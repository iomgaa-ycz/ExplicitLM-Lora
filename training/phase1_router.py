"""
training/phase1_router.py — Phase 1 Router 训练循环

功能：
    实现端到端 Router 自监督训练：每 Epoch 从预压缩 FineWeb-Edu Parquet 数据中
    采样 N 条文本，tokenize 后更新 DualKnowledgeStore，重聚类（Phase A），
    然后用这批文本做自监督路由训练（Phase B）。

训练信号：
    - 自监督：每条文本既是知识条目（存入 store），也是 query（路由目标）
    - 损失：CE(粗排行列标签) + α·KL(teacher 软标签) + CE(精排局部索引)
    - Teacher 软标签：anchor embedding 与 cluster centroids 的相似度（无 label smoothing 假设）

数据：
    - 来源：data/compressed/v2/*.parquet（预压缩 FineWeb-Edu，无需 LLMLingua）
    - 字段：text（原文，256 token）→ AnchorBank；compressed_text（~62 token）→ FusionBank

依赖：
    accelerate（分布式兼容）
    swanlab（实验追踪，可通过 cfg.swanlab.enabled=False 禁用）
    router/memory_bank.py (DualKnowledgeStore)
    router/model.py (MemoryRouter)
    models/qwen_wrapper.py (KnowledgeEncoder, load_base_model)
    config.py (Config)
"""

from __future__ import annotations

import gc
import glob
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

if TYPE_CHECKING:
    from config import Config
    from router.memory_bank import DualKnowledgeStore
    from router.model import MemoryRouter, RouterOutput
    from models.qwen_wrapper import KnowledgeEncoder
    from router.memory_gate import ProductKeyMemory


# ─────────────────────────────────────────────────────
# §1  Parquet 数据采样器
# ─────────────────────────────────────────────────────


class ParquetEpochSampler:
    """
    按文件粒度从 Parquet 目录采样 N 条文本，每 epoch 使用不同 seed 以确保多样性。

    设计：
        - 按 seed 随机打乱文件列表，顺序读取文件直到累积 N 条
        - 整文件读取后截断，保持同文件内数据连续性
        - 不使用 pandas streaming，以 list[dict] 形式返回，方便后续 tokenize

    参数：
        parquet_dir: Parquet 文件所在目录（包含 *.parquet 文件）
        n_samples: 每 epoch 采样的条目数（通常 = knowledge_num）
    """

    def __init__(self, parquet_dir: str, n_samples: int) -> None:
        """
        初始化采样器，扫描目录下所有 .parquet 文件。

        参数：
            parquet_dir: Parquet 文件目录
            n_samples: 每 epoch 需要的样本数

        返回：
            None

        异常：
            FileNotFoundError: 目录不存在或无 .parquet 文件
        """
        self.n_samples = n_samples
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"ParquetEpochSampler: 目录 {parquet_dir} 下无 .parquet 文件"
            )
        self.files: List[str] = files

    def sample_epoch_data(self, seed: int) -> List[Dict[str, str]]:
        """
        按 seed 采样 n_samples 条文本，返回包含 text/compressed_text 字段的 dict 列表。

        参数：
            seed: 随机种子（通常为 epoch 编号），确保每轮采样可重现

        返回：
            List[Dict[str, str]]，每条 dict 含 "text" 和 "compressed_text" 字段

        实现细节：
            按 seed 随机排列文件 → 顺序读取 → 累积到 n_samples 时截断
        """
        rng = random.Random(seed)
        shuffled_files = list(self.files)
        rng.shuffle(shuffled_files)

        rows: List[Dict[str, str]] = []
        for fpath in shuffled_files:
            if len(rows) >= self.n_samples:
                break
            df = pd.read_parquet(fpath, columns=["text", "compressed_text"])
            # 转换为 dict 列表，方便后续按批处理
            rows.extend(df[["text", "compressed_text"]].to_dict(orient="records"))
            del df

        return rows[: self.n_samples]


# ─────────────────────────────────────────────────────
# §2  tokenize 工具
# ─────────────────────────────────────────────────────


def tokenize_parquet_batch(
    rows: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    anchor_length: int,
    fusion_length: int,
    batch_size: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Parquet 行批量 tokenize，返回 anchor_ids（原文）和 fusion_ids（压缩文本）。

    无需 LLMLingua：数据已预压缩，直接 tokenize 两个字段即可。
    text 固定 256 token，截断至 anchor_length=128；
    compressed_text 约 62 token，pad/truncate 至 fusion_length=64。

    参数：
        rows: sample_epoch_data 返回的 dict 列表
        tokenizer: Qwen3 tokenizer
        anchor_length: AnchorBank token 长度（截断 text 至此长度）
        fusion_length: FusionBank token 长度（pad/truncate compressed_text 至此长度）
        batch_size: tokenize 批大小（避免一次性处理全部数据 OOM）

    返回：
        anchor_ids: [N, anchor_length] torch.long
        fusion_ids:  [N, fusion_length] torch.long

    异常：
        AssertionError: 输出形状不符合预期
    """
    N = len(rows)
    anchor_ids = torch.zeros(N, anchor_length, dtype=torch.long)
    fusion_ids = torch.zeros(N, fusion_length, dtype=torch.long)

    n_batches = (N + batch_size - 1) // batch_size
    for start in tqdm(range(0, N, batch_size), total=n_batches, desc="[Phase A] Tokenize", unit="batch"):
        end = min(start + batch_size, N)
        chunk = rows[start:end]

        texts = [r["text"] for r in chunk]
        compressed = [r["compressed_text"] for r in chunk]

        # Phase 1: tokenize 原文 → anchor
        enc_anchor = tokenizer(
            texts,
            max_length=anchor_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        # Phase 2: tokenize 压缩文本 → fusion
        enc_fusion = tokenizer(
            compressed,
            max_length=fusion_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )

        anchor_ids[start:end] = enc_anchor["input_ids"]
        fusion_ids[start:end] = enc_fusion["input_ids"]

    assert anchor_ids.shape == (N, anchor_length), (
        f"anchor_ids 形状不匹配：期望 ({N}, {anchor_length})，实际 {tuple(anchor_ids.shape)}"
    )
    assert fusion_ids.shape == (N, fusion_length), (
        f"fusion_ids 形状不匹配：期望 ({N}, {fusion_length})，实际 {tuple(fusion_ids.shape)}"
    )
    return anchor_ids, fusion_ids


# ─────────────────────────────────────────────────────
# §3  训练数据集
# ─────────────────────────────────────────────────────


class RouterTrainDataset(Dataset):
    """
    Phase 1 Router 训练数据集，每条样本包含：
        anchor_ids:     [anchor_length] — AnchorBank token IDs（原文截断）
        anchor_mask:    [anchor_length] — attention mask
        target_row:     scalar long — 该条目的 row cluster 标签
        target_col:     scalar long — 该条目的 col cluster 标签
        entry_id:       scalar long — 该条目在 store 中的全局索引

    自监督设计：每条文本既是知识（存入 store），也是 query（路由目标）。
    target_row/col 来自 compact_and_recluster 后的 get_rowcol_labels()。

    参数：
        anchor_ids:   [N, anchor_length] torch.long
        id_to_rowcol: [N, 2] torch.long（[:,0]=row, [:,1]=col）
    """

    def __init__(
        self,
        anchor_ids: torch.Tensor,
        id_to_rowcol: torch.Tensor,
    ) -> None:
        """
        初始化数据集。

        参数：
            anchor_ids:   [N, anchor_length] long — 知识原文的 token IDs
            id_to_rowcol: [N, 2] long — 每条知识的 (row, col) cluster 标签

        返回：
            None

        异常：
            AssertionError: 两个张量的第一维不一致
        """
        assert anchor_ids.shape[0] == id_to_rowcol.shape[0], (
            f"anchor_ids 行数 {anchor_ids.shape[0]} ≠ id_to_rowcol 行数 {id_to_rowcol.shape[0]}"
        )
        self._anchor_ids = anchor_ids  # [N, L]
        self._id_to_rowcol = id_to_rowcol  # [N, 2]
        self._n = anchor_ids.shape[0]

    def __len__(self) -> int:
        """返回数据集大小。"""
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回第 idx 条样本。

        参数：
            idx: 样本索引

        返回：
            dict 含以下键：
                anchor_ids:  [anchor_length] long
                anchor_mask: [anchor_length] long（0=pad, 1=有效）
                target_row:  scalar long
                target_col:  scalar long
                entry_id:    scalar long
        """
        ids = self._anchor_ids[idx]  # [L]
        mask = (ids != 0).long()  # [L]
        row = self._id_to_rowcol[idx, 0]  # scalar
        col = self._id_to_rowcol[idx, 1]  # scalar
        return {
            "anchor_ids": ids,
            "anchor_mask": mask,
            "target_row": row,
            "target_col": col,
            "entry_id": torch.tensor(idx, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────
# §4  损失计算工具
# ─────────────────────────────────────────────────────


@torch.no_grad()
def compute_teacher_logits(
    anchor_emb: torch.Tensor,
    pkm: "ProductKeyMemory",
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用 anchor embedding 计算 teacher 软标签（知识自身与 cluster centroids 的相似度）。

    Teacher 代表"理想路由"：知道完整的 anchor embedding 后对各 cluster 的归属概率。
    Student 的 KL loss 以此为目标，引导 PKM 学习更贴近理想分布的投影。

    参数：
        anchor_emb: [B, D] — frozen encoder 输出的 anchor 文本 embedding
        pkm:        ProductKeyMemory 实例（含 query_proj, row/col key_proj, keys）
        temperature: PKM softmax 温度（与训练时保持一致）

    返回：
        teacher_logits_1: [B, num_keys] — row cluster logits（未 softmax）
        teacher_logits_2: [B, num_keys] — col cluster logits（未 softmax）

    实现细节：
        复用 PKM 的投影层（query_proj → chunk → key_proj → L2norm → 点积），
        仅用 @torch.no_grad() 包裹，不影响反向传播路径
    """
    # Phase 1: query 侧投影（与 PKM.forward 保持完全一致的计算路径）
    q = pkm.query_proj(anchor_emb)  # [B, query_dim]
    q1, q2 = q.chunk(2, dim=-1)  # [B, key_proj_dim] × 2
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    # Phase 2: key 侧投影 + L2 归一化
    k1 = F.normalize(
        pkm.row_key_proj(pkm.row_keys), p=2, dim=-1
    )  # [num_keys, key_proj_dim]
    k2 = F.normalize(pkm.col_key_proj(pkm.col_keys), p=2, dim=-1)

    # Phase 3: 温度缩放点积 → teacher logits
    teacher_logits_1 = q1 @ k1.T / temperature  # [B, num_keys]
    teacher_logits_2 = q2 @ k2.T / temperature  # [B, num_keys]

    return teacher_logits_1, teacher_logits_2


def compute_target_local_idx(
    target_entry_ids: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    在 PKM 粗排候选集中查找 target entry ID 的局部索引。

    若 target 未被粗排命中（PKM 漏检），返回 -100，fine_loss 对该样本忽略。

    参数：
        target_entry_ids: [B] long — 每个样本的 ground truth 知识条目 ID
        candidates:       [B, C] long — PKM 粗排输出的候选 ID

    返回：
        [B] long — target 在 candidates 中的局部索引（0-based），未命中为 -100

    实现：
        对每个 batch 样本做线性搜索（C 通常 ≤ 32，开销可忽略）
    """
    B, C = candidates.shape
    local_idx = torch.full((B,), -100, dtype=torch.long, device=candidates.device)
    for b in range(B):
        match = (candidates[b] == target_entry_ids[b]).nonzero(as_tuple=True)[0]
        if match.numel() > 0:
            local_idx[b] = match[0]
    return local_idx


def compute_router_loss(
    out: "RouterOutput",
    target_row: torch.Tensor,
    target_col: torch.Tensor,
    teacher_logits_1: torch.Tensor,
    teacher_logits_2: torch.Tensor,
    target_local_idx: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 Router 完整训练损失。

    损失组成（参照 TD.md §2.2 伪代码）：
        coarse_loss = (1-α)·CE(粗排) + α·KL(teacher 软标签)
        fine_loss   = CE(精排，ignore_index=-100 跳过未命中样本)
        total_loss  = coarse_loss + fine_loss

    参数：
        out:              RouterOutput（含 coarse_scores 和 fine_scores）
        target_row:       [B] long — 粗排目标 row cluster
        target_col:       [B] long — 粗排目标 col cluster
        teacher_logits_1: [B, num_keys] — teacher row logits（未 softmax）
        teacher_logits_2: [B, num_keys] — teacher col logits（未 softmax）
        target_local_idx: [B] long — 精排目标局部索引（-100=未命中）
        alpha:            KL 蒸馏权重（默认 0.2）

    返回：
        total_loss:  scalar — 总损失（用于 backward）
        ce_loss:     scalar detach — 粗排 CE 损失（监控用）
        kl_loss:     scalar detach — KL 蒸馏损失（监控用）
        fine_loss:   scalar detach — 精排 CE 损失（监控用）
    """
    scores_1, scores_2 = out.coarse_scores  # [B, num_keys] × 2

    # Phase 1: 粗排 CE（行 + 列）
    ce_row = F.cross_entropy(scores_1, target_row)
    ce_col = F.cross_entropy(scores_2, target_col)
    ce_loss = ce_row + ce_col

    # Phase 2: KL 蒸馏（student log_softmax vs teacher softmax）
    log_row = F.log_softmax(scores_1, dim=-1)
    log_col = F.log_softmax(scores_2, dim=-1)
    teacher_prob_1 = F.softmax(teacher_logits_1, dim=-1)
    teacher_prob_2 = F.softmax(teacher_logits_2, dim=-1)
    kl_loss = F.kl_div(log_row, teacher_prob_1, reduction="batchmean") + F.kl_div(
        log_col, teacher_prob_2, reduction="batchmean"
    )

    coarse_loss = (1 - alpha) * ce_loss + alpha * kl_loss

    # Phase 3: 精排 CE（跳过未命中样本）
    # 当所有样本均 miss（全为 -100）时 cross_entropy 返回 nan，需特判为 0
    valid_mask = target_local_idx != -100
    if valid_mask.any():
        fine_loss = F.cross_entropy(
            out.fine_scores, target_local_idx, ignore_index=-100
        )
    else:
        fine_loss = torch.zeros(
            1, device=out.fine_scores.device, dtype=out.fine_scores.dtype
        ).squeeze()

    total_loss = coarse_loss + fine_loss

    return total_loss, ce_loss.detach(), kl_loss.detach(), fine_loss.detach()


# ─────────────────────────────────────────────────────
# §5  评估工具
# ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_recall_at_k(
    router: "MemoryRouter",
    store: "DualKnowledgeStore",
    dataset: RouterTrainDataset,
    encoder: "KnowledgeEncoder",
    device: torch.device,
    Ks: List[int] = None,
    max_eval_samples: int = 1000,
) -> Dict[int, float]:
    """
    计算 Router 的 Recall@K 指标（K=1,4,16）。

    对 dataset 中的样本：将 anchor_ids 作为 query，调用 router.forward()，
    检查 ground truth entry_id 是否在 top-K candidates 中。

    参数：
        router:           MemoryRouter 实例（eval 模式）
        store:            DualKnowledgeStore（含倒排索引）
        dataset:          RouterTrainDataset
        encoder:          KnowledgeEncoder（frozen）
        device:           torch.device
        Ks:               需要计算的 K 值列表，默认 [1, 4, 16]
        max_eval_samples: 最多评估的样本数（避免全量评估过慢）

    返回：
        Dict[int, float]，key=K，value=Recall@K（0.0~1.0）
    """
    if Ks is None:
        Ks = [1, 4, 16]

    router.eval()
    n_eval = min(max_eval_samples, len(dataset))

    hits: Dict[int, int] = {k: 0 for k in Ks}
    total = 0

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    for batch in loader:
        if total >= n_eval:
            break

        anchor_ids = batch["anchor_ids"].to(device)  # [B, L]
        anchor_mask = batch["anchor_mask"].to(device)  # [B, L]
        entry_ids = batch["entry_id"].to(device)  # [B]

        q_emb = encoder.encode_mean(anchor_ids, anchor_mask)  # [B, D]
        out = router(q_emb, store)
        candidates = out.candidates  # [B, C]

        batch_size = entry_ids.shape[0]
        for b in range(batch_size):
            if total >= n_eval:
                break
            for k in Ks:
                if entry_ids[b] in candidates[b, :k]:
                    hits[k] += 1
            total += 1

    del loader
    recall = {k: hits[k] / total for k in Ks} if total > 0 else {k: 0.0 for k in Ks}
    return recall


# ─────────────────────────────────────────────────────
# §6  Phase A：重聚类 + 更新 Keys
# ─────────────────────────────────────────────────────


def run_phase_a(
    epoch: int,
    sampler: ParquetEpochSampler,
    store: "DualKnowledgeStore",
    encoder: "KnowledgeEncoder",
    router: "MemoryRouter",
    cfg: "Config",
    tokenizer: AutoTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, str]]]:
    """
    Phase A：数据采样 → tokenize → 更新 Store → 重聚类 → 更新 PKM Keys。

    在主进程上执行（多卡场景下通过 Accelerate 广播结果）。

    参数：
        epoch:     当前 epoch 编号（用作 seed）
        sampler:   ParquetEpochSampler 实例
        store:     DualKnowledgeStore（in-place 更新）
        encoder:   KnowledgeEncoder（frozen，用于 compact_and_recluster 内部编码）
        router:    MemoryRouter（更新 pkm.row_keys/col_keys）
        cfg:       Config 配置对象
        tokenizer: Qwen3 tokenizer

    返回：
        id_to_rowcol: [N, 2] torch.long — 每条知识的 (row, col) cluster 标签
        anchor_ids:   [N, anchor_length] torch.long — 供 RouterTrainDataset 使用
        rows:         原始 dict 列表（含 text/compressed_text，Phase B 可复用）
    """
    N = cfg.router.knowledge_num
    chunk_size = cfg.data.phase1_recluster_chunk_size

    # ── Step 1: 采样文本 ──
    t0 = time.time()
    rows = sampler.sample_epoch_data(seed=epoch)
    assert len(rows) == N, f"Phase A 采样数量不符：期望 {N}，实际 {len(rows)}"
    print(f"[Phase A 1/6] 采样完成：{N} 条 ({time.time() - t0:.1f}s)")

    # ── Step 2: tokenize ──
    t0 = time.time()
    anchor_ids, fusion_ids = tokenize_parquet_batch(
        rows,
        tokenizer,
        anchor_length=cfg.model.anchor_length,
        fusion_length=cfg.model.fusion_length,
        batch_size=cfg.data.phase1_tokenize_batch_size,
    )
    # anchor_ids: [N, 128], fusion_ids: [N, 64]
    print(f"[Phase A 2/6] Tokenize 完成 ({time.time() - t0:.1f}s)")

    # ── Step 3: 更新双 Bank（全量覆盖，Phase 1 每 epoch 重新采样）──
    store.fusion_bank.update_all(fusion_ids)
    store.anchor_bank.update_all(anchor_ids)
    store.valid_mask[:N] = True
    store.next_free = N
    print(f"[Phase A 3/6] Bank 更新完成")

    # ── Step 4: 全量重聚类（编码 anchor → 子空间聚类 → 重建倒排索引）──
    t0 = time.time()
    print(f"[Phase A 4/6] 开始 compact_and_recluster (chunk_size={chunk_size}) ...")
    store.compact_and_recluster(encoder, chunk_size=chunk_size)
    print(f"[Phase A 4/6] compact_and_recluster 完成 ({time.time() - t0:.1f}s)")

    # ── Step 5: 更新 PKM Keys（必须在 recluster 之后，row/col centroids 已刷新）──
    assert store.row_centroids is not None, "recluster 后 row_centroids 不应为 None"
    router.pkm.update_keys(store.row_centroids, store.col_centroids)
    print(f"[Phase A 5/6] PKM Keys 更新完成")

    # ── Step 6: 构建 id_to_rowcol 映射 ──
    id_to_rowcol = store.get_rowcol_labels()  # [N, 2]
    print(f"[Phase A 6/6] id_to_rowcol 构建完成")

    # ── Step 7: 释放 tokenize 临时张量 ──
    del fusion_ids
    gc.collect()

    return id_to_rowcol, anchor_ids, rows


# ─────────────────────────────────────────────────────
# §7  SwanLab 工具
# ─────────────────────────────────────────────────────


def init_swanlab(cfg: "Config") -> None:
    """
    初始化 SwanLab 实验追踪（仅在 cfg.swanlab.enabled=True 时调用）。

    参数：
        cfg: Config 配置对象

    返回：
        None
    """
    if not cfg.swanlab.enabled:
        return
    try:
        import swanlab

        swanlab.init(
            project=cfg.swanlab.project,
            config={
                "knowledge_num": cfg.router.knowledge_num,
                "phase1_lr": cfg.train.phase1_lr,
                "phase1_batch_size": cfg.train.phase1_batch_size,
                "phase1_max_epochs": cfg.train.phase1_max_epochs,
                "phase1_warmup_steps": cfg.train.phase1_warmup_steps,
                "phase1_gradient_accumulation_steps": cfg.train.phase1_gradient_accumulation_steps,
                "encoder_depth": cfg.model.encoder_depth,
                "num_candidates": cfg.router.num_candidates,
                "temperature": cfg.router.temperature,
                "parquet_dir": cfg.data.phase1_parquet_dir,
            },
        )
    except Exception as exc:
        print(f"[WARN] SwanLab 初始化失败（继续训练不受影响）: {exc}")


def log_swanlab(
    accelerator: Accelerator,
    enabled: bool,
    step: int,
    metrics: Dict[str, float],
) -> None:
    """
    向 SwanLab 上报指标（仅主进程执行，enabled=False 时跳过）。

    参数：
        accelerator: Accelerate 加速器（用于检查 is_main_process）
        enabled:     是否启用 SwanLab（来自 cfg.swanlab.enabled）
        step:        当前全局训练步数
        metrics:     {指标名: 值} dict

    返回：
        None
    """
    if not enabled or not accelerator.is_main_process:
        return
    try:
        import swanlab

        swanlab.log(metrics, step=step)
    except Exception as exc:
        print(f"[WARN] SwanLab log 失败（step={step}）: {exc}")


# ─────────────────────────────────────────────────────
# §8  Checkpoint 保存
# ─────────────────────────────────────────────────────


def save_checkpoint(
    accelerator: Accelerator,
    router: "MemoryRouter",
    store: "DualKnowledgeStore",
    epoch: int,
    recall: Dict[int, float],
    cfg: "Config",
    best: bool = False,
) -> None:
    """
    保存 Router checkpoint（仅主进程执行）。

    保存内容：
        - router.pt：MemoryRouter.state_dict（含 PKM keys buffer）
        - store.pt：DualKnowledgeStore 全量状态（双 Bank + 倒排索引）
        - meta.txt：epoch、recall 等元信息

    参数：
        accelerator: Accelerate 加速器
        router:      MemoryRouter（wrapped by Accelerate）
        store:       DualKnowledgeStore
        epoch:       当前 epoch（从 0 起）
        recall:      {K: Recall@K}
        cfg:         Config
        best:        是否同时保存到 phase1_best 目录

    返回：
        None
    """
    if not accelerator.is_main_process:
        return

    ckpt_root = Path(cfg.paths.checkpoint_dir)
    # 每 epoch 独立目录
    epoch_dir = ckpt_root / f"phase1_epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(router)
    torch.save(unwrapped.state_dict(), epoch_dir / "router.pt")
    store.save_state(str(epoch_dir / "store.pt"))

    meta_lines = [
        f"epoch={epoch}",
        f"recall@1={recall.get(1, 0.0):.4f}",
        f"recall@4={recall.get(4, 0.0):.4f}",
        f"recall@16={recall.get(16, 0.0):.4f}",
    ]
    (epoch_dir / "meta.txt").write_text("\n".join(meta_lines))

    if best:
        best_dir = ckpt_root / "phase1_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(unwrapped.state_dict(), best_dir / "router.pt")
        store.save_state(str(best_dir / "store.pt"))
        (best_dir / "meta.txt").write_text("\n".join(meta_lines))
        print(
            f"[Phase1] Saved best checkpoint → {best_dir}  Recall@1={recall.get(1, 0.0):.4f}"
        )


# ─────────────────────────────────────────────────────
# §9  多卡广播工具
# ─────────────────────────────────────────────────────


def _broadcast_tensor(
    accelerator: Accelerator,
    tensor: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    在所有 rank 间广播张量（主进程拥有数据，其他 rank 接收）。

    参数：
        accelerator: Accelerate 加速器
        tensor:      主进程上的源张量（非主进程传 None）
        shape:       张量形状（所有 rank 必须一致）
        dtype:       张量 dtype

    返回：
        广播后的张量（所有 rank 都持有完整数据）
    """
    import torch.distributed as dist

    if accelerator.num_processes == 1:
        # 单卡无需广播
        assert tensor is not None
        return tensor.to(accelerator.device)

    if accelerator.is_main_process:
        buf = tensor.to(accelerator.device)
    else:
        buf = torch.zeros(shape, dtype=dtype, device=accelerator.device)

    dist.broadcast(buf, src=0)
    return buf


def broadcast_store_state(
    accelerator: Accelerator,
    store: "DualKnowledgeStore",
    id_to_rowcol: Optional[torch.Tensor],
    n: int,
) -> torch.Tensor:
    """
    多卡场景下将 store 关键状态和 id_to_rowcol 广播至所有 rank。

    广播内容：
        - store.fusion_bank.data[:n]
        - store.anchor_bank.data[:n]
        - store.inverted_index[:n]
        - store.cluster_offsets
        - store.cluster_counts
        - id_to_rowcol [n, 2]

    注意：row_centroids / col_centroids 不在此广播——它们通过
    router.pkm.row_keys / col_keys（register_buffer）在 train_phase1() 中单独广播，
    以避免非主进程 row_centroids=None 导致的非对称 skip 死锁。

    参数：
        accelerator:  Accelerate 加速器
        store:        DualKnowledgeStore（主进程已完成 Phase A 更新）
        id_to_rowcol: [n, 2] long — 仅主进程有效，非主进程传 None
        n:            有效知识条目数（= knowledge_num）

    返回：
        id_to_rowcol: [n, 2] long（所有 rank 同步后）
    """
    import torch.distributed as dist

    if accelerator.num_processes == 1:
        assert id_to_rowcol is not None
        return id_to_rowcol.to(accelerator.device)

    # 广播 store 的关键 buffer（供非主进程使用）
    dev = accelerator.device
    num_keys_sq = store._num_keys * store._num_keys

    for attr, shape, dtype in [
        ("fusion_bank.data", (n, store.fusion_bank.fusion_length), torch.long),
        ("anchor_bank.data", (n, store.anchor_bank.anchor_length), torch.long),
    ]:
        obj = store
        for part in attr.split("."):
            obj = getattr(obj, part)
        if accelerator.is_main_process:
            buf = obj[:n].to(dev)
        else:
            buf = torch.zeros(shape, dtype=dtype, device=dev)
        dist.broadcast(buf, src=0)
        if not accelerator.is_main_process:
            obj[:n].copy_(buf)

    # 仅广播始终非 None、shape 固定的三个属性；row/col centroids 通过 PKM keys 在外部广播
    for attr, shape, dtype in [
        ("inverted_index", (n,), torch.long),
        ("cluster_offsets", (num_keys_sq + 1,), torch.long),
        ("cluster_counts", (num_keys_sq,), torch.long),
    ]:
        src = getattr(store, attr)
        src_slice = src[:n] if attr == "inverted_index" else src
        if accelerator.is_main_process:
            buf = src_slice.to(dev)
        else:
            buf = torch.zeros(shape, dtype=dtype, device=dev)
        dist.broadcast(buf, src=0)
        if not accelerator.is_main_process:
            if attr == "inverted_index":
                store.inverted_index[:n].copy_(buf)
            else:
                setattr(store, attr, buf)

    # 广播 next_free
    nf_tensor = torch.tensor(
        [store.next_free if accelerator.is_main_process else 0],
        dtype=torch.long,
        device=dev,
    )
    dist.broadcast(nf_tensor, src=0)
    store.next_free = int(nf_tensor.item())

    # 广播 id_to_rowcol
    id_rc = _broadcast_tensor(accelerator, id_to_rowcol, (n, 2), torch.long)
    return id_rc


# ─────────────────────────────────────────────────────
# §10  主训练函数
# ─────────────────────────────────────────────────────


def train_phase1(cfg: "Config", device_str: str = "cpu") -> None:
    """
    Phase 1 Router 训练主函数，使用 Accelerate 实现单/多 GPU 兼容。

    训练流程：
        for epoch:
            Phase A（主进程）: 采样 N 条 FineWeb-Edu 文本 → tokenize → 更新 Store
                              → compact_and_recluster → pkm.update_keys → id_to_rowcol
            广播（多卡）:       store state + id_to_rowcol → 所有 rank
            Phase B（所有 rank）: RouterTrainDataset → DataLoader
                              → for batch: compute q_emb → teacher logits → router forward
                              → compute_router_loss → backward → optimizer step
            评估（主进程）:     Recall@1/4/16 → SwanLab → 若提升则保存 best checkpoint

    参数：
        cfg:        Config 配置对象
        device_str: 设备字符串（单卡时指定，多卡时被 Accelerate 覆盖）

    返回：
        None

    异常：
        FileNotFoundError: Parquet 目录或模型路径不存在
        AssertionError:    数据采样量不足 knowledge_num
    """
    # ── Phase 1: Accelerate 初始化 ──
    # Phase A 仅主进程执行（编码 1M 条约 8 分钟），扩大 NCCL 超时至 2 小时避免 rank 1 空等被杀
    _nccl_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.phase1_gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.train.bf16 else "no",
        kwargs_handlers=[_nccl_kwargs],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"[Phase1] device={device}, num_processes={accelerator.num_processes}")
        print(
            f"[Phase1] knowledge_num={cfg.router.knowledge_num}, parquet_dir={cfg.data.phase1_parquet_dir}"
        )
        init_swanlab(cfg)

    # ── Phase 2: 加载模型组件 ──
    from models.qwen_wrapper import KnowledgeEncoder, load_base_model
    from router.model import MemoryRouter
    from router.memory_bank import DualKnowledgeStore

    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.model_dir)

    base_model = load_base_model(cfg.paths.model_dir, bf16=cfg.train.bf16)
    encoder = KnowledgeEncoder(
        base_model, cfg.model.encoder_depth, cfg.model.hidden_dim
    )
    encoder.requires_grad_(False)  # Phase 1 完全冻结 encoder
    encoder = encoder.to(device)
    encoder.eval()

    # store 放 CPU（数据量大，GPU 可能放不下；重聚类时需 CPU numpy）
    store = DualKnowledgeStore(
        cfg.router,
        cfg.model.fusion_length,
        cfg.model.anchor_length,
        device="cpu",
    )

    router = MemoryRouter(cfg.router, encoder)
    router = router.to(device)

    optimizer = AdamW(
        [p for p in router.parameters() if p.requires_grad],
        lr=cfg.train.phase1_lr,
        weight_decay=1e-2,
    )

    # Accelerate prepare（router + optimizer；encoder 已手动 .to(device)，不走 prepare）
    router, optimizer = accelerator.prepare(router, optimizer)
    # 一次解包，全局复用：DDP-wrapped router 仅用于梯度同步，属性访问和 forward 均用 unwrapped_router
    unwrapped_router = accelerator.unwrap_model(router)

    # ── Phase 3: 数据采样器 ──
    sampler = ParquetEpochSampler(
        cfg.data.phase1_parquet_dir,
        cfg.router.knowledge_num,
    )

    # ── Phase 4: 学习率调度器（先设占位步数，DataLoader 构建后更新）──
    # 总步数 = max_epochs × (N // batch_size) // grad_accum
    total_steps_estimate = (
        cfg.train.phase1_max_epochs
        * (cfg.router.knowledge_num // cfg.train.phase1_batch_size)
        // cfg.train.phase1_gradient_accumulation_steps
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.phase1_warmup_steps,
        num_training_steps=total_steps_estimate,
    )
    scheduler = accelerator.prepare(scheduler)

    best_recall1 = 0.0
    global_step = 0
    N = cfg.router.knowledge_num

    for epoch in range(cfg.train.phase1_max_epochs):
        t_epoch_start = time.time()

        # ── Phase A: 数据更新 + 重聚类（主进程执行）──
        if accelerator.is_main_process:
            print(f"[Phase1] Epoch {epoch} — Phase A: 采样 + 重聚类")
            id_to_rowcol, anchor_ids, _ = run_phase_a(
                epoch, sampler, store, encoder, unwrapped_router, cfg, tokenizer
            )
            id_to_rowcol = id_to_rowcol.to(device)
            anchor_ids_dev = anchor_ids  # 保持 CPU，Dataset 直接用
        else:
            id_to_rowcol = None
            anchor_ids_dev = torch.zeros(N, cfg.model.anchor_length, dtype=torch.long)

        # ── 多卡广播 store 状态 ──
        id_to_rowcol = broadcast_store_state(accelerator, store, id_to_rowcol, N)
        # 广播 PKM 聚类 Keys（register_buffer，永不为 None；rank 0 已在 run_phase_a 中 update_keys）
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            dist.barrier()
            dist.broadcast(unwrapped_router.pkm.row_keys, src=0)
            dist.broadcast(unwrapped_router.pkm.col_keys, src=0)
            dist.barrier()
        # anchor_ids_dev 无需单独广播：anchor_bank 已在 broadcast_store_state 中同步
        # 对标参考项目（train_router.py）：所有 rank 直接从已广播的 store buffer 读取
        anchor_ids_dev = store.anchor_bank.data[:N].cpu()
        # 与参考项目对齐：广播全部完成后等待所有 rank 就绪再进 Phase B
        accelerator.wait_for_everyone()

        # ── Phase B: 路由训练 ──
        router.train()
        dataset = RouterTrainDataset(anchor_ids_dev, id_to_rowcol.cpu())
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.phase1_batch_size,
            shuffle=True,
            num_workers=0,  # Phase 1 关闭多进程（避免 fork 导致内存翻倍）
            drop_last=True,
        )
        loader = accelerator.prepare(loader)

        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_kl_sum = 0.0
        epoch_fine_sum = 0.0
        n_batches = 0

        for batch in loader:
            anchor_ids_b = batch["anchor_ids"].to(device)  # [B, L]
            anchor_mask_b = batch["anchor_mask"].to(device)  # [B, L]
            target_row_b = batch["target_row"].to(device)  # [B]
            target_col_b = batch["target_col"].to(device)  # [B]
            entry_ids_b = batch["entry_id"].to(device)  # [B]

            with accelerator.accumulate(router):
                # query embedding（frozen encoder）
                with torch.no_grad():
                    q_emb = encoder.encode_mean(anchor_ids_b, anchor_mask_b)  # [B, D]

                # teacher logits（anchor 自身 embedding 与 cluster centroids 的相似度）
                teacher_log1, teacher_log2 = compute_teacher_logits(
                    q_emb.detach(),
                    unwrapped_router.pkm,
                    cfg.router.temperature,
                )

                # router forward（粗排 + 精排）
                out = unwrapped_router(q_emb, store)

                # 精排局部索引
                target_local = compute_target_local_idx(entry_ids_b, out.candidates)

                # 损失计算
                loss, ce_loss, kl_loss, fine_loss = compute_router_loss(
                    out,
                    target_row_b,
                    target_col_b,
                    teacher_log1,
                    teacher_log2,
                    target_local,
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(router.parameters(), cfg.train.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += float(loss.detach())
            epoch_ce_sum += float(ce_loss)
            epoch_kl_sum += float(kl_loss)
            epoch_fine_sum += float(fine_loss)
            n_batches += 1

            # 定步上报
            if global_step % cfg.swanlab.log_every_n_steps == 0:
                log_swanlab(
                    accelerator,
                    cfg.swanlab.enabled,
                    global_step,
                    {
                        "train/step_loss": float(loss.detach()),
                        "train/step_ce_loss": float(ce_loss),
                        "train/step_kl_loss": float(kl_loss),
                        "train/step_fine_loss": float(fine_loss),
                    },
                )

        # ── Epoch 结束：释放内存 ──
        del loader
        gc.collect()
        torch.cuda.empty_cache()

        # ── Epoch 验证 + Checkpoint ──
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        recall = evaluate_recall_at_k(
            unwrapped_router, store, dataset, encoder, device, Ks=[1, 4, 16]
        )
        del dataset

        t_elapsed = time.time() - t_epoch_start
        if accelerator.is_main_process:
            print(
                f"[Phase1] Epoch {epoch} | loss={avg_loss:.4f} "
                f"| Recall@1={recall[1]:.4f} @4={recall[4]:.4f} @16={recall[16]:.4f} "
                f"| {t_elapsed:.1f}s"
            )

            log_swanlab(
                accelerator,
                cfg.swanlab.enabled,
                global_step,
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_ce_loss": epoch_ce_sum / max(n_batches, 1),
                    "train/epoch_kl_loss": epoch_kl_sum / max(n_batches, 1),
                    "train/epoch_fine_loss": epoch_fine_sum / max(n_batches, 1),
                    "eval/recall@1": recall[1],
                    "eval/recall@4": recall[4],
                    "eval/recall@16": recall[16],
                    "train/epoch": epoch,
                },
            )

            is_best = recall[1] > best_recall1
            if is_best:
                best_recall1 = recall[1]
            save_checkpoint(
                accelerator, router, store, epoch, recall, cfg, best=is_best
            )

        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"[Phase1] 训练完成，best Recall@1={best_recall1:.4f}")
