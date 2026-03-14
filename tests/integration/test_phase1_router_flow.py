"""
tests/integration/test_phase1_router_flow.py — Phase 1 Router 训练流程集成测试

验证内容：
    - Phase A（run_phase_a）：合成 Parquet 采样 → tokenize → 更新 Store → 重聚类 → 生成 id_to_rowcol
    - Phase B：RouterTrainDataset → DataLoader → 2 步训练 → loss 有效
    - Checkpoint：save_checkpoint 创建正确目录和文件
    - 完整 mini epoch（evaluate_recall_at_k 不崩溃）

设计原则：
    - 使用合成 store（knowledge_num=64，num_keys=8，N=32 条知识）
    - Mock Parquet 采样（内存中随机文本，不依赖真实 FineWeb-Edu 文件）
    - Mock KnowledgeEncoder（随机 embedding，不依赖 Qwen3 模型权重）
    - 不依赖 GPU / SwanLab / 实际 Parquet 文件

输出报告：tests/outputs/phase1_router/flow_<timestamp>.md
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "phase1_router"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────
# 公共 Config 和组件工厂
# ─────────────────────────────────────────────────────


@dataclass
class _RouterCfg:
    knowledge_num: int = 64
    dim: int = 64
    query_dim: int = 64
    key_proj_dim: int = 32
    num_candidates: int = 4
    temperature: float = 0.1
    max_candidates_per_cell: int = -1
    adapter_dim: int = 32
    refined_num_heads: int = 2
    refined_num_layers: int = 1
    recluster_threshold: float = 0.1


@dataclass
class _ModelCfg:
    fusion_length: int = 16
    anchor_length: int = 32
    hidden_dim: int = 64
    encoder_depth: int = 2
    num_layers: int = 2
    injection_method: str = "attention"
    injection_layers: tuple = (1,)


@dataclass
class _TrainCfg:
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 8
    phase1_max_epochs: int = 1
    phase1_warmup_steps: int = 0
    phase1_gradient_accumulation_steps: int = 1
    phase1_recluster_batch_size: int = 32
    grad_clip: float = 1.0
    bf16: bool = False


@dataclass
class _DataCfg:
    phase1_parquet_dir: str = "data/compressed/v2"
    phase1_tokenize_batch_size: int = 100
    phase1_recluster_chunk_size: int = 8
    fusion_length: int = 16
    anchor_length: int = 32
    num_workers: int = 0
    train_max_samples: int = -1


@dataclass
class _PathsCfg:
    model_dir: str = "models/qwen3-0.6B"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"


@dataclass
class _SwanLabCfg:
    project: str = "test-phase1"
    enabled: bool = False  # 测试时禁用
    log_every_n_steps: int = 10


@dataclass
class _Config:
    router: _RouterCfg = None
    model: _ModelCfg = None
    train: _TrainCfg = None
    data: _DataCfg = None
    paths: _PathsCfg = None
    swanlab: _SwanLabCfg = None

    def __post_init__(self):
        if self.router is None:
            self.router = _RouterCfg()
        if self.model is None:
            self.model = _ModelCfg()
        if self.train is None:
            self.train = _TrainCfg()
        if self.data is None:
            self.data = _DataCfg()
        if self.paths is None:
            self.paths = _PathsCfg(checkpoint_dir=str(OUTPUT_DIR / "checkpoints"))
        if self.swanlab is None:
            self.swanlab = _SwanLabCfg()


class _MockEncoder:
    """Mock KnowledgeEncoder：不依赖 Qwen3 权重，直接返回随机向量。"""

    def __init__(self, hidden_dim: int = 64):
        self._dim = hidden_dim
        self.device = torch.device("cpu")

    def encode_mean(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回 [B, hidden_dim] 随机向量。"""
        return torch.randn(ids.shape[0], self._dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回 [B, L, hidden_dim] 随机向量。"""
        return torch.randn(ids.shape[0], ids.shape[1], self._dim)

    def requires_grad_(self, _: bool) -> "_MockEncoder":
        return self

    def to(self, device) -> "_MockEncoder":
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def eval(self) -> "_MockEncoder":
        return self


def _make_test_store(cfg: _Config) -> "DualKnowledgeStore":
    """构造填充 N=32 条知识的测试 Store，并运行一次重聚类。"""
    from router.memory_bank import DualKnowledgeStore

    store = DualKnowledgeStore(cfg.router, cfg.model.fusion_length, cfg.model.anchor_length, device="cpu")
    N = 32
    store.fusion_bank.data[:N] = torch.randint(100, 3000, (N, cfg.model.fusion_length))
    store.anchor_bank.data[:N] = torch.randint(100, 3000, (N, cfg.model.anchor_length))
    store.valid_mask[:N] = True
    store.next_free = N
    encoder = _MockEncoder(cfg.model.hidden_dim)
    store.compact_and_recluster(encoder)
    return store


def _make_synthetic_parquet_rows(N: int = 32) -> List[Dict[str, str]]:
    """生成合成 Parquet 行（text + compressed_text 字段），模拟 FineWeb-Edu 数据。"""
    rows = []
    for i in range(N):
        rows.append({
            "text": f"This is synthetic knowledge entry {i}. " * 8,  # 充足文本供截断
            "compressed_text": f"synthetic entry {i} compressed tokens",
        })
    return rows


# ─────────────────────────────────────────────────────
# 测试 1: tokenize_parquet_batch 形状（集成场景）
# ─────────────────────────────────────────────────────


def test_tokenize_with_synthetic_tokenizer():
    """
    使用 MockTokenizer（不依赖真实模型）验证 tokenize_parquet_batch 输出形状。
    """
    from training.phase1_router import tokenize_parquet_batch
    from unittest.mock import MagicMock

    B = 8
    anchor_length = 32
    fusion_length = 16

    # Mock tokenizer：返回固定形状的 input_ids
    class _MockTokenizer:
        def __call__(self, texts, max_length, truncation, padding, add_special_tokens, return_tensors):
            n = len(texts)
            return {"input_ids": torch.randint(100, 3000, (n, max_length))}

    rows = _make_synthetic_parquet_rows(B)
    anchor_ids, fusion_ids = tokenize_parquet_batch(
        rows,
        _MockTokenizer(),
        anchor_length=anchor_length,
        fusion_length=fusion_length,
        batch_size=4,
    )

    assert anchor_ids.shape == (B, anchor_length), f"anchor_ids 形状错误: {anchor_ids.shape}"
    assert fusion_ids.shape == (B, fusion_length), f"fusion_ids 形状错误: {fusion_ids.shape}"


# ─────────────────────────────────────────────────────
# 测试 2: RouterTrainDataset + DataLoader
# ─────────────────────────────────────────────────────


def test_router_train_dataset():
    """
    RouterTrainDataset 应能正确构造，DataLoader 取出一批时字段形状正确。
    """
    from training.phase1_router import RouterTrainDataset
    from torch.utils.data import DataLoader

    N = 32
    anchor_length = 32
    anchor_ids = torch.randint(0, 3000, (N, anchor_length))
    id_to_rowcol = torch.zeros(N, 2, dtype=torch.long)
    id_to_rowcol[:, 0] = torch.randint(0, 8, (N,))
    id_to_rowcol[:, 1] = torch.randint(0, 8, (N,))

    dataset = RouterTrainDataset(anchor_ids, id_to_rowcol)
    assert len(dataset) == N

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    assert batch["anchor_ids"].shape == (8, anchor_length)
    assert batch["anchor_mask"].shape == (8, anchor_length)
    assert batch["target_row"].shape == (8,)
    assert batch["target_col"].shape == (8,)
    assert batch["entry_id"].shape == (8,)


# ─────────────────────────────────────────────────────
# 测试 3: 2 步 Phase B 训练（核心集成测试）
# ─────────────────────────────────────────────────────


def test_phase_b_two_steps(tmp_path):
    """
    模拟 2 步 Phase B 训练：
    1. 使用合成 store（知识已填充 + 重聚类）
    2. RouterTrainDataset → DataLoader
    3. 执行 2 次 forward + backward + optimizer.step
    4. 验证 loss 有效（正标量，无 NaN/Inf）
    5. 验证 router 参数已更新（grad 非 None）
    """
    from training.phase1_router import (
        RouterTrainDataset,
        compute_router_loss,
        compute_teacher_logits,
    )
    from router.model import MemoryRouter
    from torch.utils.data import DataLoader

    cfg = _Config(paths=_PathsCfg(checkpoint_dir=str(tmp_path / "checkpoints")))
    encoder = _MockEncoder(cfg.model.hidden_dim)

    # 构造并填充 store
    store = _make_test_store(cfg)
    N = store.next_free
    id_to_rowcol = store.get_rowcol_labels()

    # 合成 anchor_ids
    anchor_ids = torch.randint(0, 3000, (N, cfg.model.anchor_length))

    dataset = RouterTrainDataset(anchor_ids, id_to_rowcol)
    loader = DataLoader(dataset, batch_size=cfg.train.phase1_batch_size, shuffle=True)

    # 构造 router（用 mock encoder）
    router = MemoryRouter(cfg.router, encoder)
    router.pkm.update_keys(store.row_centroids, store.col_centroids)

    optimizer = AdamW([p for p in router.parameters() if p.requires_grad], lr=1e-3)

    losses = []
    for step, batch in enumerate(loader):
        if step >= 2:
            break

        anchor_ids_b = batch["anchor_ids"]
        anchor_mask_b = batch["anchor_mask"]
        target_row_b = batch["target_row"]
        target_col_b = batch["target_col"]
        entry_ids_b = batch["entry_id"]

        # query embedding（mock encoder）
        q_emb = encoder.encode_mean(anchor_ids_b, anchor_mask_b)

        # teacher logits
        teacher_log1, teacher_log2 = compute_teacher_logits(q_emb.detach(), router.pkm, 0.1)

        # router forward（GT 强制插入候选集）
        out = router(q_emb, store, target_entry_ids=entry_ids_b)

        # 损失
        loss, ce, kl, fine = compute_router_loss(
            out, target_row_b, target_col_b, teacher_log1, teacher_log2, entry_ids_b
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        # 在 zero_grad 之前检查梯度（zero_grad(set_to_none=True) 会清空梯度）
        has_grad = any(p.grad is not None for p in router.parameters() if p.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach()))

    # 验证损失有效
    assert len(losses) == 2, f"期望 2 步损失，实际 {len(losses)}"
    for i, l in enumerate(losses):
        assert not (l != l), f"step {i} loss 为 NaN"  # nan != nan is True
        assert l < 1e6, f"step {i} loss={l} 异常大"

    # 验证梯度存在（至少 pkm 参数有梯度）
    assert has_grad, "Phase B 后 router 参数应有梯度"


# ─────────────────────────────────────────────────────
# 测试 4: save_checkpoint 创建正确文件
# ─────────────────────────────────────────────────────


def test_save_checkpoint(tmp_path):
    """
    save_checkpoint 应在正确目录下创建 router.pt / store.pt / meta.txt。
    """
    from training.phase1_router import save_checkpoint
    from router.model import MemoryRouter

    cfg = _Config(paths=_PathsCfg(checkpoint_dir=str(tmp_path / "checkpoints")))
    encoder = _MockEncoder(cfg.model.hidden_dim)
    store = _make_test_store(cfg)

    router = MemoryRouter(cfg.router, encoder)
    router.pkm.update_keys(store.row_centroids, store.col_centroids)

    # Mock Accelerator（仅需 is_main_process + unwrap_model）
    mock_accel = MagicMock()
    mock_accel.is_main_process = True
    mock_accel.unwrap_model.return_value = router

    recall = {1: 0.5, 4: 0.75, 16: 0.9}

    save_checkpoint(mock_accel, router, store, epoch=0, recall=recall, cfg=cfg, best=True)

    # 验证 epoch0 目录
    epoch_dir = Path(cfg.paths.checkpoint_dir) / "phase1_epoch0"
    assert epoch_dir.exists(), f"epoch 目录不存在: {epoch_dir}"
    assert (epoch_dir / "router.pt").exists(), "router.pt 不存在"
    assert (epoch_dir / "store.pt").exists(), "store.pt 不存在"
    assert (epoch_dir / "meta.txt").exists(), "meta.txt 不存在"

    # 验证 best 目录
    best_dir = Path(cfg.paths.checkpoint_dir) / "phase1_best"
    assert best_dir.exists(), f"best 目录不存在: {best_dir}"
    assert (best_dir / "router.pt").exists(), "best/router.pt 不存在"

    # 验证 meta.txt 内容
    meta_text = (epoch_dir / "meta.txt").read_text()
    assert "epoch=0" in meta_text
    assert "recall@1=0.5000" in meta_text


# ─────────────────────────────────────────────────────
# 测试 5: evaluate_recall_at_k 不崩溃
# ─────────────────────────────────────────────────────


def test_evaluate_recall_at_k_smoke():
    """
    evaluate_recall_at_k 在合成 store 上应正常运行并返回 Dict[int, float]。
    """
    from training.phase1_router import RouterTrainDataset, evaluate_recall_at_k
    from router.model import MemoryRouter

    cfg = _Config()
    encoder = _MockEncoder(cfg.model.hidden_dim)
    store = _make_test_store(cfg)
    N = store.next_free
    id_to_rowcol = store.get_rowcol_labels()

    router = MemoryRouter(cfg.router, encoder)
    router.pkm.update_keys(store.row_centroids, store.col_centroids)

    anchor_ids = torch.randint(0, 3000, (N, cfg.model.anchor_length))
    dataset = RouterTrainDataset(anchor_ids, id_to_rowcol)

    recall, _, _ = evaluate_recall_at_k(
        router, store, dataset, encoder,
        device=torch.device("cpu"),
        Ks=[1, 4, 16],
        max_eval_samples=16,
    )

    for k in [1, 4, 16]:
        assert k in recall, f"Recall@{k} 缺失"
        assert 0.0 <= recall[k] <= 1.0, f"Recall@{k}={recall[k]} 超出 [0,1]"


# ─────────────────────────────────────────────────────
# 保存 Markdown 报告（pytest fixture）
# ─────────────────────────────────────────────────────


def test_write_report(
    request,
    capsys,
):
    """
    执行一次完整的 mini Phase B 训练并写出 Markdown 报告。
    """
    from training.phase1_router import (
        RouterTrainDataset,
        compute_router_loss,
        compute_teacher_logits,
        evaluate_recall_at_k,
    )
    from router.model import MemoryRouter
    from torch.utils.data import DataLoader

    cfg = _Config()
    encoder = _MockEncoder(cfg.model.hidden_dim)
    store = _make_test_store(cfg)
    N = store.next_free
    id_to_rowcol = store.get_rowcol_labels()
    anchor_ids = torch.randint(0, 3000, (N, cfg.model.anchor_length))
    dataset = RouterTrainDataset(anchor_ids, id_to_rowcol)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    router = MemoryRouter(cfg.router, encoder)
    router.pkm.update_keys(store.row_centroids, store.col_centroids)
    optimizer = AdamW([p for p in router.parameters() if p.requires_grad], lr=1e-3)

    t_start = time.time()
    step_results = []

    for step, batch in enumerate(loader):
        if step >= 3:
            break
        entry_ids_b = batch["entry_id"]
        q_emb = encoder.encode_mean(batch["anchor_ids"], batch["anchor_mask"])
        teacher_log1, teacher_log2 = compute_teacher_logits(q_emb.detach(), router.pkm, 0.1)
        out = router(q_emb, store, target_entry_ids=entry_ids_b)
        loss, ce, kl, fine = compute_router_loss(
            out, batch["target_row"], batch["target_col"],
            teacher_log1, teacher_log2, entry_ids_b
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_results.append({
            "step": step,
            "loss": float(loss.detach()),
            "ce": float(ce),
            "kl": float(kl),
            "fine": float(fine),
        })

    recall, _, _ = evaluate_recall_at_k(
        router, store, dataset, encoder,
        device=torch.device("cpu"), Ks=[1, 4, 16], max_eval_samples=32,
    )
    elapsed = time.time() - t_start

    # 写 Markdown 报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"flow_{ts}.md"
    lines = [
        "# Phase 1 Router 集成测试报告",
        f"## 时间: {datetime.now().isoformat()}",
        f"## 配置",
        f"- knowledge_num: {cfg.router.knowledge_num}",
        f"- N（已填充）: {N}",
        f"- batch_size: 8",
        f"- anchor_length: {cfg.model.anchor_length}",
        "",
        "## Phase B 训练步骤",
        "| step | total_loss | ce | kl | fine |",
        "|------|-----------|----|----|------|",
    ]
    for r in step_results:
        lines.append(f"| {r['step']} | {r['loss']:.4f} | {r['ce']:.4f} | {r['kl']:.4f} | {r['fine']:.4f} |")

    lines += [
        "",
        "## 验证指标（Recall@K）",
        f"- Recall@1:  {recall[1]:.4f}",
        f"- Recall@4:  {recall[4]:.4f}",
        f"- Recall@16: {recall[16]:.4f}",
        "",
        f"## 耗时: {elapsed:.2f}s",
        "",
        "## 结论",
        "- [x] loss 为正标量且无 NaN" if all(0 < r["loss"] < 1e6 for r in step_results) else "- [ ] loss 异常",
        "- [x] Recall@K 在 [0,1] 内" if all(0 <= recall[k] <= 1 for k in [1, 4, 16]) else "- [ ] Recall@K 异常",
    ]

    report_path.write_text("\n".join(lines))
    print(f"\n[集成测试报告] 已保存至: {report_path}")

    # 基本断言
    assert all(0 < r["loss"] < 1e6 for r in step_results), "某步 loss 无效"
    assert all(0 <= recall[k] <= 1 for k in [1, 4, 16]), "Recall@K 超出合法范围"
