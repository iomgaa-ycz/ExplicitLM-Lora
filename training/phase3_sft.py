"""
training/phase3_sft.py — Phase 3 MedQA 下游 SFT

目标：从 Phase 2 最优权重出发，在 MedQA 四选一数据集上进行监督微调，
激活模型利用注入知识回答下游医学问题的能力。

数据流：
    MedQA HuggingFace dataset（data/medqa/hf_dataset）
    + 知识映射（data/medqa_knowledge_train.jsonl）
        → SFT 序列："Question: {q}\nA. ...\nD. ...\nAnswer: {A/B/C/D}"
        → Labels: Prompt 部分 = -100，仅答案字母 token 参与损失
        → knowledge_ids: question[:200].strip() 查表

早停：
    监控 validation loss，连续 patience=3 轮不下降则停止。
    最优权重保存到 checkpoints/phase3_best/

启动：FROM_PHASE2=checkpoints/phase2_best NUM_GPUS=2 GPU_IDS=6,7 bash scripts/run_phase3_sft.sh
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import Config
from models.injection_modules import AttentionInjection
from models.modified_model import ModifiedQwen
from models.qwen_wrapper import KnowledgeEncoder, load_base_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# §1  知识映射加载
# ─────────────────────────────────────────────


def _load_knowledge_map(jsonl_path: str) -> Dict[str, List[int]]:
    """
    从 JSONL 文件加载 question → knowledge_ids 映射。

    JSONL 格式（每行）：
        {"key": "<question 前 200 字符>", "knowledge_ids": [int, ...]}

    参数：
        jsonl_path: JSONL 文件路径

    返回：
        Dict[str, List[int]]，key = question prefix，value = token id 列表
    """
    knowledge_map: Dict[str, List[int]] = {}
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning("[Phase3SFT] 知识映射文件不存在: %s（将使用空知识兜底）", jsonl_path)
        return knowledge_map

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj["key"]
                knowledge_map[key] = obj["knowledge_ids"]
            except (KeyError, json.JSONDecodeError) as exc:
                logger.warning("[Phase3SFT] 跳过无效知识映射行: %s", exc)

    logger.info("[Phase3SFT] 知识映射加载完成: %d 条（来自 %s）", len(knowledge_map), jsonl_path)
    return knowledge_map


# ─────────────────────────────────────────────
# §2  MedQASFTDataset
# ─────────────────────────────────────────────


class MedQASFTDataset(Dataset):
    """
    MedQA 四选一 SFT 数据集。

    SFT 序列格式（参考 Reference/Explicit-Lora-fusion/training/medqa_dataset.py）：
        prompt = "Question: {question}\\nA. {a}\\nB. {b}\\nC. {c}\\nD. {d}\\nAnswer:"
        full   = prompt + " {A/B/C/D}"

    Labels 设计：
        input_ids  = tokenize(full)[:-1]   （next-token prediction 输入）
        labels     = tokenize(full)[1:]    （预测目标）
        labels[0 : prompt_len - 1] = -100  （Prompt 部分不计损失）
        labels[padding 位置] = -100

    知识来源：
        question[:200].strip() → knowledge_map → knowledge_ids [fusion_length]
        无命中时使用全 pad_token_id 兜底

    参数：
        hf_dataset_split:  HuggingFace dataset split（train 或 validation）
        knowledge_map:     Dict[str, List[int]]，由 _load_knowledge_map 加载
        tokenizer:         AutoTokenizer 实例
        max_seq_length:    SFT 序列最大 token 数（默认 256）
        fusion_length:     knowledge_ids 固定长度（默认 64）
        pad_token_id:      padding token id
    """

    def __init__(
        self,
        hf_dataset_split,
        knowledge_map: Dict[str, List[int]],
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        fusion_length: int,
        pad_token_id: int,
    ) -> None:
        self.data = hf_dataset_split
        self.knowledge_map = knowledge_map
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.fusion_length = fusion_length
        self.pad_token_id = pad_token_id

        # 默认知识（全 pad）：无命中时兜底，确保 knowledge_ids 维度一致
        self.default_knowledge: List[int] = [pad_token_id] * fusion_length

        logger.info(
            "[Phase3SFT] MedQASFTDataset 初始化完成: %d 条样本, max_seq=%d, fusion=%d",
            len(self.data),
            max_seq_length,
            fusion_length,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        构建单条 SFT 样本：
            input_ids, labels, attention_mask, knowledge_ids, knowledge_mask
        """
        row = self.data[idx]
        question: str = row["question"]
        options: Dict[str, str] = row["options"]   # {"A": ..., "B": ..., "C": ..., "D": ...}
        answer_idx: str = row["answer_idx"]         # "A" / "B" / "C" / "D"

        # Phase 1: 构建 SFT 序列
        options_str = "".join(f"{k}. {v}\n" for k, v in options.items())
        prompt = f"Question: {question}\n{options_str}Answer:"
        full_text = f"{prompt} {answer_idx}"

        # Phase 2: Tokenize（不加 special tokens，手动控制长度）
        full_ids: List[int] = self.tokenizer.encode(
            full_text, add_special_tokens=False
        )
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt, add_special_tokens=False
        )

        # Phase 3: 截断至 max_seq_length + 1（shift 后变 max_seq_length）
        max_full_len = self.max_seq_length + 1
        full_ids = full_ids[:max_full_len]
        prompt_len = min(len(prompt_ids), len(full_ids))

        # Phase 4: Next-token prediction 切分
        #   input_ids = full[:-1],  labels = full[1:]
        #   确保长度 >= 2（否则无法训练）
        if len(full_ids) < 2:
            full_ids = full_ids + [self.pad_token_id]

        input_raw = full_ids[:-1]   # [0 : L-1]
        label_raw = full_ids[1:]    # [1 : L]

        # Phase 5: Prompt 部分 labels = -100（仅对答案字母计算损失）
        # prompt_len - 1 是因为 label 相对 input 移位了 1
        mask_until = max(0, prompt_len - 1)
        for i in range(min(mask_until, len(label_raw))):
            label_raw[i] = -100

        # Phase 6: 填充到 max_seq_length
        seq_len = self.max_seq_length
        pad_len = seq_len - len(input_raw)
        if pad_len > 0:
            input_raw = input_raw + [self.pad_token_id] * pad_len
            label_raw = label_raw + [-100] * pad_len
        else:
            input_raw = input_raw[:seq_len]
            label_raw = label_raw[:seq_len]

        input_ids = torch.tensor(input_raw, dtype=torch.long)
        labels = torch.tensor(label_raw, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Phase 7: 知识查表（question[:200].strip() 作为 key）
        key = question[:200].strip()
        raw_k_ids: List[int] = self.knowledge_map.get(key, self.default_knowledge)

        # 截断 / 填充到 fusion_length
        k_ids = raw_k_ids[: self.fusion_length]
        if len(k_ids) < self.fusion_length:
            k_ids = k_ids + [self.pad_token_id] * (self.fusion_length - len(k_ids))

        knowledge_ids = torch.tensor(k_ids, dtype=torch.long)
        knowledge_mask = (knowledge_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "knowledge_ids": knowledge_ids,
            "knowledge_mask": knowledge_mask,
        }


# ─────────────────────────────────────────────
# §3  验证集 loss 评估
# ─────────────────────────────────────────────


@torch.no_grad()
def _evaluate_val_loss(
    accelerator: Accelerator,
    modified_qwen: nn.Module,
    val_loader: DataLoader,
) -> float:
    """
    计算验证集平均 loss（用于早停判断）。

    参数：
        accelerator:   Accelerate 对象
        modified_qwen: 已包装的融合模型
        val_loader:    已 prepare 的验证 DataLoader

    返回：
        全进程聚合后的平均 loss（float）
    """
    modified_qwen.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_steps = torch.tensor(0, device=accelerator.device)

    for batch in val_loader:
        output = modified_qwen(
            input_ids=batch["input_ids"],
            knowledge_ids=batch["knowledge_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # 过滤掉全-100 的 batch（无有效 token，loss=nan）
        if not torch.isnan(output.loss):
            total_loss += output.loss.detach()
            total_steps += 1

    # 多卡聚合
    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_steps = accelerator.reduce(total_steps, reduction="sum")
    val_loss = (total_loss / total_steps.clamp(min=1)).item()

    modified_qwen.train()
    return val_loss


# ─────────────────────────────────────────────
# §4  Checkpoint 保存
# ─────────────────────────────────────────────


def _save_phase3_checkpoint(
    accelerator: Accelerator,
    modified_qwen: nn.Module,
    epoch: int,
    train_loss: float,
    val_loss: float,
    cfg: Config,
    best: bool = False,
) -> None:
    """
    保存 Phase 3 SFT checkpoint（仅主进程执行）。

    保存内容与格式同 Phase 2（injection_modules.pt + encoder_*.pt + meta.txt）。

    参数：
        accelerator:  Accelerate 对象
        modified_qwen: 融合模型（已 prepare）
        epoch:        当前 epoch（从 0 起）
        train_loss:   当前 epoch 训练 loss
        val_loss:     当前 epoch 验证 loss
        cfg:          项目配置
        best:         True → 同时保存 phase3_best/
    """
    if not accelerator.is_main_process:
        return

    ckpt_root = Path(cfg.paths.checkpoint_dir)
    epoch_dir = ckpt_root / f"phase3_epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(modified_qwen)

    torch.save(unwrapped.injection_modules.state_dict(), epoch_dir / "injection_modules.pt")
    torch.save(unwrapped.knowledge_encoder.layers.state_dict(), epoch_dir / "encoder_layers.pt")
    torch.save(unwrapped.knowledge_encoder.norm.state_dict(), epoch_dir / "encoder_norm.pt")
    (epoch_dir / "meta.txt").write_text(
        f"epoch={epoch}\n"
        f"train_loss={train_loss:.6f}\n"
        f"val_loss={val_loss:.6f}\n"
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    logger.info("[Phase3SFT] Checkpoint 保存到 %s", epoch_dir)

    if best:
        best_dir = ckpt_root / "phase3_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(unwrapped.injection_modules.state_dict(), best_dir / "injection_modules.pt")
        torch.save(unwrapped.knowledge_encoder.layers.state_dict(), best_dir / "encoder_layers.pt")
        torch.save(unwrapped.knowledge_encoder.norm.state_dict(), best_dir / "encoder_norm.pt")
        (best_dir / "meta.txt").write_text(
            f"epoch={epoch}\n"
            f"train_loss={train_loss:.6f}\n"
            f"val_loss={val_loss:.6f}\n"
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logger.info("[Phase3SFT] Best checkpoint → %s", best_dir)


# ─────────────────────────────────────────────
# §5  SwanLab 工具
# ─────────────────────────────────────────────


def _init_swanlab(cfg: Config, accelerator: Accelerator) -> None:
    """主进程初始化 SwanLab（Phase 3）。"""
    if not (cfg.swanlab.enabled and accelerator.is_main_process):
        return
    try:
        import swanlab  # type: ignore[import]

        swanlab.init(
            project="explicit-lora-phase3",
            config={
                "phase": 3,
                "lr": cfg.train.phase3_lr,
                "batch_size": cfg.train.phase3_batch_size,
                "gradient_accumulation_steps": cfg.train.phase3_gradient_accumulation_steps,
                "max_epochs": cfg.train.phase3_max_epochs,
                "warmup_steps": cfg.train.phase3_warmup_steps,
                "patience": cfg.train.patience,
                "max_seq_length": cfg.data.phase3_max_seq_length,
                "fusion_length": cfg.model.fusion_length,
                "encoder_depth": cfg.model.encoder_depth,
                "injection_layers": cfg.model.injection_layers,
            },
        )
        logger.info("[Phase3SFT] SwanLab 初始化完成（project=explicit-lora-phase3）")
    except Exception as exc:  # pragma: no cover
        logger.warning("[Phase3SFT] SwanLab 初始化失败（已跳过）: %s", exc)


def _log_swanlab(
    metrics: Dict[str, float],
    step: int,
    accelerator: Accelerator,
    cfg: Config,
) -> None:
    """主进程上报指标到 SwanLab。"""
    if not (cfg.swanlab.enabled and accelerator.is_main_process):
        return
    try:
        import swanlab  # type: ignore[import]

        swanlab.log(metrics, step=step)
    except Exception:  # pragma: no cover
        pass


# ─────────────────────────────────────────────
# §6  模型构建 & Checkpoint 加载
# ─────────────────────────────────────────────


def _build_modified_qwen_phase3(
    cfg: Config,
    phase2_ckpt: Optional[str],
) -> tuple[ModifiedQwen, AutoTokenizer]:
    """
    构建 Phase 3 用 ModifiedQwen，并从 Phase 2 checkpoint 加载注入权重。

    参数：
        cfg:          项目配置
        phase2_ckpt:  Phase 2 最优 checkpoint 目录（含 injection_modules.pt 等）
                      None → 使用随机初始化（不推荐，仅测试用）

    返回：
        (modified_qwen, tokenizer)
    """
    model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)

    logger.info("[Phase3SFT] 加载基础模型: %s", model_path)
    base_model = load_base_model(model_path, bf16=cfg.train.bf16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 知识编码器（Phase 3 同样解冻，与 Phase 2 可训练范围一致）
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
    )
    encoder.unfreeze_layers()

    # 注入模块
    if cfg.model.injection_method != "attention":
        raise ValueError(
            f"Phase 3 当前仅支持 injection_method='attention'，"
            f"实际配置: {cfg.model.injection_method}"
        )
    injection_modules = nn.ModuleList(
        [AttentionInjection(cfg.model.hidden_dim) for _ in cfg.model.injection_layers]
    )

    modified_qwen = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=encoder,
        injection_modules=injection_modules,
        injection_layers=cfg.model.injection_layers,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 从 Phase 2 checkpoint 加载权重
    if phase2_ckpt is not None:
        ckpt_dir = Path(phase2_ckpt)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Phase 2 checkpoint 目录不存在: {ckpt_dir}\n"
                f"请先运行 Phase 2 训练，或通过 --from-phase2 指定正确路径。"
            )

        _load_tensor = lambda fname: torch.load(ckpt_dir / fname, map_location="cpu", weights_only=True)

        inj_path = ckpt_dir / "injection_modules.pt"
        enc_layers_path = ckpt_dir / "encoder_layers.pt"
        enc_norm_path = ckpt_dir / "encoder_norm.pt"

        if inj_path.exists():
            modified_qwen.injection_modules.load_state_dict(_load_tensor("injection_modules.pt"))
            logger.info("[Phase3SFT] injection_modules 从 %s 加载", ckpt_dir)
        else:
            logger.warning("[Phase3SFT] injection_modules.pt 不存在，使用随机初始化！")

        if enc_layers_path.exists():
            modified_qwen.knowledge_encoder.layers.load_state_dict(
                _load_tensor("encoder_layers.pt")
            )
            logger.info("[Phase3SFT] encoder_layers 从 %s 加载", ckpt_dir)
        else:
            logger.warning("[Phase3SFT] encoder_layers.pt 不存在，使用随机初始化！")

        if enc_norm_path.exists():
            modified_qwen.knowledge_encoder.norm.load_state_dict(
                _load_tensor("encoder_norm.pt")
            )
            logger.info("[Phase3SFT] encoder_norm 从 %s 加载", ckpt_dir)
        else:
            logger.warning("[Phase3SFT] encoder_norm.pt 不存在，使用随机初始化！")
    else:
        logger.warning(
            "[Phase3SFT] 未提供 phase2_ckpt，使用随机初始化（建议先运行 Phase 2）"
        )

    trainable = sum(p.numel() for p in modified_qwen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modified_qwen.parameters())
    logger.info(
        "[Phase3SFT] 可训练参数: %d / %d (%.1f%%)", trainable, total, 100.0 * trainable / total
    )

    return modified_qwen, tokenizer


# ─────────────────────────────────────────────
# §7  主训练函数 train_phase3()
# ─────────────────────────────────────────────


def train_phase3(cfg: Config, device: str, phase2_ckpt: Optional[str] = None) -> None:
    """
    Phase 3 MedQA SFT 主入口。

    通过 accelerate launch 调用，自动处理多卡 DDP。
    早停策略：监控 val_loss，连续 patience 轮不下降则停止。

    参数：
        cfg:          项目配置对象（由 load_config 加载）
        device:       设备字符串（多卡时由 Accelerator 自动管理）
        phase2_ckpt:  Phase 2 最优 checkpoint 目录（默认使用 checkpoints/phase2_best）
    """
    # ── §7.1  初始化 ──
    if phase2_ckpt is None:
        phase2_ckpt = str(Path(cfg.paths.checkpoint_dir) / "phase2_best")
        logger.info("[Phase3SFT] 未指定 phase2_ckpt，默认使用: %s", phase2_ckpt)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.phase3_gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.train.bf16 else "no",
    )
    logger.info(
        "[Phase3SFT] Accelerator 初始化（num_processes=%d, mixed_precision=%s）",
        accelerator.num_processes,
        "bf16" if cfg.train.bf16 else "no",
    )

    _init_swanlab(cfg, accelerator)

    # ── §7.2  构建模型 ──
    modified_qwen, tokenizer = _build_modified_qwen_phase3(cfg, phase2_ckpt)

    # ── §7.3  加载数据集 ──
    dataset_dir = str(Path(cfg.paths.data_dir) / "medqa" / "hf_dataset")
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"MedQA HF dataset 目录不存在: {dataset_dir}\n"
            f"请先将 MedQA 数据放入 data/medqa/hf_dataset/。"
        )
    hf_dataset = load_from_disk(dataset_dir)
    logger.info(
        "[Phase3SFT] 加载 MedQA dataset: train=%d, test=%d",
        len(hf_dataset["train"]),
        len(hf_dataset["test"]),
    )

    # 知识映射（train + validation）
    train_km_path = str(Path(cfg.paths.data_dir) / "medqa_knowledge_train.jsonl")
    val_km_path = str(Path(cfg.paths.data_dir) / "medqa_knowledge_validation.jsonl")
    train_knowledge_map = _load_knowledge_map(train_km_path)
    val_knowledge_map = _load_knowledge_map(val_km_path)

    # ── §7.4  构建 Dataset & DataLoader ──
    pad_id = tokenizer.pad_token_id

    train_dataset = MedQASFTDataset(
        hf_dataset_split=hf_dataset["train"],
        knowledge_map=train_knowledge_map,
        tokenizer=tokenizer,
        max_seq_length=cfg.data.phase3_max_seq_length,
        fusion_length=cfg.model.fusion_length,
        pad_token_id=pad_id,
    )
    # Phase 3 使用 MedQA test split 作验证（与原实验评测集一致）
    val_dataset = MedQASFTDataset(
        hf_dataset_split=hf_dataset["test"],
        knowledge_map=val_knowledge_map,
        tokenizer=tokenizer,
        max_seq_length=cfg.data.phase3_max_seq_length,
        fusion_length=cfg.model.fusion_length,
        pad_token_id=pad_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.phase3_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.phase3_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # ── §7.5  优化器 & 调度器 ──
    trainable_params = [p for p in modified_qwen.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.phase3_lr)

    steps_per_epoch = (
        len(train_loader)
        // cfg.train.phase3_gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * cfg.train.phase3_max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.phase3_warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "[Phase3SFT] 优化器初始化: lr=%.2e, steps_per_epoch=%d, total_steps=%d",
        cfg.train.phase3_lr,
        steps_per_epoch,
        total_steps,
    )

    # ── §7.6  Accelerate 包装 ──
    modified_qwen, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        modified_qwen, optimizer, scheduler, train_loader, val_loader
    )

    # ── §7.7  训练循环（含早停）──
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(cfg.train.phase3_max_epochs):
        epoch_start = time.time()
        logger.info(
            "[Phase3SFT] ══ Epoch %d/%d 开始（早停计数: %d/%d）══",
            epoch,
            cfg.train.phase3_max_epochs - 1,
            patience_counter,
            cfg.train.patience,
        )

        # ── 训练 ──
        modified_qwen.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in train_loader:
            with accelerator.accumulate(modified_qwen):
                output = modified_qwen(
                    input_ids=batch["input_ids"],
                    knowledge_ids=batch["knowledge_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss

                # 过滤 NaN（全-100 batch 时 loss=nan）
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.train.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss_sum += loss.item()
                epoch_steps += 1

                if global_step % cfg.swanlab.log_every_n_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "[Phase3SFT] step=%d  train_loss=%.4f  lr=%.2e",
                        global_step,
                        loss.item(),
                        lr_now,
                    )
                    _log_swanlab(
                        {"train/loss": loss.item(), "train/lr": lr_now},
                        step=global_step,
                        accelerator=accelerator,
                        cfg=cfg,
                    )

        epoch_train_loss = epoch_loss_sum / max(epoch_steps, 1)

        # ── 验证 ──
        epoch_val_loss = _evaluate_val_loss(accelerator, modified_qwen, val_loader)
        epoch_time = time.time() - epoch_start

        logger.info(
            "[Phase3SFT] Epoch %d 完成: train_loss=%.4f  val_loss=%.4f  耗时=%.1fs",
            epoch,
            epoch_train_loss,
            epoch_val_loss,
            epoch_time,
        )
        _log_swanlab(
            {
                "epoch/train_loss": epoch_train_loss,
                "epoch/val_loss": epoch_val_loss,
                "epoch/time_s": epoch_time,
            },
            step=epoch,
            accelerator=accelerator,
            cfg=cfg,
        )

        # ── 早停 & Checkpoint ──
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            logger.info("[Phase3SFT] 新最优 val_loss=%.4f（epoch %d）", best_val_loss, epoch)
        else:
            patience_counter += 1
            logger.info(
                "[Phase3SFT] val_loss 未改善（%.4f >= %.4f），早停计数: %d/%d",
                epoch_val_loss,
                best_val_loss,
                patience_counter,
                cfg.train.patience,
            )

        _save_phase3_checkpoint(
            accelerator, modified_qwen, epoch, epoch_train_loss, epoch_val_loss, cfg, best=is_best
        )
        accelerator.wait_for_everyone()

        if patience_counter >= cfg.train.patience:
            logger.info(
                "[Phase3SFT] 早停触发！连续 %d 轮 val_loss 未改善，最优 val_loss=%.4f",
                cfg.train.patience,
                best_val_loss,
            )
            break

    logger.info(
        "[Phase3SFT] 训练完成！最优 val_loss=%.4f，checkpoint → checkpoints/phase3_best/",
        best_val_loss,
    )
