"""
training/phase2_fusion.py — Phase 2 Fusion 预训练

目标：在 FineWeb-Edu 语料上，给定压缩知识（Oracle），训练注入模块和知识编码器
学习通用的知识融合能力，为 Phase 3 下游 SFT 奠定基础。

数据流：
    FineWeb-Edu Parquet（同 Phase 1 数据）
        text（256 token 原文）→ 截断 128 tokens → input_ids + labels
        compressed_text（LLMLingua 压缩）→ 截断 64 tokens → knowledge_ids（Oracle）
    ModifiedQwen.forward(input_ids, knowledge_ids, attention_mask, labels)
        → 语言建模 CrossEntropy loss（pad 位置 -100 忽略）

可训练参数：
    - injection_modules（AttentionInjection × 4，~16.8M）
    - knowledge_encoder.layers（Qwen3 前 6 层，共享权重，Phase 2 解冻）
    - knowledge_encoder.norm（独立深拷贝，始终可训练）
冻结：Qwen3 全量、Router（PKM/Adapter/Selector）

启动：NUM_GPUS=2 GPU_IDS=6,7 bash scripts/run_phase2_fusion.sh
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import Config
from models.injection_modules import AttentionInjection
from models.modified_model import ModifiedQwen
from models.qwen_wrapper import KnowledgeEncoder, load_base_model

# Phase 1 工具函数复用（数据采样 + tokenize）
from training.phase1_router import ParquetEpochSampler, tokenize_parquet_batch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# §1  FusionDataset — FineWeb-Edu 语言建模数据集
# ─────────────────────────────────────────────


class FusionDataset(Dataset):
    """
    Phase 2 Fusion 预训练数据集。

    每条样本对应一条 FineWeb-Edu 文本：
        input_ids    [anchor_length]  text 截断（等同 Phase 1 anchor_ids）
        labels       [anchor_length]  同 input_ids，padding 位置 = -100
        knowledge_ids [fusion_length] compressed_text 截断（Oracle 知识）
        attention_mask [anchor_length]  1=有效 0=padding
        knowledge_mask  [fusion_length] 1=有效 0=padding

    参数：
        anchor_ids:    [N, anchor_length] int64，text tokenize 结果
        fusion_ids:    [N, fusion_length] int64，compressed_text tokenize 结果
        pad_token_id:  padding token id（Qwen3 通常为 tokenizer.pad_token_id）
    """

    def __init__(
        self,
        anchor_ids: Tensor,
        fusion_ids: Tensor,
        pad_token_id: int,
    ) -> None:
        assert anchor_ids.shape[0] == fusion_ids.shape[0], (
            f"anchor_ids ({anchor_ids.shape[0]}) 与 fusion_ids ({fusion_ids.shape[0]}) "
            f"样本数必须一致"
        )
        self.anchor_ids = anchor_ids       # [N, K_a]
        self.fusion_ids = fusion_ids       # [N, K_f]
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.anchor_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        返回单条样本字典：
            input_ids, labels, attention_mask, knowledge_ids, knowledge_mask
        """
        input_ids = self.anchor_ids[idx]       # [K_a]
        knowledge_ids = self.fusion_ids[idx]   # [K_f]

        # attention_mask: 1=有效 0=padding
        attention_mask = (input_ids != self.pad_token_id).long()

        # labels: 同 input_ids，padding 位置 = -100
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100

        knowledge_mask = (knowledge_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "knowledge_ids": knowledge_ids,
            "knowledge_mask": knowledge_mask,
        }


# ─────────────────────────────────────────────
# §2  Checkpoint 保存
# ─────────────────────────────────────────────


def save_fusion_checkpoint(
    accelerator: Accelerator,
    modified_qwen: ModifiedQwen,
    epoch: int,
    loss: float,
    cfg: Config,
    best: bool = False,
) -> None:
    """
    保存 Phase 2 Fusion checkpoint（仅主进程执行）。

    保存内容：
        injection_modules.pt   —— ModifiedQwen.injection_modules.state_dict()
        encoder_layers.pt      —— KnowledgeEncoder.layers.state_dict()
        encoder_norm.pt        —— KnowledgeEncoder.norm.state_dict()
        meta.txt               —— epoch, loss, timestamp

    参数：
        accelerator:  Accelerate 对象，用于 is_main_process 判断
        modified_qwen: 包含 injection_modules 和 knowledge_encoder 的融合模型
        epoch:        当前 epoch 号（从 0 起）
        loss:         当前 epoch 训练 loss
        cfg:          项目配置对象
        best:         True → 同时保存到 phase2_best/
    """
    if not accelerator.is_main_process:
        return

    ckpt_root = Path(cfg.paths.checkpoint_dir)
    epoch_dir = ckpt_root / f"phase2_epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # 获取去包装后的模型（Accelerate DDP 包装下需要 unwrap）
    unwrapped = accelerator.unwrap_model(modified_qwen)

    torch.save(
        unwrapped.injection_modules.state_dict(),
        epoch_dir / "injection_modules.pt",
    )
    torch.save(
        unwrapped.knowledge_encoder.layers.state_dict(),
        epoch_dir / "encoder_layers.pt",
    )
    torch.save(
        unwrapped.knowledge_encoder.norm.state_dict(),
        epoch_dir / "encoder_norm.pt",
    )
    (epoch_dir / "meta.txt").write_text(
        f"epoch={epoch}\nloss={loss:.6f}\ntimestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    logger.info("[Phase2Fusion] Checkpoint 保存到 %s", epoch_dir)

    if best:
        best_dir = ckpt_root / "phase2_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            unwrapped.injection_modules.state_dict(),
            best_dir / "injection_modules.pt",
        )
        torch.save(
            unwrapped.knowledge_encoder.layers.state_dict(),
            best_dir / "encoder_layers.pt",
        )
        torch.save(
            unwrapped.knowledge_encoder.norm.state_dict(),
            best_dir / "encoder_norm.pt",
        )
        (best_dir / "meta.txt").write_text(
            f"epoch={epoch}\nloss={loss:.6f}\ntimestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logger.info("[Phase2Fusion] Best checkpoint → %s", best_dir)


# ─────────────────────────────────────────────
# §3  SwanLab 工具（与 Phase 1 保持一致的接口风格）
# ─────────────────────────────────────────────


def _init_swanlab(cfg: Config, accelerator: Accelerator) -> None:
    """主进程初始化 SwanLab 实验追踪（Phase 2）。"""
    if not (cfg.swanlab.enabled and accelerator.is_main_process):
        return
    try:
        import swanlab  # type: ignore[import]

        swanlab.init(
            project="explicit-lora-phase2",
            config={
                "phase": 2,
                "lr": cfg.train.phase2_lr,
                "batch_size": cfg.train.phase2_batch_size,
                "gradient_accumulation_steps": cfg.train.phase2_gradient_accumulation_steps,
                "max_epochs": cfg.train.phase2_max_epochs,
                "warmup_steps": cfg.train.phase2_warmup_steps,
                "n_samples_per_epoch": cfg.data.phase2_n_samples_per_epoch,
                "fusion_length": cfg.model.fusion_length,
                "anchor_length": cfg.model.anchor_length,
                "encoder_depth": cfg.model.encoder_depth,
                "injection_layers": cfg.model.injection_layers,
            },
        )
        logger.info("[Phase2Fusion] SwanLab 初始化完成（project=explicit-lora-phase2）")
    except Exception as exc:  # pragma: no cover
        logger.warning("[Phase2Fusion] SwanLab 初始化失败（已跳过）: %s", exc)


def _log_swanlab(metrics: Dict[str, float], step: int, accelerator: Accelerator, cfg: Config) -> None:
    """主进程上报指标到 SwanLab。"""
    if not (cfg.swanlab.enabled and accelerator.is_main_process):
        return
    try:
        import swanlab  # type: ignore[import]

        swanlab.log(metrics, step=step)
    except Exception:  # pragma: no cover
        pass


# ─────────────────────────────────────────────
# §4  模型构建工具
# ─────────────────────────────────────────────


def _build_modified_qwen(cfg: Config, device: str) -> Tuple[ModifiedQwen, AutoTokenizer]:
    """
    构建 Phase 2 用 ModifiedQwen（Injection + KnowledgeEncoder）。

    流程：
        1. load_base_model → 冻结 Qwen3 全量
        2. KnowledgeEncoder（前 6 层共享权重）
        3. encoder.unfreeze_layers()  ← Phase 2 关键：解冻前 N 层
        4. AttentionInjection × len(injection_layers) → ModifiedQwen

    返回：
        (modified_qwen, tokenizer)
    """
    # Phase 1: 解析模型路径（支持 .env MODEL_PATH 覆盖）
    model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)

    logger.info("[Phase2Fusion] 加载基础模型: %s", model_path)
    base_model = load_base_model(model_path, bf16=cfg.train.bf16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Phase 2: 构建知识编码器并解冻前 N 层（联合训练）
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
    )
    encoder.unfreeze_layers()
    logger.info(
        "[Phase2Fusion] KnowledgeEncoder 已解冻前 %d 层（联合训练模式）",
        cfg.model.encoder_depth,
    )

    # Phase 3: 构建注入模块（AttentionInjection × N）
    if cfg.model.injection_method != "attention":
        raise ValueError(
            f"Phase 2 当前仅支持 injection_method='attention'，"
            f"实际配置: {cfg.model.injection_method}"
        )
    injection_modules = nn.ModuleList(
        [AttentionInjection(cfg.model.hidden_dim) for _ in cfg.model.injection_layers]
    )
    logger.info(
        "[Phase2Fusion] 构建 AttentionInjection × %d（注入层: %s）",
        len(injection_modules),
        cfg.model.injection_layers,
    )

    # Phase 4: 组装 ModifiedQwen
    modified_qwen = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=encoder,
        injection_modules=injection_modules,
        injection_layers=cfg.model.injection_layers,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 统计可训练参数
    trainable = sum(p.numel() for p in modified_qwen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modified_qwen.parameters())
    logger.info(
        "[Phase2Fusion] 可训练参数: %d / %d (%.1f%%)",
        trainable,
        total,
        100.0 * trainable / total,
    )

    return modified_qwen, tokenizer


# ─────────────────────────────────────────────
# §5  主训练函数 train_phase2()
# ─────────────────────────────────────────────


def train_phase2(cfg: Config, device: str) -> None:
    """
    Phase 2 Fusion 预训练主入口。

    通过 accelerate launch 调用，自动处理多卡 DDP。
    每 epoch：
        - ParquetEpochSampler 采样 N 条 FineWeb-Edu
        - tokenize → FusionDataset
        - 前向（ModifiedQwen）→ 语言建模 loss → backward
        - 保存 checkpoint（每 epoch + best）
        - SwanLab 上报

    参数：
        cfg:    项目配置对象（由 load_config 加载）
        device: 设备字符串（"cpu" / "cuda"），多卡时由 Accelerator 自动管理
    """
    # ── §5.1  初始化 Accelerator ──
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.phase2_gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.train.bf16 else "no",
    )
    logger.info(
        "[Phase2Fusion] Accelerator 初始化完成（num_processes=%d, mixed_precision=%s）",
        accelerator.num_processes,
        "bf16" if cfg.train.bf16 else "no",
    )

    _init_swanlab(cfg, accelerator)

    # ── §5.2  构建模型 ──
    modified_qwen, tokenizer = _build_modified_qwen(cfg, device)

    # ── §5.3  构建优化器 & 调度器 ──
    # 仅对 requires_grad=True 的参数创建 optimizer
    trainable_params = [p for p in modified_qwen.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.phase2_lr)

    # 调度器：先估算总步数（每 epoch 步数依赖数据量，此处用近似值初始化）
    # 实际步数在第 0 epoch 后确认，scheduler 在 accelerate.prepare 后会按实际步数 step
    approx_steps_per_epoch = (
        cfg.data.phase2_n_samples_per_epoch
        // (cfg.train.phase2_batch_size * accelerator.num_processes)
        // cfg.train.phase2_gradient_accumulation_steps
    )
    total_steps = approx_steps_per_epoch * cfg.train.phase2_max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.phase2_warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "[Phase2Fusion] 近似总步数=%d（每 epoch %d 步，共 %d epoch）",
        total_steps,
        approx_steps_per_epoch,
        cfg.train.phase2_max_epochs,
    )

    # ── §5.4  数据采样器（复用 Phase 1）──
    parquet_dir = cfg.data.phase1_parquet_dir  # Phase 2 复用同一 FineWeb-Edu 数据
    sampler = ParquetEpochSampler(
        parquet_dir=parquet_dir,
        n_samples=cfg.data.phase2_n_samples_per_epoch,
    )
    logger.info(
        "[Phase2Fusion] ParquetEpochSampler 初始化（dir=%s, n_samples=%d）",
        parquet_dir,
        cfg.data.phase2_n_samples_per_epoch,
    )

    # ── §5.5  Accelerate 包装（先不包装 dataloader，每 epoch 重建）──
    modified_qwen, optimizer, scheduler = accelerator.prepare(
        modified_qwen, optimizer, scheduler
    )

    # ── §5.6  训练循环 ──
    best_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.train.phase2_max_epochs):
        epoch_start = time.time()
        logger.info("[Phase2Fusion] ══ Epoch %d/%d 开始 ══", epoch, cfg.train.phase2_max_epochs - 1)

        # Phase A: 采样 & tokenize（所有 rank 使用相同 seed，保证一致性）
        rows = sampler.sample_epoch_data(seed=epoch)
        anchor_ids, fusion_ids = tokenize_parquet_batch(
            rows,
            tokenizer,
            anchor_length=cfg.model.anchor_length,
            fusion_length=cfg.model.fusion_length,
        )
        logger.info(
            "[Phase2Fusion] Epoch %d: 采样 %d 条，anchor_ids %s, fusion_ids %s",
            epoch,
            len(rows),
            tuple(anchor_ids.shape),
            tuple(fusion_ids.shape),
        )

        # Phase B: 构建 Dataset & DataLoader
        dataset = FusionDataset(anchor_ids, fusion_ids, tokenizer.pad_token_id)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.phase2_batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader = accelerator.prepare(dataloader)

        # Phase C: 训练 epoch
        modified_qwen.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:
            with accelerator.accumulate(modified_qwen):
                output = modified_qwen(
                    input_ids=batch["input_ids"],
                    knowledge_ids=batch["knowledge_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.train.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 统计（仅在完成 grad accum 后记录，避免中间值）
            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss_sum += loss.item()
                epoch_steps += 1

                # Step 级别日志 & SwanLab
                if global_step % cfg.swanlab.log_every_n_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "[Phase2Fusion] step=%d  loss=%.4f  lr=%.2e",
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

        # Phase D: Epoch 结束统计
        epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_time = time.time() - epoch_start
        logger.info(
            "[Phase2Fusion] Epoch %d 完成: loss=%.4f  耗时=%.1fs",
            epoch,
            epoch_loss,
            epoch_time,
        )
        _log_swanlab(
            {"epoch/train_loss": epoch_loss, "epoch/time_s": epoch_time},
            step=epoch,
            accelerator=accelerator,
            cfg=cfg,
        )

        # Phase E: 保存 checkpoint
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
        save_fusion_checkpoint(accelerator, modified_qwen, epoch, epoch_loss, cfg, best=is_best)

        # 等待所有进程同步（确保 checkpoint 写完再开始下一 epoch）
        accelerator.wait_for_everyone()

    logger.info(
        "[Phase2Fusion] 训练完成！共 %d epochs，最优 loss=%.4f，checkpoint → checkpoints/phase2_best/",
        cfg.train.phase2_max_epochs,
        best_loss,
    )
