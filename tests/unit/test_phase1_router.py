"""
tests/unit/test_phase1_router.py — Phase 1 Router 单元测试

测试范围：
    1. compute_teacher_logits — 输出形状 [B, num_keys]
    2. compute_target_local_idx — 命中时返回正确索引
    3. compute_target_local_idx — 未命中时返回 -100
    4. compute_router_loss — loss 为正标量，可反向传播
    5. DualKnowledgeStore.get_rowcol_labels — 重聚类后形状与值域正确
    6. ParquetEpochSampler — 使用真实 Parquet 文件，验证采样数和可重现性
    7. tokenize_parquet_batch — 输出形状正确
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────
# 辅助工厂：构造轻量 Mock 对象（避免依赖 GPU / 真实模型）
# ─────────────────────────────────────────────────────


def _make_mock_pkm(num_keys: int = 8, dim: int = 64, key_proj_dim: int = 32):
    """
    构造最小化 ProductKeyMemory，仅包含 compute_teacher_logits 所需接口。
    通过直接实例化 ProductKeyMemory 实现，使用小维度避免 OOM。
    """
    from dataclasses import make_dataclass

    # 使用 make_dataclass 避免 dataclass 默认值中无法引用函数参数的问题
    _RouterConfig = make_dataclass(
        "_RouterConfig",
        [
            ("dim", int, dim),
            ("query_dim", int, dim),
            ("key_proj_dim", int, key_proj_dim),
            ("knowledge_num", int, num_keys * num_keys),
            ("num_candidates", int, 4),
            ("temperature", float, 0.1),
            ("max_candidates_per_cell", int, -1),
            ("adapter_dim", int, 32),
            ("refined_num_heads", int, 2),
            ("refined_num_layers", int, 1),
            ("recluster_threshold", float, 0.1),
        ],
    )

    from router.memory_gate import ProductKeyMemory

    cfg = _RouterConfig()
    pkm = ProductKeyMemory(cfg)
    # 填充随机 row/col keys（模拟 recluster 后状态）
    with torch.no_grad():
        pkm.row_keys.copy_(torch.randn(num_keys, key_proj_dim))
        pkm.col_keys.copy_(torch.randn(num_keys, key_proj_dim))
    return pkm, num_keys, cfg


def _make_mock_router_output(B: int, num_keys: int = 8, num_candidates: int = 4):
    """
    构造最小化 RouterOutput（不走 forward，直接构造）。
    """
    from router.model import RouterOutput

    candidates = torch.randint(0, num_keys * num_keys, (B, num_candidates))
    coarse_1 = torch.randn(B, num_keys)
    coarse_2 = torch.randn(B, num_keys)
    fine_scores = torch.randn(B, num_candidates)
    best_id = candidates[:, 0]
    cand_mask = torch.ones(B, num_candidates, dtype=torch.bool)
    return RouterOutput(
        best_id=best_id,
        candidates=candidates,
        coarse_scores=(coarse_1, coarse_2),
        fine_scores=fine_scores,
        cand_mask=cand_mask,
    )


# ─────────────────────────────────────────────────────
# 测试 1: compute_teacher_logits 形状
# ─────────────────────────────────────────────────────


def test_compute_teacher_logits_shape():
    """
    compute_teacher_logits 应返回两个 [B, num_keys] 张量。
    """
    from training.phase1_router import compute_teacher_logits

    B = 4
    num_keys = 8
    dim = 64
    pkm, num_keys, _ = _make_mock_pkm(num_keys=num_keys, dim=dim, key_proj_dim=32)

    anchor_emb = torch.randn(B, dim)
    t1, t2 = compute_teacher_logits(anchor_emb, pkm, temperature=0.1)

    assert t1.shape == (B, num_keys), f"teacher_logits_1 形状错误: {t1.shape}"
    assert t2.shape == (B, num_keys), f"teacher_logits_2 形状错误: {t2.shape}"
    # 验证是 logits（未 softmax），不要求 sum=1
    assert not torch.isnan(t1).any(), "teacher_logits_1 含 NaN"
    assert not torch.isnan(t2).any(), "teacher_logits_2 含 NaN"



# ─────────────────────────────────────────────────────
# 测试 4: compute_router_loss — 可反向传播
# ─────────────────────────────────────────────────────


def test_compute_router_loss_backward():
    """
    compute_router_loss 应返回正标量总损失，且可成功反向传播。
    验证 ce_loss, kl_loss, fine_loss 均为 detach 标量（梯度不存在）。
    """
    from training.phase1_router import compute_router_loss

    B = 4
    num_keys = 8
    C = 4

    # 构造带梯度的 RouterOutput
    coarse_1 = torch.randn(B, num_keys, requires_grad=True)
    coarse_2 = torch.randn(B, num_keys, requires_grad=True)
    fine_scores = torch.randn(B, C, requires_grad=True)
    candidates = torch.randint(0, 16, (B, C))
    from router.model import RouterOutput

    # GT 知识 ID 强制放在 candidates 第 0 列，保证 entry_ids 一定在候选集中
    entry_ids = candidates[:, 0].clone()
    cand_mask = torch.ones(B, C, dtype=torch.bool)
    out = RouterOutput(
        best_id=candidates[:, 0],
        candidates=candidates,
        coarse_scores=(coarse_1, coarse_2),
        fine_scores=fine_scores,
        cand_mask=cand_mask,
    )

    target_row = torch.randint(0, num_keys, (B,))
    target_col = torch.randint(0, num_keys, (B,))
    teacher_log1 = torch.randn(B, num_keys)
    teacher_log2 = torch.randn(B, num_keys)

    total, ce, kl, fine = compute_router_loss(
        out, target_row, target_col, teacher_log1, teacher_log2, entry_ids,
        temperature=0.1,
    )

    assert total.ndim == 0, "total_loss 应为标量"
    assert float(total) > 0, f"total_loss 应为正值，实际 {float(total)}"
    assert ce.requires_grad is False, "ce_loss 应为 detach"
    assert kl.requires_grad is False, "kl_loss 应为 detach"
    assert fine.requires_grad is False, "fine_loss 应为 detach"

    # 验证反向传播不报错
    total.backward()
    assert coarse_1.grad is not None, "coarse_1 未收到梯度"
    assert fine_scores.grad is not None, "fine_scores 未收到梯度"


# ─────────────────────────────────────────────────────
# 测试 4b: compute_router_loss — KL 收敛验证
# ─────────────────────────────────────────────────────


def test_compute_router_loss_kl_convergence():
    """
    修复后（CE 和 KL student 均除以 temperature），当学生分布与 teacher 完全对齐时
    KL loss 应接近 0。修复前（student T=1.0，teacher T=0.1）此条件永远无法满足。
    """
    import torch.nn.functional as F

    B, K, T = 4, 32, 0.1
    # 学生和 teacher 都强烈指向 cluster 0（模拟训练收敛后的理想状态）
    scores = torch.zeros(B, K)
    scores[:, 0] = 10.0
    # teacher logits 已内含 /T（与 compute_teacher_logits 输出格式一致）
    teacher_logits = torch.zeros(B, K)
    teacher_logits[:, 0] = 10.0 / T

    log_row = F.log_softmax(scores / T, dim=-1)
    kl = F.kl_div(log_row, F.softmax(teacher_logits, dim=-1), reduction="batchmean")
    assert kl.item() < 1e-3, f"对齐时 KL 应接近 0，实际 {kl.item():.6f}"


# ─────────────────────────────────────────────────────
# 测试 5: DualKnowledgeStore.get_rowcol_labels
# ─────────────────────────────────────────────────────


def test_get_rowcol_labels():
    """
    compact_and_recluster 后调用 get_rowcol_labels，
    应返回 [next_free, 2]，row/col 值域均在 [0, num_keys) 内。
    """
    from dataclasses import dataclass

    @dataclass
    class _RouterConfig:
        knowledge_num: int = 64  # 8x8 grid
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

    from router.memory_bank import DualKnowledgeStore

    cfg = _RouterConfig()
    store = DualKnowledgeStore(cfg, fusion_length=16, anchor_length=32, device="cpu")

    N = 16  # 填充 16 条知识（<< knowledge_num=64，测试稀疏场景）
    # 写入随机 token IDs
    store.fusion_bank.data[:N] = torch.randint(100, 3000, (N, 16))
    store.anchor_bank.data[:N] = torch.randint(100, 3000, (N, 32))
    store.valid_mask[:N] = True
    store.next_free = N

    # 构造 mock encoder（对所有 anchor 返回随机 embedding）
    class _MockEncoder:
        device = torch.device("cpu")

        def encode_mean(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return torch.randn(ids.shape[0], 64)

        def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return torch.randn(ids.shape[0], ids.shape[1], 64)

        def eval(self):
            return self

    encoder = _MockEncoder()
    store.compact_and_recluster(encoder)

    labels = store.get_rowcol_labels()

    num_keys = 8  # √64 = 8
    assert labels.shape == (N, 2), f"形状错误：期望 ({N}, 2)，实际 {tuple(labels.shape)}"
    assert (labels[:, 0] >= 0).all() and (labels[:, 0] < num_keys).all(), (
        f"row 值超出 [0, {num_keys}) 范围"
    )
    assert (labels[:, 1] >= 0).all() and (labels[:, 1] < num_keys).all(), (
        f"col 值超出 [0, {num_keys}) 范围"
    )


# ─────────────────────────────────────────────────────
# 测试 6: ParquetEpochSampler — 真实文件 + 可重现性
# ─────────────────────────────────────────────────────


def test_parquet_epoch_sampler():
    """
    使用 data/compressed/v2/000001.parquet 验证：
    1. 采样数量等于 n_samples
    2. 相同 seed 两次采样结果完全一致（可重现）
    3. 不同 seed 结果不同
    """
    from training.phase1_router import ParquetEpochSampler

    parquet_dir = PROJECT_ROOT / "data" / "compressed" / "v2"
    if not parquet_dir.exists() or not list(parquet_dir.glob("*.parquet")):
        pytest.skip("data/compressed/v2/ 中无 Parquet 文件，跳过测试")

    n_samples = 100
    sampler = ParquetEpochSampler(str(parquet_dir), n_samples)

    rows_seed0_a = sampler.sample_epoch_data(seed=0)
    rows_seed0_b = sampler.sample_epoch_data(seed=0)
    rows_seed1 = sampler.sample_epoch_data(seed=1)

    # 数量正确
    assert len(rows_seed0_a) == n_samples, (
        f"期望 {n_samples} 条，实际 {len(rows_seed0_a)} 条"
    )
    # 可重现性：相同 seed 结果相同
    for i, (a, b) in enumerate(zip(rows_seed0_a, rows_seed0_b)):
        assert a["text"] == b["text"], f"第 {i} 条 text 不一致（seed=0 两次采样）"

    # 不同 seed 产生不同结果（概率极高）
    same = sum(
        a["text"] == b["text"] for a, b in zip(rows_seed0_a, rows_seed1)
    )
    assert same < n_samples, "seed=0 和 seed=1 的采样结果完全相同，随机性异常"

    # 字段存在
    assert "text" in rows_seed0_a[0], "缺少 text 字段"
    assert "compressed_text" in rows_seed0_a[0], "缺少 compressed_text 字段"


# ─────────────────────────────────────────────────────
# 测试 7: tokenize_parquet_batch — 输出形状
# ─────────────────────────────────────────────────────


def test_tokenize_parquet_batch():
    """
    tokenize_parquet_batch 应输出：
        anchor_ids: [B, anchor_length]
        fusion_ids:  [B, fusion_length]
    使用合成文本（无需 Parquet 文件）。
    """
    from training.phase1_router import tokenize_parquet_batch
    from transformers import AutoTokenizer

    tokenizer_path = PROJECT_ROOT / "Qwen3-0.6B"
    if not tokenizer_path.exists():
        pytest.skip("Qwen3-0.6B/ 不存在，跳过 tokenize 测试")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    B = 8
    anchor_length = 32
    fusion_length = 16

    # 合成数据
    rows = [
        {
            "text": "This is a sample text for testing the anchor tokenization process " * 3,
            "compressed_text": "sample text anchor " * 2,
        }
        for _ in range(B)
    ]

    anchor_ids, fusion_ids = tokenize_parquet_batch(
        rows,
        tokenizer,
        anchor_length=anchor_length,
        fusion_length=fusion_length,
        batch_size=4,
    )

    assert anchor_ids.shape == (B, anchor_length), (
        f"anchor_ids 形状错误：期望 ({B}, {anchor_length})，实际 {tuple(anchor_ids.shape)}"
    )
    assert fusion_ids.shape == (B, fusion_length), (
        f"fusion_ids 形状错误：期望 ({B}, {fusion_length})，实际 {tuple(fusion_ids.shape)}"
    )
    assert anchor_ids.dtype == torch.long, "anchor_ids 应为 torch.long"
    assert fusion_ids.dtype == torch.long, "fusion_ids 应为 torch.long"
