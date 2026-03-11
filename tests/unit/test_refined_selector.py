"""
tests/unit/test_refined_selector.py — RefinedSelector 单元测试

测试覆盖：
    - 初始化：score_head 形状、scale 初始值、num_heads 整除约束
    - forward：输出形状（scores [B,C], best_idx [B]）
    - best_idx 值域：[0, C)
    - 数值稳定性：无 NaN/Inf
    - 单候选：C=1 时 best_idx 全为 0
    - mask：被遮蔽的候选不被选中
"""

from __future__ import annotations

import torch
import pytest

from router.refined_selector import RefinedSelector


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

ADAPTER_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 2
BATCH_SIZE = 4
NUM_CANDS = 16


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────


def _make_selector(
    adapter_dim: int = ADAPTER_DIM,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
) -> RefinedSelector:
    """构造测试用 RefinedSelector。"""
    return RefinedSelector(
        adapter_dim=adapter_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )


# ─────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────


class TestRefinedSelectorInit:
    """测试 RefinedSelector 初始化：参数形状与设计约束。"""

    def test_score_head_shape(self) -> None:
        """score_head 权重形状应为 [1, adapter_dim]。"""
        sel = _make_selector()
        assert sel.score_head.weight.shape == (1, ADAPTER_DIM), (
            f"期望 [1, {ADAPTER_DIM}]，实际 {sel.score_head.weight.shape}"
        )
        assert sel.score_head.bias.shape == (1,)

    def test_scale_init_value(self) -> None:
        """scale 初始值应为 10.0。"""
        sel = _make_selector()
        assert abs(sel.scale.item() - 10.0) < 1e-6, (
            f"scale 初始值期望 10.0，实际 {sel.scale.item()}"
        )

    def test_scale_is_learnable(self) -> None:
        """scale 应为 nn.Parameter（可学习）。"""
        sel = _make_selector()
        assert isinstance(sel.scale, torch.nn.Parameter)
        assert sel.scale.requires_grad

    def test_invalid_num_heads_raises(self) -> None:
        """adapter_dim % num_heads ≠ 0 时应抛出 AssertionError。"""
        with pytest.raises(AssertionError, match="必须能被"):
            RefinedSelector(adapter_dim=512, num_heads=7, num_layers=2)

    def test_transformer_config(self) -> None:
        """TransformerEncoderLayer 应使用 batch_first=True。"""
        sel = _make_selector()
        # 访问内部 encoder layer 验证 batch_first
        layer = sel.transformer.layers[0]
        assert layer.self_attn.batch_first is True

    def test_num_transformer_layers(self) -> None:
        """Transformer 层数应与 num_layers 参数一致。"""
        sel = _make_selector(num_layers=2)
        assert len(sel.transformer.layers) == 2


class TestRefinedSelectorForward:
    """测试 RefinedSelector forward：输出形状、值域、数值稳定性。"""

    @pytest.fixture()
    def selector_and_inputs(self) -> tuple:
        """返回已初始化的 selector 和随机输入。"""
        sel = _make_selector()
        sel.eval()
        torch.manual_seed(42)
        query_vec = torch.randn(BATCH_SIZE, ADAPTER_DIM)
        cand_vecs = torch.randn(BATCH_SIZE, NUM_CANDS, ADAPTER_DIM)
        return sel, query_vec, cand_vecs

    def test_output_shapes(self, selector_and_inputs: tuple) -> None:
        """scores 形状为 [B, C]，best_idx 形状为 [B]。"""
        sel, query_vec, cand_vecs = selector_and_inputs
        with torch.no_grad():
            scores, best_idx = sel(query_vec, cand_vecs)
        assert scores.shape == (BATCH_SIZE, NUM_CANDS), (
            f"scores 形状期望 {(BATCH_SIZE, NUM_CANDS)}，实际 {scores.shape}"
        )
        assert best_idx.shape == (BATCH_SIZE,), (
            f"best_idx 形状期望 {(BATCH_SIZE,)}，实际 {best_idx.shape}"
        )

    def test_best_idx_in_range(self, selector_and_inputs: tuple) -> None:
        """best_idx 所有值应在 [0, C) 范围内。"""
        sel, query_vec, cand_vecs = selector_and_inputs
        with torch.no_grad():
            _, best_idx = sel(query_vec, cand_vecs)
        assert (best_idx >= 0).all(), "best_idx 存在负值"
        assert (best_idx < NUM_CANDS).all(), f"best_idx 存在 ≥ {NUM_CANDS} 的值"

    def test_no_nan_inf(self, selector_and_inputs: tuple) -> None:
        """scores 和 best_idx 中不应存在 NaN 或 Inf。"""
        sel, query_vec, cand_vecs = selector_and_inputs
        with torch.no_grad():
            scores, best_idx = sel(query_vec, cand_vecs)
        assert not torch.isnan(scores).any(), "scores 中存在 NaN"
        assert not torch.isinf(scores).any(), "scores 中存在 Inf"
        assert not torch.isnan(best_idx.float()).any()

    def test_best_idx_is_argmax(self, selector_and_inputs: tuple) -> None:
        """best_idx 应等于 scores 的 argmax。"""
        sel, query_vec, cand_vecs = selector_and_inputs
        with torch.no_grad():
            scores, best_idx = sel(query_vec, cand_vecs)
        expected = scores.argmax(dim=-1)
        assert torch.equal(best_idx, expected), (
            "best_idx 与 scores.argmax(dim=-1) 不一致"
        )


class TestRefinedSelectorSingleCandidate:
    """测试 C=1（单候选）边界情况。"""

    def test_single_candidate_best_idx_is_zero(self) -> None:
        """C=1 时所有 batch item 的 best_idx 应为 0。"""
        sel = _make_selector()
        sel.eval()
        query_vec = torch.randn(BATCH_SIZE, ADAPTER_DIM)
        cand_vecs = torch.randn(BATCH_SIZE, 1, ADAPTER_DIM)
        with torch.no_grad():
            scores, best_idx = sel(query_vec, cand_vecs)
        assert scores.shape == (BATCH_SIZE, 1)
        assert (best_idx == 0).all(), f"C=1 时 best_idx 应全为 0，实际: {best_idx}"


class TestRefinedSelectorMask:
    """测试 mask 参数：被遮蔽候选不应被选中。"""

    def test_masked_candidates_not_selected(self) -> None:
        """
        若 mask 中只有第 0 个候选有效，best_idx 应全为 0。
        """
        sel = _make_selector()
        sel.eval()
        torch.manual_seed(99)
        query_vec = torch.randn(BATCH_SIZE, ADAPTER_DIM)
        cand_vecs = torch.randn(BATCH_SIZE, NUM_CANDS, ADAPTER_DIM)
        # 仅第 0 个候选有效
        mask = torch.zeros(BATCH_SIZE, NUM_CANDS, dtype=torch.bool)
        mask[:, 0] = True
        with torch.no_grad():
            scores, best_idx = sel(query_vec, cand_vecs, mask=mask)
        assert (best_idx == 0).all(), (
            f"只有候选 0 有效时，best_idx 应全为 0，实际: {best_idx}"
        )

    def test_mask_shape_mismatch_raises(self) -> None:
        """mask 形状与 cand_vecs 前两维不匹配时应抛出 AssertionError。"""
        sel = _make_selector()
        query_vec = torch.randn(BATCH_SIZE, ADAPTER_DIM)
        cand_vecs = torch.randn(BATCH_SIZE, NUM_CANDS, ADAPTER_DIM)
        # 错误的 mask 形状
        bad_mask = torch.ones(BATCH_SIZE, NUM_CANDS + 1, dtype=torch.bool)
        with pytest.raises(AssertionError):
            sel(query_vec, cand_vecs, mask=bad_mask)

    def test_dim_mismatch_raises(self) -> None:
        """query_vec 或 cand_vecs 维度不匹配时应抛出 AssertionError。"""
        sel = _make_selector()
        # query_vec 维度错误
        with pytest.raises(AssertionError):
            sel(
                torch.randn(BATCH_SIZE, ADAPTER_DIM + 1),
                torch.randn(BATCH_SIZE, NUM_CANDS, ADAPTER_DIM),
            )
        # cand_vecs 维度错误
        with pytest.raises(AssertionError):
            sel(
                torch.randn(BATCH_SIZE, ADAPTER_DIM),
                torch.randn(BATCH_SIZE, NUM_CANDS, ADAPTER_DIM + 1),
            )
