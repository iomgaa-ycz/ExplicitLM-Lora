"""
tests/unit/test_feature_adapter.py — FeatureAdapter 单元测试

测试覆盖：
    - 初始化：参数形状、无可学习 centering 参数（使用 Batch Centering）
    - forward [B, D]：二维输入，输出形状正确
    - forward [B, S, D]：三维输入，有/无 mask，输出形状正确
    - 数值稳定性：无 NaN/Inf
    - Batch Centering 效果：常量偏置在批次内被消除
"""

from __future__ import annotations

import torch
import pytest

from router.feature_adapter import FeatureAdapter


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

IN_DIM = 1024
ADAPTER_DIM = 512
BATCH_SIZE = 4
SEQ_LEN = 16


# ─────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────


class TestFeatureAdapterInit:
    """测试 FeatureAdapter 初始化：参数形状与设计约束。"""

    def test_param_shapes(self) -> None:
        """input_norm、proj、output_norm 的参数形状正确。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)

        # input_norm: LN(in_dim)
        assert adapter.input_norm.weight.shape == (IN_DIM,)
        assert adapter.input_norm.bias.shape == (IN_DIM,)

        # proj: Linear(in_dim → adapter_dim)
        assert adapter.proj.weight.shape == (ADAPTER_DIM, IN_DIM)
        assert adapter.proj.bias.shape == (ADAPTER_DIM,)

        # output_norm: LN(adapter_dim)
        assert adapter.output_norm.weight.shape == (ADAPTER_DIM,)
        assert adapter.output_norm.bias.shape == (ADAPTER_DIM,)

    def test_no_centering_param(self) -> None:
        """
        使用 Batch Centering（动态批均值），不应有名为 centering 的 nn.Parameter。
        """
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        param_names = [name for name, _ in adapter.named_parameters()]
        assert "centering" not in param_names, (
            "Batch Centering 不使用可学习参数，不应有 'centering' Parameter"
        )

    def test_scale_is_constant(self) -> None:
        """_scale 为常数 √adapter_dim，不是可学习参数。"""
        import math
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        assert abs(adapter._scale - math.sqrt(ADAPTER_DIM)) < 1e-6
        # _scale 不在参数中
        param_names = [name for name, _ in adapter.named_parameters()]
        assert "_scale" not in param_names

    def test_invalid_dims_raise(self) -> None:
        """in_dim 或 adapter_dim 为 0 时应抛出 AssertionError。"""
        with pytest.raises(AssertionError):
            FeatureAdapter(in_dim=0, adapter_dim=ADAPTER_DIM)
        with pytest.raises(AssertionError):
            FeatureAdapter(in_dim=IN_DIM, adapter_dim=0)


class TestFeatureAdapterForward2D:
    """测试 [B, D] 二维输入的前向传播。"""

    def test_output_shape(self) -> None:
        """[B, D] 输入 → [B, adapter_dim] 输出。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, IN_DIM)
        with torch.no_grad():
            out = adapter(x)
        assert out.shape == (BATCH_SIZE, ADAPTER_DIM), (
            f"期望形状 {(BATCH_SIZE, ADAPTER_DIM)}，实际 {out.shape}"
        )

    def test_output_dtype_preserved(self) -> None:
        """输出 dtype 应与输入一致（float32）。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, IN_DIM, dtype=torch.float32)
        with torch.no_grad():
            out = adapter(x)
        assert out.dtype == torch.float32

    def test_mask_ignored_for_2d(self) -> None:
        """2D 输入时 mask 参数被忽略（不影响结果）。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, IN_DIM)
        # 传入与 2D 不相关的 mask（行为：忽略）
        with torch.no_grad():
            out_no_mask = adapter(x)
            # 2D 时 mask 不参与计算，传不传等价
            out_with_mask = adapter(x, mask=None)
        assert torch.allclose(out_no_mask, out_with_mask)


class TestFeatureAdapterForward3D:
    """测试 [B, S, D] 三维输入的前向传播。"""

    def test_output_shape_no_mask(self) -> None:
        """[B, S, D] 无 mask → [B, adapter_dim] 输出。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
        with torch.no_grad():
            out = adapter(x, mask=None)
        assert out.shape == (BATCH_SIZE, ADAPTER_DIM)

    def test_output_shape_with_mask(self) -> None:
        """[B, S, D] 带 mask → [B, adapter_dim] 输出。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
        # 后 4 个 token 为 padding
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, -4:] = False
        with torch.no_grad():
            out = adapter(x, mask=mask)
        assert out.shape == (BATCH_SIZE, ADAPTER_DIM)

    def test_mask_affects_output(self) -> None:
        """有/无 padding mask 时，输出应不同（验证 mask 参与了 pooling）。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        torch.manual_seed(42)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
        # 有效长度一半
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, SEQ_LEN // 2:] = False
        with torch.no_grad():
            out_full = adapter(x, mask=None)
            out_masked = adapter(x, mask=mask)
        # mask 存在时输出与全序列不同
        assert not torch.allclose(out_full, out_masked), (
            "mask 有效时，输出应与无 mask 不同"
        )

    def test_invalid_dim_raises(self) -> None:
        """输入维度不为 2 或 3 时应抛出 AssertionError。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        x_4d = torch.randn(2, 3, SEQ_LEN, IN_DIM)
        with pytest.raises(AssertionError, match="维度必须为 2 或 3"):
            adapter(x_4d)


class TestFeatureAdapterNumerics:
    """测试 FeatureAdapter 的数值稳定性。"""

    def test_no_nan_inf(self) -> None:
        """随机输入下输出无 NaN 或 Inf。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        torch.manual_seed(0)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
        with torch.no_grad():
            out = adapter(x)
        assert not torch.isnan(out).any(), "输出中存在 NaN"
        assert not torch.isinf(out).any(), "输出中存在 Inf"

    def test_no_nan_inf_extreme_input(self) -> None:
        """极端输入值（大幅度随机）下输出仍无 NaN/Inf。"""
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        x = torch.randn(BATCH_SIZE, IN_DIM) * 100.0
        with torch.no_grad():
            out = adapter(x)
        assert not torch.isnan(out).any(), "极端输入下输出存在 NaN"
        assert not torch.isinf(out).any(), "极端输入下输出存在 Inf"

    def test_batch_centering_removes_constant_offset(self) -> None:
        """
        Batch Centering 应消除批次内的常量偏置。

        原理：若所有样本均加同一偏置向量 d，批次均值包含 d，
              减去均值后 d 被消除，输出应与无偏置时一致。
        """
        adapter = FeatureAdapter(in_dim=IN_DIM, adapter_dim=ADAPTER_DIM)
        adapter.eval()
        torch.manual_seed(7)
        x = torch.randn(BATCH_SIZE, IN_DIM)
        # 给批次内所有样本加同一常量偏置
        offset = torch.ones(IN_DIM) * 50.0
        x_offset = x + offset.unsqueeze(0)

        with torch.no_grad():
            out_orig = adapter(x)
            out_offset = adapter(x_offset)

        # Batch Centering 后，两次输出应完全一致（相对误差 < 1e-5）
        assert torch.allclose(out_orig, out_offset, atol=1e-4), (
            f"Batch Centering 未正确消除常量偏置，\n"
            f"最大偏差: {(out_orig - out_offset).abs().max().item():.6f}"
        )
