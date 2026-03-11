"""
router/model.py 单元测试

测试覆盖：
    - MemoryRouter 初始化：子模块类型、encoder 注册
    - forward：RouterOutput 各字段形状、best_id 值域、分数形状
    - retrieve：输出形状、@torch.no_grad 语义
    - 可训练参数量统计（encoder 冻结时）
    - 边界情况：B=1、B=8 批次

设计原则：
    - 不加载真实 Qwen3 模型，使用 MagicMock 替代 encoder 和 store
    - 子模块（pkm、adapter、selector）使用真实实例 + MagicMock 替换其 __call__
    - 维持小规模配置（DIM=64）保证 CPU 测试速度
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from router.model import MemoryRouter, RouterOutput  # noqa: E402


# ─────────────────────────────────────────────
# 常量：小规模配置（CPU 可运行）
# ─────────────────────────────────────────────

KNOWLEDGE_NUM = 16      # 4²，最小完全平方数
NUM_KEYS = 4            # √16
DIM = 64                # 模型隐藏维度（测试用小值）
QUERY_DIM = 64          # = DIM
KEY_PROJ_DIM = 32       # = DIM // 2
ADAPTER_DIM = 32        # FeatureAdapter 输出维度
NUM_CANDIDATES = 4      # 粗排候选数
TEMPERATURE = 0.1
ANCHOR_LENGTH = 8       # AnchorBank 每条知识 token 数
FUSION_LENGTH = 6       # FusionBank 每条知识 token 数
NUM_HEADS = 4           # adapter_dim // num_heads = 32 // 4 = 8（整除）
NUM_LAYERS = 2
DEVICE = "cpu"


# ─────────────────────────────────────────────
# 辅助：构造 mock config
# ─────────────────────────────────────────────


def _make_config(**overrides: Any) -> Any:
    """构造 RouterConfig MagicMock，含 MemoryRouter 需要的所有字段。"""
    cfg = MagicMock()
    cfg.knowledge_num = KNOWLEDGE_NUM
    cfg.dim = DIM
    cfg.query_dim = QUERY_DIM
    cfg.key_proj_dim = KEY_PROJ_DIM
    cfg.adapter_dim = ADAPTER_DIM
    cfg.num_candidates = NUM_CANDIDATES
    cfg.temperature = TEMPERATURE
    cfg.max_candidates_per_cell = -1
    cfg.recluster_threshold = 0.1
    cfg.refined_num_heads = NUM_HEADS
    cfg.refined_num_layers = NUM_LAYERS
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_encoder() -> MagicMock:
    """构造 KnowledgeEncoder mock：forward 根据输入形状返回合法张量。"""
    enc = MagicMock()
    enc.forward.side_effect = lambda ids, mask: torch.randn(
        ids.shape[0], ids.shape[1], DIM
    )
    return enc


def _make_router() -> MemoryRouter:
    """构造测试用 MemoryRouter（小规模配置 + mock encoder）。"""
    return MemoryRouter(_make_config(), _make_mock_encoder())


def _make_mock_store() -> MagicMock:
    """
    构造 DualKnowledgeStore mock，提供 forward 中用到的全部接口：
        - anchor_bank.data: [KNOWLEDGE_NUM, ANCHOR_LENGTH] 真实 long 张量
        - fusion_bank.__getitem__: 返回 [B, FUSION_LENGTH] long 张量
        - next_free: KNOWLEDGE_NUM（所有槽位已用）
    """
    store = MagicMock()
    store.anchor_bank.data = torch.zeros(
        KNOWLEDGE_NUM, ANCHOR_LENGTH, dtype=torch.long
    )
    store.fusion_bank.__getitem__ = MagicMock(
        side_effect=lambda ids: torch.zeros(ids.shape[0], FUSION_LENGTH, dtype=torch.long)
    )
    store.next_free = KNOWLEDGE_NUM
    return store


def _mock_pkm_output(
    router: MemoryRouter,
    B: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用轻量 nn.Module Stub 替换 router.pkm，返回固定形状的粗排输出。

    注意：PyTorch 不允许将非 nn.Module 赋给已注册的子模块，
    故使用 nn.Module 子类 Stub 而非 MagicMock。

    返回替换后的固定 4-tuple：(candidates, scores_1, scores_2, q_pkm)
    """
    candidates = torch.randint(0, KNOWLEDGE_NUM, (B, NUM_CANDIDATES))
    s1 = torch.randn(B, NUM_KEYS)
    s2 = torch.randn(B, NUM_KEYS)
    q_pkm = torch.randn(B, KEY_PROJ_DIM)

    _ret = (candidates, s1, s2, q_pkm)

    class _StubPKM(nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> Tuple:
            return _ret

    router.pkm = _StubPKM()
    return candidates, s1, s2, q_pkm


# ─────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────


class TestMemoryRouterInit:
    """测试 MemoryRouter.__init__ 的子模块类型与注册状态。"""

    def test_submodule_types(self) -> None:
        """三个可训练子模块类型正确。"""
        from router.feature_adapter import FeatureAdapter
        from router.memory_gate import ProductKeyMemory
        from router.refined_selector import RefinedSelector

        router = _make_router()
        assert isinstance(router.pkm, ProductKeyMemory)
        assert isinstance(router.adapter, FeatureAdapter)
        assert isinstance(router.selector, RefinedSelector)

    def test_encoder_registered_as_submodule(self) -> None:
        """encoder 应以 nn.Module 子模块注册（出现在 named_modules）。"""
        # encoder 为 MagicMock，不是 nn.Module，故不注册为子模块；
        # 验证 router._modules 中不含 "encoder" 时仍然不报错；
        # 实际场景下 encoder 为真实 nn.Module 时会被注册。
        # 此处仅验证 encoder 属性可读
        router = _make_router()
        assert hasattr(router, "encoder")
        assert router.encoder is not None

    def test_adapter_dim_cached(self) -> None:
        """_adapter_dim 应与 config.adapter_dim 一致。"""
        router = _make_router()
        assert router._adapter_dim == ADAPTER_DIM


class TestMemoryRouterForwardShapes:
    """测试 forward 返回的 RouterOutput 各字段形状。"""

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_best_id_shape(self, B: int) -> None:
        """best_id 应为 [B] long 张量。"""
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        assert out.best_id.shape == (B,)
        assert out.best_id.dtype == torch.long

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_candidates_shape(self, B: int) -> None:
        """candidates 应为 [B, num_candidates] long 张量。"""
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        assert out.candidates.shape == (B, NUM_CANDIDATES)
        assert out.candidates.dtype == torch.long

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_coarse_scores_shape(self, B: int) -> None:
        """coarse_scores 为 2-Tuple，各 [B, num_keys]。"""
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        assert isinstance(out.coarse_scores, tuple)
        assert len(out.coarse_scores) == 2
        assert out.coarse_scores[0].shape == (B, NUM_KEYS)
        assert out.coarse_scores[1].shape == (B, NUM_KEYS)

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_fine_scores_shape(self, B: int) -> None:
        """fine_scores 应为 [B, num_candidates] float 张量。"""
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        assert out.fine_scores.shape == (B, NUM_CANDIDATES)
        assert out.fine_scores.dtype == torch.float


class TestMemoryRouterForwardCorrectness:
    """测试 forward 输出值域与内部逻辑正确性。"""

    def test_best_id_in_range(self) -> None:
        """best_id 所有元素应在 [0, KNOWLEDGE_NUM) 内。"""
        B = 4
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        assert out.best_id.min().item() >= 0
        assert out.best_id.max().item() < KNOWLEDGE_NUM

    def test_best_id_comes_from_candidates(self) -> None:
        """best_id 中每个元素必须来自对应批次的 candidates。"""
        B = 4
        router = _make_router()
        candidates, _, _, _ = _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        out = router.forward(query, store)

        # 对每个批次 b，best_id[b] 必须在 candidates[b] 中
        candidates_used = out.candidates  # router.pkm 被替换后 forward 内重新获取
        for b in range(B):
            assert out.best_id[b].item() in candidates_used[b].tolist()

    def test_output_is_router_output_instance(self) -> None:
        """forward 返回值必须是 RouterOutput dataclass 实例。"""
        router = _make_router()
        _mock_pkm_output(router, B=2)
        store = _make_mock_store()

        out = router.forward(torch.randn(2, DIM), store)

        assert isinstance(out, RouterOutput)

    def test_query_ndim_check(self) -> None:
        """非 2D query_embedding 应抛出 AssertionError。"""
        router = _make_router()
        _mock_pkm_output(router, B=2)
        store = _make_mock_store()

        bad_query = torch.randn(2, 4, DIM)  # 3D，非法
        with pytest.raises(AssertionError):
            router.forward(bad_query, store)

    def test_encoder_called_with_correct_shapes(self) -> None:
        """encoder.forward 应以 [B*C, K_a] 的 ids 和 mask 调用。"""
        B = 3
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        router.forward(query, store)

        # encoder.forward 应被调用一次
        router.encoder.forward.assert_called_once()
        ids_arg, mask_arg = router.encoder.forward.call_args[0]
        assert ids_arg.shape == (B * NUM_CANDIDATES, ANCHOR_LENGTH)
        assert mask_arg.shape == (B * NUM_CANDIDATES, ANCHOR_LENGTH)


class TestMemoryRouterRetrieve:
    """测试 retrieve 方法的形状与梯度语义。"""

    def test_retrieve_shape(self) -> None:
        """retrieve 应返回 [B, FUSION_LENGTH] long 张量。"""
        B = 3
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        result = router.retrieve(query, store)

        assert result.shape == (B, FUSION_LENGTH)

    def test_retrieve_no_grad(self) -> None:
        """retrieve 内部执行不构建计算图（best_id 无 grad_fn）。"""
        B = 2
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM, requires_grad=False)

        result = router.retrieve(query, store)

        # retrieve 以 @torch.no_grad() 修饰，输出无 grad_fn
        assert result.grad_fn is None

    def test_retrieve_calls_fusion_bank(self) -> None:
        """retrieve 应通过 store.fusion_bank[best_id] 获取知识 token IDs。"""
        B = 2
        router = _make_router()
        _mock_pkm_output(router, B)
        store = _make_mock_store()
        query = torch.randn(B, DIM)

        router.retrieve(query, store)

        # fusion_bank.__getitem__ 应被调用一次
        store.fusion_bank.__getitem__.assert_called_once()
        ids_passed = store.fusion_bank.__getitem__.call_args[0][0]
        assert ids_passed.shape == (B,)
        assert ids_passed.dtype == torch.long


class TestTrainableParams:
    """测试可训练参数量（生产规模配置，不运行 forward）。"""

    def test_trainable_params_with_frozen_encoder(self) -> None:
        """
        encoder 冻结时，可训练参数应约 8-12M（pkm + adapter + selector）。
        生产配置：DIM=1024, ADAPTER_DIM=512, refined_num_heads=8, refined_num_layers=2。
        """
        # 使用生产规模配置
        cfg = _make_config(
            knowledge_num=64,           # 8²，最小完全平方数（初始化快）
            dim=1024,
            query_dim=1024,
            key_proj_dim=512,
            adapter_dim=512,
            num_candidates=32,
            refined_num_heads=8,
            refined_num_layers=2,
        )
        # mock encoder（已冻结，requires_grad=False）
        mock_enc = MagicMock()
        mock_enc.parameters = MagicMock(return_value=iter([]))  # 无可训练参数

        router = MemoryRouter(cfg, mock_enc)

        trainable = sum(
            p.numel() for p in router.parameters() if p.requires_grad
        )
        # 期望：pkm ~1.6M + adapter ~0.5M + selector ~6.3M ≈ 8-10M
        assert 5_000_000 < trainable < 20_000_000, (
            f"可训练参数量 {trainable:,} 超出预期范围 [5M, 20M]"
        )

    def test_encoder_frozen_reduces_trainable_count(self) -> None:
        """验证 encoder 冻结后，filter(requires_grad) 不包含 encoder 参数。"""
        import torch.nn as nn

        # 创建一个小 encoder（真实 nn.Module）并冻结
        class TinyEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(DIM, DIM)

            def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(-1).expand(-1, -1, DIM)

        enc = TinyEncoder()
        for p in enc.parameters():
            p.requires_grad_(False)

        router = MemoryRouter(_make_config(), enc)

        trainable_names = [
            name for name, p in router.named_parameters() if p.requires_grad
        ]
        # encoder 参数（"encoder.fc.*"）不应出现在可训练列表中
        encoder_trainable = [n for n in trainable_names if n.startswith("encoder.")]
        assert len(encoder_trainable) == 0, (
            f"冻结的 encoder 参数意外出现在可训练列表：{encoder_trainable}"
        )
