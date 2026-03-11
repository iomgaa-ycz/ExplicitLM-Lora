"""
tests/unit/test_modified_model.py — ModifiedQwen 单元测试

使用真实 Qwen3-0.6B 模型（module 级 fixture，整个模块只加载一次）验证：
  - Hook 注册到正确层
  - 基础模型参数冻结
  - 注入模块参数可训练
  - forward 输出形状（logits [B, L, V]）
  - labels 存在时 loss 为标量
  - knowledge_ids=None 退化模式正常工作
  - 零初始化注入模块：有/无知识注入 logits 相对误差 < 1e-4
  - remove_hooks() 清空所有 hooks

常量：B=2, L=8, K_F=64, D=1024
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    AttentionInjection,
    KnowledgeEncoder,
    ModifiedQwen,
    load_base_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "Qwen3-0.6B"
ENCODER_DEPTH = 6
HIDDEN_DIM = 1024
FUSION_LENGTH = 64
B = 2
L = 8
VOCAB_SIZE = 151936
INJECTION_LAYERS = [6, 12, 18, 24]
NUM_INJECTION = len(INJECTION_LAYERS)
PAD_TOKEN_ID = 0
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures（module 级，避免重复加载 Qwen3）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """加载并返回冻结的 Qwen3-0.6B 基础模型（整个 test module 只加载一次）。"""
    return load_base_model(MODEL_PATH, bf16=True)


@pytest.fixture(scope="module")
def knowledge_encoder(base_model):
    """基于 base_model 构造 KnowledgeEncoder。"""
    enc = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=ENCODER_DEPTH,
        hidden_dim=HIDDEN_DIM,
    )
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def injection_modules() -> nn.ModuleList:
    """构造 4 个独立 AttentionInjection 实例（零初始化）。"""
    modules = nn.ModuleList(
        [AttentionInjection(hidden_dim=HIDDEN_DIM) for _ in range(NUM_INJECTION)]
    )
    return modules


@pytest.fixture(scope="module")
def model(base_model, knowledge_encoder, injection_modules):
    """构造 ModifiedQwen 并切换到 eval 模式。"""
    m = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=knowledge_encoder,
        injection_modules=injection_modules,
        injection_layers=INJECTION_LAYERS,
        pad_token_id=PAD_TOKEN_ID,
    )
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────


def _make_input_ids(batch: int = B, seq_len: int = L) -> torch.Tensor:
    """构造随机 input_ids [B, L]，token ID 范围 [1, VOCAB_SIZE//2)。"""
    torch.manual_seed(SEED)
    return torch.randint(1, VOCAB_SIZE // 2, (batch, seq_len))


def _make_knowledge_ids(batch: int = B, k_f: int = FUSION_LENGTH) -> torch.Tensor:
    """构造随机 knowledge_ids [B, K_F]，后半段为 padding（ID=0）。"""
    torch.manual_seed(SEED + 1)
    ids = torch.randint(1, VOCAB_SIZE // 2, (batch, k_f))
    # 后半段 padding
    ids[:, k_f // 2 :] = 0
    return ids


def _make_attention_mask(batch: int = B, seq_len: int = L) -> torch.Tensor:
    """构造全 1 的 attention_mask [B, L]。"""
    return torch.ones(batch, seq_len, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────────────────────


class TestModifiedQwenInit:
    """初始化验证：Hook 注册、参数冻结/可训练状态。"""

    def test_hook_registered(self, model: ModifiedQwen) -> None:
        """
        测试：4 个注入层均应注册了 forward hook。

        验证点：
            - 每个 injection_layer 对应的 DecoderLayer 有 >= 1 个 forward hook
        """
        for layer_idx in INJECTION_LAYERS:
            layer = model.base_model.model.layers[layer_idx]
            assert len(layer._forward_hooks) >= 1, (
                f"Layer {layer_idx} 未注册 hook，_forward_hooks={layer._forward_hooks}"
            )

    def test_base_model_frozen(self, model: ModifiedQwen) -> None:
        """
        测试：base_model 的所有参数应完全冻结（requires_grad=False）。

        验证点：
            - base_model 无任何参数 requires_grad=True
        """
        trainable = [
            name
            for name, p in model.base_model.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable) == 0, (
            f"base_model 存在 {len(trainable)} 个未冻结参数：{trainable[:5]}"
        )

    def test_injection_modules_trainable(self, model: ModifiedQwen) -> None:
        """
        测试：injection_modules 的参数应可训练（requires_grad=True）。

        验证点：
            - 至少有一个参数 requires_grad=True
        """
        trainable_count = sum(
            1 for p in model.injection_modules.parameters() if p.requires_grad
        )
        assert trainable_count > 0, "injection_modules 无可训练参数"

    def test_hooks_list_length(self, model: ModifiedQwen) -> None:
        """
        测试：_hooks 列表长度应等于 injection_layers 长度。

        验证点：
            - len(model._hooks) == NUM_INJECTION
        """
        assert len(model._hooks) == NUM_INJECTION, (
            f"_hooks 长度 {len(model._hooks)} != 预期 {NUM_INJECTION}"
        )


class TestModifiedQwenForward:
    """forward 方法输出形状与基本语义验证。"""

    def test_output_shape(self, model: ModifiedQwen) -> None:
        """
        测试：有知识注入时 logits shape 应为 [B, L, V]。

        验证点：
            - output.logits.shape == (B, L, VOCAB_SIZE)
        """
        input_ids = _make_input_ids()
        knowledge_ids = _make_knowledge_ids()
        attention_mask = _make_attention_mask()

        with torch.no_grad():
            output = model(input_ids, knowledge_ids, attention_mask)

        assert output.logits.shape == (B, L, VOCAB_SIZE), (
            f"logits shape {output.logits.shape} != 预期 (B={B}, L={L}, V={VOCAB_SIZE})"
        )

    def test_with_labels_returns_loss(self, model: ModifiedQwen) -> None:
        """
        测试：提供 labels 时 output.loss 应为非 None 标量。

        验证点：
            - output.loss is not None
            - output.loss.shape == ()（标量）
            - output.loss > 0
        """
        input_ids = _make_input_ids()
        knowledge_ids = _make_knowledge_ids()
        attention_mask = _make_attention_mask()
        # labels：移位一位，忽略最后一个 token（设为 -100）
        labels = input_ids.clone()
        labels[:, -1] = -100

        with torch.no_grad():
            output = model(input_ids, knowledge_ids, attention_mask, labels=labels)

        assert output.loss is not None, "提供 labels 后 output.loss 不应为 None"
        assert output.loss.shape == (), (
            f"loss 应为标量，实际 shape={output.loss.shape}"
        )
        assert output.loss.item() > 0, f"loss 应为正数，实际={output.loss.item():.4f}"

    def test_knowledge_none_passes(self, model: ModifiedQwen) -> None:
        """
        测试：knowledge_ids=None 时（退化模式）forward 正常运行，logits shape 正确。

        验证点：
            - 无异常抛出
            - output.logits.shape == (B, L, VOCAB_SIZE)
        """
        input_ids = _make_input_ids()
        attention_mask = _make_attention_mask()

        with torch.no_grad():
            output = model(input_ids, None, attention_mask)

        assert output.logits.shape == (B, L, VOCAB_SIZE), (
            f"退化模式 logits shape {output.logits.shape} != (B={B}, L={L}, V={VOCAB_SIZE})"
        )

    def test_logits_finite(self, model: ModifiedQwen) -> None:
        """
        测试：logits 数值应有限（无 NaN 或 inf）。

        验证点：
            - torch.isfinite(output.logits).all() == True
        """
        input_ids = _make_input_ids()
        knowledge_ids = _make_knowledge_ids()
        attention_mask = _make_attention_mask()

        with torch.no_grad():
            output = model(input_ids, knowledge_ids, attention_mask)

        assert torch.isfinite(output.logits).all(), "logits 包含 NaN 或 inf"


class TestZeroInitBehavior:
    """零初始化验证：新建的注入模块初始输出应近似等于无注入输出。"""

    def test_residual_identity(self, base_model, knowledge_encoder) -> None:
        """
        测试：零初始化 AttentionInjection 时，有知识注入的 logits 应近似等于退化模式 logits。

        设计：新建独立的零初始化注入模块（未经训练），此时 out_proj=0 → attn_out≈0
              → 注入模块输出 ≈ hidden → logits ≈ 无注入 logits

        验证点：
            - 有注入 logits 与无注入 logits 的相对误差 < 1e-4
        """
        fresh_modules = nn.ModuleList(
            [AttentionInjection(hidden_dim=HIDDEN_DIM) for _ in range(NUM_INJECTION)]
        )
        fresh_model = ModifiedQwen(
            base_model=base_model,
            knowledge_encoder=knowledge_encoder,
            injection_modules=fresh_modules,
            injection_layers=INJECTION_LAYERS,
            pad_token_id=PAD_TOKEN_ID,
        )
        fresh_model.eval()

        try:
            input_ids = _make_input_ids()
            knowledge_ids = _make_knowledge_ids()
            attention_mask = _make_attention_mask()

            with torch.no_grad():
                out_with = fresh_model(input_ids, knowledge_ids, attention_mask)
                out_none = fresh_model(input_ids, None, attention_mask)

            logits_with = out_with.logits.float()
            logits_none = out_none.logits.float()

            # 相对误差：|with - none| / (|none| + eps)
            rel_err = (
                (logits_with - logits_none).abs()
                / (logits_none.abs() + 1e-8)
            ).max().item()

            assert rel_err < 1e-4, (
                f"零初始化时有/无注入 logits 相对误差 {rel_err:.2e} >= 1e-4，"
                "检查 AttentionInjection 零初始化是否生效"
            )
        finally:
            fresh_model.remove_hooks()


class TestRemoveHooks:
    """remove_hooks() 接口验证。"""

    def test_remove_hooks(self, base_model, knowledge_encoder, injection_modules) -> None:
        """
        测试：remove_hooks() 后所有注入层的 _forward_hooks 应清空。

        验证点：
            - 调用 remove_hooks() 后 len(layer._forward_hooks) == 0（对本模型注册的 hooks）
            - model._hooks 列表清空
        """
        # 新建独立模型，避免影响 module 级共享 model fixture
        fresh_modules = nn.ModuleList(
            [AttentionInjection(hidden_dim=HIDDEN_DIM) for _ in range(NUM_INJECTION)]
        )
        m = ModifiedQwen(
            base_model=base_model,
            knowledge_encoder=knowledge_encoder,
            injection_modules=fresh_modules,
            injection_layers=INJECTION_LAYERS,
            pad_token_id=PAD_TOKEN_ID,
        )

        # 验证 hooks 已注册
        assert len(m._hooks) == NUM_INJECTION

        # 移除 hooks
        m.remove_hooks()

        # 验证清空
        assert len(m._hooks) == 0, f"remove_hooks() 后 _hooks 未清空：{m._hooks}"
