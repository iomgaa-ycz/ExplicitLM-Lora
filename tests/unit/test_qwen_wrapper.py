"""
tests/unit/test_qwen_wrapper.py — KnowledgeEncoder 集成测试

使用真实 Qwen3-0.6B 模型（本地路径 Qwen3-0.6B/）验证接口正确性。
通过 module 级别 fixture 复用同一 base_model 实例，避免重复加载。

测试覆盖：
  - 初始化后参数冻结状态
  - forward 输出形状
  - forward padding 不变性（双向注意力正确性）
  - encode_mean 输出形状
  - encode_mean padding mask 有效性
  - unfreeze_layers 接口
  - device 属性
"""

from __future__ import annotations

import pytest
import torch

from models.qwen_wrapper import KnowledgeEncoder, load_base_model

# ── 常量 ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "Qwen3-0.6B"
ENCODER_DEPTH = 6
HIDDEN_DIM = 1024
FUSION_LENGTH = 64   # K_f
ANCHOR_LENGTH = 128  # K_a
BATCH_SIZE = 2
VOCAB_SIZE = 151936  # Qwen3-0.6B 词表大小（tokenizer）


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_model():
    """
    模块级 fixture：加载并返回冻结的 Qwen3-0.6B 基础模型。
    整个 test module 只加载一次，避免重复加载开销（加载约需 5 秒）。
    """
    model = load_base_model(MODEL_PATH, bf16=True)
    return model


@pytest.fixture(scope="module")
def encoder(base_model):
    """
    模块级 fixture：构造 KnowledgeEncoder 并返回。
    依赖 base_model fixture，模块内共享同一实例。
    """
    enc = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=ENCODER_DEPTH,
        hidden_dim=HIDDEN_DIM,
    )
    enc.eval()
    return enc


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_inputs(seq_len: int, batch_size: int = BATCH_SIZE, pad_last: bool = False):
    """
    构造合法的 (knowledge_ids, attention_mask) 张量对。

    参数：
        seq_len: 序列长度
        batch_size: batch 大小
        pad_last: 是否在后半段添加 padding（用于测试 mask 有效性）

    返回：
        (knowledge_ids [B, seq_len], attention_mask [B, seq_len]) 的 LongTensor 对
    """
    # token IDs 范围 [1, VOCAB_SIZE)，避免使用 pad token ID 0
    knowledge_ids = torch.randint(1, VOCAB_SIZE // 2, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    if pad_last:
        # 后半段设为 padding（token ID = 0，mask = 0）
        half = seq_len // 2
        knowledge_ids[:, half:] = 0
        attention_mask[:, half:] = 0

    return knowledge_ids, attention_mask


# ── 测试用例 ──────────────────────────────────────────────────────────────────

class TestKnowledgeEncoderInit:
    """初始化后参数冻结状态验证"""

    def test_embed_tokens_frozen(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：embed_tokens 的所有参数应始终冻结（requires_grad=False）。

        验证点：
            - embed_tokens.weight.requires_grad == False
        """
        for p in encoder.embed_tokens.parameters():
            assert p.requires_grad is False, (
                f"embed_tokens 参数应冻结，但 requires_grad={p.requires_grad}"
            )

    def test_layers_frozen(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：初始化后 layers 的所有参数应冻结。

        验证点：
            - 前 encoder_depth 层的所有参数 requires_grad == False
        """
        for i, layer in enumerate(encoder.layers):
            for name, p in layer.named_parameters():
                assert p.requires_grad is False, (
                    f"layer[{i}].{name} 应冻结，但 requires_grad={p.requires_grad}"
                )

    def test_norm_trainable(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：独立 norm 应可训练（requires_grad=True）。

        验证点：
            - self.norm 至少有一个参数 requires_grad=True
        """
        norm_params = list(encoder.norm.parameters())
        assert len(norm_params) > 0, "norm 应有至少一个参数"
        assert all(p.requires_grad for p in norm_params), (
            "norm 的所有参数应可训练（requires_grad=True）"
        )

    def test_encoder_depth_stored(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：encoder_depth 属性正确存储。

        验证点：
            - encoder.encoder_depth == ENCODER_DEPTH
            - len(encoder.layers) == ENCODER_DEPTH
        """
        assert encoder.encoder_depth == ENCODER_DEPTH
        assert len(encoder.layers) == ENCODER_DEPTH


class TestForward:
    """forward 方法输出形状与语义正确性验证"""

    def test_output_shape(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：forward 输出形状应为 [B, K_f, D]。

        验证点：
            - 输出 shape == (BATCH_SIZE, FUSION_LENGTH, HIDDEN_DIM)
            - 输出 dtype 为浮点类型
        """
        ids, mask = _make_inputs(FUSION_LENGTH)

        with torch.no_grad():
            out = encoder(ids, mask)

        assert out.shape == (BATCH_SIZE, FUSION_LENGTH, HIDDEN_DIM), (
            f"forward 输出形状不符: 期望 {(BATCH_SIZE, FUSION_LENGTH, HIDDEN_DIM)}, "
            f"实际 {tuple(out.shape)}"
        )
        assert out.dtype in (torch.float32, torch.bfloat16, torch.float16), (
            f"输出应为浮点类型，实际 {out.dtype}"
        )

    def test_padding_invariance(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：padding token 不影响有效 token 的输出（双向注意力 mask 正确性）。

        设计：
            - 两个 batch item 的前半段 token IDs 完全相同
            - 后半段分别填充不同的随机 token，但 mask=0（均为 padding）
            - 有效位置（前半段）的输出应完全相同（误差在数值精度范围内）

        验证点：
            - 有效 token 的输出在两个 sample 间完全相同（allclose）
        """
        half = FUSION_LENGTH // 2

        # 构造相同的有效 token 前半段
        shared_valid = torch.randint(1, VOCAB_SIZE // 2, (1, half))

        # 两个 sample：前半段相同，后半段为不同 padding
        ids_a = torch.cat([shared_valid, torch.randint(1, 100, (1, half))], dim=1)
        ids_b = torch.cat([shared_valid, torch.randint(100, 200, (1, half))], dim=1)

        # 后半段均为 padding（mask=0）
        mask = torch.ones(1, FUSION_LENGTH, dtype=torch.long)
        mask[:, half:] = 0

        with torch.no_grad():
            out_a = encoder(ids_a, mask)  # [1, K_f, D]
            out_b = encoder(ids_b, mask)  # [1, K_f, D]

        # 有效位置（前半段）输出应完全一致
        assert torch.allclose(out_a[:, :half, :], out_b[:, :half, :], atol=1e-3), (
            "相同有效 token、不同 padding 的输出在有效位置应相同，"
            f"但最大差异为 {(out_a[:, :half, :] - out_b[:, :half, :]).abs().max().item()}"
        )

    def test_anchor_length_input(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：forward 支持不同序列长度（如 K_a = 128）。

        验证点：
            - 输出形状为 [B, K_a, D]
        """
        ids, mask = _make_inputs(ANCHOR_LENGTH)

        with torch.no_grad():
            out = encoder(ids, mask)

        assert out.shape == (BATCH_SIZE, ANCHOR_LENGTH, HIDDEN_DIM), (
            f"forward 对 anchor_length 输入形状不符: 期望 "
            f"{(BATCH_SIZE, ANCHOR_LENGTH, HIDDEN_DIM)}, 实际 {tuple(out.shape)}"
        )


class TestEncodeMean:
    """encode_mean 方法输出形状与 padding mask 有效性验证"""

    def test_output_shape(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：encode_mean 输出形状应为 [B, D]。

        验证点：
            - 输出 shape == (BATCH_SIZE, HIDDEN_DIM)
        """
        ids, mask = _make_inputs(FUSION_LENGTH)

        with torch.no_grad():
            out = encoder.encode_mean(ids, mask)

        assert out.shape == (BATCH_SIZE, HIDDEN_DIM), (
            f"encode_mean 输出形状不符: 期望 {(BATCH_SIZE, HIDDEN_DIM)}, "
            f"实际 {tuple(out.shape)}"
        )

    def test_padding_affects_mean(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：padding mask 正确影响 mean pooling（有效 token 数不同，均值不同）。

        设计：
            - 两次调用使用相同 token IDs
            - 第一次：全部 token 有效（mask 全为 1）
            - 第二次：后半段为 padding（mask 后半段为 0）
            - 均值结果应不同（因参与均值的 token 不同）

        验证点：
            - 两次均值输出不完全相同（至少有一个元素差异 > 1e-4）
        """
        ids = torch.randint(1, VOCAB_SIZE // 2, (1, FUSION_LENGTH))

        mask_full = torch.ones(1, FUSION_LENGTH, dtype=torch.long)
        mask_half = torch.ones(1, FUSION_LENGTH, dtype=torch.long)
        mask_half[:, FUSION_LENGTH // 2:] = 0

        with torch.no_grad():
            out_full = encoder.encode_mean(ids, mask_full)  # [1, D]
            out_half = encoder.encode_mean(ids, mask_half)  # [1, D]

        max_diff = (out_full - out_half).abs().max().item()
        assert max_diff > 1e-4, (
            f"full mask 与 half mask 的 encode_mean 输出应不同，"
            f"但最大差异仅为 {max_diff:.2e}"
        )

    def test_anchor_length_input(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：encode_mean 支持 anchor_length（K_a=128）输入。

        验证点：
            - 输出形状为 [B, D]
        """
        ids, mask = _make_inputs(ANCHOR_LENGTH)

        with torch.no_grad():
            out = encoder.encode_mean(ids, mask)

        assert out.shape == (BATCH_SIZE, HIDDEN_DIM), (
            f"encode_mean 对 anchor_length 输入形状不符: 期望 "
            f"{(BATCH_SIZE, HIDDEN_DIM)}, 实际 {tuple(out.shape)}"
        )


class TestUnfreezeAndDevice:
    """unfreeze_layers 与 device 属性验证"""

    def test_unfreeze_layers(self, base_model) -> None:
        """
        测试：unfreeze_layers() 解冻 layers，embed_tokens 保持冻结。

        使用独立 encoder 实例（避免污染其他测试的共享 encoder）。

        验证点：
            - 调用后 layers 的所有参数 requires_grad=True
            - embed_tokens 仍然 requires_grad=False
            - norm 仍然 requires_grad=True
        """
        # 使用独立实例，不影响 module 级共享 encoder
        enc = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=ENCODER_DEPTH,
            hidden_dim=HIDDEN_DIM,
        )

        # 调用前：layers 应冻结
        for p in enc.layers.parameters():
            assert p.requires_grad is False, "调用前 layers 应冻结"

        enc.unfreeze_layers()

        # 调用后：layers 应可训练
        for i, layer in enumerate(enc.layers):
            for name, p in layer.named_parameters():
                assert p.requires_grad is True, (
                    f"unfreeze_layers 后 layer[{i}].{name} 应可训练"
                )

        # embed_tokens 仍应冻结
        for p in enc.embed_tokens.parameters():
            assert p.requires_grad is False, (
                "unfreeze_layers 后 embed_tokens 仍应冻结"
            )

        # norm 仍应可训练
        for p in enc.norm.parameters():
            assert p.requires_grad is True, (
                "unfreeze_layers 后 norm 仍应可训练"
            )

    def test_device_property(self, encoder: KnowledgeEncoder) -> None:
        """
        测试：device 属性返回正确的设备对象。

        验证点：
            - 返回类型为 torch.device
            - 设备为 cpu（测试环境不使用 GPU）
        """
        device = encoder.device

        assert isinstance(device, torch.device), (
            f"device 属性应返回 torch.device，实际返回 {type(device)}"
        )
        # 测试环境在 CPU 运行
        assert device.type == "cpu", (
            f"测试环境设备应为 cpu，实际为 {device}"
        )
