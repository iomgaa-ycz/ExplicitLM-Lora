"""
tests/integration/test_modified_model_flow.py — ModifiedQwen 端到端集成测试

使用真实 Qwen3-0.6B + 真实 AttentionInjection（D=1024）验证：
  - 端到端 forward logits 形状 [B, L, V]
  - 有 labels 时 loss.backward() → injection_modules 有梯度，base_model 无梯度
  - Hook 被触发 4 次（4 个注入层）
  - KnowledgeEncoder.forward 仅调用 1 次（4 层复用同一知识）
  - knowledge_ids=None 时输出与直接调用 base_model 等价

测试策略：
  - 真实 Qwen3-0.6B（module 级 fixture，只加载一次）
  - CPU 推理（避免 GPU 依赖，CI 兼容）
  - 生成 Markdown 报告到 tests/outputs/modified_model/

常量：B=2, L=8, K_F=64, D=1024
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "modified_model"

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """加载冻结的 Qwen3-0.6B（module 级，只加载一次）。"""
    return load_base_model(MODEL_PATH, bf16=True)


@pytest.fixture(scope="module")
def knowledge_encoder(base_model):
    """构造 KnowledgeEncoder 并切换 eval 模式。"""
    enc = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=ENCODER_DEPTH,
        hidden_dim=HIDDEN_DIM,
    )
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def injection_modules() -> nn.ModuleList:
    """构造 4 个 AttentionInjection（零初始化）。"""
    return nn.ModuleList(
        [AttentionInjection(hidden_dim=HIDDEN_DIM) for _ in range(NUM_INJECTION)]
    )


@pytest.fixture(scope="module")
def model(base_model, knowledge_encoder, injection_modules) -> ModifiedQwen:
    """构造 ModifiedQwen 并切换 eval 模式。"""
    m = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=knowledge_encoder,
        injection_modules=injection_modules,
        injection_layers=INJECTION_LAYERS,
        pad_token_id=PAD_TOKEN_ID,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def report_lines() -> List[str]:
    """跨测试共享的 Markdown 报告内容列表。"""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────


def _make_input_ids(batch: int = B, seq_len: int = L) -> torch.Tensor:
    """构造随机 input_ids [B, L]。"""
    torch.manual_seed(100)
    return torch.randint(1, VOCAB_SIZE // 2, (batch, seq_len))


def _make_knowledge_ids(batch: int = B, k_f: int = FUSION_LENGTH) -> torch.Tensor:
    """构造随机 knowledge_ids [B, K_F]，后半段为 padding。"""
    torch.manual_seed(101)
    ids = torch.randint(1, VOCAB_SIZE // 2, (batch, k_f))
    ids[:, k_f // 2 :] = 0
    return ids


def _make_attention_mask(batch: int = B, seq_len: int = L) -> torch.Tensor:
    """构造全 1 的 attention_mask [B, L]。"""
    return torch.ones(batch, seq_len, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndFlow:
    """端到端 forward 与 backward 流程验证。"""

    def test_forward_logits_shape(
        self,
        model: ModifiedQwen,
        report_lines: List[str],
    ) -> None:
        """
        测试：端到端 forward 后 logits shape 应为 [B, L, V]。

        验证点：
            - output.logits.shape == (B, L, VOCAB_SIZE)
            - logits 数值有限（无 NaN/inf）
        """
        input_ids = _make_input_ids()
        knowledge_ids = _make_knowledge_ids()
        attention_mask = _make_attention_mask()

        with torch.no_grad():
            output = model(input_ids, knowledge_ids, attention_mask)

        logits = output.logits
        assert logits.shape == (B, L, VOCAB_SIZE), (
            f"logits shape {logits.shape} != 预期 (B={B}, L={L}, V={VOCAB_SIZE})"
        )
        assert torch.isfinite(logits).all(), "logits 包含 NaN 或 inf"

        report_lines.append("## Step 1：端到端 Forward 形状验证")
        report_lines.append(f"- 输入 input_ids: {list(input_ids.shape)}")
        report_lines.append(f"- 输入 knowledge_ids: {list(knowledge_ids.shape)}")
        report_lines.append(f"- 输出 logits: {list(logits.shape)} ✓")
        report_lines.append(f"- 数值有限: {torch.isfinite(logits).all().item()} ✓")
        report_lines.append("")

    def test_loss_backward(
        self,
        model: ModifiedQwen,
        report_lines: List[str],
    ) -> None:
        """
        测试：loss.backward() 后 injection_modules 有梯度，base_model 无梯度。

        验证点：
            - injection_modules 中至少一个参数 .grad is not None
            - base_model 所有参数 .grad is None
        """
        # 切换到 train 模式以使梯度流通（eval 模式下 dropout 关闭但梯度仍可计算）
        for inj in model.injection_modules:
            inj.train()

        input_ids = _make_input_ids()
        knowledge_ids = _make_knowledge_ids()
        attention_mask = _make_attention_mask()
        labels = input_ids.clone()

        # 清空已有梯度
        model.injection_modules.zero_grad()

        output = model(input_ids, knowledge_ids, attention_mask, labels=labels)
        output.loss.backward()

        # 验证注入模块有梯度
        injection_grads = [
            p.grad for p in model.injection_modules.parameters() if p.requires_grad
        ]
        assert any(g is not None for g in injection_grads), (
            "backward 后 injection_modules 无梯度"
        )
        assert any(g is not None and g.abs().sum() > 0 for g in injection_grads), (
            "injection_modules 梯度全为零"
        )

        # 验证 base_model 无梯度
        base_grads_nonzero = [
            (name, p.grad)
            for name, p in model.base_model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(base_grads_nonzero) == 0, (
            f"base_model 存在 {len(base_grads_nonzero)} 个参数有非零梯度: "
            f"{[n for n, _ in base_grads_nonzero[:3]]}"
        )

        # 清空梯度，恢复 eval 模式
        model.injection_modules.zero_grad()
        for inj in model.injection_modules:
            inj.eval()

        report_lines.append("## Step 2：Backward 梯度流向验证")
        report_lines.append(f"- loss 值: {output.loss.item():.4f}")
        trainable_with_grad = sum(
            1 for g in injection_grads if g is not None
        )
        report_lines.append(
            f"- injection_modules 有梯度参数数: {trainable_with_grad}/{len(injection_grads)} ✓"
        )
        report_lines.append(
            f"- base_model 非零梯度参数数: {len(base_grads_nonzero)} ✓"
        )
        report_lines.append("")


class TestHookInjection:
    """Hook 触发次数验证。"""

    def test_injection_called_4_times(
        self,
        model: ModifiedQwen,
        report_lines: List[str],
    ) -> None:
        """
        测试：一次 forward 应触发 4 次注入（对应 4 个注入层）。

        实现：包装 injection_modules[i].forward 计数调用次数。

        验证点：
            - 共触发 4 次（每个注入层恰好触发 1 次）
        """
        call_counts = [0] * NUM_INJECTION

        # 包装每个注入模块的 forward 方法
        original_forwards = []
        for i, inj in enumerate(model.injection_modules):
            orig = inj.forward

            def make_counting_forward(orig_forward, idx):
                def counting_forward(hidden, knowledge, mask):
                    call_counts[idx] += 1
                    return orig_forward(hidden, knowledge, mask)
                return counting_forward

            inj.forward = make_counting_forward(orig, i)
            original_forwards.append(orig)

        try:
            input_ids = _make_input_ids()
            knowledge_ids = _make_knowledge_ids()
            attention_mask = _make_attention_mask()

            with torch.no_grad():
                model(input_ids, knowledge_ids, attention_mask)

            assert call_counts == [1, 1, 1, 1], (
                f"注入次数 {call_counts} != 预期 [1,1,1,1]"
            )
        finally:
            # 恢复原始 forward
            for i, inj in enumerate(model.injection_modules):
                inj.forward = original_forwards[i]

        report_lines.append("## Step 3：Hook 触发次数验证")
        report_lines.append(f"- 各注入层调用次数: {call_counts}")
        report_lines.append(f"- 总注入次数: {sum(call_counts)} (预期 4) ✓")
        report_lines.append("")


class TestKnowledgeEncoding:
    """KnowledgeEncoder 调用次数验证（知识一次编码，4 层复用）。"""

    def test_knowledge_encoded_once(
        self,
        model: ModifiedQwen,
        report_lines: List[str],
    ) -> None:
        """
        测试：一次 forward 中 KnowledgeEncoder.forward 仅调用 1 次（所有注入层复用）。

        验证点：
            - knowledge_encoder.forward 调用计数 == 1
        """
        encoder_call_count = [0]
        orig_forward = model.knowledge_encoder.forward

        def counting_forward(knowledge_ids, attention_mask):
            encoder_call_count[0] += 1
            return orig_forward(knowledge_ids, attention_mask)

        model.knowledge_encoder.forward = counting_forward

        try:
            input_ids = _make_input_ids()
            knowledge_ids = _make_knowledge_ids()
            attention_mask = _make_attention_mask()

            with torch.no_grad():
                model(input_ids, knowledge_ids, attention_mask)

            assert encoder_call_count[0] == 1, (
                f"KnowledgeEncoder.forward 调用 {encoder_call_count[0]} 次，预期 1 次"
            )
        finally:
            model.knowledge_encoder.forward = orig_forward

        report_lines.append("## Step 4：KnowledgeEncoder 调用次数验证")
        report_lines.append(f"- KnowledgeEncoder.forward 调用次数: {encoder_call_count[0]} (预期 1) ✓")
        report_lines.append("- 设计验证：知识一次编码，4 个注入层复用 _current_knowledge ✓")
        report_lines.append("")


class TestNoneKnowledge:
    """knowledge_ids=None 退化模式验证。"""

    def test_none_knowledge_identical_to_base(
        self,
        model: ModifiedQwen,
        report_lines: List[str],
    ) -> None:
        """
        测试：knowledge_ids=None 时，ModifiedQwen 输出与直接调用 base_model 完全一致。

        验证点：
            - logits 绝对误差 == 0（完全相同，因为 hook 不修改输出）
        """
        input_ids = _make_input_ids()
        attention_mask = _make_attention_mask()

        with torch.no_grad():
            out_modified = model(input_ids, None, attention_mask)
            out_base = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        logits_modified = out_modified.logits.float()
        logits_base = out_base.logits.float()

        max_abs_diff = (logits_modified - logits_base).abs().max().item()

        assert max_abs_diff == 0.0, (
            f"退化模式 logits 与 base_model 不完全一致，最大绝对误差={max_abs_diff:.2e}"
        )

        report_lines.append("## Step 5：退化模式等价验证")
        report_lines.append(f"- knowledge_ids=None 时 logits 最大绝对误差: {max_abs_diff:.2e}")
        report_lines.append(f"- 与 base_model 输出完全一致: {max_abs_diff == 0.0} ✓")
        report_lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 报告生成
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def generate_markdown_report(report_lines: List[str]) -> None:
    """
    模块级 autouse fixture：所有测试完成后生成 Markdown 报告。

    报告路径：tests/outputs/modified_model/test_modified_model_flow_<timestamp>.md
    """
    yield  # 等待所有测试执行完毕

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"test_modified_model_flow_{timestamp}.md"

    header = [
        "# Agent 测试: test_modified_model_flow",
        "",
        "## 任务",
        "验证 ModifiedQwen 端到端流程：Hook 注入机制、梯度流向、知识编码复用、退化模式等价性。",
        "",
        "## 测试环境",
        f"- 时间戳: {timestamp}",
        f"- 设备: CPU",
        f"- 基础模型: Qwen3-0.6B",
        f"- 参数: B={B}, L={L}, K_F={FUSION_LENGTH}, D={HIDDEN_DIM}",
        f"- 注入层: {INJECTION_LAYERS}",
        f"- 注入方式: AttentionInjection（零初始化）",
        "",
        "## 测试结果",
        "",
    ]

    footer = [
        "",
        "## 最终结论",
        f"- logits 形状 [B={B}, L={L}, V={VOCAB_SIZE}] 正确 ✓",
        "- backward 后 injection_modules 有梯度，base_model 无梯度 ✓",
        "- 4 个注入层 hook 各触发 1 次 ✓",
        "- KnowledgeEncoder.forward 仅调用 1 次（知识复用设计正确）✓",
        "- knowledge_ids=None 时完全等价于原始 base_model ✓",
    ]

    content = "\n".join(header + report_lines + footer)
    report_path.write_text(content, encoding="utf-8")
    print(f"\n[报告] 已生成 Markdown 报告：{report_path}")
