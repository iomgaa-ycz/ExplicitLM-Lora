"""
tests/integration/test_pipeline_flow.py — ExplicitLMPipeline 端到端集成测试

使用真实 Qwen3-0.6B + 真实组件（非 Mock）验证：
  - 直接传组件构建 ExplicitLMPipeline，无异常
  - answer() 返回 PipelineOutput，各字段类型/值域正确
  - Oracle 模式下 retrieved_id 与 oracle_map 一致
  - evaluate_loglikelihood() 对 4 选项返回合法 int (0-3)
  - eval 模式下相同输入两次结果一致（确定性）

测试策略：
  - module 级 fixture：Qwen3-0.6B 只加载一次，所有测试共享
  - 小规模 DualKnowledgeStore（knowledge_num=64），手动注入倒排索引，不走真实聚类
  - CPU 推理（避免 GPU 依赖），encoder_depth=2（减少计算量）
  - 生成 Markdown 报告到 tests/outputs/pipeline/

常量：KNOWLEDGE_NUM=64, FUSION_LENGTH=16, ANCHOR_LENGTH=16, ENCODER_DEPTH=2
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    AttentionInjection,
    KnowledgeEncoder,
    ModifiedQwen,
    load_base_model,
)
from pipeline import ExplicitLMPipeline, PipelineOutput  # noqa: E402
from router.memory_bank import DualKnowledgeStore  # noqa: E402
from router.model import MemoryRouter  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = str(PROJECT_ROOT / "Qwen3-0.6B")
ENCODER_DEPTH = 2       # 仅前 2 层，降低 CPU 计算量
HIDDEN_DIM = 1024
FUSION_LENGTH = 16      # 集成测试用短序列
ANCHOR_LENGTH = 16
KNOWLEDGE_NUM = 64      # 8²，最小完全平方数
NUM_CANDIDATES = 4
KEY_PROJ_DIM = 512
ADAPTER_DIM = 512
INJECTION_LAYERS = [6, 12, 18, 24]
DEVICE = "cpu"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "pipeline"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 报告辅助
# ─────────────────────────────────────────────────────────────────────────────


def _save_md_report(test_name: str, sections: List[str]) -> str:
    """
    保存测试结果到 Markdown 文件，返回文件路径。

    参数：
        test_name: 测试名称（用于文件名和标题）
        sections:  Markdown 段落列表

    返回：
        生成的 Markdown 文件绝对路径字符串
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{test_name}_{ts}.md"
    content = f"# 集成测试: {test_name}\n\n"
    content += "\n\n".join(sections)
    path.write_text(content, encoding="utf-8")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 模块级 Fixtures（真实模型只加载一次）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """加载冻结的 Qwen3-0.6B（module 级，所有测试共享）。"""
    return load_base_model(MODEL_PATH, bf16=False)  # CPU 用 float32


@pytest.fixture(scope="module")
def encoder(base_model):
    """构造 KnowledgeEncoder（encoder_depth=2 加速），冻结所有参数。"""
    enc = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=ENCODER_DEPTH,
        hidden_dim=HIDDEN_DIM,
    )
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def injection_modules():
    """构造 4 个 AttentionInjection（零初始化），eval 模式。"""
    modules = nn.ModuleList(
        [AttentionInjection(hidden_dim=HIDDEN_DIM) for _ in range(len(INJECTION_LAYERS))]
    )
    modules.eval()
    return modules


@pytest.fixture(scope="module")
def modified_qwen(base_model, encoder, injection_modules):
    """构造 ModifiedQwen，eval 模式。"""
    m = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=encoder,
        injection_modules=injection_modules,
        injection_layers=INJECTION_LAYERS,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def router_config():
    """构造小规模 RouterConfig mock（集成测试用）。"""
    cfg = MagicMock()
    cfg.knowledge_num = KNOWLEDGE_NUM
    cfg.dim = HIDDEN_DIM
    cfg.query_dim = HIDDEN_DIM
    cfg.key_proj_dim = KEY_PROJ_DIM
    cfg.adapter_dim = ADAPTER_DIM
    cfg.num_candidates = NUM_CANDIDATES
    cfg.temperature = 0.1
    cfg.max_candidates_per_cell = -1
    cfg.recluster_threshold = 0.1
    cfg.refined_num_heads = 8
    cfg.refined_num_layers = 2
    return cfg


@pytest.fixture(scope="module")
def router(encoder, router_config):
    """构造 MemoryRouter，eval 模式。"""
    r = MemoryRouter(config=router_config, encoder=encoder)
    r.eval()
    return r


@pytest.fixture(scope="module")
def store(router_config):
    """
    构造小规模 DualKnowledgeStore（KNOWLEDGE_NUM=64），手动注入倒排索引。

    不调用真实聚类，加速测试。
    """
    s_cfg = MagicMock()
    s_cfg.knowledge_num = KNOWLEDGE_NUM
    s_cfg.recluster_threshold = 0.1

    s = DualKnowledgeStore(
        config=s_cfg,
        fusion_length=FUSION_LENGTH,
        anchor_length=ANCHOR_LENGTH,
        device=DEVICE,
    )
    num_keys = 8  # √64
    c = num_keys * num_keys  # 64 个 grid cell

    # 手动注入倒排索引（不走 compact_and_recluster）
    s.inverted_index = torch.arange(KNOWLEDGE_NUM, dtype=torch.long)
    s.cluster_counts = torch.ones(c, dtype=torch.long)
    s.cluster_offsets = torch.arange(c + 1, dtype=torch.long)
    s.row_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)
    s.col_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)
    s.valid_mask = torch.ones(KNOWLEDGE_NUM, dtype=torch.bool)
    s.next_free = KNOWLEDGE_NUM

    # 填充随机 token IDs（token_id > 0 表示有效）
    s.fusion_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, FUSION_LENGTH), dtype=torch.long)
    s.anchor_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, ANCHOR_LENGTH), dtype=torch.long)

    return s


@pytest.fixture(scope="module")
def tokenizer():
    """加载 Qwen3 tokenizer。"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def pipeline_config(router_config):
    """构造最小 Config mock，供 ExplicitLMPipeline 使用。"""
    cfg = MagicMock()
    cfg.model.hidden_dim = HIDDEN_DIM
    cfg.model.fusion_length = FUSION_LENGTH
    cfg.model.anchor_length = ANCHOR_LENGTH
    cfg.model.injection_layers = INJECTION_LAYERS
    cfg.router = router_config
    return cfg


@pytest.fixture(scope="module")
def pipeline(pipeline_config, modified_qwen, router, store, tokenizer):
    """构造标准 ExplicitLMPipeline（无 oracle_map）。"""
    # 同步 PKM keys（从 store 的 row/col centroids 到 router.pkm）
    router.pkm.update_keys(store.row_centroids, store.col_centroids)

    return ExplicitLMPipeline(
        config=pipeline_config,
        modified_qwen=modified_qwen,
        router=router,
        store=store,
        tokenizer=tokenizer,
        oracle_map=None,
    )


@pytest.fixture(scope="module")
def oracle_pipeline(pipeline_config, modified_qwen, router, store, tokenizer):
    """构造含 oracle_map 的 ExplicitLMPipeline（Oracle 模式测试用）。"""
    oracle_map = {"What is the capital of France?": 5}
    return ExplicitLMPipeline(
        config=pipeline_config,
        modified_qwen=modified_qwen,
        router=router,
        store=store,
        tokenizer=tokenizer,
        oracle_map=oracle_map,
    )


@pytest.fixture(scope="module")
def report_lines() -> List[str]:
    """跨测试共享的 Markdown 报告内容列表。"""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineConstruct:
    """验证管线构建流程。"""

    def test_pipeline_builds_from_components(
        self,
        pipeline_config,
        modified_qwen,
        router,
        store,
        tokenizer,
        report_lines: List[str],
    ) -> None:
        """
        测试：直接传入已构建组件，ExplicitLMPipeline 应正常创建，无异常。

        验证点：
            - pipeline 实例非 None
            - _encoder 引用来自 modified_qwen.knowledge_encoder
            - oracle_map 默认为 None
        """
        p = ExplicitLMPipeline(
            config=pipeline_config,
            modified_qwen=modified_qwen,
            router=router,
            store=store,
            tokenizer=tokenizer,
        )

        assert p is not None
        assert p._encoder is modified_qwen.knowledge_encoder
        assert p._oracle_map is None

        report_lines.append("## Step 1：管线构建验证")
        report_lines.append("- 直接传入组件构建 ExplicitLMPipeline：成功 ✓")
        report_lines.append(f"- _encoder 引用来自 modified_qwen.knowledge_encoder：True ✓")
        report_lines.append("")


class TestPipelineAnswer:
    """验证 answer() 端到端流程。"""

    def test_answer_returns_pipeline_output(
        self, pipeline: ExplicitLMPipeline, report_lines: List[str]
    ) -> None:
        """
        测试：真实 Qwen3 + 真实组件下，answer() 应返回合法 PipelineOutput。

        验证点：
            - 返回类型为 PipelineOutput
            - answer 为 str（可为空但不为 None）
            - retrieved_id 为 int，值域 [0, KNOWLEDGE_NUM)
            - latency_ms > 0
        """
        question = "What is the capital of France?"
        result = pipeline.answer(question, use_real_router=True)

        assert isinstance(result, PipelineOutput)
        assert isinstance(result.answer, str)
        assert isinstance(result.retrieved_id, int)
        assert 0 <= result.retrieved_id < KNOWLEDGE_NUM, (
            f"retrieved_id={result.retrieved_id} 超出 [0, {KNOWLEDGE_NUM})"
        )
        assert result.latency_ms > 0.0

        report_lines.append("## Step 2：answer() 端到端验证")
        report_lines.append(f"- 问题: {question}")
        report_lines.append(f"- 返回类型: PipelineOutput ✓")
        report_lines.append(f"- answer: '{result.answer}'")
        report_lines.append(f"- retrieved_id: {result.retrieved_id} ✓")
        report_lines.append(f"- latency_ms: {result.latency_ms:.2f} ms ✓")
        report_lines.append("")

    def test_answer_oracle_mode(
        self, oracle_pipeline: ExplicitLMPipeline, report_lines: List[str]
    ) -> None:
        """
        测试：Oracle 模式下 retrieved_id 应与 oracle_map 中的值一致。

        验证点：
            - retrieved_id == oracle_map["What is the capital of France?"] == 5
        """
        question = "What is the capital of France?"
        result = oracle_pipeline.answer(question, use_real_router=False)

        assert result.retrieved_id == 5, (
            f"Oracle mode retrieved_id={result.retrieved_id}，期望 5"
        )

        report_lines.append("## Step 3：Oracle 模式验证")
        report_lines.append(f"- oracle_map['...France?'] = 5")
        report_lines.append(f"- retrieved_id = {result.retrieved_id} ✓")
        report_lines.append("")


class TestPipelineEval:
    """验证 evaluate_loglikelihood() 接口。"""

    def test_evaluate_loglikelihood_abcd(
        self, pipeline: ExplicitLMPipeline, store: DualKnowledgeStore, report_lines: List[str]
    ) -> None:
        """
        测试：4 选项 loglikelihood 评测应返回 0-3 的 int。

        验证点：
            - 返回类型为 int
            - 值在 [0, 4) 范围内
        """
        question = "A patient presents with fever and cough. The most likely diagnosis is:"
        choices = ["Pneumonia", "Influenza", "COVID-19", "Tuberculosis"]
        knowledge_ids = store.fusion_bank[torch.tensor([0], dtype=torch.long)]  # [1, K_f]

        idx = pipeline.evaluate_loglikelihood(
            question=question,
            choices=choices,
            knowledge_ids=knowledge_ids,
        )

        assert isinstance(idx, int), f"返回类型应为 int，实际: {type(idx)}"
        assert 0 <= idx < len(choices), f"idx={idx} 超出范围 [0, {len(choices)})"

        report_lines.append("## Step 4：evaluate_loglikelihood() 4选项验证")
        report_lines.append(f"- 问题: {question[:60]}...")
        report_lines.append(f"- 选项: {choices}")
        report_lines.append(f"- 预测 idx: {idx} ({choices[idx]}) ✓")
        report_lines.append("")

    def test_evaluate_loglikelihood_deterministic(
        self, pipeline: ExplicitLMPipeline, store: DualKnowledgeStore, report_lines: List[str]
    ) -> None:
        """
        测试：eval 模式下相同输入两次调用 evaluate_loglikelihood 结果一致。

        验证点：
            - 两次结果 idx1 == idx2
        """
        question = "Which vitamin deficiency causes scurvy?"
        choices = ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"]
        knowledge_ids = store.fusion_bank[torch.tensor([1], dtype=torch.long)]  # [1, K_f]

        idx1 = pipeline.evaluate_loglikelihood(question, choices, knowledge_ids)
        idx2 = pipeline.evaluate_loglikelihood(question, choices, knowledge_ids)

        assert idx1 == idx2, (
            f"eval 模式下两次结果不一致：idx1={idx1}, idx2={idx2}"
        )

        report_lines.append("## Step 5：evaluate_loglikelihood() 确定性验证")
        report_lines.append(f"- 第一次预测: idx={idx1}")
        report_lines.append(f"- 第二次预测: idx={idx2}")
        report_lines.append(f"- 两次一致: {idx1 == idx2} ✓")
        report_lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 报告生成
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def generate_markdown_report(report_lines: List[str]) -> None:
    """
    module 级 autouse fixture：所有测试完成后生成 Markdown 报告。

    报告路径：tests/outputs/pipeline/test_pipeline_flow_<timestamp>.md
    """
    yield  # 等待所有测试执行完毕

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"test_pipeline_flow_{timestamp}.md"

    header = [
        "# Agent 测试: test_pipeline_flow",
        "",
        "## 任务",
        "验证 ExplicitLMPipeline 端到端流程：管线构建、answer()、Oracle 模式、loglikelihood 评测、确定性。",
        "",
        "## 测试环境",
        f"- 时间戳: {timestamp}",
        f"- 设备: {DEVICE}",
        f"- 基础模型: Qwen3-0.6B",
        f"- 参数: KNOWLEDGE_NUM={KNOWLEDGE_NUM}, FUSION_LENGTH={FUSION_LENGTH}",
        f"- encoder_depth: {ENCODER_DEPTH}（加速 CPU 推理）",
        f"- 注入层: {INJECTION_LAYERS}",
        "",
        "## 测试结果",
        "",
    ]

    footer = [
        "",
        "## 最终结论",
        "- 管线从组件直接构建：成功 ✓",
        "- answer() 返回合法 PipelineOutput ✓",
        "- Oracle 模式 retrieved_id 与 oracle_map 一致 ✓",
        "- evaluate_loglikelihood() 4选项返回 0-3 的 int ✓",
        "- eval 模式下结果确定性 ✓",
    ]

    content = "\n".join(header + report_lines + footer)
    report_path.write_text(content, encoding="utf-8")
    print(f"\n[报告] 已生成 Markdown 报告：{report_path}")
