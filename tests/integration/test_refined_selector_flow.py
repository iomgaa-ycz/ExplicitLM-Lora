"""
tests/integration/test_refined_selector_flow.py — 精排系统端到端集成测试

使用真实 Qwen3-0.6B 模型（本地路径 Qwen3-0.6B/），验证 FeatureAdapter + RefinedSelector
完整精排流程的形状、值域和数值稳定性。

测试场景：
    1. 标准流程：B=4，C=16，KnowledgeEncoder 编码候选
    2. 单批 B=1 边界情况
    3. 大候选数 C=32
    4. 带 padding mask 推理

测试结果保存为 Markdown 报告至 tests/outputs/refined_selector/。
"""

from __future__ import annotations

import sys
import datetime
from pathlib import Path
from typing import List

import pytest
import torch

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen_wrapper import KnowledgeEncoder, load_base_model  # noqa: E402
from router.feature_adapter import FeatureAdapter  # noqa: E402
from router.refined_selector import RefinedSelector  # noqa: E402


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

MODEL_PATH = str(PROJECT_ROOT / "Qwen3-0.6B")
ENCODER_DEPTH = 6
HIDDEN_DIM = 1024
ADAPTER_DIM = 512
ANCHOR_LENGTH = 16   # 集成测试用较短序列，降低显存压力
NUM_HEADS = 8
NUM_LAYERS = 2
DEVICE = "cpu"       # 集成测试在 CPU 上运行，无需 GPU

OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "refined_selector"


# ─────────────────────────────────────────────
# 辅助：Markdown 报告
# ─────────────────────────────────────────────


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


# ─────────────────────────────────────────────
# Module 级 Fixture（共享真实模型，避免重复加载）
# ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """
    模块级 fixture：加载并返回冻结的 Qwen3-0.6B 基础模型。
    整个 test module 只加载一次，避免重复加载开销。
    """
    model = load_base_model(MODEL_PATH, bf16=False)  # CPU 上用 float32
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


@pytest.fixture(scope="module")
def feature_adapter():
    """
    模块级 fixture：构造并返回 FeatureAdapter。
    """
    adapter = FeatureAdapter(in_dim=HIDDEN_DIM, adapter_dim=ADAPTER_DIM)
    adapter.eval()
    return adapter


@pytest.fixture(scope="module")
def refined_selector():
    """
    模块级 fixture：构造并返回 RefinedSelector。
    """
    sel = RefinedSelector(
        adapter_dim=ADAPTER_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )
    sel.eval()
    return sel


# ─────────────────────────────────────────────
# 辅助：生成随机 token IDs
# ─────────────────────────────────────────────


def _make_token_ids(batch_size: int, seq_len: int, vocab_size: int = 151936) -> torch.Tensor:
    """生成随机 token IDs（模拟 AnchorBank 中的知识 token）。"""
    return torch.randint(1, vocab_size, (batch_size, seq_len))


# ─────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────


class TestRefinedSelectorStandardFlow:
    """标准精排流程：B=4，C=16。"""

    def test_standard_flow_shapes_and_values(
        self,
        encoder: KnowledgeEncoder,
        feature_adapter: FeatureAdapter,
        refined_selector: RefinedSelector,
    ) -> None:
        """
        验证完整精排流程（编码 → 适配 → 精排）的形状与值域正确性。

        Step 1: 用 KnowledgeEncoder 编码候选 token → [B, C, HIDDEN_DIM]
        Step 2: FeatureAdapter 处理查询 [B, HIDDEN_DIM] → [B, ADAPTER_DIM]
        Step 3: FeatureAdapter 处理候选 [B*C, ANCHOR_LEN, HIDDEN_DIM] → [B, C, ADAPTER_DIM]
        Step 4: RefinedSelector → (scores [B,C], best_idx [B])
        """
        B, C = 4, 16
        sections = [f"## 任务\n标准精排流程 (B={B}, C={C})"]

        # Phase 1: 生成随机查询 embedding（模拟来自 base model 的 hidden states）
        torch.manual_seed(0)
        query_emb = torch.randn(B, HIDDEN_DIM)  # [B, HIDDEN_DIM]

        # Phase 2: 生成候选 token IDs，通过 KnowledgeEncoder 编码
        # 将 [B, C, L] reshape 为 [B*C, L] 批量编码，再 reshape 回来
        cand_token_ids = _make_token_ids(B * C, ANCHOR_LENGTH)  # [B*C, L]
        cand_mask = torch.ones(B * C, ANCHOR_LENGTH, dtype=torch.bool)

        with torch.no_grad():
            cand_emb_flat = encoder.encode_mean(cand_token_ids, cand_mask)  # [B*C, HIDDEN_DIM]

        cand_emb = cand_emb_flat.view(B, C, HIDDEN_DIM)   # [B, C, HIDDEN_DIM]

        sections.append(
            f"## Step 1: KnowledgeEncoder 编码\n"
            f"- 候选 token_ids: {cand_token_ids.shape}\n"
            f"- 候选 embedding: {cand_emb_flat.shape} → reshape → {cand_emb.shape}"
        )

        # Phase 3: FeatureAdapter 处理查询（[B, D] 路径）
        with torch.no_grad():
            query_vec = feature_adapter(query_emb)   # [B, ADAPTER_DIM]

        # Phase 4: FeatureAdapter 处理候选（[B*C, D] → [B, C, ADAPTER_DIM]）
        cand_emb_flat_2d = cand_emb.view(B * C, HIDDEN_DIM)  # [B*C, HIDDEN_DIM]
        with torch.no_grad():
            cand_vecs_flat = feature_adapter(cand_emb_flat_2d)  # [B*C, ADAPTER_DIM]
        cand_vecs = cand_vecs_flat.view(B, C, ADAPTER_DIM)   # [B, C, ADAPTER_DIM]

        sections.append(
            f"## Step 2: FeatureAdapter\n"
            f"- 查询输出: {query_vec.shape}\n"
            f"- 候选输出: {cand_vecs_flat.shape} → reshape → {cand_vecs.shape}\n"
            f"- NaN 检查: {'无' if not torch.isnan(query_vec).any() and not torch.isnan(cand_vecs).any() else '有 NaN!'}"
        )

        # Phase 5: RefinedSelector 精排
        with torch.no_grad():
            scores, best_idx = refined_selector(query_vec, cand_vecs)

        sections.append(
            f"## Step 3: RefinedSelector\n"
            f"- scores: {scores.shape}\n"
            f"- best_idx: {best_idx.tolist()}\n"
            f"- NaN 检查: {'无' if not torch.isnan(scores).any() else '有 NaN!'}\n"
            f"- Inf 检查: {'无' if not torch.isinf(scores).any() else '有 Inf!'}"
        )

        # ── 断言 ──
        assert query_vec.shape == (B, ADAPTER_DIM)
        assert cand_vecs.shape == (B, C, ADAPTER_DIM)
        assert scores.shape == (B, C)
        assert best_idx.shape == (B,)
        assert (best_idx >= 0).all() and (best_idx < C).all(), (
            f"best_idx 越界: {best_idx}"
        )
        assert not torch.isnan(scores).any(), "scores 中有 NaN"
        assert not torch.isinf(scores).any(), "scores 中有 Inf"

        sections.append(
            f"## 最终结果\n✅ 所有断言通过\n- best_idx 值域: [0, {C})\n"
            f"- scores 范围: [{scores.min().item():.3f}, {scores.max().item():.3f}]"
        )

        report_path = _save_md_report("standard_flow", sections)
        print(f"\nMarkdown 报告: {report_path}")


class TestRefinedSelectorBatchOne:
    """B=1 边界情况。"""

    def test_batch_size_one(
        self,
        encoder: KnowledgeEncoder,
        feature_adapter: FeatureAdapter,
        refined_selector: RefinedSelector,
    ) -> None:
        """B=1 时精排流程正确运行，best_idx 形状为 [1]。"""
        B, C = 1, 8
        sections = [f"## 任务\nB=1 边界测试 (C={C})"]

        query_emb = torch.randn(B, HIDDEN_DIM)
        cand_token_ids = _make_token_ids(B * C, ANCHOR_LENGTH)
        cand_mask = torch.ones(B * C, ANCHOR_LENGTH, dtype=torch.bool)

        with torch.no_grad():
            cand_emb = encoder.encode_mean(cand_token_ids, cand_mask)  # [B*C, D]
            cand_emb = cand_emb.view(B, C, HIDDEN_DIM)

            query_vec = feature_adapter(query_emb)
            cand_vecs = feature_adapter(cand_emb.view(B * C, HIDDEN_DIM)).view(B, C, ADAPTER_DIM)
            scores, best_idx = refined_selector(query_vec, cand_vecs)

        assert scores.shape == (1, C)
        assert best_idx.shape == (1,)
        assert 0 <= best_idx.item() < C

        sections.append(
            f"## 结果\n- scores: {scores.shape}\n- best_idx: {best_idx.item()}\n✅ 通过"
        )
        _save_md_report("batch_one", sections)


class TestRefinedSelectorLargeCandidates:
    """大候选数 C=32。"""

    def test_large_candidate_count(
        self,
        encoder: KnowledgeEncoder,
        feature_adapter: FeatureAdapter,
        refined_selector: RefinedSelector,
    ) -> None:
        """C=32 时精排流程正确运行，输出形状与值域正确。"""
        B, C = 2, 32
        sections = [f"## 任务\n大候选数测试 (B={B}, C={C})"]

        query_emb = torch.randn(B, HIDDEN_DIM)
        cand_token_ids = _make_token_ids(B * C, ANCHOR_LENGTH)
        cand_mask = torch.ones(B * C, ANCHOR_LENGTH, dtype=torch.bool)

        with torch.no_grad():
            cand_emb = encoder.encode_mean(cand_token_ids, cand_mask).view(B, C, HIDDEN_DIM)
            query_vec = feature_adapter(query_emb)
            cand_vecs = feature_adapter(cand_emb.view(B * C, HIDDEN_DIM)).view(B, C, ADAPTER_DIM)
            scores, best_idx = refined_selector(query_vec, cand_vecs)

        assert scores.shape == (B, C)
        assert best_idx.shape == (B,)
        assert (best_idx >= 0).all() and (best_idx < C).all()

        sections.append(
            f"## 结果\n- scores: {scores.shape}\n"
            f"- best_idx: {best_idx.tolist()}\n✅ 通过"
        )
        _save_md_report("large_candidates", sections)


class TestRefinedSelectorWithMask:
    """带 padding mask 的推理场景。"""

    def test_mask_excludes_invalid_candidates(
        self,
        encoder: KnowledgeEncoder,
        feature_adapter: FeatureAdapter,
        refined_selector: RefinedSelector,
    ) -> None:
        """
        当 mask 中只有前 K 个候选有效时，best_idx 应在 [0, K) 范围内。
        """
        B, C, K = 4, 16, 4   # 仅前 4 个候选有效
        sections = [f"## 任务\n带 mask 推理 (B={B}, C={C}, 有效候选 K={K})"]

        query_emb = torch.randn(B, HIDDEN_DIM)
        cand_token_ids = _make_token_ids(B * C, ANCHOR_LENGTH)
        cand_mask_enc = torch.ones(B * C, ANCHOR_LENGTH, dtype=torch.bool)

        with torch.no_grad():
            cand_emb = encoder.encode_mean(cand_token_ids, cand_mask_enc).view(B, C, HIDDEN_DIM)
            query_vec = feature_adapter(query_emb)
            cand_vecs = feature_adapter(cand_emb.view(B * C, HIDDEN_DIM)).view(B, C, ADAPTER_DIM)

            # 仅前 K 个候选有效
            valid_mask = torch.zeros(B, C, dtype=torch.bool)
            valid_mask[:, :K] = True

            scores, best_idx = refined_selector(query_vec, cand_vecs, mask=valid_mask)

        # 被遮蔽的候选分数应为 -inf
        assert torch.isinf(scores[:, K:]).all() and (scores[:, K:] < 0).all(), (
            "被 mask=False 的候选分数应为 -inf"
        )
        # best_idx 应在有效候选范围内
        assert (best_idx >= 0).all() and (best_idx < K).all(), (
            f"best_idx 应在 [0, {K})，实际: {best_idx}"
        )

        sections.append(
            f"## 结果\n"
            f"- 有效候选 mask: 前 {K} 个\n"
            f"- best_idx: {best_idx.tolist()}（均 < {K}）\n"
            f"- 无效候选分数全为 -inf: ✅\n✅ 通过"
        )
        _save_md_report("with_mask", sections)
