"""
tests/integration/test_router_model_flow.py — MemoryRouter 端到端集成测试

使用真实 Qwen3-0.6B 模型，验证 MemoryRouter 完整前向/推理流程的
形状正确性、best_id 范围、梯度流以及 retrieve 无梯度语义。

测试场景：
    1. 标准前向（B=2）：RouterOutput 各字段形状正确
    2. 批次大小 B=1 边界情况
    3. best_id 范围断言：∈ [0, store.next_free)
    4. 梯度流：冻结 encoder 后，adapter/selector 参数有梯度
    5. retrieve 无梯度：输出无 grad_fn
    6. forward/retrieve 输出一致性：两次相同输入结果一致（eval 模式）

测试结果保存为 Markdown 报告至 tests/outputs/router_model/。

说明：
    - 使用小规模 DualKnowledgeStore（knowledge_num=64，8²）加速测试
    - KnowledgeEncoder 使用 encoder_depth=2（仅前 2 层），减少 CPU 计算量
    - 所有测试在 CPU 上运行
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen_wrapper import KnowledgeEncoder, load_base_model  # noqa: E402
from router.memory_bank import DualKnowledgeStore  # noqa: E402
from router.model import MemoryRouter, RouterOutput  # noqa: E402


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

MODEL_PATH = str(PROJECT_ROOT / "Qwen3-0.6B")
ENCODER_DEPTH = 2        # 仅前 2 层，降低 CPU 压力
HIDDEN_DIM = 1024        # Qwen3-0.6B hidden dim
ADAPTER_DIM = 512
KEY_PROJ_DIM = 512
KNOWLEDGE_NUM = 64       # 8²，最小完全平方数（集成测试用）
NUM_CANDIDATES = 4       # 粗排候选数（小值加速）
ANCHOR_LENGTH = 16       # AnchorBank token 数（集成测试用短序列）
FUSION_LENGTH = 16       # FusionBank token 数
NUM_HEADS = 8
NUM_LAYERS = 2
DEVICE = "cpu"

OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "router_model"


# ─────────────────────────────────────────────
# 辅助：Markdown 报告
# ─────────────────────────────────────────────


def _save_md_report(test_name: str, sections: List[str]) -> str:
    """
    保存测试结果到 Markdown 文件，返回文件路径。

    参数：
        test_name: 测试名称（用于文件名和标题）
        sections:  Markdown 段落列表（每项为一个 ## 节）

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
# 模块级 Fixtures（模型只加载一次）
# ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """加载 Qwen3-0.6B，冻结全部参数，整个 module 共享。"""
    model = load_base_model(MODEL_PATH, bf16=False)  # CPU 用 float32
    return model


@pytest.fixture(scope="module")
def encoder(base_model):
    """构造 KnowledgeEncoder（encoder_depth=2 加速），冻结。"""
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
def router_config():
    """构造 RouterConfig mock（集成测试规模）。"""
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
    cfg.refined_num_heads = NUM_HEADS
    cfg.refined_num_layers = NUM_LAYERS
    return cfg


@pytest.fixture(scope="module")
def router(encoder, router_config):
    """构造并返回 eval 模式的 MemoryRouter。"""
    r = MemoryRouter(router_config, encoder)
    r.eval()
    return r


@pytest.fixture(scope="module")
def store(router_config):
    """
    构造小规模 DualKnowledgeStore，手动注入倒排索引（不调用真实聚类）。

    设计：KNOWLEDGE_NUM=64，8² 个 grid cell，每格 1 条知识。
    """
    store_cfg = MagicMock()
    store_cfg.knowledge_num = KNOWLEDGE_NUM
    store_cfg.recluster_threshold = 0.1
    s = DualKnowledgeStore(
        store_cfg,
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

    # 注入 PKM keys（否则粗排分数无意义）
    # 注：这里 store 已有 row_centroids/col_centroids，但 PKM keys 需单独更新

    # Anchor Bank 填充随机 token IDs（token_id > 0 表示有效）
    s.anchor_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, ANCHOR_LENGTH), dtype=torch.long)

    # Fusion Bank 填充随机 token IDs
    s.fusion_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, FUSION_LENGTH), dtype=torch.long)

    return s


# ─────────────────────────────────────────────
# 辅助：更新 PKM keys 后执行 forward
# ─────────────────────────────────────────────


def _update_pkm_keys(router: MemoryRouter, store: DualKnowledgeStore) -> None:
    """将 store 的 row/col centroids 同步到 router.pkm。"""
    router.pkm.update_keys(store.row_centroids, store.col_centroids)


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────


class TestRouterModelForwardShapes:
    """端到端 forward 形状验证。"""

    def test_standard_forward_B2(self, router, store) -> None:
        """标准场景 B=2：RouterOutput 各字段 shape 正确，生成 Markdown 报告。"""
        B = 2
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        with torch.no_grad():
            out = router.forward(query, store)

        # 断言
        assert out.best_id.shape == (B,)
        assert out.best_id.dtype == torch.long
        assert out.candidates.shape == (B, NUM_CANDIDATES)
        assert out.fine_scores.shape == (B, NUM_CANDIDATES)
        assert len(out.coarse_scores) == 2
        assert out.coarse_scores[0].shape[0] == B
        assert out.coarse_scores[1].shape[0] == B

        # Markdown 报告
        path = _save_md_report(
            "test_standard_forward_B2",
            [
                "## 任务\n标准前向传播（B=2），验证 RouterOutput 各字段形状。",
                "## 输入\n"
                f"- query: {tuple(query.shape)}\n"
                f"- store.next_free: {store.next_free}\n"
                f"- num_candidates: {NUM_CANDIDATES}",
                "## 输出\n"
                f"- best_id shape: {tuple(out.best_id.shape)}\n"
                f"- candidates shape: {tuple(out.candidates.shape)}\n"
                f"- fine_scores shape: {tuple(out.fine_scores.shape)}\n"
                f"- coarse_scores[0] shape: {tuple(out.coarse_scores[0].shape)}\n"
                f"- coarse_scores[1] shape: {tuple(out.coarse_scores[1].shape)}",
                "## 结论\n所有形状验证通过。",
            ],
        )
        print(f"\n[报告] {path}")

    def test_forward_B1_edge_case(self, router, store) -> None:
        """边界情况 B=1：单样本不应报错，形状正确。"""
        B = 1
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        with torch.no_grad():
            out = router.forward(query, store)

        assert out.best_id.shape == (1,)
        assert out.candidates.shape == (1, NUM_CANDIDATES)
        assert out.fine_scores.shape == (1, NUM_CANDIDATES)


class TestRouterModelBestIdRange:
    """验证 best_id 值域约束。"""

    def test_best_id_in_valid_range(self, router, store) -> None:
        """best_id 所有元素应在 [0, store.next_free=KNOWLEDGE_NUM) 内。"""
        B = 4
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        with torch.no_grad():
            out = router.forward(query, store)

        assert out.best_id.min().item() >= 0
        assert out.best_id.max().item() < store.next_free, (
            f"best_id 越界：{out.best_id.max().item()} >= {store.next_free}"
        )

    def test_best_id_is_candidate_member(self, router, store) -> None:
        """best_id[b] 必须是 candidates[b] 中的成员（来自精排 argmax）。"""
        B = 3
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        with torch.no_grad():
            out = router.forward(query, store)

        for b in range(B):
            assert out.best_id[b].item() in out.candidates[b].tolist(), (
                f"批次 {b}：best_id={out.best_id[b].item()} 不在 candidates 中"
            )

        # Markdown 报告
        path = _save_md_report(
            "test_best_id_range",
            [
                "## 任务\n验证 best_id 值域：∈ [0, store.next_free) 且来自 candidates。",
                "## 输入\n"
                f"- B={B}, KNOWLEDGE_NUM={KNOWLEDGE_NUM}, NUM_CANDIDATES={NUM_CANDIDATES}",
                "## 输出\n"
                f"- best_id: {out.best_id.tolist()}\n"
                f"- candidates[0]: {out.candidates[0].tolist()}\n"
                f"- max(best_id)={out.best_id.max().item()} < next_free={store.next_free}",
                "## 结论\nbest_id 值域验证通过。",
            ],
        )
        print(f"\n[报告] {path}")


class TestRouterModelGradientFlow:
    """验证 Phase 1 训练场景：冻结 encoder，adapter/selector 有梯度。"""

    def test_adapter_selector_have_gradients(self, encoder, router_config) -> None:
        """
        冻结 encoder 时，经 forward + loss.backward() 后：
            - adapter 参数应有梯度
            - selector 参数应有梯度
            - encoder 参数不应有梯度
        """
        # 构造独立 router（不共享 eval fixture，需要 train 模式）
        r = MemoryRouter(router_config, encoder)
        r.train()

        # encoder 已冻结（fixture 中设置 requires_grad=False）

        # 构造小 store（同 module fixture）
        s_cfg = MagicMock()
        s_cfg.knowledge_num = KNOWLEDGE_NUM
        s_cfg.recluster_threshold = 0.1
        s = DualKnowledgeStore(s_cfg, fusion_length=FUSION_LENGTH, anchor_length=ANCHOR_LENGTH, device=DEVICE)
        num_keys = 8
        c = num_keys * num_keys
        s.inverted_index = torch.arange(KNOWLEDGE_NUM, dtype=torch.long)
        s.cluster_counts = torch.ones(c, dtype=torch.long)
        s.cluster_offsets = torch.arange(c + 1, dtype=torch.long)
        s.row_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)
        s.col_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)
        s.valid_mask = torch.ones(KNOWLEDGE_NUM, dtype=torch.bool)
        s.next_free = KNOWLEDGE_NUM
        s.anchor_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, ANCHOR_LENGTH), dtype=torch.long)
        s.fusion_bank.data = torch.randint(1, 1000, (KNOWLEDGE_NUM, FUSION_LENGTH), dtype=torch.long)

        r.pkm.update_keys(s.row_centroids, s.col_centroids)

        B = 2
        query = torch.randn(B, HIDDEN_DIM)
        out = r.forward(query, s)

        # 构造简单 loss：fine_scores 的均值
        loss = out.fine_scores.mean()
        loss.backward()

        # 验证 adapter 参数有梯度
        adapter_grads = [
            (name, p.grad)
            for name, p in r.adapter.named_parameters()
            if p.requires_grad
        ]
        assert len(adapter_grads) > 0, "adapter 无可训练参数"
        for name, grad in adapter_grads:
            assert grad is not None, f"adapter.{name} 梯度为 None"

        # 验证 selector 参数有梯度
        selector_grads = [
            (name, p.grad)
            for name, p in r.selector.named_parameters()
            if p.requires_grad
        ]
        assert len(selector_grads) > 0, "selector 无可训练参数"
        for name, grad in selector_grads:
            assert grad is not None, f"selector.{name} 梯度为 None"

        # 验证 encoder 参数无梯度（requires_grad=False 时 grad 为 None）
        for name, p in r.encoder.named_parameters():
            assert p.grad is None, f"encoder.{name} 意外产生梯度"

        # Markdown 报告
        path = _save_md_report(
            "test_gradient_flow",
            [
                "## 任务\nPhase 1 场景：冻结 encoder，验证 adapter/selector 梯度流。",
                "## 设置\n"
                f"- encoder 冻结（requires_grad=False）\n"
                f"- loss = fine_scores.mean()，backward()",
                "## 结果\n"
                f"- adapter 有梯度的参数数：{len(adapter_grads)}\n"
                f"- selector 有梯度的参数数：{len(selector_grads)}\n"
                f"- encoder 参数全部无梯度：True",
                "## 结论\n梯度流验证通过。",
            ],
        )
        print(f"\n[报告] {path}")


class TestRouterModelRetrieve:
    """验证 retrieve 接口。"""

    def test_retrieve_shape(self, router, store) -> None:
        """retrieve 返回 [B, FUSION_LENGTH] long 张量。"""
        B = 3
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        result = router.retrieve(query, store)

        assert result.shape == (B, FUSION_LENGTH)
        assert result.dtype == torch.long

    def test_retrieve_no_grad_fn(self, router, store) -> None:
        """retrieve 输出无 grad_fn（@torch.no_grad() 语义）。"""
        B = 2
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        result = router.retrieve(query, store)

        assert result.grad_fn is None, "retrieve 输出不应有 grad_fn"

    def test_retrieve_consistency(self, router, store) -> None:
        """相同输入两次 retrieve 应得到相同结果（eval 模式）。"""
        B = 2
        _update_pkm_keys(router, store)
        query = torch.randn(B, HIDDEN_DIM)

        with torch.no_grad():
            result1 = router.retrieve(query, store)
            result2 = router.retrieve(query, store)

        assert torch.equal(result1, result2), "eval 模式下 retrieve 结果不稳定"

        # Markdown 报告
        path = _save_md_report(
            "test_retrieve",
            [
                "## 任务\n验证 retrieve 接口：形状、无梯度、eval 一致性。",
                "## 输入\n"
                f"- query: {tuple(query.shape)}\n"
                f"- B={B}, FUSION_LENGTH={FUSION_LENGTH}",
                "## 输出\n"
                f"- result shape: {tuple(result1.shape)}\n"
                f"- grad_fn: {result1.grad_fn}\n"
                f"- 两次结果一致: {torch.equal(result1, result2)}",
                "## 结论\nretrieve 接口验证通过。",
            ],
        )
        print(f"\n[报告] {path}")
