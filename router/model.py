"""
router/model.py — MemoryRouter 端到端路由器

功能：
    将 ProductKeyMemory（粗排）、FeatureAdapter（特征适配）、RefinedSelector（精排）
    整合为统一路由器，提供两个对外接口：
      - forward：训练用，返回完整 RouterOutput（含用于损失计算的分数）
      - retrieve：推理用，返回 Fusion Bank 的压缩知识 token IDs

设计要点：
    1. 同一个 FeatureAdapter 分别用于 query 和候选编码，Batch Centering 各批次独立计算
    2. 候选先经 KnowledgeEncoder（Qwen3 前 encoder_depth 层）上下文化，再经 adapter mean pool 降维
    3. encoder 注册为 nn.Module 子模块；Phase 1 冻结时通过 filter(requires_grad) 跳过
    4. retrieve 以 @torch.no_grad() 修饰，避免训练时误用

依赖：
    router/memory_gate.py (ProductKeyMemory)
    router/feature_adapter.py (FeatureAdapter)
    router/refined_selector.py (RefinedSelector)
    router/memory_bank.py (DualKnowledgeStore)
    models/qwen_wrapper.py (KnowledgeEncoder)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

from router.feature_adapter import FeatureAdapter
from router.memory_gate import ProductKeyMemory
from router.refined_selector import RefinedSelector

if TYPE_CHECKING:
    from config import RouterConfig
    from models.qwen_wrapper import KnowledgeEncoder
    from router.memory_bank import DualKnowledgeStore


@dataclass
class RouterOutput:
    """
    MemoryRouter.forward 的完整输出，训练时用于计算 Router 损失。

    参数：
        best_id:       [B] — 精排后最优知识条目的全局 ID（long）
        candidates:    [B, num_candidates] — 粗排输出的候选知识 ID（long）
        coarse_scores: (scores_1, scores_2) — PKM 行/列匹配分数，各 [B, num_keys]
        fine_scores:   [B, num_candidates] — RefinedSelector 精排原始分数（未 softmax）
    """

    best_id: torch.Tensor
    candidates: torch.Tensor
    coarse_scores: Tuple[torch.Tensor, torch.Tensor]
    fine_scores: torch.Tensor


class MemoryRouter(nn.Module):
    """
    端到端知识路由器：将 query embedding 映射到最优知识条目 ID。

    可训练参数（Phase 1，约 15M）：
        - pkm: query_proj + row_key_proj + col_key_proj（~8.2M）
        - adapter: LayerNorm + Linear + LayerNorm（~0.5M）
        - selector: 2 层 Transformer + score_head + scale（~6.3M）

    冻结参数（Phase 1）：
        - encoder: KnowledgeEncoder（Qwen3 前 encoder_depth 层），训练循环负责设置 requires_grad=False

    路由流程：
        query_embedding [B, D]
          ↓ Step 1: FeatureAdapter → q_adapted [B, 512]
          ↓ Step 2: ProductKeyMemory → candidates [B, C], scores_1/2 [B, num_keys]
          ↓ Step 3: encoder(anchor_bank[candidates]) + FeatureAdapter → cand_vecs [B, C, 512]
          ↓ Step 4: RefinedSelector(q_adapted, cand_vecs) → fine_scores [B, C], best_local [B]
          ↓ Step 5: best_id = candidates[arange(B), best_local] → [B]

    参数：
        config: RouterConfig，包含 dim, adapter_dim, refined_num_heads, refined_num_layers 等
        encoder: KnowledgeEncoder 共享实例（跨模块共享，不重复创建）
    """

    def __init__(self, config: "RouterConfig", encoder: "KnowledgeEncoder") -> None:
        """
        初始化 MemoryRouter，构建三个可训练子模块并注册共享 encoder。

        参数：
            config:  RouterConfig，所有超参显式来源于此，不使用默认值
            encoder: KnowledgeEncoder 共享实例，Phase 1 应已冻结

        返回：
            None
        """
        super().__init__()

        # Phase 1: 三个可训练子模块
        self.pkm = ProductKeyMemory(config)
        self.adapter = FeatureAdapter(config.dim, config.adapter_dim)
        self.selector = RefinedSelector(
            config.adapter_dim, config.refined_num_heads, config.refined_num_layers
        )

        # Phase 2: 共享编码器（注册为子模块，支持 state_dict / device 迁移）
        # Phase 1 冻结时，训练循环通过 filter(lambda p: p.requires_grad, parameters()) 跳过
        self.encoder = encoder

        # Phase 3: 缓存超参，供 forward 内部维度检查
        self._adapter_dim: int = config.adapter_dim

    def forward(
        self,
        query_embedding: torch.Tensor,
        store: "DualKnowledgeStore",
    ) -> RouterOutput:
        """
        训练用完整前向传播，返回含分数的 RouterOutput 供损失计算。

        参数：
            query_embedding: [B, D] — 问题的 Qwen3 embedding（D = config.dim）
            store:           DualKnowledgeStore — 双存储库，提供倒排索引与 AnchorBank

        返回：
            RouterOutput，包含：
                best_id:       [B] long — 最优知识全局 ID
                candidates:    [B, C] long — 粗排候选 ID
                coarse_scores: (Tensor[B, num_keys], Tensor[B, num_keys])
                fine_scores:   [B, C] float — 精排原始分数

        异常：
            AssertionError: 输入形状不合法
        """
        assert query_embedding.ndim == 2, (
            f"query_embedding 必须为 2D [B, D]，实际 ndim={query_embedding.ndim}"
        )

        B = query_embedding.shape[0]

        # ─── Step 1: FeatureAdapter — query 侧特征适配 ───────────────────────────
        # 输入 [B, D]（2D），adapter 跳过 mean pooling 直接输出 [B, adapter_dim]
        q_adapted = self.adapter(query_embedding)  # [B, adapter_dim]

        # ─── Step 2: ProductKeyMemory — 粗排候选检索 ─────────────────────────────
        # 返回 4-tuple；第 4 项 q_pkm 为 PKM 内部 L2 归一化子查询，此处不使用
        candidates, scores_1, scores_2, _ = self.pkm(query_embedding, store)
        # candidates: [B, num_candidates] long
        # scores_1/2: [B, num_keys] float

        C = candidates.shape[1]

        # ─── Step 3: KnowledgeEncoder + FeatureAdapter — 候选侧编码 ──────────────
        # 3a. 从 AnchorBank 取候选原文 token IDs：[B, C, K_a]
        anchor_ids = store.anchor_bank.data[candidates]  # [B, C, K_a]
        K_a = anchor_ids.shape[2]

        # 3b. 展平为 [B*C, K_a]，方便批量编码
        flat_ids = anchor_ids.reshape(B * C, K_a)  # [B*C, K_a]

        # 3c. 构造 attention mask（0=pad，1=有效）
        flat_mask = (flat_ids != 0).long()  # [B*C, K_a]

        # 3d. KnowledgeEncoder 上下文化（Qwen3 前 encoder_depth 层 + 双向注意力）
        cand_enc = self.encoder.forward(flat_ids, flat_mask)  # [B*C, K_a, D]

        # 3e. FeatureAdapter mean pooling + 投影 → [B*C, adapter_dim]
        # adapter 3D 路径：[B*C, K_a, D] → masked mean pool → [B*C, adapter_dim]
        cand_vecs_flat = self.adapter(cand_enc, flat_mask.bool())  # [B*C, adapter_dim]

        # 3f. 恢复批次维度 [B, C, adapter_dim]
        cand_vecs = cand_vecs_flat.view(B, C, self._adapter_dim)  # [B, C, adapter_dim]

        # ─── Step 4: RefinedSelector — 精排评分 ──────────────────────────────────
        fine_scores, best_local_idx = self.selector(q_adapted, cand_vecs)
        # fine_scores:     [B, C] float（原始分数，训练时用 softmax + CE loss）
        # best_local_idx:  [B] long，值域 [0, C)

        # ─── Step 5: 局部 ID → 全局知识 ID ───────────────────────────────────────
        arange_b = torch.arange(B, device=candidates.device)
        best_id = candidates[arange_b, best_local_idx]  # [B] long

        # 范围断言：best_id 必须在已使用槽位内
        assert best_id.max().item() < store.next_free, (
            f"best_id 越界：max(best_id)={best_id.max().item()} >= store.next_free={store.next_free}"
        )

        return RouterOutput(
            best_id=best_id,
            candidates=candidates,
            coarse_scores=(scores_1, scores_2),
            fine_scores=fine_scores,
        )

    @torch.no_grad()
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        store: "DualKnowledgeStore",
    ) -> torch.Tensor:
        """
        推理专用接口（无梯度），从 Fusion Bank 取出最优知识的压缩 token IDs。

        与 forward 的区别：
            - @torch.no_grad() 修饰，完全不构建计算图
            - 只返回 knowledge_ids，丢弃路由分数

        参数：
            query_embedding: [B, D] — 问题的 Qwen3 embedding
            store:           DualKnowledgeStore

        返回：
            [B, K_f] long — 最优知识条目的压缩 token IDs（来自 FusionBank）
        """
        out = self.forward(query_embedding, store)
        return store.fusion_bank[out.best_id]  # [B, K_f]
