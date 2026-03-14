"""
Product Key Memory 粗排模块

功能：
    - ProductKeyMemory：两维独立 Product Key Memory，O(√N) 时间将输入 embedding 映射到
      约 num_candidates 个候选知识条目 ID。
    - 行/列 Keys 以 register_buffer 存储于模块内，不参与梯度，每 epoch 通过 update_keys() 更新。
    - 候选查询采用倒排索引（DualKnowledgeStore.inverted_index），支持每格多条目（热更新场景）。
      通过 config.max_candidates_per_cell 控制：-1=全量，1=1:1 简单映射。

设计约束：
    - knowledge_num 必须是完全平方数（num_keys = √knowledge_num 为整数）
    - update_keys 必须在首次 compact_and_recluster 后调用，之后 forward 才能得到有意义的分数
    - row_keys / col_keys 均为 [num_keys, key_proj_dim] 的 float 张量
    - forward 返回 4-tuple: (candidates, scores_1, scores_2, q_adapted)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from config import RouterConfig
    from router.memory_bank import DualKnowledgeStore


class ProductKeyMemory(nn.Module):
    """
    两维独立 Product Key Memory，粗排检索约 num_candidates 个候选知识条目。

    工作原理：
        1. 将输入 embedding 投影并分裂为行/列两个子查询（各 D//2 维）
        2. 行/列子查询分别与 register_buffer 中的 row_keys/col_keys 做点积评分
        3. 各取 top-K_COARSE=4，笛卡儿积得到 16 个候选 grid cell
        4. 从 DualKnowledgeStore 倒排索引中查询每个 grid cell 内的知识条目 ID
        5. 汇聚后截断/填充至固定长度 num_candidates

    参数：
        config: RouterConfig，包含 dim, query_dim, key_proj_dim, knowledge_num,
                num_candidates, temperature, max_candidates_per_cell
    """

    def __init__(self, config: "RouterConfig") -> None:
        """
        初始化 ProductKeyMemory。

        参数：
            config: RouterConfig 实例

        返回：
            None

        异常：
            AssertionError: knowledge_num 不是完全平方数
        """
        super().__init__()

        num_keys = int(config.knowledge_num**0.5)
        assert num_keys * num_keys == config.knowledge_num, (
            f"knowledge_num 必须是完全平方数，实际 {config.knowledge_num}（√N={config.knowledge_num**0.5:.4f}）"
        )

        # Phase 1: 可训练投影层（无 bias，遵循 PKM 惯例）
        self.query_proj = nn.Linear(config.dim, config.query_dim, bias=False)
        self.row_key_proj = nn.Linear(
            config.key_proj_dim, config.key_proj_dim, bias=False
        )
        self.col_key_proj = nn.Linear(
            config.key_proj_dim, config.key_proj_dim, bias=False
        )

        # Phase 2: 非训练 keys（register_buffer：随模型 save/load/device 迁移，无梯度）
        # 初始化为全零；首次 compact_and_recluster 后通过 update_keys() 写入真实聚类中心
        self.register_buffer(
            "row_keys", torch.zeros(num_keys, config.key_proj_dim, dtype=torch.float)
        )
        self.register_buffer(
            "col_keys", torch.zeros(num_keys, config.key_proj_dim, dtype=torch.float)
        )

        # Phase 3: 超参数
        self.num_keys: int = num_keys
        self.K_COARSE: int = 4  # 每维 top-k，共 4×4=16 个候选 cluster
        self.temperature: float = config.temperature
        self.num_candidates: int = config.num_candidates
        self.max_candidates_per_cell: int = config.max_candidates_per_cell

    def update_keys(self, row_keys: torch.Tensor, col_keys: torch.Tensor) -> None:
        """
        更新行/列聚类 Keys（每 epoch 在 compact_and_recluster 后调用）。

        Keys 不参与梯度，此方法通过 copy_ 就地更新 register_buffer。
        典型调用：
            store.compact_and_recluster(encoder)
            pkm.update_keys(store.row_centroids, store.col_centroids)

        参数：
            row_keys: [num_keys, key_proj_dim] 的行聚类中心，dtype=torch.float
            col_keys: [num_keys, key_proj_dim] 的列聚类中心，dtype=torch.float

        返回：
            None

        异常：
            AssertionError: 形状或 dtype 不匹配
        """
        expected_shape = (self.num_keys, self.row_keys.shape[1])
        assert row_keys.shape == expected_shape, (
            f"row_keys 形状不匹配: 期望 {expected_shape}，实际 {tuple(row_keys.shape)}"
        )
        assert col_keys.shape == expected_shape, (
            f"col_keys 形状不匹配: 期望 {expected_shape}，实际 {tuple(col_keys.shape)}"
        )
        assert row_keys.dtype == torch.float, (
            f"row_keys 必须为 torch.float，实际 {row_keys.dtype}"
        )
        assert col_keys.dtype == torch.float, (
            f"col_keys 必须为 torch.float，实际 {col_keys.dtype}"
        )
        self.row_keys.copy_(row_keys.to(self.row_keys.device))
        self.col_keys.copy_(col_keys.to(self.col_keys.device))

    def forward(
        self,
        embedding: torch.Tensor,
        store: "DualKnowledgeStore",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        粗排检索主流程。

        参数：
            embedding: [B, dim] 的输入向量（来自 Qwen3 hidden state 或 FeatureAdapter 输出）
            store: DualKnowledgeStore 实例，提供倒排索引用于候选查询

        返回：
            candidates:  Tensor[B, num_candidates]  — 候选知识条目 ID（long）
            scores_1:    Tensor[B, num_keys]         — 行匹配分数（训练损失用）
            scores_2:    Tensor[B, num_keys]         — 列匹配分数（训练损失用）
            q_adapted:   Tensor[B, key_proj_dim]     — L2 归一化行子查询
            valid_mask:  Tensor[B, num_candidates]   — bool，True=真实候选，False=空位

        异常：
            AssertionError: embedding 形状不合法
        """
        assert embedding.ndim == 2, (
            f"embedding 必须为 2D 张量 [B, D]，实际 ndim={embedding.ndim}"
        )
        assert embedding.shape[1] == self.query_proj.in_features, (
            f"embedding dim={embedding.shape[1]} 与 query_proj.in_features="
            f"{self.query_proj.in_features} 不匹配"
        )

        # Phase 1: 投影 → 分割 → L2 归一化（查询侧）
        q = self.query_proj(embedding)  # [B, query_dim=1024]
        q1, q2 = q.chunk(2, dim=-1)  # [B, key_proj_dim=512] × 2
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)

        # Phase 2: Key 投影 + L2 归一化（key 侧，register_buffer 无梯度）
        # row_keys: [num_keys, key_proj_dim]
        k1 = F.normalize(self.row_key_proj(self.row_keys), p=2, dim=-1)  # [√N, 512]
        k2 = F.normalize(self.col_key_proj(self.col_keys), p=2, dim=-1)  # [√N, 512]

        # Phase 3: 温度缩放点积分数
        scores_1 = (q1 @ k1.T) / self.temperature  # [B, √N]
        scores_2 = (q2 @ k2.T) / self.temperature  # [B, √N]

        # Phase 4: Top-K + 笛卡儿积 → grid_indices [B, K_COARSE²]
        top_rows = scores_1.topk(self.K_COARSE, dim=-1).indices  # [B, 4]
        top_cols = scores_2.topk(self.K_COARSE, dim=-1).indices  # [B, 4]
        # 广播笛卡儿积：top_rows[b, i] * num_keys + top_cols[b, j]
        grid_indices = (
            top_rows.unsqueeze(2).expand(-1, -1, self.K_COARSE) * self.num_keys
            + top_cols.unsqueeze(1).expand(-1, self.K_COARSE, -1)
        ).reshape(embedding.size(0), -1)  # [B, K_COARSE * K_COARSE = 16]

        # Phase 5: 倒排索引查询 → 候选知识条目 ID + 有效位掩码
        candidates, valid_mask = self._lookup_candidates(grid_indices, store)  # [B, num_candidates]

        return candidates, scores_1, scores_2, q1, valid_mask

    def _lookup_candidates(
        self,
        grid_indices: torch.Tensor,
        store: "DualKnowledgeStore",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        倒排索引查询：从选中的 K_COARSE² 个 grid cell 收集候选知识条目 ID。

        支持两种模式（由 max_candidates_per_cell 控制）：
            - max_candidates_per_cell = -1：全量倒排（每格取所有条目）
            - max_candidates_per_cell > 0：每格最多取该数目（=1 时退化为 1:1 简单映射）

        不足 num_candidates 时用 0 填充并在 valid_mask 中标记为 False，
        不再循环重复填充（循环填充会导致 GT 多次出现，精排产生矛盾梯度）。

        参数：
            grid_indices: [B, K_COARSE²] 的 grid cell ID 张量（long）
            store: DualKnowledgeStore 实例

        返回：
            candidates:  [B, num_candidates] long — 候选知识条目 ID，空位填 0
            valid_mask:  [B, num_candidates] bool — True=真实候选，False=空位
        """
        b_size = grid_indices.size(0)
        result = torch.zeros(
            (b_size, self.num_candidates), dtype=torch.long, device=grid_indices.device
        )
        valid_mask = torch.zeros(
            (b_size, self.num_candidates), dtype=torch.bool, device=grid_indices.device
        )
        cap = self.max_candidates_per_cell  # -1=不限，>0=每格上限

        for b in range(b_size):
            ids_list = []
            for gidx in grid_indices[b]:
                gidx_int = int(gidx.item())
                offset = int(store.cluster_offsets[gidx_int].item())
                count = int(store.cluster_counts[gidx_int].item())
                if count > 0:
                    chunk = store.inverted_index[offset : offset + count]
                    # 1:1 模式：每格只取前 cap 条
                    if cap > 0:
                        chunk = chunk[:cap]
                    ids_list.append(chunk)

            if ids_list:
                combined = torch.cat(ids_list)  # [total_count]
                total = combined.size(0)
                fill = min(total, self.num_candidates)
                result[b, :fill] = combined[:fill]
                valid_mask[b, :fill] = True
            # else: result[b] 保持全零，valid_mask[b] 保持全 False（空 store 的安全处理）

        return result, valid_mask
