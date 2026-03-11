"""
router/refined_selector.py — 精排系统：2 层 Transformer Post-LN 交叉编码器

功能：
    从粗排（ProductKeyMemory）返回的 ~32 候选中精选 Top-1 知识条目。
    通过拼接查询向量与候选向量后用 Transformer 建模交互，捕捉查询与候选的语义匹配。

参数：
    adapter_dim: FeatureAdapter 输出维度（通常为 router.key_proj_dim=512）
    num_heads:   Transformer 注意力头数（必须整除 adapter_dim）
    num_layers:  Transformer 层数（默认 2，MVP 轻量设计）

输入/输出：
    query_vec: [B, adapter_dim]   — FeatureAdapter 处理后的查询向量
    cand_vecs: [B, C, adapter_dim] — C 个候选知识编码向量
    mask:      [B, C] bool，True=有效候选（可选，推理时候选数不足时使用）
    → scores:  [B, C]   — 精排原始分数（训练时做 softmax + CE loss）
    → best_idx:[B]      — argmax 候选 ID（推理时直接使用）

核心设计（参考项目验证方案）：
    [query; cand_1;...;cand_C] → TransformerEncoder(2层 Post-LN)
    → 取候选部分 [:,1:,:] → score_head(adapter_dim→1) → ×scale → scores
    Post-LN（norm_first=False）是参考项目实际验证的稳定方案。

可学习温度（scale=10.0 初始化）：
    scale 参数放大分数以使 softmax 分布更尖锐，提升精排区分度。
    参考项目验证该初始化有效，无需手动调整温度超参。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RefinedSelector(nn.Module):
    """
    精排选择器：2 层 Transformer 交叉编码器，从候选中精选 Top-1 知识条目。

    参数：
        adapter_dim: FeatureAdapter 输出维度（必须能被 num_heads 整除）
        num_heads:   Transformer 注意力头数
        num_layers:  Transformer 层数
    """

    def __init__(
        self,
        adapter_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        """
        初始化 RefinedSelector。

        参数：
            adapter_dim: FeatureAdapter 输出维度（通常为 512）
            num_heads:   Transformer 注意力头数（adapter_dim 必须能被 num_heads 整除）
            num_layers:  Transformer 层数（通常为 2）

        异常：
            AssertionError: adapter_dim 不能被 num_heads 整除
        """
        super().__init__()
        assert adapter_dim % num_heads == 0, (
            f"adapter_dim={adapter_dim} 必须能被 num_heads={num_heads} 整除"
        )

        self.adapter_dim = adapter_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 2 层 Post-LN Transformer 编码器（batch_first=True，无需转置）
        # dim_feedforward = adapter_dim * 4 = 2048（标准 4x 比例）
        # norm_first=False（Post-LN，参考项目验证方案）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=adapter_dim,
            nhead=num_heads,
            dim_feedforward=adapter_dim * 4,
            batch_first=True,
            norm_first=False,  # Post-LN，参考项目验证稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # 候选评分头：adapter_dim → 1（每个候选输出一个标量分数）
        self.score_head = nn.Linear(adapter_dim, 1)

        # 可学习温度参数：放大分数使 softmax 分布更尖锐
        # 参考项目验证初始值 10.0 有效
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        query_vec: Tensor,
        cand_vecs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        交叉编码查询与候选，输出精排分数和最优候选索引。

        参数：
            query_vec: 查询向量 Tensor[B, adapter_dim]（FeatureAdapter 输出）
            cand_vecs: 候选向量 Tensor[B, C, adapter_dim]（FeatureAdapter 批量输出）
            mask:      可选 padding mask Tensor[B, C]，True=有效候选
                       推理时候选数不足时传入，训练时通常为 None

        返回：
            scores:   Tensor[B, C] — 精排原始分数（训练时做 softmax + CE loss）
            best_idx: Tensor[B]    — argmax 候选 ID，值域 [0, C)（推理时直接使用）

        异常：
            AssertionError: 向量维度不匹配，或 mask 形状与 cand_vecs 不一致
        """
        assert query_vec.dim() == 2, (
            f"query_vec 维度必须为 2 [B, adapter_dim]，实际: {query_vec.dim()}"
        )
        assert cand_vecs.dim() == 3, (
            f"cand_vecs 维度必须为 3 [B, C, adapter_dim]，实际: {cand_vecs.dim()}"
        )
        assert query_vec.shape[-1] == self.adapter_dim, (
            f"query_vec 最后一维 {query_vec.shape[-1]} 与 adapter_dim={self.adapter_dim} 不匹配"
        )
        assert cand_vecs.shape[-1] == self.adapter_dim, (
            f"cand_vecs 最后一维 {cand_vecs.shape[-1]} 与 adapter_dim={self.adapter_dim} 不匹配"
        )

        B, C, _ = cand_vecs.shape

        if mask is not None:
            assert mask.shape == (B, C), (
                f"mask 形状 {mask.shape} 与 cand_vecs 前两维 ({B}, {C}) 不匹配"
            )

        # Phase 1: 拼接 [query; cand_1; ...; cand_C] → [B, 1+C, adapter_dim]
        query_expanded = query_vec.unsqueeze(1)          # [B, 1, adapter_dim]
        seq = torch.cat([query_expanded, cand_vecs], dim=1)  # [B, 1+C, adapter_dim]

        # Phase 2: 2 层 Post-LN Transformer 交叉编码（建模查询与候选交互）
        encoded = self.transformer(seq)  # [B, 1+C, adapter_dim]

        # Phase 3: 提取候选部分，计算精排分数
        cand_encoded = encoded[:, 1:, :]               # [B, C, adapter_dim]
        scores = self.score_head(cand_encoded).squeeze(-1)  # [B, C]
        scores = scores * self.scale                    # 可学习温度缩放

        # Phase 4: 处理 padding mask（推理时候选不足的格子填充 -inf）
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        best_idx = scores.argmax(dim=-1)  # [B]，值域 [0, C)

        return scores, best_idx
