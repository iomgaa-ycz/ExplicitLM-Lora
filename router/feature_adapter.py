"""
router/feature_adapter.py — 特征适配器：冻结 embedding 投影 + 防 Feature Collapse

功能：
    将冻结 Qwen3 embedding 投影到适配空间（adapter_dim），供精排系统（RefinedSelector）使用。
    通过 Batch Centering + 双 LayerNorm 解决生成模型 embedding 的各向异性问题（cone effect）。

参数：
    in_dim:      输入 embedding 维度（通常为 model.hidden_dim，如 1024）
    adapter_dim: 输出适配维度（通常为 router.key_proj_dim，如 512）

输入/输出：
    输入 x:   [B, S, D] 序列形式，或 [B, D] 已 pool 形式
    输出:     [B, adapter_dim]

核心流程（参考项目验证方案）：
    [B,S,D] → Batch Centering → LayerNorm(in_dim) → Linear → Tanh → ×√adapter_dim
            → masked mean pool → LayerNorm(adapter_dim) → [B, adapter_dim]
    [B,D]   → Batch Centering → LayerNorm(in_dim) → Linear → Tanh → ×√adapter_dim
            → LayerNorm(adapter_dim) → [B, adapter_dim]

设计说明：
    - Batch Centering：动态减去批次均值，无需训练即可消除冻结 embedding 的模式偏差
    - scale_factor = √adapter_dim：防止 Linear 投影后方差收缩
    - Tanh：有界激活，避免均值偏移
    - Output LayerNorm：强制修正分布，防止特征坍缩
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class FeatureAdapter(nn.Module):
    """
    特征适配器：将冻结 Qwen3 embedding 投影到适配空间。

    参数：
        in_dim:      输入维度（通常为 hidden_dim=1024）
        adapter_dim: 输出适配维度（通常为 key_proj_dim=512）
    """

    def __init__(self, in_dim: int, adapter_dim: int) -> None:
        """
        初始化 FeatureAdapter。

        参数：
            in_dim:      输入维度（通常为 model.hidden_dim，如 1024）
            adapter_dim: 输出适配维度（通常为 router.key_proj_dim，如 512）
        """
        super().__init__()
        assert in_dim > 0, f"in_dim 必须为正整数，实际: {in_dim}"
        assert adapter_dim > 0, f"adapter_dim 必须为正整数，实际: {adapter_dim}"

        self.in_dim = in_dim
        self.adapter_dim = adapter_dim
        # √adapter_dim 常数缩放因子（非可学习），防止投影后方差收缩
        self._scale: float = math.sqrt(adapter_dim)

        # Phase 1 正则化：输入归一化（在去中心化之后）
        self.input_norm = nn.LayerNorm(in_dim)
        # 特征投影（无 bias 会导致 Tanh 对称输出，保留 bias 允许偏移学习）
        self.proj = nn.Linear(in_dim, adapter_dim, bias=True)
        # Phase 2 正则化：输出归一化，强制修正分布防止 Feature Collapse
        self.output_norm = nn.LayerNorm(adapter_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播：冻结 embedding → 适配向量。

        参数：
            x:    输入张量，形状为 [B, S, D]（序列）或 [B, D]（已 pool）
            mask: 可选 padding mask，形状为 [B, S]，True 表示有效 token
                  仅在 x 为 [B, S, D] 时使用；传入 [B, D] 时忽略

        返回：
            适配向量 Tensor[B, adapter_dim]

        异常：
            AssertionError: 输入维度不为 2 或 3，或 mask 形状不匹配
        """
        assert x.dim() in (2, 3), f"输入维度必须为 2 或 3，实际: {x.dim()}"
        is_3d = x.dim() == 3

        # Phase 1: Batch Centering — 动态减去批次均值，消除冻结 embedding 的模式偏差
        if is_3d:
            # [B, S, D]：跨 batch 和 seq 维度计算均值
            mean_vec = x.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
        else:
            # [B, D]：仅跨 batch 维度计算均值
            mean_vec = x.mean(dim=0, keepdim=True)  # [1, D]
        x = x - mean_vec

        # Phase 2: Input LayerNorm — 归一化当前批次
        x = self.input_norm(x)

        # Phase 3: 投影 + Tanh + 缩放
        # x: [B, S, D] → [B, S, adapter_dim] 或 [B, D] → [B, adapter_dim]
        x = self.proj(x)
        x = torch.tanh(x)
        x = x * self._scale

        # Phase 4: Masked Mean Pooling（仅 3D 路径需要）
        if is_3d:
            if mask is not None:
                assert mask.dim() == 2, (
                    f"mask 维度必须为 2 [B, S]，实际: {mask.dim()}"
                )
                assert mask.shape[0] == x.shape[0] and mask.shape[1] == x.shape[1], (
                    f"mask 形状 {mask.shape} 与 x 前两维 {x.shape[:2]} 不匹配"
                )
                # True 位置参与平均，False 位置（padding）不参与
                mask_float = mask.unsqueeze(-1).float()  # [B, S, 1]
                x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
            else:
                x = x.mean(dim=1)  # [B, S, adapter_dim] → [B, adapter_dim]

        # Phase 5: Output LayerNorm — 强制修正分布，防止 Feature Collapse
        x = self.output_norm(x)

        return x
