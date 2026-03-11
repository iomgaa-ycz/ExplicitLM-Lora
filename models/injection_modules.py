"""
models/injection_modules.py — 知识注入模块

定义三种知识注入方式，共享统一接口 forward(hidden, knowledge, mask) → hidden。
所有注入方式均采用零初始化策略，确保训练初期等价于原始模型（残差恒等）。

模块结构：
  - RMSNorm：RMS Layer Normalization，与 Qwen3 内部一致
  - masked_mean_pool：屏蔽 padding 的均值池化工具函数
  - BaseInjection：抽象基类，统一接口定义
  - AttentionInjection：Cross-Attention + Null KV，主力注入方案（~4.2M 参数/层）
  - ConcatProjection：mean_pool + concat + MLP，备选方案（~12.6M 参数/层）
  - GatedInjection：per-dim gate × knowledge，轻量备选（~1K 参数/层）

关键设计原则：
  - PreNorm 残差：fn(norm(x), context) + x
  - 零初始化：训练初期 forward(x) ≈ x（无注入退化）
  - Null KV：AttentionInjection 在知识全 padding 时数值稳定
  - mask 约定：0 = padding（无效），1 = 有效 token
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助组件
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """
    RMS Layer Normalization。

    与 Qwen3 内部实现保持一致，使用可学习缩放参数 gamma，无偏置项。
    适合作为 Transformer 模块的 PreNorm 使用。

    参数：
        dim: 归一化维度（即最后一维大小）
        eps: 数值稳定性 epsilon，默认 1e-8

    输入/输出：
        x: [..., dim] → 归一化后 [..., dim]，形状不变
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        计算 RMS 归一化。

        参数：
            x: 输入张量，形状 [..., dim]

        返回：
            归一化后张量，形状 [..., dim]
        """
        # Phase 1：计算每行 L2 范数（沿最后一维）
        norm = x.norm(keepdim=True, dim=-1) * self.scale
        # Phase 2：归一化并应用可学习缩放
        return (x / norm.clamp(min=self.eps)) * self.gamma


def masked_mean_pool(
    knowledge: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    对知识张量做均值池化，屏蔽 padding 位置。

    参数：
        knowledge: 知识编码器输出，形状 [B, K, D]
        mask: padding mask，形状 [B, K]，约定 1=有效 0=padding。
              为 None 时对全部 token 做均值。

    返回：
        池化后的知识向量，形状 [B, 1, D]
        若某条样本全为 padding，返回全零向量（数值稳定兜底）

    关键实现：
        - 使用 clamp(min=1) 避免除以零
        - 广播 valid_mask 到 D 维后相乘，保持梯度流通畅
    """
    if mask is None:
        return knowledge.mean(dim=1, keepdim=True)

    # [B, K] → [B, K, 1]，1=有效位置
    valid = mask.unsqueeze(-1).to(knowledge.dtype)  # [B, K, 1]

    # 加权求和 / 有效 token 数量
    pooled = (knowledge * valid).sum(dim=1, keepdim=True)  # [B, 1, D]
    count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
    return pooled / count


# ─────────────────────────────────────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────────────────────────────────────


class BaseInjection(nn.Module, ABC):
    """
    知识注入模块抽象基类。

    所有注入方式必须实现 forward 方法，接受相同签名，返回注入后的 hidden states。

    mask 约定：0 = padding（无效），1 = 有效 token。
    所有子类应在初始化时使用零初始化，确保初始 forward(hidden, ...) ≈ hidden。
    """

    @abstractmethod
    def forward(
        self,
        hidden: Tensor,
        knowledge: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        注入知识到隐藏状态。

        参数：
            hidden:    当前层隐藏状态，形状 [B, L, D]
            knowledge: 知识编码器输出，形状 [B, K_f, D]
            mask:      知识 padding mask，形状 [B, K_f]，1=有效 0=padding

        返回：
            注入后的隐藏状态，形状 [B, L, D]
        """


# ─────────────────────────────────────────────────────────────────────────────
# AttentionInjection — 主力注入方案
# ─────────────────────────────────────────────────────────────────────────────


class AttentionInjection(BaseInjection):
    """
    Cross-Attention 注入模块（主力方案）。

    使用 PreNorm 残差 + Multi-Head Cross-Attention + Null KV 机制，
    将知识编码器输出通过交叉注意力融合到隐藏层。

    核心设计：
      - out_proj 零初始化：初始 attn_out ≈ 0 → output = hidden（无注入退化）
      - Null KV：在知识序列前拼接可学习 null_k/null_v，
                 知识全 padding 时 Query attend to null，数值稳定
      - PreNorm：Q = W_q(RMSNorm(hidden))，K/V 直接投影知识向量
      - 使用 F.scaled_dot_product_attention 支持 Flash Attention

    参数量：~4.2M/层（@D=1024, num_heads=8）
      - W_q/W_k/W_v/out_proj: 4 × (1024 × 1024 + 1024) ≈ 4.2M

    参数：
        hidden_dim: 隐藏层维度，须与基础模型一致，默认 1024
        num_heads:  多头注意力头数，默认 8（每头维度 128）
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) 必须整除 num_heads ({num_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # PreNorm
        self.pre_norm = RMSNorm(hidden_dim)

        # Q/K/V 投影（Query 来自 hidden，K/V 来自 knowledge）
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 输出投影：零初始化 → 初始 attn_out ≈ 0
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # 可学习 Null KV：知识全 pad 时 Query attend to null，避免 NaN
        self.null_k = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.null_v = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        logger.debug(
            "AttentionInjection 初始化完毕：hidden_dim=%d, num_heads=%d, head_dim=%d",
            hidden_dim,
            num_heads,
            self.head_dim,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        将 [B, S, D] 转换为 [B, H, S, head_dim]，便于 SDPA。

        参数：
            x: 形状 [B, S, D]

        返回：
            形状 [B, H, S, head_dim]
        """
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        knowledge: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Cross-Attention 注入前向。

        流程：
          Phase 1：PreNorm + Q 投影
            normed = RMSNorm(hidden)                 [B, L, D]
            Q = W_q(normed)                          [B, L, D]

          Phase 2：Null KV 拼接 + K/V 投影（K/V 分别拼接各自 null 向量）
            k_input = cat([null_k.expand(B,-1,-1), knowledge])  [B, K_f+1, D]
            v_input = cat([null_v.expand(B,-1,-1), knowledge])  [B, K_f+1, D]
            K = W_k(k_input)                                    [B, K_f+1, D]
            V = W_v(v_input)                                    [B, K_f+1, D]

          Phase 3：构建 key_padding_mask（True = 忽略位置）
            原始 mask: 1=有效 0=pad → 取反得 pad_flag: True=pad
            Null 位置始终有效（False），pad 到 [B, K_f+1]

          Phase 4：SDPA（支持 Flash Attention）
            attn_out = SDPA(Q, K, V, mask) → [B, H, L, head_dim]
            → reshape 为 [B, L, D]

          Phase 5：零初始化残差
            output = hidden + out_proj(attn_out)

        参数：
            hidden:    [B, L, D]
            knowledge: [B, K_f, D]
            mask:      [B, K_f]，1=有效 0=padding

        返回：
            注入后 hidden states，形状 [B, L, D]
        """
        B, L, D = hidden.shape
        K = knowledge.shape[1]

        # Phase 1：PreNorm + Q 投影
        normed = self.pre_norm(hidden)  # [B, L, D]
        Q = self.W_q(normed)  # [B, L, D]

        # Phase 2：Null KV 拼接，K/V 分别拼接各自的 null 向量
        null_k = self.null_k.expand(B, -1, -1)  # [B, 1, D]
        null_v = self.null_v.expand(B, -1, -1)  # [B, 1, D]
        k_input = torch.cat([null_k, knowledge], dim=1)  # [B, K_f+1, D]
        v_input = torch.cat([null_v, knowledge], dim=1)  # [B, K_f+1, D]
        K_proj = self.W_k(k_input)  # [B, K_f+1, D]
        V_proj = self.W_v(v_input)  # [B, K_f+1, D]

        # Phase 3：构建 attention mask
        # mask: [B, K_f]，1=有效 → 转为 bool mask（True=忽略）
        pad_flag = mask == 0  # [B, K_f]，True=padding 位置
        # Null KV 始终有效（False），pad 到前端
        attn_mask_bool = F.pad(pad_flag, (1, 0), value=False)  # [B, K_f+1]，True=忽略
        # SDPA 的 attn_mask 约定：True=参与计算，False=忽略 → 取反
        # 使用 float mask：-inf 表示忽略
        attn_mask_float = torch.zeros(
            B, 1, L, K + 1, dtype=hidden.dtype, device=hidden.device
        )
        attn_mask_float = attn_mask_float.masked_fill(
            attn_mask_bool.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, K_f+1]
            float("-inf"),
        )

        # Phase 4：多头分割 + SDPA
        Q_h = self._split_heads(Q)  # [B, H, L, head_dim]
        K_h = self._split_heads(K_proj)  # [B, H, K_f+1, head_dim]
        V_h = self._split_heads(V_proj)  # [B, H, K_f+1, head_dim]

        # scaled_dot_product_attention：[B, H, L, head_dim]
        attn_out = F.scaled_dot_product_attention(
            Q_h, K_h, V_h, attn_mask=attn_mask_float
        )

        # 合并多头：[B, H, L, head_dim] → [B, L, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Phase 5：零初始化残差（out_proj=0 → 初始 output ≈ hidden）
        return hidden + self.out_proj(attn_out)

    def get_out_proj_norm(self) -> float:
        """
        获取 out_proj 权重的平均绝对值，用于监控注入通道是否打开。

        返回：
            out_proj.weight 绝对值均值（float），训练初期应接近 0
        """
        return self.out_proj.weight.data.abs().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# ConcatProjection — 备选注入方案（消融实验 E5-D）
# ─────────────────────────────────────────────────────────────────────────────


class ConcatProjection(BaseInjection):
    """
    Concat + MLP 注入模块（备选，用于消融实验）。

    将知识 mean pool 后与 hidden 拼接，通过两层 MLP 投影，残差相加。
    末层零初始化确保训练初期 output ≈ hidden。

    参数量：~12.6M/层（@D=1024）
      - proj_in: (2D × 4D) + 4D ≈ 8.4M
      - proj_out: (4D × D) + D ≈ 4.2M

    参数：
        hidden_dim: 隐藏层维度，须与基础模型一致，默认 1024
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # MLP：[B, L, 2D] → [B, L, 4D] → [B, L, D]
        self.proj_in = nn.Linear(hidden_dim * 2, hidden_dim * 4, bias=True)
        self.proj_out = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)

        # LayerNorm（与 TD.md 规格一致，ConcatProjection 使用 LayerNorm）
        self.norm = nn.LayerNorm(hidden_dim)

        # 末层零初始化：初始 delta ≈ 0 → output = hidden（无注入退化）
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        logger.debug("ConcatProjection 初始化完毕：hidden_dim=%d", hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        knowledge: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Concat + MLP 注入前向。

        流程：
          Phase 1：Masked mean pool 压缩知识序列
            k_pooled = masked_mean_pool(knowledge, mask)  [B, 1, D]
            k_pooled = k_pooled.expand(-1, L, -1)         [B, L, D]

          Phase 2：拼接 + MLP 投影
            concat = cat([hidden, k_pooled], dim=-1)      [B, L, 2D]
            delta = proj_out(gelu(proj_in(concat)))       [B, L, D]
            delta = norm(delta)                           [B, L, D]

          Phase 3：残差相加（末层零初始化确保初始 delta ≈ 0）
            output = hidden + delta

        参数：
            hidden:    [B, L, D]
            knowledge: [B, K_f, D]
            mask:      [B, K_f]，1=有效 0=padding

        返回：
            注入后 hidden states，形状 [B, L, D]
        """
        B, L, D = hidden.shape

        # Phase 1：均值池化知识向量
        k_pooled = masked_mean_pool(knowledge, mask)  # [B, 1, D]
        k_pooled = k_pooled.expand(-1, L, -1)  # [B, L, D]

        # Phase 2：拼接 + MLP
        concat = torch.cat([hidden, k_pooled], dim=-1)  # [B, L, 2D]
        delta = self.proj_out(F.gelu(self.proj_in(concat)))  # [B, L, D]
        delta = self.norm(delta)  # [B, L, D]

        # Phase 3：残差
        return hidden + delta


# ─────────────────────────────────────────────────────────────────────────────
# GatedInjection — 轻量备选注入方案（消融实验 E5-D）
# ─────────────────────────────────────────────────────────────────────────────


class GatedInjection(BaseInjection):
    """
    Per-dimension Gate 注入模块（轻量备选，用于消融实验）。

    对知识向量 mean pool 后，通过可学习的 per-dim sigmoid gate 控制注入强度。
    参数量极少（D 个参数），计算开销最小。

    gate 零初始化：初始 gate=0 → sigmoid(0)=0.5 → 弱注入，训练中动态调整。

    参数量：~1K/层（@D=1024，仅 gate 向量）

    参数：
        hidden_dim: 隐藏层维度，须与基础模型一致，默认 1024
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # per-dim gate：零初始化 → sigmoid(0) = 0.5
        self.gate = nn.Parameter(torch.zeros(hidden_dim))

        logger.debug("GatedInjection 初始化完毕：hidden_dim=%d", hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        knowledge: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Per-dim Gate 注入前向。

        流程：
          Phase 1：Masked mean pool 压缩知识序列
            k_pooled = masked_mean_pool(knowledge, mask)  [B, 1, D]

          Phase 2：门控注入
            gate_val = sigmoid(gate)                      [D]
            output = hidden + gate_val * k_pooled         广播到 [B, L, D]

        参数：
            hidden:    [B, L, D]
            knowledge: [B, K_f, D]
            mask:      [B, K_f]，1=有效 0=padding

        返回：
            注入后 hidden states，形状 [B, L, D]
        """
        # Phase 1：均值池化知识向量
        k_pooled = masked_mean_pool(knowledge, mask)  # [B, 1, D]

        # Phase 2：门控相乘并残差相加（广播：[D] × [B, 1, D] → [B, L, D]）
        gate_val = torch.sigmoid(self.gate)  # [D]
        return hidden + gate_val * k_pooled

    def get_gate_stats(self) -> float:
        """
        获取 gate 绝对值均值，用于监控注入强度。

        返回：
            gate 绝对值均值（float）
        """
        return self.gate.data.abs().mean().item()
