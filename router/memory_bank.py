"""
双存储知识库架构

功能：
    - FusionBank：存储 LLMLingua 压缩后的高密度知识 token IDs（K_f=64），供知识编码器注入使用
    - AnchorBank：存储原文截断的 token IDs（K_a=128），供聚类计算 embedding 及更新路由索引使用
    - DualKnowledgeStore：统一管理双 Bank + 倒排索引，支持动态增删、近似 cluster 分配和物理压缩重聚类

设计约束：
    - FusionBank 服务于注入（压缩语义），AnchorBank 服务于路由索引（原文语义），两者不可混用
    - 热更新（add_entries）必须在首次 compact_and_recluster 之后才可调用（否则 RuntimeError）
    - knowledge_num 满时 add_entries 直接 RuntimeError，不自动扩容
    - 所有写操作通过 threading.Lock 保护，支持推理（读）与更新（写）并发
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from config import RouterConfig
    from models.qwen_wrapper import KnowledgeEncoder
    from router.clustering import ClusteringResult


class FusionBank:
    """
    存储 LLMLingua 压缩后的 facts token IDs，形状 [N, K_f]，供知识编码器读取注入。

    参数：
        knowledge_num: 知识库最大条目数 N
        fusion_length: 每条知识的压缩 token 数 K_f（通常为 64）
        device: 存储设备，如 "cpu" 或 "cuda:0"
    """

    def __init__(self, knowledge_num: int, fusion_length: int, device: str) -> None:
        """
        初始化空 FusionBank，预分配显存。

        参数：
            knowledge_num: 知识库最大条目数
            fusion_length: 每条知识压缩 token 数
            device: 存储设备

        返回：
            None
        """
        self.knowledge_num = knowledge_num
        self.fusion_length = fusion_length
        self.device = device
        # Phase 1: 预分配 [N, K_f] 零填充张量（0 通常是 pad_token_id 附近的安全值，由外部覆盖）
        self.data: torch.Tensor = torch.zeros(
            (knowledge_num, fusion_length), dtype=torch.long, device=device
        )

    def update_all(self, token_ids: torch.Tensor) -> None:
        """
        全量替换 Fusion Bank 数据（Phase 0 离线构建或训练期批量写入）。

        参数：
            token_ids: [N, K_f] 的 token ID 张量，dtype=torch.long

        返回：
            None

        异常：
            AssertionError: 形状或 dtype 不匹配
        """
        assert token_ids.shape == (self.knowledge_num, self.fusion_length), (
            f"update_all 形状不匹配: 期望 {(self.knowledge_num, self.fusion_length)}, "
            f"实际 {tuple(token_ids.shape)}"
        )
        assert token_ids.dtype == torch.long, (
            f"token_ids 必须为 torch.long，实际 {token_ids.dtype}"
        )
        self.data.copy_(token_ids.to(self.device))

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        """
        批量读取指定条目的压缩 token IDs。

        参数：
            ids: [B] 的条目索引张量，dtype=torch.long

        返回：
            [B, K_f] 的 token ID 张量

        异常：
            AssertionError: 索引越界
        """
        assert ids.dtype == torch.long, f"ids 必须为 torch.long，实际 {ids.dtype}"
        assert ids.max().item() < self.knowledge_num, (
            f"索引越界: max(ids)={ids.max().item()} >= knowledge_num={self.knowledge_num}"
        )
        return self.data[ids]


class AnchorBank:
    """
    存储原文截断的 token IDs，形状 [N, K_a]，供聚类计算 embedding、更新路由 Keys 使用。

    原文语义（与 query 同一语义空间）是 Router 精确检索的前提，
    故不能与压缩后的 Fusion Bank 混用。

    参数：
        knowledge_num: 知识库最大条目数 N
        anchor_length: 每条知识的原文截断 token 数 K_a（通常为 128）
        device: 存储设备
    """

    def __init__(self, knowledge_num: int, anchor_length: int, device: str) -> None:
        """
        初始化空 AnchorBank，预分配存储。

        参数：
            knowledge_num: 知识库最大条目数
            anchor_length: 每条知识原文截断 token 数
            device: 存储设备

        返回：
            None
        """
        self.knowledge_num = knowledge_num
        self.anchor_length = anchor_length
        self.device = device
        self.data: torch.Tensor = torch.zeros(
            (knowledge_num, anchor_length), dtype=torch.long, device=device
        )

    def update_all(self, token_ids: torch.Tensor) -> None:
        """
        全量替换 Anchor Bank 数据。

        参数：
            token_ids: [N, K_a] 的 token ID 张量，dtype=torch.long

        返回：
            None

        异常：
            AssertionError: 形状或 dtype 不匹配
        """
        assert token_ids.shape == (self.knowledge_num, self.anchor_length), (
            f"update_all 形状不匹配: 期望 {(self.knowledge_num, self.anchor_length)}, "
            f"实际 {tuple(token_ids.shape)}"
        )
        assert token_ids.dtype == torch.long, (
            f"token_ids 必须为 torch.long，实际 {token_ids.dtype}"
        )
        self.data.copy_(token_ids.to(self.device))

    def get_embeddings(
        self,
        encoder: "KnowledgeEncoder",
        valid_mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        对所有有效条目流式编码，返回 mean-pool 后的稠密向量。

        仅编码 valid_mask=True 的条目，分 chunk 进行以避免 OOM。
        编码器调用 encoder.encode_mean(ids, mask)，其中 mask 为 padding mask（0=pad）。

        参数：
            encoder: KnowledgeEncoder 实例（Qwen3 前 N 层）
            valid_mask: [N] 的 bool 张量，True 表示有效条目
            chunk_size: 每次编码的批量大小，默认 64

        返回：
            [N_valid, D] 的 float 张量，D 为 encoder 隐藏维度

        异常：
            AssertionError: valid_mask 形状不匹配
        """
        assert valid_mask.shape == (self.knowledge_num,), (
            f"valid_mask 形状不匹配: 期望 ({self.knowledge_num},), 实际 {tuple(valid_mask.shape)}"
        )
        # Phase 1: 收集有效条目索引
        valid_indices = torch.where(valid_mask)[0]  # [N_valid]
        n_valid = valid_indices.shape[0]
        if n_valid == 0:
            # 无有效条目，返回空张量（D 维度由调用方处理）
            return torch.zeros((0,), dtype=torch.float, device=self.device)

        # Phase 2: 分 chunk 流式编码
        n_chunks = (n_valid + chunk_size - 1) // chunk_size
        embeddings_list: List[torch.Tensor] = []
        for start in tqdm(range(0, n_valid, chunk_size), total=n_chunks, desc="[Recluster] 编码 AnchorBank", unit="batch"):
            end = min(start + chunk_size, n_valid)
            chunk_ids_idx = valid_indices[start:end]  # [chunk, ]
            chunk_token_ids = self.data[chunk_ids_idx]  # [chunk, K_a]
            # 构造 padding mask（非零位置为有效 token）
            chunk_mask = (chunk_token_ids != 0).long()  # [chunk, K_a]
            # encoder.encode_mean 返回 [chunk, D]
            chunk_emb = encoder.encode_mean(
                chunk_token_ids.to(encoder.device),
                chunk_mask.to(encoder.device),
            )
            embeddings_list.append(chunk_emb.to(self.device))

        return torch.cat(embeddings_list, dim=0)  # [N_valid, D]


class DualKnowledgeStore:
    """
    统一管理 FusionBank + AnchorBank + 倒排索引，提供动态增删和重聚类接口。

    关键数据结构：
        fusion_bank:       [N, K_f]    — 压缩知识 token IDs
        anchor_bank:       [N, K_a]    — 原文截断 token IDs
        valid_mask:        [N]         — 有效条目标记（逻辑删除用）
        inverted_index:    [N]         — 按 cluster 排序的全局数据 ID
        cluster_offsets:   [C+1]       — 每个 cluster 在 inverted_index 中的起始偏移
        cluster_counts:    [C]         — 每个 cluster 的有效条目数
        pca_matrix:        [D, D]      — PCA 旋转矩阵（全量 recluster 后保存，用于近似分配）
        pca_mean:          [D]         — PCA 均值向量
        row_centroids:     [num_keys, D//2]  — 行聚类中心
        col_centroids:     [num_keys, D//2]  — 列聚类中心

    C = num_keys²，num_keys = √knowledge_num

    参数：
        config: RouterConfig 实例
        device: 存储设备
    """

    def __init__(
        self,
        config: "RouterConfig",  # type: ignore[name-defined]
        fusion_length: int,
        anchor_length: int,
        device: str,
    ) -> None:
        """
        初始化双 Bank 和所有索引 buffer。

        参数：
            config: RouterConfig，包含 knowledge_num、recluster_threshold 等字段
            fusion_length: Fusion Bank 每条知识的压缩 token 数（来自 ModelConfig.fusion_length）
            anchor_length: Anchor Bank 每条知识的原文截断 token 数（来自 ModelConfig.anchor_length）
            device: 存储设备

        返回：
            None

        异常：
            AssertionError: knowledge_num 必须是完全平方数
        """
        self._config = config
        self.device = device

        n = config.knowledge_num
        num_keys = int(n**0.5)
        assert num_keys * num_keys == n, (
            f"knowledge_num 必须是完全平方数，实际 {n}（√N={n**0.5:.4f}）"
        )
        self._num_keys = num_keys
        c = num_keys * num_keys  # = N

        # Phase 1: 初始化双 Bank
        self.fusion_bank = FusionBank(n, fusion_length, device)
        self.anchor_bank = AnchorBank(n, anchor_length, device)

        # Phase 2: 初始化索引 buffer
        self.valid_mask: torch.Tensor = torch.zeros(n, dtype=torch.bool, device=device)
        self.inverted_index: torch.Tensor = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )
        self.cluster_offsets: torch.Tensor = torch.zeros(
            c + 1, dtype=torch.long, device=device
        )
        self.cluster_counts: torch.Tensor = torch.zeros(
            c, dtype=torch.long, device=device
        )

        # Phase 3: 近似分配状态（recluster 后才有值）
        self.pca_matrix: Optional[torch.Tensor] = None  # [D, D]
        self.pca_mean: Optional[torch.Tensor] = None  # [D]
        self.row_centroids: Optional[torch.Tensor] = None  # [num_keys, D//2]
        self.col_centroids: Optional[torch.Tensor] = None  # [num_keys, D//2]

        # Phase 4: 动态更新状态
        self.next_free: int = 0
        self.change_counter: int = 0

        # Phase 5: 并发锁（保护写操作）
        self._lock = threading.Lock()

    def add_entries(
        self,
        fusion_token_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        encoder: "KnowledgeEncoder",
    ) -> None:
        """
        热更新：批量添加新知识条目（调用方负责预处理/压缩/截断）。

        分配策略：使用 encoder 对 anchor_token_ids 做真实编码后，
        调用 SubspaceClustering.assign_approximate 做高精度 cluster 分配，
        复用上次 recluster 的 PCA 状态，避免全量重聚类。

        参数：
            fusion_token_ids: [B, K_f] 的压缩 token IDs，dtype=torch.long
            anchor_token_ids: [B, K_a] 的原文截断 token IDs，dtype=torch.long
            encoder: KnowledgeEncoder 实例，用于编码 anchor_token_ids

        返回：
            None

        异常：
            RuntimeError: 尚未完成首次 recluster（pca_matrix 为 None）
            RuntimeError: knowledge_num 已满
            AssertionError: 输入形状或 dtype 不合法
        """
        # 延迟导入避免循环依赖
        from router.clustering import SubspaceClustering  # type: ignore[import]

        # Phase 1: 前置校验
        assert fusion_token_ids.dtype == torch.long, (
            f"fusion_token_ids 必须为 torch.long，实际 {fusion_token_ids.dtype}"
        )
        assert anchor_token_ids.dtype == torch.long, (
            f"anchor_token_ids 必须为 torch.long，实际 {anchor_token_ids.dtype}"
        )
        assert fusion_token_ids.ndim == 2 and anchor_token_ids.ndim == 2, (
            "fusion_token_ids 和 anchor_token_ids 必须为 2D 张量"
        )
        b = fusion_token_ids.shape[0]
        assert anchor_token_ids.shape[0] == b, (
            f"fusion 和 anchor 条目数不一致: {b} vs {anchor_token_ids.shape[0]}"
        )

        if self.pca_matrix is None:
            raise RuntimeError(
                "add_entries 失败：尚未完成首次聚类初始化。"
                "请先调用 compact_and_recluster(encoder) 初始化 pca_matrix 和 centroids，"
                "之后才能使用热更新功能。"
            )

        with self._lock:
            if self.next_free + b > self._config.knowledge_num:
                raise RuntimeError(
                    f"knowledge_num 已满（next_free={self.next_free}, "
                    f"新增={b}, 上限={self._config.knowledge_num}）。"
                    f"请先调用 compact_and_recluster 进行物理压缩后再添加。"
                )

            # Phase 2: 写入双 Bank
            start, end = self.next_free, self.next_free + b
            self.fusion_bank.data[start:end].copy_(fusion_token_ids.to(self.device))
            self.anchor_bank.data[start:end].copy_(anchor_token_ids.to(self.device))
            self.valid_mask[start:end] = True

            # Phase 3: 高精度 cluster 分配（真实 encoder 编码 + PCA 最近邻）
            with torch.no_grad():
                mask = (anchor_token_ids != 0).long()
                anchor_emb = (
                    encoder.encode_mean(
                        anchor_token_ids.to(encoder.device),
                        mask.to(encoder.device),
                    )
                    .cpu()
                    .float()
                    .numpy()
                )  # [B, D]
            grid_indices_np = SubspaceClustering.assign_approximate(
                anchor_emb,
                self.pca_matrix.cpu().numpy(),
                self.pca_mean.cpu().numpy(),
                self.row_centroids.cpu().numpy(),
                self.col_centroids.cpu().numpy(),
                self._num_keys,
            )  # np.ndarray [B]
            grid_indices = torch.tensor(
                grid_indices_np, dtype=torch.long, device=self.device
            )  # [B]
            self._append_to_inverted_index(
                torch.arange(start, end, dtype=torch.long, device=self.device),
                grid_indices,
            )

            # Phase 4: 更新计数器
            self.next_free += b
            self.change_counter += b

    def delete_entries(self, ids: List[int]) -> None:
        """
        逻辑删除指定条目（将 valid_mask 置为 False，不立即释放存储）。

        路由检索时会自动跳过 valid_mask=False 的条目。
        物理释放需通过 compact_and_recluster 完成。

        参数：
            ids: 待删除的条目全局索引列表

        返回：
            None

        异常：
            AssertionError: 索引越界
        """
        if not ids:
            return
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        assert ids_tensor.max().item() < self._config.knowledge_num, (
            f"delete_entries 索引越界: max(ids)={ids_tensor.max().item()}"
        )
        with self._lock:
            self.valid_mask[ids_tensor] = False
            self.change_counter += len(ids)

    def should_recluster(self) -> bool:
        """
        判断是否需要触发全量重聚类。

        触发条件：change_counter / N_valid > recluster_threshold（默认 0.1）
        无有效条目时返回 False。

        返回：
            bool，True 表示应调用 compact_and_recluster
        """
        n_valid = int(self.valid_mask.sum().item())
        if n_valid == 0:
            return False
        return (self.change_counter / n_valid) > self._config.recluster_threshold

    def compact_and_recluster(self, encoder: "KnowledgeEncoder", chunk_size: int = 512) -> None:
        """
        物理压缩 + 全量重聚类（重建所有索引）。

        执行步骤：
            Phase 1: 收集所有 valid_mask=True 的条目 → 紧凑排列到 bank[0..N_valid-1]
            Phase 2: 流式编码 anchor_bank → [N_valid, D]
            Phase 3: 调用 SubspaceClustering.fit 做独立子空间聚类
            Phase 4: 更新 pca_matrix、pca_mean、row/col_centroids、inverted_index
            Phase 5: 重置 next_free = N_valid，change_counter = 0

        整个操作受 self._lock 保护，期间写操作被阻塞。

        参数：
            encoder:    KnowledgeEncoder 实例，用于编码 anchor_bank
            chunk_size: 编码时每批大小（默认 512，越大 GPU 利用率越高，越小越省显存）

        返回：
            None
        """
        # 延迟导入避免循环依赖（clustering 依赖本文件）
        from router.clustering import SubspaceClustering  # type: ignore[import]

        with self._lock:
            # Phase 1: 收集 valid 条目索引并紧凑排列
            t1 = time.time()
            valid_indices = torch.where(self.valid_mask)[0]  # [N_valid]
            n_valid = valid_indices.shape[0]

            if n_valid == 0:
                self.change_counter = 0
                return

            # 紧凑拷贝 fusion_bank 和 anchor_bank
            self.fusion_bank.data[:n_valid].copy_(self.fusion_bank.data[valid_indices])
            self.anchor_bank.data[:n_valid].copy_(self.anchor_bank.data[valid_indices])

            # 清空后续槽位
            self.fusion_bank.data[n_valid:].zero_()
            self.anchor_bank.data[n_valid:].zero_()

            # 重建 valid_mask
            self.valid_mask.zero_()
            self.valid_mask[:n_valid] = True
            print(f"[Recluster] Phase 1 紧凑排列完成，n_valid={n_valid} ({time.time() - t1:.1f}s)")

            # Phase 2: 全量编码（仅对有效条目）
            t2 = time.time()
            print(f"[Recluster] Phase 2 编码开始 (n={n_valid}, chunk_size={chunk_size}) ...")
            valid_mask_compact = self.valid_mask.clone()
            embeddings = self.anchor_bank.get_embeddings(
                encoder, valid_mask_compact, chunk_size=chunk_size
            )  # [N_valid, D]
            print(f"[Recluster] Phase 2 编码完成 ({time.time() - t2:.1f}s)")

            # Phase 3: 独立子空间聚类
            t3 = time.time()
            print(f"[Recluster] Phase 3 子空间聚类开始 ...")
            embeddings_np = embeddings.cpu().float().numpy()
            result: "ClusteringResult" = SubspaceClustering.fit(
                embeddings_np, self._num_keys
            )
            print(f"[Recluster] Phase 3 聚类完成 ({time.time() - t3:.1f}s)")

            # Phase 4: 更新聚类状态
            t4 = time.time()
            self.pca_matrix = torch.tensor(
                result.pca_matrix, dtype=torch.float, device=self.device
            )
            self.pca_mean = torch.tensor(
                result.pca_mean, dtype=torch.float, device=self.device
            )
            self.row_centroids = torch.tensor(
                result.row_centroids, dtype=torch.float, device=self.device
            )
            self.col_centroids = torch.tensor(
                result.col_centroids, dtype=torch.float, device=self.device
            )

            # 重建倒排索引
            row_labels = torch.tensor(
                result.row_labels, dtype=torch.long, device=self.device
            )  # [N_valid]
            col_labels = torch.tensor(
                result.col_labels, dtype=torch.long, device=self.device
            )  # [N_valid]
            grid_indices = row_labels * self._num_keys + col_labels  # [N_valid]
            self._rebuild_inverted_index(
                torch.arange(n_valid, dtype=torch.long, device=self.device),
                grid_indices,
            )
            print(f"[Recluster] Phase 4-5 索引重建完成，max_cluster={result.max_cluster_size} ({time.time() - t4:.1f}s)")

            # Phase 5: 重置计数器
            self.next_free = n_valid
            self.change_counter = 0

    def _rebuild_inverted_index(
        self,
        data_indices: torch.Tensor,
        grid_indices: torch.Tensor,
    ) -> None:
        """
        全量重建倒排索引（recluster 后调用）。

        按 grid_indices 排序 data_indices，构造 inverted_index、cluster_offsets、cluster_counts。

        参数：
            data_indices: [N_valid] 的全局数据 ID，dtype=torch.long
            grid_indices: [N_valid] 的 cluster grid 索引，dtype=torch.long

        返回：
            None
        """
        n = data_indices.shape[0]
        c = self._num_keys * self._num_keys

        # 按 grid_indices 排序
        sorted_order = torch.argsort(grid_indices, stable=True)
        sorted_data = data_indices[sorted_order]
        sorted_grid = grid_indices[sorted_order]

        # 写入 inverted_index（前 N_valid 槽位）
        self.inverted_index[:n].copy_(sorted_data)
        self.inverted_index[n:].fill_(-1)

        # 计算 cluster_counts 和 cluster_offsets
        counts = torch.bincount(sorted_grid, minlength=c)  # [C]
        self.cluster_counts[:c].copy_(counts)
        offsets = torch.zeros(c + 1, dtype=torch.long, device=self.device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        self.cluster_offsets.copy_(offsets)

    def _append_to_inverted_index(
        self,
        data_indices: torch.Tensor,
        grid_indices: torch.Tensor,
    ) -> None:
        """
        热更新时追加新条目到倒排索引（近似，不重排已有条目）。

        将新条目简单追加到对应 cluster 的末尾（通过更新 cluster_counts），
        精确性低于全量重建，在下次 compact_and_recluster 时纠正。

        参数：
            data_indices: [B] 新条目的全局 ID（= next_free..next_free+B-1）
            grid_indices: [B] 新条目的近似 cluster 索引

        返回：
            None
        """
        b = data_indices.shape[0]
        for i in range(b):
            gidx = int(grid_indices[i].item())
            # 当前该 cluster 末尾位置 = offsets[gidx] + counts[gidx]
            pos = int(self.cluster_offsets[gidx].item()) + int(
                self.cluster_counts[gidx].item()
            )
            if pos < self._config.knowledge_num:
                self.inverted_index[pos] = data_indices[i]
                self.cluster_counts[gidx] += 1

    def get_rowcol_labels(self) -> torch.Tensor:
        """
        从倒排索引重建每条知识条目的 (row, col) cluster 标签。

        实现：遍历所有非空 cluster（cluster_counts > 0），
        将 inverted_index 中对应位置的 entry_id 映射到该 cluster 的 (row, col)。
        时间复杂度 O(min(N, num_cells))，仅遍历非空格。

        返回：
            [next_free, 2] torch.long 张量：
                [:, 0] = row cluster 标签（值域 [0, num_keys)）
                [:, 1] = col cluster 标签（值域 [0, num_keys)）
            若 next_free=0 返回空张量 shape=(0, 2)

        前置：
            必须在 compact_and_recluster 之后调用（pca_matrix 不为 None）
            否则所有条目 row/col 均为 0（倒排索引未初始化）

        异常：
            无，但若 pca_matrix 为 None，结果无意义（静默返回全 0）
        """
        n = self.next_free
        if n == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=self.device)

        num_keys = self._num_keys
        id_to_rowcol = torch.zeros(n, 2, dtype=torch.long, device=self.device)

        # Phase 1: 仅遍历非空 cluster，避免全量枚举 num_keys² 个格子
        nonempty_cells = torch.where(self.cluster_counts > 0)[0]  # [n_nonempty]

        for g_tensor in nonempty_cells:
            g = int(g_tensor.item())
            row = g // num_keys
            col = g % num_keys
            start = int(self.cluster_offsets[g].item())
            count = int(self.cluster_counts[g].item())
            # Phase 2: 取出该格中的 entry_id，过滤 -1（填充值）
            entries = self.inverted_index[start : start + count]
            valid = entries[entries >= 0]
            if valid.numel() > 0:
                id_to_rowcol[valid, 0] = row
                id_to_rowcol[valid, 1] = col

        return id_to_rowcol

    def save_state(self, path: str) -> None:
        """
        序列化所有状态到文件（供 checkpoint 保存）。

        保存内容：双 Bank 数据、所有索引 buffer、聚类状态、动态计数器。

        参数：
            path: 保存文件路径（.pt 格式）

        返回：
            None
        """
        state = {
            "fusion_bank": self.fusion_bank.data.cpu(),
            "anchor_bank": self.anchor_bank.data.cpu(),
            "valid_mask": self.valid_mask.cpu(),
            "inverted_index": self.inverted_index.cpu(),
            "cluster_offsets": self.cluster_offsets.cpu(),
            "cluster_counts": self.cluster_counts.cpu(),
            "pca_matrix": self.pca_matrix.cpu()
            if self.pca_matrix is not None
            else None,
            "pca_mean": self.pca_mean.cpu() if self.pca_mean is not None else None,
            "row_centroids": self.row_centroids.cpu()
            if self.row_centroids is not None
            else None,
            "col_centroids": self.col_centroids.cpu()
            if self.col_centroids is not None
            else None,
            "next_free": self.next_free,
            "change_counter": self.change_counter,
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """
        从文件恢复所有状态。

        参数：
            path: 保存文件路径（.pt 格式）

        返回：
            None
        """
        state = torch.load(path, map_location=self.device)
        self.fusion_bank.data.copy_(state["fusion_bank"].to(self.device))
        self.anchor_bank.data.copy_(state["anchor_bank"].to(self.device))
        self.valid_mask.copy_(state["valid_mask"].to(self.device))
        self.inverted_index.copy_(state["inverted_index"].to(self.device))
        self.cluster_offsets.copy_(state["cluster_offsets"].to(self.device))
        self.cluster_counts.copy_(state["cluster_counts"].to(self.device))
        self.pca_matrix = (
            state["pca_matrix"].to(self.device)
            if state["pca_matrix"] is not None
            else None
        )
        self.pca_mean = (
            state["pca_mean"].to(self.device) if state["pca_mean"] is not None else None
        )
        self.row_centroids = (
            state["row_centroids"].to(self.device)
            if state["row_centroids"] is not None
            else None
        )
        self.col_centroids = (
            state["col_centroids"].to(self.device)
            if state["col_centroids"] is not None
            else None
        )
        self.next_free = int(state["next_free"])
        self.change_counter = int(state["change_counter"])
