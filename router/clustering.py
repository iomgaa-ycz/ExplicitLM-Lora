"""
独立子空间聚类模块

功能：
    - ClusteringResult：聚类结果数据类，包含 PCA 旋转矩阵、各子空间聚类中心、标签及诊断信息
    - SubspaceClustering：纯静态方法类，实现独立子空间聚类（PCA 旋转 + 交错分割 + 轴对齐递归二分）
      及近似 cluster 分配接口（供热更新场景调用）

设计约束：
    - SubspaceClustering 无实例状态，所有方法均为 @staticmethod
    - fit() 接收 numpy array，返回 ClusteringResult（含 numpy array 字段），由调用方转换为 torch tensor
    - num_keys 必须为 2 的幂次（保证递归二分终止且标签连续）
    - 输入 embedding 维度 D 必须为偶数（子空间均等分割前提）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np


@dataclass
class ClusteringResult:
    """
    独立子空间聚类的完整输出。

    所有张量字段均为 numpy array，由 DualKnowledgeStore.compact_and_recluster
    按需转换为 torch tensor 后写入 store 状态。

    参数：
        pca_matrix: [D, D] float32，特征向量矩阵 V（np.linalg.eigh 降序排列后），
                    用于 PCA 旋转：x_rot = (x - pca_mean) @ pca_matrix
        pca_mean: [D] float32，训练集均值向量（去中心化用）
        pca_components: [D] float32，对应特征值（降序），诊断用
        row_centroids: [num_keys, D//2] float32，Row 子空间各 cluster 中心，
                       直接传给 ProductKeyMemory.update_keys 作为行 keys
        col_centroids: [num_keys, D//2] float32，Col 子空间各 cluster 中心，
                       直接传给 ProductKeyMemory.update_keys 作为列 keys
        row_labels: [N_valid] int64，每条知识对应的行 cluster ID ∈ [0, num_keys)
        col_labels: [N_valid] int64，每条知识对应的列 cluster ID ∈ [0, num_keys)
        max_cluster_size: 网格（row_label, col_label）中最大条目数（负载均衡诊断）
        mean_cluster_size: 网格平均条目数（= N_valid / num_keys²）
    """

    pca_matrix: np.ndarray  # [D, D] float32
    pca_mean: np.ndarray  # [D] float32
    pca_components: np.ndarray  # [D] float32，特征值降序
    row_centroids: np.ndarray  # [num_keys, D//2] float32
    col_centroids: np.ndarray  # [num_keys, D//2] float32
    row_labels: np.ndarray  # [N_valid] int64
    col_labels: np.ndarray  # [N_valid] int64
    max_cluster_size: int
    mean_cluster_size: float


class SubspaceClustering:
    """
    独立子空间聚类器（纯静态方法类，无实例状态）。

    核心算法：
        1. PCA 旋转：协方差矩阵特征分解，将 embedding 旋转到去相关主成分空间
        2. 交错分割：偶数索引主成分 → Row 子空间，奇数索引 → Col 子空间，保证正交
        3. 轴对齐递归二分：对每个子空间独立聚类，循环遍历维度，均等对半分割，
           保证每个 cluster 大小相差不超过 1

    参考：Reference/Explicit-Lora-router/router/clustering.py（balanced_clustering + perform_clustering）
    """

    @staticmethod
    def fit(
        embeddings: np.ndarray,
        num_keys: int,
    ) -> ClusteringResult:
        """
        全量独立子空间聚类（每次 compact_and_recluster 调用一次）。

        参数：
            embeddings: [N, D] float32 numpy array，来自 AnchorBank.get_embeddings
            num_keys: int，√knowledge_num，必须为 2 的幂次（如 4, 64, 1024）

        返回：
            ClusteringResult，包含 PCA 状态、聚类中心、标签及诊断指标

        异常：
            AssertionError: N == 0 / D 为奇数 / num_keys 不为 2 的幂次
            ValueError: num_keys 不为 2 的幂次（明确异常类型供测试捕获）
        """
        # Phase 1: 前置校验
        assert embeddings.ndim == 2, (
            f"embeddings 必须为 2D 数组 [N, D]，实际 ndim={embeddings.ndim}"
        )
        n, d = embeddings.shape
        assert n > 0, "embeddings 不能为空（N == 0）"
        assert d % 2 == 0, f"embedding 维度 D={d} 必须为偶数（子空间均等分割前提）"
        if num_keys <= 0 or (num_keys & (num_keys - 1)) != 0:
            raise ValueError(
                f"num_keys 必须为正整数且为 2 的幂次，实际 num_keys={num_keys}"
            )

        # Phase 2: PCA 旋转（np.linalg.eigh 对对称正半定矩阵更稳定）
        t_pca = time.time()
        mean = embeddings.mean(axis=0).astype(np.float32)  # [D]
        x_c = embeddings.astype(np.float32) - mean  # [N, D]
        cov = (x_c.T @ x_c) / max(n - 1, 1)  # [D, D]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 升序特征值

        # 降序排列（按方差贡献大小）
        desc_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[desc_idx].astype(np.float32)
        eigenvectors = eigenvectors[:, desc_idx].astype(np.float32)  # [D, D]

        # PCA 旋转到主成分空间
        x_rot = x_c @ eigenvectors  # [N, D]
        print(f"[Clustering] PCA 完成 ({time.time() - t_pca:.1f}s)")

        # Phase 3: 交错分割 → 两个正交子空间
        sub1 = x_rot[:, 0::2]  # [N, D//2] Row 子空间（偶数 PC）
        sub2 = x_rot[:, 1::2]  # [N, D//2] Col 子空间（奇数 PC）

        # Phase 4: 对两个子空间分别进行轴对齐递归二分聚类
        t_bisect = time.time()
        row_labels, row_centroids = SubspaceClustering._recursive_bisect(sub1, num_keys)
        col_labels, col_centroids = SubspaceClustering._recursive_bisect(sub2, num_keys)

        # Phase 5: 计算诊断指标（网格级别负载均衡）
        grid_labels = row_labels * num_keys + col_labels  # [N]
        grid_sizes = np.bincount(grid_labels, minlength=num_keys * num_keys)
        max_cluster_size = int(grid_sizes.max())
        mean_cluster_size = float(grid_sizes.mean())
        print(f"[Clustering] 递归二分完成 ({time.time() - t_bisect:.1f}s), max_cluster={max_cluster_size}, mean_cluster={mean_cluster_size:.1f}")

        return ClusteringResult(
            pca_matrix=eigenvectors,
            pca_mean=mean,
            pca_components=eigenvalues,
            row_centroids=row_centroids,
            col_centroids=col_centroids,
            row_labels=row_labels,
            col_labels=col_labels,
            max_cluster_size=max_cluster_size,
            mean_cluster_size=mean_cluster_size,
        )

    @staticmethod
    def _recursive_bisect(
        data: np.ndarray,
        num_keys: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        轴对齐递归二分聚类（Axis-Aligned Recursive Bisection）。

        算法与参考项目 balanced_clustering 一致：循环遍历维度，每轮按该维度值升序
        排列后均等对半分割，保证所有 cluster 大小相差不超过 1。

        参数：
            data: [N, d] float32 numpy array（某一子空间数据）
            num_keys: int，聚类数，必须为 2 的幂次

        返回：
            labels: [N] int64，每条数据的 cluster ID ∈ [0, num_keys)
            centroids: [num_keys, d] float32，各 cluster 均值中心（空 cluster 填零）

        实现细节：
            - 维度选择：level % d，循环遍历，第 i 轮分割第 (i % d) 维
            - 分割方式：排序后取前半（左），后半（右），确保平衡
            - 空 cluster：仅当 N < num_keys 时出现，中心保持零向量
        """
        n, d = data.shape
        log_k = num_keys.bit_length() - 1  # log2(num_keys) 轮递归

        # 初始状态：一个包含所有索引的组
        current_clusters = [np.arange(n)]

        # Phase 1: 递归二分
        for level in range(log_k):
            dim = level % d  # 循环遍历维度
            next_clusters = []
            for indices in current_clusters:
                if len(indices) == 0:
                    # 空组：左右各一个空组（维持 cluster 总数不变）
                    next_clusters.append(indices)
                    next_clusters.append(indices)
                    continue
                # 按该维度值排序
                values = data[indices, dim]
                sorted_local = np.argsort(values)
                sorted_indices = indices[sorted_local]
                # 均等对半分（奇数长度：左 ⌊N/2⌋，右 ⌈N/2⌉）
                mid = len(sorted_indices) // 2
                next_clusters.append(sorted_indices[:mid])
                next_clusters.append(sorted_indices[mid:])
            current_clusters = next_clusters

        # Phase 2: 分配标签与中心点
        labels = np.zeros(n, dtype=np.int64)
        centroids = np.zeros((num_keys, d), dtype=np.float32)
        for cid, indices in enumerate(current_clusters):
            if len(indices) > 0:
                labels[indices] = cid
                centroids[cid] = data[indices].mean(axis=0)
            # 空 cluster → centroids[cid] 保持零向量

        return labels, centroids

    @staticmethod
    def assign_approximate(
        embeddings: Union[np.ndarray, Any],  # np.ndarray 或 torch.Tensor
        pca_matrix: np.ndarray,
        pca_mean: np.ndarray,
        row_centroids: np.ndarray,
        col_centroids: np.ndarray,
        num_keys: int,
    ) -> Union[int, np.ndarray]:
        """
        高精度近似 cluster 分配（使用真实 embedding 做 PCA 旋转 + 最近邻）。

        与 fit() 的 PCA + 分割逻辑完全一致，复用上次 recluster 保存的状态，
        避免全量重新聚类开销。支持单条 [D] 和批量 [B, D] 输入。

        参数：
            embeddings: [D] 或 [B, D] 的 numpy array 或 torch.Tensor（真实 encoder 编码）
            pca_matrix: [D, D] float32，来自 ClusteringResult.pca_matrix
            pca_mean: [D] float32，来自 ClusteringResult.pca_mean
            row_centroids: [num_keys, D//2] float32，来自 ClusteringResult.row_centroids
            col_centroids: [num_keys, D//2] float32，来自 ClusteringResult.col_centroids
            num_keys: int，= √knowledge_num

        返回：
            单条输入 → int，grid_idx ∈ [0, num_keys²)
            批量输入 → np.ndarray [B] int64，各条目 grid_idx

        关键实现：
            - 与 fit() 的 Step 2-4 完全对应（PCA 旋转 → 交错分割 → 最近邻中心）
            - 距离计算用 L2 平方（避免开根号，不影响 argmin 结果）
        """
        # Phase 1: 统一为 numpy float32
        if hasattr(embeddings, "numpy"):
            emb = embeddings.detach().cpu().float().numpy()
        else:
            emb = np.asarray(embeddings, dtype=np.float32)

        single = emb.ndim == 1
        if single:
            emb = emb[np.newaxis]  # [1, D]

        # Phase 2: PCA 旋转
        x_c = emb - pca_mean.astype(np.float32)  # [B, D]
        x_rot = x_c @ pca_matrix.astype(np.float32)  # [B, D]

        # Phase 3: 交错分割
        sub1 = x_rot[:, 0::2]  # [B, D//2]
        sub2 = x_rot[:, 1::2]  # [B, D//2]

        # Phase 4: 最近邻 cluster 中心（L2 平方距离）
        # row_dist [B, K] = ||sub1[b] - row_centroids[k]||²（广播计算）
        row_centroids_f = row_centroids.astype(np.float32)
        col_centroids_f = col_centroids.astype(np.float32)
        row_dist = np.sum(
            (sub1[:, np.newaxis, :] - row_centroids_f[np.newaxis, :, :]) ** 2, axis=-1
        )  # [B, K]
        col_dist = np.sum(
            (sub2[:, np.newaxis, :] - col_centroids_f[np.newaxis, :, :]) ** 2, axis=-1
        )  # [B, K]

        row_labels = row_dist.argmin(axis=-1).astype(np.int64)  # [B]
        col_labels = col_dist.argmin(axis=-1).astype(np.int64)  # [B]
        result = row_labels * num_keys + col_labels  # [B]

        return int(result[0]) if single else result

    @staticmethod
    def validate_orthogonality(
        row_centroids: np.ndarray,
        col_centroids: np.ndarray,
    ) -> float:
        """
        验证行列聚类中心的正交性（诊断用）。

        计算 Frobenius 范数归一化的正交偏差：
          ||row_centroids @ col_centroids.T||_F / (||row_centroids||_F × ||col_centroids||_F)
        期望值接近 0（严格正交时 = 0）。

        参数：
            row_centroids: [num_keys, D//2] 行子空间中心
            col_centroids: [num_keys, D//2] 列子空间中心

        返回：
            float，正交偏差值（越小越好）
        """
        cross = row_centroids @ col_centroids.T  # [K, K]
        numerator = float(np.linalg.norm(cross, "fro"))
        denom = float(
            np.linalg.norm(row_centroids, "fro") * np.linalg.norm(col_centroids, "fro")
        )
        if denom < 1e-12:
            return 0.0
        return numerator / denom


# 避免循环导入：torch 仅在类型注解中使用，实际运行时不强制导入
try:
    import torch as _torch_check  # noqa: F401
except ImportError:
    pass
