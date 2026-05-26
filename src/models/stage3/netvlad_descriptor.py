#!/usr/bin/env python3
"""
NetVLAD Descriptor Head for Latent Space
Created: 2025-11-10 16:25
Purpose: 实现标准NetVLAD描述符提取头，用于BEV潜在表示

NetVLAD (Vector of Locally Aggregated Descriptors) 是一种可微分的VLAD实现，
广泛应用于图像检索和地理定位任务。

核心公式:
    V(j, k) = Σ_i a_k(x_i) * (x_i - c_k)

其中:
- x_i: 第i个局部特征
- c_k: 第k个聚类中心（可学习参数）
- a_k(x_i): 软分配权重（通过softmax计算）

关键特性:
1. 软分配：使用softmax而非硬分配
2. 残差聚合：计算特征与聚类中心的残差
3. Intra-normalization：每个cluster独立L2归一化
4. Ghost clusters处理：避免空聚类

References:
- Arandjelovic et al., "NetVLAD: CNN architecture for weakly supervised place recognition", CVPR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NetVLADDescriptor(nn.Module):
    """
    NetVLAD描述符提取头

    用于从去噪后的潜在表示提取全局描述符，适用于图像检索和地理定位。

    输入: 潜在表示 (B, latent_dim, H, W)
    输出: 全局描述符 (B, num_clusters * latent_dim), L2归一化

    参数:
        latent_dim: 输入潜在表示的通道维度（如DINOv2-B=768）
        num_clusters: VLAD聚类中心数量（通常64或128）
        normalize_input: 是否对输入特征进行L2归一化
        use_intra_norm: 是否使用intra-normalization（推荐）
        add_batch_norm: 是否在卷积后添加BatchNorm
        ghost_cluster_threshold: Ghost cluster处理阈值
    """

    def __init__(
        self,
        latent_dim: int = 768,
        num_clusters: int = 64,
        normalize_input: bool = True,
        use_intra_norm: bool = True,
        add_batch_norm: bool = False,
        ghost_cluster_threshold: float = 1e-6,
        output_dim: Optional[int] = None
    ):
        """
        初始化NetVLAD描述头

        Args:
            latent_dim: 输入特征维度
            num_clusters: 聚类中心数量
            normalize_input: 是否归一化输入
            use_intra_norm: 是否使用intra-normalization
            add_batch_norm: 是否使用BatchNorm
            ghost_cluster_threshold: Ghost cluster阈值
            output_dim: 输出维度（用于验证，应等于latent_dim * num_clusters）
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.use_intra_norm = use_intra_norm
        self.ghost_cluster_threshold = ghost_cluster_threshold

        # 输出维度验证
        expected_dim = latent_dim * num_clusters
        if output_dim is not None:
            assert output_dim == expected_dim, \
                f"Output dim mismatch: expected {expected_dim}, got {output_dim}"
        self.output_dim = expected_dim

        # 聚类中心（可学习参数）
        # shape: (num_clusters, latent_dim)
        self.centroids = nn.Parameter(
            torch.randn(num_clusters, latent_dim)
        )

        # 软分配卷积层
        # 输入: (B, latent_dim, H, W)
        # 输出: (B, num_clusters, H, W)
        self.conv = nn.Conv2d(
            latent_dim, num_clusters,
            kernel_size=1,
            bias=True
        )

        # 可选的BatchNorm
        if add_batch_norm:
            self.bn = nn.BatchNorm2d(num_clusters)
        else:
            self.bn = None

        # 初始化参数
        self._init_params()

    def _init_params(self):
        """初始化网络参数"""
        # 聚类中心：Xavier均匀初始化
        nn.init.xavier_uniform_(self.centroids)

        # 卷积层：Xavier初始化
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 潜在表示 (B, latent_dim, H, W)
               如DINOv2-B: (B, 768, 32, 32)

        Returns:
            vlad: NetVLAD描述符 (B, latent_dim * num_clusters)
                 如: (B, 49152) for 768×64
                 L2归一化
        """
        B, C, H, W = x.shape
        N = H * W  # 空间位置数量

        assert C == self.latent_dim, \
            f"Input channel mismatch: expected {self.latent_dim}, got {C}"

        # ===== 1. 可选的输入归一化 =====
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize along channel dim

        # ===== 2. 计算软分配权重 =====
        # 通过1x1卷积计算每个位置对每个cluster的分配分数
        soft_assign = self.conv(x)  # (B, K, H, W)

        # 可选的BatchNorm
        if self.bn is not None:
            soft_assign = self.bn(soft_assign)

        # Softmax归一化（沿cluster维度）
        # 确保每个空间位置的权重和为1
        soft_assign = F.softmax(soft_assign, dim=1)  # (B, K, H, W)

        # ===== 3. 展平空间维度 =====
        x_flatten = x.view(B, C, -1)  # (B, C, N)
        soft_assign_flatten = soft_assign.view(B, self.num_clusters, -1)  # (B, K, N)

        # ===== 4. VLAD残差聚合 =====
        # 对每个cluster k:
        #   V_k = Σ_i a_k(x_i) * (x_i - c_k)

        vlad = torch.zeros(
            [B, self.num_clusters, C],
            dtype=x.dtype,
            device=x.device
        )

        for k in range(self.num_clusters):
            # 获取cluster k的中心
            centroid_k = self.centroids[k:k+1].T  # (C, 1)

            # 计算残差: x_i - c_k
            # x_flatten: (B, C, N)
            # centroid_k: (C, 1) -> broadcast to (B, C, N)
            residual = x_flatten - centroid_k.unsqueeze(0)  # (B, C, N)

            # 加权残差: a_k(x_i) * (x_i - c_k)
            # soft_assign_flatten[:, k:k+1, :]: (B, 1, N)
            weighted_residual = residual * soft_assign_flatten[:, k:k+1, :]  # (B, C, N)

            # 求和: Σ_i a_k(x_i) * (x_i - c_k)
            vlad[:, k, :] = weighted_residual.sum(dim=2)  # (B, C)

        # ===== 5. Intra-normalization =====
        # 对每个cluster的特征向量进行L2归一化
        if self.use_intra_norm:
            vlad = F.normalize(vlad, p=2, dim=2)  # (B, K, C)

            # Ghost cluster处理
            # 如果某个cluster的norm接近0，说明没有特征被分配到它
            # 添加小扰动避免数值问题
            vlad = self._handle_ghost_clusters(vlad)

        # ===== 6. 展平并最终归一化 =====
        vlad = vlad.view(B, -1)  # (B, K*C)

        # 最终L2归一化
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad

    def _handle_ghost_clusters(self, vlad: torch.Tensor) -> torch.Tensor:
        """
        处理ghost clusters（空聚类）

        Ghost clusters是指没有或很少有特征被分配到的聚类。
        这会导致归一化后的向量接近零向量，影响特征表达能力。

        处理方法：为接近零的cluster添加小扰动

        Args:
            vlad: (B, K, C) VLAD特征

        Returns:
            vlad: 处理后的VLAD特征
        """
        # 计算每个cluster的L2 norm
        cluster_norms = torch.norm(vlad, p=2, dim=2, keepdim=True)  # (B, K, 1)

        # 找到norm接近0的cluster（ghost clusters）
        ghost_mask = cluster_norms < self.ghost_cluster_threshold

        # 为ghost clusters添加小的随机扰动
        if ghost_mask.any():
            perturbation = torch.randn_like(vlad) * 1e-3
            vlad = torch.where(
                ghost_mask.expand_as(vlad),
                perturbation,
                vlad
            )

        return vlad

    def get_output_dim(self) -> int:
        """返回输出描述符维度"""
        return self.output_dim

    def get_cluster_assignments(
        self,
        x: torch.Tensor,
        return_soft: bool = True
    ) -> torch.Tensor:
        """
        获取cluster分配（用于可视化和分析）

        Args:
            x: 潜在表示 (B, C, H, W)
            return_soft: 是否返回软分配（softmax）或硬分配（argmax）

        Returns:
            assignments:
                - 如果return_soft=True: (B, K, H, W) 软分配概率
                - 如果return_soft=False: (B, H, W) 硬分配索引
        """
        # 计算软分配
        soft_assign = self.conv(x)  # (B, K, H, W)

        if self.bn is not None:
            soft_assign = self.bn(soft_assign)

        soft_assign = F.softmax(soft_assign, dim=1)  # (B, K, H, W)

        if return_soft:
            return soft_assign
        else:
            # 硬分配：argmax
            hard_assign = torch.argmax(soft_assign, dim=1)  # (B, H, W)
            return hard_assign

    def visualize_cluster_activations(
        self,
        x: torch.Tensor,
        cluster_idx: int
    ) -> torch.Tensor:
        """
        可视化特定cluster的激活图

        Args:
            x: 潜在表示 (B, C, H, W)
            cluster_idx: 要可视化的cluster索引

        Returns:
            activation_map: (B, H, W) cluster激活图
        """
        assert 0 <= cluster_idx < self.num_clusters, \
            f"Invalid cluster index: {cluster_idx}"

        soft_assign = self.get_cluster_assignments(x, return_soft=True)  # (B, K, H, W)
        activation_map = soft_assign[:, cluster_idx, :, :]  # (B, H, W)

        return activation_map


# 工厂函数：创建标准NetVLAD实例
def create_netvlad_descriptor(
    latent_dim: int = 768,
    num_clusters: int = 64,
    **kwargs
) -> NetVLADDescriptor:
    """
    创建标准NetVLAD描述符实例

    Args:
        latent_dim: 输入特征维度（如DINOv2-B=768）
        num_clusters: 聚类数量
        **kwargs: 其他参数

    Returns:
        NetVLADDescriptor实例
    """
    return NetVLADDescriptor(
        latent_dim=latent_dim,
        num_clusters=num_clusters,
        normalize_input=True,
        use_intra_norm=True,
        add_batch_norm=False,
        **kwargs
    )
