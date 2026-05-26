#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
潜在空间描述符提取头
基于SALAD聚合机制，直接作用于RAE去噪后的潜在表示

Created: 2025-10-21 15:55
Author: Claude Code Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sinkhorn_utils import get_matching_probs


class LatentDescriptorHead(nn.Module):
    """
    潜在空间描述符提取头

    设计理念:
    1. 输入: RAE去噪后的潜在表示 (B, 768, 32, 32)
    2. 输出: 全局描述符 (B, 8448)
    3. 机制: SALAD聚合 (Sinkhorn Aggregation of Local Descriptors)

    组件:
    - token_features: 全局token提取（通过空间平均池化）
    - cluster_features: 局部聚类特征提取
    - score: Sinkhorn匹配分数计算
    - SALAD聚合: 软分配+加权求和

    输出维度计算:
    - global_token: token_dim (256)
    - clustered_features: cluster_dim × num_clusters (128 × 64 = 8192)
    - total: 256 + 8192 = 8448维
    """

    def __init__(
        self,
        latent_dim: int = 768,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        use_global_token: bool = True,
        output_dim: int = 8448
    ):
        """
        Args:
            latent_dim: 输入潜在表示维度 (DINOv2-Base = 768)
            num_clusters: 聚类中心数量
            cluster_dim: 每个聚类的特征维度
            token_dim: 全局token维度
            use_global_token: 是否使用全局token
            output_dim: 输出描述符维度（用于验证）
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.use_global_token = use_global_token

        # ===== 1. 全局Token提取 =====
        if use_global_token:
            self.token_features = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # (B, 768, 32, 32) → (B, 768, 1, 1)
                nn.Flatten(),             # (B, 768, 1, 1) → (B, 768)
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, token_dim)  # (B, 768) → (B, 256)
            )

        # ===== 2. 聚类特征提取 =====
        self.cluster_features = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, kernel_size=1)  # (B, 768, 32, 32) → (B, 128, 32, 32)
        )

        # ===== 3. Sinkhorn匹配分数 =====
        self.score = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, num_clusters, kernel_size=1)  # (B, 768, 32, 32) → (B, 64, 32, 32)
        )

        # ===== 4. Dustbin参数 =====
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        # 验证输出维度
        expected_dim = token_dim + cluster_dim * num_clusters if use_global_token else cluster_dim * num_clusters
        assert output_dim == expected_dim, f"Output dim mismatch: expected {expected_dim}, got {output_dim}"
        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            z: 去噪后的潜在表示 (B, 768, 32, 32)

        Returns:
            descriptor: 全局描述符 (B, 8448), L2归一化
        """
        batch_size = z.size(0)

        # ===== 1. 提取全局Token（如果使用）=====
        if self.use_global_token:
            t = self.token_features(z)  # (B, 256)
            t = F.normalize(t, p=2, dim=-1)  # L2归一化

        # ===== 2. 提取聚类特征 =====
        f = self.cluster_features(z)  # (B, 128, 32, 32)
        f = f.flatten(2)  # (B, 128, 1024)

        # ===== 3. 计算匹配分数 =====
        p = self.score(z)  # (B, 64, 32, 32)
        p = p.flatten(2)  # (B, 64, 1024)

        # ===== 4. Sinkhorn匹配 =====
        # p: (B, 64, 1024) - 64个聚类中心对1024个空间位置的分数
        p = get_matching_probs(p, self.dust_bin, num_iters=3)
        p = torch.exp(p)[:, :-1, :]  # 去除dustbin行, (B, 64, 1024)

        # ===== 5. 软分配聚合 =====
        # p: (B, 64, 1024)
        # f: (B, 128, 1024)
        # 目标: (B, 128, 64) → 每个聚类中心的聚合特征

        p_expanded = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # (B, 128, 64, 1024)
        f_expanded = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # (B, 128, 64, 1024)

        # 加权求和
        aggregated = (f_expanded * p_expanded).sum(dim=-1)  # (B, 128, 64)

        # L2归一化（沿cluster_dim维度）
        aggregated = F.normalize(aggregated, p=2, dim=1)  # (B, 128, 64)

        # 展平
        aggregated = aggregated.flatten(1)  # (B, 8192)

        # ===== 6. 拼接全局Token（如果使用）=====
        if self.use_global_token:
            descriptor = torch.cat([t, aggregated], dim=-1)  # (B, 256+8192=8448)
        else:
            descriptor = aggregated  # (B, 8192)

        # ===== 7. 最终L2归一化 =====
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        return descriptor

    def get_output_dim(self) -> int:
        """返回输出描述符维度"""
        return self.output_dim

    def extra_repr(self) -> str:
        """打印额外信息"""
        return (
            f"latent_dim={self.latent_dim}, num_clusters={self.num_clusters}, "
            f"cluster_dim={self.cluster_dim}, token_dim={self.token_dim}, "
            f"output_dim={self.output_dim}"
        )


def test_latent_descriptor_head():
    """测试潜在空间描述符头"""
    print("Testing LatentDescriptorHead...")

    # 创建模型
    model = LatentDescriptorHead(
        latent_dim=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        use_global_token=True,
        output_dim=8448
    )

    print(f"Model:\n{model}")
    print(f"Output dim: {model.get_output_dim()}")

    # 测试前向传播
    batch_size = 2
    z = torch.randn(batch_size, 768, 32, 32)

    descriptor = model(z)

    print(f"\nInput shape: {z.shape}")
    print(f"Output shape: {descriptor.shape}")
    print(f"Output dim matches: {descriptor.shape[1] == 8448}")

    # 验证L2归一化
    norms = torch.norm(descriptor, p=2, dim=1)
    print(f"L2 norms (should be close to 1): {norms}")

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 normalization failed!"

    # 测试不使用全局token的情况
    model_no_token = LatentDescriptorHead(
        latent_dim=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        use_global_token=False,
        output_dim=8192  # 128 * 64
    )

    descriptor_no_token = model_no_token(z)
    print(f"\nWithout global token:")
    print(f"Output shape: {descriptor_no_token.shape}")
    print(f"Output dim matches: {descriptor_no_token.shape[1] == 8192}")

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✓ All LatentDescriptorHead tests passed!")


if __name__ == "__main__":
    test_latent_descriptor_head()
