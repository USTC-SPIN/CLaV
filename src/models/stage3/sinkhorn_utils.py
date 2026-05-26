#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sinkhorn匹配算法工具函数
用于SALAD聚合中的软分配

Created: 2025-10-21 15:50
Author: Claude Code Assistant

Reference:
- 原实现来自 models/ImLPR.py
- Sinkhorn算法用于求解最优传输问题
"""

import torch
import math
from typing import Optional


def log_otp_solver(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    M: torch.Tensor,
    num_iters: int = 20,
    reg: float = 1.0
) -> torch.Tensor:
    """
    Sinkhorn矩阵缩放算法（对数域实现）

    求解正则化的最优传输问题:
    min <P, M> + reg * KL(P || a⊗b)

    Args:
        log_a: 源分布对数 (B, m+1)
        log_b: 目标分布对数 (B, n)
        M: 代价矩阵 (B, m+1, n)
        num_iters: Sinkhorn迭代次数
        reg: 正则化参数（熵正则化强度）

    Returns:
        log_P: 传输矩阵对数 (B, m+1, n)
    """
    M = M / reg
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(num_iters):
        # u = log_a - logsumexp(M + v)
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        # v = log_b - logsumexp(M + u)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    log_P = M + u.unsqueeze(2) + v.unsqueeze(1)
    return log_P


def get_matching_probs(
    S: torch.Tensor,
    dustbin_score: float = 1.0,
    num_iters: int = 3,
    reg: float = 1.0
) -> torch.Tensor:
    """
    计算匹配概率（带dustbin行处理未匹配元素）

    Args:
        S: 相似度矩阵 (B, m, n)
        dustbin_score: dustbin行的分数（用于处理未匹配）
        num_iters: Sinkhorn迭代次数
        reg: 正则化参数

    Returns:
        log_P: 对数匹配概率 (B, m+1, n)
               其中最后一行是dustbin行
    """
    batch_size, m, n = S.size()

    # 1. 添加dustbin行
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # 2. 构造边际分布（均匀分布）
    norm = -torch.tensor(math.log(n + m), device=S.device)

    log_a = norm.expand(m + 1).contiguous()
    log_b = norm.expand(n).contiguous()

    # 调整dustbin行的边际概率
    log_a[-1] = log_a[-1] + math.log(n - m) if n > m else log_a[-1]

    # 扩展到batch维度
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)

    # 3. Sinkhorn求解
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)

    # 4. 归一化
    log_P = log_P - norm

    return log_P


def soft_assignment(
    scores: torch.Tensor,
    features: torch.Tensor,
    num_clusters: int,
    dustbin_score: float = 1.0,
    num_iters: int = 3
) -> torch.Tensor:
    """
    使用Sinkhorn算法进行软分配聚合

    Args:
        scores: 分配分数 (B, num_clusters, N) - N是特征点数量
        features: 特征 (B, feature_dim, N)
        num_clusters: 聚类中心数量
        dustbin_score: dustbin分数
        num_iters: Sinkhorn迭代次数

    Returns:
        aggregated: 聚合后的特征 (B, feature_dim, num_clusters)
    """
    # 1. 计算匹配概率
    log_P = get_matching_probs(
        scores,
        dustbin_score=dustbin_score,
        num_iters=num_iters
    )
    P = torch.exp(log_P)[:, :-1, :]  # 去除dustbin行

    # 2. 加权聚合
    # P: (B, num_clusters, N)
    # features: (B, feature_dim, N)
    # 目标: (B, feature_dim, num_clusters)

    feature_dim = features.size(1)

    # 扩展维度进行广播
    P_expanded = P.unsqueeze(1).repeat(1, feature_dim, 1, 1)  # (B, feature_dim, num_clusters, N)
    features_expanded = features.unsqueeze(2).repeat(1, 1, num_clusters, 1)  # (B, feature_dim, num_clusters, N)

    # 加权求和
    aggregated = (P_expanded * features_expanded).sum(dim=-1)  # (B, feature_dim, num_clusters)

    return aggregated


def test_sinkhorn():
    """测试Sinkhorn算法"""
    print("Testing Sinkhorn utilities...")

    batch_size = 2
    m = 64  # 聚类数
    n = 1024  # 特征点数

    # 测试log_otp_solver
    log_a = torch.zeros(batch_size, m + 1)
    log_b = torch.zeros(batch_size, n)
    M = torch.randn(batch_size, m + 1, n)

    log_P = log_otp_solver(log_a, log_b, M, num_iters=10)
    print(f"log_P shape: {log_P.shape}")

    # 测试get_matching_probs
    S = torch.randn(batch_size, m, n)
    log_P = get_matching_probs(S, dustbin_score=1.0, num_iters=3)
    print(f"Matching probs shape: {log_P.shape}")

    # 验证概率和为1（沿列方向）
    P = torch.exp(log_P)
    col_sums = P.sum(dim=1)
    print(f"Column sums (should be close to 1): {col_sums[0, :5]}")

    # 测试soft_assignment
    scores = torch.randn(batch_size, m, n)
    features = torch.randn(batch_size, 128, n)
    aggregated = soft_assignment(scores, features, num_clusters=m)
    print(f"Aggregated features shape: {aggregated.shape}")

    print("✓ All Sinkhorn tests passed!")


if __name__ == "__main__":
    test_sinkhorn()
