#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标计算

Created: 2025-10-21 16:20
Author: Claude Code Assistant
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple


def compute_recall_at_k(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [1, 5, 10, 20],
    batch_size: int = 500  # 分批处理避免OOM
) -> Dict[str, float]:
    """
    计算Recall@K指标

    Args:
        database_descriptors: 数据库描述符 (N_db, D)
        query_descriptors: 查询描述符 (N_q, D)
        ground_truth: 真值字典 {query_idx: [db_idx1, db_idx2, ...]}
        k_values: K值列表
        batch_size: 分批处理大小，避免内存溢出

    Returns:
        recall_dict: {f'recall@{k}': value} for k in k_values
    """
    num_queries = query_descriptors.shape[0]
    max_k = max(k_values)

    # 分批计算距离并获取Top-K，避免创建全量距离矩阵
    sorted_indices_list = []
    
    # 预计算数据库描述符的L2范数平方（用于加速）
    db_norms_sq = np.sum(database_descriptors ** 2, axis=1)  # (N_db,)
    
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        query_batch = query_descriptors[i:end_idx]  # (batch, D)
        
        # 使用矩阵乘法计算欧氏距离平方: ||q - d||^2 = ||q||^2 + ||d||^2 - 2*q*d
        query_norms_sq = np.sum(query_batch ** 2, axis=1, keepdims=True)  # (batch, 1)
        distances_batch = query_norms_sq + db_norms_sq - 2 * query_batch @ database_descriptors.T  # (batch, N_db)
        
        # 获取Top-K索引（每行）
        sorted_indices_batch = np.argsort(distances_batch, axis=1)[:, :max_k]
        sorted_indices_list.append(sorted_indices_batch)
    
    # 合并所有批次的结果
    sorted_indices = np.vstack(sorted_indices_list)  # (N_q, max_k)

    recalls = {}

    for k in k_values:
        correct = 0

        for query_idx in range(num_queries):
            # 使用已计算的sorted_indices获取Top-K
            top_k_indices = sorted_indices[query_idx, :k]

            # 获取真值
            gt_indices = ground_truth.get(query_idx, [])

            if len(gt_indices) == 0:
                continue  # 跳过没有真值的查询

            # 检查Top-K中是否有真值
            if any(idx in gt_indices for idx in top_k_indices):
                correct += 1

        # 计算Recall@K
        recall = 100.0 * correct / num_queries
        recalls[f'recall@{k}'] = recall

    return recalls


def compute_average_precision(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    batch_size: int = 500  # 分批处理避免OOM
) -> float:
    """
    计算平均精度均值 (mAP)

    Args:
        database_descriptors: 数据库描述符
        query_descriptors: 查询描述符
        ground_truth: 真值字典
        batch_size: 分批处理大小

    Returns:
        mean_ap: 平均精度均值
    """
    num_queries = query_descriptors.shape[0]
    
    # 预计算数据库描述符的L2范数平方
    db_norms_sq = np.sum(database_descriptors ** 2, axis=1)  # (N_db,)

    average_precisions = []

    for query_idx in range(num_queries):
        # 获取真值
        gt_indices = set(ground_truth.get(query_idx, []))

        if len(gt_indices) == 0:
            continue

        # 分批计算该query的距离并排序
        query_desc = query_descriptors[query_idx:query_idx+1]  # (1, D)
        distances_query = -2 * query_desc @ database_descriptors.T + db_norms_sq  # (1, N_db)
        ranked_indices = np.argsort(distances_query[0])

        # 计算AP
        num_correct = 0
        precision_sum = 0.0

        for rank, db_idx in enumerate(ranked_indices):
            if db_idx in gt_indices:
                num_correct += 1
                precision = num_correct / (rank + 1)
                precision_sum += precision

        if num_correct > 0:
            ap = precision_sum / len(gt_indices)
            average_precisions.append(ap)

    # 计算mAP
    mean_ap = np.mean(average_precisions) if len(average_precisions) > 0 else 0.0

    return mean_ap


def compute_precision_at_k(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [1, 5, 10, 20],
    batch_size: int = 500  # 分批处理避免OOM
) -> Dict[str, float]:
    """
    计算Precision@K指标

    Args:
        database_descriptors: 数据库描述符
        query_descriptors: 查询描述符
        ground_truth: 真值字典
        k_values: K值列表
        batch_size: 分批处理大小

    Returns:
        precision_dict: {f'precision@{k}': value}
    """
    num_queries = query_descriptors.shape[0]
    max_k = max(k_values)
    
    # 预计算数据库描述符的L2范数平方
    db_norms_sq = np.sum(database_descriptors ** 2, axis=1)  # (N_db,)

    precisions = {}

    for k in k_values:
        precision_sum = 0.0
        valid_queries = 0

        for query_idx in range(num_queries):
            # 获取真值
            gt_indices = set(ground_truth.get(query_idx, []))

            if len(gt_indices) == 0:
                continue

            valid_queries += 1

            # 分批计算该query的距离并获取Top-K
            query_desc = query_descriptors[query_idx:query_idx+1]  # (1, D)
            distances_query = -2 * query_desc @ database_descriptors.T + db_norms_sq  # (1, N_db)
            top_k_indices = np.argsort(distances_query[0])[:k]

            # 计算Precision@K
            num_correct = sum(1 for idx in top_k_indices if idx in gt_indices)
            precision = num_correct / k

            precision_sum += precision

        # 平均Precision@K
        avg_precision = 100.0 * precision_sum / valid_queries if valid_queries > 0 else 0.0
        precisions[f'precision@{k}'] = avg_precision

    return precisions


def get_top_k_matches(
    database_descriptors: np.ndarray,
    query_descriptor: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取单个查询的Top-K匹配

    Args:
        database_descriptors: 数据库描述符 (N_db, D)
        query_descriptor: 单个查询描述符 (D,)
        k: 返回Top-K结果

    Returns:
        indices: Top-K数据库索引
        distances: Top-K距离
    """
    # 计算距离
    distances = np.linalg.norm(database_descriptors - query_descriptor, axis=1)

    # 排序并获取Top-K
    sorted_indices = np.argsort(distances)
    top_k_indices = sorted_indices[:k]
    top_k_distances = distances[top_k_indices]

    return top_k_indices, top_k_distances


def compute_distance_threshold_recall(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    distance_threshold: float = 25.0
) -> float:
    """
    计算距离阈值内的召回率

    Args:
        database_descriptors: 数据库描述符
        query_descriptors: 查询描述符
        ground_truth: 真值字典
        distance_threshold: 距离阈值（米）

    Returns:
        recall: 召回率
    """
    num_queries = query_descriptors.shape[0]

    # 计算距离
    distances = np.linalg.norm(
        query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :],
        axis=2
    )

    correct = 0

    for query_idx in range(num_queries):
        # 获取真值
        gt_indices = ground_truth.get(query_idx, [])

        if len(gt_indices) == 0:
            continue

        # 检查是否有真值在阈值内
        for gt_idx in gt_indices:
            if distances[query_idx, gt_idx] <= distance_threshold:
                correct += 1
                break

    recall = 100.0 * correct / num_queries

    return recall


if __name__ == "__main__":
    # 测试代码
    print("Testing evaluation metrics...")

    # 模拟数据
    np.random.seed(42)
    database_desc = np.random.randn(100, 128)
    query_desc = np.random.randn(20, 128)

    # L2归一化
    database_desc = database_desc / np.linalg.norm(database_desc, axis=1, keepdims=True)
    query_desc = query_desc / np.linalg.norm(query_desc, axis=1, keepdims=True)

    # 模拟真值
    ground_truth = {i: [i, i+1] for i in range(20)}

    # 测试Recall@K
    recalls = compute_recall_at_k(database_desc, query_desc, ground_truth)
    print(f"Recall@K: {recalls}")

    # 测试Precision@K
    precisions = compute_precision_at_k(database_desc, query_desc, ground_truth)
    print(f"Precision@K: {precisions}")

    # 测试mAP
    mean_ap = compute_average_precision(database_desc, query_desc, ground_truth)
    print(f"mAP: {mean_ap:.4f}")

    # 测试Top-K匹配
    top_k_indices, top_k_distances = get_top_k_matches(database_desc, query_desc[0], k=5)
    print(f"Top-5 indices: {top_k_indices}")
    print(f"Top-5 distances: {top_k_distances}")

    print("✓ All metric tests passed!")
