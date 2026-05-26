#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Denoising Dataset Utilities - DataLoader和mask构建工具

Created: 2025-10-22 09:35
Author: Claude Code Assistant
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


def build_masks_from_positions(
    positions,
    indices,
    positive_threshold=25.0,
    negative_threshold=50.0
):
    """
    根据位置信息构建positives_mask和negatives_mask

    Args:
        positions: 所有样本的位置数组 (N, 2) - [northing, easting] in degrees
        indices: 当前batch的索引列表
        positive_threshold: 正样本距离阈值(米)
        negative_threshold: 负样本距离阈值(米)

    Returns:
        positives_mask: (B, B) bool tensor
        negatives_mask: (B, B) bool tensor
    """
    batch_size = len(indices)

    # 获取batch中的位置
    batch_positions = positions[indices]  # (B, 2)

    # 判断坐标系统（NCLT使用UTM坐标，单位已经是米）
    # 如果坐标值很大（如>180），说明是UTM坐标（米），否则是GPS坐标（度）
    max_coord = np.max(np.abs(batch_positions))

    # 计算距离矩阵(米)
    # 扩展维度: (B, 1, 2) - (1, B, 2) = (B, B, 2)
    diff = batch_positions[:, None, :] - batch_positions[None, :, :]  # (B, B, 2)

    if max_coord > 180:
        # UTM坐标，单位已经是米，直接使用
        diff_meters = diff
    else:
        # GPS坐标，需要转换
        LAT_TO_METERS = 111000  # 1度纬度约111km
        LON_TO_METERS = 74000   # 1度经度在德国纬度约74km
        # 转换为米: [northing_diff, easting_diff] * [LAT_TO_METERS, LON_TO_METERS]
        diff_meters = diff * np.array([LAT_TO_METERS, LON_TO_METERS])

    # 计算欧氏距离
    distances = np.linalg.norm(diff_meters, axis=2)  # (B, B) in meters

    # 构建masks
    positives_mask = (distances <= positive_threshold) & (distances > 0)
    negatives_mask = distances >= negative_threshold

    # 转换为tensor
    positives_mask = torch.from_numpy(positives_mask)
    negatives_mask = torch.from_numpy(negatives_mask)

    return positives_mask, negatives_mask


def denoising_collate_fn(batch, dataset, positive_threshold=25.0, negative_threshold=50.0):
    """
    自定义collate函数,用于DenoisingBEVDataset

    Args:
        batch: list of (noisy_img, clean_img, idx, weather)
        dataset: DenoisingBEVDataset实例(用于获取positions)
        positive_threshold: 正样本阈值
        negative_threshold: 负样本阈值

    Returns:
        noisy_imgs: (B, C, H, W) tensor
        clean_imgs: (B, C, H, W) tensor
        positives_mask: (B, B) bool tensor
        negatives_mask: (B, B) bool tensor
        indices: batch索引列表
        weather_list: 天气条件列表
    """
    # 解包batch
    noisy_imgs, clean_imgs, indices, weather_list = zip(*batch)

    # Stack tensors
    noisy_imgs = torch.stack(noisy_imgs)
    clean_imgs = torch.stack(clean_imgs)
    indices = list(indices)

    # 构建masks
    # 支持torch.utils.data.Subset：优先获取底层原始数据集
    base_dataset = getattr(dataset, 'dataset', dataset)
    all_positions = base_dataset.get_all_positions()
    positives_mask, negatives_mask = build_masks_from_positions(
        all_positions,
        indices,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold
    )

    return noisy_imgs, clean_imgs, positives_mask, negatives_mask, indices, weather_list


def make_denoising_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    positive_threshold=25.0,
    negative_threshold=50.0,
    use_batch_sampler=False
):
    """
    创建DenoisingBEVDataset的DataLoader

    Args:
        dataset: DenoisingBEVDataset实例
        batch_size: batch大小
        shuffle: 是否打乱 (当use_batch_sampler=True时忽略)
        num_workers: worker数量
        positive_threshold: 正样本阈值
        negative_threshold: 负样本阈值
        use_batch_sampler: 是否使用BatchSampler (确保batch内有正样本)

    Returns:
        DataLoader
    """
    # 创建collate函数(绑定dataset和thresholds)
    def collate_fn(batch):
        return denoising_collate_fn(
            batch,
            dataset,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold
        )

    if use_batch_sampler:
        # 使用BatchSampler确保batch内有足够的正样本
        from src.datasets.denoising_sampler import DenoisingBatchSampler

        batch_sampler = DenoisingBatchSampler(
            dataset,
            batch_size=batch_size,
            k=2,  # 每组2个正样本
            positive_threshold=positive_threshold
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        # 使用默认的随机采样
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True  # 确保batch大小一致
        )

    return dataloader


def compute_positives_per_query(positives_mask):
    """
    计算每个query有多少个正样本

    Args:
        positives_mask: (B, B) bool tensor

    Returns:
        avg_positives: 平均正样本数
        min_positives: 最少正样本数
        max_positives: 最多正样本数
    """
    num_positives = positives_mask.sum(dim=1).float()  # (B,)

    return {
        'avg': num_positives.mean().item(),
        'min': num_positives.min().item(),
        'max': num_positives.max().item(),
    }


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path

    # 添加路径
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_DIR))

    from src.datasets.denoising_dataset import DenoisingBEVDataset

    print("Testing denoising dataset utilities...")

    # 创建测试数据
    pickle_path = "data/bev_denoising_tuples_448.pkl"

    try:
        dataset = DenoisingBEVDataset(pickle_path, split='train')
        print(f"\nDataset size: {len(dataset)}")

        # 创建DataLoader
        dataloader = make_denoising_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # 测试时使用0
            positive_threshold=25.0
        )

        print(f"DataLoader created: {len(dataloader)} batches")

        # 测试第一个batch
        for batch_idx, batch in enumerate(dataloader):
            noisy_imgs, clean_imgs, pos_mask, neg_mask, indices, weather = batch

            print(f"\nBatch {batch_idx}:")
            print(f"  Noisy imgs: {noisy_imgs.shape}")
            print(f"  Clean imgs: {clean_imgs.shape}")
            print(f"  Positives mask: {pos_mask.shape}")
            print(f"  Negatives mask: {neg_mask.shape}")
            print(f"  Indices: {indices}")
            print(f"  Weather: {weather}")

            # 统计正样本
            pos_stats = compute_positives_per_query(pos_mask)
            print(f"  Positives per query: avg={pos_stats['avg']:.1f}, "
                  f"min={pos_stats['min']}, max={pos_stats['max']}")

            # 只测试第一个batch
            break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")
