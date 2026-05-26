#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Sampler for Denoising Dataset - 确保batch内有足够的正样本

参考: /home/cxw/pro/ImLPR/datasets/samplers.py
Created: 2025-10-22 12:00
"""

import random
import copy
import numpy as np
from torch.utils.data import Sampler


class ListDict(object):
    """高效的列表+字典数据结构,支持O(1)的add/remove/random_choice"""
    def __init__(self, items=None):
        if items is not None:
            self.items = copy.deepcopy(items)
            self.item_to_position = {item: ndx for ndx, item in enumerate(items)}
        else:
            self.items = []
            self.item_to_position = {}

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class DenoisingBatchSampler(Sampler):
    """
    Batch采样器 - 确保每个batch内有空间邻近的样本(正样本)

    策略:
    - 每次随机选择一个样本
    - 从其正样本中选择k-1个样本
    - 将这k个样本作为一组加入batch
    - 重复直到batch大小达到要求

    这样可以保证batch内有足够的正样本,避免描述符损失NaN

    Args:
        dataset: DenoisingBEVDataset实例
        batch_size: batch大小
        k: 每组正样本数量(默认2)
        positive_threshold: 正样本距离阈值(米)
        max_batches: 最大batch数(用于调试)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        k: int = 2,
        positive_threshold: float = 10.0,
        max_batches: int = None
    ):
        # 支持torch.utils.data.Subset：优先获取底层原始数据集
        self.dataset = getattr(dataset, 'dataset', dataset)
        self._wrapped_dataset = dataset  # 可能为 Subset
        self.batch_size = batch_size
        self.k = k
        self.positive_threshold = positive_threshold
        self.max_batches = max_batches

        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print(f'WARNING: Batch too small. Batch size increased to {self.batch_size}.')

        self.batch_idx = []  # 每个epoch重新生成
        # 注意：若传入的是Subset，这里 elems_ndx 使用分片后的 [0, subset_len)
        self.elems_ndx = list(range(len(self._wrapped_dataset)))

        # 预先计算所有样本的正样本
        print(f'[DenoisingBatchSampler] 预计算正样本...')
        self._is_subset = hasattr(self._wrapped_dataset, 'indices')
        # 基于底层数据集（完整）预计算正样本（使用底层索引）
        base_positives = self._precompute_positives()

        if self._is_subset:
            # 构建 base <-> subset 映射
            subset_to_base = list(self._wrapped_dataset.indices)
            base_to_subset = {b: s for s, b in enumerate(subset_to_base)}

            # 将底层正样本映射到分片子集命名空间，并做过滤
            positives_subset = {}
            for subset_idx, base_idx in enumerate(subset_to_base):
                base_pos_list = base_positives.get(base_idx, [])
                # 仅保留仍落在当前子集的正样本
                mapped = [base_to_subset[b] for b in base_pos_list if b in base_to_subset]
                positives_subset[subset_idx] = mapped

            self.positives = positives_subset
        else:
            # 非子集情形：直接使用底层索引空间
            self.positives = base_positives

        mean_pos = np.mean([len(p) for p in self.positives.values()]) if len(self.positives) > 0 else 0.0
        print(f'  完成! 平均正样本数: {mean_pos:.1f}')

    def _precompute_positives(self):
        """预先计算每个样本的正样本索引（优化版）"""
        from sklearn.neighbors import NearestNeighbors

        positives = {}
        all_positions = self.dataset.get_all_positions()  # (N, 2) 基于底层完整数据集
        n_samples = len(self.dataset)

        print(f'  样本总数: {n_samples}')

        # 获取序列信息
        sequence_ids = self.dataset.get_sequence_ids()
        if sequence_ids is not None:
            print(f'  检测到序列信息，将排除同序列样本')

        # 判断坐标系统（NCLT使用UTM坐标，单位已经是米）
        # 如果坐标值很大（如>180），说明是UTM坐标（米），否则是GPS坐标（度）
        max_coord = np.max(np.abs(all_positions))
        if max_coord > 180:
            # UTM坐标，单位已经是米，直接使用
            print(f'  检测到UTM坐标系统（最大坐标值: {max_coord:.0f}）')
            all_positions_meters = all_positions
        else:
            # GPS坐标，需要转换
            print(f'  检测到GPS坐标系统')
            LAT_TO_METERS = 111000
            LON_TO_METERS = 74000
            all_positions_meters = all_positions * np.array([LAT_TO_METERS, LON_TO_METERS])

        # 使用KD树高效查找邻居（比O(N^2)循环快得多）
        print(f'  构建KD树...')
        nn = NearestNeighbors(radius=self.positive_threshold, algorithm='kd_tree', metric='euclidean')
        nn.fit(all_positions_meters)

        print(f'  查找半径内邻居（阈值={self.positive_threshold}m）...')
        distances, indices = nn.radius_neighbors(all_positions_meters, return_distance=True)

        # 构建正样本字典（排除自己和同序列样本）
        print(f'  构建正样本字典（排除自身和同序列）...')
        for i in range(n_samples):
            # 排除距离为0的样本（自己）
            mask = distances[i] > 0

            # 如果有序列信息，还要排除同序列的样本
            if sequence_ids is not None:
                current_seq = sequence_ids[i]
                if current_seq is not None:
                    neighbor_indices = indices[i][mask]
                    # 进一步过滤：排除同序列样本
                    same_seq_mask = np.array([
                        sequence_ids[idx] != current_seq
                        for idx in neighbor_indices
                    ], dtype=bool)  # 显式指定为bool类型
                    positives[i] = neighbor_indices[same_seq_mask].tolist()
                else:
                    # 当前样本没有序列信息，保留所有邻居
                    positives[i] = indices[i][mask].tolist()
            else:
                # 没有序列信息，只排除自己
                positives[i] = indices[i][mask].tolist()

        return positives

    def __iter__(self):
        # 每个epoch重新生成batches
        self.generate_batches()
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self):
        """生成训练batches"""
        self.batch_idx = []
        unused_elements_ndx = ListDict(self.elems_ndx)
        current_batch = []

        while True:
            # 如果batch满了或没有剩余元素,flush batch
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                if len(current_batch) >= 2 * self.k:
                    # 确保至少有两组正样本(才能找到负样本)
                    assert len(current_batch) % self.k == 0, \
                        f'Incorrect batch size: {len(current_batch)}'
                    self.batch_idx.append(current_batch)
                    current_batch = []

                    if (self.max_batches is not None) and \
                       (len(self.batch_idx) >= self.max_batches):
                        break

                if len(unused_elements_ndx) == 0:
                    break

            # 随机选择一个元素
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)

            # 获取它的正样本
            positives = self.positives.get(selected_element, [])
            if len(positives) == 0:
                # 没有正样本,跳过
                continue

            # 添加selected_element到batch
            current_batch.append(selected_element)

            # 从正样本中选择k-1个元素
            unused_positives = [e for e in positives if e in unused_elements_ndx]

            # 优先从未使用的正样本中选择
            if len(unused_positives) >= (self.k - 1):
                selected_positives = random.sample(unused_positives, self.k - 1)
                for pos in selected_positives:
                    unused_elements_ndx.remove(pos)
            elif len(positives) >= (self.k - 1):
                # 如果未使用的正样本不够,从所有正样本中选择
                selected_positives = random.sample(positives, self.k - 1)
                for pos in selected_positives:
                    if pos in unused_elements_ndx:
                        unused_elements_ndx.remove(pos)
            else:
                # 正样本不够,尽可能多选
                selected_positives = positives[:self.k-1]
                for pos in selected_positives:
                    if pos in unused_elements_ndx:
                        unused_elements_ndx.remove(pos)

            current_batch.extend(selected_positives)

        # 验证所有batch大小
        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, f'Incorrect batch size: {len(batch)}'


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_DIR))

    from src.datasets.denoising_dataset import DenoisingBEVDataset

    print("Testing DenoisingBatchSampler...")

    pickle_path = "data/stage2_denoising_tuples_448.pkl"
    dataset = DenoisingBEVDataset(pickle_path, split='train')

    print(f"\nDataset size: {len(dataset)}")

    # 创建sampler
    sampler = DenoisingBatchSampler(
        dataset,
        batch_size=8,
        k=2,
        positive_threshold=10.0
    )

    print(f"\nGenerated {len(sampler)} batches")

    # 测试前3个batch
    for i, batch_indices in enumerate(sampler):
        if i >= 3:
            break

        print(f"\nBatch {i}:")
        print(f"  Indices: {batch_indices}")
        print(f"  Size: {len(batch_indices)}")

        # 检查batch内的距离
        positions = dataset.get_all_positions()[batch_indices]
        LAT_TO_METERS = 111000
        LON_TO_METERS = 74000
        positions_meters = positions * np.array([LAT_TO_METERS, LON_TO_METERS])

        # 计算距离矩阵
        diff = positions_meters[:, None, :] - positions_meters[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        print(f"  距离统计(米):")
        non_zero_dists = distances[distances > 0]
        if len(non_zero_dists) > 0:
            print(f"    最小: {non_zero_dists.min():.2f}")
            print(f"    最大: {non_zero_dists.max():.2f}")
            print(f"    平均: {non_zero_dists.mean():.2f}")

            # 统计正样本数
            pos_count = (distances <= 10.0) & (distances > 0)
            print(f"    正样本对数: {pos_count.sum()}")

    print("\nDone!")
