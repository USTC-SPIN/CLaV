#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boreas BEV去噪配对数据生成脚本（空间分离版本）
生成用于Stage2扩散模型训练的noisy-clean配对数据

特点：
- 基于GPS坐标匹配配对（真实数据，非仿真）
- Clear天气作为clean reference
- Snow/Rain天气作为noisy query
- 使用距离阈值确保配对质量
- 遵循空间分离原则：训练集在测试区域外，验证集和测试集在测试区域内

创建时间: 2025-11-07
"""

import numpy as np
import os
import pandas as pd
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KDTree

# Boreas BEV数据集配置
DATASET_FOLDER = "boreas-skip-bev"
POSES_FILENAME = "poses.txt"
BEV_SUBFOLDER = "bev_448"  # Boreas BEV图像目录

# 天气条件
CLEAN_WEATHER = 'clear'  # 清晰天气
NOISY_WEATHERS = ['snow', 'rain']  # 噪声天气

# Boreas序列列表（包含天气后缀）
SEQUENCES = [
    'boreas-2020-12-01-13-26-snow',
    'boreas-2021-01-26-11-22-snow',
    'boreas-2021-04-08-12-44-clear',
    'boreas-2021-04-29-15-55-rain',
]

# 测试区域配置（与训练元组生成保持一致）
TEST_REGION_CENTER = [4850082.44, 622826.56]  # [northing, easting]
TEST_REGION_HALF_WIDTH = 410  # 米

# GPS匹配阈值
DISTANCE_THRESHOLD = 5.0  # 米，配对的最大允许距离


def check_in_test_set(northing, easting, center, half_width):
    """
    检查点是否在测试区域内（矩形区域）

    参数:
    - northing: UTM Northing坐标（米）
    - easting: UTM Easting坐标（米）
    - center: 测试区域中心点 [northing, easting]
    - half_width: 矩形半宽（米）

    返回:
    - bool: True if 在测试区域
    """
    center_n, center_e = center
    return (center_n - half_width < northing < center_n + half_width and
            center_e - half_width < easting < center_e + half_width)


def load_poses(base_path, sequence_name):
    """
    加载单个序列的poses数据

    返回:
    - DataFrame with columns: timestamp, easting, northing, ...
    """
    poses_path = Path(base_path) / DATASET_FOLDER / sequence_name / POSES_FILENAME

    if not poses_path.exists():
        return None

    # 读取poses.txt（格式：timestamp easting northing ...）
    data = np.loadtxt(poses_path)

    df = pd.DataFrame({
        'timestamp': data[:, 0].astype(int),
        'easting': data[:, 1],
        'northing': data[:, 2],
    })

    return df.sort_values('timestamp').reset_index(drop=True)


def get_weather_from_sequence(sequence_name):
    """从序列名称提取天气信息"""
    if 'snow' in sequence_name:
        return 'snow'
    elif 'rain' in sequence_name:
        return 'rain'
    elif 'clear' in sequence_name:
        return 'clear'
    else:
        return 'unknown'


def find_nearest_clean_images(noisy_df, clean_data_list, distance_threshold):
    """
    为每个noisy图像找到最近的clean图像

    参数:
    - noisy_df: noisy序列的DataFrame
    - clean_data_list: clean序列的数据列表 [(sequence_name, df), ...]
    - distance_threshold: 最大允许距离（米）

    返回:
    - 配对列表: [(noisy_idx, clean_sequence, clean_idx, distance), ...]
    """
    # 收集所有clean图像的位置
    all_clean_positions = []
    all_clean_metadata = []  # (sequence_name, idx)

    for seq_name, clean_df in clean_data_list:
        for idx, row in clean_df.iterrows():
            all_clean_positions.append([row['northing'], row['easting']])
            all_clean_metadata.append((seq_name, idx))

    if len(all_clean_positions) == 0:
        return []

    all_clean_positions = np.array(all_clean_positions)

    # 构建KDTree
    tree = KDTree(all_clean_positions)

    # 为每个noisy图像找最近的clean图像
    pairs = []
    for noisy_idx, noisy_row in noisy_df.iterrows():
        noisy_pos = np.array([[noisy_row['northing'], noisy_row['easting']]])

        # 查询最近的clean图像
        distances, indices = tree.query(noisy_pos, k=1)
        distance = distances[0][0]
        clean_global_idx = indices[0][0]

        # 检查距离阈值
        if distance <= distance_threshold:
            clean_seq_name, clean_idx = all_clean_metadata[clean_global_idx]
            pairs.append((noisy_idx, clean_seq_name, clean_idx, distance))

    return pairs


def generate_denoising_pairs(base_path, output_path, val_ratio=0.15, distance_threshold=DISTANCE_THRESHOLD):
    """
    生成Boreas BEV去噪配对数据

    数据划分：
    - 训练集: 测试区域外
    - 验证集: 测试区域内（15%）
    - 测试集: 测试区域内（85%，用于评估）

    输出格式:
    {
        'train': [(noisy_path, clean_path, weather), ...],
        'val': [(noisy_path, clean_path, weather), ...],
        'test': [(noisy_path, clean_path, weather), ...],
        'root_dir': str,
        'metadata': {...}
    }
    """
    print(f"\n{'='*70}")
    print(f"生成Boreas BEV去噪配对数据（空间分离版本）")
    print(f"{'='*70}")
    print(f"数据源: {base_path}/{DATASET_FOLDER}")
    print(f"GPS距离阈值: {distance_threshold}m")
    print(f"测试区域: {TEST_REGION_CENTER} ± {TEST_REGION_HALF_WIDTH}m")

    # 加载所有序列的数据
    sequence_data = {}
    for seq_name in SEQUENCES:
        df = load_poses(base_path, seq_name)
        if df is None:
            print(f"⚠ 未找到序列: {seq_name}")
            continue

        weather = get_weather_from_sequence(seq_name)
        sequence_data[seq_name] = {
            'df': df,
            'weather': weather
        }
        print(f"✓ 加载序列: {seq_name} ({weather}, {len(df)} frames)")

    # 分类序列
    clean_sequences = [(name, data['df']) for name, data in sequence_data.items()
                      if data['weather'] == CLEAN_WEATHER]
    noisy_sequences = [(name, data['df'], data['weather']) for name, data in sequence_data.items()
                      if data['weather'] in NOISY_WEATHERS]

    if len(clean_sequences) == 0:
        raise ValueError("未找到clean序列")
    if len(noisy_sequences) == 0:
        raise ValueError("未找到noisy序列")

    print(f"\n清晰序列: {len(clean_sequences)} 个")
    for seq_name, _ in clean_sequences:
        print(f"  - {seq_name}")

    print(f"\n噪声序列: {len(noisy_sequences)} 个")
    for seq_name, _, weather in noisy_sequences:
        print(f"  - {seq_name} ({weather})")

    # 为每个noisy序列找配对
    all_train_pairs = []  # 训练区域外
    all_test_pairs = []   # 测试区域内

    print(f"\n{'='*70}")
    print(f"开始配对...")
    print(f"{'='*70}")

    for noisy_seq_name, noisy_df, weather in noisy_sequences:
        print(f"\n处理: {noisy_seq_name} ({weather})")

        # 找配对
        pairs = find_nearest_clean_images(noisy_df, clean_sequences, distance_threshold)

        if len(pairs) == 0:
            print(f"  ⚠ 未找到匹配的配对")
            continue

        print(f"  找到 {len(pairs)} 个配对")

        # 统计距离
        distances = [p[3] for p in pairs]
        print(f"  距离统计: min={min(distances):.2f}m, max={max(distances):.2f}m, "
              f"mean={np.mean(distances):.2f}m, median={np.median(distances):.2f}m")

        # 创建配对数据
        train_count = 0
        test_count = 0

        noisy_dir = Path(base_path) / DATASET_FOLDER / noisy_seq_name / BEV_SUBFOLDER

        for noisy_idx, clean_seq_name, clean_idx, distance in pairs:
            # 获取noisy图像信息
            noisy_row = noisy_df.iloc[noisy_idx]
            noisy_ts = int(noisy_row['timestamp'])
            noisy_file = f"{noisy_ts}.png"
            noisy_full_path = noisy_dir / noisy_file

            # 获取clean图像信息
            clean_df = next(df for name, df in clean_sequences if name == clean_seq_name)
            clean_row = clean_df.iloc[clean_idx]
            clean_ts = int(clean_row['timestamp'])
            clean_file = f"{clean_ts}.png"

            clean_dir = Path(base_path) / DATASET_FOLDER / clean_seq_name / BEV_SUBFOLDER
            clean_full_path = clean_dir / clean_file

            # 检查文件是否存在
            if not (noisy_full_path.exists() and clean_full_path.exists()):
                continue

            # 相对路径
            noisy_rel_path = str(Path(DATASET_FOLDER) / noisy_seq_name / BEV_SUBFOLDER / noisy_file)
            clean_rel_path = str(Path(DATASET_FOLDER) / clean_seq_name / BEV_SUBFOLDER / clean_file)

            # 构建pair信息
            pair_info = {
                'noisy_path': noisy_rel_path,
                'clean_path': clean_rel_path,
                'weather': weather,
                'noisy_timestamp': noisy_ts,
                'clean_timestamp': clean_ts,
                'noisy_position': [noisy_row['northing'], noisy_row['easting']],
                'clean_position': [clean_row['northing'], clean_row['easting']],
                'distance': distance,
                'noisy_sequence': noisy_seq_name,
                'clean_sequence': clean_seq_name,
            }

            # 根据noisy图像的位置划分：测试区域内/外
            if check_in_test_set(noisy_row['northing'], noisy_row['easting'],
                                TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH):
                all_test_pairs.append(pair_info)
                test_count += 1
            else:
                all_train_pairs.append(pair_info)
                train_count += 1

        print(f"  ✓ train={train_count}, test={test_count}")

    # 打乱数据
    np.random.seed(42)
    np.random.shuffle(all_train_pairs)
    np.random.shuffle(all_test_pairs)

    # 从test区域中分出一部分作为验证集
    num_val = int(len(all_test_pairs) * val_ratio)
    all_val_pairs = all_test_pairs[:num_val]
    all_test_pairs = all_test_pairs[num_val:]

    print(f"\n{'='*70}")
    print(f"空间划分结果:")
    print(f"{'='*70}")
    print(f"  训练集（区域外）: {len(all_train_pairs)} pairs")
    print(f"  验证集（区域内）: {len(all_val_pairs)} pairs ({val_ratio*100:.0f}%)")
    print(f"  测试集（区域内）: {len(all_test_pairs)} pairs ({(1-val_ratio)*100:.0f}%)")

    # 统计天气分布
    print(f"\n天气分布:")
    for split_name, split_data in [('训练集', all_train_pairs),
                                    ('验证集', all_val_pairs),
                                    ('测试集', all_test_pairs)]:
        weather_counts = defaultdict(int)
        for pair in split_data:
            weather_counts[pair['weather']] += 1
        print(f"  {split_name}:")
        for weather in NOISY_WEATHERS:
            count = weather_counts[weather]
            if len(split_data) > 0:
                print(f"    {weather}: {count} ({count/len(split_data)*100:.1f}%)")
            else:
                print(f"    {weather}: 0")

    # 统计距离信息
    print(f"\n配对距离统计:")
    for split_name, split_data in [('训练集', all_train_pairs),
                                    ('验证集', all_val_pairs),
                                    ('测试集', all_test_pairs)]:
        if len(split_data) > 0:
            distances = [p['distance'] for p in split_data]
            print(f"  {split_name}: min={min(distances):.2f}m, max={max(distances):.2f}m, "
                  f"mean={np.mean(distances):.2f}m, median={np.median(distances):.2f}m")

    # 转换为简单元组格式
    train_tuples = [
        (p['noisy_path'], p['clean_path'], p['weather'])
        for p in all_train_pairs
    ]
    val_tuples = [
        (p['noisy_path'], p['clean_path'], p['weather'])
        for p in all_val_pairs
    ]
    test_tuples = [
        (p['noisy_path'], p['clean_path'], p['weather'])
        for p in all_test_pairs
    ]

    # 构建输出数据
    denoising_data = {
        'train': train_tuples,
        'val': val_tuples,
        'test': test_tuples,
        'root_dir': base_path,
        'metadata': {
            'train_pairs_full': all_train_pairs,
            'val_pairs_full': all_val_pairs,
            'test_pairs_full': all_test_pairs,
            'sequences': SEQUENCES,
            'weather_conditions': NOISY_WEATHERS,
            'clean_weather': CLEAN_WEATHER,
            'split_method': 'spatial_rectangular_region',
            'val_ratio': val_ratio,
            'test_region_center': TEST_REGION_CENTER,
            'test_region_half_width': TEST_REGION_HALF_WIDTH,
            'distance_threshold': distance_threshold,
            'total_pairs': len(train_tuples) + len(val_tuples) + len(test_tuples),
            'matching_method': 'GPS_coordinate_nearest_neighbor'
        }
    }

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(denoising_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'='*70}")
    print(f"✓ 去噪配对数据已保存: {output_path}")
    print(f"{'='*70}")
    print(f"数据特性:")
    print(f"  - 训练集: 测试区域外（空间分离）")
    print(f"  - 验证集: 测试区域内 ({val_ratio*100:.0f}%)")
    print(f"  - 测试集: 测试区域内 ({(1-val_ratio)*100:.0f}%)")
    print(f"  - 配对方式: GPS坐标最近邻匹配")
    print(f"  - 距离阈值: {distance_threshold}m")
    print(f"  - 坐标系统: UTM（米）")
    print(f"  - 格式: (noisy_path, clean_path, weather)")

    return denoising_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成Boreas BEV去噪配对数据（空间分离版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/data/users/cxw/pro/clav',
        help='数据集根目录'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='/data/users/cxw/pro/clav/data/boreas_bev_denoising_pairs_spatial.pickle',
        help='输出pickle文件路径'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集占测试区域的比例'
    )

    parser.add_argument(
        '--distance_threshold',
        type=float,
        default=5.0,
        help='GPS配对的最大允许距离（米）'
    )

    args = parser.parse_args()

    # 生成去噪配对数据
    denoising_data = generate_denoising_pairs(
        base_path=args.dataset_root,
        output_path=args.output_file,
        val_ratio=args.val_ratio,
        distance_threshold=args.distance_threshold
    )

    return 0


if __name__ == '__main__':
    exit(main())
