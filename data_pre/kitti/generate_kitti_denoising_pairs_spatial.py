#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI BEV去噪配对数据生成脚本（空间分离版本）
生成用于Stage2扩散模型训练的noisy-clean配对数据

使用UTM坐标系统（单位：米）
遵循空间分离原则：
- 训练集：后80%累积距离
- 验证集：前20%累积距离的15%
- 测试集：前20%累积距离的85%

创建时间: 2025-11-06
"""

import numpy as np
import os
import pandas as pd
import pickle
import argparse
from pathlib import Path
from collections import defaultdict


# KITTI BEV数据集配置
DATASET_FOLDER = "kitti-bev-skip"  # 新的数据集文件夹
BEV_SUBFOLDER = "bev_448"  # BEV图像子目录
POSES_FILENAME = "poses.txt"

# 天气条件
CLEAN_WEATHER = 'orin'  # 清晰天气
NOISY_WEATHERS = ['fog', 'rain', 'snow']  # 噪声天气
ALL_WEATHERS = [CLEAN_WEATHER] + NOISY_WEATHERS

# KITTI序列（默认使用6个序列）
DEFAULT_SEQUENCES = [
    "00-10-03-27", "01-10-03-42", "02-10-03-14",
    "04-09-30-16", "08-09-30-28", "09-09-30-33",
    "10-09-30-34"
]

# 空间划分比例
TEST_RATIO = 0.2  # 前20%累积距离作为测试/验证集
TRAIN_RATIO = 0.8  # 后80%累积距离作为训练集


def calculate_cumulative_distance(positions):
    """
    计算UTM轨迹的累积距离（单位：米）

    参数:
    - positions: Nx2数组，[northing, easting] UTM坐标（单位：米）

    返回:
    - cum_dist: N维数组，每个点的累积距离（米）
    """
    # UTM坐标已经是米为单位，直接计算欧氏距离
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))

    # 计算累积距离
    cum_dist = np.zeros(len(positions))
    cum_dist[1:] = np.cumsum(distances)

    return cum_dist


def spatial_split_by_distance(df, test_ratio=0.2):
    """
    按累积距离进行空间分离，返回时间戳阈值

    返回:
    - split_info: {
        'total_distance': float,
        'threshold_distance': float,
        'split_idx': int,
        'split_timestamp': str,  # 分割时间戳
        'total_points': int,
        'timestamps': array,
        'positions': array,
        'cumulative_distances': array
    }
    """
    positions = df[['northing', 'easting']].values
    timestamps = df['timestamp'].values

    cum_dist = calculate_cumulative_distance(positions)
    total_dist = cum_dist[-1]
    threshold_dist = total_dist * test_ratio

    # 找到分割点
    split_idx = np.searchsorted(cum_dist, threshold_dist)

    # 获取分割时间戳
    split_timestamp = df.iloc[split_idx]['timestamp'] if split_idx < len(df) else df.iloc[-1]['timestamp']

    return {
        'total_distance': total_dist,
        'threshold_distance': threshold_dist,
        'split_idx': split_idx,
        'split_timestamp': split_timestamp,  # 添加分割时间戳
        'total_points': len(positions),
        'timestamps': timestamps,
        'positions': positions,
        'cumulative_distances': cum_dist
    }


def load_poses(base_path, seq, weather):
    """
    加载单个序列和天气的poses数据

    返回:
    - DataFrame with columns: timestamp, easting, northing, ...
    """
    seq_weather = f"{seq}-{weather}"
    poses_path = os.path.join(base_path, DATASET_FOLDER, seq_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        return None

    # 读取poses文件
    # 格式: timestamp easting northing z qx qy qz qw ...
    df = pd.read_csv(
        poses_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],  # 只读取需要的列
        names=['timestamp', 'easting', 'northing']
    )

    # 转换timestamp为字符串格式（10位数字）
    df['timestamp'] = df['timestamp'].astype(str).str.zfill(10)

    return df.sort_index().reset_index(drop=True)


def get_image_files(base_path, seq, weather):
    """
    获取指定序列和天气的图像文件列表
    """
    seq_weather = f"{seq}-{weather}"
    img_dir = os.path.join(base_path, DATASET_FOLDER, seq_weather, BEV_SUBFOLDER)

    if not os.path.exists(img_dir):
        return set()

    # 获取所有PNG文件的时间戳
    img_files = set()
    for f in os.listdir(img_dir):
        if f.endswith('.png'):
            ts = os.path.splitext(f)[0]
            img_files.add(ts)

    return img_files


def generate_denoising_pairs(base_path, sequences=None, val_ratio=0.15):
    """
    生成KITTI BEV去噪配对数据

    数据划分：
    - 训练集: 后80%累积距离
    - 验证集: 前20%累积距离的15%
    - 测试集: 前20%累积距离的85%

    输出格式:
    {
        'train': [(noisy_path, clean_path, weather), ...],
        'val': [(noisy_path, clean_path, weather), ...],
        'test': [(noisy_path, clean_path, weather), ...],
        'root_dir': str,
        'metadata': {...}
    }
    """
    if sequences is None:
        sequences = DEFAULT_SEQUENCES

    print(f"\n{'='*70}")
    print(f"生成KITTI BEV去噪配对数据（空间分离版本 - UTM坐标）")
    print(f"{'='*70}")
    print(f"数据集根目录: {base_path}")
    print(f"处理序列: {sequences}")
    print(f"坐标系统: UTM（单位：米）")

    all_train_pairs = []  # 后80%累积距离
    all_test_pairs = []   # 前20%累积距离

    for seq in sequences:
        print(f"\n处理序列: {seq}")

        # 加载clean (orin) poses
        clean_df = load_poses(base_path, seq, CLEAN_WEATHER)
        if clean_df is None:
            print(f"  ⚠ 未找到{CLEAN_WEATHER} poses")
            continue

        # 空间分离
        split_info = spatial_split_by_distance(clean_df, TEST_RATIO)
        print(f"  总距离: {split_info['total_distance']:.2f}m")
        print(f"  分割阈值: {split_info['threshold_distance']:.2f}m")
        print(f"  分割点: {split_info['split_idx']}/{split_info['total_points']}")
        print(f"  分割时间戳: {split_info['split_timestamp']}")

        # 获取clean图像文件
        clean_imgs = get_image_files(base_path, seq, CLEAN_WEATHER)
        if len(clean_imgs) == 0:
            print(f"  ⚠ 未找到{CLEAN_WEATHER}图像")
            continue

        # 创建timestamp到位置的映射
        clean_ts_to_pos = {
            row['timestamp']: (row['northing'], row['easting'])
            for _, row in clean_df.iterrows()
        }

        # 基于时间戳划分训练集和测试集
        split_timestamp = split_info['split_timestamp']
        test_timestamps = set(clean_df[clean_df['timestamp'] < split_timestamp]['timestamp'])
        train_timestamps = set(clean_df[clean_df['timestamp'] >= split_timestamp]['timestamp'])

        # 为每种噪声天气创建配对
        for weather in NOISY_WEATHERS:
            # 获取noisy图像文件
            noisy_imgs = get_image_files(base_path, seq, weather)
            if len(noisy_imgs) == 0:
                print(f"  ⚠ 未找到{weather}图像")
                continue

            # 找到共同的时间戳（图像必须都存在）
            common_timestamps = clean_imgs & noisy_imgs

            if len(common_timestamps) == 0:
                print(f"  ⚠ {weather}: 无匹配的图像")
                continue

            train_count = 0
            test_count = 0

            for ts in common_timestamps:
                # 构建文件路径
                clean_file = f"{ts}.png"
                noisy_file = f"{ts}.png"

                # 相对路径
                clean_rel_path = os.path.join(
                    DATASET_FOLDER, f"{seq}-{CLEAN_WEATHER}",
                    BEV_SUBFOLDER, clean_file
                )
                noisy_rel_path = os.path.join(
                    DATASET_FOLDER, f"{seq}-{weather}",
                    BEV_SUBFOLDER, noisy_file
                )

                # 获取位置
                if ts not in clean_ts_to_pos:
                    continue

                northing, easting = clean_ts_to_pos[ts]

                # 构建pair信息
                pair_info = {
                    'noisy_path': noisy_rel_path,
                    'clean_path': clean_rel_path,
                    'weather': weather,
                    'timestamp': int(ts),
                    'position': [northing, easting],  # UTM坐标（米）
                    'sequence': seq,
                }

                # 根据时间戳划分数据集
                if ts in test_timestamps:
                    all_test_pairs.append(pair_info)
                    test_count += 1
                elif ts in train_timestamps:
                    all_train_pairs.append(pair_info)
                    train_count += 1

            print(f"  ✓ {weather}: train={train_count}, test={test_count}")

    # 打乱数据
    np.random.shuffle(all_train_pairs)
    np.random.shuffle(all_test_pairs)

    # 从test区域中分出一部分作为验证集
    num_val = int(len(all_test_pairs) * val_ratio)
    all_val_pairs = all_test_pairs[:num_val]
    all_test_pairs = all_test_pairs[num_val:]

    print(f"\n空间划分结果:")
    print(f"  训练集（后80%）: {len(all_train_pairs)} pairs")
    print(f"  验证集（前20%的{val_ratio*100:.0f}%）: {len(all_val_pairs)} pairs")
    print(f"  测试集（前20%的{(1-val_ratio)*100:.0f}%）: {len(all_test_pairs)} pairs")

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
            print(f"    {weather}: {weather_counts[weather]}")

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
        'root_dir': os.path.join(base_path, DATASET_FOLDER),
        'metadata': {
            'train_pairs_full': all_train_pairs,
            'val_pairs_full': all_val_pairs,
            'test_pairs_full': all_test_pairs,
            'sequences': sequences,
            'weather_conditions': NOISY_WEATHERS,
            'split_method': 'spatial_cumulative_distance',
            'test_ratio': TEST_RATIO,
            'val_ratio': val_ratio,
            'coordinate_system': 'UTM',
            'total_pairs': len(train_tuples) + len(val_tuples) + len(test_tuples),
        }
    }

    print(f"\n总计配对数:")
    print(f"  总数: {denoising_data['metadata']['total_pairs']}")

    # 统计序列分布
    print(f"\n序列分布:")
    for split_name, split_data in [('训练集', all_train_pairs),
                                    ('验证集', all_val_pairs),
                                    ('测试集', all_test_pairs)]:
        seq_counts = defaultdict(int)
        for pair in split_data:
            seq_counts[pair['sequence']] += 1
        print(f"  {split_name}:")
        for seq in sorted(seq_counts.keys()):
            print(f"    {seq}: {seq_counts[seq]}")

    return denoising_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成KITTI BEV去噪配对数据（空间分离版本 - UTM坐标）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/data/users/cxw/pro/clav',
        help='数据集根目录'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（默认: dataset_root/data/kitti_denoising_tuples.pkl）'
    )

    parser.add_argument(
        '--sequences',
        nargs='+',
        default=None,
        help='要处理的序列列表（默认使用所有序列）'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例（从测试区域中分出）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 确定序列
    sequences = args.sequences if args.sequences else DEFAULT_SEQUENCES

    # 生成去噪配对数据
    denoising_data = generate_denoising_pairs(
        args.dataset_root,
        sequences,
        args.val_ratio
    )

    # 确定输出路径
    if args.output is None:
        output_path = os.path.join(args.dataset_root, 'data', 'kitti_denoising_tuples.pkl')
    else:
        output_path = args.output

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(denoising_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'='*70}")
    print(f"✓ 去噪配对数据已保存: {output_path}")
    print(f"{'='*70}")
    print(f"数据特性:")
    print(f"  - 训练集: 后80%累积距离（空间分离）")
    print(f"  - 验证集: 前20%累积距离的{args.val_ratio*100:.0f}%")
    print(f"  - 测试集: 前20%累积距离的{(1-args.val_ratio)*100:.0f}%")
    print(f"  - 配对方式: 精确时间戳匹配")
    print(f"  - 坐标系统: UTM（米）")
    print(f"  - 格式: (noisy_path, clean_path, weather)")


if __name__ == "__main__":
    main()