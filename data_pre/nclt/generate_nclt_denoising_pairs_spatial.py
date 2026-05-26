#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCLT BEV去噪配对数据生成脚本（空间分离版本）
生成用于Stage2扩散模型训练的noisy-clean配对数据

遵循空间分离原则：
- 训练集：测试区域外
- 验证集：测试区域内（但不用于评估）
- 测试集：测试区域内（用于评估）

创建时间: 2025-11-05
"""

import numpy as np
import os
import pandas as pd
import pickle
import argparse
from pathlib import Path
from collections import defaultdict


# NCLT BEV数据集配置
DATASET_FOLDER = "nclt-bev-12-frame-skip"
POSES_FILENAME = "poses.txt"

# 天气条件
CLEAN_WEATHER = 'orin'  # 清晰天气
NOISY_WEATHERS = ['fog', 'rain', 'snow']  # 噪声天气

# NCLT数据集日期
ALL_DATES = [
    '2012-01-08', '2012-01-15', '2012-01-22', '2012-08-20',
    '2012-09-28', '2012-10-28', '2012-11-04', '2012-11-16',
    '2012-11-17', '2012-12-01', '2013-02-23', '2013-04-05'
]

# 矩形测试区域参数
X_WIDTH = 75  # 米（半宽）
Y_WIDTH = 75  # 米（半高）

# 测试区域中心点（UTM坐标，米）
P1 = [4685682.634858, 276089.034153]
P2 = [4685707.728925, 276309.850477]
P3 = [4685905.456411, 276116.466847]
P_TEST_REGIONS = [P1, P2, P3]


def check_in_test_set(northing, easting, test_regions):
    """
    检查点是否在测试区域内

    参数:
    - northing: UTM Northing坐标（米）
    - easting: UTM Easting坐标（米）
    - test_regions: 测试区域中心点列表

    返回:
    - bool: True if 在测试区域
    """
    for center_n, center_e in test_regions:
        if (center_n - Y_WIDTH < northing < center_n + Y_WIDTH and
            center_e - X_WIDTH < easting < center_e + X_WIDTH):
            return True
    return False


def load_poses(base_path, date, weather):
    """
    加载单个日期和天气的poses数据

    返回:
    - DataFrame with columns: timestamp, easting, northing, z, qx, qy, qz, qw
    """
    date_weather = f"{date}-{weather}"
    poses_path = os.path.join(base_path, DATASET_FOLDER, date_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        return None

    df = pd.read_csv(
        poses_path,
        sep=r"\s+",
        header=None,
        names=['timestamp', 'easting', 'northing', 'z', 'qx', 'qy', 'qz', 'qw']
    )

    return df.sort_values('timestamp').reset_index(drop=True)


def generate_denoising_pairs(base_path, output_path, val_ratio=0.15):
    """
    生成NCLT BEV去噪配对数据

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
    print(f"生成NCLT BEV去噪配对数据（空间分离版本）")
    print(f"{'='*70}")

    all_train_pairs = []  # 训练区域外
    all_test_pairs = []   # 测试区域内

    for date in ALL_DATES:
        print(f"\n处理日期: {date}")

        # 加载clean (orin) poses
        clean_df = load_poses(base_path, date, CLEAN_WEATHER)
        if clean_df is None:
            print(f"  ⚠ 未找到orin poses")
            continue

        # 检查clean图像是否存在
        clean_dir = os.path.join(base_path, DATASET_FOLDER, f"{date}-{CLEAN_WEATHER}")
        if not os.path.exists(clean_dir):
            print(f"  ⚠ Clean图像目录不存在")
            continue

        # 为每种噪声天气创建配对
        for weather in NOISY_WEATHERS:
            noisy_df = load_poses(base_path, date, weather)
            if noisy_df is None:
                print(f"  ⚠ 未找到{weather} poses")
                continue

            noisy_dir = os.path.join(base_path, DATASET_FOLDER, f"{date}-{weather}")
            if not os.path.exists(noisy_dir):
                print(f"  ⚠ {weather}图像目录不存在")
                continue

            # 找到共同时间戳（精确匹配）
            clean_timestamps = set(clean_df['timestamp'].astype(str))
            noisy_timestamps = set(noisy_df['timestamp'].astype(str))
            common_timestamps = clean_timestamps & noisy_timestamps

            if len(common_timestamps) == 0:
                print(f"  ⚠ {weather}: 无匹配时间戳")
                continue

            # 创建timestamp到位置的映射
            clean_ts_to_pos = {
                str(int(row['timestamp'])): (row['northing'], row['easting'])
                for _, row in clean_df.iterrows()
            }

            # 创建配对
            train_count = 0
            test_count = 0

            for ts in common_timestamps:
                # 检查图像文件是否存在
                clean_file = f"{ts}.png"
                noisy_file = f"{ts}.png"

                clean_full_path = os.path.join(clean_dir, clean_file)
                noisy_full_path = os.path.join(noisy_dir, noisy_file)

                if not (os.path.exists(clean_full_path) and os.path.exists(noisy_full_path)):
                    continue

                # 相对路径
                clean_rel_path = os.path.join(DATASET_FOLDER, f"{date}-{CLEAN_WEATHER}", clean_file)
                noisy_rel_path = os.path.join(DATASET_FOLDER, f"{date}-{weather}", noisy_file)

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
                    'position': [northing, easting],
                    'date': date,
                }

                # 根据位置划分：测试区域内/外
                if check_in_test_set(northing, easting, P_TEST_REGIONS):
                    all_test_pairs.append(pair_info)
                    test_count += 1
                else:
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
        'root_dir': base_path,
        'metadata': {
            'train_pairs_full': all_train_pairs,
            'val_pairs_full': all_val_pairs,
            'test_pairs_full': all_test_pairs,
            'dates': ALL_DATES,
            'weather_conditions': NOISY_WEATHERS,
            'split_method': 'spatial_rectangular_regions',
            'val_ratio': val_ratio,
            'test_regions': P_TEST_REGIONS,
            'total_pairs': len(train_tuples) + len(val_tuples) + len(test_tuples),
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
    print(f"  - 配对方式: 精确时间戳匹配")
    print(f"  - 坐标系统: UTM（米）")
    print(f"  - 格式: (noisy_path, clean_path, weather)")

    return denoising_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成NCLT BEV去噪配对数据（空间分离版本）',
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
        help='输出文件路径（默认: dataset_root/data/nclt_denoising_tuples.pkl）'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例（从测试区域中分出）'
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(42)

    # 确定输出路径
    if args.output is None:
        output_path = os.path.join(args.dataset_root, 'data', 'nclt_denoising_tuples.pkl')
    else:
        output_path = args.output

    # 生成去噪配对数据
    generate_denoising_pairs(args.dataset_root, output_path, args.val_ratio)


if __name__ == "__main__":
    main()
