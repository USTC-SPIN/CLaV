#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boreas BEV图像评估文件生成脚本（空间分离版本）
使用矩形测试区域进行空间分离
生成用于评估的database和query文件（包含ground truth）

评估配置：
- Database: Clear天气作为参考数据库
- Query: Snow和Rain天气作为查询集
- 每个query包含针对所有database sets的ground truth邻居

创建时间: 2025-11-07
"""

import numpy as np
import os
import pandas as pd
import pickle
import argparse
import sys
from collections import defaultdict

# 添加父目录到Python路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Boreas BEV数据集配置
DATASET_FOLDER = "boreas-skip-bev"
BEV_SUBFOLDER = "bev_448"  # Boreas BEV图像目录
POSES_FILENAME = "poses.txt"

# 天气条件列表
CLEAN_WEATHER = 'clear'  # 清晰天气，用于database
NOISY_WEATHERS = ['snow', 'rain']  # 噪声天气，用于query
ALL_WEATHERS = [CLEAN_WEATHER] + NOISY_WEATHERS

# Boreas序列列表（包含天气后缀）
SEQUENCES = [
    'boreas-2020-12-01-13-26-snow',
    'boreas-2021-01-26-11-22-snow',
    'boreas-2021-04-08-12-44-clear',
    'boreas-2021-04-29-15-55-rain',
]

# 测试区域配置（将在main中根据数据自动计算）
TEST_REGION_CENTER = None  # [northing, easting]
TEST_REGION_HALF_WIDTH = None  # 米

# Ground truth距离阈值
POSITIVE_DIST_THRESHOLD = 10.0  # 10米内为正确匹配


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
    if center is None:
        return False

    center_n, center_e = center
    if (center_n - half_width < northing < center_n + half_width and
        center_e - half_width < easting < center_e + half_width):
        return True
    return False


def extract_weather_from_dirname(dirname):
    """从目录名提取天气条件"""
    # 格式: boreas-2020-12-01-13-26-snow
    parts = dirname.split('-')
    if len(parts) >= 7:
        return parts[-1]  # 最后一部分是天气
    return None


def calculate_test_region(base_path, target_ratio=0.2):
    """
    根据所有数据计算测试区域
    目标是覆盖约target_ratio（默认20%）的空间范围

    返回:
    - center: [northing, easting]
    - half_width: 米
    """
    print("\n计算测试区域...")

    # 加载所有序列的位置数据
    all_positions = []

    for sequence in SEQUENCES:
        poses_path = os.path.join(base_path, DATASET_FOLDER, sequence, POSES_FILENAME)
        if not os.path.exists(poses_path):
            continue

        # 使用numpy直接读取，确保数据类型正确
        data = np.loadtxt(poses_path)
        # 格式: timestamp(0), easting(1), northing(2), ...
        positions = data[:, [2, 1]]  # [northing, easting]
        all_positions.append(positions)

    if len(all_positions) == 0:
        raise ValueError("无法加载任何位置数据来计算测试区域")

    all_positions = np.vstack(all_positions)

    # 计算数据范围
    min_n, max_n = all_positions[:, 0].min(), all_positions[:, 0].max()
    min_e, max_e = all_positions[:, 1].min(), all_positions[:, 1].max()
    range_n = max_n - min_n
    range_e = max_e - min_e

    print(f"  数据范围:")
    print(f"    Northing: [{min_n:.2f}, {max_n:.2f}] (范围: {range_n:.2f}m)")
    print(f"    Easting:  [{min_e:.2f}, {max_e:.2f}] (范围: {range_e:.2f}m)")

    # 选择测试区域中心（路线中部偏北）
    center_n = min_n + range_n * 0.6  # 60%位置（偏北）
    center_e = (min_e + max_e) / 2.0  # 东西方向居中

    # 计算半宽使得测试区域覆盖约20%的空间
    total_area = range_n * range_e
    target_area = total_area * target_ratio
    half_width = np.sqrt(target_area) / 2.0

    # 四舍五入到整十米
    half_width = round(half_width / 10) * 10

    center = [center_n, center_e]

    print(f"  测试区域配置:")
    print(f"    中心点: [{center_n:.2f}, {center_e:.2f}]")
    print(f"    半宽: {half_width:.0f}m")
    print(f"    区域大小: {2*half_width:.0f}m × {2*half_width:.0f}m")
    print(f"    预估覆盖率: {(2*half_width)**2 / total_area * 100:.1f}%")

    return center, half_width


def load_sequence_data(base_path, sequence):
    """
    加载单个序列的数据（仅测试区域）

    参数:
    - base_path: 数据集根目录
    - sequence: 序列名（包含天气后缀）

    返回测试区域内的数据和天气
    """
    # 构建poses路径
    poses_path = os.path.join(base_path, DATASET_FOLDER, sequence, POSES_FILENAME)

    if not os.path.exists(poses_path):
        print(f"  ⚠ 未找到: {poses_path}")
        return None, None

    # 提取天气条件
    weather = extract_weather_from_dirname(sequence)
    if weather is None or weather not in ALL_WEATHERS:
        print(f"  ⚠ {sequence} 无效天气条件: {weather}")
        return None, None

    # 读取poses: timestamp easting northing z qx qy qz qw [额外列...]
    # Boreas使用UTM坐标（米），但poses文件有13+列，只读前8列
    data = np.loadtxt(poses_path)

    # 创建DataFrame并确保数据类型正确
    df = pd.DataFrame({
        'timestamp': data[:, 0].astype(int),
        'easting': data[:, 1].astype(float),
        'northing': data[:, 2].astype(float),
        'z': data[:, 3].astype(float),
        'qx': data[:, 4].astype(float),
        'qy': data[:, 5].astype(float),
        'qz': data[:, 6].astype(float),
        'qw': data[:, 7].astype(float)
    })

    # 按timestamp排序
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 只保留测试区域内的点
    if TEST_REGION_CENTER is not None:
        test_mask = df.apply(
            lambda row: check_in_test_set(row['northing'], row['easting'],
                                         TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH),
            axis=1
        )
        df = df[test_mask].reset_index(drop=True)

    if len(df) == 0:
        print(f"  ⚠ {sequence} 测试区域无数据")
        return None, None

    # 构建BEV图像文件路径
    # Format: boreas-bev-skip/<sequence>/lidar/<timestamp>.png
    df['file'] = (
        DATASET_FOLDER + '/' + sequence + '/' +
        BEV_SUBFOLDER + '/' +
        df['timestamp'].astype(str) + '.png'
    )

    # 过滤掉不存在的文件
    valid_indices = []
    for idx, row in df.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_indices.append(idx)

    df = df.loc[valid_indices].reset_index(drop=True)

    return df, weather


def generate_evaluation_sets(base_path):
    """
    生成Boreas BEV评估集（使用矩形测试区域）
    Database: Clear天气作为参考数据库（按序列分组）
    Query: Snow和Rain天气作为查询集
    包含ground truth邻居索引
    """
    print("\n" + "="*70)
    print("生成Boreas BEV评估文件（空间分离版本）")
    print("="*70)
    print(f"测试区域: 单个矩形区域 ({2*TEST_REGION_HALF_WIDTH:.0f}m × {2*TEST_REGION_HALF_WIDTH:.0f}m)")
    print(f"Ground truth阈值: {POSITIVE_DIST_THRESHOLD}m")
    print(f"Database: {CLEAN_WEATHER} 天气")
    print(f"Query: {', '.join(NOISY_WEATHERS)} 天气")

    # 按天气分类序列
    clean_sequences = [s for s in SEQUENCES if extract_weather_from_dirname(s) == CLEAN_WEATHER]
    noisy_sequences = [s for s in SEQUENCES if extract_weather_from_dirname(s) in NOISY_WEATHERS]

    print(f"\n使用序列:")
    print(f"  Clear: {clean_sequences}")
    print(f"  Noisy: {noisy_sequences}")

    # 加载清晰天气作为database（按序列分组）
    print(f"\n加载Database ({CLEAN_WEATHER}天气 - 测试区域)...")
    database_list = []
    database_frames_list = []  # 保存每个database set的完整帧信息

    for sequence in clean_sequences:
        db_df, weather = load_sequence_data(base_path, sequence)
        if db_df is None or len(db_df) == 0:
            print(f"  ⚠ 跳过序列 {sequence}")
            continue

        print(f"  {sequence}: {len(db_df)} 个BEV图像")

        # 构建database字典
        database_dict = {}
        frames_list = []

        for idx, row in db_df.iterrows():
            database_dict[idx] = {
                'query': row['file'],
                'position': [row['northing'], row['easting']],  # UTM坐标（米）
                'timestamp': int(row['timestamp']),
                'northing': row['northing'],
                'easting': row['easting']
            }

            # 保存帧信息用于计算ground truth
            frames_list.append({
                'position': np.array([row['northing'], row['easting']], dtype=np.float64),
                'timestamp': int(row['timestamp']),
                'file': row['file']
            })

        database_list.append(database_dict)
        database_frames_list.append(frames_list)

    if len(database_list) == 0:
        raise ValueError("无法加载任何database数据")

    total_db = sum(len(d) for d in database_list)
    print(f"\nDatabase总计: {len(database_list)} 个sets, {total_db} 个BEV图像")

    # 为每种噪声天气创建query字典（包含ground truth）
    print(f"\n生成Query字典（包含ground truth）...")
    all_query_lists = []

    for sequence in noisy_sequences:
        query_df, weather = load_sequence_data(base_path, sequence)
        if query_df is None or len(query_df) == 0:
            print(f"  ⚠ 跳过 {sequence}")
            continue

        print(f"  {sequence}: {len(query_df)} 个BEV图像 (天气: {weather})")

        # 采样20%作为查询（减少评估时间）
        query_ratio = 0.2
        num_queries = max(1, int(len(query_df) * query_ratio))
        if num_queries < len(query_df):
            query_indices = np.random.choice(len(query_df), size=num_queries, replace=False)
            query_df = query_df.iloc[query_indices].reset_index(drop=True)

        # 构建query字典
        query_dict = {}
        for q_idx, row in query_df.iterrows():
            query_pos = np.array([row['northing'], row['easting']], dtype=np.float64)

            query_entry = {
                'query': row['file'],
                'position': [row['northing'], row['easting']],
                'timestamp': int(row['timestamp']),
                'northing': row['northing'],
                'easting': row['easting'],
                'weather': weather,
                'sequence': sequence
            }

            # 计算针对每个database set的ground truth
            for db_idx, db_frames in enumerate(database_frames_list):
                true_neighbors = []
                for db_frame_idx, db_frame in enumerate(db_frames):
                    # 计算UTM距离（米）
                    dist = np.linalg.norm(query_pos - db_frame['position'])
                    if dist <= POSITIVE_DIST_THRESHOLD:
                        true_neighbors.append(db_frame_idx)

                # 存储ground truth（使用database set索引作为键）
                query_entry[db_idx] = np.array(true_neighbors, dtype=np.int32)

            query_dict[q_idx] = query_entry

        if query_dict:
            all_query_lists.append(query_dict)
            print(f"    生成 {len(query_dict)} 个queries")

    print(f"\n生成 {len(all_query_lists)} 个query sets")

    # 保存文件
    output_dir = os.path.join(base_path, 'data')
    os.makedirs(output_dir, exist_ok=True)

    print("\n保存评估文件...")

    # 保存database（单个文件，包含所有clear序列）
    db_file = os.path.join(output_dir, 'boreas_bev_evaluation_database_spatial.pickle')
    with open(db_file, 'wb') as f:
        pickle.dump(database_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ Database: {db_file}")

    # 保存query（单个文件，包含所有噪声天气序列）
    query_file = os.path.join(output_dir, 'boreas_bev_evaluation_query_spatial.pickle')
    with open(query_file, 'wb') as f:
        pickle.dump(all_query_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ Query: {query_file}")

    # 额外保存按天气分离的版本
    print("\n保存天气特定评估文件...")
    weather_query_dict = defaultdict(list)

    # 重新按天气分组query
    for query_set in all_query_lists:
        for query_entry in query_set.values():
            weather = query_entry['weather']
            weather_query_dict[weather].append(query_entry)

    # 保存每种天气的评估文件
    for weather in NOISY_WEATHERS:
        if weather not in weather_query_dict:
            continue

        # Database文件（所有天气共享同一个database）
        weather_db_file = os.path.join(output_dir, f'boreas_bev_{weather}_evaluation_database_spatial.pickle')
        with open(weather_db_file, 'wb') as f:
            pickle.dump(database_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Database ({weather}): {weather_db_file}")

        # Query文件（特定天气）
        # 需要重新组织为list of dicts格式
        weather_queries = weather_query_dict[weather]
        query_list_for_weather = [{}]  # 单个query set
        for idx, q in enumerate(weather_queries):
            query_list_for_weather[0][idx] = q

        weather_query_file = os.path.join(output_dir, f'boreas_bev_{weather}_evaluation_query_spatial.pickle')
        with open(weather_query_file, 'wb') as f:
            pickle.dump(query_list_for_weather, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Query ({weather}): {weather_query_file}")

    # 统计信息
    print("\n" + "="*70)
    print("评估集统计")
    print("="*70)
    print(f"Database ({CLEAN_WEATHER}):")
    print(f"  {len(database_list)} 个sets")
    for idx, db_set in enumerate(database_list):
        print(f"    Set {idx}: {len(db_set)} 个BEV图像")

    print(f"\nQuery (噪声天气):")
    for weather in NOISY_WEATHERS:
        if weather in weather_query_dict:
            print(f"  {weather}: {len(weather_query_dict[weather])} 个queries")

    # Ground truth统计
    total_gt = 0
    num_queries_with_gt = 0
    for query_set in all_query_lists:
        for query_entry in query_set.values():
            for db_idx in range(len(database_list)):
                if db_idx in query_entry and len(query_entry[db_idx]) > 0:
                    num_queries_with_gt += 1
                    total_gt += len(query_entry[db_idx])

    if num_queries_with_gt > 0:
        avg_gt = total_gt / num_queries_with_gt
        print(f"\nGround truth统计:")
        print(f"  有ground truth的queries: {num_queries_with_gt}")
        print(f"  平均每query的邻居数: {avg_gt:.1f}")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"生成的评估文件:")
    print(f"  - boreas_bev_evaluation_database_spatial.pickle")
    print(f"  - boreas_bev_evaluation_query_spatial.pickle")
    for weather in NOISY_WEATHERS:
        print(f"  - boreas_bev_{weather}_evaluation_database_spatial.pickle")
        print(f"  - boreas_bev_{weather}_evaluation_query_spatial.pickle")

    print(f"\n数据特性:")
    print(f"  - 来源: Boreas数据集测试区域数据")
    print(f"  - Database: 始终使用{CLEAN_WEATHER}天气")
    print(f"  - Query: 噪声天气 ({', '.join(NOISY_WEATHERS)})")
    print(f"  - Ground truth: {POSITIVE_DIST_THRESHOLD}m阈值")
    print(f"  - 坐标系统: UTM（米）")
    print(f"  - 数据格式: BEV PNG图像")
    print(f"  - 用于测试泛化性能")

    return database_list, all_query_lists


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成Boreas BEV图像评估文件（空间分离版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/data/users/cxw/pro/clav',
        help='数据集根目录'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2,
        help='测试区域比例（默认0.2即20%）'
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(42)

    # 计算测试区域
    global TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH
    TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH = calculate_test_region(
        args.dataset_root, args.test_ratio
    )

    # 生成评估集
    generate_evaluation_sets(args.dataset_root)


if __name__ == "__main__":
    main()
