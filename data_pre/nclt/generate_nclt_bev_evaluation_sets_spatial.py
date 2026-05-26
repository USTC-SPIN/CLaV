#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCLT BEV图像评估文件生成脚本（空间分离版本）
使用矩形测试区域进行空间分离
生成用于评估的database和query文件（包含ground truth）

遵循PointNetVLAD评估协议：
- Database: 清晰天气(orin)
- Query: 噪声天气(fog, rain, snow)
- 每个query包含针对所有database sets的ground truth邻居

适配自 generate_kitti_bev_evaluation_sets_spatial.py，用于NCLT数据集
创建时间: 2025-11-05
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

# NCLT BEV数据集配置
DATASET_FOLDER = "nclt-bev-12-frame-skip"
BEV_SUBFOLDER = ""  # NCLT图像直接在日期-天气目录下
POSES_FILENAME = "poses.txt"

# 天气条件列表
CLEAN_WEATHER = 'orin'  # 清晰天气，用于database
NOISY_WEATHERS = ['fog', 'rain', 'snow']  # 噪声天气，用于query
ALL_WEATHERS = [CLEAN_WEATHER] + NOISY_WEATHERS

# NCLT数据集日期
EVAL_DATES = [
    '2012-01-08', '2012-01-15', '2012-01-22', '2012-08-20',
    '2012-09-28', '2012-10-28', '2012-11-04', '2012-11-16',
    '2012-11-17', '2012-12-01', '2013-02-23', '2013-04-05'
]

# 矩形测试区域参数
X_WIDTH = 75  # 米（半宽）
Y_WIDTH = 75  # 米（半高）

# 测试区域中心点（UTM坐标，米）
P1 = [4685682.634858, 276089.034153]  # 西南象限
P2 = [4685707.728925, 276309.850477]  # 东南象限
P3 = [4685905.456411, 276116.466847]  # 西北象限
P_TEST_REGIONS = [P1, P2, P3]

# Ground truth距离阈值
POSITIVE_DIST_THRESHOLD = 10.0  # 10米内为正确匹配


def check_in_test_set(northing, easting, test_regions):
    """
    检查点是否在测试区域内（矩形区域）

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


def load_date_data(base_path, date, weather):
    """
    加载单个日期和天气条件的数据（仅测试区域）

    参数:
    - base_path: 数据集根目录
    - date: 日期
    - weather: 天气条件

    返回测试区域内的数据
    """
    date_weather = f"{date}-{weather}"

    # 构建poses路径
    if BEV_SUBFOLDER:
        poses_path = os.path.join(base_path, DATASET_FOLDER, date_weather, BEV_SUBFOLDER, POSES_FILENAME)
    else:
        poses_path = os.path.join(base_path, DATASET_FOLDER, date_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        print(f"  ⚠ 未找到: {poses_path}")
        return None

    # 读取poses: timestamp easting northing z qx qy qz qw
    # NCLT使用UTM坐标（米）
    df = pd.read_csv(poses_path, sep=r"\s+", header=None,
                     names=['timestamp', 'easting', 'northing', 'z', 'qx', 'qy', 'qz', 'qw'])

    # 按timestamp排序
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 只保留测试区域内的点
    test_mask = df.apply(
        lambda row: check_in_test_set(row['northing'], row['easting'], P_TEST_REGIONS),
        axis=1
    )
    df = df[test_mask].reset_index(drop=True)

    if len(df) == 0:
        print(f"  ⚠ {date}-{weather} 测试区域无数据")
        return None

    # 构建BEV图像文件路径
    if BEV_SUBFOLDER:
        df['file'] = (
            DATASET_FOLDER + '/' + date_weather + '/' +
            BEV_SUBFOLDER + '/' +
            df['timestamp'].astype(str) + '.png'
        )
    else:
        df['file'] = (
            DATASET_FOLDER + '/' + date_weather + '/' +
            df['timestamp'].astype(str) + '.png'
        )

    # 过滤掉不存在的文件
    valid_indices = []
    for idx, row in df.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_indices.append(idx)

    df = df.loc[valid_indices].reset_index(drop=True)

    return df


def generate_evaluation_sets(base_path):
    """
    生成NCLT BEV评估集（使用矩形测试区域）
    Database: 清晰天气(orin)作为参考数据库（按日期分组）
    Query: 噪声天气(fog, rain, snow)作为查询集
    包含ground truth邻居索引
    """
    print("\n" + "="*70)
    print("生成NCLT BEV评估文件（空间分离版本）")
    print("="*70)
    print(f"测试区域: {len(P_TEST_REGIONS)} 个矩形区域 ({2*X_WIDTH}m × {2*Y_WIDTH}m)")
    print(f"Ground truth阈值: {POSITIVE_DIST_THRESHOLD}m")
    print(f"Database: {CLEAN_WEATHER} 天气")
    print(f"Query: {', '.join(NOISY_WEATHERS)} 天气")

    # 加载清晰天气作为database（按日期分组）
    print(f"\n加载Database ({CLEAN_WEATHER}天气 - 测试区域)...")
    database_list = []
    database_frames_list = []  # 保存每个database set的完整帧信息

    for date in EVAL_DATES:
        db_df = load_date_data(base_path, date, CLEAN_WEATHER)
        if db_df is None or len(db_df) == 0:
            print(f"  ⚠ 跳过日期 {date}")
            continue

        print(f"  {date}-{CLEAN_WEATHER}: {len(db_df)} 个BEV图像")

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

    for weather in NOISY_WEATHERS:
        print(f"\n处理 {weather} 天气...")

        for date in EVAL_DATES:
            query_df = load_date_data(base_path, date, weather)
            if query_df is None or len(query_df) == 0:
                print(f"  ⚠ 跳过 {date}-{weather}")
                continue

            print(f"  {date}-{weather}: {len(query_df)} 个BEV图像")

            # 采样20%作为查询
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
                    'weather': weather
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

    # 保存database（单个文件，包含所有日期）
    db_file = os.path.join(output_dir, 'nclt_bev_evaluation_database.pickle')
    with open(db_file, 'wb') as f:
        pickle.dump(database_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ Database: {db_file}")

    # 保存query（单个文件，包含所有噪声天气）
    query_file = os.path.join(output_dir, 'nclt_bev_evaluation_query.pickle')
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
        weather_db_file = os.path.join(output_dir, f'nclt_bev_{weather}_evaluation_database.pickle')
        with open(weather_db_file, 'wb') as f:
            pickle.dump(database_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Database ({weather}): {weather_db_file}")

        # Query文件（特定天气）
        # 需要重新组织为list of dicts格式
        weather_queries = weather_query_dict[weather]
        query_list_for_weather = [{}]  # 单个query set
        for idx, q in enumerate(weather_queries):
            query_list_for_weather[0][idx] = q

        weather_query_file = os.path.join(output_dir, f'nclt_bev_{weather}_evaluation_query.pickle')
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
    print(f"  - nclt_bev_evaluation_database.pickle")
    print(f"  - nclt_bev_evaluation_query.pickle")
    for weather in NOISY_WEATHERS:
        print(f"  - nclt_bev_{weather}_evaluation_database.pickle")
        print(f"  - nclt_bev_{weather}_evaluation_query.pickle")

    print(f"\n数据特性:")
    print(f"  - 来源: {len(EVAL_DATES)} 个日期的测试区域数据")
    print(f"  - Database: 始终使用{CLEAN_WEATHER}天气")
    print(f"  - Query: 噪声天气 ({', '.join(NOISY_WEATHERS)})")
    print(f"  - Ground truth: {POSITIVE_DIST_THRESHOLD}m阈值")
    print(f"  - 坐标系统: UTM（米）")
    print(f"  - 数据格式: BEV PNG图像")
    print(f"  - 遵循PointNetVLAD评估协议")

    return database_list, all_query_lists


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成NCLT BEV图像评估文件（空间分离版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/cxw/pro/best',
        help='数据集根目录'
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(42)

    # 生成评估集
    generate_evaluation_sets(args.dataset_root)


if __name__ == "__main__":
    main()
