#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCLT BEV图像训练数据集生成脚本（空间分离版本）
使用矩形测试区域进行空间分离
测试区域：3个矩形区域（150m × 150m）
训练区域：所有其他区域

适配自 generate_kitti_bev_tuples_spatial.py，用于NCLT数据集
创建时间: 2025-11-05
"""

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import sys
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.datasets.base import TrainingTuple

# NCLT BEV数据集配置
DATASET_FOLDER = "nclt-bev-12-frame-skip"
BEV_SUBFOLDER = ""  # NCLT图像直接在日期-天气目录下
POSES_FILENAME = "poses.txt"

# 天气条件列表
WEATHER_CONDITIONS = ['orin', 'fog', 'rain', 'snow']

# NCLT数据集日期
DATES = [
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


def check_in_test_set(northing, easting, test_regions):
    """
    检查点是否在测试区域内（矩形区域）

    参数:
    - northing: UTM Northing坐标（米）
    - easting: UTM Easting坐标（米）
    - test_regions: 测试区域中心点列表 [[northing, easting], ...]

    返回:
    - bool: True if 在测试区域, False otherwise
    """
    for region_center in test_regions:
        center_n, center_e = region_center
        # 检查是否在矩形边界内
        if (center_n - Y_WIDTH < northing < center_n + Y_WIDTH and
            center_e - X_WIDTH < easting < center_e + X_WIDTH):
            return True
    return False


def load_date_data(base_path, date, weather, split='train'):
    """
    加载单个日期和天气条件的数据（按矩形区域分离）

    参数:
    - base_path: 数据集根目录
    - date: 日期（如 '2012-01-08'）
    - weather: 天气条件
    - split: 'train' (训练区域) 或 'test' (测试区域)
    """
    date_weather = f"{date}-{weather}"

    # 构建poses路径
    if BEV_SUBFOLDER:
        poses_path = os.path.join(base_path, DATASET_FOLDER, date_weather, BEV_SUBFOLDER, POSES_FILENAME)
    else:
        poses_path = os.path.join(base_path, DATASET_FOLDER, date_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        return None

    # 读取poses: timestamp easting northing z qx qy qz qw
    # 注意：NCLT使用UTM坐标（米）
    df_locations = pd.read_csv(poses_path, sep=r"\s+", header=None,
                               names=['timestamp', 'easting', 'northing', 'z', 'qx', 'qy', 'qz', 'qw'])

    # 按timestamp排序
    df_locations = df_locations.sort_values('timestamp').reset_index(drop=True)

    # 按矩形区域分离
    test_mask = df_locations.apply(
        lambda row: check_in_test_set(row['northing'], row['easting'], P_TEST_REGIONS),
        axis=1
    )

    if split == 'test':
        # 测试集：在测试区域内
        df_locations = df_locations[test_mask].reset_index(drop=True)
    elif split == 'train':
        # 训练集：不在测试区域内
        df_locations = df_locations[~test_mask].reset_index(drop=True)
    else:
        raise ValueError(f"Invalid split: {split}")

    # 构建BEV图像文件路径
    # Format: nclt-bev-12-frame-skip/<date>-<weather>/<timestamp>.png
    if BEV_SUBFOLDER:
        df_locations['file'] = (
            DATASET_FOLDER + '/' + date_weather + '/' +
            BEV_SUBFOLDER + '/' +
            df_locations['timestamp'].astype(str) + '.png'
        )
    else:
        df_locations['file'] = (
            DATASET_FOLDER + '/' + date_weather + '/' +
            df_locations['timestamp'].astype(str) + '.png'
        )

    # 过滤掉不存在的文件
    valid_indices = []
    for idx, row in df_locations.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_indices.append(idx)

    df_locations = df_locations.loc[valid_indices].reset_index(drop=True)

    return df_locations


def generate_symmetric_queries(base_path, dates, ind_nn_r, ind_r_r, split='train'):
    """
    生成对称的查询元组（空间分离）

    参数:
    - base_path: 数据集根目录
    - dates: 日期列表
    - ind_nn_r: 正样本距离阈值（米）
    - ind_r_r: 负样本距离阈值（米）
    - split: 'train' 或 'test'
    """

    print(f"  正样本距离阈值: {ind_nn_r}m (UTM坐标)")
    print(f"  负样本距离阈值: {ind_r_r}m (UTM坐标)")
    print(f"  测试区域数量: {len(P_TEST_REGIONS)} 个矩形区域 ({2*X_WIDTH}m × {2*Y_WIDTH}m)")

    # 加载所有数据并分配全局ID
    all_data = []
    global_id = 0
    id_to_info = {}

    split_name = "训练区域（所有其他区域）" if split == 'train' else "测试区域（矩形区域内）"
    print(f"\n加载数据（{split} split - {split_name}）...")

    for date in dates:
        for weather in WEATHER_CONDITIONS:
            df = load_date_data(base_path, date, weather, split=split)
            if df is None or len(df) == 0:
                continue

            print(f"  {date}-{weather}: {len(df)} 个BEV图像")

            for local_idx, row in df.iterrows():
                # 存储UTM坐标（米）: [northing, easting]
                position = np.array([row['northing'], row['easting']], dtype=np.float64)

                id_to_info[global_id] = {
                    'date': date,
                    'weather': weather,
                    'local_idx': local_idx,
                    'position': position,
                    'file': row['file'],
                    'timestamp': int(row['timestamp'])
                }
                all_data.append((global_id, position, date, weather))
                global_id += 1

    print(f"总共加载 {len(all_data)} 个BEV图像")

    if len(all_data) == 0:
        print("错误: 没有加载到任何数据！")
        return {}

    # 构建全局KDTree（使用UTM米制坐标）
    print("\n构建空间索引...")
    positions = np.array([item[1] for item in all_data])
    global_kdtree = KDTree(positions)

    # 查找所有正样本对（对称，只考虑不同天气）
    print("\n查找正样本对（只考虑不同天气）...")
    positive_pairs = set()

    for gid, pos, date, weather in all_data:
        # 直接使用米制阈值查询（UTM坐标）
        neighbors = global_kdtree.query_radius([pos], r=ind_nn_r)[0]

        for neighbor_id in neighbors:
            if neighbor_id == gid:
                continue

            neighbor_info = id_to_info[neighbor_id]

            # 只考虑不同天气条件的点作为正样本
            if neighbor_info['weather'] != weather:
                if gid < neighbor_id:
                    positive_pairs.add((gid, neighbor_id))
                else:
                    positive_pairs.add((neighbor_id, gid))

    print(f"找到 {len(positive_pairs):,d} 对正样本关系")

    # 构建对称的正样本字典
    print("\n构建对称正样本映射...")
    positives_dict = defaultdict(set)

    for id1, id2 in positive_pairs:
        positives_dict[id1].add(id2)
        positives_dict[id2].add(id1)

    # 为每个查询构建非负样本集
    print("\n构建非负样本集...")
    non_negatives_dict = {}

    for gid in range(len(all_data)):
        pos = id_to_info[gid]['position']
        # 直接使用米制阈值查询（UTM坐标）
        neighbors = global_kdtree.query_radius([pos], r=ind_r_r)[0]
        non_negatives_dict[gid] = set(neighbors) - {gid}

    # 创建查询元组
    print("\n创建查询元组...")
    queries = {}

    for gid in range(len(all_data)):
        info = id_to_info[gid]

        positives = sorted(list(positives_dict.get(gid, [])))
        non_negatives = sorted(list(non_negatives_dict.get(gid, [])))

        # 限制非负样本数量
        if len(non_negatives) > 1000:
            non_negatives = sorted(np.random.choice(non_negatives, 1000, replace=False))

        query = TrainingTuple(
            id=gid,
            rel_scan_filepath=info['file'],
            positives=np.array(positives, dtype=np.int32),
            non_negatives=np.array(non_negatives, dtype=np.int32),
            position=info['position'],  # [northing, easting] UTM坐标（米）
            timestamp=info['timestamp']
        )

        query.weather = info['weather']
        query.date = info['date']

        queries[gid] = query

    # 验证对称性
    print("\n验证正样本对称性...")
    asymmetric = 0
    total_relations = 0

    for qid, query in queries.items():
        for pos_id in query.positives:
            total_relations += 1
            if pos_id < len(queries):
                if qid not in queries[pos_id].positives:
                    asymmetric += 1

    if asymmetric == 0:
        print(f"✓ 完美对称！所有 {total_relations:,d} 个正样本关系都是对称的")
    else:
        print(f"⚠ 发现 {asymmetric}/{total_relations} 个不对称关系")

    return queries


def analyze_queries(queries, title="数据集"):
    """分析查询统计信息"""
    from collections import Counter

    if len(queries) == 0:
        print(f"\n{title}: 无数据")
        return

    weather_stats = Counter()
    date_stats = Counter()
    pos_counts = []

    for query in queries.values():
        weather_stats[query.weather] += 1
        date_stats[query.date] += 1
        pos_counts.append(len(query.positives))

    # 计算位置范围
    positions = np.array([q.position[:2] for q in queries.values()])  # [northing, easting]

    print(f"\n{'='*70}")
    print(f"{title} 统计")
    print(f"{'='*70}")
    print(f"总查询数: {len(queries):,d}")

    print("\n空间范围 (UTM坐标，米):")
    print(f"  Northing: [{positions[:, 0].min():.2f}m, {positions[:, 0].max():.2f}m]")
    print(f"  Easting:  [{positions[:, 1].min():.2f}m, {positions[:, 1].max():.2f}m]")

    print("\n天气条件分布:")
    for weather in sorted(weather_stats.keys()):
        count = weather_stats[weather]
        print(f"  {weather:8s}: {count:6,d} ({count*100.0/len(queries):5.1f}%)")

    print("\n日期分布:")
    for date in sorted(date_stats.keys()):
        count = date_stats[date]
        print(f"  {date}: {count:6,d} ({count*100.0/len(queries):5.1f}%)")

    print(f"\n正样本统计:")
    if len(pos_counts) > 0:
        print(f"  平均: {np.mean(pos_counts):.1f}")
        print(f"  中位数: {np.median(pos_counts):.0f}")
        print(f"  范围: [{min(pos_counts)}, {max(pos_counts)}]")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='生成NCLT BEV图像训练数据集（空间分离版本）')
    parser.add_argument('--dataset_root', type=str, default='/home/cxw/pro/best',
                       help='数据集根目录路径')
    parser.add_argument('--ind_nn_r', type=float, default=10,
                       help='正样本距离阈值(米)')
    parser.add_argument('--ind_r_r', type=float, default=50,
                       help='负样本距离阈值(米)')

    args = parser.parse_args()

    print(f'\n{"="*70}')
    print(f'NCLT BEV图像训练数据集生成（空间分离版本）')
    print(f'{"="*70}')
    print(f'数据集根目录: {args.dataset_root}')
    print(f'使用日期: {len(DATES)} 个 - {DATES[:3]}...')
    print(f'划分方式: 矩形测试区域分离')
    print(f'           测试区域: {len(P_TEST_REGIONS)} 个矩形区域 ({2*X_WIDTH}m × {2*Y_WIDTH}m)')
    print(f'           训练区域: 所有其他区域')
    print(f'正样本阈值: {args.ind_nn_r}m')
    print(f'负样本阈值: {args.ind_r_r}m')

    # 生成训练集（训练区域）
    print("\n" + "="*70)
    print("生成训练集（训练区域）")
    print("="*70)

    train_queries = generate_symmetric_queries(
        args.dataset_root, DATES,
        args.ind_nn_r, args.ind_r_r,
        split='train'
    )

    # 保存训练集
    output_path = os.path.join(args.dataset_root, 'data')
    os.makedirs(output_path, exist_ok=True)

    train_file = os.path.join(output_path, "nclt_bev_training_queries.pickle")
    with open(train_file, 'wb') as f:
        pickle.dump(train_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存训练集: {train_file}")

    analyze_queries(train_queries, "训练集（训练区域）")

    # 生成测试集（测试区域）
    print("\n" + "="*70)
    print("生成测试集（测试区域）")
    print("="*70)

    test_queries = generate_symmetric_queries(
        args.dataset_root, DATES,
        args.ind_nn_r, args.ind_r_r,
        split='test'
    )

    # 保存测试集
    test_file = os.path.join(output_path, "nclt_bev_test_queries.pickle")
    with open(test_file, 'wb') as f:
        pickle.dump(test_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存测试集: {test_file}")

    analyze_queries(test_queries, "测试集（测试区域）")

    # 总结
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    total_queries = len(train_queries) + len(test_queries)
    print(f"训练集: {len(train_queries):,d} ({len(train_queries)/total_queries*100:.1f}%)")
    print(f"测试集: {len(test_queries):,d} ({len(test_queries)/total_queries*100:.1f}%)")
    print(f"总计: {total_queries:,d}")
    print(f"\n数据格式: BEV PNG图像")
    print(f"坐标系统: UTM（米）")
    print(f"划分方式: 矩形测试区域分离")
    print(f"优势: 空间上完全分离，避免数据泄漏")


if __name__ == '__main__':
    main()
