#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boreas BEV图像训练数据集生成脚本（空间分离版本）
使用矩形测试区域进行空间分离
测试区域：单个矩形区域（约20%的空间范围）
训练区域：其他区域（约80%）

天气配置：
- Clear作为参考(database)
- Snow/Rain作为查询(query)
- 正样本必须跨天气条件

创建时间: 2025-11-07
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

from imlpr_datasets.base_datasets import TrainingTuple

# Boreas BEV数据集配置
DATASET_FOLDER = "boreas-skip-bev"
BEV_SUBFOLDER = "bev_448"  # Boreas BEV图像目录
POSES_FILENAME = "poses.txt"

# Boreas序列列表（包含天气后缀）
SEQUENCES = [
    'boreas-2020-12-01-13-26-snow',
    'boreas-2021-01-26-11-22-snow',
    'boreas-2021-04-08-12-44-clear',
    'boreas-2021-04-29-15-55-rain',
]

# 天气条件（从目录名提取）
WEATHER_CONDITIONS = ['clear', 'snow', 'rain']

# 测试区域配置（将在main中根据数据自动计算）
# 目标：单个矩形区域覆盖约20%的空间范围
TEST_REGION_CENTER = None  # [northing, easting]
TEST_REGION_HALF_WIDTH = None  # 米


def check_in_test_set(northing, easting, center, half_width):
    """
    检查点是否在测试区域内（矩形区域）

    参数:
    - northing: UTM Northing坐标（米）
    - easting: UTM Easting坐标（米）
    - center: 测试区域中心点 [northing, easting]
    - half_width: 矩形半宽（米）

    返回:
    - bool: True if 在测试区域, False otherwise
    """
    if center is None:
        return False

    center_n, center_e = center
    # 检查是否在矩形边界内
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


def load_sequence_data(base_path, sequence, split='train'):
    """
    加载单个序列的数据（按矩形区域分离）

    参数:
    - base_path: 数据集根目录
    - sequence: 序列名（包含天气后缀，如 'boreas-2020-12-01-13-26-snow'）
    - split: 'train' (训练区域) 或 'test' (测试区域)
    """
    # 构建poses路径
    poses_path = os.path.join(base_path, DATASET_FOLDER, sequence, POSES_FILENAME)

    if not os.path.exists(poses_path):
        return None, None

    # 提取天气条件
    weather = extract_weather_from_dirname(sequence)
    if weather is None or weather not in WEATHER_CONDITIONS:
        print(f"警告: 无法从 {sequence} 提取有效天气条件")
        return None, None

    # 读取poses: timestamp easting northing z qx qy qz qw [额外列...]
    # 注意：Boreas使用UTM坐标（米），但poses文件有13+列，只读前8列
    data = np.loadtxt(poses_path)

    # 创建DataFrame并确保数据类型正确
    df_locations = pd.DataFrame({
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
    df_locations = df_locations.sort_values('timestamp').reset_index(drop=True)

    # 按矩形区域分离
    if TEST_REGION_CENTER is not None:
        test_mask = df_locations.apply(
            lambda row: check_in_test_set(row['northing'], row['easting'],
                                         TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH),
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
    # Format: boreas-bev-skip/<sequence>/lidar/<timestamp>.png
    df_locations['file'] = (
        DATASET_FOLDER + '/' + sequence + '/' +
        BEV_SUBFOLDER + '/' +
        df_locations['timestamp'].astype(str) + '.png'
    )

    # 过滤掉不存在的文件
    valid_indices = []
    for idx, row in df_locations.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_indices.append(idx)

    df_locations = df_locations.loc[valid_indices].reset_index(drop=True)

    return df_locations, weather


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
    # 假设数据大致均匀分布，矩形区域面积 = (2*half_width)^2
    # 总区域面积约 = range_n * range_e
    # 目标: (2*half_width)^2 / (range_n * range_e) ≈ target_ratio
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


def generate_symmetric_queries(base_path, sequences, ind_nn_r, ind_r_r, split='train'):
    """
    生成对称的查询元组（空间分离，跨天气条件匹配）

    参数:
    - base_path: 数据集根目录
    - sequences: 序列列表（完整名称，包含天气）
    - ind_nn_r: 正样本距离阈值（米）
    - ind_r_r: 负样本距离阈值（米）
    - split: 'train' 或 'test'
    """

    print(f"  正样本距离阈值: {ind_nn_r}m (UTM坐标)")
    print(f"  负样本距离阈值: {ind_r_r}m (UTM坐标)")
    print(f"  正样本策略: 必须跨天气条件（clear ↔ snow/rain, snow ↔ rain）")

    # 加载所有数据并分配全局ID
    all_data = []
    global_id = 0
    id_to_info = {}

    split_name = "训练区域（约80%）" if split == 'train' else "测试区域（约20%）"
    print(f"\n加载数据（{split} split - {split_name}）...")

    for sequence in sequences:
        df, weather = load_sequence_data(base_path, sequence, split=split)
        if df is None or len(df) == 0:
            continue

        print(f"  {sequence}: {len(df)} 个BEV图像 (天气: {weather})")

        for local_idx, row in df.iterrows():
            # 存储UTM坐标（米）: [northing, easting]
            position = np.array([row['northing'], row['easting']], dtype=np.float64)

            id_to_info[global_id] = {
                'sequence': sequence,
                'weather': weather,
                'local_idx': local_idx,
                'position': position,
                'file': row['file'],
                'timestamp': int(row['timestamp'])
            }
            all_data.append((global_id, position, sequence, weather))
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

    for gid, pos, sequence, weather in all_data:
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
        query.sequence = info['sequence']

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
    sequence_stats = Counter()
    pos_counts = []

    for query in queries.values():
        weather_stats[query.weather] += 1
        sequence_stats[query.sequence] += 1
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

    print("\n序列分布:")
    for sequence in sorted(sequence_stats.keys()):
        count = sequence_stats[sequence]
        print(f"  {sequence}: {count:6,d} ({count*100.0/len(queries):5.1f}%)")

    print(f"\n正样本统计:")
    if len(pos_counts) > 0:
        print(f"  平均: {np.mean(pos_counts):.1f}")
        print(f"  中位数: {np.median(pos_counts):.0f}")
        print(f"  范围: [{min(pos_counts)}, {max(pos_counts)}]")
        print(f"  有正样本的查询: {sum(1 for c in pos_counts if c > 0):,d} ({sum(1 for c in pos_counts if c > 0)*100.0/len(pos_counts):.1f}%)")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='生成Boreas BEV图像训练数据集（空间分离版本）')
    parser.add_argument('--dataset_root', type=str, default='/data/users/cxw/pro/clav',
                       help='数据集根目录路径')
    parser.add_argument('--ind_nn_r', type=float, default=10,
                       help='正样本距离阈值(米)')
    parser.add_argument('--ind_r_r', type=float, default=50,
                       help='负样本距离阈值(米)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='测试区域比例（默认0.2即20%）')

    args = parser.parse_args()

    # 设置全局变量
    global TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH
    TEST_REGION_CENTER, TEST_REGION_HALF_WIDTH = calculate_test_region(
        args.dataset_root, args.test_ratio
    )

    print(f'\n{"="*70}')
    print(f'Boreas BEV图像训练数据集生成（空间分离版本）')
    print(f'{"="*70}')
    print(f'数据集根目录: {args.dataset_root}')
    print(f'使用序列: {len(SEQUENCES)} 个')
    for seq in SEQUENCES:
        print(f'  - {seq}')
    print(f'划分方式: 矩形测试区域分离（训练:测试 ≈ 8:2）')
    print(f'正样本阈值: {args.ind_nn_r}m')
    print(f'负样本阈值: {args.ind_r_r}m')
    print(f'天气匹配: 必须跨天气条件（Clear作为参考）')

    # 验证序列是否存在
    seq_dir = os.path.join(args.dataset_root, DATASET_FOLDER)
    actual_sequences = []
    for seq in SEQUENCES:
        seq_path = os.path.join(seq_dir, seq)
        if os.path.exists(seq_path):
            actual_sequences.append(seq)
        else:
            print(f"\n⚠ 警告: 序列目录不存在: {seq}")

    print(f"\n实际找到 {len(actual_sequences)} 个序列:")
    for seq in actual_sequences:
        print(f"  - {seq}")

    # 生成训练集（训练区域）
    print("\n" + "="*70)
    print("生成训练集（训练区域）")
    print("="*70)

    train_queries = generate_symmetric_queries(
        args.dataset_root, actual_sequences,
        args.ind_nn_r, args.ind_r_r,
        split='train'
    )

    # 保存训练集
    output_path = os.path.join(args.dataset_root, 'data')
    os.makedirs(output_path, exist_ok=True)

    train_file = os.path.join(output_path, "boreas_bev_training_queries_spatial.pickle")
    with open(train_file, 'wb') as f:
        pickle.dump(train_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存训练集: {train_file}")

    analyze_queries(train_queries, "训练集（训练区域）")

    # 生成测试集（测试区域）
    print("\n" + "="*70)
    print("生成测试集（测试区域）")
    print("="*70)

    test_queries = generate_symmetric_queries(
        args.dataset_root, actual_sequences,
        args.ind_nn_r, args.ind_r_r,
        split='test'
    )

    # 保存测试集
    test_file = os.path.join(output_path, "boreas_bev_test_queries_spatial.pickle")
    with open(test_file, 'wb') as f:
        pickle.dump(test_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存测试集: {test_file}")

    analyze_queries(test_queries, "测试集（测试区域）")

    # 总结
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    total_queries = len(train_queries) + len(test_queries)

    if total_queries > 0:
        print(f"训练集: {len(train_queries):,d} ({len(train_queries)/total_queries*100:.1f}%)")
        print(f"测试集: {len(test_queries):,d} ({len(test_queries)/total_queries*100:.1f}%)")
        print(f"总计: {total_queries:,d}")
        print(f"\n数据格式: BEV PNG图像")
        print(f"坐标系统: UTM（米）")
        print(f"划分方式: 矩形测试区域分离")
        print(f"天气策略: Clear作为参考，跨天气条件匹配")
        print(f"优势: 空间上完全分离，避免数据泄漏")
    else:
        print(f"⚠️ 警告: 没有生成任何查询数据")
        print(f"训练集: {len(train_queries):,d}")
        print(f"测试集: {len(test_queries):,d}")
        print(f"\n请检查:")
        print(f"  1. BEV图像文件是否存在于 {DATASET_FOLDER}/<sequence>/{BEV_SUBFOLDER}/")
        print(f"  2. poses.txt文件是否正确")
        print(f"  3. 序列目录名是否匹配")


if __name__ == '__main__':
    main()
