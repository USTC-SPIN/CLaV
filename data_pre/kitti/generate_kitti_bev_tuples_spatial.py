#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI BEV图像训练数据集生成脚本（空间分离版本）
对每个序列单独计算空间划分，沿轨迹方向按累积距离划分
前20%累积距离作为测试集，后80%作为训练集

适配自 generate_kitti_part_tuples_spatial.py，用于BEV PNG图像
"""

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import sys
from collections import defaultdict

# TrainingTuple类定义
class TrainingTuple:
    """训练样本元组"""
    def __init__(self, id, rel_scan_filepath, positives, non_negatives, position):
        self.id = id
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position

# KITTI BEV数据集配置
DATASET_FOLDER = "kitti-bev-skip"
BEV_SUBFOLDER = "bev_448"  # BEV图像子目录
POSES_FILENAME = "poses.txt"

# 天气条件列表
WEATHER_CONDITIONS = ['orin', 'fog', 'rain', 'snow']

# 快速训练配置 - 空间分离
#SEQUENCES = ["00-10-03-27","01-10-03-42","02-10-03-14","04-09-30-16","05-09-30-18","06-09-30-20","07-09-30-27","08-09-30-28","09-09-30-33","10-09-30-34"]
#SEQUENCES = ["01-10-03-42","02-10-03-14","04-09-30-16","07-09-30-27","08-09-30-28","09-09-30-33","10-09-30-34"]
SEQUENCES = ["01-10-03-42","02-10-03-14","04-09-30-16","08-09-30-28","09-09-30-33","10-09-30-34"]
TEST_RATIO = 0.2  # 前20%累积距离作为测试集
TRAIN_RATIO = 0.8  # 后80%累积距离作为训练集

def calculate_cumulative_distance(positions):
    """
    计算轨迹的累积距离（单位：米）

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


def calculate_sequence_split_threshold(base_path, seq):
    """
    对单个序列计算空间分割阈值（按累积距离）

    使用orin天气条件计算分割的时间戳阈值
    返回基于时间戳的分割阈值
    """
    print(f"\n计算序列 {seq} 的分割阈值...")

    # 使用orin天气作为参考轨迹（因为它有完整数据）
    seq_weather = f"{seq}-orin"
    poses_path = os.path.join(base_path, DATASET_FOLDER, seq_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        print(f"  ⚠ 未找到参考文件: {poses_path}")
        return None

    # 读取poses: timestamp easting northing ...（共31列，我们只需要前3列）
    # 注意：poses.txt中UTM坐标，单位是米
    df = pd.read_csv(poses_path, sep=r"\s+", header=None,
                     usecols=[0, 1, 2],
                     names=['timestamp', 'easting', 'northing'])

    # 按timestamp排序（确保轨迹顺序）
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 提取位置信息（GPS坐标，度）
    # 注意顺序：[northing, easting] 即 [纬度, 经度]
    positions = df[['northing', 'easting']].values

    # 计算累积距离
    cum_dist = calculate_cumulative_distance(positions)
    total_distance = cum_dist[-1]

    # 计算20%的累积距离阈值
    threshold_distance = total_distance * TEST_RATIO

    # 找到最接近阈值的点的索引
    split_idx = np.searchsorted(cum_dist, threshold_distance)

    # 获取分割点的时间戳
    split_timestamp = int(df.iloc[split_idx]['timestamp'])

    print(f"  总轨迹长度: {total_distance:.2f}m")
    print(f"  20%阈值距离: {threshold_distance:.2f}m")
    print(f"  分割点索引: {split_idx} / {len(df)}")
    print(f"  分割时间戳: {split_timestamp}")
    print(f"  测试集: timestamp < {split_timestamp}")
    print(f"  训练集: timestamp >= {split_timestamp}")

    # 返回基于时间戳的分割信息
    return {
        'split_timestamp': split_timestamp,
        'split_idx': split_idx,  # 保留用于兼容性
        'threshold_distance': threshold_distance,
        'total_distance': total_distance,
        'total_points': len(df)
    }


def load_sequence_data(base_path, seq, weather, split='train', split_info=None):
    """
    加载单个序列和天气条件的数据（按时间戳分离）

    参数:
    - base_path: 数据集根目录
    - seq: 序列名称
    - weather: 天气条件
    - split: 'train' (timestamp >= split_timestamp) 或 'test' (timestamp < split_timestamp)
    - split_info: 分割信息字典，包含split_timestamp
    """
    seq_weather = f"{seq}-{weather}"
    poses_path = os.path.join(base_path, DATASET_FOLDER, seq_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        return None

    # 读取poses: timestamp easting northing ...（共31列，我们只需要前3列）
    # 注意：poses.txt中UTM坐标，单位是米
    df_locations = pd.read_csv(poses_path, sep=r"\s+", header=None,
                               usecols=[0, 1, 2],
                               names=['timestamp', 'easting', 'northing'])

    # 按timestamp排序（确保与参考轨迹一致）
    df_locations = df_locations.sort_values('timestamp').reset_index(drop=True)

    # 基于时间戳分离
    if split_info is not None and 'split_timestamp' in split_info:
        split_timestamp = split_info['split_timestamp']

        if split == 'test':
            # timestamp < split_timestamp - 测试集
            df_locations = df_locations[df_locations['timestamp'] < split_timestamp].reset_index(drop=True)
        elif split == 'train':
            # timestamp >= split_timestamp - 训练集
            df_locations = df_locations[df_locations['timestamp'] >= split_timestamp].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}")

    # 构建BEV图像文件路径
    # Format: KITTI_org_bev/<seq>-<weather>/bev_512/<timestamp>.png
    df_locations['file'] = (
        DATASET_FOLDER + '/' + seq_weather + '/' +
        BEV_SUBFOLDER + '/' +
        df_locations['timestamp'].astype(str).str.zfill(10) + '.png'
    )

    # 过滤掉不存在的文件
    valid_indices = []
    for idx, row in df_locations.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_indices.append(idx)

    df_locations = df_locations.loc[valid_indices].reset_index(drop=True)

    return df_locations


def generate_symmetric_queries(base_path, sequences, ind_nn_r, ind_r_r, split='train'):
    """
    生成对称的查询元组（空间分离）

    参数:
    - base_path: 数据集根目录
    - sequences: 序列列表
    - ind_nn_r: 正样本距离阈值（米）
    - ind_r_r: 负样本距离阈值（米）
    - split: 'train' 或 'test'
    """

    print(f"  正样本距离阈值: {ind_nn_r}m")
    print(f"  负样本距离阈值: {ind_r_r}m")

    # 第一步：计算每个序列的分割阈值
    print(f"\n计算各序列的分割阈值...")
    split_info_dict = {}
    for seq in sequences:
        split_info = calculate_sequence_split_threshold(base_path, seq)
        if split_info is not None:
            split_info_dict[seq] = split_info

    if len(split_info_dict) == 0:
        print("错误: 无法计算任何序列的分割阈值！")
        return {}

    # 第二步：加载所有数据并分配全局ID
    all_data = []
    global_id = 0
    id_to_info = {}

    print(f"\n加载数据（{split} split - {'时间戳 < split_timestamp' if split=='test' else '时间戳 >= split_timestamp'}）...")
    for seq in sequences:
        if seq not in split_info_dict:
            print(f"  跳过序列 {seq}（无分割信息）")
            continue

        split_info = split_info_dict[seq]

        for weather in WEATHER_CONDITIONS:
            df = load_sequence_data(base_path, seq, weather, split=split, split_info=split_info)
            if df is None or len(df) == 0:
                continue

            print(f"  {seq}-{weather}: {len(df)} 个BEV图像")

            for local_idx, row in df.iterrows():
                # 存储GPS坐标（度）: [northing, easting] = [纬度, 经度]
                position = np.array([row['northing'], row['easting']], dtype=np.float64)

                id_to_info[global_id] = {
                    'seq': seq,
                    'weather': weather,
                    'local_idx': local_idx,
                    'position': position,
                    'file': row['file'],
                    'timestamp': int(row['timestamp'])
                }
                all_data.append((global_id, position, seq, weather))
                global_id += 1

    print(f"总共加载 {len(all_data)} 个BEV图像")

    if len(all_data) == 0:
        print("错误: 没有加载到任何数据！")
        return {}

    # 第三步：构建全局KDTree
    print("\n构建空间索引...")
    positions = np.array([item[1] for item in all_data])
    global_kdtree = KDTree(positions)

    # 第四步：查找所有正样本对（对称）
    print("\n查找正样本对（只考虑不同天气）...")
    positive_pairs = set()

    for gid, pos, seq, weather in all_data:
        # 使用米作为半径进行查询（因为positions是UTM坐标，单位是米）
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

    # 第五步：构建对称的正样本字典
    print("\n构建对称正样本映射...")
    positives_dict = defaultdict(set)

    for id1, id2 in positive_pairs:
        positives_dict[id1].add(id2)
        positives_dict[id2].add(id1)

    # 第六步：为每个查询构建非负样本集
    print("\n构建非负样本集...")
    non_negatives_dict = {}

    for gid in range(len(all_data)):
        pos = id_to_info[gid]['position']
        # 使用米作为半径进行查询（因为positions是UTM坐标，单位是米）
        neighbors = global_kdtree.query_radius([pos], r=ind_r_r)[0]
        non_negatives_dict[gid] = set(neighbors) - {gid}

    # 第七步：创建查询元组
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
            position=info['position']  # [northing, easting] GPS坐标（度）
        )

        query.weather = info['weather']
        query.sequence = info['seq']

        queries[gid] = query

    # 第八步：验证对称性
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
    positions = np.array([q.position[:2] for q in queries.values()])  # 只取northing, easting

    print(f"\n{'='*70}")
    print(f"{title} 统计")
    print(f"{'='*70}")
    print(f"总查询数: {len(queries):,d}")

    print("\n空间范围 (GPS坐标):")
    print(f"  Northing (纬度): [{positions[:, 0].min():.6f}°, {positions[:, 0].max():.6f}°]")
    print(f"  Easting (经度):  [{positions[:, 1].min():.6f}°, {positions[:, 1].max():.6f}°]")

    print("\n天气条件分布:")
    for weather in sorted(weather_stats.keys()):
        count = weather_stats[weather]
        print(f"  {weather:8s}: {count:6,d} ({count*100.0/len(queries):5.1f}%)")

    print("\n序列分布:")
    for seq in sorted(sequence_stats.keys()):
        count = sequence_stats[seq]
        print(f"  {seq}: {count:6,d} ({count*100.0/len(queries):5.1f}%)")

    print(f"\n正样本统计:")
    if len(pos_counts) > 0:
        print(f"  平均: {np.mean(pos_counts):.1f}")
        print(f"  中位数: {np.median(pos_counts):.0f}")
        print(f"  范围: [{min(pos_counts)}, {max(pos_counts)}]")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='生成KITTI BEV图像训练数据集（空间分离版本）')
    parser.add_argument('--dataset_root', type=str, default='/home/cxw/pro/1031-all',
                       help='数据集根目录路径')
    parser.add_argument('--ind_nn_r', type=float, default=5,
                       help='正样本距离阈值(米)')
    parser.add_argument('--ind_r_r', type=float, default=50,
                       help='负样本距离阈值(米)')

    args = parser.parse_args()

    print(f'\n{"="*70}')
    print(f'KITTI BEV图像训练数据集生成（空间分离版本）')
    print(f'{"="*70}')
    print(f'数据集根目录: {args.dataset_root}')
    print(f'使用序列: {len(SEQUENCES)} 个 - {SEQUENCES}')
    print(f'划分方式: 基于时间戳分离（使用orin天气条件确定分割点）')
    print(f'           前{int(TEST_RATIO*100)}%累积距离对应的时间戳作为分割点')
    print(f'正样本阈值: {args.ind_nn_r}m')
    print(f'负样本阈值: {args.ind_r_r}m')

    # 生成训练集（时间戳 >= split_timestamp）
    print("\n" + "="*70)
    print("生成训练集（时间戳 >= 分割时间戳）")
    print("="*70)

    train_queries = generate_symmetric_queries(
        args.dataset_root, SEQUENCES,
        args.ind_nn_r, args.ind_r_r,
        split='train'
    )

    # 保存训练集
    output_path = os.path.join(args.dataset_root, 'data')
    os.makedirs(output_path, exist_ok=True)

    train_file = os.path.join(output_path, "kitti_bev_training_queries.pickle")
    with open(train_file, 'wb') as f:
        pickle.dump(train_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存训练集: {train_file}")

    analyze_queries(train_queries, "训练集（时间戳 >= 分割时间戳）")

    # 生成测试集（时间戳 < split_timestamp）
    print("\n" + "="*70)
    print("生成测试集（时间戳 < 分割时间戳）")
    print("="*70)

    test_queries = generate_symmetric_queries(
        args.dataset_root, SEQUENCES,
        args.ind_nn_r, args.ind_r_r,
        split='test'
    )

    # 保存测试集
    test_file = os.path.join(output_path, "kitti_bev_test_queries.pickle")
    with open(test_file, 'wb') as f:
        pickle.dump(test_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存测试集: {test_file}")

    analyze_queries(test_queries, "测试集（时间戳 < 分割时间戳）")

    # 总结
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    total_queries = len(train_queries) + len(test_queries)
    if total_queries > 0:
        print(f"训练集: {len(train_queries):,d} ({len(train_queries)/total_queries*100:.1f}%)")
        print(f"测试集: {len(test_queries):,d} ({len(test_queries)/total_queries*100:.1f}%)")
    else:
        print(f"训练集: {len(train_queries):,d}")
        print(f"测试集: {len(test_queries):,d}")
    print(f"总计: {total_queries:,d}")
    print(f"\n数据格式: BEV PNG图像 (512×512×3)")
    print(f"划分方式: 基于时间戳分离（使用orin天气确定分割点）")
    print(f"优势: 避免了不同天气条件数据不一致导致的分割错误")


if __name__ == '__main__':
    main()
