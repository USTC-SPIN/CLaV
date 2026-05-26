#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI BEV图像评估文件生成脚本（空间分离版本）
对每个序列单独按累积距离划分，前20%累积距离对应的时间戳作为分割点
基于时间戳分离：timestamp < split_timestamp作为评估集
生成用于评估的database和query文件

适配自 generate_kitti_part_evaluation_sets_spatial.py，用于BEV PNG图像
"""

import numpy as np
import os
import pandas as pd
import pickle
import argparse
import sys

# 添加父目录到Python路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# KITTI BEV数据集配置
DATASET_FOLDER = "kitti-bev-skip"
BEV_SUBFOLDER = "bev_448"
POSES_FILENAME = "poses.txt"

# 天气条件列表
WEATHER_CONDITIONS = ['orin', 'fog', 'rain', 'snow']

# 评估配置 - 使用累积距离的前20%
# EVAL_SEQUENCES = ["00-10-03-27","01-10-03-42","02-10-03-14","04-09-30-16","05-09-30-18","06-09-30-20","07-09-30-27","08-09-30-28","09-09-30-33","10-09-30-34"]
#EVAL_SEQUENCES  = ["01-10-03-42","02-10-03-14","04-09-30-16","07-09-30-27","08-09-30-28","09-09-30-33","10-09-30-34"]
EVAL_SEQUENCES  = ["01-10-03-42","02-10-03-14","04-09-30-16","08-09-30-28","09-09-30-33","10-09-30-34"]
TEST_RATIO = 0.2  # 前20%累积距离作为测试/评估集


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

    return {
        'split_timestamp': split_timestamp,
        'split_idx': split_idx,  # 保留用于兼容性
        'threshold_distance': threshold_distance,
        'total_distance': total_distance,
        'total_points': len(df)
    }


def load_sequence_data(base_path, seq, weather, split_info):
    """
    加载单个序列和天气条件的数据（timestamp < split_timestamp用于评估）

    参数:
    - base_path: 数据集根目录
    - seq: 序列名称
    - weather: 天气条件
    - split_info: 分割信息字典，包含split_timestamp

    返回timestamp < split_timestamp的数据作为评估集
    """
    seq_weather = f"{seq}-{weather}"
    poses_path = os.path.join(base_path, DATASET_FOLDER, seq_weather, POSES_FILENAME)

    if not os.path.exists(poses_path):
        print(f"  ⚠ 未找到: {poses_path}")
        return None

    # 读取poses: timestamp easting northing ...（共31列，我们只需要前3列）
    # 注意：poses.txt中UTM坐标，单位是米
    df_locations = pd.read_csv(poses_path, sep=r"\s+", header=None,
                               usecols=[0, 1, 2],
                               names=['timestamp', 'easting', 'northing'])

    # 按timestamp排序（确保与参考轨迹一致）
    df_locations = df_locations.sort_values('timestamp').reset_index(drop=True)

    # 基于时间戳分离：只保留timestamp < split_timestamp作为评估集
    if 'split_timestamp' in split_info:
        split_timestamp = split_info['split_timestamp']
        df_locations = df_locations[df_locations['timestamp'] < split_timestamp].reset_index(drop=True)
    else:
        # 向后兼容：如果没有split_timestamp，使用split_idx
        split_idx = split_info['split_idx']
        df_locations = df_locations.iloc[:split_idx].reset_index(drop=True)

    if len(df_locations) == 0:
        print(f"  ⚠ {seq}-{weather} 无数据")
        return None

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


def generate_evaluation_sets(base_path):
    """
    生成KITTI BEV评估集（使用各序列前20%累积距离）
    Database: orin天气作为参考数据库
    Query: 其他天气(fog, rain, snow)作为查询集
    """
    print("\n" + "="*70)
    print("生成KITTI BEV评估文件（空间分离版本）")
    print("="*70)

    # 计算每个序列的分割阈值
    print("\n计算各序列的分割阈值...")
    split_info_dict = {}
    for seq in EVAL_SEQUENCES:
        print(f"\n序列 {seq}:")
        split_info = calculate_sequence_split_threshold(base_path, seq)
        if split_info is not None:
            split_info_dict[seq] = split_info
            print(f"  总轨迹长度: {split_info['total_distance']:.2f}m")
            print(f"  20%阈值距离: {split_info['threshold_distance']:.2f}m")
            print(f"  分割时间戳: {split_info['split_timestamp']}")
            print(f"  评估集: timestamp < {split_info['split_timestamp']}")

    if len(split_info_dict) == 0:
        raise ValueError("无法计算任何序列的分割阈值")

    # 加载所有序列的orin天气作为database（timestamp < split_timestamp）
    print(f"\n加载Database (所有序列的orin天气 - timestamp < 分割时间戳)...")
    all_db_files = []
    all_db_positions = []

    for seq in EVAL_SEQUENCES:
        if seq not in split_info_dict:
            print(f"  ⚠ 跳过序列 {seq}")
            continue

        db_df = load_sequence_data(base_path, seq, 'orin', split_info_dict[seq])
        if db_df is None:
            print(f"  ⚠ 跳过序列 {seq}")
            continue

        print(f"  {seq}-orin: {len(db_df)} 个BEV图像")
        all_db_files.extend(db_df['file'].tolist())
        # 存储GPS坐标（度）: [northing, easting] = [纬度, 经度]
        all_db_positions.append(np.column_stack([
            db_df['northing'].values,
            db_df['easting'].values
        ]))

    if len(all_db_files) == 0:
        raise ValueError("无法加载任何database数据")

    all_db_positions = np.vstack(all_db_positions)
    print(f"  Database总计: {len(all_db_files)} 个BEV图像")
    print(f"  Database空间范围 (GPS坐标):")
    print(f"    Northing (纬度): [{all_db_positions[:, 0].min():.6f}°, {all_db_positions[:, 0].max():.6f}°]")
    print(f"    Easting (经度):  [{all_db_positions[:, 1].min():.6f}°, {all_db_positions[:, 1].max():.6f}°]")

    # 构建database字典 - 标准格式 {idx: {'query': file, 'position': [northing,easting], 'timestamp': idx}}
    database_dict = {}
    for idx, (file_path, position) in enumerate(zip(all_db_files, all_db_positions)):
        database_dict[idx] = {
            'query': file_path,
            'position': position.tolist(),  # [northing, easting] GPS坐标（度）
            'timestamp': idx,
            'northing': position[0],  # 纬度
            'easting': position[1]    # 经度
        }

    # 为每种天气条件创建query字典
    print(f"\n生成Query字典 (所有天气条件 - timestamp < 分割时间戳)...")

    # 存储所有天气条件的query字典
    all_query_dicts = {}

    # 处理每种天气条件
    for weather in WEATHER_CONDITIONS:
        print(f"\n处理 {weather} 天气...")
        query_dict = {}
        idx = 0

        # 收集该天气条件下所有序列的数据
        for seq in EVAL_SEQUENCES:
            if seq not in split_info_dict:
                print(f"  ⚠ 跳过序列 {seq}-{weather}")
                continue

            query_df = load_sequence_data(base_path, seq, weather, split_info_dict[seq])
            if query_df is None:
                print(f"  ⚠ 跳过序列 {seq}-{weather}")
                continue

            print(f"  {seq}-{weather}: {len(query_df)} 个BEV图像")

            # 添加每个文件到query字典
            for _, row in query_df.iterrows():
                query_dict[idx] = {
                    'query': row['file'],
                    'position': [row['northing'], row['easting']],  # GPS坐标（度）
                    'timestamp': idx,
                    'northing': row['northing'],  # 纬度
                    'easting': row['easting'],    # 经度
                    'weather': weather
                }
                idx += 1

        print(f"  {weather}总计: {idx} 个BEV图像")
        all_query_dicts[weather] = query_dict

    # 保存文件
    output_dir = os.path.join(base_path, 'data')
    os.makedirs(output_dir, exist_ok=True)

    print("\n保存评估文件...")

    # 生成所有评估文件对（normal + weather-specific）
    saved_files = []

    # 1. 正常评估：kitti_bev_evaluation_database.pickle 和 kitti_bev_evaluation_query.pickle
    # 这两个都使用orin数据
    db_file = os.path.join(output_dir, 'kitti_bev_evaluation_database.pickle')
    query_file = os.path.join(output_dir, 'kitti_bev_evaluation_query.pickle')

    with open(db_file, 'wb') as f:
        pickle.dump(database_dict, f)
    print(f"  ✓ Database (normal): {db_file}")
    saved_files.append(db_file)

    # Normal query也使用orin数据
    with open(query_file, 'wb') as f:
        pickle.dump(all_query_dicts['orin'], f)
    print(f"  ✓ Query (normal): {query_file}")
    saved_files.append(query_file)

    # 2. 天气特定评估文件
    for weather in ['fog', 'rain', 'snow']:
        # Database文件（仍使用orin作为参考）
        weather_db_file = os.path.join(output_dir, f'kitti_bev_{weather}_evaluation_database.pickle')
        with open(weather_db_file, 'wb') as f:
            pickle.dump(database_dict, f)  # 所有天气条件都使用同样的database
        print(f"  ✓ Database ({weather}): {weather_db_file}")
        saved_files.append(weather_db_file)

        # Query文件（使用特定天气的数据）
        weather_query_file = os.path.join(output_dir, f'kitti_bev_{weather}_evaluation_query.pickle')
        with open(weather_query_file, 'wb') as f:
            pickle.dump(all_query_dicts[weather], f)
        print(f"  ✓ Query ({weather}): {weather_query_file}")
        saved_files.append(weather_query_file)

    # 打印统计信息
    print("\n" + "="*70)
    print("评估集统计")
    print("="*70)
    print(f"Database (参考):")
    print(f"  orin: {len(database_dict)} 个BEV图像")
    print(f"\nQuery (查询):")
    for weather, query_dict in all_query_dicts.items():
        print(f"  {weather}: {len(query_dict)} 个BEV图像")

    # 验证数据一致性
    print("\n验证数据一致性...")
    expected_count = len(database_dict)
    all_consistent = True

    for weather, query_dict in all_query_dicts.items():
        if len(query_dict) != expected_count:
            print(f"  ⚠ {weather} 数量不一致: {len(query_dict)} vs {expected_count}")
            all_consistent = False

    if all_consistent:
        print("  ✓ 所有天气条件的BEV图像数量一致")

    # 打印分割信息
    print("\n各序列分割信息:")
    for seq in EVAL_SEQUENCES:
        if seq in split_info_dict:
            info = split_info_dict[seq]
            print(f"  {seq}:")
            print(f"    总点数: {info['total_points']}")
            print(f"    评估集: 前 {info['split_idx']} 点 ({info['split_idx']/info['total_points']*100:.1f}%)")
            print(f"    训练集: 后 {info['total_points']-info['split_idx']} 点 ({(info['total_points']-info['split_idx'])/info['total_points']*100:.1f}%)")
            print(f"    轨迹长度: {info['total_distance']:.2f}m")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"生成的评估文件 ({len(saved_files)}个):")
    for f in saved_files:
        print(f"  - {os.path.basename(f)}")

    print(f"\n数据来源:")
    print(f"  - 使用所有{len(EVAL_SEQUENCES)}个序列的前{int(TEST_RATIO*100)}%累积距离")
    print(f"  - Database: 始终使用orin天气作为参考")
    print(f"  - Query: 每种天气条件单独评估")
    print(f"  - 数据格式: BEV PNG图像 (448×512×3)")
    print(f"  - 划分方式: 每个序列单独按轨迹累积距离划分")
    print(f"  - 与训练集(后{int((1-TEST_RATIO)*100)}%)在时间和空间上分离")

    return database_dict, all_query_dicts


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成KITTI BEV图像评估文件（空间分离版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/cxw/pro/1031-all',
        help='数据集根目录'
    )

    args = parser.parse_args()

    # 生成评估集
    generate_evaluation_sets(args.dataset_root)


if __name__ == "__main__":
    main()
