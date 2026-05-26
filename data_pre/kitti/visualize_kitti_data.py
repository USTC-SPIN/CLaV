#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI数据集可视化和统计脚本
可视化空间分割情况并统计数据数量

创建时间: 2025-11-06
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
import pickle
import argparse
import os
from collections import defaultdict, Counter
from pathlib import Path


# TrainingTuple类定义（用于加载pickle文件）
class TrainingTuple:
    """训练样本元组"""
    def __init__(self, id, rel_scan_filepath, positives, non_negatives, position):
        self.id = id
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


def load_pickle_file(filepath):
    """加载pickle文件"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_training_data(train_file, test_file):
    """分析训练和测试数据"""
    print("\n" + "="*70)
    print("训练/测试数据分析")
    print("="*70)

    # 加载数据
    train_data = load_pickle_file(train_file)
    test_data = load_pickle_file(test_file)

    if train_data is None or test_data is None:
        return None, None

    # 统计基本信息
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_data):,} 样本")
    print(f"  测试集: {len(test_data):,} 样本")
    print(f"  总计: {len(train_data) + len(test_data):,} 样本")
    print(f"  训练/测试比例: {len(train_data)/(len(train_data)+len(test_data))*100:.1f}% / {len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")

    # 分析天气分布
    train_weather = Counter()
    test_weather = Counter()

    for sample in train_data.values():
        if hasattr(sample, 'weather'):
            train_weather[sample.weather] += 1

    for sample in test_data.values():
        if hasattr(sample, 'weather'):
            test_weather[sample.weather] += 1

    print(f"\n天气条件分布:")
    print(f"  训练集:")
    for weather in sorted(train_weather.keys()):
        print(f"    {weather}: {train_weather[weather]:,} ({train_weather[weather]/len(train_data)*100:.1f}%)")

    print(f"  测试集:")
    for weather in sorted(test_weather.keys()):
        print(f"    {weather}: {test_weather[weather]:,} ({test_weather[weather]/len(test_data)*100:.1f}%)")

    # 分析序列分布
    train_sequences = Counter()
    test_sequences = Counter()

    for sample in train_data.values():
        if hasattr(sample, 'sequence'):
            train_sequences[sample.sequence] += 1

    for sample in test_data.values():
        if hasattr(sample, 'sequence'):
            test_sequences[sample.sequence] += 1

    print(f"\n序列分布:")
    print(f"  训练集:")
    for seq in sorted(train_sequences.keys()):
        print(f"    {seq}: {train_sequences[seq]:,} ({train_sequences[seq]/len(train_data)*100:.1f}%)")

    print(f"  测试集:")
    for seq in sorted(test_sequences.keys()):
        print(f"    {seq}: {test_sequences[seq]:,} ({test_sequences[seq]/len(test_data)*100:.1f}%)")

    # 分析正负样本
    train_positives = []
    train_non_negatives = []
    test_positives = []
    test_non_negatives = []

    for sample in train_data.values():
        if hasattr(sample, 'positives'):
            train_positives.append(len(sample.positives))
        if hasattr(sample, 'non_negatives'):
            train_non_negatives.append(len(sample.non_negatives))

    for sample in test_data.values():
        if hasattr(sample, 'positives'):
            test_positives.append(len(sample.positives))
        if hasattr(sample, 'non_negatives'):
            test_non_negatives.append(len(sample.non_negatives))

    if train_positives:
        print(f"\n正样本统计:")
        print(f"  训练集: 平均 {np.mean(train_positives):.1f}, 中位数 {np.median(train_positives):.0f}, 范围 [{min(train_positives)}, {max(train_positives)}]")
        print(f"  测试集: 平均 {np.mean(test_positives):.1f}, 中位数 {np.median(test_positives):.0f}, 范围 [{min(test_positives)}, {max(test_positives)}]")

    if train_non_negatives:
        print(f"\n非负样本统计:")
        print(f"  训练集: 平均 {np.mean(train_non_negatives):.1f}, 中位数 {np.median(train_non_negatives):.0f}, 范围 [{min(train_non_negatives)}, {max(train_non_negatives)}]")
        print(f"  测试集: 平均 {np.mean(test_non_negatives):.1f}, 中位数 {np.median(test_non_negatives):.0f}, 范围 [{min(test_non_negatives)}, {max(test_non_negatives)}]")

    return train_data, test_data


def analyze_denoising_data(denoising_file):
    """分析去噪配对数据"""
    print("\n" + "="*70)
    print("去噪配对数据分析")
    print("="*70)

    data = load_pickle_file(denoising_file)
    if data is None:
        return None

    # 基本统计
    print(f"\n数据集大小:")
    print(f"  训练集: {len(data['train']):,} pairs")
    print(f"  验证集: {len(data['val']):,} pairs")
    if 'test' in data:
        print(f"  测试集: {len(data['test']):,} pairs")

    total = len(data['train']) + len(data['val']) + len(data.get('test', []))
    print(f"  总计: {total:,} pairs")

    # 天气分布
    for split_name in ['train', 'val', 'test']:
        if split_name not in data:
            continue

        weather_counts = Counter()
        for _, _, weather in data[split_name]:
            weather_counts[weather] += 1

        print(f"\n{split_name}集天气分布:")
        for weather in sorted(weather_counts.keys()):
            print(f"  {weather}: {weather_counts[weather]:,} ({weather_counts[weather]/len(data[split_name])*100:.1f}%)")

    # 元数据
    if 'metadata' in data:
        metadata = data['metadata']
        print(f"\n元数据信息:")
        print(f"  坐标系统: {metadata.get('coordinate_system', '未知')}")
        print(f"  划分方法: {metadata.get('split_method', '未知')}")
        print(f"  序列数: {len(metadata.get('sequences', []))}")

        # 如果有完整信息，分析序列分布
        if 'train_pairs_full' in metadata:
            train_full = metadata['train_pairs_full']
            seq_counts = Counter()
            for pair in train_full:
                seq_counts[pair.get('sequence', 'unknown')] += 1

            print(f"\n训练集序列分布:")
            for seq in sorted(seq_counts.keys()):
                print(f"  {seq}: {seq_counts[seq]:,}")

    return data


def analyze_evaluation_data(db_file, query_file):
    """分析评估数据"""
    print("\n" + "="*70)
    print("评估数据分析")
    print("="*70)

    db_data = load_pickle_file(db_file)
    query_data = load_pickle_file(query_file)

    if db_data is None or query_data is None:
        return None, None

    print(f"\n数据集大小:")
    print(f"  Database: {len(db_data):,} 个样本")
    print(f"  Query: {len(query_data):,} 个样本")

    # 分析位置范围
    if db_data:
        db_positions = np.array([item['position'] for item in db_data.values()])
        print(f"\nDatabase位置范围 (UTM):")
        print(f"  Northing: [{db_positions[:, 0].min():.2f}, {db_positions[:, 0].max():.2f}] m")
        print(f"  Easting: [{db_positions[:, 1].min():.2f}, {db_positions[:, 1].max():.2f}] m")

    if query_data:
        query_positions = np.array([item['position'] for item in query_data.values()])
        print(f"\nQuery位置范围 (UTM):")
        print(f"  Northing: [{query_positions[:, 0].min():.2f}, {query_positions[:, 0].max():.2f}] m")
        print(f"  Easting: [{query_positions[:, 1].min():.2f}, {query_positions[:, 1].max():.2f}] m")

    return db_data, query_data


def visualize_spatial_split(train_data, test_data, output_dir):
    """可视化空间分割"""
    print("\n生成空间分割可视化...")

    # 收集位置数据
    train_positions = []
    test_positions = []

    # 按序列组织数据
    train_by_seq = defaultdict(list)
    test_by_seq = defaultdict(list)

    for sample in train_data.values():
        if hasattr(sample, 'position') and hasattr(sample, 'sequence'):
            pos = sample.position
            train_positions.append(pos)
            train_by_seq[sample.sequence].append(pos)

    for sample in test_data.values():
        if hasattr(sample, 'position') and hasattr(sample, 'sequence'):
            pos = sample.position
            test_positions.append(pos)
            test_by_seq[sample.sequence].append(pos)

    train_positions = np.array(train_positions)
    test_positions = np.array(test_positions)

    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 总体空间分布
    ax1 = fig.add_subplot(gs[0, :2])
    if len(train_positions) > 0:
        ax1.scatter(train_positions[:, 1], train_positions[:, 0],
                   c='blue', alpha=0.3, s=1, label=f'训练集 ({len(train_positions)})')
    if len(test_positions) > 0:
        ax1.scatter(test_positions[:, 1], test_positions[:, 0],
                   c='red', alpha=0.3, s=1, label=f'测试集 ({len(test_positions)})')

    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_title('空间分割总览（UTM坐标）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. 数据分布柱状图
    ax2 = fig.add_subplot(gs[0, 2])

    # 统计每个序列的数据量
    sequences = sorted(set(list(train_by_seq.keys()) + list(test_by_seq.keys())))
    train_counts = [len(train_by_seq.get(seq, [])) for seq in sequences]
    test_counts = [len(test_by_seq.get(seq, [])) for seq in sequences]

    x = np.arange(len(sequences))
    width = 0.35

    ax2.bar(x - width/2, train_counts, width, label='训练集', color='blue', alpha=0.7)
    ax2.bar(x + width/2, test_counts, width, label='测试集', color='red', alpha=0.7)

    ax2.set_xlabel('序列')
    ax2.set_ylabel('样本数')
    ax2.set_title('各序列数据分布')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split('-')[0] for s in sequences], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 每个序列的轨迹（显示所有序列）
    all_sequences = sorted(set(list(train_by_seq.keys()) + list(test_by_seq.keys())))

    # 在2x3的网格中显示所有6个序列
    for idx, seq in enumerate(all_sequences[:6]):
        row = 1 + idx // 3  # 第2行或第3行
        col = idx % 3       # 第0、1或2列
        ax = fig.add_subplot(gs[row, col])

        # 训练集轨迹
        if seq in train_by_seq and len(train_by_seq[seq]) > 0:
            train_seq_pos = np.array(train_by_seq[seq])
            ax.plot(train_seq_pos[:, 1], train_seq_pos[:, 0],
                   'b-', alpha=0.5, linewidth=1, label='训练')
            ax.scatter(train_seq_pos[:, 1], train_seq_pos[:, 0],
                      c='blue', s=2, alpha=0.5)

        # 测试集轨迹
        if seq in test_by_seq and len(test_by_seq[seq]) > 0:
            test_seq_pos = np.array(test_by_seq[seq])
            ax.plot(test_seq_pos[:, 1], test_seq_pos[:, 0],
                   'r-', alpha=0.5, linewidth=1, label='测试')
            ax.scatter(test_seq_pos[:, 1], test_seq_pos[:, 0],
                      c='red', s=2, alpha=0.5)

        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title(f'序列 {seq}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 保存图片
    output_file = os.path.join(output_dir, 'kitti_spatial_split.png')
    plt.suptitle('KITTI数据集空间分割可视化', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  保存到: {output_file}")

    return fig


def visualize_cumulative_distance(base_path, sequences, output_dir):
    """可视化累积距离分割"""
    print("\n生成累积距离可视化...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, seq in enumerate(sequences[:6]):
        ax = axes[idx]

        # 读取poses文件
        poses_path = os.path.join(base_path, "kitti-bev-skip", f"{seq}-orin", "poses.txt")
        if not os.path.exists(poses_path):
            ax.text(0.5, 0.5, f'序列 {seq}\n数据不存在',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 读取数据
        df = pd.read_csv(poses_path, sep=r"\s+", header=None,
                        usecols=[0, 1, 2],
                        names=['timestamp', 'easting', 'northing'])

        positions = df[['northing', 'easting']].values

        # 计算累积距离
        diffs = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.zeros(len(positions))
        cum_dist[1:] = np.cumsum(distances)

        # 找到20%分割点
        total_dist = cum_dist[-1]
        threshold_dist = total_dist * 0.2
        split_idx = np.searchsorted(cum_dist, threshold_dist)

        # 绘制累积距离曲线
        indices = np.arange(len(cum_dist))
        ax.plot(indices, cum_dist, 'b-', linewidth=2)

        # 标记分割点
        ax.axvline(x=split_idx, color='red', linestyle='--', linewidth=2,
                  label=f'20%分割点 (idx={split_idx})')
        ax.axhline(y=threshold_dist, color='red', linestyle=':', alpha=0.5)

        # 填充区域
        ax.fill_between(indices[:split_idx], 0, cum_dist[:split_idx],
                       alpha=0.3, color='red', label='测试集(20%)')
        ax.fill_between(indices[split_idx:], 0, cum_dist[split_idx:],
                       alpha=0.3, color='blue', label='训练集(80%)')

        ax.set_xlabel('帧索引')
        ax.set_ylabel('累积距离 (m)')
        ax.set_title(f'序列 {seq}\n总距离: {total_dist:.1f}m')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('KITTI数据集累积距离分割', fontsize=16)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'kitti_cumulative_distance.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  保存到: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='KITTI数据集可视化和统计',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/users/cxw/pro/clav/data',
        help='数据目录路径'
    )

    parser.add_argument(
        '--base_path',
        type=str,
        default='/data/users/cxw/pro/clav',
        help='数据集根目录'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/users/cxw/pro/clav/data',
        help='输出目录'
    )

    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='不生成可视化图表'
    )

    args = parser.parse_args()

    # 数据文件路径
    train_file = os.path.join(args.data_dir, 'kitti_bev_training_queries.pickle')
    test_file = os.path.join(args.data_dir, 'kitti_bev_test_queries.pickle')
    denoising_file = os.path.join(args.data_dir, 'kitti_denoising_tuples.pkl')
    db_file = os.path.join(args.data_dir, 'kitti_bev_evaluation_database.pickle')
    query_file = os.path.join(args.data_dir, 'kitti_bev_evaluation_query.pickle')

    print("="*70)
    print("KITTI数据集可视化和统计分析")
    print("="*70)
    print(f"数据目录: {args.data_dir}")

    # 分析训练/测试数据
    train_data, test_data = analyze_training_data(train_file, test_file)

    # 分析去噪数据
    denoising_data = analyze_denoising_data(denoising_file)

    # 分析评估数据
    db_data, query_data = analyze_evaluation_data(db_file, query_file)

    # 生成可视化
    if not args.no_plot and train_data is not None and test_data is not None:
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 空间分割可视化
        visualize_spatial_split(train_data, test_data, args.output_dir)

        # 累积距离可视化
        sequences = ["01-10-03-42", "02-10-03-14", "04-09-30-16",
                    "08-09-30-28", "09-09-30-33", "10-09-30-34"]
        visualize_cumulative_distance(args.base_path, sequences, args.output_dir)

        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            pass

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


if __name__ == "__main__":
    main()