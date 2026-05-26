#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证去噪配对数据的正确性
"""

import pickle
import os
import numpy as np
import argparse


def verify_denoising_data(pkl_path):
    """验证去噪配对数据文件"""
    print(f"\n{'='*70}")
    print(f"验证去噪配对数据: {pkl_path}")
    print(f"{'='*70}")

    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 基本结构检查
    print("\n1. 数据结构检查:")
    required_keys = ['train', 'val', 'test', 'root_dir', 'metadata']
    for key in required_keys:
        if key in data:
            print(f"  ✓ {key}: 存在")
        else:
            print(f"  ✗ {key}: 缺失")

    # 数据集大小
    print(f"\n2. 数据集大小:")
    print(f"  训练集: {len(data['train'])} pairs")
    print(f"  验证集: {len(data['val'])} pairs")
    print(f"  测试集: {len(data['test'])} pairs")
    print(f"  总计: {len(data['train']) + len(data['val']) + len(data['test'])} pairs")

    # 元数据
    print(f"\n3. 元数据信息:")
    metadata = data.get('metadata', {})
    print(f"  坐标系统: {metadata.get('coordinate_system', '未知')}")
    print(f"  划分方法: {metadata.get('split_method', '未知')}")
    print(f"  测试比例: {metadata.get('test_ratio', '未知')}")
    print(f"  验证比例: {metadata.get('val_ratio', '未知')}")
    print(f"  序列: {metadata.get('sequences', [])}")
    print(f"  天气条件: {metadata.get('weather_conditions', [])}")

    # 检查几个样本的路径
    print(f"\n4. 样本路径检查:")
    for split_name, split_data in [('train', data['train']),
                                    ('val', data['val']),
                                    ('test', data['test'])]:
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"\n  {split_name}集第一个样本:")
            print(f"    噪声路径: {sample[0]}")
            print(f"    清晰路径: {sample[1]}")
            print(f"    天气: {sample[2]}")

    # 检查完整信息（如果存在）
    print(f"\n5. 完整位置信息检查:")
    for split_name in ['train_pairs_full', 'val_pairs_full', 'test_pairs_full']:
        full_data = metadata.get(split_name, [])
        if len(full_data) > 0:
            sample = full_data[0]
            print(f"\n  {split_name.replace('_pairs_full', '')}集第一个样本:")
            print(f"    时间戳: {sample.get('timestamp', '未知')}")
            print(f"    位置: {sample.get('position', '未知')}")
            print(f"    序列: {sample.get('sequence', '未知')}")

            # 检查位置范围（UTM坐标）
            if len(full_data) > 10:
                positions = np.array([p['position'] for p in full_data])
                print(f"  {split_name.replace('_pairs_full', '')}集位置范围:")
                print(f"    Northing: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
                print(f"    Easting: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")

    # 检查文件是否存在（抽样）
    if data.get('root_dir'):
        print(f"\n6. 文件存在性检查（抽样5个）:")
        root_dir = data['root_dir']
        print(f"  根目录: {root_dir}")

        # 如果root_dir不是绝对路径，构建绝对路径
        if not os.path.isabs(root_dir):
            root_dir = os.path.join('/data/users/cxw/pro/clav', root_dir)

        for split_name, split_data in [('train', data['train'][:5]),
                                        ('val', data['val'][:5]),
                                        ('test', data['test'][:5])]:
            if len(split_data) > 0:
                print(f"\n  {split_name}集:")
                for i, (noisy_path, clean_path, weather) in enumerate(split_data):
                    # 路径已包含数据集文件夹，直接从根目录的上一级开始
                    base_dir = os.path.dirname(root_dir)
                    noisy_full = os.path.join(base_dir, noisy_path)
                    clean_full = os.path.join(base_dir, clean_path)

                    noisy_exists = "✓" if os.path.exists(noisy_full) else "✗"
                    clean_exists = "✓" if os.path.exists(clean_full) else "✗"

                    print(f"    样本{i+1}: 噪声[{noisy_exists}] 清晰[{clean_exists}] ({weather})")

    print(f"\n{'='*70}")
    print("验证完成!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='验证去噪配对数据文件')
    parser.add_argument(
        'pkl_file',
        type=str,
        help='要验证的pickle文件路径'
    )

    args = parser.parse_args()

    if not os.path.exists(args.pkl_file):
        print(f"错误: 文件不存在 - {args.pkl_file}")
        return

    verify_denoising_data(args.pkl_file)


if __name__ == "__main__":
    main()