#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV图片序列可视化脚本
用于按顺序可视化某个序列特定天气的BEV图片，生成视频

使用方法:
    python test/visualize_bev_sequence_20251120_1952.py --weather fog --sequence 08
    python test/visualize_bev_sequence_20251120_1952.py --weather rain --sequence 01 --fps 10
"""

import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


def load_dataset(data_file):
    """加载数据集"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        first_keys = list(data[0].keys())
        if all(isinstance(k, int) for k in first_keys[:5]):
            flattened = []
            for data_set in data:
                for idx in sorted(data_set.keys()):
                    flattened.append(data_set[idx])
            data = flattened
    elif isinstance(data, dict):
        data = [data[i] for i in sorted(data.keys())]

    return data


def get_available_sequences(queries):
    """获取可用的序列列表"""
    sequences = set()
    for q in queries:
        seq = q['query'].split('/')[1].split('-')[0]
        sequences.add(seq)
    return sorted(sequences)


def filter_by_sequence(queries, sequence):
    """按序列号筛选数据"""
    filtered = [q for q in queries if q['query'].split('/')[1].startswith(sequence + '-')]
    return filtered


def sort_by_timestamp(queries):
    """按时间戳排序"""
    return sorted(queries, key=lambda x: x.get('timestamp', 0))


def load_bev_image(image_path, size=(512, 512)):
    """加载BEV图片"""
    try:
        img = Image.open(image_path)
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        # 返回黑色图片
        return np.zeros((*size, 3), dtype=np.uint8)


def create_sequence_video(queries, bev_root, output_path, fps=10, show_info=True):
    """创建序列视频"""
    num_frames = len(queries)

    print(f"Creating video with {num_frames} frames at {fps} fps")
    print(f"Video duration: {num_frames/fps:.1f} seconds")

    # 创建figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)

    # 获取序列信息
    first_query = queries[0]
    seq_info = first_query['query'].split('/')[1]  # e.g., "08-09-30-28-fog"
    seq_parts = seq_info.split('-')
    seq_num = seq_parts[0]
    weather = seq_parts[-1]

    def update_frame(frame_idx):
        ax.clear()

        # 获取当前帧数据
        query = queries[frame_idx]
        img_path = bev_root / query['query']

        # 加载并显示图片
        img = load_bev_image(img_path, size=(512, 512))
        ax.imshow(img)

        # 添加信息
        if show_info:
            position = query.get('position', [0, 0])
            timestamp = query.get('timestamp', frame_idx)

            title = f"Sequence {seq_num} - {weather.upper()}\n"
            title += f"Frame {frame_idx + 1}/{num_frames}\n"
            title += f"Position: ({position[0]:.2f}, {position[1]:.2f})"

            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        ax.axis('off')

        return []

    # 保存视频
    print("Saving video...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)

    with writer.saving(fig, str(output_path), dpi=100):
        for frame_idx in range(num_frames):
            update_frame(frame_idx)
            writer.grab_frame()

            if (frame_idx + 1) % 50 == 0:
                print(f"Progress: {frame_idx + 1}/{num_frames} ({(frame_idx+1)/num_frames*100:.1f}%)")

    print(f"Video saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='BEV图片序列可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化序列08的fog天气
  python test/visualize_bev_sequence_20251120_1952.py --weather fog --sequence 08

  # 可视化序列01的rain天气，帧率设为15
  python test/visualize_bev_sequence_20251120_1952.py --weather rain --sequence 01 --fps 15

  # 列出所有可用序列
  python test/visualize_bev_sequence_20251120_1952.py --weather fog --list
        """
    )

    parser.add_argument('--weather', type=str, required=True,
                       choices=['orin', 'fog', 'rain', 'snow'],
                       help='天气条件')
    parser.add_argument('--sequence', type=str, default=None,
                       help='序列号 (如 "08", "01"等)')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率 (默认: 10)')
    parser.add_argument('--output_dir', type=str, default='outputs/sequence_videos',
                       help='输出目录 (默认: outputs/sequence_videos)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用序列')
    parser.add_argument('--no_info', action='store_true',
                       help='不在视频中显示信息')

    args = parser.parse_args()

    # 路径设置
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / 'data'
    bev_root = project_root  # BEV图片在项目根目录下

    # 选择数据文件
    if args.weather == 'orin':
        query_file = data_root / 'kitti_bev_evaluation_query.pickle'
    else:
        query_file = data_root / f'kitti_bev_{args.weather}_evaluation_query.pickle'

    if not query_file.exists():
        print(f"Error: Query file not found: {query_file}")
        sys.exit(1)

    # 加载数据
    print(f"\nLoading dataset: {args.weather}")
    queries = load_dataset(query_file)
    print(f"Total samples: {len(queries)}")

    # 获取可用序列
    available_sequences = get_available_sequences(queries)
    print(f"Available sequences: {available_sequences}")

    # 如果只是列出序列
    if args.list:
        print("\n序列统计:")
        for seq in available_sequences:
            seq_queries = filter_by_sequence(queries, seq)
            print(f"  Sequence {seq}: {len(seq_queries)} frames")
        sys.exit(0)

    # 检查是否指定了序列
    if not args.sequence:
        print("\nError: Please specify a sequence with --sequence")
        print(f"Available sequences: {available_sequences}")
        sys.exit(1)

    # 筛选序列
    filtered_queries = filter_by_sequence(queries, args.sequence)

    if not filtered_queries:
        print(f"\nError: Sequence '{args.sequence}' not found")
        print(f"Available sequences: {available_sequences}")
        sys.exit(1)

    # 按时间戳排序
    sorted_queries = sort_by_timestamp(filtered_queries)
    print(f"\nSequence {args.sequence}: {len(sorted_queries)} frames")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件名
    output_filename = f"bev_sequence_{args.sequence}_{args.weather}_fps{args.fps}.mp4"
    output_path = output_dir / output_filename

    # 创建视频
    print(f"\nCreating video...")
    create_sequence_video(
        queries=sorted_queries,
        bev_root=bev_root,
        output_path=output_path,
        fps=args.fps,
        show_info=not args.no_info
    )

    print(f"\n✓ Done! Video saved to: {output_path}")
    print(f"  Frames: {len(sorted_queries)}")
    print(f"  FPS: {args.fps}")
    print(f"  Duration: {len(sorted_queries)/args.fps:.1f}s")


if __name__ == '__main__':
    main()
