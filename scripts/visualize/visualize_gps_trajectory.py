#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS轨迹地图可视化脚本
在真实地图上展示指定序列的GPS轨迹运动

使用方法:
    python test/visualize_gps_trajectory_map_20251120_2020.py --weather orin --sequence 08
    python test/visualize_gps_trajectory_map_20251120_2020.py --weather fog --sequence 01 --fps 15
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

# 地图库
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily not installed. Run: pip install contextily")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not installed. Run: pip install pyproj")


# KITTI坐标转换器 (UTM Zone 32N -> WGS84 -> Web Mercator)
if HAS_PYPROJ:
    utm_to_wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    wgs84_to_webmercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def utm_to_webmercator(x, y):
    """将UTM坐标转换为Web Mercator投影（用于地图显示）"""
    if not HAS_PYPROJ:
        return y, x
    lon, lat = utm_to_wgs84.transform(y, x)  # UTM: easting=Y, northing=X
    mx, my = wgs84_to_webmercator.transform(lon, lat)
    return mx, my


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


def create_trajectory_video(
    queries,
    output_path,
    fps=10,
    map_type='osm',
    trail_length=50
):
    """
    创建GPS轨迹地图视频

    参数:
    - queries: 查询数据列表
    - output_path: 输出视频路径
    - fps: 帧率
    - map_type: 地图类型 ('osm' 或 'satellite')
    - trail_length: 轨迹尾迹长度（显示最近N个点）
    """
    if not HAS_CONTEXTILY or not HAS_PYPROJ:
        print("Error: contextily and pyproj are required for map background")
        print("Install with: pip install contextily pyproj")
        sys.exit(1)

    num_frames = len(queries)
    print(f"Creating trajectory video with {num_frames} frames at {fps} fps")
    print(f"Video duration: {num_frames/fps:.1f} seconds")

    # 获取序列信息
    first_query = queries[0]
    seq_info = first_query['query'].split('/')[1]
    seq_parts = seq_info.split('-')
    seq_num = seq_parts[0]
    weather = seq_parts[-1]

    # 提取所有GPS位置（UTM坐标）
    positions_utm = np.array([q['position'][:2] for q in queries])

    # 转换为Web Mercator坐标
    print("Converting coordinates to Web Mercator...")
    positions_wm = np.array([utm_to_webmercator(p[0], p[1]) for p in positions_utm])

    # 计算地图范围
    x_min, x_max = positions_wm[:, 0].min(), positions_wm[:, 0].max()
    y_min, y_max = positions_wm[:, 1].min(), positions_wm[:, 1].max()

    # 计算轨迹的中心点和范围
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 使用较大的范围来确保是正方形，并添加额外边距
    max_range = max(x_range, y_range)
    # 增加边距以显示更多周围环境（50%的额外空间）
    map_size = max_range * 1.5

    # 设置正方形地图范围
    x_min = x_center - map_size / 2
    x_max = x_center + map_size / 2
    y_min = y_center - map_size / 2
    y_max = y_center + map_size / 2

    # 创建figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
    fig.patch.set_facecolor('white')

    # 设置地图范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    # 添加地图背景
    print("Loading map tiles...")
    if map_type == 'satellite':
        source = ctx.providers.Esri.WorldImagery
    else:  # osm
        source = ctx.providers.OpenStreetMap.Mapnik

    try:
        ctx.add_basemap(ax, source=source, zoom='auto')
    except Exception as e:
        print(f"Warning: Could not load map tiles: {e}")
        print("Continuing without map background...")

    # 绘制完整轨迹（浅色）
    ax.plot(positions_wm[:, 0], positions_wm[:, 1],
            'b-', alpha=0.3, linewidth=1.5, zorder=1)

    # 显示所有轨迹点（小点）
    ax.scatter(positions_wm[:, 0], positions_wm[:, 1],
               c='blue', s=10, alpha=0.3, zorder=2)

    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 添加标题
    weather_display = weather.upper() if weather != 'orin' else 'SUNNY'
    #title_text = f'GPS Trajectory - Sequence {seq_num} ({weather_display})'
    title_text = f'GPS Trajectory - Sequence {seq_num}'
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)

    # 动态元素
    current_point = None
    trail_line = None
    trail_scatter = None
    frame_text = None

    def update_frame(frame_idx):
        nonlocal current_point, trail_line, trail_scatter, frame_text

        # 清除之前的动态元素
        if current_point is not None:
            current_point.remove()
        if trail_line is not None:
            trail_line.remove()
        if trail_scatter is not None:
            trail_scatter.remove()
        if frame_text is not None:
            frame_text.remove()

        # 当前位置
        current_pos = positions_wm[frame_idx]

        # 绘制轨迹尾迹（最近N个点）
        start_idx = max(0, frame_idx - trail_length)
        trail_positions = positions_wm[start_idx:frame_idx+1]

        if len(trail_positions) > 1:
            # 渐变色轨迹
            trail_line, = ax.plot(trail_positions[:, 0], trail_positions[:, 1],
                                  'r-', linewidth=3, alpha=0.8, zorder=3)

            # 轨迹点
            trail_scatter = ax.scatter(trail_positions[:, 0], trail_positions[:, 1],
                                      c='red', s=30, alpha=0.6, zorder=4)

        # 绘制当前点（大星号）
        current_point = ax.scatter(current_pos[0], current_pos[1],
                                  c='red', s=300, marker='*',
                                  edgecolors='yellow', linewidths=2,
                                  zorder=10)

        # 添加帧数信息（左下角）
        frame_text = ax.text(0.02, 0.02, f'Frame: {frame_idx + 1}/{num_frames}',
                            transform=ax.transAxes,
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            verticalalignment='bottom')

        return []

    # 保存视频
    print("Saving video...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)

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
        description='GPS轨迹地图可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化序列08的orin天气轨迹
  python test/visualize_gps_trajectory_map_20251120_2020.py --weather orin --sequence 08

  # 使用卫星地图，帧率15
  python test/visualize_gps_trajectory_map_20251120_2020.py --weather fog --sequence 01 --map_type satellite --fps 15

  # 列出所有可用序列
  python test/visualize_gps_trajectory_map_20251120_2020.py --weather fog --list
        """
    )

    parser.add_argument('--weather', type=str, required=True,
                       choices=['orin', 'fog', 'rain', 'snow'],
                       help='天气条件')
    parser.add_argument('--sequence', type=str, default=None,
                       help='序列号 (如 "08", "01"等)')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率 (默认: 10)')
    parser.add_argument('--map_type', type=str, default='osm',
                       choices=['osm', 'satellite'],
                       help='地图类型: osm (街道地图) 或 satellite (卫星地图)')
    parser.add_argument('--trail_length', type=int, default=50,
                       help='轨迹尾迹长度 (默认: 50)')
    parser.add_argument('--output_dir', type=str, default='outputs/trajectory_videos',
                       help='输出目录 (默认: outputs/trajectory_videos)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用序列')

    args = parser.parse_args()

    # 路径设置
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / 'data'

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
    output_filename = f"gps_trajectory_{args.sequence}_{args.weather}_{args.map_type}_fps{args.fps}.mp4"
    output_path = output_dir / output_filename

    # 创建视频
    print(f"\nCreating trajectory video...")
    create_trajectory_video(
        queries=sorted_queries,
        output_path=output_path,
        fps=args.fps,
        map_type=args.map_type,
        trail_length=args.trail_length
    )

    print(f"\n✓ Done! Video saved to: {output_path}")
    print(f"  Frames: {len(sorted_queries)}")
    print(f"  FPS: {args.fps}")
    print(f"  Duration: {len(sorted_queries)/args.fps:.1f}s")
    print(f"  Map type: {args.map_type}")


if __name__ == '__main__':
    main()
