#!/usr/bin/env python3
"""
消融实验结果可视化
时间: 2024-11-13 12:17
功能: 生成图表展示消融实验结果
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 设置中文字体（避免乱码）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 设置非交互模式
import matplotlib
matplotlib.use('Agg')

# 数据（基于评估结果）
data = {
    'KITTI': {
        'ExpB1': {'R@1': 11.17, 'R@5': 25.75, 'R@10': 35.72},
        'ExpB2': {'R@1': 7.38, 'R@5': 22.44, 'R@10': 32.25},
        'ExpB3': {'R@1': 25.80, 'R@5': 54.26, 'R@10': 67.00},
        'ExpB4': {'R@1': 8.07, 'R@5': 19.82, 'R@10': 27.06},
    },
    'NCLT': {
        'ExpB1': {'R@1': 5.40, 'R@5': 17.61, 'R@10': 28.30},
        'ExpB2': {'R@1': 4.86, 'R@5': 17.24, 'R@10': 26.53},
        'ExpB3': {'R@1': 21.80, 'R@5': 52.70, 'R@10': 64.48},
        'ExpB4': {'R@1': 3.95, 'R@5': 13.18, 'R@10': 20.22},
    }
}

# 实验配置
configs = {
    'ExpB1': 'Small+DDPM+NetVLAD',
    'ExpB2': 'Base+DDPM+NetVLAD',
    'ExpB3': 'Base+Flow+NetVLAD',
    'ExpB4': 'Base+Flow+SALAD',
}

# 创建输出目录
output_dir = '/data/users/cxw/pro/clav/ablation/visualizations'
os.makedirs(output_dir, exist_ok=True)

# 颜色方案
colors = {
    'ExpB1': '#3498db',  # 蓝色
    'ExpB2': '#2ecc71',  # 绿色
    'ExpB3': '#e74c3c',  # 红色
    'ExpB4': '#f39c12',  # 橙色
}

def create_comparison_bar_chart():
    """创建Recall@1和Recall@5的对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    experiments = ['ExpB1', 'ExpB2', 'ExpB3', 'ExpB4']
    x = np.arange(len(experiments))
    width = 0.35

    # KITTI Recall@1
    ax = axes[0, 0]
    values = [data['KITTI'][exp]['R@1'] for exp in experiments]
    bars = ax.bar(x, values, width, color=[colors[exp] for exp in experiments])
    ax.set_ylabel('Recall@1 (%)')
    ax.set_title('KITTI Dataset - Recall@1')
    ax.set_xticks(x)
    ax.set_xticklabels([configs[exp].replace('+', '\n') for exp in experiments], rotation=0, fontsize=8)
    ax.set_ylim(0, 30)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')

    # KITTI Recall@5
    ax = axes[0, 1]
    values = [data['KITTI'][exp]['R@5'] for exp in experiments]
    bars = ax.bar(x, values, width, color=[colors[exp] for exp in experiments])
    ax.set_ylabel('Recall@5 (%)')
    ax.set_title('KITTI Dataset - Recall@5')
    ax.set_xticks(x)
    ax.set_xticklabels([configs[exp].replace('+', '\n') for exp in experiments], rotation=0, fontsize=8)
    ax.set_ylim(0, 60)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')

    # NCLT Recall@1
    ax = axes[1, 0]
    values = []
    for exp in experiments:
        val = data['NCLT'][exp]['R@1']
        values.append(val if val is not None else 0)

    bars = ax.bar(x, values, width, color=[colors[exp] for exp in experiments])
    ax.set_ylabel('Recall@1 (%)')
    ax.set_title('NCLT Dataset - Recall@1')
    ax.set_xticks(x)
    ax.set_xticklabels([configs[exp].replace('+', '\n') for exp in experiments], rotation=0, fontsize=8)
    ax.set_ylim(0, 25)

    for bar, val, exp in zip(bars, values, experiments):
        if data['NCLT'][exp]['R@1'] is not None:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 1,
                    'Evaluating', ha='center', va='bottom', fontsize=8)

    # NCLT Recall@5
    ax = axes[1, 1]
    values = []
    for exp in experiments:
        val = data['NCLT'][exp]['R@5']
        values.append(val if val is not None else 0)

    bars = ax.bar(x, values, width, color=[colors[exp] for exp in experiments])
    ax.set_ylabel('Recall@5 (%)')
    ax.set_title('NCLT Dataset - Recall@5')
    ax.set_xticks(x)
    ax.set_xticklabels([configs[exp].replace('+', '\n') for exp in experiments], rotation=0, fontsize=8)
    ax.set_ylim(0, 60)

    for bar, val, exp in zip(bars, values, experiments):
        if data['NCLT'][exp]['R@5'] is not None:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom')

    plt.suptitle('Ablation Study Results - BEV Denoising and Localization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'ablation_comparison_bars.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_component_impact_chart():
    """创建组件影响分析图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 组件升级的影响（KITTI数据集）
    components = ['Backbone\n(Small→Base)', 'Denoiser\n(DDPM→Flow)', 'Descriptor\n(NetVLAD→SALAD)']
    recall1_impact = [-3.79, 18.42, -17.73]
    recall5_impact = [-3.31, 31.82, -34.44]

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax.bar(x - width/2, recall1_impact, width, label='Recall@1', color='#3498db')
    bars2 = ax.bar(x + width/2, recall5_impact, width, label='Recall@5', color='#e74c3c')

    ax.set_ylabel('Performance Change (%)')
    ax.set_title('Component Impact Analysis (KITTI Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%', ha='center',
                    va='bottom' if height > 0 else 'top')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'component_impact_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_radar_chart():
    """创建雷达图比较各实验的多维性能"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))

    metrics = ['R@1', 'R@5', 'R@10']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # KITTI Radar Chart
    ax = axes[0]
    for exp in ['ExpB1', 'ExpB3', 'ExpB4']:  # Skip ExpB2 for clarity
        values = [data['KITTI'][exp][m] for m in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=configs[exp], color=colors[exp])
        ax.fill(angles, values, alpha=0.15, color=colors[exp])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 70)
    ax.set_title('KITTI Dataset Performance', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # NCLT Radar Chart
    ax = axes[1]
    for exp in ['ExpB1', 'ExpB3', 'ExpB4']:
        if data['NCLT'][exp]['R@1'] is not None:
            values = [data['NCLT'][exp][m] for m in metrics]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=configs[exp], color=colors[exp])
            ax.fill(angles, values, alpha=0.15, color=colors[exp])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 70)
    ax.set_title('NCLT Dataset Performance', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.suptitle('Multi-metric Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'performance_radar_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_table():
    """创建汇总表格图片"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    headers = ['Experiment', 'Configuration', 'KITTI R@1', 'KITTI R@5', 'NCLT R@1', 'NCLT R@5']

    for exp in ['ExpB1', 'ExpB2', 'ExpB3', 'ExpB4']:
        row = [
            exp,
            configs[exp],
            f"{data['KITTI'][exp]['R@1']:.2f}%" if data['KITTI'][exp]['R@1'] else 'N/A',
            f"{data['KITTI'][exp]['R@5']:.2f}%" if data['KITTI'][exp]['R@5'] else 'N/A',
            f"{data['NCLT'][exp]['R@1']:.2f}%" if data['NCLT'][exp]['R@1'] else 'Evaluating',
            f"{data['NCLT'][exp]['R@5']:.2f}%" if data['NCLT'][exp]['R@5'] else 'Evaluating',
        ]
        table_data.append(row)

    # 创建表格
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1, 0.3, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best results
    best_kitti_r1_idx = 3  # ExpB3
    best_nclt_r1_idx = 3   # ExpB3

    table[(best_kitti_r1_idx, 2)].set_facecolor('#e8f8f5')
    table[(best_kitti_r1_idx, 3)].set_facecolor('#e8f8f5')
    table[(best_nclt_r1_idx, 4)].set_facecolor('#e8f8f5')
    table[(best_nclt_r1_idx, 5)].set_facecolor('#e8f8f5')

    plt.title('Ablation Study Results Summary', fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(output_dir, 'results_summary_table.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("生成消融实验可视化")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()

    # 生成各种图表
    print("1. 生成对比柱状图...")
    create_comparison_bar_chart()

    print("2. 生成组件影响分析图...")
    create_component_impact_chart()

    print("3. 生成性能雷达图...")
    create_radar_chart()

    print("4. 生成汇总表格...")
    create_summary_table()

    print()
    print("="*60)
    print(f"所有图表已保存到: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()