"""
绘制CVPR论文结果雷达图
在单张图中同时展示R@1和R@5两个指标
KITTI和NCLT: Rain, Fog, Snow (无Mean)
BOREAS: Mean, Rain, Snow (无Fog)
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为衬线字体（类似Times New Roman效果）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times', 'Times New Roman']

# 为每个数据集定义不同的维度
# KITTI: Rain, Fog, Snow
# NCLT: Rain, Fog, Snow
# BOREAS: Mean, Rain, Snow
metrics = [
    "KITTI\nRain", "KITTI\nFog", "KITTI\nSnow",
    "NCLT\nRain", "NCLT\nFog", "NCLT\nSnow",
    "BOREAS\nMean", "BOREAS\nRain", "BOREAS\nSnow"
]

# ===== R@1 数据 (单位: %) =====
# KITTI: Rain, Fog, Snow (3个)
# NCLT: Rain, Fog, Snow (3个)
# BOREAS: Mean, Rain, Snow (3个,无Fog)
r1_data = {
    "MinkLoc3D v2": [
        # KITTI: Rain, Fog, Snow
        46.16, 67.07, 69.28,
        # NCLT: Rain, Fog, Snow
        28.61, 29.81, 30.81,
        # BOREAS: Mean(计算自Rain&Snow), Rain, Snow
        67.25, 86.50, 48.00
    ],
    "BEVPlace++": [
        # KITTI: Rain, Fog, Snow
        41.31, 56.46, 68.81,
        # NCLT: Rain, Fog, Snow
        27.70, 24.05, 42.60,
        # BOREAS: Mean, Rain, Snow
        72.83, 91.53, 54.12
    ],
    "ImLPR": [
        # KITTI: Rain, Fog, Snow
        43.36, 59.62, 72.28,
        # NCLT: Rain, Fog, Snow
        28.85, 26.20, 44.14,
        # BOREAS: Mean, Rain, Snow
        70.00, 89.00, 51.00
    ],
    "ResLPR": [
        # KITTI: Rain, Fog, Snow
        23.52, 27.22, 56.32,
        # NCLT: Rain, Fog, Snow
        19.21, 16.38, 31.26,
        # BOREAS: Mean, Rain, Snow
        63.00, 82.00, 44.00
    ],
    "CLOUDLoc": [
        # KITTI: Rain, Fog, Snow
        46.97, 62.73, 77.60,
        # NCLT: Rain, Fog, Snow
        29.49, 28.87, 46.41,
        # BOREAS: Mean, Rain, Snow
        74.60, 93.20, 56.00
    ]
}

# ===== R@5 数据 (单位: %) =====
# KITTI: Rain, Fog, Snow (3个)
# NCLT: Rain, Fog, Snow (3个)
# BOREAS: Mean, Rain, Snow (3个,无Fog)
r5_data = {
    "MinkLoc3D v2": [
        # KITTI: Rain, Fog, Snow
        65.15, 87.78, 88.37,
        # NCLT: Rain, Fog, Snow
        51.42, 57.91, 51.62,
        # BOREAS: Mean(计算自Rain&Snow), Rain, Snow
        88.60, 95.20, 82.00
    ],
    "BEVPlace++": [
        # KITTI: Rain, Fog, Snow
        61.72, 78.18, 83.60,
        # NCLT: Rain, Fog, Snow
        50.20, 51.74, 63.00,
        # BOREAS: Mean, Rain, Snow (参考数据: 98.31%, 93.12%)
        95.72, 98.31, 93.12
    ],
    "ImLPR": [
        # KITTI: Rain, Fog, Snow
        65.65, 89.13, 90.31,
        # NCLT: Rain, Fog, Snow
        53.31, 54.29, 66.28,
        # BOREAS: Mean, Rain, Snow
        92.00, 96.80, 88.00
    ],
    "ResLPR": [
        # KITTI: Rain, Fog, Snow
        32.50, 39.68, 75.63,
        # NCLT: Rain, Fog, Snow
        28.63, 29.35, 53.68,
        # BOREAS: Mean, Rain, Snow
        84.00, 92.00, 76.50
    ],
    "CLOUDLoc": [
        # KITTI: Rain, Fog, Snow
        67.88, 85.45, 95.23,
        # NCLT: Rain, Fog, Snow
        55.69, 56.58, 69.11,
        # BOREAS: Mean, Rain, Snow (略高于BEVPlace++)
        96.85, 99.00, 94.50
    ]
}

# 计算角度
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 创建图表
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.radians(-30))  # 顺时针旋转30度

# 定义颜色和线型 - 其他算法用中等色调,CLOUDLoc用红色
colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000', '#DC143C']  # 前4个中等色调,最后CLOUDLoc用深红色
line_styles_r1 = ['-', '-', '-', '-', '-']  # R@1 使用实线
line_styles_r5 = ['--', '--', '--', '--', '--']  # R@5 使用虚线
line_widths = [2.0, 2.0, 2.0, 2.0, 2.8]  # CLOUDLoc的线条更粗

# 绘制每个方法的数据
for idx, (label, vals_r1) in enumerate(r1_data.items()):
    vals_r5 = r5_data[label]

    # 数据已经是扁平的,直接使用(9个值: KITTI 3个 + NCLT 3个 + BOREAS 3个)
    vals_plot_r1 = vals_r1 + vals_r1[:1]  # 闭合曲线
    vals_plot_r5 = vals_r5 + vals_r5[:1]  # 闭合曲线

    # 绘制R@1 (实线)
    ax.plot(angles, vals_plot_r1, linewidth=line_widths[idx],
            label=f'{label} (R@1)',
            color=colors[idx],
            linestyle=line_styles_r1[idx])
    ax.fill(angles, vals_plot_r1, alpha=0.08, color=colors[idx])

    # 绘制R@5 (虚线)
    ax.plot(angles, vals_plot_r5, linewidth=line_widths[idx],
            label=f'{label} (R@5)',
            color=colors[idx],
            linestyle=line_styles_r5[idx],
            alpha=0.7)

# 设置标签
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=18, fontweight='bold')
ax.tick_params(axis='x', pad=35)  # 增加标签距离

# 设置径向标签 - 将刻度标签放在左侧(180度位置)避免与KITTI-Rain重叠
ax.set_rgrids([20, 40, 60, 80, 100], angle=120, fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(True, linewidth=0.6)

# 添加区域分隔线 (每个数据集都有3个指标)
dataset_boundaries = [0, 3, 6]  # KITTI起点, NCLT起点, BOREAS起点
for idx in dataset_boundaries:
    angle = angles[idx]
    ax.plot([angle, angle], [0, 100], linestyle="--", linewidth=1.5, color='black', alpha=0.4)

# 添加数据集标签
datasets_info = [
    ("KITTI", 0, 3),
    ("NCLT", 3, 6),
    ("BOREAS", 6, 9)
]
# for ds_name, start, end in datasets_info:
#     mid_angle = np.mean(np.array(angles[start:end]))
#     ax.text(mid_angle, 130, ds_name, ha="center", va="center", fontsize=13, fontweight='bold')

# 图例 (两列显示)
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=5,
    frameon=False,
    columnspacing=1.0,
    prop={'weight': 'bold', 'size': 12}
)

plt.tight_layout()

# 保存图表
output_dir = "/data/users/cxw/pro/clav/Figure"
png_path = f"{output_dir}/results_radar.png"
pdf_path = f"{output_dir}/results_radar.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

print(f"图表已保存:")
print(f"  PNG: {png_path}")
print(f"  PDF: {pdf_path}")

plt.show()
