#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练实验追踪器

功能:
1. 记录训练指标(loss, AP, recall等)
2. 实时绘制训练曲线
3. 保存可视化图表
4. 可选支持TensorBoard

Created: 2025-10-22 14:30
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json


class ExperimentTracker:
    """实验追踪器 - 记录和可视化训练过程"""

    def __init__(
        self,
        save_dir: str,
        experiment_name: str = "training",
        use_tensorboard: bool = False
    ):
        """
        Args:
            save_dir: 保存目录
            experiment_name: 实验名称
            use_tensorboard: 是否使用TensorBoard
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard

        # 指标存储
        self.metrics = defaultdict(list)  # {metric_name: [values]}
        self.epochs = []

        # TensorBoard writer (可选)
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = self.save_dir / 'tensorboard'
                self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
                print(f"[Tracker] TensorBoard enabled: {tensorboard_dir}")
            except ImportError:
                print("[Tracker] Warning: tensorboard not installed, skipping")
                self.use_tensorboard = False

        # 可视化目录
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        print(f"[Tracker] Experiment: {experiment_name}")
        print(f"[Tracker] Save dir: {self.save_dir}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        记录一个epoch的指标

        Args:
            epoch: Epoch编号
            metrics: 指标字典, e.g. {'train_loss': 1.23, 'val_ap': 0.65}
        """
        # 只在第一次或epoch增加时添加
        if len(self.epochs) == 0 or epoch > self.epochs[-1]:
            self.epochs.append(epoch)

        for metric_name, value in metrics.items():
            # 确保所有指标长度一致
            while len(self.metrics[metric_name]) < len(self.epochs):
                self.metrics[metric_name].append(None)

            # 记录当前值
            if len(self.metrics[metric_name]) == len(self.epochs):
                self.metrics[metric_name][-1] = value
            else:
                self.metrics[metric_name].append(value)

            # TensorBoard记录 (跳过NaN)
            if self.writer is not None and value is not None and not (isinstance(value, float) and value != value):
                self.writer.add_scalar(metric_name, value, epoch)

    def plot_metrics(
        self,
        save_path: Optional[str] = None,
        plot_types: Optional[List[str]] = None
    ):
        """
        绘制训练曲线

        Args:
            save_path: 保存路径,默认为save_dir/visualizations/training_curves.png
            plot_types: 绘制哪些图,默认['loss', 'ap', 'recall']
        """
        if save_path is None:
            save_path = self.vis_dir / 'training_curves.png'

        if plot_types is None:
            plot_types = ['loss', 'ap', 'recall']

        # 检测可用指标
        available_metrics = list(self.metrics.keys())
        if not available_metrics:
            print("[Tracker] No metrics to plot")
            return

        # 根据指标类型分组
        metric_groups = self._group_metrics(available_metrics, plot_types)

        # 计算需要的子图数量
        num_plots = len(metric_groups)
        if num_plots == 0:
            print("[Tracker] No matching metrics to plot")
            return

        # 创建子图 (2列布局)
        ncols = 2
        nrows = (num_plots + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))

        # 确保axes是二维数组
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        # 绘制每组指标
        for idx, (group_name, metric_names) in enumerate(metric_groups.items()):
            ax = axes[idx]

            for metric_name in metric_names:
                values = self.metrics[metric_name]

                # 判断是train还是val
                if 'train' in metric_name:
                    label = metric_name.replace('train_', '').replace('_', ' ').title()
                    color = 'blue'
                    linestyle = '-'
                elif 'val' in metric_name:
                    label = metric_name.replace('val_', '').replace('_', ' ').title()
                    color = 'red'
                    linestyle = '-'
                else:
                    label = metric_name.replace('_', ' ').title()
                    color = None
                    linestyle = '-'

                # 过滤NaN值
                import numpy as np
                epochs_array = np.array(self.epochs)
                values_array = np.array(values)

                # 确保values是1D数组
                if values_array.ndim > 1:
                    values_array = values_array.flatten()

                # 确保长度匹配
                if len(epochs_array) != len(values_array):
                    min_len = min(len(epochs_array), len(values_array))
                    epochs_array = epochs_array[:min_len]
                    values_array = values_array[:min_len]

                # 找到非NaN且非None索引
                try:
                    # 尝试转换为float并检查NaN
                    values_float = np.array([float(v) if v is not None else np.nan for v in values_array])
                    valid_mask = ~np.isnan(values_float)
                except (ValueError, TypeError):
                    # 如果无法转换，跳过此metric
                    continue

                if np.any(valid_mask):
                    ax.plot(epochs_array[valid_mask], values_float[valid_mask],
                           label=label, color=color,
                           linestyle=linestyle, linewidth=2, marker='o', markersize=3)

            # 设置标题和标签
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(group_name.title(), fontsize=11)
            ax.set_title(f'{group_name.title()} Over Epochs', fontsize=12, weight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(metric_groups), len(axes)):
            axes[idx].axis('off')

        # 添加总标题
        fig.suptitle(f'Training Progress - {self.experiment_name}',
                    fontsize=16, weight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Tracker] Training curves saved: {save_path}")
        plt.close()

    def _group_metrics(
        self,
        metric_names: List[str],
        plot_types: List[str]
    ) -> Dict[str, List[str]]:
        """
        将指标按类型分组

        Args:
            metric_names: 所有指标名称
            plot_types: 要绘制的类型

        Returns:
            分组字典, e.g. {'loss': ['train_loss', 'val_loss']}
        """
        groups = defaultdict(list)

        for metric_name in metric_names:
            # 移除train/val前缀
            base_name = metric_name.replace('train_', '').replace('val_', '')

            # 根据关键词分组
            if 'loss' in base_name and 'loss' in plot_types:
                # 进一步细分loss类型
                if 'diffusion' in base_name:
                    groups['diffusion_loss'].append(metric_name)
                elif 'descriptor' in base_name:
                    groups['descriptor_loss'].append(metric_name)
                elif base_name == 'loss':
                    groups['total_loss'].append(metric_name)
                else:
                    groups['loss'].append(metric_name)

            elif 'ap' in base_name and 'ap' in plot_types:
                groups['average_precision'].append(metric_name)

            elif 'recall' in base_name and 'recall' in plot_types:
                groups['recall'].append(metric_name)

            elif 'precision' in base_name and 'precision' in plot_types:
                groups['precision'].append(metric_name)

            elif 'lr' in base_name or 'learning_rate' in base_name:
                groups['learning_rate'].append(metric_name)

        return groups

    def save_metrics(self, filename: str = 'metrics.json'):
        """
        保存指标到JSON文件

        Args:
            filename: 文件名
        """
        save_path = self.save_dir / filename

        data = {
            'experiment_name': self.experiment_name,
            'epochs': self.epochs,
            'metrics': dict(self.metrics)
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Tracker] Metrics saved: {save_path}")

    def load_metrics(self, filename: str = 'metrics.json'):
        """
        从JSON文件加载指标

        Args:
            filename: 文件名
        """
        load_path = self.save_dir / filename

        if not load_path.exists():
            print(f"[Tracker] Warning: Metrics file not found: {load_path}")
            return

        with open(load_path, 'r') as f:
            data = json.load(f)

        self.experiment_name = data['experiment_name']
        self.epochs = data['epochs']
        self.metrics = defaultdict(list, data['metrics'])

        print(f"[Tracker] Metrics loaded: {load_path}")
        print(f"  - Epochs: {len(self.epochs)}")
        print(f"  - Metrics: {list(self.metrics.keys())}")

    def get_summary(self) -> Dict:
        """
        获取训练摘要

        Returns:
            摘要字典,包含最佳指标等
        """
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.epochs),
            'best_metrics': {}
        }

        # 找到最佳指标
        for metric_name, values in self.metrics.items():
            if not values:
                continue

            # 过滤掉None值
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                continue

            # 判断是minimize还是maximize
            if 'loss' in metric_name.lower():
                best_value = min(valid_values)
                best_epoch = self.epochs[values.index(best_value)]
                mode = 'min'
            else:  # ap, recall等
                best_value = max(valid_values)
                best_epoch = self.epochs[values.index(best_value)]
                mode = 'max'

            summary['best_metrics'][metric_name] = {
                'value': best_value,
                'epoch': best_epoch,
                'mode': mode
            }

        return summary

    def print_summary(self):
        """打印训练摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print(f"Training Summary - {summary['experiment_name']}")
        print("=" * 80)
        print(f"Total Epochs: {summary['total_epochs']}")
        print("\nBest Metrics:")

        for metric_name, info in summary['best_metrics'].items():
            print(f"  {metric_name:30s}: {info['value']:.4f} (Epoch {info['epoch']})")

        print("=" * 80 + "\n")

    def close(self):
        """关闭追踪器"""
        if self.writer is not None:
            self.writer.close()

        # 保存最终指标
        self.save_metrics()

        # 绘制最终曲线
        self.plot_metrics()

        # 打印摘要
        self.print_summary()

        print("[Tracker] Experiment tracker closed")


def test_tracker():
    """测试追踪器"""
    import tempfile
    import shutil

    print("Testing ExperimentTracker...")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        tracker = ExperimentTracker(
            save_dir=temp_dir,
            experiment_name="test_experiment"
        )

        # 模拟训练过程
        for epoch in range(10):
            metrics = {
                'train_loss': 1.0 - epoch * 0.08 + np.random.rand() * 0.1,
                'val_loss': 1.1 - epoch * 0.07 + np.random.rand() * 0.1,
                'train_diffusion_loss': 0.5 - epoch * 0.04 + np.random.rand() * 0.05,
                'val_diffusion_loss': 0.52 - epoch * 0.03 + np.random.rand() * 0.05,
                'train_descriptor_loss': 0.5 - epoch * 0.04 + np.random.rand() * 0.05,
                'val_descriptor_loss': 0.58 - epoch * 0.04 + np.random.rand() * 0.05,
                'train_ap': 0.3 + epoch * 0.05 + np.random.rand() * 0.05,
                'val_ap': 0.25 + epoch * 0.05 + np.random.rand() * 0.05,
                'train_recall@1': 0.4 + epoch * 0.04 + np.random.rand() * 0.05,
                'val_recall@1': 0.35 + epoch * 0.04 + np.random.rand() * 0.05,
            }

            tracker.log_epoch(epoch, metrics)

        # 关闭追踪器
        tracker.close()

        print(f"\n✓ Test completed!")
        print(f"  Results saved to: {temp_dir}")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"  Cleaned up temporary directory")


if __name__ == '__main__':
    test_tracker()
