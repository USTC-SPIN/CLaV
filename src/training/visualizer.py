#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练可视化工具

功能:
1. 解析训练日志,绘制训练曲线
2. 可视化去噪效果(noisy/denoised/clean对比)
3. 可视化检索结果(Top-K)

Created: 2025-10-22 14:00
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Optional
import argparse


class TrainingLogger:
    """训练日志解析器"""

    def __init__(self, log_file: str):
        """
        Args:
            log_file: 日志文件路径 (e.g., checkpoints/joint_training/logs/train.log)
        """
        self.log_file = Path(log_file)
        self.metrics = self._parse_log()

    def _parse_log(self) -> Dict[str, List[float]]:
        """解析训练日志"""
        metrics = {
            'epoch': [],
            'train_loss': [],
            'train_diffusion_loss': [],
            'train_descriptor_loss': [],
            'train_ap': [],
            'train_recall@1': [],
            'val_loss': [],
            'val_diffusion_loss': [],
            'val_descriptor_loss': [],
            'val_ap': [],
            'val_recall@1': [],
        }

        if not self.log_file.exists():
            print(f"Warning: Log file not found: {self.log_file}")
            return metrics

        with open(self.log_file, 'r', encoding='utf-8') as f:
            current_epoch = None

            for line in f:
                # 匹配epoch号
                epoch_match = re.search(r'Epoch (\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))

                # 匹配训练指标
                if 'Train -' in line:
                    loss_match = re.search(r'loss=([\d.]+)', line)
                    diff_match = re.search(r'diff=([\d.]+)', line)
                    desc_match = re.search(r'desc=([\d.]+)', line)
                    ap_match = re.search(r'AP=([\d.]+)', line)
                    recall_match = re.search(r'R@1=([\d.]+)', line)

                    if current_epoch is not None:
                        if loss_match:
                            metrics['epoch'].append(current_epoch)
                            metrics['train_loss'].append(float(loss_match.group(1)))
                        if diff_match:
                            metrics['train_diffusion_loss'].append(float(diff_match.group(1)))
                        if desc_match:
                            metrics['train_descriptor_loss'].append(float(desc_match.group(1)))
                        if ap_match:
                            metrics['train_ap'].append(float(ap_match.group(1)))
                        if recall_match:
                            metrics['train_recall@1'].append(float(recall_match.group(1)))

                # 匹配验证指标
                if 'Val -' in line:
                    loss_match = re.search(r'loss=([\d.]+)', line)
                    diff_match = re.search(r'diff=([\d.]+)', line)
                    desc_match = re.search(r'desc=([\d.]+)', line)
                    ap_match = re.search(r'AP=([\d.]+)', line)
                    recall_match = re.search(r'R@1=([\d.]+)', line)

                    if loss_match:
                        metrics['val_loss'].append(float(loss_match.group(1)))
                    if diff_match:
                        metrics['val_diffusion_loss'].append(float(diff_match.group(1)))
                    if desc_match:
                        metrics['val_descriptor_loss'].append(float(desc_match.group(1)))
                    if ap_match:
                        metrics['val_ap'].append(float(ap_match.group(1)))
                    if recall_match:
                        metrics['val_recall@1'].append(float(recall_match.group(1)))

        return metrics

    def plot_training_curves(self, save_path: str):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Curves - Joint Training (Stage2 + Stage3)', fontsize=16)

        epochs = self.metrics['epoch']

        # 1. Total Loss
        ax = axes[0, 0]
        if self.metrics['train_loss']:
            ax.plot(epochs, self.metrics['train_loss'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_loss']:
            val_epochs = epochs[::len(epochs)//len(self.metrics['val_loss']) or 1][:len(self.metrics['val_loss'])]
            ax.plot(val_epochs, self.metrics['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Diffusion Loss
        ax = axes[0, 1]
        if self.metrics['train_diffusion_loss']:
            ax.plot(epochs, self.metrics['train_diffusion_loss'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_diffusion_loss']:
            val_epochs = epochs[::len(epochs)//len(self.metrics['val_diffusion_loss']) or 1][:len(self.metrics['val_diffusion_loss'])]
            ax.plot(val_epochs, self.metrics['val_diffusion_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Diffusion Loss (MSE)')
        ax.set_title('Stage2 Diffusion Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Descriptor Loss
        ax = axes[0, 2]
        if self.metrics['train_descriptor_loss']:
            ax.plot(epochs, self.metrics['train_descriptor_loss'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_descriptor_loss']:
            val_epochs = epochs[::len(epochs)//len(self.metrics['val_descriptor_loss']) or 1][:len(self.metrics['val_descriptor_loss'])]
            ax.plot(val_epochs, self.metrics['val_descriptor_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Descriptor Loss')
        ax.set_title('Stage3 Descriptor Loss (TruncatedSmoothAP)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Average Precision (AP)
        ax = axes[1, 0]
        if self.metrics['train_ap']:
            ax.plot(epochs, [x*100 for x in self.metrics['train_ap']], 'b-', label='Train', linewidth=2)
        if self.metrics['val_ap']:
            val_epochs = epochs[::len(epochs)//len(self.metrics['val_ap']) or 1][:len(self.metrics['val_ap'])]
            ax.plot(val_epochs, [x*100 for x in self.metrics['val_ap']], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP (%)')
        ax.set_title('Average Precision (AP)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # 5. Recall@1
        ax = axes[1, 1]
        if self.metrics['train_recall@1']:
            ax.plot(epochs, [x*100 for x in self.metrics['train_recall@1']], 'b-', label='Train', linewidth=2)
        if self.metrics['val_recall@1']:
            val_epochs = epochs[::len(epochs)//len(self.metrics['val_recall@1']) or 1][:len(self.metrics['val_recall@1'])]
            ax.plot(val_epochs, [x*100 for x in self.metrics['val_recall@1']], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall@1 (%)')
        ax.set_title('Recall@1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # 6. Learning Rate (可选,从日志中提取)
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Expected Range:\nAP: 50-70%\nRecall@1: 60-80%\n\n(After data leakage fix)',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('Expected Performance')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.close()


class DenoisingVisualizer:
    """去噪效果可视化"""

    @staticmethod
    def visualize_denoising(
        model,
        noisy_img: torch.Tensor,
        clean_img: torch.Tensor,
        save_path: str,
        device: str = 'cuda'
    ):
        """
        可视化去噪效果

        Args:
            model: CLaV模型
            noisy_img: 恶劣天气BEV图像 (1, 3, H, W) or (3, H, W)
            clean_img: 晴天BEV图像 (1, 3, H, W) or (3, H, W)
            save_path: 保存路径
            device: 设备
        """
        model.eval()

        # 确保输入shape正确
        if noisy_img.dim() == 3:
            noisy_img = noisy_img.unsqueeze(0)
        if clean_img.dim() == 3:
            clean_img = clean_img.unsqueeze(0)

        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)

        with torch.no_grad():
            # 编码
            z_noisy = model.encoder(noisy_img)

            # 去噪
            z_denoised = model.denoise_latent(z_noisy)

            # 注意: 我们没有decoder,所以只能可视化latent的统计特性
            # 这里我们展示输入图像对比

        # 转换为numpy用于可视化
        noisy_np = noisy_img[0].cpu().permute(1, 2, 0).numpy()
        clean_np = clean_img[0].cpu().permute(1, 2, 0).numpy()

        # 反归一化 (假设ImageNet归一化)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        noisy_np = np.clip(noisy_np * std + mean, 0, 1)
        clean_np = np.clip(clean_np * std + mean, 0, 1)

        # 可视化latent统计
        z_noisy_stats = {
            'mean': z_noisy.mean().item(),
            'std': z_noisy.std().item(),
            'min': z_noisy.min().item(),
            'max': z_noisy.max().item(),
        }
        z_denoised_stats = {
            'mean': z_denoised.mean().item(),
            'std': z_denoised.std().item(),
            'min': z_denoised.min().item(),
            'max': z_denoised.max().item(),
        }

        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Denoising Effect Visualization', fontsize=14)

        # Noisy input
        ax = axes[0]
        ax.imshow(noisy_np)
        ax.set_title(f'Noisy Input (Bad Weather)\n'
                     f'Latent: mean={z_noisy_stats["mean"]:.3f}, std={z_noisy_stats["std"]:.3f}')
        ax.axis('off')

        # Clean target
        ax = axes[1]
        ax.imshow(clean_np)
        ax.set_title(f'Clean Target (Sunny)\n'
                     f'Denoised Latent: mean={z_denoised_stats["mean"]:.3f}, std={z_denoised_stats["std"]:.3f}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Denoising visualization saved to: {save_path}")
        plt.close()


class RetrievalVisualizer:
    """检索结果可视化"""

    @staticmethod
    def visualize_retrieval(
        query_img: np.ndarray,
        top_k_imgs: List[np.ndarray],
        top_k_distances: List[float],
        top_k_labels: List[str],
        save_path: str,
        k: int = 5
    ):
        """
        可视化Top-K检索结果

        Args:
            query_img: 查询图像 (H, W, 3), numpy array
            top_k_imgs: Top-K图像列表
            top_k_distances: Top-K距离列表
            top_k_labels: Top-K标签列表 (e.g., 'Positive', 'Negative')
            save_path: 保存路径
            k: 显示前k个结果
        """
        k = min(k, len(top_k_imgs))

        fig, axes = plt.subplots(1, k+1, figsize=(3*(k+1), 3))
        fig.suptitle(f'Top-{k} Retrieval Results', fontsize=14)

        # Query
        ax = axes[0]
        ax.imshow(query_img)
        ax.set_title('Query\n(Bad Weather)', fontsize=10)
        ax.axis('off')
        ax.set_facecolor('lightblue')

        # Top-K
        for i in range(k):
            ax = axes[i+1]
            ax.imshow(top_k_imgs[i])

            label = top_k_labels[i]
            dist = top_k_distances[i]

            color = 'lightgreen' if label == 'Positive' else 'lightcoral'
            ax.set_facecolor(color)
            ax.set_title(f'Rank {i+1}\n{label}\nDist: {dist:.3f}', fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Retrieval visualization saved to: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析日志并绘制训练曲线
    print("Parsing training log...")
    logger = TrainingLogger(args.log_file)

    print("Plotting training curves...")
    logger.plot_training_curves(str(output_dir / 'training_curves.png'))

    print("\nVisualization completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
