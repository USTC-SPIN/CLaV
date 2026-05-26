#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAE-ImLPR统一训练脚本

支持三种训练模式:
1. stage2_only: 只训练Stage2扩散去噪模型 (冻结Stage1和Stage3)
2. descriptor_only: 只训练Stage3描述符头 (冻结Stage1和Stage2)
3. joint: 联合训练Stage2+Stage3 (冻结Stage1) - 推荐使用

配置文件控制训练模式:
- configs/train_config_stage2.yaml: Stage2训练
- configs/train_config_descriptor.yaml: Stage3训练
- configs/train_config_joint.yaml: 联合训练

Created: 2025-10-21 16:15
Updated: 2025-10-22 14:15 - 重命名为统一训练脚本
Author: Claude Code Assistant
"""

# ============ PATH SETUP (MUST BE FIRST) ============
import sys
import os
from pathlib import Path

# 添加路径 - 确保从项目根目录导入
SCRIPT_DIR = Path(__file__).resolve().parent  # src/training/
PROJECT_DIR = SCRIPT_DIR.parent.parent         # 项目根目录 (clav/)

# 添加项目根目录到sys.path
sys.path.insert(0, str(PROJECT_DIR))

# 切换工作目录到项目根目录
os.chdir(PROJECT_DIR)
# ================== END PATH SETUP ==================

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
from tqdm import tqdm
import time
from datetime import datetime

from src.models.clav import CLaV
from src.training.train_utils import (
    setup_optimizer, setup_scheduler, save_checkpoint, load_checkpoint,
    Logger, AverageMeter, count_parameters, set_seed, EMA
)
from src.training.experiment_tracker import ExperimentTracker
from src.training.visualizer import DenoisingVisualizer

# 导入ImLPR的数据加载器和损失函数
from src.datasets.dataset_utils import make_dataloaders
from src.models.losses.loss import make_losses
from src.utils.utils import TrainingParams

# 导入联合训练所需模块
from src.datasets.denoising_dataset import DenoisingBEVDataset, DenoisingBEVDatasetWithPositions
from src.datasets.denoising_dataset_utils import make_denoising_dataloader
from src.datasets.base_datasets import TrainingTuple, EvaluationTuple  # pickle反序列化需要

# 导入可视化相关
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import numpy as np


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """反归一化图像 (ImageNet normalization)"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return np.clip(img, 0, 1)


class EarlyStopping:
    """
    早停机制

    监控验证指标，如果连续patience个epoch没有改善，则停止训练
    """
    def __init__(self, patience=30, min_delta=0.3, monitor='recall@1', mode='max'):
        """
        Args:
            patience: 容忍的无改善epoch数
            min_delta: 最小改善阈值（百分比，如0.3表示0.3%）
            monitor: 监控的指标名称
            mode: 'max'表示指标越大越好，'min'表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.should_stop_flag = False

    def step(self, current_value):
        """
        更新早停状态

        Args:
            current_value: 当前监控指标的值

        Returns:
            improved: 是否有改善
        """
        if self.mode == 'max':
            improved = (current_value - self.best_value) > self.min_delta
        else:
            improved = (self.best_value - current_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop_flag = True
            return False

    def should_stop(self):
        """是否应该停止训练"""
        return self.should_stop_flag

    def state_dict(self):
        """保存状态"""
        return {
            'counter': self.counter,
            'best_value': self.best_value,
            'should_stop_flag': self.should_stop_flag
        }

    def load_state_dict(self, state_dict):
        """加载状态"""
        self.counter = state_dict['counter']
        self.best_value = state_dict['best_value']
        self.should_stop_flag = state_dict['should_stop_flag']


def evaluate_validation_recall(model, config, device, logger):
    """
    使用现有评估脚本进行验证

    调用evaluation/eval_with_kdtree.py的逻辑进行评估
    只在主进程调用，避免DDP同步问题

    Args:
        model: 模型（不带DDP包装）
        config: 完整配置字典
        device: 设备
        logger: 日志记录器

    Returns:
        avg_recall: 平均Recall@1（用于early stopping）
    """
    import tempfile
    import yaml
    from evaluation.eval_with_kdtree import evaluate_all_conditions

    # 确保模型不是DDP包装的
    if hasattr(model, 'module'):
        model = model.module

    model.eval()

    try:
        # 创建临时文件保存checkpoint和config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            temp_config_path = config_file.name
            yaml.dump(config, config_file)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as ckpt_file:
            temp_checkpoint_path = ckpt_file.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, ckpt_file)

        # 重定向评估脚本的print输出（避免干扰训练日志）
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # 调用评估函数（使用文件路径）
            results = evaluate_all_conditions(
                checkpoint_path=temp_checkpoint_path,
                config_path=temp_config_path,
                device=device,
                batch_size=16  # 使用较小的batch size避免OOM
            )
        finally:
            # 恢复stdout
            sys.stdout = old_stdout

        # 删除临时文件
        import os
        os.unlink(temp_checkpoint_path)
        os.unlink(temp_config_path)

        # 计算平均Recall（results是字典，格式: {'snow': {'recall@1': 85.0, ...}, 'rain': {...}}）
        if results:
            # 提取所有天气条件的Recall@1
            recall_at_1_values = [metrics.get('recall@1', 0.0) for metrics in results.values() if isinstance(metrics, dict)]
            recall_at_5_values = [metrics.get('recall@5', 0.0) for metrics in results.values() if isinstance(metrics, dict)]
            recall_at_10_values = [metrics.get('recall@10', 0.0) for metrics in results.values() if isinstance(metrics, dict)]

            avg_recall = sum(recall_at_1_values) / len(recall_at_1_values) if recall_at_1_values else 0.0
            avg_recall_5 = sum(recall_at_5_values) / len(recall_at_5_values) if recall_at_5_values else 0.0
            avg_recall_10 = sum(recall_at_10_values) / len(recall_at_10_values) if recall_at_10_values else 0.0
        else:
            avg_recall = 0.0
            avg_recall_5 = 0.0
            avg_recall_10 = 0.0

        if logger is not None:
            logger.log(f"\n[Validation] Average Recall@1: {avg_recall:.2f}%")
            logger.log(f"[Validation] Average Recall@5: {avg_recall_5:.2f}%")
            logger.log(f"[Validation] Average Recall@10: {avg_recall_10:.2f}%")
            # 记录每个天气条件的结果
            for weather_name, metrics in results.items():
                if isinstance(metrics, dict):
                    r1 = metrics.get('recall@1', 0.0)
                    logger.log(f"[Validation]   {weather_name}: Recall@1={r1:.2f}%")

        model.train()
        return avg_recall

    except Exception as e:
        if logger is not None:
            logger.log(f"\n[Validation] Error during evaluation: {e}")
            logger.log("  Skipping validation for this epoch")

        # 确保删除临时文件
        try:
            if 'temp_checkpoint_path' in locals():
                os.unlink(temp_checkpoint_path)
            if 'temp_config_path' in locals():
                os.unlink(temp_config_path)
        except:
            pass

        model.train()
        return 0.0

def visualize_denoising_samples(
    model,
    dataloader,
    save_dir: Path,
    epoch: int,
    num_samples: int = 3,
    device: str = 'cuda'
):
    """
    可视化去噪效果

    Args:
        model: CLaV模型
        dataloader: 数据加载器
        save_dir: 保存目录
        epoch: 当前epoch
        num_samples: 可视化样本数
        device: 设备
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 检查dataloader是否为空
    if len(dataloader) == 0:
        logger.log("[Warning] DataLoader is empty, skipping visualization")
        return

    # 获取一个batch的数据
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.log("[Warning] Failed to get batch from dataloader, skipping visualization")
        return

    noisy_imgs, clean_imgs, _, _, _, weather = batch

    # 只取前num_samples个
    noisy_imgs = noisy_imgs[:num_samples].to(device)
    clean_imgs = clean_imgs[:num_samples].to(device)
    weather = weather[:num_samples]

    with torch.no_grad():
        # 编码
        if hasattr(model.encoder, 'encode'):
            # SharedDINOv2
            z_noisy = model.encoder.encode(
                noisy_imgs,
                normalize_input=True,
                normalize_latent=True,
                reshape_to_2d=True
            )
            z_clean = model.encoder.encode(
                clean_imgs,
                normalize_input=True,
                normalize_latent=True,
                reshape_to_2d=True
            )
        else:
            # AdaptedDINOv2
            z_noisy, _ = model.encoder(noisy_imgs)
            z_clean, _ = model.encoder(clean_imgs)

        # 去噪
        z_denoised = model.denoise_latent(z_noisy)

        # 提取descriptor
        desc_noisy = model.descriptor_head(z_noisy)
        desc_denoised = model.descriptor_head(z_denoised)
        desc_clean = model.descriptor_head(z_clean)

    # 计算相似度
    cos_sim = torch.nn.functional.cosine_similarity

    # 创建可视化 - 改为4列布局以显示更多信息
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # 图像
        noisy_np = noisy_imgs[i].cpu().permute(1, 2, 0).numpy()
        clean_np = clean_imgs[i].cpu().permute(1, 2, 0).numpy()

        noisy_np = denormalize_image(noisy_np)
        clean_np = denormalize_image(clean_np)

        # Latent统计
        z_noisy_stats = f"mean={z_noisy[i].mean():.3f}, std={z_noisy[i].std():.3f}"
        z_denoised_stats = f"mean={z_denoised[i].mean():.3f}, std={z_denoised[i].std():.3f}"
        z_clean_stats = f"mean={z_clean[i].mean():.3f}, std={z_clean[i].std():.3f}"

        # Descriptor相似度
        sim_noisy_clean = cos_sim(desc_noisy[i:i+1], desc_clean[i:i+1]).item()
        sim_denoised_clean = cos_sim(desc_denoised[i:i+1], desc_clean[i:i+1]).item()
        sim_improvement = sim_denoised_clean - sim_noisy_clean

        # 绘制
        # Column 1: Noisy
        ax = axes[i, 0]
        ax.imshow(noisy_np)
        ax.set_title(
            f'Noisy Input ({weather[i]})\n'
            f'Latent: {z_noisy_stats}\n'
            f'Sim to Clean: {sim_noisy_clean:.3f}',
            fontsize=9
        )
        ax.axis('off')

        # Column 2: Noisy latent (平均激活图)
        ax = axes[i, 1]
        latent_noisy_vis = z_noisy[i].mean(dim=0).cpu().numpy()  # (32, 32)
        im1 = ax.imshow(latent_noisy_vis, cmap='RdYlBu_r', aspect='auto')
        ax.set_title(
            f'Noisy Latent\n{z_noisy_stats}',
            fontsize=9
        )
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Column 3: Denoised latent (平均激活图)
        ax = axes[i, 2]
        latent_denoised_vis = z_denoised[i].mean(dim=0).cpu().numpy()  # (32, 32)
        im2 = ax.imshow(latent_denoised_vis, cmap='RdYlBu_r', aspect='auto')
        ax.set_title(
            f'Denoised Latent\n{z_denoised_stats}\nSim↑: {sim_improvement:+.3f}',
            fontsize=9
        )
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        # Column 4: Clean target
        ax = axes[i, 3]
        ax.imshow(clean_np)
        ax.set_title(
            f'Clean Target (Sunny)\n'
            f'Latent: {z_clean_stats}\n'
            f'Denoised→Clean Sim: {sim_denoised_clean:.3f}',
            fontsize=9
        )
        ax.axis('off')

    fig.suptitle(f'Denoising Effect Visualization - Epoch {epoch}', fontsize=14, weight='bold')
    plt.tight_layout()

    # 保存
    save_path = save_dir / f'denoising_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"[Visualizer] Saved denoising visualization: {save_path}")
    plt.close()

    model.train()  # 恢复训练模式


def run_retrieval_evaluation(model, config, device, logger):
    """
    运行完整的检索评估（效仿原项目）

    Args:
        model: CLaV模型（应该已经是model.module，不包含DDP包装）
        config: 完整配置字典
        device: 设备
        logger: 日志记录器

    Returns:
        metrics: 评估指标字典
    """
    from evaluation.evaluate import evaluate_weather_condition

    # 确保模型不是DDP包装的
    if hasattr(model, 'module'):
        model = model.module
    
    # 设置为eval模式
    model.eval()

    # 从配置中获取评估pickle路径
    dataset_folder = config['data']['dataset_folder']
    eval_config = config.get('evaluation', {})

    # 默认评估设置：orin作为database，fog/rain/snow作为query
    weather_conditions = eval_config.get('weather_conditions', ['fog', 'rain', 'snow'])

    # 累积指标
    all_recalls = {}
    total_queries = 0

    for weather in weather_conditions:
        # 构造pickle路径
        # database_pickle = f"{dataset_folder}/kitti_bev_orin_eval_database.pkl"
        # query_pickle = f"{dataset_folder}/kitti_bev_{weather}_eval_query.pkl"
        database_pickle = f"{dataset_folder}/data/kitti_bev_{weather}_evaluation_database.pickle"
        query_pickle = f"{dataset_folder}/data/kitti_bev_{weather}_evaluation_query.pickle"
        try:
            logger.log(f"  Evaluating {weather}...")

            # 运行评估
            metrics = evaluate_weather_condition(
                model=model,
                database_pickle=database_pickle,
                query_pickle=query_pickle,
                device=device,
                config=config,
                weather_name=weather
            )

            # 累积recall
            # evaluate_weather_condition返回'num_query'（单数）
            num_queries = metrics.get('num_query', 0)
            total_queries += num_queries

            for k in [1, 5, 10, 20]:
                recall_key = f'recall@{k}'
                if recall_key in metrics:
                    if recall_key not in all_recalls:
                        all_recalls[recall_key] = 0.0
                    all_recalls[recall_key] += metrics[recall_key] * num_queries

            logger.log(f"    Recall@1: {metrics.get('recall@1', 0):.2f}%")

        except FileNotFoundError:
            logger.log(f"    Warning: {weather} pickle files not found, skipping")
        except Exception as e:
            logger.log(f"    Error: {weather} evaluation failed - {e}")

    # 计算平均recall
    avg_metrics = {}
    if total_queries > 0:
        for recall_key, total_recall in all_recalls.items():
            avg_metrics[recall_key] = total_recall / total_queries

    logger.log(f"\n  Average Recall@1: {avg_metrics.get('recall@1', 0):.2f}%")
    logger.log(f"  Average Recall@5: {avg_metrics.get('recall@5', 0):.2f}%")
    logger.log(f"  Average Recall@10: {avg_metrics.get('recall@10', 0):.2f}%")

    model.train()

    return avg_metrics


def train_one_epoch_joint(
    model,
    dataloader,
    loss_fn,
    optimizer,
    epoch,
    config,
    logger,
    device
):
    """
    联合训练一个epoch - 同时训练扩散模型和描述符头

    Args:
        model: CLaV模型
        dataloader: 联合训练数据加载器(DenoisingBEVDataset)
        loss_fn: 描述符损失函数
        optimizer: 优化器
        epoch: 当前epoch
        config: 配置字典
        logger: 日志记录器
        device: 设备

    Returns:
        dict: 包含loss, diffusion_loss, descriptor_loss, ap, recall@1
    """
    model.train()

    # 损失权重
    diffusion_weight = config.get('diffusion_loss', {}).get('weight', 1.0)
    descriptor_weight = config.get('descriptor_loss', {}).get('weight', 1.0)

    # 统计meters
    loss_meter = AverageMeter()
    diffusion_loss_meter = AverageMeter()
    descriptor_loss_meter = AverageMeter()
    ap_meter = AverageMeter()
    recall1_meter = AverageMeter()
    grad_accum_steps = config['grad_accum_steps']

    # 进度条（仅主进程显示）
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f'Epoch {epoch} (Joint)',
        ncols=140,
        disable=bool(int(os.environ.get('RANK', '0')) != 0)
    )

    optimizer.zero_grad()

    for batch_idx, batch in pbar:
        # 解包batch
        # DenoisingBEVDataset返回: (noisy_imgs, clean_imgs, pos_mask, neg_mask, indices, weather)
        noisy_imgs, clean_imgs, positives_mask, negatives_mask, indices, weather = batch

        # 移动到device
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # 获取空间加权配置
        diffusion_loss_config = config.get('diffusion_loss', {})
        use_spatial_weighting = diffusion_loss_config.get('use_spatial_weighting', True)
        bg_weight = diffusion_loss_config.get('bg_weight', 0.1)
        bg_threshold = diffusion_loss_config.get('bg_threshold', 10.0)

        # 联合前向传播（支持空间加权）
        descriptors, diffusion_loss, _ = model.forward_with_denoising_loss(
            noisy_imgs, clean_imgs,
            return_intermediate=False,
            use_spatial_weighting=use_spatial_weighting,
            bg_weight=bg_weight,
            bg_threshold=bg_threshold
        )

        # 计算描述符损失
        loss_fn_truncated, loss_fn_infonce = loss_fn
        descriptor_loss, stats = loss_fn_truncated(descriptors, positives_mask, negatives_mask)

        # 组合损失
        total_loss = diffusion_weight * diffusion_loss + descriptor_weight * descriptor_loss

        # 梯度累积
        total_loss = total_loss / grad_accum_steps
        total_loss.backward()

        # 更新参数
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            if config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['gradient_clip']
                )

            optimizer.step()
            optimizer.zero_grad()

        # 记录损失和统计
        loss_meter.update(total_loss.item() * grad_accum_steps, noisy_imgs.size(0))
        diffusion_loss_meter.update(diffusion_loss.item(), noisy_imgs.size(0))
        descriptor_loss_meter.update(descriptor_loss.item(), noisy_imgs.size(0))
        ap_meter.update(stats.get('ap', 0.0), noisy_imgs.size(0))
        recall1_meter.update(stats.get('recall', {}).get(1, 0.0) * 100, noisy_imgs.size(0))

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'diff': f'{diffusion_loss_meter.avg:.4f}',
            'desc': f'{descriptor_loss_meter.avg:.4f}',
            'AP': f'{ap_meter.avg:.3f}',
            'R@1': f'{recall1_meter.avg:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # 定期日志
        if (batch_idx + 1) % config.get('log_interval', 50) == 0 and int(os.environ.get('RANK', '0')) == 0:
            logger.log(
                f'Epoch [{epoch}][{batch_idx+1}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'(Diff: {diffusion_loss_meter.avg:.4f}, Desc: {descriptor_loss_meter.avg:.4f}) '
                f'AP: {ap_meter.avg:.4f} '
                f'Recall@1: {recall1_meter.avg:.2f}% '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # 返回epoch统计
    return {
        'loss': loss_meter.avg,
        'diffusion_loss': diffusion_loss_meter.avg,
        'descriptor_loss': descriptor_loss_meter.avg,
        'ap': ap_meter.avg,
        'recall@1': recall1_meter.avg
    }


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    epoch,
    config,
    logger,
    device
):
    """
    训练一个epoch

    Args:
        model: CLaV模型
        dataloader: 训练数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    model.train()

    # 训练时跳过扩散去噪（加速）
    # 从model配置中读取skip_denoising设置
    skip_denoising = config.get('model', {}).get('skip_denoising', True)

    loss_meter = AverageMeter()
    ap_meter = AverageMeter()
    recall1_meter = AverageMeter()
    grad_accum_steps = config['grad_accum_steps']

    # 进度条（仅主进程显示）
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f'Epoch {epoch}',
        ncols=120,
        disable=bool(int(os.environ.get('RANK', '0')) != 0)
    )

    optimizer.zero_grad()

    for batch_idx, batch in pbar:
        # 解包batch
        # batch包含: (batch_data, positives_mask, negatives_mask, sampled_pairs, positive_pairs)
        batch_data, positives_mask, negatives_mask, sampled_pairs, positive_pairs = batch

        # 处理batch_data (可能是列表或tensor)
        if isinstance(batch_data, list):
            if len(batch_data) == 1:
                # 单个sub-batch
                images = batch_data[0].to(device)
            else:
                # 多个sub-batches，合并
                images = torch.cat([b.to(device) for b in batch_data], dim=0)
        else:
            # 直接是tensor
            images = batch_data.to(device)

        # 前向传播 - 跳过扩散去噪加速训练
        descriptors = model(images, skip_denoising=skip_denoising)

        # 计算损失
        # loss_fn是tuple: (truncated_smoothap, pointinfonce)
        loss_fn_truncated, loss_fn_infonce = loss_fn
        loss, stats = loss_fn_truncated(descriptors, positives_mask.to(device), negatives_mask.to(device))

        # 梯度累积
        loss = loss / grad_accum_steps
        loss.backward()

        # 更新参数
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            if config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['gradient_clip']
                )

            optimizer.step()
            optimizer.zero_grad()

        # 记录损失和统计
        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))
        ap_meter.update(stats.get('ap', 0.0), images.size(0))
        recall1_meter.update(stats.get('recall', {}).get(1, 0.0) * 100, images.size(0))

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'AP': f'{ap_meter.avg:.3f}',
            'R@1': f'{recall1_meter.avg:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # 定期日志
        if (batch_idx + 1) % config.get('log_interval', 50) == 0 and int(os.environ.get('RANK', '0')) == 0:
            logger.log(
                f'Epoch [{epoch}][{batch_idx+1}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'AP: {ap_meter.avg:.4f} '
                f'Recall@1: {recall1_meter.avg:.2f}% '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # 返回epoch统计
    return {
        'loss': loss_meter.avg,
        'ap': ap_meter.avg,
        'recall@1': recall1_meter.avg
    }


@torch.no_grad()
def validate_joint(model, dataloader, loss_fn, epoch, config, logger, device):
    """
    联合训练验证

    Args:
        model: CLaV模型
        dataloader: 联合训练验证数据加载器
        loss_fn: 损失函数
        epoch: 当前epoch
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    model.eval()

    # 损失权重
    diffusion_weight = config.get('diffusion_loss', {}).get('weight', 1.0)
    descriptor_weight = config.get('descriptor_loss', {}).get('weight', 1.0)

    loss_meter = AverageMeter()
    diffusion_loss_meter = AverageMeter()
    descriptor_loss_meter = AverageMeter()
    ap_meter = AverageMeter()
    recall1_meter = AverageMeter()

    # 获取空间加权配置（验证时也使用相同配置）
    diffusion_loss_config = config.get('diffusion_loss', {})
    use_spatial_weighting = diffusion_loss_config.get('use_spatial_weighting', True)
    bg_weight = diffusion_loss_config.get('bg_weight', 0.1)
    bg_threshold = diffusion_loss_config.get('bg_threshold', 10.0)

    for batch in tqdm(dataloader, desc=f'Val Epoch {epoch} (Joint)', ncols=140, disable=bool(int(os.environ.get('RANK', '0')) != 0)):
        # 解包batch
        noisy_imgs, clean_imgs, positives_mask, negatives_mask, indices, weather = batch

        # 移动到device
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # 联合前向传播（支持空间加权）
        descriptors, diffusion_loss, _ = model.forward_with_denoising_loss(
            noisy_imgs, clean_imgs,
            return_intermediate=False,
            use_spatial_weighting=use_spatial_weighting,
            bg_weight=bg_weight,
            bg_threshold=bg_threshold
        )

        # 计算描述符损失
        loss_fn_truncated, loss_fn_infonce = loss_fn
        descriptor_loss, stats = loss_fn_truncated(descriptors, positives_mask, negatives_mask)

        # 检查是否有有效的正样本（避免NaN）
        has_valid_positives = positives_mask.sum() > 0

        # 组合损失
        if has_valid_positives and not torch.isnan(descriptor_loss):
            total_loss = diffusion_weight * diffusion_loss + descriptor_weight * descriptor_loss
            descriptor_loss_value = descriptor_loss.item()
            ap_value = stats.get('ap', 0.0)
            recall1_value = stats.get('recall', {}).get(1, 0.0) * 100
        else:
            # 如果没有正样本，只计算diffusion loss
            total_loss = diffusion_weight * diffusion_loss
            descriptor_loss_value = float('nan')
            ap_value = float('nan')
            recall1_value = stats.get('recall', {}).get(1, 0.0) * 100  # Recall@1仍然有效

        loss_meter.update(total_loss.item(), noisy_imgs.size(0))
        diffusion_loss_meter.update(diffusion_loss.item(), noisy_imgs.size(0))

        # 只在有效时更新descriptor相关指标
        if not torch.isnan(torch.tensor(descriptor_loss_value)):
            descriptor_loss_meter.update(descriptor_loss_value, noisy_imgs.size(0))
            ap_meter.update(ap_value, noisy_imgs.size(0))

        # Recall@1总是更新（即使没有正样本，也能计算）
        recall1_meter.update(recall1_value, noisy_imgs.size(0))

    # 格式化输出（处理可能的NaN）
    desc_str = f'{descriptor_loss_meter.avg:.4f}' if descriptor_loss_meter.count > 0 else 'N/A'
    ap_str = f'{ap_meter.avg:.4f}' if ap_meter.count > 0 else 'N/A'

    if int(os.environ.get('RANK', '0')) == 0:
        logger.log(
        f'Validation Epoch [{epoch}] '
        f'Loss: {loss_meter.avg:.4f} '
        f'(Diff: {diffusion_loss_meter.avg:.4f}, Desc: {desc_str}) '
        f'AP: {ap_str} '
        f'Recall@1: {recall1_meter.avg:.2f}%'
        )

    return {
        'loss': loss_meter.avg,
        'diffusion_loss': diffusion_loss_meter.avg,
        'descriptor_loss': descriptor_loss_meter.avg if descriptor_loss_meter.count > 0 else float('nan'),
        'ap': ap_meter.avg if ap_meter.count > 0 else float('nan'),
        'recall@1': recall1_meter.avg
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, epoch, config, logger, device):
    """
    验证

    Args:
        model: CLaV模型
        dataloader: 验证数据加载器
        loss_fn: 损失函数
        epoch: 当前epoch
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    model.eval()

    loss_meter = AverageMeter()
    ap_meter = AverageMeter()
    recall1_meter = AverageMeter()

    # 验证时执行完整去噪
    for batch in tqdm(dataloader, desc=f'Val Epoch {epoch}', ncols=120, disable=bool(int(os.environ.get('RANK', '0')) != 0)):
        # 解包batch
        batch_data, positives_mask, negatives_mask, sampled_pairs, positive_pairs = batch

        # 处理batch_data (可能是列表或tensor)
        if isinstance(batch_data, list):
            if len(batch_data) == 1:
                images = batch_data[0].to(device)
            else:
                images = torch.cat([b.to(device) for b in batch_data], dim=0)
        else:
            images = batch_data.to(device)

        # 前向传播 - 使用与训练相同的skip_denoising设置
        skip_denoising = config.get('model', {}).get('skip_denoising', True)
        descriptors = model(images, skip_denoising=skip_denoising)

        # 计算损失
        loss_fn_truncated, loss_fn_infonce = loss_fn
        loss, stats = loss_fn_truncated(descriptors, positives_mask.to(device), negatives_mask.to(device))
        loss_meter.update(loss.item(), images.size(0))
        ap_meter.update(stats.get('ap', 0.0), images.size(0))
        recall1_meter.update(stats.get('recall', {}).get(1, 0.0) * 100, images.size(0))

    if int(os.environ.get('RANK', '0')) == 0:
        logger.log(
        f'Validation Epoch [{epoch}] '
        f'Loss: {loss_meter.avg:.4f} '
        f'AP: {ap_meter.avg:.4f} '
        f'Recall@1: {recall1_meter.avg:.2f}%'
        )

    return {
        'loss': loss_meter.avg,
        'ap': ap_meter.avg,
        'recall@1': recall1_meter.avg
    }


def main():
    # ===== 参数解析 =====
    parser = argparse.ArgumentParser(description='Train Descriptor Head')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Training config file')
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml',
                        help='Base config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (fewer iterations)')
    args = parser.parse_args()

    # ===== 分布式初始化 =====
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    local_rank_env = os.environ.get('LOCAL_RANK')

    if distributed:
        if local_rank_env is None:
            raise RuntimeError('DDP: LOCAL_RANK not set. Please launch with torchrun/torch.distributed.run')
        local_rank = int(local_rank_env)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        local_rank = 0

    is_main_process = (int(os.environ.get('RANK', '0')) == 0)

    # ===== 加载配置 =====
    print("[Train] Loading configuration...")

    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)

    # 深度合并配置的辅助函数
    def deep_merge(base, override):
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # 合并配置（深度合并）
    config = deep_merge(base_config, train_config)

    # 兼容多种配置格式: 'stage2_training', 'stage3_training', 'training'
    if 'stage2_training' in config:
        stage2_config = config['stage2_training']
        config['training'] = stage2_config
    elif 'stage3_training' in config:
        stage2_config = config['stage3_training']
        config['training'] = stage2_config
    elif 'training' in config:
        stage2_config = config['training']
    else:
        raise KeyError("Configuration must contain one of: 'stage2_training', 'stage3_training', or 'training'")

    # 合并data_override（如果存在）
    if 'data_override' in train_config:
        print(f"[Train] Applying data overrides from config")
        for key, value in train_config['data_override'].items():
            config['data'][key] = value
            print(f"  {key}: {value}")

    # 保存融合后的完整配置到变量，用于后续保存到checkpoint
    merged_config = config.copy()

    # 设置设备
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Train] Using device: {device}")

    # 设置随机种子
    seed = train_config.get('common', {}).get('seed', 42)
    set_seed(seed)
    print(f"[Train] Random seed: {seed}")

    # ===== 创建保存目录(添加时间戳) =====
    base_save_dir = Path(stage2_config['save_dir'])

    # 添加时间戳到保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = stage2_config.get('experiment_name', 'training')

    # 如果save_dir已经包含实验名称,则直接添加时间戳
    # 否则创建新的路径: base_save_dir / experiment_name_timestamp
    if experiment_name in str(base_save_dir):
        # save_dir已包含实验名称,添加时间戳后缀
        save_dir = Path(str(base_save_dir) + f"_{timestamp}")
    else:
        # 创建新路径
        save_dir = base_save_dir.parent / f"{base_save_dir.name}_{timestamp}"

    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Train] Save directory: {save_dir}")

        # 保存融合后的完整配置到yaml文件
        config_save_path = save_dir / 'merged_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)
        print(f"[Train] Merged config saved to: {config_save_path}")

    # 创建日志记录器
    class _DummyLogger:
        def log(self, *args, **kwargs):
            return
        def log_metrics(self, *args, **kwargs):
            return
        def get_metrics(self):
            return {}

    logger = Logger(save_dir / 'logs') if is_main_process else _DummyLogger()
    logger.log("=" * 80)
    logger.log("Training Descriptor Head - Stage 2")
    logger.log("=" * 80)

    # ===== 创建模型 =====
    print("[Train] Creating CLaV model...")
    model = CLaV(config)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 应用冻结策略
    freeze_config = stage2_config['freeze']
    (model.module if isinstance(model, DDP) else model).apply_freeze_strategy(freeze_config)

    # 统计参数
    param_stats = count_parameters(model.module if isinstance(model, DDP) else model)
    if is_main_process:
        logger.log(f"\nModel Parameters:")
        logger.log(f"  Total: {param_stats['total']:,}")
        logger.log(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_ratio']*100:.2f}%)")
        logger.log(f"  Frozen: {param_stats['frozen']:,}")

        # 显示skip_denoising设置
        skip_denoising_config = config.get('model', {}).get('skip_denoising', True)
        logger.log(f"\nDenoising Configuration:")
        logger.log(f"  Skip denoising: {skip_denoising_config}")
        logger.log(f"  Mode: {'Training on noisy latents (Stage3)' if skip_denoising_config else 'Training with full denoising (Stage2/Joint)'}")

        # 显示空间加权配置
        diffusion_loss_config = stage2_config.get('diffusion_loss', {})
        use_spatial_weighting = diffusion_loss_config.get('use_spatial_weighting', True)
        if not skip_denoising_config and use_spatial_weighting:
            logger.log(f"\nSpatial Weighting Configuration:")
            logger.log(f"  Enabled: {use_spatial_weighting}")
            logger.log(f"  Background weight: {diffusion_loss_config.get('bg_weight', 0.1)}")
            logger.log(f"  Background threshold: {diffusion_loss_config.get('bg_threshold', 10.0)}")

    # ===== 创建数据加载器 =====
    print("[Train] Creating data loaders...")

    # 检查训练模式
    data_mode = stage2_config.get('data_mode', 'descriptor_only')  # 'descriptor_only' 或 'denoising_pairs'

    if data_mode == 'denoising_pairs':
        # 联合训练模式 - 使用配对的(noisy, clean)数据
        print("[Train] Using joint training mode with paired denoising data")

        denoising_pickle = stage2_config.get('denoising_pickle', 'data/bev_denoising_tuples_448.pkl')
        position_pickle = config['data'].get('train_pickle', None)

        # Get augmentation config (pass to dataset)
        augmentation_config = stage2_config.get('augmentation', None)

        if position_pickle and os.path.exists(position_pickle):
            # 使用带位置信息的数据集
            train_dataset = DenoisingBEVDatasetWithPositions(
                denoising_pickle,
                position_pickle,
                split='train',
                image_size=config['data']['image_size'],
                augmentation_config=augmentation_config  # Pass augmentation config
            )
            val_dataset = DenoisingBEVDatasetWithPositions(
                denoising_pickle,
                position_pickle,
                split='val',
                image_size=config['data']['image_size'],
                augmentation_config=None  # No augmentation for validation
            ) if stage2_config.get('eval_interval', 0) > 0 else None
        else:
            # 使用基础数据集（位置信息从metadata中获取）
            train_dataset = DenoisingBEVDataset(
                denoising_pickle,
                split='train',
                image_size=config['data']['image_size'],
                augmentation_config=augmentation_config  # Pass augmentation config
            )
            val_dataset = DenoisingBEVDataset(
                denoising_pickle,
                split='val',
                image_size=config['data']['image_size'],
                augmentation_config=None  # No augmentation for validation
            ) if stage2_config.get('eval_interval', 0) > 0 else None

        # 分片训练集（Subset）保持BatchSampler逻辑
        if distributed:
            total_len = len(train_dataset)
            indices = list(range(total_len))[int(os.environ.get('RANK', '0'))::world_size]
            from torch.utils.data import Subset
            train_dataset_shard = Subset(train_dataset, indices)
        else:
            train_dataset_shard = train_dataset

        # 检查是否需要BatchSampler（仅联合训练描述符头时需要）
        freeze_descriptor = stage2_config['freeze'].get('freeze_descriptor', False)
        use_batch_sampler = not freeze_descriptor  # 只有训练描述符时才需要正样本对

        if is_main_process:
            print(f"[Train] Freeze descriptor: {freeze_descriptor}")
            print(f"[Train] Use batch sampler (for positive pairs): {use_batch_sampler}")

        train_dataloader = make_denoising_dataloader(
            train_dataset_shard,
            batch_size=stage2_config['batch_size'],
            shuffle=not distributed,
            num_workers=config['data'].get('num_workers', 16),
            positive_threshold=10.0,
            negative_threshold=50.0,
            use_batch_sampler=use_batch_sampler
        )

        if is_main_process and val_dataset is not None:
            val_dataloader = make_denoising_dataloader(
                val_dataset,
                batch_size=stage2_config['batch_size'],
                shuffle=False,
                num_workers=config['data'].get('num_workers', 16),
                positive_threshold=10.0,
                negative_threshold=50.0,
                use_batch_sampler=False
            )
        else:
            val_dataloader = None

        dataloaders = {'train': train_dataloader}
        if val_dataloader is not None:
            dataloaders['val'] = val_dataloader

        if is_main_process:
            logger.log(f"\nData Loaders (Joint Training):")
            logger.log(f"  Train batches: {len(dataloaders['train'])}")
            if 'val' in dataloaders and dataloaders['val'] is not None:
                logger.log(f"  Val batches: {len(dataloaders['val'])}")

        # 设置训练和验证函数
        train_fn = train_one_epoch_joint
        validate_fn = validate_joint

    else:
        # 描述符训练模式 - 使用原始ImLPR数据
        print("[Train] Using descriptor-only training mode")

        # 使用ImLPR的数据加载器
        # 需要创建TrainingParams对象
        dataset_folder = config['data']['dataset_folder']
        train_file = config['data']['train_pickle']

        # 创建临时配置文件（兼容ImLPR的数据加载器）
        import configparser
        import tempfile

        temp_config = configparser.ConfigParser()
        temp_config['DEFAULT'] = {
            'dataset_folder': dataset_folder,
        }
        temp_config['TRAIN'] = {
            'num_workers': str(config['data'].get('num_workers', 16)),
            'batch_size': str(stage2_config['batch_size']),
            'val_batch_size': str(stage2_config['batch_size']),
            'train_file': train_file,
            'val_file': '',  # 可选
            'loss': stage2_config['loss']['type'],
            'positives_per_query': str(stage2_config['loss']['positives_per_query']),
            'tau1': str(stage2_config['loss']['tau1']),
            'image_H': str(config['data']['image_size']),
            'image_W': str(config['data']['image_size']),
            'similarity': 'euclidean',
        }

        # 保存临时配置
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_config.write(f)
            temp_config_path = f.name

        # 创建模型配置
        temp_model_config = configparser.ConfigParser()
        temp_model_config['BACKBONE'] = {
            'model_name': 'dinov2_vitb14',
            'num_trainable_blocks': '0',
            'adapter_frequency': '3'
        }
        temp_model_config['AGGREGATOR'] = {
            'num_channels': str(config['model']['descriptor_head']['latent_dim']),
            'num_clusters': str(config['model']['descriptor_head']['num_clusters']),
            'cluster_dim': str(config['model']['descriptor_head']['cluster_dim']),
            'token_dim': str(config['model']['descriptor_head']['token_dim']),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_model_config.write(f)
            temp_model_config_path = f.name

        # 创建TrainingParams
        params = TrainingParams(temp_config_path, temp_model_config_path)

        # 创建数据加载器
        dataloaders = make_dataloaders(params)
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data import DataLoader as TorchDataLoader

            def rebuild_loader(loader, shuffle=True):
                if not hasattr(loader, 'dataset'):
                    return loader
                try:
                    sampler = DistributedSampler(loader.dataset, shuffle=shuffle, drop_last=True)
                    return TorchDataLoader(
                        loader.dataset,
                        batch_size=loader.batch_size,
                        sampler=sampler,
                        num_workers=loader.num_workers,
                        collate_fn=getattr(loader, 'collate_fn', None),
                        pin_memory=getattr(loader, 'pin_memory', True),
                        drop_last=True
                    )
                except Exception:
                    return loader

            dataloaders['train'] = rebuild_loader(dataloaders['train'], shuffle=True)
            if 'val' in dataloaders and dataloaders['val'] is not None:
                if is_main_process:
                    dataloaders['val'] = rebuild_loader(dataloaders['val'], shuffle=False)
                else:
                    dataloaders['val'] = None

        # 清理临时文件
        os.unlink(temp_config_path)
        os.unlink(temp_model_config_path)

        if is_main_process:
            logger.log(f"\nData Loaders (Descriptor Only):")
            logger.log(f"  Train batches: {len(dataloaders['train'])}")
            if 'val' in dataloaders and dataloaders['val'] is not None:
                logger.log(f"  Val batches: {len(dataloaders['val'])}")

        # 设置训练和验证函数
        train_fn = train_one_epoch
        validate_fn = validate

    # ===== 创建损失函数 =====
    print("[Train] Creating loss function...")
    if data_mode == 'denoising_pairs':
        # 联合训练模式 - 需要创建params用于loss函数
        import configparser
        import tempfile

        temp_config = configparser.ConfigParser()
        temp_config['DEFAULT'] = {'dataset_folder': config['data']['dataset_folder']}
        temp_config['TRAIN'] = {
            'loss': stage2_config['loss']['type'],
            'positives_per_query': str(stage2_config['loss']['positives_per_query']),
            'tau1': str(stage2_config['loss']['tau1']),
            'similarity': 'euclidean',
        }
        temp_model_config = configparser.ConfigParser()
        temp_model_config['BACKBONE'] = {'model_name': 'dinov2_vitb14'}
        temp_model_config['AGGREGATOR'] = {'num_channels': '768'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_config.write(f)
            temp_config_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_model_config.write(f)
            temp_model_config_path = f.name

        params = TrainingParams(temp_config_path, temp_model_config_path)
        loss_fn = make_losses(params)

        os.unlink(temp_config_path)
        os.unlink(temp_model_config_path)
    else:
        loss_fn = make_losses(params)

    # ===== 创建优化器和调度器 =====
    print("[Train] Setting up optimizer and scheduler...")
    optimizer = setup_optimizer(model.module if isinstance(model, DDP) else model, stage2_config)
    scheduler = setup_scheduler(optimizer, stage2_config)

    if is_main_process:
        logger.log(f"\nOptimizer: {optimizer.__class__.__name__}")
        logger.log(f"Scheduler: {scheduler.__class__.__name__}")

    # ===== EMA（可选）=====
    ema = None
    if stage2_config.get('use_ema', False):
        base_model = model.module if isinstance(model, DDP) else model
        ema = EMA(base_model, decay=stage2_config.get('ema_decay', 0.9995))
        if is_main_process:
            logger.log(f"EMA enabled with decay={ema.decay}")

    # ===== Early Stopping（可选）=====
    early_stopping = None
    if stage2_config.get('early_stopping', {}).get('enabled', False):
        es_config = stage2_config['early_stopping']
        early_stopping = EarlyStopping(
            patience=es_config.get('patience', 30),
            min_delta=es_config.get('min_delta', 0.3),
            monitor=es_config.get('monitor', 'eval_avg_recall@1'),
            mode='max' if 'recall' in es_config.get('monitor', '') else 'min'
        )
        if is_main_process:
            logger.log(f"\nEarly Stopping enabled:")
            logger.log(f"  Monitor: {early_stopping.monitor}")
            logger.log(f"  Patience: {early_stopping.patience}")
            logger.log(f"  Min Delta: {early_stopping.min_delta}%")

    # ===== 恢复训练（如果指定）=====
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_recall = 0.0  # 用于基于recall保存最佳模型

    if args.resume:
        start_epoch, checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler, device, strict=False
        )
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # ===== 实验追踪器 =====
    use_tensorboard = stage2_config.get('use_tensorboard', False)
    tracker = ExperimentTracker(
        save_dir=save_dir,
        experiment_name=stage2_config.get('experiment_name', 'training'),
        use_tensorboard=use_tensorboard
    )
    logger.log(f"\nExperiment Tracker initialized")
    logger.log(f"  TensorBoard: {'Enabled' if use_tensorboard else 'Disabled'}")

    # ===== 训练循环 =====
    if is_main_process:
        logger.log("\n" + "=" * 80)
        logger.log("Starting Training")
        logger.log("=" * 80)

    num_epochs = stage2_config['epochs']
    eval_interval = stage2_config.get('eval_interval', 5)
    save_interval = stage2_config.get('save_interval', 5)

    # 初始化最佳loss追踪（用于保存最佳模型）
    best_train_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # 训练 - 使用动态选择的训练函数
        train_stats = train_fn(
            model.module if isinstance(model, DDP) else model, dataloaders['train'], loss_fn, optimizer,
            epoch, stage2_config, logger, device
        )

        # ===== 完整检索评估 (已禁用) =====
        # 训练阶段禁用评估以避免NCCL超时问题
        # 只在训练结束后单独运行评估脚本进行验证
        eval_stats = None

        # 更新调度器（基于训练loss）
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_stats['loss'])
            else:
                scheduler.step()

        # 更新EMA
        if ema is not None:
            ema.update()

        # 记录指标
        metrics = {
            'train_loss': train_stats['loss'],
            'train_ap': train_stats['ap'],
            'train_recall@1': train_stats['recall@1'],
            'learning_rate': optimizer.param_groups[0]['lr'],
        }

        # 联合训练额外指标
        if data_mode == 'denoising_pairs':
            metrics['train_diffusion_loss'] = train_stats.get('diffusion_loss', 0.0)
            metrics['train_descriptor_loss'] = train_stats.get('descriptor_loss', 0.0)

        # ===== 验证评估（仅主进程，仅descriptor_only模式）=====
        val_recall_at_1 = 0.0
        if is_main_process and data_mode == 'descriptor_only' and (epoch + 1) % eval_interval == 0:
            logger.log(f"\n[Epoch {epoch}] Running validation evaluation...")
            val_recall_at_1 = evaluate_validation_recall(
                model=(model.module if isinstance(model, DDP) else model),
                config=merged_config,
                device=device,
                logger=logger
            )
            metrics['eval_avg_recall@1'] = val_recall_at_1

        if is_main_process:
            logger.log_metrics(epoch, metrics)

        # 记录到tracker
        if is_main_process:
            tracker.log_epoch(epoch, metrics)

        # 定期绘制训练曲线
        plot_interval = stage2_config.get('plot_interval', 5)
        if (epoch + 1) % plot_interval == 0:
            tracker.plot_metrics()

        # 定期可视化去噪效果
        vis_interval = stage2_config.get('vis_denoising_interval', 5)
        if is_main_process and data_mode == 'denoising_pairs' and (epoch + 1) % vis_interval == 0:
            logger.log(f"Visualizing denoising effect at epoch {epoch}...")
            visualize_denoising_samples(
                model=(model.module if isinstance(model, DDP) else model),
                dataloader=dataloaders.get('val', dataloaders['train']),
                save_dir=save_dir / 'denoising_vis',
                epoch=epoch,
                num_samples=3,
                device=device
            )

        # ===== Early Stopping检查（仅descriptor_only模式）=====
        if early_stopping is not None and data_mode == 'descriptor_only' and val_recall_at_1 > 0:
            improved = early_stopping.step(val_recall_at_1)
            if is_main_process:
                if improved:
                    logger.log(f"[Early Stopping] Validation improved: {val_recall_at_1:.2f}% (best: {early_stopping.best_value:.2f}%)")
                else:
                    logger.log(f"[Early Stopping] No improvement for {early_stopping.counter} epochs (patience: {early_stopping.patience})")

                if early_stopping.should_stop():
                    logger.log(f"\n[Early Stopping] Stopping training at epoch {epoch}")
                    logger.log(f"  Best {early_stopping.monitor}: {early_stopping.best_value:.2f}%")

        # 保存检查点
        # descriptor_only模式：基于验证Recall@1保存最佳模型
        # 其他模式：基于训练loss保存
        current_train_loss = train_stats['loss']

        if data_mode == 'descriptor_only' and val_recall_at_1 > 0:
            # 基于验证Recall@1
            is_best = val_recall_at_1 > best_val_recall
            if is_best:
                best_val_recall = val_recall_at_1
        else:
            # 基于训练loss（兼容旧逻辑）
            is_best = current_train_loss < best_train_loss
            if is_best:
                best_train_loss = current_train_loss

        if is_main_process:
            # 每个epoch保存last.pt
            last_path = save_dir / 'last.pt'
            checkpoint_state = {
                'train_loss': train_stats['loss'],
                'val_recall': best_val_recall if data_mode == 'descriptor_only' else None,
                'best_val_loss': best_train_loss,
                'config': merged_config
            }

            # 保存early stopping状态
            if early_stopping is not None:
                checkpoint_state['early_stopping'] = early_stopping.state_dict()

            save_checkpoint(
                (model.module if isinstance(model, DDP) else model), optimizer, scheduler, epoch,
                str(last_path),
                is_best=False,
                **checkpoint_state
            )

            # 保存best.pt
            if is_best:
                best_path = save_dir / 'best.pt'
                save_checkpoint(
                    (model.module if isinstance(model, DDP) else model), optimizer, scheduler, epoch,
                    str(best_path),
                    is_best=True,
                    **checkpoint_state
                )
                if data_mode == 'descriptor_only' and val_recall_at_1 > 0:
                    logger.log(f'[Checkpoint] New best model saved (val_recall@1: {best_val_recall:.2f}%)')
                else:
                    logger.log(f'[Checkpoint] New best model saved (train_loss: {best_train_loss:.4f})')

        epoch_time = time.time() - epoch_start_time
        if is_main_process:
            logger.log(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")

        # ===== Early Stopping中断训练 =====
        if early_stopping is not None and early_stopping.should_stop():
            break

    if is_main_process:
        logger.log("=" * 80)
        logger.log("Training Completed!")
        logger.log(f"Best Train Loss: {best_train_loss:.4f}")
        logger.log("=" * 80)
        logger.log("Note: Evaluation is disabled during training.")
        logger.log("Please run evaluation script separately after training completes.")
        logger.log("=" * 80)

    # 关闭tracker - 保存最终指标和绘制最终曲线
    if is_main_process:
        tracker.close()

    # 结束分布式
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
