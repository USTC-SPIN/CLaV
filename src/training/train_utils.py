#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练工具函数

Created: 2025-10-21 16:10
Author: Claude Code Assistant
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, Optional
from pathlib import Path


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_optimizer(model, config: Dict):
    """
    配置优化器

    支持多学习率（用于联合微调）
    """
    optimizer_config = config['optimizer']
    optimizer_type = optimizer_config['type']

    # 检查是否使用分层学习率（layer-wise learning rates）
    if optimizer_config.get('use_layer_wise_lr', False):
        # 分层学习率设置（新增功能，向后兼容）
        base_lr = optimizer_config['lr']
        encoder_lr = base_lr * optimizer_config.get('encoder_lr_multiplier', 0.1)
        descriptor_lr = base_lr * optimizer_config.get('descriptor_lr_multiplier', 1.0)

        param_groups = []

        # 编码器参数（较低学习率）
        if hasattr(model, 'encoder'):
            encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
            param_groups.append({
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'encoder'
            })

        # Stage2去噪器参数（如果存在）
        if hasattr(model, 'stage2_model'):
            stage2_lr = base_lr * optimizer_config.get('stage2_lr_multiplier', 0.5)
            stage2_params = filter(lambda p: p.requires_grad, model.stage2_model.parameters())
            param_groups.append({
                'params': stage2_params,
                'lr': stage2_lr,
                'name': 'stage2'
            })

        # 描述符头参数（正常学习率）
        if hasattr(model, 'descriptor_head'):
            descriptor_params = filter(lambda p: p.requires_grad, model.descriptor_head.parameters())
            param_groups.append({
                'params': descriptor_params,
                'lr': descriptor_lr,
                'name': 'descriptor_head'
            })

        # 如果没有找到任何参数组，回退到标准方式
        if not param_groups:
            params = filter(lambda p: p.requires_grad, model.parameters())
            param_groups = [{'params': params, 'lr': base_lr}]

        optimizer = getattr(torch.optim, optimizer_type)(
            param_groups,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            eps=optimizer_config.get('eps', 1e-8)
        )

        # 打印学习率信息
        print(f"[Optimizer] Using layer-wise learning rates:")
        for group in param_groups:
            if 'name' in group:
                print(f"  - {group['name']}: lr={group['lr']:.2e}")

    # 检查是否有多学习率配置（原有功能）
    elif 'lr_groups' in optimizer_config:
        # 多学习率设置（阶段3）
        param_groups = []
        for group_config in optimizer_config['lr_groups']:
            group_name = group_config['name']
            lr = group_config['lr']

            if group_name == 'stage2_denoiser':
                params = model.stage2_model.parameters()
            elif group_name == 'descriptor_head':
                params = model.descriptor_head.parameters()
            else:
                raise ValueError(f"Unknown parameter group: {group_name}")

            param_groups.append({
                'params': filter(lambda p: p.requires_grad, params),
                'lr': lr
            })

        optimizer = getattr(torch.optim, optimizer_type)(
            param_groups,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    else:
        # 单学习率设置（阶段2）
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = getattr(torch.optim, optimizer_type)(
            params,
            lr=optimizer_config['lr'],
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )

    return optimizer


def setup_scheduler(optimizer, config: Dict):
    """
    配置学习率调度器
    """
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')

    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10)
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 20),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 1.0e-7)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    save_path: str,
    is_best: bool = False,
    **kwargs
):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 调度器
        epoch: 当前epoch
        save_path: 保存路径
        is_best: 是否是最佳模型
        **kwargs: 其他要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        **kwargs
    }

    # 保存当前检查点
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to {save_path}")

    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = str(Path(save_path).parent / 'best.pt')
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device='cpu',
    strict=False
):
    """
    加载训练检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 调度器（可选）
        device: 设备
        strict: 是否严格匹配参数（默认False，允许部分加载）

    Returns:
        start_epoch: 起始epoch
        checkpoint_dict: 检查点字典
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        return 0, {}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型（使用strict=False允许部分参数不匹配）
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict
        )

        if missing_keys:
            print(f"[Warning] Missing keys in checkpoint ({len(missing_keys)} keys):")
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys)-5} more")
            print("[Info] These parameters will use initialized values")

        if unexpected_keys:
            print(f"[Warning] Unexpected keys in checkpoint ({len(unexpected_keys)} keys):")
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys)-5} more")
            print("[Info] These parameters will be ignored")

    except RuntimeError as e:
        print(f"[Error] Failed to load model state dict: {e}")
        if not strict:
            print("[Info] Trying to load with partial matching...")
            # 尝试部分加载匹配的参数
            model_dict = model.state_dict()
            checkpoint_dict = checkpoint['model_state_dict']

            # 过滤出匹配的参数
            matched_dict = {k: v for k, v in checkpoint_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}

            print(f"[Info] Matched {len(matched_dict)}/{len(model_dict)} parameters")

            # 更新模型参数
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
        else:
            raise

    # 加载优化器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1

    print(f"[Checkpoint] Loaded from {checkpoint_path}")
    print(f"[Checkpoint] Resuming from epoch {start_epoch}")

    return start_epoch, checkpoint


class Logger:
    """简单的日志记录器"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'training.log'
        self.metrics_file = self.log_dir / 'metrics.json'

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }

    def log(self, message: str, print_console: bool = True):
        """记录消息到文件"""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

        if print_console:
            print(message)

    def log_metrics(self, epoch: int, metrics: Dict):
        """记录训练指标"""
        self.metrics['epoch'].append(epoch)

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # 保存到JSON文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics(self):
        """获取所有指标"""
        return self.metrics


def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性算法（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制，用于防止过拟合"""

    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: 多少个epoch没有改善后停止
            min_delta: 最小改善量
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, metric):
        """
        检查是否应该早停

        Args:
            metric: 当前的验证指标

        Returns:
            bool: 是否应该停止训练
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == 'min':
            improved = metric < (self.best - self.min_delta)
        else:
            improved = metric > (self.best + self.min_delta)

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best = None
        self.early_stop = False


class EMA:
    """指数移动平均（Exponential Moving Average）"""

    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    # 测试代码
    print("Testing training utilities...")

    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}, Count: {meter.count}")

    # 测试Logger
    logger = Logger('/tmp/test_log')
    logger.log("Test message")
    logger.log_metrics(1, {'loss': 0.5, 'acc': 0.9})
    print(f"Metrics: {logger.get_metrics()}")

    print("✓ All utility tests passed!")
