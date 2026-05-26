#!/usr/bin/env python3
"""
DDPM Helper Utilities
Created: 2025-11-10 16:20
Purpose: 提供DDPM训练和推理的辅助工具，与现有Transport框架配合使用

本文件实现DDPM的核心组件：
1. 噪声调度（Beta Schedule）
2. 前向过程（加噪）
3. 反向过程辅助函数
4. 与Transport框架的适配

注意：本实现作为Transport框架的补充，主要用于：
- DDPM特定的噪声调度
- 方便的前向过程采样
- 训练时的辅助计算
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DDPMHelper:
    """
    DDPM辅助工具类

    提供DDPM训练和推理所需的辅助函数，与Transport框架配合使用。
    Transport框架负责模型定义和ODE/SDE求解，本类提供DDPM特定的工具。

    主要功能：
    1. 噪声调度：beta_t, alpha_t, alpha_cumprod_t
    2. 前向过程：q(x_t | x_0)
    3. 后验计算：q(x_{t-1} | x_t, x_0)
    4. 训练辅助：loss加权，SNR计算等
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        device: str = 'cuda'
    ):
        """
        初始化DDPM辅助工具

        Args:
            num_timesteps: 扩散步数（训练用，推理可以fewer steps）
            beta_start: beta起始值
            beta_end: beta终止值
            beta_schedule: 调度类型 ('linear', 'cosine', 'quadratic')
            device: 计算设备
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device

        # 计算噪声调度
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alphas_cumprod[:-1]
        ])

        # 后验方差: \tilde{\beta}_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )

        # 后验均值系数
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod)
        )

        # 计算SNR（Signal-to-Noise Ratio）用于loss加权
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

    def _get_beta_schedule(self) -> torch.Tensor:
        """
        获取beta调度

        Returns:
            betas: (num_timesteps,) tensor
        """
        if self.beta_schedule == 'linear':
            # 线性调度（DDPM原始）
            betas = torch.linspace(
                self.beta_start, self.beta_end,
                self.num_timesteps,
                device=self.device
            )

        elif self.beta_schedule == 'cosine':
            # Cosine调度（Improved DDPM）
            # Nichol & Dhariwal, 2021
            steps = self.num_timesteps + 1
            s = 0.008  # offset
            x = torch.linspace(0, self.num_timesteps, steps, device=self.device)
            alphas_cumprod = torch.cos(
                ((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)

        elif self.beta_schedule == 'quadratic':
            # 二次调度
            betas = torch.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5,
                self.num_timesteps,
                device=self.device
            ) ** 2

        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        return betas

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向过程：q(x_t | x_0) 采样

        使用重参数化技巧：
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon

        Args:
            x_start: 干净数据 x_0, shape (B, C, H, W)
            t: 时间步, shape (B,), 值域 [0, num_timesteps-1]
            noise: 可选的预生成噪声

        Returns:
            x_t: 加噪后的数据
            noise: 使用的噪声（如果未提供则生成）
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取 alpha_cumprod_t
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_start.shape)

        # 重参数化
        x_t = (
            torch.sqrt(alpha_cumprod_t) * x_start +
            torch.sqrt(1.0 - alpha_cumprod_t) * noise
        )

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从噪声预测恢复 x_0

        x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)

        Args:
            x_t: 噪声数据 at timestep t
            t: 时间步
            noise: 预测的噪声

        Returns:
            x_0: 预测的干净数据
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        x_0 = (
            x_t - torch.sqrt(1.0 - alpha_cumprod_t) * noise
        ) / torch.sqrt(alpha_cumprod_t)

        return x_0

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        后验分布：q(x_{t-1} | x_t, x_0)

        这是DDPM采样的关键，用于从x_t和预测的x_0计算x_{t-1}

        Args:
            x_start: 预测的 x_0
            x_t: 当前 x_t
            t: 时间步

        Returns:
            posterior_mean: 后验均值
            posterior_variance: 后验方差
            posterior_log_variance_clipped: log方差（clipped避免数值问题）
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        posterior_mean = coef1 * x_start + coef2 * x_t

        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_loss_weights(
        self,
        t: torch.Tensor,
        weighting_type: str = 'none'
    ) -> torch.Tensor:
        """
        获取损失加权（用于训练）

        Args:
            t: 时间步
            weighting_type: 加权类型
                - 'none': 无加权（标准MSE）
                - 'snr': SNR加权
                - 'truncated_snr': Truncated SNR (clip to [0, 5])
                - 'min_snr': Min-SNR加权 (Hang et al., 2023)

        Returns:
            weights: (B,) 损失权重
        """
        if weighting_type == 'none':
            return torch.ones(t.shape[0], device=self.device)

        snr_t = self._extract(self.snr, t, (t.shape[0],))

        if weighting_type == 'snr':
            return snr_t

        elif weighting_type == 'truncated_snr':
            return torch.clamp(snr_t, 0, 5.0)

        elif weighting_type == 'min_snr':
            # Min-SNR gamma weighting
            gamma = 5.0
            return torch.minimum(snr_t, torch.ones_like(snr_t) * gamma)

        else:
            raise ValueError(f"Unknown weighting type: {weighting_type}")

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        提取a[t]并reshape为适合broadcast的形状

        Args:
            a: 系数数组 (num_timesteps,)
            t: 时间步索引 (B,)
            x_shape: 目标形状 (B, C, H, W)

        Returns:
            extracted: (B, 1, 1, 1) 形状的系数
        """
        batch_size = t.shape[0]
        out = a.gather(0, t.long())

        # Reshape to (B, 1, 1, 1) for broadcasting
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))

    def compute_v_prediction_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算v-prediction目标（用于v-prediction训练）

        v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * x_start

        Args:
            x_start: 干净数据 x_0
            noise: 噪声 epsilon
            t: 时间步

        Returns:
            v: v-prediction目标
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod_t)

        v = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start

        return v

    def predict_noise_from_v(
        self,
        x_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        从v-prediction恢复噪声预测

        noise = sqrt(alpha_cumprod_t) * v + sqrt(1 - alpha_cumprod_t) * x_t

        Args:
            x_t: 噪声数据
            v: v-prediction输出
            t: 时间步

        Returns:
            noise: 噪声预测
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod_t)

        noise = sqrt_alpha * v + sqrt_one_minus_alpha * x_t

        return noise

    def get_sampling_timesteps(
        self,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        获取采样时间步（用于DDIM等快速采样）

        Args:
            num_inference_steps: 推理步数（可以少于训练步数）

        Returns:
            timesteps: (num_inference_steps,) 采样时间步
        """
        # 均匀采样
        step = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(
            0, self.num_timesteps, step,
            device=self.device
        ).long()

        return timesteps


# 辅助函数：创建默认DDPM helper
def create_ddpm_helper(
    num_timesteps: int = 1000,
    beta_schedule: str = 'linear',
    device: str = 'cuda'
) -> DDPMHelper:
    """
    创建默认的DDPM helper实例

    Args:
        num_timesteps: 扩散步数
        beta_schedule: 调度类型
        device: 计算设备

    Returns:
        DDPMHelper实例
    """
    return DDPMHelper(
        num_timesteps=num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=beta_schedule,
        device=device
    )
