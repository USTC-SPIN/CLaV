#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLaV集成模型

"""

import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Dict, Tuple
import yaml

from .stage1.encoders.adapted_dinov2 import AdaptedDINOv2
from .stage1.encoders.adapted_dinov3 import AdaptedDINOv3
from .stage3.latent_descriptor import LatentDescriptorHead
from .stage3.netvlad_descriptor import NetVLADDescriptor
from .stage2.models.conditional_dit import ConditionalDiT
from .stage2.transport.transport import Transport, Sampler


class CLaV(nn.Module):
    """
    CLaV集成模型

    架构:
    噪声BEV图像 (B, 3, 448, 448)
        ↓
    [DINOv2编码器] (AdaptedDINOv2)
        ↓
    噪声潜在表示 (B, 768, 32, 32)
        ↓
    [Stage2扩散去噪 - 50步ODE] (可冻结/可训练)
        ↓
    干净潜在表示 (B, 768, 32, 32)
        ↓
    [描述符提取头 - SALAD聚合] (可训练)
        ↓
    全局描述符 (B, 8448)

    编码器:
    - AdaptedDINOv2: DINOv2 + MultiConvAdapter，部分可训练
    - AdaptedDINOv3: DINOv3 + MultiConvAdapter，支持ViT/ConvNeXt，支持LVD/SAT预训练

    训练策略:
    - 阶段2: 冻结编码器+Stage2, 训练DescriptorHead
    - 阶段3: 冻结/部分解冻编码器, 解冻Stage2最后几层, 联合训练
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含model配置
        """
        super().__init__()

        self.config = config
        model_config = config['model']

        # ===== 1. 编码器初始化 =====
        encoder_type = model_config['stage1'].get('encoder_type', 'dinov2')
        encoder_config = model_config['stage1']['encoder_params']

        if encoder_type == 'dinov3':
            # DINOv3编码器
            print("[CLaV] 初始化AdaptedDINOv3编码器...")
            dinov3_config = encoder_config.get('dinov3', encoder_config)
            self.encoder = AdaptedDINOv3(
                model_name=dinov3_config.get('model_name', 'dinov3_vitb16'),
                pretrain_data=dinov3_config.get('pretrain_data', 'lvd'),
                num_trainable_blocks=dinov3_config.get('num_trainable_blocks', 0),
                adapter_frequency=dinov3_config.get('adapter_frequency', 3),
                input_size=dinov3_config.get('input_size', 512),
                norm_layer=dinov3_config.get('norm_layer', True),
                return_token=dinov3_config.get('return_token', True)
            )
            print(f"  - 使用AdaptedDINOv3编码器")
            print(f"  - 模型: {dinov3_config.get('model_name', 'dinov3_vitb16')}")
            print(f"  - 预训练数据: {dinov3_config.get('pretrain_data', 'lvd')}")
            print(f"  - 输入尺寸: {dinov3_config.get('input_size', 512)}")
        else:
            # 默认DINOv2编码器
            print("[CLaV] 初始化AdaptedDINOv2编码器...")
            self.encoder = AdaptedDINOv2(
                model_name=encoder_config.get('model_name', 'dinov2_vitb14'),
                num_trainable_blocks=encoder_config.get('num_trainable_blocks', 0),
                adapter_frequency=encoder_config.get('adapter_frequency', 3),
                norm_layer=encoder_config.get('norm_layer', True),
                return_token=encoder_config.get('return_token', True)
            )
            print(f"  - 使用AdaptedDINOv2编码器")

        print(f"  - 可训练块数: {encoder_config.get('num_trainable_blocks', 0)}")
        print(f"  - Adapter频率: {encoder_config.get('adapter_frequency', 3)}")
        self.encoder_type = encoder_type

        # ===== 2. Stage2 扩散去噪模型 =====
        print("[CLaV] 加载Stage2扩散模型...")
        self.stage2_config = model_config['stage2']
        self.stage2_model = self._load_stage2_model()

        # ===== 3. 扩散传输与采样器 =====
        print("[CLaV] 初始化扩散传输...")
        self.transport = self._setup_transport()
        self.sampler = Sampler(self.transport)
        self.num_diffusion_steps = self.stage2_config['num_diffusion_steps']
        self.sampling_method = self.stage2_config['sampling_method']

        # 创建采样函数
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.sampling_method,
            num_steps=self.num_diffusion_steps
        )

        # ===== 4. 描述符提取头 =====
        print("[CLaV] 创建描述符提取头...")
        descriptor_config = model_config['descriptor_head']

        # 支持多种描述头类型
        descriptor_type = descriptor_config.get('type', 'SALAD')  # 默认SALAD

        # 定义各描述头类型不需要的参数
        salad_only_params = {'cluster_dim', 'token_dim', 'use_global_token'}  # SALAD特定参数
        common_exclude = {'type'}  # 所有类型都要排除的参数

        if descriptor_type == 'SALAD':
            # SALAD使用所有参数，只移除type
            descriptor_params = {k: v for k, v in descriptor_config.items()
                               if k not in common_exclude}
            self.descriptor_head = LatentDescriptorHead(**descriptor_params)
            print(f"  - 使用SALAD描述头 (Sinkhorn聚合)")
        elif descriptor_type == 'NetVLAD':
            # NetVLAD移除type和SALAD特定参数
            descriptor_params = {k: v for k, v in descriptor_config.items()
                               if k not in (common_exclude | salad_only_params)}
            self.descriptor_head = NetVLADDescriptor(**descriptor_params)
            print(f"  - 使用NetVLAD描述头 (VLAD聚合)")
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")

        # ===== 5. 输出维度 =====
        self.output_dim = self.descriptor_head.get_output_dim()

        print(f"[CLaV] 初始化完成:")
        encoder_name = f"AdaptedDINOv3" if self.encoder_type == 'dinov3' else "AdaptedDINOv2"
        print(f"  - 编码器: {encoder_name} ({self.encoder.num_channels}维)")
        # 安全打印Stage2检查点路径
        stage2_ckpt = self.stage2_config.get('checkpoint_path', '未指定(随机初始化)')
        print(f"  - Stage2检查点: {stage2_ckpt}")
        print(f"  - 扩散步数: {self.num_diffusion_steps}")
        print(f"  - 采样方法: {self.sampling_method}")
        print(f"  - 输出维度: {self.output_dim}")

    def _load_stage2_model(self) -> nn.Module:
        """加载Stage2扩散模型"""
        # 获取checkpoint路径（如果有）
        checkpoint_path = self.stage2_config.get('checkpoint_path', None)

        # 创建模型 - 使用配置文件中的所有参数
        model = ConditionalDiT(
            input_size=self.stage2_config['input_size'],
            patch_size=self.stage2_config['patch_size'],
            in_channels=self.stage2_config['in_channels'],
            hidden_size=self.stage2_config['hidden_size'],
            depth=self.stage2_config['depth'],
            num_heads=self.stage2_config['num_heads'],
            mlp_ratio=self.stage2_config.get('mlp_ratio', 4.0),
            learn_sigma=self.stage2_config.get('learn_sigma', False),
            use_qknorm=self.stage2_config.get('use_qknorm', False),
            use_swiglu=self.stage2_config.get('use_swiglu', True),
            use_rope=self.stage2_config.get('use_rope', True),
            use_rmsnorm=self.stage2_config.get('use_rmsnorm', True),
            wo_shift=self.stage2_config.get('wo_shift', False),
            use_cross_attn=self.stage2_config.get('use_cross_attn', False),
        )

        # 加载权重（如果提供了checkpoint路径）
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"  加载检查点: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')

            # 处理可能的EMA权重
            if 'ema' in state_dict and self.stage2_config.get('use_ema', False):
                state_dict = state_dict['ema']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            model.load_state_dict(state_dict, strict=False)
        else:
            if checkpoint_path:
                print(f"  ⚠ 警告: 检查点不存在 {checkpoint_path}")
            else:
                print(f"  ✓ 未指定检查点，使用随机初始化权重")

        model.eval()
        return model

    def _setup_transport(self) -> Transport:
        """配置扩散传输 - 从配置文件读取参数"""
        from .stage2.transport.transport import ModelType, PathType, WeightType

        # 获取transport配置
        transport_config = self.stage2_config.get('transport', {})

        # 映射配置字符串到枚举类型
        model_type_map = {
            'velocity': ModelType.VELOCITY,
            'noise': ModelType.NOISE,
            'score': ModelType.SCORE
        }
        path_type_map = {
            'linear': PathType.LINEAR,
            'gvp': PathType.GVP,
            'vp': PathType.VP
        }
        loss_type_map = {
            'none': WeightType.NONE,
            'velocity': WeightType.VELOCITY,
            'likelihood': WeightType.LIKELIHOOD
        }

        transport = Transport(
            model_type=model_type_map.get(transport_config.get('model_type', 'velocity'), ModelType.VELOCITY),
            path_type=path_type_map.get(transport_config.get('path_type', 'linear'), PathType.LINEAR),
            loss_type=loss_type_map.get(transport_config.get('loss_type', 'none'), WeightType.NONE),
            time_dist_type=transport_config.get('time_dist_type', 'uniform'),
            time_dist_shift=transport_config.get('time_dist_shift', 1.0),
            train_eps=transport_config.get('train_eps', 1e-3),
            sample_eps=transport_config.get('sample_eps', 1e-3)
        )
        return transport

    def apply_freeze_strategy(self, freeze_config: Dict):
        """
        应用冻结策略

        Args:
            freeze_config: 冻结配置
                - freeze_encoder: bool (冻结编码器)
                - encoder_trainable_blocks: int (编码器可训练块数，仅AdaptedDINOv2)
                - freeze_stage2: bool
                - freeze_descriptor: bool
                - stage2_unfreeze_layers: int (解冻Stage2最后几层)
        """
        # 编码器冻结策略
        if freeze_config.get('freeze_encoder', True):
            # 完全冻结编码器
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # 部分解冻编码器
            num_trainable = freeze_config.get('encoder_trainable_blocks', 0)
            if num_trainable > 0:
                print(f"[CLaV] 解冻编码器最后{num_trainable}层")
                # 冻结所有块
                for param in self.encoder.parameters():
                    param.requires_grad = False
                # 解冻最后几层
                for blk in self.encoder.model.blocks[-num_trainable:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                # 解冻adapters
                for param in self.encoder.multi_conv_adapters.parameters():
                    param.requires_grad = True
            else:
                # 解冻全部编码器
                for param in self.encoder.parameters():
                    param.requires_grad = True

        # Stage2扩散模型
        if freeze_config.get('freeze_stage2', True):
            for param in self.stage2_model.parameters():
                param.requires_grad = False
        else:
            # 解冻最后几层
            unfreeze_layers = freeze_config.get('stage2_unfreeze_layers', 0)
            if unfreeze_layers > 0:
                print(f"[CLaV] 解冻Stage2最后{unfreeze_layers}层")
                # 冻结所有层
                for param in self.stage2_model.parameters():
                    param.requires_grad = False
                # 解冻最后几层
                for layer in self.stage2_model.blocks[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                # 解冻全部Stage2
                for param in self.stage2_model.parameters():
                    param.requires_grad = True

        # 描述符头
        if freeze_config.get('freeze_descriptor', False):
            for param in self.descriptor_head.parameters():
                param.requires_grad = False
        else:
            for param in self.descriptor_head.parameters():
                param.requires_grad = True

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CLaV] 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    @torch.no_grad()
    def denoise_latent(self, z_noisy: torch.Tensor) -> torch.Tensor:
        """
        在潜在空间进行扩散去噪

        Args:
            z_noisy: 噪声图像的潜在表示 (B, 768, 32, 32)

        Returns:
            z_clean: 去噪后的潜在表示 (B, 768, 32, 32)
        """
        # ConditionalDiT期望输入格式: (B, C, H, W)
        # 初始化纯噪声（与z_noisy同样的形状）
        x0 = torch.randn_like(z_noisy)  # (B, 768, 32, 32)

        # ODE求解 - 使用条件去噪
        # cond也应该是(B, C, H, W)格式
        model_kwargs = {'cond': z_noisy}
        trajectory = self.sample_fn(x0, self.stage2_model, **model_kwargs)
        z_clean = trajectory[-1]  # 取最后一步 (B, 768, 32, 32)

        return z_clean

    def forward(
        self,
        x: torch.Tensor,
        skip_denoising: bool = False,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入BEV图像 (B, 3, H, W)
            skip_denoising: 是否跳过扩散去噪（训练时可设为True加速）
            return_intermediate: 是否返回中间结果

        Returns:
            descriptor: 全局描述符 (B, output_dim)
            或
            (descriptor, intermediate_dict) 如果return_intermediate=True
        """
        # 1. 编码到潜在空间
        # AdaptedDINOv2返回(feature_map, token)元组
        z_noisy, _ = self.encoder(x)  # (B, 768, 32, 32), (B, 768)

        # 2. 扩散去噪
        if self.training and skip_denoising:
            # 训练模式且跳过去噪：直接使用噪声潜在表示
            z_clean = z_noisy
        else:
            # 推理模式或训练时不跳过：执行完整去噪
            z_clean = self.denoise_latent(z_noisy)

        # 3. 提取全局描述符
        descriptor = self.descriptor_head(z_clean)

        if return_intermediate:
            intermediate = {
                'z_noisy': z_noisy,
                'z_clean': z_clean,
            }
            return descriptor, intermediate

        return descriptor

    def compute_spatial_weight_mask(
        self,
        images: torch.Tensor,
        latent_shape: tuple,
        bg_weight: float = 0.1,
        bg_threshold: float = 10.0
    ) -> torch.Tensor:
        """
        计算空间加权mask（针对稀疏BEV图像）

        核心思想：背景区域（低强度）权重低，前景区域权重高

        Args:
            images: 原始图像 (B, 3, H, W)
            latent_shape: 潜在表示的形状 (B, C, H, W)
            bg_weight: 背景权重 (0-1)
            bg_threshold: 背景像素阈值

        Returns:
            weight_mask: (B, 1, latent_h, latent_w)
        """
        B, C, latent_h, latent_w = latent_shape

        # 归一化图像到[0, 255]范围（如果需要）
        if images.max() <= 1.0:
            images = images * 255.0

        # 计算像素强度
        intensity = images.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 下采样到潜在空间尺寸
        intensity_down = torch.nn.functional.adaptive_avg_pool2d(
            intensity, (latent_h, latent_w)
        )

        # 背景mask: 强度 < 阈值*3 (RGB三通道)
        bg_mask = intensity_down < (bg_threshold * 3)

        # 权重mask
        weight_mask = torch.ones_like(bg_mask, dtype=images.dtype)
        weight_mask[bg_mask] = bg_weight

        return weight_mask

    def forward_with_denoising_loss(
        self,
        x_noisy: torch.Tensor,
        x_clean: torch.Tensor,
        return_intermediate: bool = False,
        use_spatial_weighting: bool = True,  # 新增：是否使用空间加权
        bg_weight: float = 0.1,  # 新增：背景权重
        bg_threshold: float = 10.0  # 新增：背景阈值
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        联合训练前向传播 - 同时计算扩散损失和描述符

        使用Flow Matching训练扩散模型:
        1. 编码noisy和clean图像到潜在空间
        2. 采样时间步 t ~ U(0,1)
        3. 线性插值: xt = (1-t)*x0 + t*z_clean
        4. 计算真实速度: ut = z_clean - x0
        5. 预测速度并计算加权MSE损失（可选空间加权）
        6. 对noisy latent执行去噪: z_denoised = denoise(z_noisy)
        7. 从去噪后的latent提取描述符

        注意: 步骤6-7确保训练/推理一致性,避免数据泄漏
        - 训练: descriptor从denoise(z_noisy)提取
        - 推理: descriptor从denoise(z_noisy)提取

        Args:
            x_noisy: 带噪声的BEV图像 (恶劣天气) (B, 3, H, W)
            x_clean: 干净的BEV图像 (晴天,用于扩散训练target) (B, 3, H, W)
            return_intermediate: 是否返回中间结果
            use_spatial_weighting: 是否使用空间加权损失
            bg_weight: 背景区域权重（0-1）
            bg_threshold: 背景像素阈值

        Returns:
            descriptor: 全局描述符 (B, output_dim)
            diffusion_loss: Flow Matching损失 (scalar)
            intermediate: 中间结果字典(可选)
        """
        batch_size = x_noisy.size(0)

        # 1. 编码到潜在空间 (编码器始终冻结)
        with torch.no_grad():
            # AdaptedDINOv2返回(feature_map, token)元组
            z_noisy, _ = self.encoder(x_noisy)  # (B, 768, 32, 32)
            z_clean, _ = self.encoder(x_clean)  # (B, 768, 32, 32)

        # 2. Flow Matching训练
        # 采样时间步 t ~ U(0,1)
        t = torch.rand(batch_size, device=z_noisy.device, dtype=z_noisy.dtype)

        # 初始噪声 x0 ~ N(0, I)
        x0 = torch.randn_like(z_clean)

        # 线性插值路径: xt = (1-t)*x0 + t*z_clean
        t_expanded = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * z_clean

        # 真实速度: ut = z_clean - x0
        ut = z_clean - x0

        # 预测速度(conditioned on z_noisy)
        model_kwargs = {'cond': z_noisy}
        predicted_velocity = self.stage2_model(xt, t, **model_kwargs)

        # 计算Flow Matching损失（支持空间加权）
        if use_spatial_weighting:
            # 计算空间权重mask
            weight_mask = self.compute_spatial_weight_mask(
                x_clean,  # 使用干净图像计算mask
                latent_shape=xt.shape,
                bg_weight=bg_weight,
                bg_threshold=bg_threshold
            )

            # 加权MSE损失
            squared_error = (predicted_velocity - ut) ** 2  # (B, C, H, W)
            weighted_error = squared_error * weight_mask  # 应用权重
            diffusion_loss = weighted_error.mean()
        else:
            # 标准MSE损失
            diffusion_loss = torch.nn.functional.mse_loss(predicted_velocity, ut)

        # 3. 执行去噪并提取描述符
        # 修复数据泄漏: descriptor应该从去噪后的noisy latent提取,而不是直接用clean latent
        # 这样训练和推理保持一致
        with torch.no_grad():
            # 对noisy latent执行去噪
            z_denoised = self.denoise_latent(z_noisy)

        # 从去噪后的latent提取描述符
        descriptor = self.descriptor_head(z_denoised)

        if return_intermediate:
            intermediate = {
                'z_noisy': z_noisy,
                'z_clean': z_clean,
                'z_denoised': z_denoised,
                'x0': x0,
                'xt': xt,
                't': t,
                'ut': ut,
                'predicted_velocity': predicted_velocity,
            }
            return descriptor, diffusion_loss, intermediate

        return descriptor, diffusion_loss, None

    def extract_descriptor(self, x: torch.Tensor, skip_denoising: bool = False) -> torch.Tensor:
        """
        提取描述符（推理接口）

        Args:
            x: 输入BEV图像 (B, 3, H, W)
            skip_denoising: 是否跳过扩散去噪（应该与训练时保持一致）

        Returns:
            descriptor: 全局描述符 (B, output_dim)
        """
        with torch.no_grad():
            return self.forward(x, skip_denoising=skip_denoising)

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, **kwargs):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前epoch
            optimizer: 优化器（可选）
            **kwargs: 其他要保存的信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"[CLaV] 检查点已保存: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device='cpu'):
        """
        加载检查点

        Args:
            path: 检查点路径
            device: 设备

        Returns:
            model, checkpoint_dict
        """
        checkpoint = torch.load(path, map_location=device)

        # 创建模型
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"[CLaV] 检查点已加载: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")

        return model, checkpoint


def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_clav():
    """测试CLaV模型"""
    print("Testing CLaV...")

    # 创建测试配置
    config = {
        'model': {
            'stage1': {
                'encoder_config_path': 'facebook/dinov2-with-registers-base',
                'encoder_input_size': 448,
                'normalization_stat_path': None,
                'eps': 1e-5
            },
            'stage2': {
                'checkpoint_path': '../RAE/results/stage2_bev/best.pt',
                'input_size': 32,
                'patch_size': 1,
                'in_channels': 768,
                'hidden_size': 384,
                'depth': 12,
                'num_heads': 6,
                'num_diffusion_steps': 10,  # 少一些用于测试
                'sampling_method': 'euler',
                'use_ema': False
            },
            'descriptor_head': {
                'latent_dim': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
                'use_global_token': True,
                'output_dim': 8448
            }
        }
    }

    # 创建模型
    try:
        model = CLaV(config)
        print(f"\nModel created successfully!")
        print(f"Output dimension: {model.output_dim}")

        # 应用冻结策略
        freeze_config = {
            'freeze_stage1': True,
            'freeze_stage2': True,
            'freeze_descriptor': False
        }
        model.apply_freeze_strategy(freeze_config)

        # 测试前向传播
        x = torch.randn(1, 3, 448, 448)
        descriptor = model(x, skip_denoising=True)  # 跳过去噪加速测试

        print(f"\nForward pass test:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {descriptor.shape}")
        print(f"  L2 norm: {torch.norm(descriptor, p=2, dim=1)}")

        print("\n✓ CLaV test passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_clav()
