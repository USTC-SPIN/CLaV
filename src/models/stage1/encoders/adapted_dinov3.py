#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaptedDINOv3编码器
支持ViT和ConvNeXt架构，支持LVD和SAT预训练数据

Created: 2025-12-03 15:30
Author: Claude Code Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from pathlib import Path
import os


# DINOv3 ViT架构 (patch_size=16)
DINOV3_VIT_ARCHS = {
    'dinov3_vits16': 384,
    'dinov3_vits16plus': 384,  # plus版本
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vith16plus': 1280,  # H+ 版本
    'dinov3_vit7b16': 1536,  # 7B版本
}

# DINOv3 ConvNeXt架构
DINOV3_CONVNEXT_ARCHS = {
    'dinov3_convnext_tiny': 768,
    'dinov3_convnext_small': 768,
    'dinov3_convnext_base': 1024,
    'dinov3_convnext_large': 1536,
}

# 合并所有架构
DINOV3_ARCHS = {**DINOV3_VIT_ARCHS, **DINOV3_CONVNEXT_ARCHS}

# 官方权重文件名映射 (用于本地加载)
# 格式: (model_name, pretrain_data) -> 权重文件名
OFFICIAL_WEIGHT_FILES = {
    # ViT + LVD (lvd1689m)
    ('dinov3_vits16', 'lvd'): 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    ('dinov3_vits16plus', 'lvd'): 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
    ('dinov3_vitb16', 'lvd'): 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    ('dinov3_vitl16', 'lvd'): 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    ('dinov3_vith16plus', 'lvd'): 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    ('dinov3_vit7b16', 'lvd'): 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    # ViT + SAT (sat493m) - 只有L和7B有SAT版本
    ('dinov3_vitl16', 'sat'): 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
    ('dinov3_vit7b16', 'sat'): 'dinov3_vit7b16_pretrain_sat493m-a6675841.pth',
    # ConvNeXt + LVD
    ('dinov3_convnext_tiny', 'lvd'): 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
    ('dinov3_convnext_small', 'lvd'): 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
    ('dinov3_convnext_base', 'lvd'): 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
    ('dinov3_convnext_large', 'lvd'): 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
}


def get_local_weights_config():
    """获取本地权重配置"""
    project_dir = Path(__file__).parent.parent.parent.parent.parent  # encoders -> stage1 -> models -> src -> clav
    weights_dir = project_dir / "pretrained_weights" / "dinov3"

    return {
        "weights_dir": weights_dir,  # 存放.pth文件的目录
        "torch_hub_dir": project_dir / "pretrained_weights" / "torch_hub",  # torch.hub缓存
    }


class MultiConvAdapter(nn.Module):
    """
    并行1x1/3x3/5x5卷积适配器
    来自SelaVPR++论文，用于增强多尺度特征表示

    输入: (B, N, C) token序列
    输出: (B, N, C) 增强后的token序列
    """

    def __init__(self, in_channels: int, patch_size: int = 16):
        """
        Args:
            in_channels: 输入通道数（DINOv3特征维度）
            patch_size: patch大小，用于计算特征图尺寸
        """
        super(MultiConvAdapter, self).__init__()
        self.patch_size = patch_size
        out_channels = in_channels // 3

        # 三个并行分支
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_for_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_for_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, H: int) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入token序列 (B, N, C)
            H: 原始图像高度（用于计算特征图尺寸）

        Returns:
            y: 增强后的token序列 (B, N, C)
        """
        B, N, C = x.shape

        # Reshape为2D特征图
        # H // patch_size = 特征图高度
        h = w = H // self.patch_size
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)

        x = self.relu(x)

        # 三个并行分支
        o1 = self.conv1x1(x)  # 1x1卷积
        o3 = self.conv3x3(self.conv1x1_for_3(x))  # 1x1 -> 3x3
        o5 = self.conv5x5(self.conv1x1_for_5(x))  # 1x1 -> 5x5

        # 拼接并添加残差连接
        y = torch.cat([o1, o3, o5], dim=1) + x  # (B, C, h, w)

        # Reshape回token序列
        y = y.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        return y


class AdaptedDINOv3(nn.Module):
    """
    适配的DINOv3编码器

    特性:
    1. 支持ViT和ConvNeXt两种架构
    2. 支持LVD(web图像)和SAT(卫星图像)预训练数据
    3. 输入自动上采样到512x512以保持32x32输出
    4. MultiConvAdapter增强特征
    5. 可选的部分可训练块（后N层可微调）

    输出:
    - feature_map: 局部特征图 (B, C, 32, 32)
    - global_token: 全局token (B, C)
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        pretrain_data: str = 'lvd',
        num_trainable_blocks: int = 0,
        adapter_frequency: int = 3,
        input_size: int = 512,
        norm_layer: bool = True,
        return_token: bool = True
    ):
        """
        Args:
            model_name: DINOv3模型名称
            pretrain_data: 预训练数据类型 ('lvd' 或 'sat')
            num_trainable_blocks: 可训练块数量（从后往前）
            adapter_frequency: adapter频率（每N层一个adapter）
            input_size: 输入图像尺寸（默认512，确保输出32x32）
            norm_layer: 是否应用归一化层
            return_token: 是否返回全局token
        """
        super().__init__()

        assert model_name in DINOV3_ARCHS, f'Unknown model name {model_name}. Available: {list(DINOV3_ARCHS.keys())}'
        assert pretrain_data in ['lvd', 'sat'], f'Unknown pretrain_data {pretrain_data}. Use "lvd" or "sat"'

        self.model_name = model_name
        self.pretrain_data = pretrain_data
        self.input_size = input_size
        self.num_channels = DINOV3_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.is_convnext = 'convnext' in model_name
        self.patch_size = 16  # DINOv3使用patch_size=16

        print(f"  [AdaptedDINOv3] Initializing {model_name} with {pretrain_data} pretrain...")
        print(f"    - Input size: {input_size} (will upsample from 448 if needed)")
        print(f"    - Output channels: {self.num_channels}")

        # 加载DINOv3模型
        self.model = self._load_dinov3_model()

        # 获取块数量
        if self.is_convnext:
            # ConvNeXt架构
            self.num_blocks = len(self.model.stages) if hasattr(self.model, 'stages') else 4
        else:
            # ViT架构
            self.num_blocks = len(self.model.blocks) if hasattr(self.model, 'blocks') else 12

        # 创建MultiConvAdapter (仅用于ViT架构)
        if not self.is_convnext and adapter_frequency > 0:
            num_adapters = self.num_blocks // adapter_frequency
            print(f"    - Creating {num_adapters} MultiConvAdapters (frequency={adapter_frequency})...")
            self.multi_conv_adapters = nn.ModuleList([
                MultiConvAdapter(self.num_channels, patch_size=self.patch_size)
                for _ in range(num_adapters)
            ])
            self.adapter_frequency = adapter_frequency
        else:
            self.multi_conv_adapters = None
            self.adapter_frequency = 0

        # 冻结块
        self._freeze_blocks()

        print(f"  [AdaptedDINOv3] Initialized:")
        print(f"    - Model: {model_name} ({self.num_channels}D)")
        print(f"    - Pretrain: {pretrain_data}")
        print(f"    - Architecture: {'ConvNeXt' if self.is_convnext else 'ViT'}")
        print(f"    - Total blocks: {self.num_blocks}")
        print(f"    - Trainable blocks: {self.num_trainable_blocks}")
        if self.multi_conv_adapters:
            print(f"    - Adapters: {len(self.multi_conv_adapters)}")

    def _load_dinov3_model(self):
        """
        加载DINOv3模型
        优先顺序:
        1. 本地.pth权重文件 + torch.hub模型结构
        2. torch.hub在线加载(带权重)
        """
        local_config = get_local_weights_config()
        model_key = (self.model_name, self.pretrain_data)

        # 检查是否支持该模型+预训练数据组合
        if model_key not in OFFICIAL_WEIGHT_FILES:
            # 如果SAT版本不可用，提示用户
            if self.pretrain_data == 'sat':
                available_sat = [k[0] for k in OFFICIAL_WEIGHT_FILES.keys() if k[1] == 'sat']
                raise ValueError(
                    f"Model {self.model_name} with SAT pretrain not available. "
                    f"SAT pretrain only available for: {available_sat}"
                )
            raise ValueError(f"Model {self.model_name} with {self.pretrain_data} pretrain not available")

        weight_filename = OFFICIAL_WEIGHT_FILES[model_key]
        weights_dir = local_config["weights_dir"]
        local_weight_path = weights_dir / weight_filename

        # 设置torch.hub缓存目录
        torch_hub_dir = local_config["torch_hub_dir"]
        if torch_hub_dir.exists():
            os.environ['TORCH_HOME'] = str(torch_hub_dir)
            torch.hub.set_dir(str(torch_hub_dir))

        # 方法1: 本地权重文件 + torch.hub模型结构
        if local_weight_path.exists():
            print(f"    - Found local weights: {local_weight_path}")
            try:
                # 加载模型结构(不带权重)
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=False  # 不加载预训练权重
                )

                # 加载本地权重
                state_dict = torch.load(local_weight_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=True)

                print(f"    - Successfully loaded with local weights")
                return model

            except Exception as e:
                print(f"    - Failed to load local weights: {e}")
                print(f"    - Trying online loading...")

        # 方法2: torch.hub在线加载
        try:
            print(f"    - Loading from torch.hub: facebookresearch/dinov3/{self.model_name}")

            # 注意: torch.hub.load默认会下载权重，但默认是LVD版本
            # 对于SAT版本，需要手动加载权重
            if self.pretrain_data == 'sat':
                # SAT版本需要手动加载权重
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=False
                )
                print(f"    - SAT pretrain requires manual weight loading")
                print(f"    - Please download weights from Meta official website and place at:")
                print(f"      {local_weight_path}")
                raise FileNotFoundError(f"SAT weights not found: {local_weight_path}")
            else:
                # LVD版本可以直接在线加载
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True
                )

            print(f"    - Successfully loaded via torch.hub")
            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load {self.model_name}. "
                f"Please download weights from Meta official website and place at: "
                f"{local_weight_path}\n"
                f"Error: {e}"
            )

    def _freeze_blocks(self):
        """冻结前面的块，保留后面num_trainable_blocks层可训练"""
        if self.is_convnext:
            self._freeze_convnext_blocks()
        else:
            self._freeze_vit_blocks()

    def _freeze_vit_blocks(self):
        """冻结ViT块"""
        frozen_blocks = self.num_blocks - self.num_trainable_blocks

        if hasattr(self.model, 'blocks'):
            for i, blk in enumerate(self.model.blocks):
                if i < frozen_blocks:
                    for param in blk.parameters():
                        param.requires_grad = False
                else:
                    for param in blk.parameters():
                        param.requires_grad = True

        # 冻结embedding层
        if hasattr(self.model, 'patch_embed'):
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'cls_token'):
            if isinstance(self.model.cls_token, nn.Parameter):
                self.model.cls_token.requires_grad = False

        if hasattr(self.model, 'pos_embed'):
            if isinstance(self.model.pos_embed, nn.Parameter):
                self.model.pos_embed.requires_grad = False

    def _freeze_convnext_blocks(self):
        """冻结ConvNeXt块"""
        # ConvNeXt有4个stages
        frozen_stages = max(0, 4 - self.num_trainable_blocks)

        if hasattr(self.model, 'stages'):
            for i, stage in enumerate(self.model.stages):
                if i < frozen_stages:
                    for param in stage.parameters():
                        param.requires_grad = False
                else:
                    for param in stage.parameters():
                        param.requires_grad = True

        # 冻结stem
        if hasattr(self.model, 'stem'):
            for param in self.model.stem.parameters():
                param.requires_grad = False

    def _forward_vit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ViT架构的前向传播 - 适配DINOv3"""
        B, _, H, W = x.shape

        # DINOv3使用forward_features方法，直接调用
        # 这样可以正确处理RoPE位置编码等
        if hasattr(self.model, 'forward_features'):
            # 使用DINOv3的forward_features
            result = self.model.forward_features(x, masks=None)

            # forward_features返回字典或字典列表
            if isinstance(result, list):
                result = result[0]

            # 提取CLS token和patch tokens
            t = result['x_norm_clstoken']  # (B, C) - 全局token
            f = result['x_norm_patchtokens']  # (B, N, C) - patch tokens

            # Reshape为2D特征图
            h = w = H // self.patch_size
            f = f.reshape((B, h, w, self.num_channels)).permute(0, 3, 1, 2)  # (B, C, h, w)

            return f, t

        # 如果没有forward_features，手动处理（兼容其他模型）
        # 准备tokens（包含CLS token）
        if hasattr(self.model, 'prepare_tokens_with_masks'):
            tokens, (feat_h, feat_w) = self.model.prepare_tokens_with_masks(x, masks=None)
        elif hasattr(self.model, 'prepare_tokens'):
            tokens = self.model.prepare_tokens(x)
            feat_h = feat_w = H // self.patch_size
        else:
            # 手动处理
            tokens = self.model.patch_embed(x)
            if tokens.dim() == 4:  # (B, H, W, C)
                tokens = tokens.flatten(1, 2)  # (B, N, C)
            if hasattr(self.model, 'cls_token'):
                cls_tokens = self.model.cls_token.expand(B, -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
            feat_h = feat_w = H // self.patch_size

        # 获取RoPE编码（如果存在）
        rope_sincos = None
        if hasattr(self.model, 'rope_embed') and self.model.rope_embed is not None:
            rope_sincos = self.model.rope_embed(H=feat_h, W=feat_w)

        # 获取storage tokens数量
        n_storage = getattr(self.model, 'n_storage_tokens', 0)

        # 保存中间输出用于adapter融合
        output = []

        # 前面的冻结块
        frozen_blocks = self.num_blocks - self.num_trainable_blocks

        if hasattr(self.model, 'blocks'):
            for blk in self.model.blocks[:frozen_blocks]:
                tokens = blk(tokens, rope_sincos)
                # 保存patch tokens（去除CLS token和storage tokens）
                output.append(tokens[:, n_storage + 1:])

            # 梯度截断
            if frozen_blocks > 0:
                tokens = tokens.detach()

            # 后面的可训练块
            for blk in self.model.blocks[frozen_blocks:]:
                tokens = blk(tokens, rope_sincos)
                output.append(tokens[:, n_storage + 1:])

        # MultiConvAdapter融合
        if self.multi_conv_adapters and len(output) > 0:
            y = None
            for i, adapter in enumerate(self.multi_conv_adapters):
                if i == 0:
                    # 第一个adapter: 融合第0层和第2层
                    idx1, idx2 = 0, min(2, len(output) - 1)
                    y = adapter(output[idx1] + output[idx2], H=H) + output[idx1]
                else:
                    # 后续adapter: 融合前一层adapter输出和当前层输出
                    idx = min(3 * (i + 1) - 1, len(output) - 1)
                    y = adapter(y + output[idx], H=H) + y
        else:
            # 没有adapter，直接使用最后一层输出
            y = tokens[:, n_storage + 1:]  # 去除CLS token和storage tokens

        # 全局token
        output_token = tokens[:, 0]  # (B, C)

        # 拼接CLS token和适配后的patch tokens
        tokens = torch.cat([output_token.unsqueeze(1), y], dim=1)  # (B, 1+N, C)

        # 归一化
        if self.norm_layer and hasattr(self.model, 'norm'):
            tokens = self.model.norm(tokens)

        # 分离token和特征
        t = tokens[:, 0]  # 全局token (B, C)
        f = tokens[:, 1:]  # patch tokens (B, N, C)

        # Reshape为2D特征图
        h = w = H // self.patch_size
        f = f.reshape((B, h, w, self.num_channels)).permute(0, 3, 1, 2)  # (B, C, h, w)

        return f, t

    def _forward_convnext(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ConvNeXt架构的前向传播"""
        B = x.shape[0]

        # ConvNeXt前向传播
        if hasattr(self.model, 'forward_features'):
            features = self.model.forward_features(x)
        else:
            # 手动处理
            x = self.model.stem(x)
            for stage in self.model.stages:
                x = stage(x)
            features = x

        # 处理输出格式
        if features.dim() == 3:
            # (B, N, C) -> (B, C, H, W)
            h = w = int(features.shape[1] ** 0.5)
            f = features.permute(0, 2, 1).reshape(B, -1, h, w)
        elif features.dim() == 4:
            # 已经是 (B, C, H, W)
            f = features
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        # 全局平均池化得到token
        t = f.mean(dim=[2, 3])  # (B, C)

        return f, t

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)，通常是448x448

        Returns:
            feature_map: 局部特征图 (B, C, 32, 32)
            global_token: 全局token (B, C) [如果return_token=True]
        """
        B, _, H, W = x.shape

        # 输入上采样: 448x448 → 512x512
        if H != self.input_size or W != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
            H = W = self.input_size

        # 根据架构选择前向传播方法
        if self.is_convnext:
            f, t = self._forward_convnext(x)
        else:
            f, t = self._forward_vit(x)

        # 确保输出尺寸为32x32
        if f.shape[-1] != 32 or f.shape[-2] != 32:
            f = F.interpolate(f, size=(32, 32), mode='bilinear', align_corners=False)

        if self.return_token:
            return f, t
        else:
            return f


def test_adapted_dinov3():
    """测试AdaptedDINOv3"""
    print("\n" + "=" * 80)
    print("Testing AdaptedDINOv3")
    print("=" * 80)

    # 测试ViT架构
    print("\n[Test 1] ViT-B/16 with LVD pretrain")
    try:
        model = AdaptedDINOv3(
            model_name='dinov3_vitb16',
            pretrain_data='lvd',
            num_trainable_blocks=0,
            adapter_frequency=3,
            input_size=512,
            norm_layer=True,
            return_token=True
        )
        model.eval()

        # 测试448输入（应自动上采样到512）
        x = torch.randn(2, 3, 448, 448)
        print(f"\nInput shape: {x.shape}")

        with torch.no_grad():
            f, t = model(x)

        print(f"Feature map shape: {f.shape}")  # 期望 (2, 768, 32, 32)
        print(f"Global token shape: {t.shape}")  # 期望 (2, 768)

        assert f.shape == (2, 768, 32, 32), f"Expected (2, 768, 32, 32), got {f.shape}"
        assert t.shape == (2, 768), f"Expected (2, 768), got {t.shape}"

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")

        print("\n[OK] ViT test passed!")

    except Exception as e:
        print(f"\n[SKIP] ViT test skipped: {e}")

    print("\n" + "=" * 80)
    print("AdaptedDINOv3 test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_adapted_dinov3()
