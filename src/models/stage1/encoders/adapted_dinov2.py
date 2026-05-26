#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaptedDINOv2编码器
从原始ImLPR移植并适配，包含MultiConvAdapter

Created: 2025-10-21 17:45
Author: Claude Code Assistant
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path
import os
import sys


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

# 本地权重配置
def get_local_weights_config():
    """获取本地权重配置"""
    project_dir = Path(__file__).parent.parent.parent  # src/models -> src -> clav/
    weights_dir = project_dir / "pretrained_weights"

    return {
        "dinov2_vitb14": {
            "weights_path": weights_dir / "torch_hub/dinov2/dinov2_vitb14.pth",
            "cache_dir": weights_dir / "torch_hub"
        },
        "dinov2_vits14": {
            "weights_path": weights_dir / "torch_hub/dinov2/dinov2_vits14.pth",
            "cache_dir": weights_dir / "torch_hub"
        },
        "dinov2_vitl14": {
            "weights_path": weights_dir / "torch_hub/dinov2/dinov2_vitl14.pth",
            "cache_dir": weights_dir / "torch_hub"
        },
        "dinov2_vitg14": {
            "weights_path": weights_dir / "torch_hub/dinov2/dinov2_vitg14.pth",
            "cache_dir": weights_dir / "torch_hub"
        }
    }


class MultiConvAdapter(nn.Module):
    """
    并行1x1/3x3/5x5卷积适配器
    来自SelaVPR++论文，用于增强多尺度特征表示

    输入: (B, N, C) token序列
    输出: (B, N, C) 增强后的token序列
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: 输入通道数（DINOv2特征维度）
        """
        super(MultiConvAdapter, self).__init__()
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
        # H // 14 = 特征图高度（DINOv2 patch_size=14）
        h = w = H // 14
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


class AdaptedDINOv2(nn.Module):
    """
    适配的DINOv2编码器

    特性:
    1. 从torch.hub加载预训练DINOv2
    2. MultiConvAdapter增强特征
    3. 可选的部分可训练块（后N层可微调）
    4. 中间特征融合

    输出:
    - feature_map: 局部特征图 (B, C, H/14, W/14)
    - global_token: 全局token (B, C)
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        num_trainable_blocks: int = 0,
        adapter_frequency: int = 3,
        norm_layer: bool = True,
        return_token: bool = True
    ):
        """
        Args:
            model_name: DINOv2模型名称（dinov2_vits14/vitb14/vitl14/vitg14）
            num_trainable_blocks: 可训练块数量（从后往前）
            adapter_frequency: adapter频率（每N层一个adapter）
            norm_layer: 是否应用归一化层
            return_token: 是否返回全局token
        """
        super().__init__()

        assert model_name in DINOV2_ARCHS, f'Unknown model name {model_name}'

        # 尝试加载本地权重，如果不存在则从torch.hub下载
        self.model = self._load_dinov2_model(model_name)

        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        # 创建MultiConvAdapter
        self.num_blocks = len(self.model.blocks)
        num_adapters = self.num_blocks // adapter_frequency

        print(f"  Creating {num_adapters} MultiConvAdapters...")
        self.multi_conv_adapters = nn.ModuleList([
            MultiConvAdapter(self.num_channels) for _ in range(num_adapters)
        ])

        # 冻结前面的块
        self._freeze_blocks()

        print(f"  AdaptedDINOv2 initialized:")
        print(f"    - Model: {model_name} ({self.num_channels}D)")
        print(f"    - Total blocks: {self.num_blocks}")
        print(f"    - Trainable blocks: {self.num_trainable_blocks}")
        print(f"    - Adapters: {len(self.multi_conv_adapters)}")

    def _load_dinov2_model(self, model_name: str):
        """
        加载DINOv2模型，优先使用本地缓存

        Args:
            model_name: 模型名称

        Returns:
            加载的模型
        """
        local_config = get_local_weights_config()

        # 设置torch hub目录为本地缓存
        cache_dir = local_config.get(model_name, {}).get("cache_dir", Path(__file__).parent.parent.parent / "pretrained_weights/torch_hub")

        if cache_dir.exists():
            os.environ['TORCH_HOME'] = str(cache_dir)
            torch.hub.set_dir(str(cache_dir))
            print(f"  Using local cache: {cache_dir}")

        try:
            # 尝试从本地缓存加载（source='local'表示不检查更新）
            print(f"  Loading {model_name} from torch.hub...")
            model = torch.hub.load('facebookresearch/dinov2', model_name, source='local')

            # DINOv2模型会自动处理不同输入尺寸的位置编码插值
            # prepare_tokens_with_masks方法会在运行时动态处理
            print(f"  ✓ Successfully loaded {model_name}")
            return model

        except Exception as e:
            print(f"  ⚠ Failed to load from local cache: {e}")

            # 如果本地缓存失败，尝试使用手动加载权重的方式
            if model_name in local_config:
                weights_path = local_config[model_name]["weights_path"]

                if weights_path.exists():
                    print(f"  Trying manual weight loading from: {weights_path}")

                    # 最后的备选方案：在线加载模型结构，本地加载权重
                    try:
                        print(f"  Loading model structure...")
                        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)

                        print(f"  Loading weights from: {weights_path}")
                        state_dict = torch.load(weights_path, map_location='cpu')

                        # DINOv2会自动处理位置编码插值，所以使用strict=False
                        model.load_state_dict(state_dict, strict=False)
                        print(f"  ✓ Successfully loaded with manual weight loading")
                        return model

                    except Exception as e2:
                        print(f"  ⚠ Manual loading also failed: {e2}")

            raise RuntimeError(f"Failed to load {model_name}. "
                              f"Please ensure the model is properly cached in {cache_dir}")

    def _freeze_blocks(self):
        """冻结前面的块，保留后面num_trainable_blocks层可训练"""
        frozen_blocks = self.num_blocks - self.num_trainable_blocks

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
            self.model.cls_token.requires_grad = False

        if hasattr(self.model, 'pos_embed'):
            self.model.pos_embed.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            feature_map: 局部特征图 (B, C, H/14, W/14)
            global_token: 全局token (B, C) [如果return_token=True]
        """
        B, _, H, W = x.shape

        # 准备tokens（包含CLS token）
        x = self.model.prepare_tokens_with_masks(x)  # (B, 1+N, C)

        # 保存中间输出用于adapter融合
        output = []

        # 前面的冻结块
        frozen_blocks = self.num_blocks - self.num_trainable_blocks
        for blk in self.model.blocks[:frozen_blocks]:
            x = blk(x)
            output.append(x[:, 1:])  # 保存patch tokens（去除CLS token）

        # 梯度截断
        x = x.detach()

        # 后面的可训练块
        for blk in self.model.blocks[frozen_blocks:]:
            x = blk(x)
            output.append(x[:, 1:])

        # MultiConvAdapter融合
        y = None
        for i, adapter in enumerate(self.multi_conv_adapters):
            if i == 0:
                # 第一个adapter: 融合第0层和第2层
                y = adapter(output[0] + output[2], H=H) + output[0]
            else:
                # 后续adapter: 融合前一层adapter输出和当前层输出
                idx = 3 * (i + 1) - 1
                if idx < len(output):
                    y = adapter(y + output[idx], H=H) + y

        # 全局token
        output_token = x[:, 0]  # (B, C)

        # 拼接CLS token和适配后的patch tokens
        x = torch.cat([output_token.unsqueeze(1), y], dim=1)  # (B, 1+N, C)

        # 归一化
        if self.norm_layer:
            x = self.model.norm(x)

        # 分离token和特征
        t = x[:, 0]  # 全局token (B, C)
        f = x[:, 1:]  # patch tokens (B, N, C)

        # Reshape为2D特征图
        h = w = H // 14
        f = f.reshape((B, h, w, self.num_channels)).permute(0, 3, 1, 2)  # (B, C, h, w)

        if self.return_token:
            return f, t
        else:
            return f


def test_adapted_dinov2():
    """测试AdaptedDINOv2"""
    print("\n" + "="*80)
    print("Testing AdaptedDINOv2")
    print("="*80)

    # 创建模型
    model = AdaptedDINOv2(
        model_name='dinov2_vitb14',
        num_trainable_blocks=0,
        adapter_frequency=3,
        norm_layer=True,
        return_token=True
    )
    model.eval()

    # 测试输入
    x = torch.randn(2, 3, 448, 448)

    print(f"\nInput shape: {x.shape}")

    # 前向传播
    with torch.no_grad():
        f, t = model(x)

    print(f"Feature map shape: {f.shape}")  # (2, 768, 32, 32)
    print(f"Global token shape: {t.shape}")  # (2, 768)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    print("\n✓ AdaptedDINOv2 test passed!")


if __name__ == "__main__":
    test_adapted_dinov2()
