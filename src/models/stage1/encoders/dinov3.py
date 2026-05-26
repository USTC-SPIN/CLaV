#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3 HuggingFace Transformers接口
使用注册器模式，类似dinov2.py

Created: 2025-12-03 15:45
Author: Claude Code Assistant
"""

from torch import nn
import torch
from pathlib import Path
from . import register_encoder


def get_local_huggingface_path(model_id: str) -> str:
    """获取本地HuggingFace模型路径"""
    # src/models/stage1/encoders -> src/models/stage1 -> src/models -> src -> clav/
    project_dir = Path(__file__).parent.parent.parent.parent.parent
    weights_dir = project_dir / "pretrained_weights" / "huggingface"

    # 转换模型ID为本地路径格式
    local_path = weights_dir / model_id.replace('/', '_')

    if local_path.exists():
        return str(local_path)
    return model_id  # 返回原始ID，使用在线下载


@register_encoder()
class Dinov3withNorm(nn.Module):
    """
    DINOv3编码器 (HuggingFace Transformers版本)

    支持的模型ID格式:
    - facebook/dinov3-vits16-pretrain-lvd
    - facebook/dinov3-vitb16-pretrain-lvd
    - facebook/dinov3-vitl16-pretrain-lvd
    - facebook/dinov3-vits16-pretrain-sat
    - facebook/dinov3-vitb16-pretrain-sat
    - facebook/dinov3-vitl16-pretrain-sat
    - facebook/dinov3-convnext-tiny-pretrain-lvd
    - facebook/dinov3-convnext-base-pretrain-lvd
    - 等等...
    """

    def __init__(
        self,
        dinov3_path: str = 'facebook/dinov3-vitb16-pretrain-lvd',
        normalize: bool = True,
    ):
        """
        Args:
            dinov3_path: HuggingFace模型ID或本地路径
            normalize: 是否归一化输出
        """
        super().__init__()

        from transformers import AutoModel

        # 检查是否有本地版本
        local_path = get_local_huggingface_path(dinov3_path)

        # Support both local paths and HuggingFace model IDs
        if local_path != dinov3_path and Path(local_path).exists():
            print(f"  Loading DINOv3 from local path: {local_path}")
            try:
                self.encoder = AutoModel.from_pretrained(local_path, local_files_only=True)
                print(f"  Successfully loaded local weights")
            except Exception as e:
                print(f"  Failed to load local weights: {e}")
                print(f"  Falling back to online download...")
                self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=False)
        else:
            # 尝试先使用本地文件，如果失败则在线下载
            try:
                self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                print(f"  Loading DINOv3 from HuggingFace (online): {dinov3_path}")
                self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=False)

        self.encoder.requires_grad_(False)

        # 处理归一化层
        if normalize and hasattr(self.encoder, 'layernorm'):
            self.encoder.layernorm.elementwise_affine = False
            if hasattr(self.encoder.layernorm, 'weight'):
                self.encoder.layernorm.weight = None
            if hasattr(self.encoder.layernorm, 'bias'):
                self.encoder.layernorm.bias = None

        # 获取配置信息
        if hasattr(self.encoder, 'config'):
            self.patch_size = getattr(self.encoder.config, 'patch_size', 16)
            self.hidden_size = getattr(self.encoder.config, 'hidden_size', 768)
        else:
            self.patch_size = 16
            self.hidden_size = 768

        print(f"  DINOv3 initialized: patch_size={self.patch_size}, hidden_size={self.hidden_size}")

    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，提取图像特征

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            image_features: 图像特征 (B, N, C)
        """
        outputs = self.encoder(x, output_hidden_states=True)

        # 获取最后一层hidden state
        last_hidden_state = outputs.last_hidden_state

        # 去除特殊tokens (CLS + register tokens)
        # DINOv3可能有不同数量的特殊token
        if hasattr(self.encoder.config, 'num_register_tokens'):
            num_special = 1 + self.encoder.config.num_register_tokens  # CLS + registers
        else:
            num_special = 1  # 只有CLS token

        image_features = last_hidden_state[:, num_special:]

        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            image_features: 图像特征 (B, N, C)
        """
        return self.dinov3_forward(x)


@register_encoder()
class Dinov3ViT(nn.Module):
    """
    DINOv3 ViT编码器，返回2D特征图
    """

    def __init__(
        self,
        dinov3_path: str = 'facebook/dinov3-vitb16-pretrain-lvd',
        normalize: bool = True,
        return_token: bool = True,
    ):
        """
        Args:
            dinov3_path: HuggingFace模型ID或本地路径
            normalize: 是否归一化输出
            return_token: 是否返回全局token
        """
        super().__init__()

        from transformers import AutoModel

        # 检查是否有本地版本
        local_path = get_local_huggingface_path(dinov3_path)

        if local_path != dinov3_path and Path(local_path).exists():
            print(f"  Loading DINOv3 from local path: {local_path}")
            self.encoder = AutoModel.from_pretrained(local_path, local_files_only=True)
        else:
            try:
                self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                print(f"  Loading DINOv3 from HuggingFace (online): {dinov3_path}")
                self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=False)

        self.encoder.requires_grad_(False)
        self.return_token = return_token

        # 获取配置
        if hasattr(self.encoder, 'config'):
            self.patch_size = getattr(self.encoder.config, 'patch_size', 16)
            self.hidden_size = getattr(self.encoder.config, 'hidden_size', 768)
        else:
            self.patch_size = 16
            self.hidden_size = 768

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            feature_map: 特征图 (B, C, H//patch_size, W//patch_size)
            global_token: 全局token (B, C) [如果return_token=True]
        """
        B, _, H, W = x.shape

        outputs = self.encoder(x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state

        # 提取CLS token
        cls_token = last_hidden_state[:, 0]  # (B, C)

        # 获取特殊token数量
        if hasattr(self.encoder.config, 'num_register_tokens'):
            num_special = 1 + self.encoder.config.num_register_tokens
        else:
            num_special = 1

        # 提取patch tokens
        patch_tokens = last_hidden_state[:, num_special:]  # (B, N, C)

        # Reshape为2D特征图
        h = H // self.patch_size
        w = W // self.patch_size
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.hidden_size, h, w)

        if self.return_token:
            return feature_map, cls_token
        else:
            return feature_map
