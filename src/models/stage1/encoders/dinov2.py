from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from pathlib import Path
import os
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
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()

        # 检查是否有本地版本
        local_path = get_local_huggingface_path(dinov2_path)

        # Support both local paths and HuggingFace model IDs
        if local_path != dinov2_path and Path(local_path).exists():
            print(f"  Loading DINOv2 from local path: {local_path}")
            try:
                self.encoder = Dinov2WithRegistersModel.from_pretrained(local_path, local_files_only=True)
                print(f"  ✓ Successfully loaded local weights")
            except Exception as e:
                print(f"  ⚠ Failed to load local weights: {e}")
                print(f"  Falling back to online download...")
                self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        else:
            # 尝试先使用本地文件，如果失败则在线下载
            try:
                self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                print(f"  Loading DINOv2 from HuggingFace (online): {dinov2_path}")
                self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)
