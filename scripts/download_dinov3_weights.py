#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3预训练权重下载脚本

支持两种下载方式:
1. torch.hub下载 (推荐，仅LVD版本)
2. HuggingFace下载 (需要访问权限)

支持的模型:
- ViT架构: dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16
- ConvNeXt架构: dinov3_convnext_tiny/small/base/large
- 预训练数据: LVD (web图像) 或 SAT (卫星图像，仅vitl16/vit7b16)

使用方法:
    # 使用torch.hub下载 (推荐)
    python download_dinov3_weights.py --model dinov3_vitb16 --method hub

    # 下载并保存权重到本地
    python download_dinov3_weights.py --model dinov3_vitb16 --method hub --save

    # 列出所有可用模型
    python download_dinov3_weights.py --list

Created: 2025-12-03 16:00
Updated: 2025-12-03 21:45
Author: Claude Code Assistant
"""

import argparse
import os
from pathlib import Path
import sys


# 官方权重文件名 (用于本地加载)
OFFICIAL_WEIGHT_FILES = {
    # ViT + LVD
    ('dinov3_vits16', 'lvd'): 'dinov3_vits16_pretrain_lvd1689m-d0e0eb6d.pth',
    ('dinov3_vits16plus', 'lvd'): 'dinov3_vits16plus_pretrain_lvd1689m-fb04d891.pth',
    ('dinov3_vitb16', 'lvd'): 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    ('dinov3_vitl16', 'lvd'): 'dinov3_vitl16_pretrain_lvd1689m-33e5a6fe.pth',
    ('dinov3_vith16plus', 'lvd'): 'dinov3_vith16plus_pretrain_lvd1689m-5d696ada.pth',
    ('dinov3_vit7b16', 'lvd'): 'dinov3_vit7b16_pretrain_lvd1689m-c3a099cc.pth',
    # ViT + SAT (只有L和7B)
    ('dinov3_vitl16', 'sat'): 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
    ('dinov3_vit7b16', 'sat'): 'dinov3_vit7b16_pretrain_sat493m-28d52f52.pth',
}

# HuggingFace模型ID映射 (受限访问)
HF_MODELS = {
    ('dinov3_vits16', 'lvd'): 'facebook/dinov3-vits16-pretrain-lvd1689m',
    ('dinov3_vitb16', 'lvd'): 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    ('dinov3_vitl16', 'lvd'): 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    ('dinov3_vit7b16', 'lvd'): 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    ('dinov3_vitl16', 'sat'): 'facebook/dinov3-vitl16-pretrain-sat493m',
    ('dinov3_vit7b16', 'sat'): 'facebook/dinov3-vit7b16-pretrain-sat493m',
}

# torch.hub支持的模型 (仅LVD版本)
HUB_MODELS = [
    'dinov3_vits16',
    'dinov3_vits16plus',
    'dinov3_vitb16',
    'dinov3_vitl16',
    'dinov3_vith16plus',
    'dinov3_vit7b16',
    'dinov3_convnext_tiny',
    'dinov3_convnext_small',
    'dinov3_convnext_base',
    'dinov3_convnext_large',
]

# 模型简称列表
MODEL_NAMES = [
    'dinov3_vits16',
    'dinov3_vits16plus',
    'dinov3_vitb16',
    'dinov3_vitl16',
    'dinov3_vith16plus',
    'dinov3_vit7b16',
    'dinov3_convnext_tiny',
    'dinov3_convnext_small',
    'dinov3_convnext_base',
    'dinov3_convnext_large',
]

PRETRAIN_TYPES = ['lvd', 'sat']


def get_output_dir():
    """获取输出目录"""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    return project_dir / "pretrained_weights" / "dinov3"


def download_via_hub(model_name: str, save_path: Path = None):
    """
    使用torch.hub下载模型

    Args:
        model_name: 模型名称
        save_path: 可选，保存权重的路径

    Returns:
        成功返回True，失败返回False
    """
    if model_name not in HUB_MODELS:
        print(f"[ERROR] Model {model_name} not supported via torch.hub")
        return False

    print(f"[DOWNLOAD] Loading {model_name} via torch.hub...")
    print("  Note: torch.hub only supports LVD pretrain data")

    try:
        import torch

        # 加载模型
        model = torch.hub.load(
            'facebookresearch/dinov3',
            model_name,
            pretrained=True
        )

        print(f"[OK] Model loaded successfully!")

        # 保存权重到本地
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[OK] Weights saved to: {save_path}")

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to load via torch.hub: {e}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Try again later (GitHub may be rate-limiting)")
        print("  3. Download manually from Meta's official website")
        return False


def download_via_hf(model_name: str, pretrain_data: str, output_dir: Path):
    """
    使用HuggingFace下载模型 (需要访问权限)

    Args:
        model_name: 模型名称
        pretrain_data: 预训练数据类型 (lvd/sat)
        output_dir: 输出目录
    """
    model_key = (model_name, pretrain_data)

    if model_key not in HF_MODELS:
        print(f"[ERROR] Model {model_name} with {pretrain_data} pretrain not available on HuggingFace.")
        return False

    hf_model_id = HF_MODELS[model_key]
    local_path = output_dir / hf_model_id.replace('/', '_')

    if local_path.exists():
        print(f"[SKIP] Model already exists: {local_path}")
        return True

    print(f"[DOWNLOAD] {hf_model_id}")
    print(f"  -> {local_path}")
    print("\n  NOTE: HuggingFace repos are gated. You may need to:")
    print("  1. Login with: huggingface-cli login")
    print("  2. Request access at: https://huggingface.co/{hf_model_id}")

    try:
        from huggingface_hub import snapshot_download

        local_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=hf_model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )

        print(f"[OK] Downloaded successfully!")
        return True

    except ImportError:
        print("[ERROR] huggingface_hub library not installed.")
        print("  Install with: pip install huggingface_hub")
        return False

    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        print("\n  The model repository is likely gated. Please:")
        print("  1. Download manually from Meta's official website")
        print("  2. Place the .pth file in: pretrained_weights/dinov3/")
        return False


def list_models():
    """列出所有可用模型"""
    print("\n" + "=" * 70)
    print("Available DINOv3 Models")
    print("=" * 70)

    print("\n[torch.hub] Supported models (LVD only, recommended):")
    print("-" * 50)
    for name in HUB_MODELS:
        print(f"  {name}")

    print("\n[Local .pth] Official weight files:")
    print("-" * 50)
    for (name, pretrain), filename in OFFICIAL_WEIGHT_FILES.items():
        print(f"  {name} + {pretrain}: {filename}")

    print("\n[HuggingFace] Gated repositories (requires access):")
    print("-" * 50)
    for (name, pretrain), repo in HF_MODELS.items():
        print(f"  {name} + {pretrain}: {repo}")

    print("\n" + "=" * 70)
    print("RECOMMENDED: Use torch.hub for LVD models:")
    print("  python download_dinov3_weights.py --model dinov3_vitb16 --method hub --save")
    print("\nFor SAT models, download manually from Meta's website and place in:")
    print("  pretrained_weights/dinov3/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download DINOv3 pretrained weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download via torch.hub (recommended for LVD models)
  python download_dinov3_weights.py --model dinov3_vitb16 --method hub

  # Download and save weights locally
  python download_dinov3_weights.py --model dinov3_vitb16 --method hub --save

  # Try HuggingFace (requires access)
  python download_dinov3_weights.py --model dinov3_vitb16 --pretrain sat --method hf

  # List available models
  python download_dinov3_weights.py --list
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='dinov3_vitb16',
        choices=MODEL_NAMES,
        help='Model to download (default: dinov3_vitb16)'
    )

    parser.add_argument(
        '--pretrain', '-p',
        type=str,
        default='lvd',
        choices=PRETRAIN_TYPES,
        help='Pretrain data type (default: lvd). Note: SAT only for vitl16/vit7b16'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='hub',
        choices=['hub', 'hf'],
        help='Download method: hub (torch.hub, LVD only) or hf (HuggingFace, requires access)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save weights to local file when using torch.hub'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: pretrained_weights/dinov3)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available models'
    )

    args = parser.parse_args()

    # 列出模型
    if args.list:
        list_models()
        return

    # 设置输出目录
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)

    if args.method == 'hub':
        # 使用torch.hub下载
        if args.pretrain == 'sat':
            print("[WARNING] torch.hub only supports LVD pretrain data.")
            print("  For SAT models, download manually from Meta's website.")
            return

        save_path = None
        if args.save:
            # 使用官方文件名
            key = (args.model, 'lvd')
            if key in OFFICIAL_WEIGHT_FILES:
                save_path = output_dir / OFFICIAL_WEIGHT_FILES[key]
            else:
                save_path = output_dir / f"{args.model}_pretrain_lvd.pth"

        download_via_hub(args.model, save_path)

    else:
        # 使用HuggingFace下载
        download_via_hf(args.model, args.pretrain, output_dir)

    print("=" * 60)


if __name__ == '__main__':
    main()
