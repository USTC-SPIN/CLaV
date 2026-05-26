#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEV Denoising Dataset - 用于加载配对的(noisy, clean) BEV图像

Created: 2025-10-22 09:30
Updated: 2025-10-31 23:20 - Added augmentation support
Author: Claude Code Assistant
"""

import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

# Import augmentation utilities
try:
    from .augmentations import create_transform
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("Warning: augmentations.py not found, using basic transforms")


class DenoisingBEVDataset(Dataset):
    """
    加载配对的(noisy, clean) BEV图像用于联合训练扩散模型和描述符头

    数据格式(来自generate_stage2_denoising_tuples_202510211030.py):
    {
        'train': [(noisy_path, clean_path, weather), ...],
        'val': [(noisy_path, clean_path, weather), ...],
        'root_dir': dataset_folder,
        'metadata': {
            'train_pairs_full': [
                {
                    'noisy_path': str,
                    'clean_path': str,
                    'weather': str,
                    'timestamp': int,
                    'sequence': str,
                    'position': [x, y, z]  # 从OXTS数据提取
                },
                ...
            ],
            'val_pairs_full': [...]
        }
    }
    """

    def __init__(
        self,
        pickle_path,
        split='train',
        image_size=448,
        transform=None,
        augmentation_config=None
    ):
        """
        Args:
            pickle_path: 包含配对数据的pickle文件路径
            split: 'train' 或 'val'
            image_size: 图像大小
            transform: 可选的transforms (如果提供，会覆盖augmentation_config)
            augmentation_config: 增强配置字典 (仅在训练时使用)
        """
        self.split = split
        self.image_size = image_size
        self.is_training = (split == 'train')

        # 加载pickle数据
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.root_dir = Path(data['root_dir'])
        self.pairs = data[split]  # [(noisy_path, clean_path, weather), ...]

        # 获取完整的pair信息(包含位置)
        self.pairs_full = data['metadata'].get(f'{split}_pairs_full', [])

        # 构建位置索引
        self.positions = self._load_positions()

        # Setup transforms
        if transform is not None:
            # User-provided transform
            self.transform = transform
            self.use_custom_transform = True
        elif AUGMENTATION_AVAILABLE and augmentation_config is not None and self.is_training:
            # Use enhanced augmentation pipeline (only for training)
            self.transform = create_transform(
                image_size=image_size,
                augmentation_config=augmentation_config,
                is_training=True
            )
            self.use_custom_transform = False
            print(f"[DenoisingBEVDataset] Using enhanced augmentation pipeline")
        else:
            # Default basic transforms
            self.transform = self._create_basic_transform()
            self.use_custom_transform = True

        print(f"[DenoisingBEVDataset] Loaded {len(self.pairs)} {split} pairs")
        print(f"  Root dir: {self.root_dir}")
        print(f"  Positions shape: {self.positions.shape}")
        print(f"  Sample position: {self.positions[0] if len(self.positions) > 0 else 'None'}")

    def _create_basic_transform(self):
        """Create basic transform pipeline (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_positions(self):
        """从metadata中提取位置信息"""
        if not self.pairs_full:
            print("  Warning: No full pair info, using dummy positions")
            return np.zeros((len(self.pairs), 2), dtype=np.float32)

        positions = []
        for pair_info in self.pairs_full:
            if 'position' in pair_info:
                # Position格式: [northing, easting] (2D)
                pos = pair_info['position']
                positions.append(pos[:2] if len(pos) >= 2 else [0.0, 0.0])
            else:
                positions.append([0.0, 0.0])

        return np.array(positions, dtype=np.float32)

    def get_position(self, idx):
        """获取指定索引的位置信息(用于构建positives/negatives masks)"""
        return self.positions[idx]

    def get_all_positions(self):
        """获取所有位置信息"""
        return self.positions

    def get_sequence_ids(self):
        """获取所有样本的序列ID"""
        if not self.pairs_full:
            # 如果没有序列信息，返回None
            return None

        sequences = []
        for pair_info in self.pairs_full:
            seq = pair_info.get('sequence', None)
            sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            noisy_img: 带噪声的BEV图像 (C, H, W)
            clean_img: 干净的BEV图像 (C, H, W)
            idx: 索引(用于构建masks)
            weather: 天气条件字符串
        """
        noisy_path, clean_path, weather = self.pairs[idx]

        # 加载图像 - 修复路径重复问题
        # 如果相对路径已经包含了root_dir的basename，去除它
        root_basename = self.root_dir.name  # 'KITTI_org_bev_448'
        if noisy_path.startswith(root_basename + '/'):
            noisy_path = noisy_path[len(root_basename) + 1:]
        if clean_path.startswith(root_basename + '/'):
            clean_path = clean_path[len(root_basename) + 1:]

        noisy_img_path = self.root_dir / noisy_path
        clean_img_path = self.root_dir / clean_path

        try:
            noisy_img = Image.open(noisy_img_path).convert('RGB')
            clean_img = Image.open(clean_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {noisy_img_path}, {clean_img_path}")
            print(f"Error: {e}")
            # 返回零图像作为占位符
            noisy_img = Image.new('RGB', (self.image_size, self.image_size))
            clean_img = Image.new('RGB', (self.image_size, self.image_size))

        # Apply transforms
        if self.use_custom_transform:
            # Custom transform (legacy support)
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        else:
            # Enhanced transform (handles both images together)
            noisy_img, clean_img = self.transform(noisy_img, clean_img)

        return noisy_img, clean_img, idx, weather


class DenoisingBEVDatasetWithPositions(DenoisingBEVDataset):
    """
    扩展版本 - 从单独的位置pickle文件加载位置信息

    如果denoising tuple pickle中没有位置信息,
    可以从训练用的pickle文件(如kitti_bev_train.pickle)中加载
    """

    def __init__(
        self,
        pickle_path,
        position_pickle_path,
        split='train',
        image_size=448,
        transform=None,
        augmentation_config=None
    ):
        """
        Args:
            pickle_path: 配对数据pickle路径
            position_pickle_path: 包含位置信息的pickle路径
            split: 'train' 或 'val'
            image_size: 图像大小
            transform: 可选transforms
            augmentation_config: 增强配置字典 (仅在训练时使用)
        """
        # 先加载基础数据
        super().__init__(pickle_path, split, image_size, transform, augmentation_config)

        # 加载位置信息
        print(f"[DenoisingBEVDataset] Loading positions from: {position_pickle_path}")
        with open(position_pickle_path, 'rb') as f:
            position_data = pickle.load(f)

        # 构建文件名到位置的映射
        self.position_map = {}
        for idx, entry in position_data.items():
            # entry是TrainingTuple对象，使用属性访问
            file_path = entry.rel_scan_filepath  # 相对路径
            position = entry.position
            self.position_map[file_path] = position

        # 重新加载位置信息
        self.positions = self._load_positions_from_map()
        print(f"  Positions loaded: {self.positions.shape}")

    def _load_positions_from_map(self):
        """从position_map中提取位置"""
        positions = []

        for noisy_path, clean_path, weather in self.pairs:
            # 尝试使用clean path(orin)获取位置
            # 因为orin是参考序列,更可能有准确的位置信息

            # 从路径中提取文件名部分
            # noisy_path格式: '02-10-03-14-fog/bev_448/0000.png'
            # clean_path格式: '02-10-03-14-orin/bev_448/0000.png'

            if clean_path in self.position_map:
                positions.append(self.position_map[clean_path])
            elif noisy_path in self.position_map:
                positions.append(self.position_map[noisy_path])
            else:
                # 尝试构建可能的路径
                # 提取文件名
                parts = clean_path.split('/')
                if len(parts) >= 2:
                    filename = parts[-1]
                    sequence = parts[0].replace('-orin', '').replace('-fog', '').replace('-rain', '').replace('-snow', '')
                    # 尝试匹配
                    found = False
                    for key in self.position_map.keys():
                        if filename in key and sequence in key:
                            positions.append(self.position_map[key])
                            found = True
                            break
                    if not found:
                        positions.append([0.0, 0.0])
                else:
                    positions.append([0.0, 0.0])

        return np.array(positions, dtype=np.float32)


if __name__ == "__main__":
    # 测试代码
    print("Testing DenoisingBEVDataset...")

    # 测试基础版本
    pickle_path = "data/bev_denoising_tuples_448.pkl"

    try:
        dataset = DenoisingBEVDataset(pickle_path, split='train')
        print(f"\nDataset size: {len(dataset)}")

        # 测试加载第一个样本
        noisy, clean, idx, weather = dataset[0]
        print(f"\nSample 0:")
        print(f"  Noisy shape: {noisy.shape}")
        print(f"  Clean shape: {clean.shape}")
        print(f"  Index: {idx}")
        print(f"  Weather: {weather}")
        print(f"  Position: {dataset.get_position(idx)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the pickle file exists and has the correct format")

    print("\nDone!")
