"""
BEV (Bird's Eye View) Dataset for ImLPR training.
Loads PNG BEV images instead of RIV .npy files.
"""

import os
from typing import List, Dict, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import cv2


# =========================
# Augmentation utilities for BEV
# =========================

def random_rotation_bev(image_hwc: np.ndarray, max_angle_deg: float = 180.0) -> Tuple[np.ndarray, float]:
    """
    Randomly rotate BEV image around center.
    Returns (rotated_image, rotation_angle_rad).

    Args:
        image_hwc: (H, W, C) BEV image
        max_angle_deg: maximum rotation angle in degrees
    """
    H, W, C = image_hwc.shape
    angle_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = np.radians(angle_deg)

    # Rotation matrix
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    # Rotate image
    rotated = cv2.warpAffine(image_hwc, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)

    return rotated, angle_rad


def random_flip_bev(image_hwc: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, bool]:
    """
    Randomly flip BEV image horizontally or vertically.
    Returns (flipped_image, was_flipped).
    """
    if np.random.random() < prob:
        # Randomly choose horizontal or vertical flip
        if np.random.random() < 0.5:
            return np.flip(image_hwc, axis=1).copy(), True  # horizontal
        else:
            return np.flip(image_hwc, axis=0).copy(), True  # vertical
    return image_hwc, False


def normalize_bev(image_chw: np.ndarray) -> np.ndarray:
    """
    Normalize BEV image from uint8 [0, 255] to float32 [0, 1].
    Expects CHW format.
    """
    image_chw = image_chw.astype(np.float32, copy=False)
    return image_chw / 255.0


def random_block_mask_bev(height: int,
                          width: int,
                          mask_ratio: float = 0.0,
                          patch_size_min: int = 8,
                          patch_size_max: int = 32) -> np.ndarray:
    """
    Randomly zero out multiple square/rect blocks in BEV.
    """
    mask = np.ones((height, width), dtype=np.float32)
    if mask_ratio <= 0:
        return mask

    patch_size = np.random.randint(patch_size_min, patch_size_max + 1)
    total_area = height * width
    patch_area = patch_size * patch_size
    total_patches_possible = max(1, total_area // patch_area)
    num_masked = int(total_patches_possible * mask_ratio)

    max_h = height - patch_size
    max_w = width - patch_size
    if max_h < 0 or max_w < 0 or num_masked <= 0:
        return mask

    h_starts = np.random.randint(0, max_h + 1, size=num_masked)
    w_starts = np.random.randint(0, max_w + 1, size=num_masked)

    for hs, ws in zip(h_starts, w_starts):
        mask[hs:hs + patch_size, ws:ws + patch_size] = 0.0
    return mask


# =========================
# Import TrainingTuple from base_datasets to ensure compatibility
# =========================

from src.datasets.base_datasets import TrainingTuple, EvaluationTuple


# 自定义Unpickler来处理模块重命名
class RenameUnpickler(pickle.Unpickler):
    """处理pickle文件中旧的模块名,自动转换为新的模块名"""
    def find_class(self, module, name):
        # 处理TrainingTuple从数据生成脚本(__main__)的情况
        if name == 'TrainingTuple' and module == '__main__':
            # 返回我们定义的TrainingTuple类
            return TrainingTuple
        # 重命名旧的模块路径
        if module == 'datasets.base_datasets':
            module = 'imlpr_datasets.base_datasets'
        elif module.startswith('datasets.'):
            module = module.replace('datasets.', 'imlpr_datasets.')
        elif module == 'misc.utils':
            module = 'imlpr_misc.utils'
        elif module.startswith('misc.'):
            module = module.replace('misc.', 'imlpr_misc.')
        return super().find_class(module, name)


class BEVTrainingDataset(Dataset):
    """
    Training dataset for BEV images (PNG format).
    Similar to TrainingDataset but loads PNG instead of .npy.
    """
    def __init__(self, dataset_path, query_filename, image_size=512, transform=None, set_transform=None):
        assert os.path.exists(dataset_path), f'Cannot access dataset path: {dataset_path}'
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(self.dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), f'Cannot access query file: {self.query_filepath}'

        self.image_size = image_size

        with open(self.query_filepath, 'rb') as f:
            data_dict = RenameUnpickler(f).load()

        self.queries: Dict[int, TrainingTuple] = {}
        for key in data_dict.keys():
            item = data_dict[key]
            # Pickle file contains TrainingTuple objects directly
            self.queries[key] = item

        print(f'{len(self.queries)} queries in the BEV dataset')
        self.transform = transform
        self.set_transform = set_transform

    @staticmethod
    def apply_rotation_to_position_2d(position, rotation_angle_rad: float):
        """
        Apply 2D rotation to BEV position [x, y].

        Args:
            position: [x, y] 2D position array
            rotation_angle_rad: rotation angle in radians (around Z-axis)

        Returns:
            [x_new, y_new] rotated 2D position
        """
        cos_a = np.cos(rotation_angle_rad)
        sin_a = np.sin(rotation_angle_rad)
        rot_matrix_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        position_new = rot_matrix_2d @ position
        return position_new

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        # Load BEV PNG image (H, W, 3) in BGR format
        file_pathname = os.path.join(self.dataset_path, self.queries[index].rel_scan_filepath)

        # Handle both .png and .npy extensions
        if not os.path.exists(file_pathname):
            # Try replacing extension
            base_path = os.path.splitext(file_pathname)[0]
            if os.path.exists(base_path + '.png'):
                file_pathname = base_path + '.png'
            elif os.path.exists(base_path + '.npy'):
                file_pathname = base_path + '.npy'

        if file_pathname.endswith('.png'):
            query = cv2.imread(file_pathname, cv2.IMREAD_COLOR)  # (H, W, 3) BGR
            if query is None:
                raise ValueError(f"Failed to load image: {file_pathname}")
            # Convert BGR to RGB
            query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to .npy (for compatibility)
            query = np.load(file_pathname)  # (H, W, 3)
            query = query.astype(np.uint8)

        # Resize to ensure dimensions are divisible by 14 (DINOv2 patch size)
        # If self.image_size is not divisible by 14, adjust to nearest valid size
        target_size = self.image_size
        if target_size % 14 != 0:
            # Round to nearest multiple of 14
            target_size = int(round(target_size / 14) * 14)

        if query.shape[0] != target_size or query.shape[1] != target_size:
            query = cv2.resize(query, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # 1) Random rotation augmentation
        query, rotation_angle = random_rotation_bev(query, max_angle_deg=180.0)

        # 2) Random flip augmentation
        query, was_flipped = random_flip_bev(query, prob=0.5)

        # 3) HWC -> CHW
        query = query.transpose(2, 0, 1)  # (3, H, W)

        # 4) Normalize to [0, 1]
        query = normalize_bev(query)

        # 5) Random block masking (similar to RIV)
        block_ratio = float(np.random.uniform(0.0, 0.3))
        if block_ratio > 0:
            H, W = query.shape[1], query.shape[2]
            mask = random_block_mask_bev(H, W, mask_ratio=block_ratio)
            query = query * mask  # broadcast over channels

        # Keep original position for geo-localization (rotation only affects image)
        # Position is used for evaluation to compute GPS-based ground truth
        query_position = self.queries[index].position

        return query, index, query_position

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    """Evaluation set (same as base_datasets)."""
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        query_l = [e.to_tuple() for e in self.query_set]
        map_l = [e.to_tuple() for e in self.map_set]
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        with open(pickle_filepath, 'rb') as f:
            query_l, map_l = RenameUnpickler(f).load()
        self.query_set = [EvaluationTuple(*e) for e in query_l]
        self.map_set = [EvaluationTuple(*e) for e in map_l]

    def get_map_positions(self):
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions
