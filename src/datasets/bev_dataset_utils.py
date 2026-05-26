"""
BEV Dataset utilities for ImLPR training.
Simplified dataloader without ICP/point cloud processing.
"""

import numpy as np
from typing import Dict
import torch
from torch.utils.data import DataLoader

from src.datasets.bev_datasets import BEVTrainingDataset
from src.datasets.samplers import BatchSampler
from src.utils.utils import TrainingParams


def make_bev_datasets(params: TrainingParams, validation: bool = False) -> Dict[str, BEVTrainingDataset]:
    """Create BEV training/validation datasets."""
    datasets = {}
    datasets['train'] = BEVTrainingDataset(
        params.dataset_folder,
        params.train_file,
        image_size=params.image_H,  # BEV images are square (512x512)
        transform=None,
        set_transform=None
    )
    if validation and params.val_file:
        datasets['val'] = BEVTrainingDataset(
            params.dataset_folder,
            params.val_file,
            image_size=params.image_H
        )
    return datasets


def make_bev_dataloaders(params: TrainingParams, validation: bool = False) -> Dict[str, DataLoader]:
    """
    Create BEV dataloaders with simplified collate function (no ICP).
    Returns a dict with 'train' and optional 'val'.
    """
    datasets = make_bev_datasets(params, validation=validation)
    dataloaders = {}

    train_sampler = BatchSampler(
        datasets['train'],
        batch_size=params.batch_size,
        batch_size_limit=params.batch_size_limit,
        batch_expansion_rate=params.batch_expansion_rate
    )
    train_collate_fn = make_bev_collate_fn(datasets['train'], params)
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=params.num_workers,
        pin_memory=True
    )

    if validation and 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        val_collate_fn = make_bev_collate_fn(datasets['val'], params)
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True
        )

    return dataloaders


def make_bev_collate_fn(dataset: BEVTrainingDataset, params: TrainingParams):
    """
    BEV collate function - simplified without ICP/point cloud processing.
    BEV images don't need 3D geometric alignment, so we skip patch-level matching.
    """

    def in_sorted_array(e: int, array: np.ndarray) -> bool:
        """Binary search in a sorted array; returns True if e is present."""
        pos = np.searchsorted(array, e)
        return (pos < array.size) and (array[pos] == e)

    def collate_fn(data_list):
        """
        Simplified batch builder for BEV images.
        Returns:
          batch (split into chunks of params.batch_split_size),
          positives_mask (N,N),
          negatives_mask (N,N),
          sampled_pairs (empty for BEV - no patch matching),
          positive_pairs (empty for BEV - no patch matching)
        """
        # Extract images and metadata
        images_list = [e[0] for e in data_list]  # (C, H, W) tensors
        images_np = np.asarray(images_list)
        images = torch.from_numpy(images_np)

        labels = [e[1] for e in data_list]
        poses = [e[2] for e in data_list]

        # Build positives/negatives masks (NxN)
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
        negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)

        # For BEV, we don't do patch-level matching
        # Return empty lists for sampled_pairs and positive_pairs
        sampled_pairs = np.array([], dtype=np.int64).reshape(0, 2)
        positive_pairs = []

        # Split batch into chunks for multi-stage training (if enabled)
        bss = params.batch_split_size
        if bss is None or bss <= 0:
            # No batch splitting - return entire batch directly as a list with one element
            # (to maintain consistent interface with multi-stage mode)
            batch = [images]
        else:
            # Split into sub-batches
            batch = []
            for k in range(0, len(images), bss):
                batch.append(images[k:k + bss])

        return batch, positives_mask, negatives_mask, sampled_pairs, positive_pairs

    return collate_fn
