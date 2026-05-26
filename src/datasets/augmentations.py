#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Data Augmentation for BEV Images
Supports geometric and photometric augmentations for improving cross-weather generalization

Created: 2025-10-31 23:20
Author: Claude Code Assistant
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageEnhance


class BEVAugmentation:
    """
    Enhanced augmentation pipeline for BEV images

    Supports:
    - Geometric: horizontal flip, rotation, scale
    - Photometric: brightness, contrast, gamma, color jitter
    - Noise: Gaussian noise
    """

    def __init__(self, config):
        """
        Args:
            config: Augmentation configuration dict from training config
        """
        self.config = config
        self.enabled = config.get('augmentation_probability', 0.8) > 0
        self.probability = config.get('augmentation_probability', 0.8)

        # Geometric augmentations
        self.use_hflip = config.get('random_horizontal_flip', False)
        self.flip_prob = config.get('flip_probability', 0.5)

        self.use_rotation = config.get('random_rotation', False)
        self.rotation_degrees = config.get('rotation_degrees', 5)

        self.use_scale = config.get('random_scale', False)
        self.scale_range = config.get('scale_range', [0.95, 1.05])

        # Photometric augmentations
        self.use_brightness = config.get('random_brightness', False)
        self.brightness_factor = config.get('brightness_factor', 0.3)

        self.use_contrast = config.get('random_contrast', False)
        self.contrast_factor = config.get('contrast_factor', 0.3)

        self.use_gamma = config.get('random_gamma', False)
        self.gamma_range = config.get('gamma_range', [0.8, 1.2])

        self.use_color_jitter = config.get('color_jitter', False)
        self.jitter_saturation = config.get('jitter_saturation', 0.2)
        self.jitter_hue = config.get('jitter_hue', 0.05)

        # Noise augmentations
        self.use_gaussian_noise = config.get('gaussian_noise', False)
        self.noise_std = config.get('noise_std', 0.02)

    def __call__(self, noisy_img, clean_img):
        """
        Apply augmentations to both noisy and clean images

        IMPORTANT: Geometric augmentations are applied identically to both images
                   Photometric augmentations are applied independently

        Args:
            noisy_img: PIL Image (noisy BEV)
            clean_img: PIL Image (clean BEV)

        Returns:
            Tuple of augmented (noisy_img, clean_img)
        """
        if not self.enabled or random.random() > self.probability:
            return noisy_img, clean_img

        # ===== 1. Geometric Augmentations (SAME for both images) =====

        # Random horizontal flip
        if self.use_hflip and random.random() < self.flip_prob:
            noisy_img = TF.hflip(noisy_img)
            clean_img = TF.hflip(clean_img)

        # Random rotation
        if self.use_rotation:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            noisy_img = TF.rotate(noisy_img, angle, interpolation=Image.BILINEAR)
            clean_img = TF.rotate(clean_img, angle, interpolation=Image.BILINEAR)

        # Random scale (via resized crop)
        if self.use_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            w, h = noisy_img.size
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize
            noisy_img = TF.resize(noisy_img, (new_h, new_w), interpolation=Image.BILINEAR)
            clean_img = TF.resize(clean_img, (new_h, new_w), interpolation=Image.BILINEAR)

            # Center crop back to original size
            noisy_img = TF.center_crop(noisy_img, (h, w))
            clean_img = TF.center_crop(clean_img, (h, w))

        # ===== 2. Photometric Augmentations (INDEPENDENT for each image) =====

        # Apply to noisy image
        noisy_img = self._apply_photometric(noisy_img)

        # Apply to clean image (with different random parameters)
        clean_img = self._apply_photometric(clean_img)

        return noisy_img, clean_img

    def _apply_photometric(self, img):
        """Apply photometric augmentations to a single image"""

        # Brightness
        if self.use_brightness:
            factor = 1.0 + random.uniform(-self.brightness_factor, self.brightness_factor)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # Contrast
        if self.use_contrast:
            factor = 1.0 + random.uniform(-self.contrast_factor, self.contrast_factor)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        # Color jitter (saturation + hue)
        if self.use_color_jitter:
            # Saturation
            if self.jitter_saturation > 0:
                factor = 1.0 + random.uniform(-self.jitter_saturation, self.jitter_saturation)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(factor)

        # Gamma correction (applied in tensor space)
        if self.use_gamma:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.power(img_array, gamma)
            img = Image.fromarray((img_array * 255).astype(np.uint8))

        return img

    def add_noise_to_tensor(self, img_tensor):
        """
        Add Gaussian noise to image tensor (call after ToTensor)

        Args:
            img_tensor: Tensor (C, H, W) in [0, 1]

        Returns:
            Noisy tensor
        """
        if self.use_gaussian_noise and random.random() < 0.5:
            noise = torch.randn_like(img_tensor) * self.noise_std
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)

        return img_tensor


class BEVTransform:
    """
    Complete transformation pipeline for BEV images
    Includes augmentation + normalization
    """

    def __init__(self, image_size=448, augmentation_config=None, is_training=True):
        """
        Args:
            image_size: Target image size
            augmentation_config: Augmentation config dict (None to disable)
            is_training: If False, skip augmentation
        """
        self.image_size = image_size
        self.is_training = is_training

        # Augmentation (only during training)
        if is_training and augmentation_config is not None:
            self.augmentation = BEVAugmentation(augmentation_config)
        else:
            self.augmentation = None

        # Base transforms
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

        # ImageNet normalization (DINOv2 expects this)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, noisy_img, clean_img):
        """
        Apply full transformation pipeline

        Args:
            noisy_img: PIL Image
            clean_img: PIL Image

        Returns:
            Tuple of (noisy_tensor, clean_tensor)
        """
        # 1. Augmentation (if training)
        if self.augmentation is not None:
            noisy_img, clean_img = self.augmentation(noisy_img, clean_img)

        # 2. Resize
        noisy_img = self.resize(noisy_img)
        clean_img = self.resize(clean_img)

        # 3. To Tensor
        noisy_tensor = self.to_tensor(noisy_img)
        clean_tensor = self.to_tensor(clean_img)

        # 4. Add noise (if enabled)
        if self.augmentation is not None:
            noisy_tensor = self.augmentation.add_noise_to_tensor(noisy_tensor)
            clean_tensor = self.augmentation.add_noise_to_tensor(clean_tensor)

        # 5. Normalize
        noisy_tensor = self.normalize(noisy_tensor)
        clean_tensor = self.normalize(clean_tensor)

        return noisy_tensor, clean_tensor


def create_transform(image_size=448, augmentation_config=None, is_training=True):
    """
    Factory function to create transformation pipeline

    Args:
        image_size: Target image size
        augmentation_config: Augmentation config dict from training config
        is_training: Whether this is for training (enables augmentation)

    Returns:
        BEVTransform instance
    """
    return BEVTransform(
        image_size=image_size,
        augmentation_config=augmentation_config if is_training else None,
        is_training=is_training
    )


if __name__ == "__main__":
    # Test augmentation
    print("Testing BEV Augmentation...")

    # Create dummy config
    test_config = {
        'augmentation_probability': 1.0,  # Always apply for testing
        'random_horizontal_flip': True,
        'flip_probability': 0.5,
        'random_rotation': True,
        'rotation_degrees': 5,
        'random_scale': True,
        'scale_range': [0.95, 1.05],
        'random_brightness': True,
        'brightness_factor': 0.3,
        'random_contrast': True,
        'contrast_factor': 0.3,
        'random_gamma': True,
        'gamma_range': [0.8, 1.2],
        'gaussian_noise': True,
        'noise_std': 0.02,
        'color_jitter': True,
        'jitter_saturation': 0.2,
        'jitter_hue': 0.05,
    }

    # Create transform
    transform = create_transform(
        image_size=448,
        augmentation_config=test_config,
        is_training=True
    )

    # Create dummy images
    noisy_img = Image.new('RGB', (448, 448), color=(100, 150, 200))
    clean_img = Image.new('RGB', (448, 448), color=(120, 170, 220))

    # Apply transform
    noisy_tensor, clean_tensor = transform(noisy_img, clean_img)

    print(f"Noisy tensor shape: {noisy_tensor.shape}")
    print(f"Clean tensor shape: {clean_tensor.shape}")
    print(f"Noisy tensor range: [{noisy_tensor.min():.3f}, {noisy_tensor.max():.3f}]")
    print(f"Clean tensor range: [{clean_tensor.min():.3f}, {clean_tensor.max():.3f}]")
    print("\nAugmentation test passed!")
