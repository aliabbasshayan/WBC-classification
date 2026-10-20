"""
Preprocessing and Data Augmentation Pipeline for White Blood Cell (WBC) Classification.

This module provides reproducible transformations for medical microscopy images:
- Training transformations include morphology-preserving augmentations:
  * Horizontal and vertical flips (WBCs have no canonical biological orientation on a slide).
  * Gentle rotations (+/- 15 degrees) without clipping nuclear structures.
  * Mild color jitter (brightness and contrast +/- 10%) simulating minor stain variations.
- Validation and Test transformations apply deterministic resizing and ImageNet normalization.
- Denormalization helper functions for Grad-CAM overlay and Streamlit visualization.
- Loss weighting utilities for handling class imbalance.
"""

from typing import Tuple, List, Dict
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Standard ImageNet normalization parameters (compatible with pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 128) -> T.Compose:
    """
    Constructs the data augmentation pipeline for training WBC classifiers.

    Args:
        image_size: Target square image dimension (e.g., 128 or 160 for CPU efficiency).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 128) -> T.Compose:
    """
    Constructs deterministic preprocessing for validation and test evaluation.

    Args:
        image_size: Target square image dimension.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms(image_size: int = 128) -> T.Compose:
    """Alias for validation transforms ensuring identical evaluation preprocessing."""
    return get_val_transforms(image_size=image_size)


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor (C, H, W) back to an RGB uint8 NumPy array (H, W, C).

    Args:
        tensor: Float tensor normalized with IMAGENET_MEAN and IMAGENET_STD.

    Returns:
        NumPy uint8 array with values in [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.detach().cpu().clone().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0.0, 1.0)
    return (img_np * 255.0).astype(np.uint8)


def compute_class_weights(class_counts: Dict[str, int], 
                          class_names: List[str], 
                          method: str = "balanced") -> torch.Tensor:
    """
    Computes class weights for CrossEntropyLoss to mitigate severe class imbalance
    (e.g., ~61% Neutrophil vs ~2% Basophil).

    Args:
        class_counts: Dictionary mapping class name to count in training set.
        class_names: Ordered list of class names.
        method: Weighting scheme ('balanced' or 'smoothed_inverse').

    Returns:
        1D torch.FloatTensor of class weights matching len(class_names).
    """
    total_samples = sum(class_counts[c] for c in class_names)
    n_classes = len(class_names)
    counts = np.array([class_counts[c] for c in class_names], dtype=np.float32)

    if method == "balanced":
        # Standard scikit-learn formula: n_samples / (n_classes * count)
        weights = total_samples / (n_classes * counts)
    elif method == "smoothed_inverse":
        # Square root smoothed inverse frequency (less extreme penalties)
        weights = np.sqrt(total_samples / counts)
        weights = weights / weights.sum() * n_classes
    else:
        raise ValueError(f"Unknown class weighting method: {method}")

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
