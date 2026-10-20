"""
Unit tests for WBC dataset loading, manifests, and preprocessing transforms.
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    build_or_load_split_manifest,
    WBCDataset,
    get_dataloaders
)
from src.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    denormalize_tensor,
    compute_class_weights
)


def test_class_definitions():
    """Verify standard 5 leukocyte classes are defined with bidirectional mappings."""
    assert len(CLASS_NAMES) == 5
    assert "Basophil" in CLASS_NAMES
    assert "Neutrophil" in CLASS_NAMES
    for idx, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[name] == idx
        assert IDX_TO_CLASS[idx] == name


def test_class_weights_computation():
    """Verify class weights produce positive float tensor with mean ~1.0."""
    counts = {
        "Basophil": 100,
        "Eosinophil": 300,
        "Lymphocyte": 1000,
        "Monocyte": 250,
        "Neutrophil": 3000
    }
    weights = compute_class_weights(counts, CLASS_NAMES, method="balanced")
    assert isinstance(weights, torch.Tensor)
    assert len(weights) == 5
    assert torch.all(weights > 0)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-4)
    # Minority class (Basophil) must have higher weight than majority class (Neutrophil)
    assert weights[CLASS_TO_IDX["Basophil"]] > weights[CLASS_TO_IDX["Neutrophil"]]


def test_preprocessing_transforms():
    """Verify image resizing, tensor conversion, and denormalization."""
    # Create a synthetic PIL image
    dummy_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
    
    transform = get_val_transforms(image_size=128)
    tensor = transform(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32

    # Test denormalization
    denorm = denormalize_tensor(tensor)
    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (128, 128, 3)
    assert denorm.dtype == np.uint8


def test_manifest_generation():
    """Verify manifest creation on real raw dataset directory."""
    manifest = build_or_load_split_manifest(
        raw_dir="data/raw",
        processed_dir="data/processed",
        val_ratio=0.15,
        seed=42
    )

    assert "train" in manifest
    assert "val" in manifest
    assert "test" in manifest
    assert "metadata" in manifest

    train_paths = set(item["path"] for item in manifest["train"])
    val_paths = set(item["path"] for item in manifest["val"])
    test_paths = set(item["path"] for item in manifest["test"])

    # Verify zero data leakage across splits
    assert len(train_paths.intersection(val_paths)) == 0, "Leakage detected between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage detected between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage detected between val and test!"

    # Verify all 5 classes are represented in each split
    for split_name, items in [("train", manifest["train"]), ("val", manifest["val"]), ("test", manifest["test"])]:
        split_classes = set(item["class_name"] for item in items)
        assert split_classes == set(CLASS_NAMES), f"Missing classes in {split_name} split!"


def test_manifest_rebuilds_when_split_configuration_changes(tmp_path):
    """A cached manifest must not silently ignore a changed validation ratio or seed."""
    first = build_or_load_split_manifest(
        raw_dir="data/raw",
        processed_dir=str(tmp_path),
        val_ratio=0.15,
        seed=42,
    )
    second = build_or_load_split_manifest(
        raw_dir="data/raw",
        processed_dir=str(tmp_path),
        val_ratio=0.20,
        seed=7,
    )

    assert first["metadata"]["val_ratio"] == 0.15
    assert first["metadata"]["seed"] == 42
    assert second["metadata"]["val_ratio"] == 0.20
    assert second["metadata"]["seed"] == 7
    assert second["metadata"]["val_count"] != first["metadata"]["val_count"]


def test_dataloaders_smoke():
    """Verify dataloaders return valid batches on a tiny stratified subset."""
    train_loader, val_loader, test_loader, weights, meta = get_dataloaders(
        raw_dir="data/raw",
        processed_dir="data/processed",
        image_size=64,
        batch_size=8,
        subset_fraction=0.01,
        seed=42
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    images, labels, paths = next(iter(train_loader))
    assert images.shape == (8, 3, 64, 64)
    assert labels.shape == (8,)
    assert len(paths) == 8
    assert all(0 <= lbl < 5 for lbl in labels)
