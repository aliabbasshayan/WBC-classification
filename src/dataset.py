"""
Dataset Loading, Splitting, and Manifest Management for WBC Classification.

This module provides:
- The 5 canonical White Blood Cell target classes:
  Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil.
- Manifest generation preserving official held-out TestA and creating a reproducible,
  leakage-free stratified validation split from Train.
- WBCDataset PyTorch Dataset class.
- get_dataloaders factory supporting configurable image size, batch size, and subset fractions.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    compute_class_weights
)

CLASS_NAMES: List[str] = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil"
]

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


class WBCDataset(Dataset):
    """
    PyTorch Dataset for White Blood Cell microscopy images.
    Loads images from pre-verified file manifests, converts to RGB,
    and applies corresponding torchvision transforms.
    """

    def __init__(self, 
                 items: List[Dict[str, Any]], 
                 transform=None):
        """
        Args:
            items: List of dicts, each containing:
                   {"path": <abs_path>, "class_name": <str>, "label": <int>, "slide_id": <str>}
            transform: Optional torchvision transform to apply.
        """
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        image_path = item["path"]
        label = item["label"]

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Error reading image {image_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Fallback to standard ToTensor if no transform provided
            image_tensor = get_val_transforms()(image)

        return image_tensor, label, image_path


def build_or_load_split_manifest(raw_dir: str = "data/raw",
                                 processed_dir: str = "data/processed",
                                 val_ratio: float = 0.15,
                                 seed: int = 42,
                                 force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Builds or loads a reproducible dataset manifest.
    
    Guarantees:
    - Official held-out TestA is strictly preserved as the test split.
    - A stratified validation split is created from the official Train split.
    - TestB is noted and excluded from 5-class evaluation (it only contains 2 classes).
    - Random state is deterministic.

    Returns:
        Dictionary with keys: 'train', 'val', 'test', 'metadata'.
    """
    os.makedirs(processed_dir, exist_ok=True)
    manifest_path = os.path.join(processed_dir, "split_manifest.json")

    if os.path.exists(manifest_path) and not force_rebuild:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        metadata = manifest.get("metadata", {})
        # A manifest is only reusable when its split-defining inputs match.
        # This prevents silently training with a stale seed or validation ratio.
        if (
            metadata.get("seed") == seed
            and metadata.get("val_ratio") == val_ratio
            and metadata.get("classes") == CLASS_NAMES
        ):
            return manifest

    raw_dir_abs = os.path.abspath(raw_dir)
    train_root = os.path.join(raw_dir_abs, "Train", "Train")
    test_root = os.path.join(raw_dir_abs, "TestA", "TestA")

    if not os.path.exists(train_root):
        raise FileNotFoundError(f"Expected Train directory at: {train_root}")
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"Expected TestA directory at: {test_root}")

    def collect_items(split_root: str) -> List[Dict[str, Any]]:
        collected = []
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(split_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(cls_dir, fname)
                    base = os.path.splitext(fname)[0]
                    slide_id = base.split("_")[0] if "_" in base else "unknown"
                    collected.append({
                        "path": fpath,
                        "filename": fname,
                        "class_name": cls_name,
                        "label": CLASS_TO_IDX[cls_name],
                        "slide_id": slide_id
                    })
        return collected

    train_val_pool = collect_items(train_root)
    test_items = collect_items(test_root)

    # Perform stratified split on Train pool
    labels = [item["label"] for item in train_val_pool]
    train_items, val_items = train_test_split(
        train_val_pool,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed
    )

    manifest = {
        "metadata": {
            "dataset": "Raabin-WBC",
            "val_ratio": val_ratio,
            "seed": seed,
            "train_count": len(train_items),
            "val_count": len(val_items),
            "test_count": len(test_items),
            "classes": CLASS_NAMES,
            "class_to_idx": CLASS_TO_IDX
        },
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def get_dataloaders(raw_dir: str = "data/raw",
                    processed_dir: str = "data/processed",
                    image_size: int = 128,
                    batch_size: int = 32,
                    num_workers: int = 0,
                    val_ratio: float = 0.15,
                    subset_fraction: Optional[float] = None,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, Dict[str, Any]]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.

    Args:
        raw_dir: Path to raw dataset root.
        processed_dir: Path to save processed metadata/manifests.
        image_size: Image resolution (e.g. 128 or 160).
        batch_size: Batch size for DataLoader (e.g. 16 or 32 for CPU).
        num_workers: Number of DataLoader worker processes (0 recommended on Windows).
        val_ratio: Validation fraction from Train set.
        subset_fraction: Optional fraction (e.g. 0.05) to sample a stratified subset for smoke tests.
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader, class_weights_tensor, manifest_metadata)
    """
    manifest = build_or_load_split_manifest(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        val_ratio=val_ratio,
        seed=seed
    )

    train_items = manifest["train"]
    val_items = manifest["val"]
    test_items = manifest["test"]

    # Stratified subset sampling if requested (for smoke testing / fast iterations)
    if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
        def sample_subset(items: List[Dict[str, Any]], frac: float) -> List[Dict[str, Any]]:
            by_cls = {c: [] for c in CLASS_NAMES}
            for it in items:
                by_cls[it["class_name"]].append(it)
            sampled = []
            rng = random.Random(seed)
            for c, cls_items in by_cls.items():
                k = max(2, int(len(cls_items) * frac))
                sampled.extend(rng.sample(cls_items, min(k, len(cls_items))))
            rng.shuffle(sampled)
            return sampled

        train_items = sample_subset(train_items, subset_fraction)
        val_items = sample_subset(val_items, subset_fraction)
        test_items = sample_subset(test_items, subset_fraction)

    # Compute class weights from training items
    train_counts = {c: 0 for c in CLASS_NAMES}
    for item in train_items:
        train_counts[item["class_name"]] += 1
    class_weights = compute_class_weights(train_counts, CLASS_NAMES)

    # Instantiate datasets with appropriate transforms
    train_dataset = WBCDataset(train_items, transform=get_train_transforms(image_size))
    val_dataset = WBCDataset(val_items, transform=get_val_transforms(image_size))
    test_dataset = WBCDataset(test_items, transform=get_test_transforms(image_size))

    # Construct DataLoaders
    # pin_memory is False on CPU-only machines to avoid overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    metadata = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_class_counts": train_counts,
        "class_weights": class_weights.tolist(),
        "image_size": image_size,
        "batch_size": batch_size
    }

    return train_loader, val_loader, test_loader, class_weights, metadata
