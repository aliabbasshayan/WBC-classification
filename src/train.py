"""
Model Training and Checkpoint Management Pipeline for WBC Classification.

Supports:
- Baseline CNN and MobileNetV3-Small architectures.
- Class-weighted CrossEntropyLoss for severe class imbalance.
- Early stopping with configurable patience.
- Learning rate scheduling with ReduceLROnPlateau.
- Checkpoint persistence and resumable training.
- Metric logging (loss, accuracy, macro-F1).
- Automated training curve visualization and report export.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import argparse
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score

from src.dataset import CLASS_NAMES, get_dataloaders
from src.models import get_model



def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, 
                    dataloader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device) -> Tuple[float, float, float]:
    """Runs a single training epoch and returns (loss, accuracy, macro_f1)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = float(accuracy_score(all_targets, all_preds))
    epoch_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model: nn.Module, 
             dataloader, 
             criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float, float]:
    """Runs validation evaluation and returns (loss, accuracy, macro_f1)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = float(accuracy_score(all_targets, all_preds))
    epoch_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return epoch_loss, epoch_acc, epoch_f1


def plot_training_curves(history: Dict[str, list], save_path: str, model_title: str) -> None:
    """Plots and saves loss, accuracy, and macro-F1 curves over epochs."""
    plt.switch_backend('Agg')
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'r-s', label='Val Loss')
    axes[0].set_title(f"{model_title} - CrossEntropy Loss", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], 'b-o', label='Train Accuracy')
    axes[1].plot(epochs, history["val_acc"], 'r-s', label='Val Accuracy')
    axes[1].set_title(f"{model_title} - Accuracy", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # Macro-F1
    axes[2].plot(epochs, history["train_f1"], 'b-o', label='Train Macro-F1')
    axes[2].plot(epochs, history["val_f1"], 'r-s', label='Val Macro-F1')
    axes[2].set_title(f"{model_title} - Macro-F1 Score", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def run_training(model_name: str = "baseline_cnn",
                 image_size: int = 128,
                 batch_size: int = 32,
                 epochs: int = 10,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 4,
                 subset_fraction: Optional[float] = None,
                 use_class_weights: bool = True,
                 pretrained: bool = True,
                 resume_path: Optional[str] = None,
                 artifacts_dir: str = "artifacts",
                 results_dir: str = "results",
                 seed: int = 42) -> Dict[str, Any]:
    """
    Executes the full training loop with checkpointing, early stopping, and metric tracking.
    """
    set_seed(seed)
    device = torch.device("cpu")
    print(f"[{model_name.upper()}] Starting training on device: {device}")

    os.makedirs(artifacts_dir, exist_ok=True)
    figures_dir = os.path.join(results_dir, "figures")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data (image_size={image_size}, batch_size={batch_size}, subset_fraction={subset_fraction})...")
    train_loader, val_loader, test_loader, class_weights, metadata = get_dataloaders(
        raw_dir="data/raw",
        processed_dir="data/processed",
        image_size=image_size,
        batch_size=batch_size,
        subset_fraction=subset_fraction,
        seed=seed
    )

    print(f"Train samples: {metadata['train_samples']} | Val samples: {metadata['val_samples']}")
    print(f"Class distribution: {metadata['train_class_counts']}")

    # 2. Build Model
    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=pretrained)
    model.to(device)

    # 3. Criterion & Optimizer
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Applied class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    start_epoch = 1
    best_val_f1 = -1.0
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": []
    }

    # Resume from checkpoint if requested
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("best_metric", -1.0)
        if "history" in checkpoint:
            history = checkpoint["history"]
        print(f"Resumed from epoch {start_epoch} with prior best Val Macro-F1: {best_val_f1:.4f}")

    best_checkpoint_path = os.path.join(artifacts_dir, f"{model_name}_best.pt")
    latest_checkpoint_path = os.path.join(artifacts_dir, f"{model_name}_latest.pt")

    epochs_without_improvement = 0
    total_training_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | LR: {current_lr:.6f}")

        # Save latest checkpoint (for resume)
        torch.save({
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_val_f1,
            "history": history,
            "config": {
                "image_size": image_size,
                "batch_size": batch_size,
                "classes": CLASS_NAMES
            }
        }, latest_checkpoint_path)

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            print(f"  --> [NEW BEST] Val Macro-F1: {best_val_f1:.4f}. Saving checkpoint: {best_checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "best_metric": best_val_f1,
                "val_accuracy": val_acc,
                "config": {
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "classes": CLASS_NAMES
                }
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs without improvement in {patience} epochs.")
                break

    total_training_time = time.time() - total_training_start
    print(f"[{model_name.upper()}] Training finished in {total_training_time:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")

    # Plot and save curves
    curve_path = os.path.join(figures_dir, f"{model_name}_training_curves.png")
    plot_training_curves(history, curve_path, model_name)
    print(f"Saved training curves: {curve_path}")

    # Save training report / history
    history_path = os.path.join(reports_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "best_val_macro_f1": best_val_f1,
            "total_epochs_trained": len(history["train_loss"]),
            "total_time_seconds": total_training_time,
            "history": history,
            "config": {
                "image_size": image_size,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "use_class_weights": use_class_weights,
                "subset_fraction": subset_fraction,
                "seed": seed
            }
        }, f, indent=2)
    print(f"Saved history: {history_path}")

    return {
        "model_name": model_name,
        "best_val_macro_f1": best_val_f1,
        "best_checkpoint_path": best_checkpoint_path,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WBC Deep Learning Classifier")
    parser.add_argument("--model", type=str, default="baseline_cnn", choices=["baseline_cnn", "mobilenet_v3_small"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--subset", type=float, default=None, help="Fraction of data to sample (e.g. 0.05 for smoke test)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_training(
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        subset_fraction=args.subset,
        resume_path=args.resume,
        seed=args.seed
    )
