"""
Comprehensive Model Evaluation and Error Analysis Pipeline for WBC Classification.

Reports and generates:
- Overall Accuracy, Macro Precision, Macro Recall, Macro F1-score.
- Per-class breakdown (Precision, Recall, F1, Support).
- Normalized and Raw Confusion Matrix Heatmaps.
- One-vs-Rest (OvR) ROC Curves with per-class and Macro ROC-AUC.
- Misclassified Image Gallery with true label, predicted label, and probability.
- CPU Inference Latency per sample (milliseconds).
- Exportable CSV and JSON reports saved under results/.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import argparse
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

from src.dataset import CLASS_NAMES, IDX_TO_CLASS, get_dataloaders
from src.models import get_model


def evaluate_model_on_test(checkpoint_path: str,
                           image_size: int = 128,
                           batch_size: int = 32,
                           subset_fraction: Optional[float] = None,
                           results_dir: str = "results",
                           device_name: str = "cpu") -> Dict[str, Any]:
    """
    Evaluates a trained checkpoint on held-out test data.
    """
    device = torch.device(device_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    figures_dir = os.path.join(results_dir, "figures")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "model")
    config = checkpoint.get("config", {})
    actual_image_size = config.get("image_size", image_size)

    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Get Held-out Test DataLoader
    print(f"Loading held-out test data (image_size={actual_image_size}, subset={subset_fraction})...")
    _, _, test_loader, _, _ = get_dataloaders(
        raw_dir="data/raw",
        processed_dir="data/processed",
        image_size=actual_image_size,
        batch_size=batch_size,
        subset_fraction=subset_fraction,
        seed=42
    )

    print(f"Test samples to evaluate: {len(test_loader.dataset)}")

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_paths: List[str] = []
    latencies: List[float] = []

    # 3. Inference Loop
    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / images.size(0))

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_paths.extend(paths)

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # 4. Compute Metrics
    acc = float(accuracy_score(all_targets_np, all_preds_np))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets_np, all_preds_np, average="macro", zero_division=0
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_targets_np, all_preds_np, labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0
    )

    avg_latency_ms = float(np.mean(latencies) * 1000.0)

    # Compute OvR ROC-AUC if at least 2 distinct classes present in targets
    unique_targets = np.unique(all_targets_np)
    roc_auc_dict = {}
    if len(unique_targets) == len(CLASS_NAMES):
        try:
            macro_roc_auc = float(roc_auc_score(all_targets_np, all_probs_np, multi_class="ovr", average="macro"))
            for i, cls in enumerate(CLASS_NAMES):
                binary_target = (all_targets_np == i).astype(int)
                roc_auc_dict[cls] = float(roc_auc_score(binary_target, all_probs_np[:, i]))
            roc_auc_dict["macro_ovr"] = macro_roc_auc
        except Exception as e:
            print(f"[Warning] ROC-AUC calculation error: {e}")
            macro_roc_auc = None
    else:
        macro_roc_auc = None

    print("\n================== EVALUATION REPORT ==================")
    print(f"Model Architecture  : {model_name}")
    print(f"Held-out Samples    : {len(all_targets_np)}")
    print(f"Accuracy            : {acc * 100:.2f}%")
    print(f"Macro F1-Score      : {f1_macro:.4f}")
    print(f"Macro Precision     : {precision_macro:.4f}")
    print(f"Macro Recall        : {recall_macro:.4f}")
    if macro_roc_auc is not None:
        print(f"Macro OvR ROC-AUC   : {macro_roc_auc:.4f}")
    print(f"Avg CPU Latency     : {avg_latency_ms:.2f} ms/image")
    print("=======================================================\n")

    # 5. Save Per-Class Metrics to CSV
    per_class_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Precision": [round(p, 4) for p in precision_per],
        "Recall": [round(r, 4) for r in recall_per],
        "F1-Score": [round(f, 4) for f in f1_per],
        "Support": [int(s) for s in support_per],
        "ROC-AUC": [round(roc_auc_dict.get(c, 0.0), 4) for c in CLASS_NAMES]
    })
    csv_per_class_path = os.path.join(reports_dir, f"{model_name}_per_class_metrics.csv")
    per_class_df.to_csv(csv_per_class_path, index=False)
    print(f"Saved per-class CSV: {csv_per_class_path}")

    # 6. Save JSON Metrics
    summary_report = {
        "model_name": model_name,
        "checkpoint": os.path.basename(checkpoint_path),
        "total_test_samples": len(all_targets_np),
        "accuracy": round(acc, 4),
        "macro_precision": round(precision_macro, 4),
        "macro_recall": round(recall_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "macro_roc_auc": round(macro_roc_auc, 4) if macro_roc_auc is not None else None,
        "average_cpu_latency_ms": round(avg_latency_ms, 2),
        "per_class": {
            cls: {
                "precision": round(float(precision_per[i]), 4),
                "recall": round(float(recall_per[i]), 4),
                "f1": round(float(f1_per[i]), 4),
                "support": int(support_per[i]),
                "roc_auc": round(float(roc_auc_dict.get(cls, 0.0)), 4)
            } for i, cls in enumerate(CLASS_NAMES)
        }
    }
    json_report_path = os.path.join(reports_dir, f"{model_name}_test_metrics.json")
    with open(json_report_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved JSON metrics: {json_report_path}")

    # 7. Generate and Save Confusion Matrix Plot
    plt.switch_backend('Agg')
    cm = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=True)
    ax.set_title(f"Confusion Matrix: {model_name} on Held-Out Test Set\n(Values: raw count, Heat: normalized recall)",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("True WBC Class", fontsize=11)
    ax.set_xlabel("Predicted WBC Class", fontsize=11)
    fig.tight_layout()
    cm_fig_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix: {cm_fig_path}")

    # 8. Generate and Save ROC Curves Plot
    if len(unique_targets) == len(CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve((all_targets_np == i).astype(int), all_probs_np[:, i])
            class_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {class_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title(f"One-vs-Rest ROC Curves: {model_name}", fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        roc_fig_path = os.path.join(figures_dir, f"{model_name}_roc_curves.png")
        fig.savefig(roc_fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved ROC curves: {roc_fig_path}")

    # 9. Extract Misclassified Images and Save Gallery
    misclassified_indices = np.where(all_targets_np != all_preds_np)[0]
    misclassified_rows = []
    for idx in misclassified_indices:
        misclassified_rows.append({
            "image_path": all_paths[idx],
            "filename": os.path.basename(all_paths[idx]),
            "true_class": IDX_TO_CLASS[all_targets_np[idx]],
            "predicted_class": IDX_TO_CLASS[all_preds_np[idx]],
            "confidence": round(float(all_probs_np[idx, all_preds_np[idx]]), 4)
        })

    misclassified_df = pd.DataFrame(misclassified_rows)
    csv_misc_path = os.path.join(reports_dir, f"{model_name}_misclassified_samples.csv")
    misclassified_df.to_csv(csv_misc_path, index=False)
    print(f"Saved misclassifications CSV: {csv_misc_path} (Total errors: {len(misclassified_rows)})")

    if len(misclassified_indices) > 0:
        n_show = min(12, len(misclassified_indices))
        n_cols = 4
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i in range(n_show):
            ax = axes_flat[i]
            idx = misclassified_indices[i]
            img = Image.open(all_paths[idx]).convert("RGB")
            ax.imshow(img)
            true_lbl = IDX_TO_CLASS[all_targets_np[idx]]
            pred_lbl = IDX_TO_CLASS[all_preds_np[idx]]
            conf = all_probs_np[idx, all_preds_np[idx]]
            ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf*100:.1f}%)",
                         color="red", fontsize=10, fontweight='bold')
            ax.axis("off")

        # Hide any unused subplots
        for j in range(n_show, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"Sample Misclassified Leukocyte Cases ({model_name})", fontsize=14, fontweight='bold')
        fig.tight_layout()
        misc_fig_path = os.path.join(figures_dir, f"{model_name}_misclassifications.png")
        fig.savefig(misc_fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved misclassifications gallery: {misc_fig_path}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WBC Deep Learning Classifier")
    parser.add_argument("--checkpoint", type=str, default="artifacts/baseline_cnn_best.pt")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset", type=float, default=None, help="Fraction of test data to evaluate")
    args = parser.parse_args()

    evaluate_model_on_test(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        subset_fraction=args.subset
    )
