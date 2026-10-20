"""
Model Calibration and Reliability Pipeline using Temperature Scaling.

Implements:
1. ModelWithTemperature: Post-processing temperature scaling wrapper (Guo et al., ICML 2017).
2. Tuning on validation data ONLY (strict protocol preventing test data leakage).
3. Expected Calibration Error (ECE) and Brier Score evaluation before and after scaling.
4. Publication-quality Reliability Diagrams (Calibration Curves).
5. Documented safeguards: Softmax values are model scores, not clinical certainty.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


def compute_calibration_metrics(probs: np.ndarray, 
                                targets: np.ndarray, 
                                n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and multi-class Brier Score.

    Args:
        probs: Softmax probability matrix (N, K).
        targets: True integer labels (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dict with 'ece', 'mce', 'brier_score', and bin details for reliability diagrams.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count_in_bin = int(np.sum(in_bin))
        bin_counts.append(count_in_bin)

        if count_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(accuracy_in_bin - confidence_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    # Multi-class Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    n_classes = probs.shape[1]
    one_hot_targets = np.eye(n_classes)[targets]
    brier_score = float(np.mean(np.sum((probs - one_hot_targets) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist()
    }


class ModelWithTemperature(nn.Module):
    """
    Wraps a trained model with a learned temperature scalar T.
    Scaled logits = z / T.
    Optimized strictly on the held-out validation set using negative log-likelihood.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Divides logits by temperature."""
        # Keep the effective temperature positive even if an optimizer step
        # briefly crosses zero; negative temperatures reverse class ranking.
        temperature = self.temperature.clamp_min(1e-3).unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def calibrate(self, val_loader, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """
        Learns optimal temperature T using validation data ONLY.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # 1. Collect all validation logits and targets
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(targets)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # 2. Compute pre-calibration metrics
        unscaled_probs = F.softmax(logits, dim=1).cpu().numpy()
        targets_np = labels.cpu().numpy()
        pre_metrics = compute_calibration_metrics(unscaled_probs, targets_np)

        # 3. Optimize Temperature via L-BFGS
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)

        # Ensure temperature remains positive
        optimal_t = float(self.temperature.clamp_min(1e-3).item())
        print(f"Optimal Temperature found on validation data: T = {optimal_t:.4f}")

        # 4. Compute post-calibration metrics
        scaled_logits = self.temperature_scale(logits)
        post_probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()
        post_metrics = compute_calibration_metrics(post_probs, targets_np)

        return {
            "optimal_temperature": optimal_t,
            "pre_calibration": pre_metrics,
            "post_calibration": post_metrics
        }


def plot_reliability_diagram(pre_metrics: Dict[str, Any], 
                             post_metrics: Dict[str, Any], 
                             save_path: str, 
                             model_name: str) -> None:
    """Plots comparative Reliability Diagrams (ECE calibration curves) before and after scaling."""
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bin_centers = np.linspace(0.05, 0.95, 10)

    # Pre-Calibration
    axes[0].bar(bin_centers, pre_metrics["bin_accs"], width=0.08, color="#1f77b4", alpha=0.7,
                edgecolor="black", label="Empirical Accuracy")
    axes[0].plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Calibration")
    axes[0].set_title(f"Uncalibrated {model_name}\nECE: {pre_metrics['ece']*100:.2f}% | Brier: {pre_metrics['brier_score']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Confidence Bin", fontsize=11)
    axes[0].set_ylabel("Empirical Accuracy", fontsize=11)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Post-Calibration
    axes[1].bar(bin_centers, post_metrics["bin_accs"], width=0.08, color="#2ca02c", alpha=0.7,
                edgecolor="black", label="Empirical Accuracy")
    axes[1].plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Calibration")
    axes[1].set_title(f"Temperature-Scaled {model_name}\nECE: {post_metrics['ece']*100:.2f}% | Brier: {post_metrics['brier_score']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Confidence Bin", fontsize=11)
    axes[1].set_ylabel("Empirical Accuracy", fontsize=11)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
