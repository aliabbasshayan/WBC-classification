"""
Explainable AI (XAI) Pipeline using Grad-CAM for WBC Classification.

Implements:
1. GradCAM: Gradient-weighted Class Activation Mapping for PyTorch CNNs.
2. Layer target resolution (conv4 for Baseline CNN, features[-1] for MobileNetV3).
3. Class-specific explanations (explain predicted vs alternative candidate classes).
4. Overlay generation with publication-quality matplotlib colormaps (JET / TURBO / VIRIDIS).
5. Formal XAI Sanity Checks:
   - Model parameter randomization check (Adebayo et al., NeurIPS 2018).
   - High-attribution region perturbation fidelity check.
6. Clinical and morphological disclaimer documentation.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import CLASS_NAMES, IDX_TO_CLASS
from src.preprocessing import denormalize_tensor


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Maps (Grad-CAM)
    for a specified convolutional target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self) -> None:
        """Removes PyTorch forward and backward hooks to prevent memory leaks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM for an input tensor (1, C, H, W).

        Args:
            input_tensor: Normalized image tensor of shape (1, 3, H, W).
            target_class: Integer index of class to explain. If None, uses model's argmax prediction.

        Returns:
            Tuple of:
            - heatmap: 2D NumPy float32 array in [0.0, 1.0] of shape (H, W)
            - target_class: Integer class index explained
            - class_score: Float probability or logit for the target class
        """
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        class_prob = float(probabilities[0, target_class].item())

        # Backward pass on target class score
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer registration.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum_i sum_j (dY / dA_k_ij)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of activation maps: L_GradCAM = ReLU( sum_k alpha_k * A_k )
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)

        # Upsample heatmap to match input image resolution (H, W)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), target_class, class_prob


def create_gradcam_overlay(original_image: np.ndarray, 
                           heatmap: np.ndarray, 
                           alpha: float = 0.45, 
                           colormap_name: str = "jet") -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays Grad-CAM heatmap onto the original RGB image.

    Args:
        original_image: NumPy uint8 RGB image of shape (H, W, 3).
        heatmap: NumPy float32 array in [0, 1] of shape (H, W).
        alpha: Overlay opacity weight between 0.0 and 1.0.
        colormap_name: Matplotlib colormap ('jet', 'turbo', 'viridis').

    Returns:
        Tuple of (colored_heatmap_uint8, blended_overlay_uint8).
    """
    cmap = plt.colormaps[colormap_name]
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel -> (H, W, 3) in [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255.0).astype(np.uint8)

    # Normalize original image to [0, 1] for linear blending
    orig_norm = original_image.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * orig_norm + alpha * colored_heatmap
    blended = np.clip(blended, 0.0, 1.0)
    overlay_uint8 = (blended * 255.0).astype(np.uint8)

    return colored_heatmap_uint8, overlay_uint8


def save_explanation_figure(original_image: np.ndarray,
                            heatmap: np.ndarray,
                            overlay: np.ndarray,
                            class_name: str,
                            probability: float,
                            save_path: str,
                            alternative_info: Optional[str] = None) -> None:
    """
    Renders and saves a comprehensive 3-panel explanation figure with colorbar and clinical note.
    """
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Input WBC Microscopy Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Standalone Grad-CAM Heatmap with Colorbar
    im1 = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Grad-CAM Attribution Map\nClass: {class_name} ({probability*100:.1f}%)",
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Feature Importance", fontsize=10)

    # Panel 3: Blended Overlay
    axes[2].imshow(overlay)
    title_text = f"Overlay Visualization\n{class_name}"
    if alternative_info:
        title_text += f"\n({alternative_info})"
    axes[2].set_title(title_text, fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Clinical & Biological Disclaimer footnote
    fig.text(0.5, 0.01,
             "DISCLAIMER: Grad-CAM highlights discriminative convolutional receptive fields (e.g. nucleus, granules).\n"
             "It represents algorithmic attribution, not guaranteed medical etiology or clinical diagnostic truth.",
             ha='center', fontsize=9, style='italic', color='#555555')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =====================================================================
# Sanity Checks for Saliency / Attribution Maps (Adebayo et al., 2018)
# =====================================================================

def model_randomization_sanity_check(model: nn.Module,
                                     input_tensor: torch.Tensor,
                                     target_layer: nn.Module,
                                     target_class: int) -> Dict[str, Any]:
    """
    Performs the cascading/independent model parameter randomization test.
    If the explanation is genuinely sensitive to learned features, randomizing
    the model's weights should destroy the attribution map structure (low correlation).
    """
    import copy

    # 1. Compute original CAM
    cam_explainer = GradCAM(model, target_layer)
    orig_cam, _, _ = cam_explainer.generate_cam(input_tensor, target_class=target_class)
    cam_explainer.remove_hooks()

    # 2. Deepcopy and randomize weights of the classifier & final conv layer
    rand_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in rand_model.parameters():
            param.normal_(0.0, 0.02)

    # Resolve target layer in randomized model
    rand_target = rand_model.get_gradcam_target_layer()
    rand_explainer = GradCAM(rand_model, rand_target)
    rand_cam, _, _ = rand_explainer.generate_cam(input_tensor, target_class=target_class)
    rand_explainer.remove_hooks()

    # 3. Compute Pearson correlation and Mean Absolute Difference
    orig_flat = orig_cam.flatten()
    rand_flat = rand_cam.flatten()

    std_orig = np.std(orig_flat)
    std_rand = np.std(rand_flat)

    if std_orig > 1e-6 and std_rand > 1e-6:
        correlation = float(np.corrcoef(orig_flat, rand_flat)[0, 1])
    else:
        correlation = 0.0

    mean_abs_diff = float(np.mean(np.abs(orig_cam - rand_cam)))

    # A valid attribution method has low correlation (< 0.5) when weights are scrambled
    passed_randomization = correlation < 0.50

    return {
        "passed_sanity_check": passed_randomization,
        "pearson_correlation": round(correlation, 4),
        "mean_absolute_difference": round(mean_abs_diff, 4),
        "conclusion": (
            "Attribution is strongly dependent on learned weights (Passed sanity check)."
            if passed_randomization else
            "Warning: Attribution is partially invariant to model weights."
        )
    }


def region_perturbation_fidelity_check(model: nn.Module,
                                       input_tensor: torch.Tensor,
                                       heatmap: np.ndarray,
                                       target_class: int,
                                       top_k_percent: float = 0.20) -> Dict[str, Any]:
    """
    Fidelity check: Masking the top-k% most influential pixels according to Grad-CAM
    should significantly drop the model's target class probability compared to masking random pixels.
    """
    model.eval()
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        orig_out = model(input_tensor)
        orig_prob = float(F.softmax(orig_out, dim=1)[0, target_class].item())

    # Find threshold for top-k% highest activation
    threshold = np.percentile(heatmap, 100 * (1.0 - top_k_percent))
    mask_high = torch.tensor(heatmap >= threshold, device=input_tensor.device).unsqueeze(0).unsqueeze(0)

    # 1. Perturb (zero out) high-attribution region
    perturbed_high = input_tensor.clone()
    perturbed_high[mask_high.expand_as(perturbed_high)] = 0.0

    with torch.no_grad():
        high_out = model(perturbed_high)
        high_prob = float(F.softmax(high_out, dim=1)[0, target_class].item())

    # 2. Perturb random pixels of equal count
    random_mask = torch.rand_like(mask_high.float()) >= (1.0 - top_k_percent)
    perturbed_random = input_tensor.clone()
    perturbed_random[random_mask.expand_as(perturbed_random)] = 0.0

    with torch.no_grad():
        rand_out = model(perturbed_random)
        rand_prob = float(F.softmax(rand_out, dim=1)[0, target_class].item())

    high_drop = orig_prob - high_prob
    rand_drop = orig_prob - rand_prob
    passed_fidelity = high_drop > rand_drop

    return {
        "original_probability": round(orig_prob, 4),
        "probability_after_high_attr_mask": round(high_prob, 4),
        "probability_after_random_mask": round(rand_prob, 4),
        "high_attr_drop": round(high_drop, 4),
        "random_attr_drop": round(rand_drop, 4),
        "fidelity_verified": passed_fidelity,
        "conclusion": (
            "High-attribution regions causally drive model confidence (Fidelity confirmed)."
            if passed_fidelity else
            "Marginal fidelity difference."
        )
    }
