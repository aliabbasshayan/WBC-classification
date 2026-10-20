"""
Tests for Inference Engine, Grad-CAM generation, and Temperature Calibration.
"""

import os
import glob
import pytest
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASS_NAMES
from src.models import WBCBaselineCNN
from src.inference import WBCInferenceEngine
from src.calibration import ModelWithTemperature, compute_calibration_metrics
from src.explainability import (
    GradCAM,
    create_gradcam_overlay,
    model_randomization_sanity_check,
    region_perturbation_fidelity_check
)


def test_inference_engine_on_real_image():
    """Verify that WBCInferenceEngine executes on real dataset image and returns all expected fields."""
    checkpoint_path = "artifacts/baseline_cnn_best.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint not found yet.")

    engine = WBCInferenceEngine(checkpoint_path)
    
    # Find a real image from dataset
    sample_images = glob.glob("data/raw/Train/Train/*/*.jpg")
    if not sample_images:
        pytest.skip("No real dataset images found.")

    real_img = Image.open(sample_images[0]).convert("RGB")
    res = engine.predict_single(real_img)

    assert "predicted_class" in res
    assert res["predicted_class"] in CLASS_NAMES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "score_breakdown" in res
    assert len(res["score_breakdown"]) == 5
    assert "heatmap" in res
    assert res["heatmap"].shape == (128, 128)
    assert np.all(res["heatmap"] >= 0.0) and np.all(res["heatmap"] <= 1.0)
    assert "overlay_rgb" in res
    assert res["overlay_rgb"].shape == (128, 128, 3)
    assert res["quality_report"]["status"] == "PASSED"


def test_quality_safeguard_flags_blank_image():
    """Verify quality safeguard detects low-variance / blank synthetic images."""
    checkpoint_path = "artifacts/baseline_cnn_best.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint not found yet.")

    engine = WBCInferenceEngine(checkpoint_path)
    blank_img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    quality = engine.validate_image_quality(blank_img)
    assert quality["status"] == "WARNING"
    assert any("variance" in w.lower() for w in quality["warnings"])


def test_quality_safeguard_preserves_failed_status_for_small_images():
    """A failed dimension check must not be downgraded to a warning."""
    checkpoint_path = "artifacts/baseline_cnn_best.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint not found yet.")

    engine = WBCInferenceEngine(checkpoint_path)
    tiny_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    quality = engine.validate_image_quality(tiny_img)
    assert quality["status"] == "FAILED"


def test_gradcam_sanity_checks():
    """Verify Adebayo et al. weight randomization and fidelity tests on a small model."""
    model = WBCBaselineCNN(num_classes=5)
    target_layer = model.get_gradcam_target_layer()
    x = torch.randn(1, 3, 128, 128)

    # 1. Randomization test
    res_rand = model_randomization_sanity_check(model, x, target_layer, target_class=0)
    assert "passed_sanity_check" in res_rand
    assert "pearson_correlation" in res_rand

    # 2. Fidelity test
    explainer = GradCAM(model, target_layer)
    heatmap, _, _ = explainer.generate_cam(x, target_class=0)
    explainer.remove_hooks()

    res_fid = region_perturbation_fidelity_check(model, x, heatmap, target_class=0)
    assert "original_probability" in res_fid
    assert "high_attr_drop" in res_fid


def test_calibration_computation():
    """Verify ECE and Brier Score computation formulas."""
    probs = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.2, 0.1, 0.5, 0.1, 0.1]
    ])
    targets = np.array([0, 1, 2])

    metrics = compute_calibration_metrics(probs, targets, n_bins=5)
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_temperature_scaling_stays_positive():
    """Temperature scaling must not reverse logits by using a negative T."""
    model = WBCBaselineCNN(num_classes=5)
    wrapper = ModelWithTemperature(model)
    wrapper.temperature.data.fill_(-2.0)
    logits = torch.randn(2, 5)
    scaled = wrapper.temperature_scale(logits)
    assert torch.isfinite(scaled).all()
    assert wrapper.temperature.clamp_min(1e-3).item() > 0
