"""
Inference and Clinical Safeguard Engine for WBC Classification.

Provides:
- WBCInferenceEngine: High-performance single-session model loader.
- Pre-inference quality and safeguard checks (blur, brightness, size).
- Full 5-class score distribution with clear disclaimer labels.
- Integrated Grad-CAM generation for visual explanation.
- Morphological summary descriptions for educational understanding.
- Resilient batch processing with error isolation.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image, ImageStat
import torch
import torch.nn.functional as F

from src.dataset import CLASS_NAMES, IDX_TO_CLASS
from src.preprocessing import get_val_transforms, denormalize_tensor
from src.models import get_model
from src.explainability import GradCAM, create_gradcam_overlay


# Biological and morphological educational references
WBC_MORPHOLOGY_NOTES = {
    "Basophil": (
        "Characterized by large, coarse, dark-purple/blue-black granules that frequently obscure "
        "the bi-lobed or S-shaped nucleus. Granules contain histamine and heparin, playing a key "
        "role in allergic and inflammatory responses."
    ),
    "Eosinophil": (
        "Distinguished by distinctive coarse orange-red to bright pink cytoplasmic granules and "
        "typically a bi-lobed (spectacle-shaped) nucleus. Involved in allergic reactions and parasitic defense."
    ),
    "Lymphocyte": (
        "Features a prominent, round or slightly indented, dense nucleus with deep purple chromatin "
        "and a narrow rim of light blue cytoplasm. Central effector cells in adaptive immunity."
    ),
    "Monocyte": (
        "The largest normal leukocyte in peripheral blood. Features an abundant gray-blue cytoplasm "
        "often with vacuoles or fine dust-like granules, and a folded, kidney-shaped or indented nucleus."
    ),
    "Neutrophil": (
        "The most abundant circulating leukocyte, characterized by a segmented, multilobed nucleus "
        "(typically 3 to 5 lobes connected by thin chromatin filaments) and pale pink cytoplasm with fine neutral granules."
    )
}


class WBCInferenceEngine:
    """
    Production-grade inference and explanation engine.
    Loads PyTorch weights once and caches them for fast, repeated CPU evaluations.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        print(f"Loading inference checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_name = checkpoint.get("model_name", "baseline_cnn")
        self.config = checkpoint.get("config", {})
        self.image_size = self.config.get("image_size", 128)

        # Initialize model
        self.model = get_model(self.model_name, num_classes=len(CLASS_NAMES), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM
        self.target_layer = self.model.get_gradcam_target_layer()
        self.gradcam = GradCAM(self.model, self.target_layer)
        self.transform = get_val_transforms(self.image_size)

    def validate_image_quality(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Runs automated quality safeguards:
        - Dimension check (minimum 64x64)
        - Brightness check (rejecting pitch black or bleached white frames)
        - Contrast / variance check (detecting extreme blur or blank images)
        """
        w, h = pil_image.size
        status = "PASSED"
        warnings = []

        if w < 64 or h < 64:
            status = "FAILED"
            warnings.append(f"Image resolution too low ({w}x{h}). Minimum recommended is 64x64.")

        # Brightness & contrast
        grayscale = pil_image.convert("L")
        stat = ImageStat.Stat(grayscale)
        mean_brightness = stat.mean[0]
        std_dev = stat.stddev[0]

        if mean_brightness < 20:
            if status != "FAILED":
                status = "WARNING"
            warnings.append("Image is severely underexposed (too dark).")
        elif mean_brightness > 240:
            if status != "FAILED":
                status = "WARNING"
            warnings.append("Image is severely overexposed (washed out).")

        if std_dev < 10:
            if status != "FAILED":
                status = "WARNING"
            warnings.append("Very low image variance / contrast; image may be completely blank or severely blurred.")

        return {
            "status": status,
            "width": w,
            "height": h,
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(std_dev, 2),
            "warnings": warnings,
            "disclaimer": "Quality checks are automated heuristic safeguards, not a clinical guarantee."
        }

    def predict_single(self, 
                       image_input: Union[str, Image.Image], 
                       target_explain_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs full inference and Grad-CAM explanation on a single input image.

        Args:
            image_input: File path (str) or PIL.Image instance.
            target_explain_class: Optional name of class to generate Grad-CAM for.
                                  If None, explains the top predicted class.

        Returns:
            Dictionary containing predicted class, confidence scores, Grad-CAM arrays, and quality checks.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            pil_image = Image.open(image_input).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")

        # 1. Quality validation
        quality_report = self.validate_image_quality(pil_image)

        # 2. Preprocessing
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = IDX_TO_CLASS[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # 4. Resolve explanation target
        if target_explain_class is not None and target_explain_class in CLASS_NAMES:
            explain_idx = CLASS_NAMES.index(target_explain_class)
        else:
            explain_idx = predicted_idx

        # 5. Grad-CAM Computation
        heatmap, explained_class_idx, explained_score = self.gradcam.generate_cam(
            input_tensor, target_class=explain_idx
        )

        # 6. Generate Overlays
        # Resize original PIL image to match tensor resolution for pixel-accurate overlay
        resized_orig = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        orig_np = np.array(resized_orig)
        colored_heatmap, overlay = create_gradcam_overlay(orig_np, heatmap, alpha=0.45, colormap_name="jet")

        # 7. Formulate structured response
        score_breakdown = {
            cls: round(float(probabilities[i]), 4) for i, cls in enumerate(CLASS_NAMES)
        }

        # Educational explanation narrative
        explained_name = IDX_TO_CLASS[explained_class_idx]
        morphology_text = WBC_MORPHOLOGY_NOTES.get(explained_name, "")
        explanation_narrative = (
            f"The deep learning model assigned its highest score ({confidence*100:.1f}%) to {predicted_class}. "
            f"Grad-CAM visualizes the convolutional receptive fields that most positively contributed to the score for {explained_name}. "
            f"Warmer regions (red/yellow) indicate features such as nuclear lobulation, chromatin condensation, or cytoplasmic granules "
            f"that influenced the prediction."
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "score_breakdown": score_breakdown,
            "explained_class": explained_name,
            "quality_report": quality_report,
            "original_image_rgb": orig_np,
            "heatmap": heatmap,
            "colored_heatmap_rgb": colored_heatmap,
            "overlay_rgb": overlay,
            "explanation_narrative": explanation_narrative,
            "morphological_reference": morphology_text,
            "model_name": self.model_name,
            "disclaimer": (
                "EDUCATIONAL RESEARCH PROTOTYPE ONLY: This software is designed for academic study and algorithm benchmarking. "
                "It is NOT a medical device, is NOT approved for clinical diagnostics, and MUST NOT be used for patient management."
            )
        }

    def predict_batch(self, 
                      image_inputs: List[Tuple[str, Union[str, Image.Image]]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of images with individual error resilience.
        """
        results = []
        for item_id, img_input in image_inputs:
            try:
                res = self.predict_single(img_input)
                res["item_id"] = item_id
                res["status"] = "SUCCESS"
                results.append(res)
            except Exception as e:
                results.append({
                    "item_id": item_id,
                    "status": "ERROR",
                    "error_message": str(e),
                    "predicted_class": "N/A",
                    "confidence": 0.0,
                    "score_breakdown": {}
                })
        return results
