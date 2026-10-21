"""
FastAPI REST Service for Explainable White Blood Cell Classification.

Endpoints:
- GET  /health      : Service health and active model metadata
- GET  /model/info  : Model architecture, target classes, and image dimensions
- POST /predict     : Single-image classification with quality checks and Grad-CAM overlay
- POST /explain     : Class-specific visual attribution with Base64 heatmap and overlay
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import io
import base64
from typing import Dict, Any, Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.dataset import CLASS_NAMES
from src.inference import WBCInferenceEngine
from src.explainability import create_gradcam_overlay

app = FastAPI(
    title="Explainable WBC Classification API",
    description="REST API providing deep learning leukocyte predictions and Grad-CAM visual explanations.",
    version="1.0.0"
)

# Enable CORS for cross-origin access (e.g. Android devices, web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
_engine: Optional[WBCInferenceEngine] = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def get_engine() -> WBCInferenceEngine:
    global _engine
    if _engine is None:
        # Priority: MobileNetV3-Small > Baseline CNN
        mobile_checkpoint = os.path.join(project_root, "artifacts", "mobilenet_v3_small_best.pt")
        baseline_checkpoint = os.path.join(project_root, "artifacts", "baseline_cnn_best.pt")
        if os.path.exists(mobile_checkpoint):
            ckpt = mobile_checkpoint
        elif os.path.exists(baseline_checkpoint):
            ckpt = baseline_checkpoint
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model checkpoint found in artifacts/. Please train a model first."
            )
        _engine = WBCInferenceEngine(ckpt)
    return _engine


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    score_breakdown: Dict[str, float]
    quality_status: str
    quality_warnings: list
    explanation_narrative: str
    overlay_base64: Optional[str] = None
    model_name: str
    disclaimer: str


@app.get("/health", tags=["Status"])
def health_check():
    """Health check endpoint reporting API status and model readiness."""
    try:
        engine = get_engine()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": engine.model_name,
            "image_size": engine.image_size,
            "classes": CLASS_NAMES
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": str(e)
        }


@app.get("/model/info", tags=["Metadata"])
def model_info():
    """Returns technical specifications of the active deep learning classifier."""
    engine = get_engine()
    return {
        "model_name": engine.model_name,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "input_resolution": f"{engine.image_size}x{engine.image_size}",
        "device": str(engine.device),
        "disclaimer": "Academic educational prototype. Not for medical diagnosis."
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Leukocyte microscopic image (JPEG or PNG)"),
    explain_class: Optional[str] = Query(None, description="Optional WBC class to explain with Grad-CAM")
):
    """
    Classifies an uploaded WBC image, runs quality validation, and returns Grad-CAM overlay as Base64.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Must be JPEG or PNG."
        )

    engine = get_engine()

    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file is too large. Maximum supported size is 10 MB."
            )
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unreadable image file: {e}"
        )

    # Perform inference and Grad-CAM
    try:
        res = engine.predict_single(pil_image, target_explain_class=explain_class)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e}"
        )

    # Encode overlay image to Base64 PNG
    overlay_pil = Image.fromarray(res["overlay_rgb"])
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return PredictionResponse(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        score_breakdown=res["score_breakdown"],
        quality_status=res["quality_report"]["status"],
        quality_warnings=res["quality_report"]["warnings"],
        explanation_narrative=res["explanation_narrative"],
        overlay_base64=overlay_b64,
        model_name=res["model_name"],
        disclaimer=res["disclaimer"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
