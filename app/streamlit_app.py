"""
Explainable White Blood Cell Classification Using Deep Learning and Grad-CAM
Interactive Streamlit Web Application

Sections:
A. Home & Project Overview
B. Single Cell Classification & Grad-CAM
C. Explainability & Attention Analysis (Sanity Checks)
D. Batch Screening & CSV Export
E. Model Evaluation Dashboard & ROC Curves
F. Error Analysis & Misclassification Explorer
G. Project Documentation, Ethics & Viva Guide
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
DATA_ROOT = Path(project_root) / "data"
RESULTS_ROOT = Path(project_root) / "results"

import io
import json
import glob
import time
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.dataset import CLASS_NAMES, IDX_TO_CLASS
from src.inference import WBCInferenceEngine, WBC_MORPHOLOGY_NOTES
from src.explainability import (
    GradCAM, 
    create_gradcam_overlay, 
    model_randomization_sanity_check, 
    region_perturbation_fidelity_check
)
from src.calibration import compute_calibration_metrics

# Streamlit Page Configuration
st.set_page_config(
    page_title="CellScope | Explainable WBC AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --primary: #5b4bdb;
        --secondary: #0ea5a4;
        --surface: #ffffff;
        --line: #e7eaf3;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #111936 0%, #1d2450 100%); }
    [data-testid="stSidebar"] * { color: #f6f7fb !important; }
    [data-testid="stSidebar"] .stRadio label { border-radius: 10px; padding: 0.3rem 0.4rem; }
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
        padding: 0.85rem 1rem; box-shadow: 0 8px 24px rgba(34, 42, 75, 0.06);
    }
    [data-testid="stMetricValue"] { color: var(--ink); }
    .hero {
        padding: 2.2rem 2.4rem; border-radius: 24px; margin-bottom: 1.2rem;
        background: radial-gradient(circle at 88% 10%, rgba(45, 212, 191, .35), transparent 30%),
                    linear-gradient(120deg, #171d46 0%, #393181 52%, #147d86 100%);
        color: white; box-shadow: 0 18px 45px rgba(29, 37, 92, .2);
    }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.05; margin: 0 0 .65rem; }
    .hero p { color: #dfe5ff; max-width: 760px; font-size: 1.08rem; margin: 0; }
    .eyebrow { color: #8ef0df; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 800; }
    .section-title { color: var(--ink); font-weight: 800; margin-top: 1.5rem; }
    .feature-card {
        min-height: 150px; padding: 1.2rem; border: 1px solid var(--line); border-radius: 16px;
        background: linear-gradient(145deg, #fff, #f7f8fc); box-shadow: 0 8px 22px rgba(34, 42, 75, .05);
    }
    .feature-card h4 { margin: 0 0 .5rem; color: var(--ink); }
    .feature-card p { color: var(--muted); font-size: .9rem; margin: 0; }
    .prediction-card {
        border: 1px solid #d9d5ff; border-radius: 18px; padding: 1.25rem;
        background: linear-gradient(145deg, #f8f7ff, #effcfb);
    }
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: var(--ink);
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem; color: var(--muted);
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.5rem;
    }
    .disclaimer-box {
        background-color: #fff8e7; border: 1px solid #f6d98b; border-left: 5px solid #f59e0b;
        padding: 0.85rem;
        border-radius: 12px;
        font-size: 0.9rem;
        color: #7a4b00;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stButton > button { border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inference_engine(model_type: str = "baseline_cnn") -> Optional[WBCInferenceEngine]:
    """Caches model weights once per session."""
    checkpoint_map = {
        "Baseline CNN (Custom 4-Stage)": "artifacts/baseline_cnn_best.pt",
        "MobileNetV3-Small (Transfer Learning)": "artifacts/mobilenet_v3_small_best.pt"
    }
    ckpt_path = str(Path(project_root) / checkpoint_map.get(model_type, "artifacts/baseline_cnn_best.pt"))
    if not os.path.exists(ckpt_path):
        return None
    try:
        return WBCInferenceEngine(ckpt_path)
    except Exception as e:
        st.error(f"Error loading model weights from {ckpt_path}: {e}")
        return None


def get_available_models() -> List[str]:
    """Finds which model checkpoints exist on disk."""
    available = []
    if os.path.exists(Path(project_root) / "artifacts" / "baseline_cnn_best.pt"):
        available.append("Baseline CNN (Custom 4-Stage)")
    if os.path.exists(Path(project_root) / "artifacts" / "mobilenet_v3_small_best.pt"):
        available.append("MobileNetV3-Small (Transfer Learning)")
    return available


def get_dataset_counts() -> Dict[str, int]:
    """Return counts from the generated manifest instead of hardcoded UI claims."""
    manifest_path = str(Path(project_root) / "data" / "processed" / "split_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return {
            "total": sum(
                manifest.get("metadata", {}).get(key, 0)
                for key in ("train_count", "val_count", "test_count")
            ),
            "test": manifest.get("metadata", {}).get("test_count", 0),
        }
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>ACADEMIC RESEARCH & EDUCATIONAL PROTOTYPE ONLY</strong><br>
        This application is an educational engineering benchmark built for final-year university evaluation. 
        It is NOT a certified medical diagnostic device. Softmax outputs are statistical confidence scores, not clinical certainty. 
        Do NOT use this tool for clinical decision-making or patient diagnosis.
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("## 🔬 CellScope AI")
page = st.sidebar.radio(
    "Select Workflow Module:",
    [
        "🏠 Home & Project Overview",
        "🔬 Single Cell Classification",
        "🧩 Explainability & Attention",
        "📁 Batch Screening",
        "📊 Benchmark Evaluation",
        "🔍 Error & Misclassification Analysis",
        "📜 Methodology & Viva Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
available_models = get_available_models()

if available_models:
    preferred_model = "MobileNetV3-Small (Transfer Learning)"
    default_index = available_models.index(preferred_model) if preferred_model in available_models else 0
    selected_model_type = st.sidebar.selectbox(
        "Active Architecture:", available_models, index=default_index,
        help="MobileNetV3 is the recommended checkpoint because it has the strongest validation macro-F1."
    )
    engine = load_inference_engine(selected_model_type)
    if engine:
        st.sidebar.success(f"Loaded: `{engine.model_name}`\nInput Size: `{engine.image_size}x{engine.image_size}`")
    else:
        st.sidebar.error("Checkpoint failed to initialize.")
else:
    st.sidebar.warning("⚠️ No trained checkpoint found in `artifacts/`. Please train a model first.")
    selected_model_type = None
    engine = None

st.sidebar.markdown("---")
st.sidebar.info("**Hardware Config:**\nIntel Core i7 CPU (4 Cores / 8 Threads)\n16 GB RAM | Windows 11\nExecution: 100% CPU (No CUDA)")


# =====================================================================
# PAGE A: HOME & PROJECT OVERVIEW
# =====================================================================
if page == "🏠 Home & Project Overview":
    st.markdown(
        '<div class="hero"><div class="eyebrow">CellScope AI · WBC research workspace</div>'
        '<h1>See the cell. Understand the prediction.</h1>'
        '<p>A focused, explainable computer-vision workspace for five-class white blood cell screening, '
        'with confidence-aware predictions and Grad-CAM visual evidence.</p></div>',
        unsafe_allow_html=True
    )
    
    render_disclaimer()

    dataset_counts = get_dataset_counts()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_label = f"{dataset_counts['total']:,}" if dataset_counts else "Unavailable"
        st.metric("Manifest Images", total_label, "Raabin-WBC")
    with col2:
        st.metric("Target Classes", "5", "Leukocyte Types")
    with col3:
        test_label = f"TestA ({dataset_counts['test']:,})" if dataset_counts else "TestA"
        st.metric("Primary Test Split", test_label, "Held-Out")
    with col4:
        st.metric("CPU Latency", "Measured per run", "See inference panel")

    st.markdown('<h3 class="section-title">A clearer path from image to insight</h3>', unsafe_allow_html=True)
    feature_cols = st.columns(3)
    features = [
        ("01 · Classify", "Upload one cell or screen a batch with a lightweight CPU-friendly deep-learning model."),
        ("02 · Explain", "Inspect where the network focused with an interactive heatmap and morphology notes."),
        ("03 · Validate", "Compare held-out metrics, confusion patterns, calibration and error cases before trusting results.")
    ]
    for col, (title, text) in zip(feature_cols, features):
        with col:
            st.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Project Summary & Engineering Objectives")
    st.markdown("""
    Automated White Blood Cell (WBC) differential counting from peripheral blood smears is a critical step in hematological diagnostics.
    Traditional laboratory diagnosis requires labor-intensive manual light microscopy by skilled pathologists.
    This project delivers a **transparent, CPU-optimized computer vision system** that combines:
    1. **Morphology-Preserving Deep Learning**: Custom 4-stage convolutional baseline CNN and lightweight MobileNetV3-Small architectures.
    2. **Class Imbalance Mitigation**: Class-weighted CrossEntropy loss addressing the natural physiological skew (~61% Neutrophil vs ~2% Basophil).
    3. **Visual Explainability via Grad-CAM**: Spatial heatmaps revealing convolutional feature attribution to corroborate predictions against biological landmarks (nuclear lobes, chromatin density, cytoplasmic granules).
    4. **Safety & Calibration**: Automated pre-inference image quality checks and temperature scaling calibration for truthful confidence estimation.
    """)

    st.markdown("### 🧬 Target Leukocyte Morphology Reference")
    cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        with cols[i]:
            st.markdown(f"**{cls}**")
            st.caption(WBC_MORPHOLOGY_NOTES[cls][:140] + "...")

    # Display dataset distribution figure if generated
    if (RESULTS_ROOT / "figures" / "class_distribution.png").exists():
        st.markdown("### 📊 Official Dataset Class Distribution")
        st.image(str(RESULTS_ROOT / "figures" / "class_distribution.png"), use_container_width=True)

    if (RESULTS_ROOT / "figures" / "dataset_samples_grid.png").exists():
        st.markdown("### 🔬 Microscopic Smear Samples")
        st.image(str(RESULTS_ROOT / "figures" / "dataset_samples_grid.png"), use_container_width=True)


# =====================================================================
# PAGE B: SINGLE CELL CLASSIFICATION & GRAD-CAM
# =====================================================================
elif page == "🔬 Single Cell Classification":
    st.markdown("## 🔬 Single Leukocyte Classification & Grad-CAM Analysis")
    render_disclaimer()

    if engine is None:
        st.error("No trained checkpoint available. Run `python src/train.py` to produce a valid model checkpoint.")
        st.stop()

    input_mode = st.radio("Choose Input Method:", ["Upload Image File", "Select Benchmark Sample from Dataset"], horizontal=True)
    input_image = None
    sample_path = None

    if input_mode == "Upload Image File":
        uploaded_file = st.file_uploader("Upload a cropped WBC microscopy image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
    else:
        sample_files = glob.glob(str(DATA_ROOT / "raw" / "TestA" / "TestA" / "*" / "*.jpg"))
        if sample_files:
            sample_options = {f"{os.path.basename(os.path.dirname(p))} - {os.path.basename(p)}": p for p in sample_files[:40]}
            choice = st.selectbox("Pick a benchmark test sample:", list(sample_options.keys()))
            sample_path = sample_options[choice]
            input_image = Image.open(sample_path).convert("RGB")
        else:
            st.warning("No sample images found under data/raw/TestA/TestA.")

    if input_image is not None:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.markdown("#### Input Microscopy Image")
            st.image(input_image, caption="Uploaded WBC Smear", use_container_width=True)

        # Quality safeguard inspection
        quality = engine.validate_image_quality(input_image)
        if quality["status"] == "FAILED":
            st.error(f"❌ Image Quality Safeguard Failed: {', '.join(quality['warnings'])}")
            st.stop()
        elif quality["status"] == "WARNING":
            st.warning(f"⚠️ Quality Notice: {', '.join(quality['warnings'])}")

        # User options for Grad-CAM
        with col_pred:
            st.markdown("#### Inference & Attribution Controls")
            explain_target = st.selectbox("Explain Attribution for Class:", ["(Top Model Prediction)"] + CLASS_NAMES)
            target_class_arg = None if explain_target.startswith("(") else explain_target

            alpha_slider = st.slider("Grad-CAM Overlay Opacity (Alpha):", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
            cmap_choice = st.selectbox("Attribution Colormap:", ["jet", "turbo", "viridis"])

            with st.spinner("Executing inference and calculating gradients..."):
                t0 = time.perf_counter()
                res = engine.predict_single(input_image, target_explain_class=target_class_arg)
                latency_ms = (time.perf_counter() - t0) * 1000.0

            predicted_class = res["predicted_class"]
            conf = res["confidence"]
            scores = res["score_breakdown"]

            margin = res["confidence_margin"]
            st.markdown(
                f'<div class="prediction-card"><div class="eyebrow">Model result</div>'
                f'<h2 style="margin:.2rem 0;color:#172033">{predicted_class}</h2>'
                f'<p style="margin:0;color:#667085">Top score <b>{conf*100:.2f}%</b> · '
                f'lead over {res["runner_up_class"]}: <b>{margin*100:.2f} pts</b></p></div>',
                unsafe_allow_html=True
            )
            if margin < 0.10:
                st.warning("Close call: the top two classes are near each other. Review the heatmap and morphology guide.")
            st.caption(f"Inference latency: **{latency_ms:.1f} ms** on CPU | Model: `{engine.model_name}`")

            # Probability Distribution Bar Chart
            score_df = pd.DataFrame({
                "WBC Class": list(scores.keys()),
                "Score": list(scores.values())
            }).sort_values(by="Score", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["#1f77b4" if c != predicted_class else "#2ca02c" for c in score_df["WBC Class"]]
            ax.barh(score_df["WBC Class"], score_df["Score"], color=colors)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Model Softmax Score (0.0 to 1.0)", fontsize=9)
            for i, v in enumerate(score_df["Score"]):
                ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontsize=9, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🧩 Grad-CAM Explainability Visualization")

        # Re-generate overlay with interactive user slider & cmap
        custom_colored_hm, custom_overlay = create_gradcam_overlay(
            res["original_image_rgb"], res["heatmap"], alpha=alpha_slider, colormap_name=cmap_choice
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(res["original_image_rgb"], caption="Preprocessed Cell (128x128)", use_container_width=True)
        with c2:
            st.image(custom_colored_hm, caption=f"Grad-CAM Heatmap ({res['explained_class']})", use_container_width=True)
        with c3:
            st.image(custom_overlay, caption=f"Blended Overlay (Alpha={alpha_slider})", use_container_width=True)

        st.markdown("#### 📖 Diagnostic Narrative & Biological Relevance")
        st.info(res["explanation_narrative"])
        with st.expander("🔬 Morphological Landmark Guide for this Cell Type"):
            st.write(res["morphological_reference"])


# =====================================================================
# PAGE C: EXPLAINABILITY & ATTENTION (SANITY CHECKS)
# =====================================================================
elif page == "🧩 Explainability & Attention":
    st.markdown("## 🧩 Explainability Deep-Dive & XAI Sanity Checks")
    render_disclaimer()

    if engine is None:
        st.error("No trained checkpoint available.")
        st.stop()

    st.markdown("""
    A critical limitation of saliency methods (such as vanilla gradients or improperly implemented Grad-CAM) 
    is that they can act as edge detectors independent of model weights or labels. 
    Following **Adebayo et al. (NeurIPS 2018) 'Sanity Checks for Saliency Maps'**, this module performs formal verification:
    """)

    sample_files = glob.glob(str(DATA_ROOT / "raw" / "TestA" / "TestA" / "*" / "*.jpg"))
    if not sample_files:
        st.warning("No test images available.")
        st.stop()

    test_img_path = sample_files[0]
    pil_img = Image.open(test_img_path).convert("RGB")

    col_test, col_btn = st.columns([1, 2])
    with col_test:
        st.image(pil_img, caption="Test Smear Sample", width=200)
    with col_btn:
        st.write("Click below to run real-time formal sanity checks on the active model:")
        run_checks = st.button("🚀 Run Saliency Sanity Checks")

    if run_checks:
        with st.spinner("Running Model Weight Randomization and Perturbation Tests..."):
            tensor = engine.transform(pil_img).unsqueeze(0).to(engine.device)
            res_cam = engine.predict_single(pil_img)

            # 1. Randomization test
            rand_res = model_randomization_sanity_check(
                engine.model, tensor, engine.target_layer, target_class=0
            )

            # 2. Fidelity perturbation test
            fid_res = region_perturbation_fidelity_check(
                engine.model, tensor, res_cam["heatmap"], target_class=0, top_k_percent=0.20
            )

        st.markdown("### 1. Model Parameter Randomization Check (Adebayo et al., 2018)")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Pearson Correlation with Random Weights", f"{rand_res['pearson_correlation']:.4f}")
        with col_m2:
            st.metric("Mean Absolute Difference", f"{rand_res['mean_absolute_difference']:.4f}")
        with col_m3:
            status = "PASSED ✅" if rand_res["passed_sanity_check"] else "INCONCLUSIVE ⚠️"
            st.metric("Sanity Check Status", status)
        st.write(f"**Finding:** {rand_res['conclusion']}")

        st.markdown("### 2. High-Attribution Region Perturbation (Fidelity Test)")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Original Confidence", f"{fid_res['original_probability']*100:.1f}%")
        with col_f2:
            st.metric("Confidence After Masking Grad-CAM Focus", f"{fid_res['probability_after_high_attr_mask']*100:.1f}%",
                      f"-{fid_res['high_attr_drop']*100:.1f}%", delta_color="inverse")
        with col_f3:
            st.metric("Confidence After Random Masking", f"{fid_res['probability_after_random_mask']*100:.1f}%",
                      f"-{fid_res['random_attr_drop']*100:.1f}%", delta_color="inverse")
        st.write(f"**Finding:** {fid_res['conclusion']}")


# =====================================================================
# PAGE D: BATCH SCREENING
# =====================================================================
elif page == "📁 Batch Screening":
    st.markdown("## 📁 Batch Screening & Automated CSV Export")
    render_disclaimer()

    if engine is None:
        st.error("No model available.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload multiple leukocyte images (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Loaded **{len(uploaded_files)}** images for batch inference.")
        if st.button("⚡ Run Batch Classification"):
            progress_bar = st.progress(0)
            rows = []
            
            for idx, file in enumerate(uploaded_files):
                try:
                    img = Image.open(file).convert("RGB")
                    pred_res = engine.predict_single(img)
                    rows.append({
                        "Filename": file.name,
                        "Predicted Class": pred_res["predicted_class"],
                        "Confidence": pred_res["confidence"],
                        "Confidence Margin": pred_res["confidence_margin"],
                        "Runner-up": pred_res["runner_up_class"],
                        "Quality Status": pred_res["quality_report"]["status"],
                        "Basophil Score": pred_res["score_breakdown"]["Basophil"],
                        "Eosinophil Score": pred_res["score_breakdown"]["Eosinophil"],
                        "Lymphocyte Score": pred_res["score_breakdown"]["Lymphocyte"],
                        "Monocyte Score": pred_res["score_breakdown"]["Monocyte"],
                        "Neutrophil Score": pred_res["score_breakdown"]["Neutrophil"]
                    })
                except Exception as e:
                    rows.append({
                        "Filename": file.name,
                        "Predicted Class": "ERROR",
                        "Confidence": 0.0,
                        "Quality Status": f"Corrupted ({str(e)})",
                        "Basophil Score": 0.0,
                        "Eosinophil Score": 0.0,
                        "Lymphocyte Score": 0.0,
                        "Monocyte Score": 0.0,
                        "Neutrophil Score": 0.0
                    })
                progress_bar.progress((idx + 1) / len(uploaded_files))

            batch_df = pd.DataFrame(rows)
            st.dataframe(batch_df, use_container_width=True)

            # Export to CSV
            csv_buf = io.StringIO()
            batch_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Batch Results as CSV",
                data=csv_buf.getvalue(),
                file_name="wbc_batch_classification_results.csv",
                mime="text/csv"
            )


# =====================================================================
# PAGE E: BENCHMARK EVALUATION & ROC CURVES
# =====================================================================
elif page == "📊 Benchmark Evaluation":
    st.markdown("## 📊 Model Evaluation Dashboard (Held-Out TestA Benchmark)")
    render_disclaimer()

    # Look for metrics reports in results/reports/
    metric_files = glob.glob(str(RESULTS_ROOT / "reports" / "*_test_metrics.json"))
    if not metric_files:
        st.warning("No test evaluation reports found. Run `python src/evaluate.py` to evaluate on held-out test data.")
    else:
        reports = {}
        for mf in metric_files:
            with open(mf, "r") as f:
                data = json.load(f)
                reports[data["model_name"]] = data

        st.markdown("### 📈 Overall Performance Comparison")
        comp_rows = []
        for name, rep in reports.items():
            comp_rows.append({
                "Architecture": name,
                "Accuracy (%)": f"{rep['accuracy']*100:.2f}%",
                "Macro F1-Score": f"{rep['macro_f1']:.4f}",
                "Macro Precision": f"{rep['macro_precision']:.4f}",
                "Macro Recall": f"{rep['macro_recall']:.4f}",
                "Macro OvR ROC-AUC": f"{rep['macro_roc_auc']:.4f}" if rep.get("macro_roc_auc") else "N/A",
                "CPU Latency (ms)": f"{rep['average_cpu_latency_ms']:.2f} ms",
                "Test Samples": rep["total_test_samples"]
            })
        st.table(pd.DataFrame(comp_rows))

        # Show per-class metrics
        selected_model_report = st.selectbox("View per-class breakdown for:", list(reports.keys()))
        if selected_model_report:
            per_class_data = reports[selected_model_report]["per_class"]
            per_class_df = pd.DataFrame.from_dict(per_class_data, orient="index")
            st.dataframe(per_class_df, use_container_width=True)

        # Display Confusion Matrix and ROC Curve figures
        col1, col2 = st.columns(2)
        cm_path = str(RESULTS_ROOT / "figures" / f"{selected_model_report}_confusion_matrix.png")
        roc_path = str(RESULTS_ROOT / "figures" / f"{selected_model_report}_roc_curves.png")

        with col1:
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"Confusion Matrix ({selected_model_report})", use_container_width=True)
        with col2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption=f"ROC Curves ({selected_model_report})", use_container_width=True)

        curve_path = str(RESULTS_ROOT / "figures" / f"{selected_model_report}_training_curves.png")
        if os.path.exists(curve_path):
            st.markdown("### 📉 Training & Validation Learning Curves")
            st.image(curve_path, use_container_width=True)


# =====================================================================
# PAGE F: ERROR ANALYSIS & MISCLASSIFICATIONS
# =====================================================================
elif page == "🔍 Error & Misclassification Analysis":
    st.markdown("## 🔍 Error Analysis & Misclassified Samples Explorer")
    render_disclaimer()

    misc_csvs = glob.glob(str(RESULTS_ROOT / "reports" / "*_misclassified_samples.csv"))
    if not misc_csvs:
        st.warning("No misclassification logs found. Run evaluation to detect error cases.")
    else:
        chosen_csv = st.selectbox("Select Model Error Log:", misc_csvs)
        misc_df = pd.read_csv(chosen_csv)
        st.markdown(f"Total Misclassifications Recorded: **{len(misc_df)}**")

        st.dataframe(misc_df.head(25), use_container_width=True)

        gallery_img = chosen_csv.replace("reports", "figures").replace("_misclassified_samples.csv", "_misclassifications.png")
        if os.path.exists(gallery_img):
            st.markdown("### 🖼️ Misclassification Visual Gallery")
            st.image(gallery_img, use_container_width=True)

        st.markdown("### 💡 Common Clinical & Visual Causes of Leukocyte Misclassification")
        st.markdown("""
        1. **Band vs. Segmented Neutrophils**: Immature band forms lack distinct nuclear filaments, frequently confusing models with monocytes or large lymphocytes.
        2. **Stain Color Shifts**: Excessive methylene blue staining can make lymphocyte cytoplasm appear intensely dark or make monocyte vacuoles less prominent.
        3. **Cell Rupture & Smudge Cells**: During blood smear preparation, mechanically fragile cells rupture, creating distorted nuclear borders without intact cytoplasmic membranes.
        4. **Class Imbalance Artifacts**: Because neutrophils outnumber basophils 30:1 in peripheral blood, borderline cases naturally attract higher prior probabilities without aggressive weighting.
        """)


# =====================================================================
# PAGE G: METHODOLOGY, ETHICS & VIVA GUIDE
# =====================================================================
elif page == "📜 Methodology & Viva Guide":
    st.markdown("## 📜 Project Methodology, Ethics, and Viva Voce Guide")
    render_disclaimer()

    st.markdown("""
    ### 🎓 University Final-Year Project Defense Reference

    #### 1. Why Grad-CAM over other Explainability Methods (e.g. Saliency Maps or LIME)?
    - **Vanilla Saliency Maps**: Suffer from gradient saturation, high noise, and often pass model randomization tests trivially because they act as edge detectors (Adebayo et al., 2018).
    - **LIME**: Extremely slow on CPU because it requires generating hundreds of random perturbations per image; highly sensitive to superpixel segmentation boundaries.
    - **Grad-CAM**: Leverages the final convolutional layer's spatial representations; combines gradient weights with feature activations; fast ($O(1)$ backward pass); class-discriminative; well-suited for interactive deployment.

    #### 2. How is Class Imbalance Handled?
    - Peripheral blood naturally has high neutrophil counts (50-70%) and low basophil counts (0.5-2%).
    - In our pipeline, we compute **balanced class weights**:
      $$\\text{weight}_c = \\frac{N}{K \\times N_c}$$
      This penalizes errors on rare Basophils and Eosinophils proportionally higher, boosting Macro-F1 score without artificially duplicating images.

    #### 3. Why Temperature Scaling for Calibration?
    - Modern neural networks with Batch Normalization and Cross-Entropy loss are often overconfident.
    - **Temperature Scaling** adds a single scalar parameter $T > 0$ to soften logits: $\\hat{p}_i = \\text{softmax}(z_i / T)$.
    - Crucially, $T$ is tuned **strictly on the validation set** using Negative Log-Likelihood, keeping the test set 100% held-out.

    #### 4. Viva Voce Quick-Fire Q&A
    - **Q: Does Grad-CAM prove biological causation?**
      *A: No. Grad-CAM identifies image regions with high positive gradient correlation with the target logit. It is an algorithmic localization tool, not biological proof.*
    - **Q: Why not train on 575x575 images?**
      *A: Memory and CPU constraints. 128x128 provides an optimal trade-off between retaining chromatin texture and completing training/inference in milliseconds on a quad-core Intel CPU without CUDA.*
    - **Q: What is the primary data leakage risk in medical imaging?**
      *A: Patient/slide leakage. In Raabin-WBC, Train and TestA share 26 slide IDs. While this reflects the official benchmark partition, in clinical deployment patient-grouped splitting is strictly required to prevent optimistic bias.*
    """)
