import sys
import importlib
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

# Hot-reload any updated src modules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src."):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import MODELS_DIR, DISCLAIMER_TEXT
from src.predict import NeuroVisionPredictor
from src.data.dataset_manager import DatasetManager

st.set_page_config(page_title="Prediction - NeuroVision", page_icon="🎯", layout="wide")

st.title("🎯 Dedicated Research Inference & Staging")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

# Discover both PyTorch (.pt, .pth), Keras (.keras, .h5), and Simulation (.json) checkpoints
available_checkpoints = []
if Path(MODELS_DIR).exists():
    available_checkpoints.extend(list(Path(MODELS_DIR).glob("**/*.pt")))
    available_checkpoints.extend(list(Path(MODELS_DIR).glob("**/*.pth")))
    available_checkpoints.extend(list(Path(MODELS_DIR).glob("**/*.keras")))
    available_checkpoints.extend(list(Path(MODELS_DIR).glob("**/*.h5")))
    available_checkpoints.extend(list(Path(MODELS_DIR).glob("**/*.json")))

available_checkpoints = sorted(list(set(available_checkpoints)))

c_side, c_main = st.columns([1, 2])

with c_side:
    st.subheader("1. Model Selection")
    if not available_checkpoints:
        st.warning("No model checkpoints found in `models/`. Train an architecture first on the **Model Training** page.")
        selected_checkpoint = None
    else:
        def format_ckpt(p: Path) -> str:
            if p.suffix in [".pt", ".pth"]:
                fw = "PyTorch"
            elif p.suffix == ".json":
                fw = "Simulation"
            else:
                fw = "Keras"
            return f"[{fw}] {p.parent.name}/{p.name}"

        selected_checkpoint = st.selectbox(
            "Select Checkpoint",
            available_checkpoints,
            format_func=format_ckpt
        )

    explain_toggle = st.checkbox("Generate Attribution Overlay", value=True)
    
    st.subheader("2. Scan Source")
    cohort_source = st.selectbox("Cohort Source", ["Falah_Alzheimer_MRI", "OASIS-1"], index=0)
    input_source = st.radio("Choose Input Type", ["Upload Custom File", "Pick from Test Cohort"])

with c_main:
    st.subheader("3. Image Input & Analysis")
    target_img = None
    true_label = None
    
    if input_source == "Upload Custom File":
        uploaded = st.file_uploader("Upload Brain MRI (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded:
            target_img = Image.open(uploaded).convert("RGB")
    else:
        mgr = DatasetManager(default_dataset=cohort_source)
        sample_list = mgr.get_sample_images("test", count=16)
        if sample_list:
            chosen_sample_idx = st.selectbox(
                "Sample Index",
                range(len(sample_list)),
                format_func=lambda i: f"Sample #{i} ({sample_list[i]['class_name']})"
            )
            target_img = sample_list[chosen_sample_idx]["image"].convert("RGB")
            true_label = sample_list[chosen_sample_idx]["class_name"]
        else:
            st.info(f"No samples currently indexed for {cohort_source} test split.")

    if target_img is not None:
        col_img, col_pred = st.columns(2)
        with col_img:
            caption_text = f"Input Brain MRI Slice (Ground Truth: {true_label})" if true_label else "Input Brain MRI Slice"
            st.image(target_img, caption=caption_text, use_container_width=True)

        with col_pred:
            if selected_checkpoint:
                with st.spinner("Running deep learning model inference..."):
                    predictor = NeuroVisionPredictor(checkpoint_path=selected_checkpoint)
                    res = predictor.predict(target_img, explain=explain_toggle)

                pred_class = res["predicted_class"]
                conf = res["confidence"]

                # Badge
                if "Non" in pred_class:
                    st.success(f"### Predicted Stage: **{pred_class.replace('_', ' ')}** ({conf:.1f}%)")
                elif "Very" in pred_class:
                    st.info(f"### Predicted Stage: **{pred_class.replace('_', ' ')}** ({conf:.1f}%)")
                elif "Mild" in pred_class:
                    st.warning(f"### Predicted Stage: **{pred_class.replace('_', ' ')}** ({conf:.1f}%)")
                else:
                    st.error(f"### Predicted Stage: **{pred_class.replace('_', ' ')}** ({conf:.1f}%)")

                st.markdown("##### Stage Probability Spectrum")
                for cname, p in res["probabilities"].items():
                    st.progress(p / 100.0, text=f"{cname.replace('_', ' ')}: {p:.1f}%")

                if explain_toggle:
                    if "gradcam_overlay" in res:
                        st.markdown(f"##### 🧠 Anatomical Attention ({res.get('method', 'Grad-CAM')})")
                        st.image(res["gradcam_overlay"], use_container_width=True)
                    elif "attention_overlay" in res:
                        st.markdown(f"##### 🧠 Transformer Attention Attribution ({res.get('method', 'ViT Rollout')})")
                        st.image(res["attention_overlay"], use_container_width=True)
                    elif "explainability_error" in res:
                        st.caption(f"Attribution note: {res['explainability_error']}")
            else:
                st.info("Select a trained checkpoint to execute inference.")
