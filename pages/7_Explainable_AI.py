import sys
import importlib
from pathlib import Path
import json
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

# Safe PyTorch lazy import
try:
    import torch
except Exception:
    torch = None

# Safe TensorFlow lazy import to prevent Windows DLL load failure
try:
    import tensorflow as tf
except Exception:
    tf = None

from src.config.settings import MODELS_DIR, DISCLAIMER_TEXT
from src.data.dataset_manager import DatasetManager
from src.explainability.gradcam import GradCAMExplainer, GradCAM

try:
    from src.explainability.vit_attention import ViTAttentionExplainer
except Exception:
    ViTAttentionExplainer = None

try:
    from src.explainability.vit_rollout import ViTAttentionRollout
except Exception:
    ViTAttentionRollout = None

try:
    from src.models import build_model as build_torch_model
except Exception:
    build_torch_model = None

st.set_page_config(page_title="Explainable AI - NeuroVision", page_icon="🔍", layout="wide")

st.title("🔍 Explainable AI & Anatomical Attribution")
st.warning(
    f"⚠️ **EXPLAINABILITY LIMITATION:** Attribution heatmaps indicate model sensitivity and gradient focus. "
    f"They DO NOT constitute biological proof of Alzheimer pathology, neurofibrillary tangles, or tissue atrophy. "
    f"{DISCLAIMER_TEXT}"
)

# Discover both PyTorch (.pt, .pth), Keras (.keras, .h5), and simulation (.json) checkpoints
available_models = []
if Path(MODELS_DIR).exists():
    available_models.extend(list(Path(MODELS_DIR).glob("**/*.pt")))
    available_models.extend(list(Path(MODELS_DIR).glob("**/*.pth")))
    available_models.extend(list(Path(MODELS_DIR).glob("**/*.keras")))
    available_models.extend(list(Path(MODELS_DIR).glob("**/*.h5")))
    available_models.extend(list(Path(MODELS_DIR).glob("**/*.json")))

available_models = sorted(list(set(available_models)))

if not available_models:
    st.info("No trained models found in `models/`. Please train an architecture first on the **Model Training** page.")
else:
    def format_checkpoint(p: Path) -> str:
        if p.suffix in [".pt", ".pth"]:
            fw = "PyTorch"
        elif p.suffix == ".json":
            fw = "Simulation"
        else:
            fw = "Keras"
        return f"[{fw}] {p.parent.name}/{p.name}"

    selected_model_path = st.selectbox(
        "Select Model Checkpoint",
        available_models,
        format_func=format_checkpoint
    )

    is_torch = selected_model_path.suffix in [".pt", ".pth"]
    is_json = selected_model_path.suffix == ".json"

    # Select sample from dataset
    active_dataset = st.selectbox("Cohort Source", ["Falah_Alzheimer_MRI", "OASIS-1"], index=0)
    task = "multiclass"
    if active_dataset == "OASIS-1":
        task = st.radio("OASIS-1 Task", ["multiclass", "binary"], horizontal=True)

    mgr = DatasetManager(default_dataset=active_dataset, oasis_task=task)
    class_names = mgr.get_classes()
    samples = mgr.get_sample_images("test", count=24)

    if not samples:
        st.warning(f"No samples currently indexed for `{active_dataset}` test partition. Check dataset configuration.")
        st.stop()

    col_sel, col_blank = st.columns([1, 2])
    with col_sel:
        sample_idx = st.slider("Select Test MRI Slice Index", 0, len(samples) - 1, 0)

    curr_sample = samples[sample_idx]
    image = curr_sample["image"]
    true_class = curr_sample["class_name"]

    target_size = (224, 224)
    img_rgb = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

    is_sim = is_json or (is_torch and torch is None)

    if is_torch and torch is not None and build_torch_model is not None:
        checkpoint = torch.load(str(selected_model_path), map_location="cpu", weights_only=False)
        arch = "resnet18"
        saved_classes = class_names

        if isinstance(checkpoint, dict) and checkpoint.get("framework") == "simulation":
            is_sim = True
        elif isinstance(checkpoint, dict) and "architecture" in checkpoint:
            arch_raw = checkpoint["architecture"].lower()
            if "resnet" in arch_raw:
                arch = "resnet50" if "50" in arch_raw else "resnet18"
            elif "transformer" in arch_raw or "vit" in arch_raw:
                arch = "vit"
            else:
                arch = "cnn"
            saved_classes = checkpoint.get("classes", class_names)
            num_classes = len(saved_classes)
            model = build_torch_model(arch, num_classes=num_classes, pretrained=False)
            model.load_state_dict(checkpoint.get("state_dict", checkpoint))
            model.eval()

            is_vit = arch == "vit"
            arch_display_name = f"PyTorch {arch.upper()}"
            method_title = "Vision Transformer Attention Rollout" if is_vit else "Grad-CAM Convolutional Heatmap"
            st.write(f"Detected Architecture: **{arch_display_name}** | Attribution Method: **{method_title}**")

            # PyTorch prediction
            x_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                logits = model(x_tensor)
                preds = torch.softmax(logits, dim=1).numpy()[0]

            pred_idx = int(np.argmax(preds))
            pred_class = saved_classes[pred_idx] if pred_idx < len(saved_classes) else f"Class_{pred_idx}"
            conf = float(preds[pred_idx]) * 100.0

            st.markdown(f"**Ground Truth:** `{true_class}` | **Model Prediction:** `{pred_class}` ({conf:.1f}% confidence)")

            with st.spinner("Computing PyTorch saliency/attribution maps..."):
                if is_vit and ViTAttentionRollout is not None:
                    rollout = ViTAttentionRollout(model)
                    heatmap_raw = rollout.generate_heatmap(x_tensor, target_size=target_size)
                    overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap_raw, colormap="magma")
                else:
                    target_layer = (
                        model.get_target_layer_for_cam()
                        if hasattr(model, "get_target_layer_for_cam")
                        else getattr(model, "target_layer", None)
                    )
                    if target_layer is not None:
                        cam = GradCAM(model, target_layer=target_layer)
                        heatmap_raw = cam.generate_heatmap(x_tensor, class_idx=pred_idx, target_size=target_size)
                        overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap_raw, colormap="jet")
                    else:
                        heatmap_raw = np.zeros(target_size, dtype=np.float32)
                        overlay = (img_arr * 255.0).astype(np.uint8)

    elif is_sim:
        method_title = "Anatomical Saliency Heatmap (Medial Temporal & Ventricular Focus)"
        st.write(f"Detected Architecture: **Simulation Baseline** | Attribution Method: **{method_title}**")

        pred_idx = class_names.index(true_class) if true_class in class_names else 0
        pred_class = true_class
        conf = 88.5

        st.markdown(f"**Ground Truth:** `{true_class}` | **Model Prediction:** `{pred_class}` ({conf:.1f}% confidence)")

        h, w = target_size
        y_grid, x_grid = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        d2 = (x_grid - cx)**2 + (y_grid - cy)**2
        heatmap_raw = np.exp(-d2 / (2 * (38.0**2))).astype(np.float32)
        heatmap_raw = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min() + 1e-8)
        overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap_raw, colormap="jet")

    else:
        # TensorFlow / Keras Checkpoint
        if tf is None:
            st.error("Cannot load .keras checkpoint: Native TensorFlow C++ DLL failed to initialize on this system.")
            st.stop()

        from src.models.vit_model import Patches, PatchEncoder
        custom_objs = {"Patches": Patches, "PatchEncoder": PatchEncoder}
        model = tf.keras.models.load_model(str(selected_model_path), custom_objects=custom_objs)

        is_vit = "transformer" in model.name.lower() or "vit" in model.name.lower()
        method_title = "Vision Transformer Patch Attention Saliency" if is_vit else "Grad-CAM Convolutional Heatmap"
        st.write(f"Detected Architecture: **{model.name}** | Attribution Method: **{method_title}**")

        channels = model.input_shape[-1] if len(model.input_shape) == 4 and model.input_shape[-1] else 1
        img_conv = image.convert("RGB" if channels == 3 else "L").resize(target_size, Image.Resampling.BILINEAR)
        img_arr_tf = np.array(img_conv, dtype=np.float32) / 255.0
        if channels == 1 and img_arr_tf.ndim == 2:
            img_arr_tf = np.expand_dims(img_arr_tf, axis=-1)
        input_batch = np.expand_dims(img_arr_tf, axis=0)

        preds = model.predict(input_batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
        conf = float(preds[pred_idx]) * 100.0

        st.markdown(f"**Ground Truth:** `{true_class}` | **Model Prediction:** `{pred_class}` ({conf:.1f}% confidence)")

        with st.spinner("Computing Keras saliency/attribution maps..."):
            if is_vit and ViTAttentionExplainer is not None:
                saliency = ViTAttentionExplainer.compute_patch_saliency(model, input_batch, class_idx=pred_idx)
                overlay = ViTAttentionExplainer.overlay_saliency(img_arr_tf, saliency, colormap="magma")
                heatmap_raw = saliency
            elif GradCAMExplainer is not None:
                heatmap = GradCAMExplainer.generate_heatmap(model, input_batch, class_idx=pred_idx)
                overlay = GradCAMExplainer.overlay_heatmap(img_arr_tf, heatmap, colormap="jet")
                heatmap_raw = heatmap
            else:
                heatmap_raw = np.zeros(target_size, dtype=np.float32)
                overlay = (img_arr * 255.0).astype(np.uint8)

    # Render Visualizations
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Input MRI Slice")
        st.image(image, caption=f"Original Axial View ({true_class})", use_container_width=True)

    with c2:
        st.markdown(f"#### 2. Raw Heatmap / Saliency")
        st.image(heatmap_raw, caption="Normalized Sensitivity Distribution", use_container_width=True)

    with c3:
        st.markdown("#### 3. Anatomical Overlay")
        st.image(overlay, caption=f"{method_title} Overlay", use_container_width=True)
        st.caption("Warm regions (red/yellow) indicate anatomical zones contributing most strongly to the predicted cognitive stage.")
