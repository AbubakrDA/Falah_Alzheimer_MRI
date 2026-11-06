import sys
import importlib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Refresh any stale src.data or src.config modules cached in the running Streamlit process
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src.data") or mod_name.startswith("src.config"):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import DEFAULT_IMAGE_SIZE, DISCLAIMER_TEXT
from src.data.falah_loader import FalahDatasetLoader

st.set_page_config(page_title="Image Preprocessing - NeuroVision", page_icon="🔬", layout="wide")

st.title("🔬 Biomedical Image Preprocessing Pipeline")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

st.markdown("""
Configures radiometric normalization, dimension resizing, and data augmentation.
Preprocessed representations feed into the Custom CNN, ResNet50, and Vision Transformer models.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Pipeline Configuration")
    target_dim = st.selectbox("Standardized Resolution", [128, 224, 256], index=0)
    target_size = (target_dim, target_dim)
    
    channel_mode = st.radio("Channel Configuration", ["Grayscale (1 channel)", "RGB Projection (3 channels)"])
    channels = 1 if "1" in channel_mode else 3
    
    norm_method = st.selectbox("Intensity Normalization", ["Min-Max Scaling [0.0, 1.0]", "Z-Score Standardization (zero mean, unit variance)"])
    
    st.markdown("#### Augmentation (Training Partition Only)")
    apply_aug = st.checkbox("Enable Stochastic Data Augmentation", value=True)
    rot_range = st.slider("Random Rotation (+/- degrees)", 0, 20, 5) if apply_aug else 0
    zoom_range = st.slider("Random Zoom (+/- %)", 0, 20, 5) if apply_aug else 0

with col2:
    st.subheader("2. Interactive Preprocessing Preview")
    loader = FalahDatasetLoader()
    sample_img = loader.raw_dataset["train"][0]["image"]
    
    st.markdown("##### Original Raw Slice")
    st.image(sample_img, caption=f"Original Size: {sample_img.size}, Mode: {sample_img.mode}", width=250)
    
    # Process
    img_conv = sample_img.convert("RGB" if channels == 3 else "L")
    img_resized = img_conv.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    
    if norm_method.startswith("Min-Max"):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
    else:
        arr = (arr - arr.mean()) / (arr.std() + 1e-7)
        # Scale to [0, 1] for visual display
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
        
    st.markdown("##### Preprocessed Output")
    st.image(arr, caption=f"Normalized Matrix: {arr.shape}, Range: [{arr.min():.2f}, {arr.max():.2f}]", width=250)
