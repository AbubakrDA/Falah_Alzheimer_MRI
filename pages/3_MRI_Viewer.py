import sys
import importlib
from pathlib import Path
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

from src.data.falah_loader import FalahDatasetLoader
from src.data.nifti_analyze_parser import MedicalImageReader
from src.config.settings import OASIS1_EXTRACTED_DIR, DISCLAIMER_TEXT

st.set_page_config(page_title="MRI Viewer - NeuroVision", page_icon="👁️", layout="wide")

st.title("👁️ Interactive MRI Exploration Viewer")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

viewer_mode = st.radio("Cohort Source", ["Falah 2D Axial Slices", "OASIS-1 3D Volumetric Exploration"], horizontal=True)

if viewer_mode == "Falah 2D Axial Slices":
    loader = FalahDatasetLoader()
    split = st.selectbox("Split", ["train", "val", "test"])
    
    # Filter by class
    classes = loader.class_names
    selected_class = st.selectbox("Filter by Category", ["All"] + classes)
    
    samples = loader.get_sample_images(split=split, count=32, class_filter=selected_class)
        
    if not samples:
        st.info("No matching samples found for selected filter.")
    else:
        sample_idx = st.slider("Select Slice Sample", 0, len(samples) - 1, 0)
        curr = samples[sample_idx]
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(curr["image"], caption=f"Sample {curr['index']} - {curr['class_name']}", use_container_width=True)
            
        with c2:
            st.markdown("### Slice Properties & Radiometric Profile")
            img_arr = np.array(curr["image"])
            st.write(f"- **Category:** `{curr['class_name']}`")
            st.write(f"- **Dimensions:** `{img_arr.shape}`")
            st.write(f"- **Data Type:** `{img_arr.dtype}`")
            st.write(f"- **Min Pixel Intensity:** `{img_arr.min()}`")
            st.write(f"- **Max Pixel Intensity:** `{img_arr.max()}`")
            st.write(f"- **Mean Intensity:** `{img_arr.mean():.2f}`")
            st.write(f"- **Standard Deviation:** `{img_arr.std():.2f}`")
            
            # Histogram
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(img_arr.ravel(), bins=32, color="teal", alpha=0.7)
            ax.set_title("Voxel Intensity Distribution")
            ax.set_xlabel("Pixel Value [0-255]")
            ax.set_ylabel("Voxel Count")
            st.pyplot(fig)

else: # OASIS-1 3D Volumetric
    st.subheader("OASIS-1 3D Multi-Planar Orthogonal Slicer")
    extracted_path = Path(OASIS1_EXTRACTED_DIR)
    
    vol_files = list(extracted_path.rglob("*.nii")) + list(extracted_path.rglob("*.nii.gz")) + list(extracted_path.rglob("*.hdr"))
    
    if not vol_files:
        st.info("No 3D MRI volumes (.nii, .nii.gz, .hdr/.img) currently found in `data/OASIS-1/extracted/`.")
        st.caption("Once you extract OASIS-1 data in **Dataset Management**, volumes will appear here for multi-planar slicing.")
    else:
        selected_file = st.selectbox("Select 3D Volume", vol_files, format_func=lambda p: p.name)
        with st.spinner("Loading 3D MRI volume header and matrix..."):
            vol, hdr = MedicalImageReader.load_volume(selected_file)
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Volume Matrix Shape", str(vol.shape))
        c2.metric("Voxel Spacing (mm)", str(hdr.get("zooms", ())))
        c3.metric("Storage Datatype", hdr.get("dtype", "unknown"))
        
        col_view1, col_view2 = st.columns([2, 1])
        with col_view2:
            plane = st.radio("Anatomical Plane", ["axial", "coronal", "sagittal"])
            max_slice = vol.shape[2] - 1 if plane == "axial" else (vol.shape[1] - 1 if plane == "coronal" else vol.shape[0] - 1)
            slice_num = st.slider("Slice Index", 0, max_slice, max_slice // 2)
            colormap = st.selectbox("Colormap", ["gray", "bone", "viridis", "plasma"])
            
        with col_view1:
            slc = MedicalImageReader.extract_slice(vol, plane=plane, slice_idx=slice_num)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(slc, cmap=colormap)
            ax.set_title(f"{plane.capitalize()} View - Slice {slice_num}/{max_slice}")
            ax.axis("off")
            st.pyplot(fig)
