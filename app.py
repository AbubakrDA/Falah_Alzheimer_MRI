"""NeuroVision 2.0 - Clinical AI Assistant & Dataset Management Platform

Features:
1. Unified Dataset Management:
   - Falah Alzheimer MRI (Hugging Face Arrow 2D slice inspection, class distributions, slice leakage audit)
   - OASIS-1 (NIfTI & Analyze 7.5 ingestion, CSV/Excel metadata matching, CDR mapping, participant-level splitting, manifest generation)
2. Clinical Diagnosis & Explainability:
   - Multi-class Alzheimer's disease staging
   - Probability distribution visualizer
   - Grad-CAM and ViT Attention Rollout anatomical overlays
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import sys
import importlib

# Refresh any stale src.data or src.config modules cached in the running Streamlit process
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src.data") or mod_name.startswith("src.config"):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

try:
    from src.config import CLASSES, MODELS_DIR, DEFAULT_BACKBONE
except ImportError:
    try:
        from src.config.settings import CLASSES, MODELS_DIR, DEFAULT_BACKBONE
    except ImportError:
        from src.config.settings import FALAH_CLASSES as CLASSES, MODELS_DIR
        DEFAULT_BACKBONE = "resnet18"
from src.predict import NeuroVisionPredictor
from src.data.falah_loader import FalahDatasetLoader
from src.data.oasis1_loader import OASIS1DatasetLoader
from src.data.oasis1_validator import OASIS1Validator
from src.data.oasis1_metadata import OASIS1MetadataManager
from src.data.oasis1_label_mapper import OASIS1LabelMapper, BINARY_CLASSES, MULTICLASS_CLASSES
from src.data.oasis1_manifest import OASIS1ManifestManager

st.set_page_config(
    page_title="NeuroVision 2.0 - Alzheimer's Diagnostic & Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 NeuroVision 2.0")
st.caption("Advanced Neuroimaging AI Architecture for Alzheimer's Staging & Multimodal Dataset Management")

# Navigation Mode
nav_mode = st.sidebar.radio(
    "Navigation Menu",
    options=["📁 Dataset Management", "🧠 Clinical Diagnosis & Inference", "🗺️ System Flowchart & Architecture"],
    index=0,
)

# ==============================================================================
# SECTION 1: DATASET MANAGEMENT
# ==============================================================================
if nav_mode == "📁 Dataset Management":
    st.header("Unified Dataset Management")
    st.write(
        "Manage, validate, and inspect local neuroimaging datasets. "
        "Select an active dataset below to configure loaders, inspect provenance, "
        "and audit clinical splits."
    )

    selected_dataset = st.selectbox(
        "Select Active Dataset:",
        options=["Falah Alzheimer MRI", "OASIS-1 (Open Access Series of Imaging Studies)"],
        index=0,
    )

    # --------------------------------------------------------------------------
    # DATASET A: FALAH ALZHEIMER MRI
    # --------------------------------------------------------------------------
    if selected_dataset == "Falah Alzheimer MRI":
        st.subheader("📊 Falah Alzheimer MRI Dataset")
        st.info(
            "**Dataset Status:** Locally Available & Verified (Hugging Face Arrow format). "
            "No download required."
        )

        try:
            falah_loader = FalahDatasetLoader()
            info = falah_loader.get_dataset_info()

            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total MRI Slices", f"{info['total_samples']:,}")
            m2.metric("Training Split (80%)", f"{info['train_samples']:,}")
            m3.metric("Testing Split (20%)", f"{info['test_samples']:,}")
            m4.metric("Native Resolution", f"{info['native_image_size'][0]}x{info['native_image_size'][1]}")

            tab_dist, tab_samples, tab_leakage, tab_report = st.tabs([
                "📈 Class Distribution",
                "🖼️ Image Samples",
                "⚠️ Patient Leakage Audit",
                "📄 Summary Report",
            ])

            with tab_dist:
                st.markdown("#### Class Distribution Breakdown")
                df_dist = pd.DataFrame([
                    {
                        "Class": c,
                        "Train": info["train_distribution"].get(c, 0),
                        "Test": info["test_distribution"].get(c, 0),
                        "Total": info["total_distribution"].get(c, 0),
                        "% Dataset": round((info["total_distribution"].get(c, 0) / info["total_samples"]) * 100, 2),
                    }
                    for c in info["class_names"]
                ])
                st.dataframe(df_dist, use_container_width=True)

                st.bar_chart(df_dist.set_index("Class")[["Train", "Test"]])

                st.warning(
                    "⚠️ **Severe Class Imbalance Detected:** `Moderate_Demented` accounts for only **1.0%** "
                    "(64 slices total). Training without inverse-frequency class weighting will cause sensitivity collapse."
                )

            with tab_samples:
                st.markdown("#### 🖼️ Representative Train & Test MRI Slice Gallery")
                gallery_view = st.radio(
                    "Select Gallery Mode:",
                    ["Multi-Class Progression (Train vs. Test)", "Deep Split Explorer"],
                    horizontal=True,
                )

                if gallery_view == "Multi-Class Progression (Train vs. Test)":
                    st.write(
                        "Comparative axial MRI slices across all 4 cognitive impairment stages from both the "
                        "**Training Split** (80% partition) and **Testing Split** (20% evaluation partition)."
                    )

                    # Display side-by-side comparison for Train and Test
                    for split_label, split_ds, badge_color in [
                        ("Training Split (5,120 Slices - 80%)", falah_loader.raw_train, "blue"),
                        ("Testing Split (1,280 Slices - 20%)", falah_loader.raw_test, "green"),
                    ]:
                        st.markdown(f"##### :{badge_color}[{split_label}]")
                        p_cols = st.columns(4)
                        for c_idx, c_name in enumerate(info["class_names"]):
                            # Find first representative slice for this class
                            sample_found = None
                            for idx in range(len(split_ds)):
                                if int(split_ds[idx]["label"]) == c_idx:
                                    sample_found = split_ds[idx]
                                    break

                            with p_cols[c_idx]:
                                if sample_found:
                                    img_pil = sample_found["image"]
                                    st.image(
                                        img_pil.resize((200, 200)),
                                        caption=f"{c_name.replace('_', ' ')} (Idx #{idx})",
                                        use_container_width=True,
                                    )
                                    st.caption(f"Native: {img_pil.size} • Mode: {img_pil.mode}")
                                else:
                                    st.info(f"No sample for {c_name}")

                else:
                    st.write("Browse consecutive slices within a specific partition and class category:")
                    col_split, col_cls, col_cnt = st.columns(3)
                    with col_split:
                        chosen_split = st.selectbox("Partition", ["train", "test"])
                    with col_cls:
                        chosen_class = st.selectbox("Class Filter", ["All"] + info["class_names"])
                    with col_cnt:
                        num_samples = st.slider("Sample Count", 4, 16, 8, step=4)

                    ds_to_browse = falah_loader.raw_train if chosen_split == "train" else falah_loader.raw_test
                    target_label_idx = info["class_names"].index(chosen_class) if chosen_class != "All" else None

                    filtered_items = []
                    for idx in range(len(ds_to_browse)):
                        lbl = int(ds_to_browse[idx]["label"])
                        if target_label_idx is None or lbl == target_label_idx:
                            filtered_items.append((idx, ds_to_browse[idx]["image"], info["class_names"][lbl]))
                        if len(filtered_items) >= num_samples:
                            break

                    grid_cols = st.columns(4)
                    for i, (orig_idx, img_obj, c_name) in enumerate(filtered_items):
                        with grid_cols[i % 4]:
                            st.image(
                                img_obj.resize((180, 180)),
                                caption=f"#{orig_idx} - {c_name}",
                                use_container_width=True,
                            )

            with tab_leakage:
                st.markdown("#### Cross-Split Duplicate & Slice-Leakage Analysis")
                st.write(
                    "Because original participant/subject IDs are missing from the Falah dataset, "
                    "pre-slicing into 2D axial views can place consecutive slices from the same subject into both "
                    "train and test sets."
                )

                if st.button("Run Duplicate & Near-Duplicate Scan (Sample Batch)"):
                    with st.spinner("Analyzing image hashes across train and test splits..."):
                        dup_res = falah_loader.detect_duplicates(max_samples=100)
                    st.success("Duplicate analysis complete!")
                    st.json(dup_res)

                st.error(
                    "**Methodological Finding:** Participant-level independence **cannot be claimed** for Falah. "
                    "Reported test accuracies over 95% frequently reflect skull/slice recognition rather than generalizable pathology."
                )

            with tab_report:
                st.markdown("#### Generated Clinical Audit Report")
                st.text(falah_loader.generate_summary_report())

        except Exception as e:
            st.error(f"Error loading Falah dataset: {str(e)}")

    # --------------------------------------------------------------------------
    # DATASET B: OASIS-1
    # --------------------------------------------------------------------------
    elif selected_dataset == "OASIS-1 (Open Access Series of Imaging Studies)":
        st.subheader("🌐 OASIS-1 Cross-Sectional Neuroimaging Dataset")
        st.write(
            "Official Source: [sites.wustl.edu/oasisbrains/oasis-1](https://sites.wustl.edu/oasisbrains/oasis-1/)  \n"
            "Features **416 participants** (ages 18–96) with T1-weighted structural MRI, complete clinical CDR ratings, "
            "and participant IDs supporting true **disjoint participant-level splitting**."
        )

        oasis_loader = OASIS1DatasetLoader()
        validator = oasis_loader.validator
        meta_mgr = oasis_loader.metadata_mgr

        # Dropzone & Directory Status
        dir_info = validator.detect_directories()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Raw Archives:** `{'✅ Found' if dir_info['raw_dir_exists'] else '❌ Missing'}`")
        c2.markdown(f"**Extracted Scans:** `{'✅ Found' if dir_info['extracted_dir_exists'] else '❌ Missing'}`")
        c3.markdown(f"**Clinical Metadata:** `{'✅ Found' if dir_info['metadata_dir_exists'] else '❌ Missing'}`")

        tab_import, tab_scans, tab_labels, tab_split, tab_manifest = st.tabs([
            "📥 Manual Import Dropzone",
            "🔍 Scans & Metadata Matching",
            "🏷️ Clinical Label Mapping",
            "🛡️ Participant Splitting",
            "📋 Manifest & Export",
        ])

        with tab_import:
            st.markdown("#### Manual Archive Import & Safe Extraction")
            st.info(
                "In compliance with data access requirements, NeuroVision **does not automatically download** OASIS-1. "
                "Place authorized archive files into `data/oasis1/raw/` and clinical metadata into `data/oasis1/metadata/`."
            )

            archive_res = validator.validate_raw_archives()
            if archive_res["total_archives"] == 0:
                st.warning("No archives found in `data/oasis1/raw/`. Place `.tar.gz` or `.zip` files there.")
            else:
                st.success(f"Detected {archive_res['total_archives']} raw archive(s).")
                for arc in archive_res["archives"]:
                    st.write(f"- **{arc['file_name']}** ({arc['size_mb']} MB) - Valid: {arc['is_valid']}")

                st.markdown("##### Manual Extraction Safeguard")
                sel_arc = st.selectbox(
                    "Select Archive to Extract:",
                    options=[a["file_name"] for a in archive_res["archives"]],
                )
                confirm_check = st.checkbox("I confirm manual extraction of this archive into data/oasis1/extracted/")
                if st.button("Extract Selected Archive"):
                    with st.spinner("Extracting archive..."):
                        ext_res = validator.extract_archive(sel_arc, confirm=confirm_check)
                    if ext_res.get("extracted"):
                        st.success(f"Extracted to: {ext_res['destination']}")
                    else:
                        st.error(ext_res.get("message") or ext_res.get("error"))

        with tab_scans:
            st.markdown("#### Scans & Clinical Metadata Matching")
            st.caption("Distinguishing: **Participants** vs **Sessions** vs **Acquisitions** vs **Individual Image Files**")

            if st.button("Scan Local Directories & Index Scans"):
                with st.spinner("Indexing scans and matching with metadata..."):
                    index_res = oasis_loader.scan_and_index()
                st.success("Indexing complete!")
                st.json(index_res)

            # Metadata files
            meta_files = meta_mgr.discover_metadata_files()
            st.markdown(f"**Discovered Metadata Files:** `{len(meta_files)}`")
            for mf in meta_files:
                st.write(f"- `{mf.name}`")

        with tab_labels:
            st.markdown("#### Clinical Dementia Rating (CDR) Mapping Configuration")
            task_choice = st.radio(
                "Select Research Task:",
                options=["Multiclass (4 Stages)", "Binary (Demented vs Non-Demented)"],
                index=0,
            )

            if "Multiclass" in task_choice:
                st.markdown("""
                | CDR Score | Clinical Designation | NeuroVision Research Label |
                | :---: | :--- | :--- |
                | **0.0** | None | **Non-Demented** |
                | **0.5** | Very Mild | **Very Mild Dementia** |
                | **1.0** | Mild | **Mild Dementia** |
                | **2.0** | Moderate | **Moderate Dementia** |
                | *Null / Blank* | Unrated (<60 yrs controls) | **Unrated / Excluded** |
                """)
            else:
                st.markdown("""
                | CDR Score | Clinical Designation | NeuroVision Research Label |
                | :---: | :--- | :--- |
                | **0.0** | None | **Non-Demented** |
                | **> 0.0** (0.5, 1.0, 2.0) | Impairment | **Demented** |
                | *Null / Blank* | Unrated (<60 yrs controls) | **Unrated / Excluded** |
                """)

        with tab_split:
            st.markdown("#### Participant-Level Disjoint Splitting")
            st.success(
                "🛡️ **Zero Patient Leakage Guarantee:** Partitioning occurs strictly at the unique `participant_id` level. "
                "All repeat/reliability sessions (e.g. `_MR1` and `_MR2`) remain in the same split."
            )

            col_s1, col_s2, col_s3 = st.columns(3)
            train_r = col_s1.slider("Training Ratio", 0.50, 0.85, 0.70, 0.05)
            val_r = col_s2.slider("Validation Ratio", 0.05, 0.25, 0.15, 0.05)
            test_r = col_s3.slider("Testing Ratio", 0.05, 0.25, 0.15, 0.05)

            seed_val = st.number_input("Reproducible Random Seed", value=42, step=1)

        with tab_manifest:
            st.markdown("#### Dataset Manifest (`oasis1_manifest.csv`)")
            st.write("Generates the standardized manifest with complete clinical and audit fields.")

            manifest_file = Path("data/oasis1/oasis1_manifest.csv")
            if manifest_file.exists():
                st.success(f"Existing manifest detected: `{manifest_file}`")
                df_man = pd.read_csv(manifest_file)
                st.dataframe(df_man.head(15), use_container_width=True)

                st.download_button(
                    label="📥 Download oasis1_manifest.csv",
                    data=manifest_file.read_bytes(),
                    file_name="oasis1_manifest.csv",
                    mime="text/csv",
                )
            else:
                st.info("Manifest has not yet been exported for this session. Run indexing above to generate it.")

# ==============================================================================
# SECTION 2: CLINICAL INFERENCE & EXPLAINABILITY (EXISTING WORKFLOW PRESERVED)
# ==============================================================================
elif nav_mode == "🧠 Clinical Diagnosis & Inference":
    st.header("Clinical Brain MRI Diagnostic Assistant")
    st.write(
        "Upload a brain MRI slice to diagnose cognitive impairment stage: "
        "**Non-Demented**, **Very Mild Demented**, **Mild Demented**, or **Moderate Demented**."
    )

    # Sidebar settings
    st.sidebar.header("Model Configuration")
    available_checkpoints = []
    if MODELS_DIR.exists():
        available_checkpoints.extend(list(MODELS_DIR.glob("**/*.pt")))
        available_checkpoints.extend(list(MODELS_DIR.glob("**/*.pth")))
        available_checkpoints.extend(list(MODELS_DIR.glob("**/*.keras")))
        available_checkpoints.extend(list(MODELS_DIR.glob("**/*.h5")))
    available_checkpoints = sorted(list(set(available_checkpoints)))

    if not available_checkpoints:
        st.sidebar.warning("⚠️ No trained checkpoint found in `models/` yet. Train one via **Model Training** page.")
        selected_checkpoint = None
    else:
        def format_model_opt(p: Path) -> str:
            fw = "PyTorch" if p.suffix in [".pt", ".pth"] else "Keras"
            return f"[{fw}] {p.parent.name}/{p.name}"

        selected_checkpoint = st.sidebar.selectbox(
            "Select Model Checkpoint",
            options=available_checkpoints,
            format_func=format_model_opt,
        )

    explain_mode = st.sidebar.checkbox("Generate Grad-CAM Heatmap", value=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Brain MRI")
        uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG/JPEG)...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with col2:
        st.subheader("2. Diagnostic Analysis")
        if uploaded_file is not None:
            if selected_checkpoint is None:
                st.error("Please ensure a trained model exists in the `models/` folder.")
            else:
                with st.spinner("Analyzing brain MRI scan..."):
                    predictor = NeuroVisionPredictor(checkpoint_path=selected_checkpoint)
                    result = predictor.predict(image, explain=explain_mode)

                predicted_class = result["predicted_class"]
                confidence = result["confidence"]

                # Severity badge styling
                if "Non" in predicted_class:
                    st.success(f"**Diagnosis: {predicted_class.replace('_', ' ')}** ({confidence}%)")
                elif "Very" in predicted_class:
                    st.info(f"**Diagnosis: {predicted_class.replace('_', ' ')}** ({confidence}%)")
                elif "Mild" in predicted_class:
                    st.warning(f"**Diagnosis: {predicted_class.replace('_', ' ')}** ({confidence}%)")
                else:
                    st.error(f"**Diagnosis: {predicted_class.replace('_', ' ')}** ({confidence}%)")

                # Probability chart
                st.markdown("#### Stage Probability Distribution")
                for cls_name, prob in result["probabilities"].items():
                    st.progress(prob / 100.0, text=f"{cls_name.replace('_', ' ')}: {prob}%")

                # Grad-CAM visualization
                if explain_mode and "gradcam_heatmap" in result:
                    st.markdown("#### 🧠 Anatomical Attention (Grad-CAM)")
                    heatmap = result["gradcam_heatmap"]

                    resized_img = np.array(image.resize((224, 224))) / 255.0
                    colored_cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB) / 255.0

                    overlay = 0.6 * resized_img + 0.4 * colored_cam
                    overlay = np.clip(overlay, 0, 1)

                    st.image(overlay, caption="Grad-CAM Attention Overlay", use_container_width=True)
                    st.caption("Red/yellow highlights signify the key anatomical features influencing the prediction.")
        else:
            st.info("Awaiting MRI scan upload to run diagnosis.")

# ==============================================================================
# SECTION 3: SYSTEM FLOWCHART & ARCHITECTURE
# ==============================================================================
elif nav_mode == "🗺️ System Flowchart & Architecture":
    st.header("🗺️ System Flowchart & Architectural Topology")
    st.write(
        "Interactive end-to-end operational blueprint of **NeuroVision 2.0**, depicting "
        "data ingestion, participant-level splitting, preprocessing, multi-backbone training, "
        "visual explainability, and multi-service deployment."
    )

    flowchart_mermaid = """
flowchart TD
    %% User and Client Interfaces
    subgraph UI_API["Client and User Interfaces"]
        streamlit_app["Streamlit Research Studio - 10 Interactive Pages on Port 8501"]
        fastapi_server["FastAPI REST Microservice - OpenAPI and Swagger on Port 8000"]
    end

    %% Data Ingestion Layer
    subgraph DATA_LAYER["Unified Multi-Cohort Ingestion and Governance"]
        falah_data["Falah Alzheimer MRI - 2D Axial Arrow Slices 6400 Images"]
        oasis_data["OASIS-1 Cohort - 3D Volumetric T1 NIfTI and Analyze 7.5"]
        falah_loader["FalahDatasetLoader - Verification and Hash Duplicate Scan"]
        oasis_loader["OASIS1DatasetLoader - Ingestion and CDR Clinical Mapping"]
        oasis_manifest["OASIS1ManifestManager - Participant-Level Disjoint 70/15/15 Split"]
        falah_leakage_warn["Falah Provenance Audit - Slice Leakage Warning"]
    end

    %% Preprocessing Pipeline
    subgraph PREPROC["Radiometric Preprocessing and Augmentation Engine"]
        norm_engine["Intensity Normalization - Min-Max Scaling 0 to 1 or Z-Score"]
        aug_engine["Spatial Augmentation - Rotations, Flips, and Affine Transforms"]
        weight_engine["Inverse-Frequency Balancing - Weighted Cross-Entropy Weights"]
    end

    %% Model Zoo
    subgraph MODELS["Deep Learning Model Zoo and Baselines"]
        cnn_model["Custom Medical CNN - 4-Block Conv2D + BatchNorm + LeakyReLU"]
        resnet_model["ResNet-18 / ResNet-50 - Deep Residual Transfer Learning"]
        vit_model["Vision Transformer ViT - 16x16 Patch Self-Attention"]
        sim_engine["Simulation Engine - NumPy Zero-Dependency Baseline"]
    end

    %% Training & Tracking
    subgraph TRAIN["Training and Optimization Pipeline"]
        loss_fn["Loss Functions - Weighted Cross-Entropy and Focal Loss"]
        optimizer["Optimization - AdamW and Cosine Annealing LR Scheduler"]
        trainer["UnifiedMRITrainer - Live Metrics Tracking and Early Stopping"]
        checkpoints[("Model Checkpoints Store - pt, pth, keras, and json in models")]
    end

    %% Evaluation & Explainability
    subgraph EVAL_XAI["Diagnostic Evaluation and Explainable AI"]
        eval_engine["Clinical Evaluation - Confusion Matrix, Sens, Spec, Macro-F1"]
        roc_engine["Multi-Class ROC Analysis - One-vs-Rest AUC and 95% Bootstrap CI"]
        gradcam["Grad-CAM Engine - Convolutional Feature Map Gradients"]
        vit_rollout["ViT Attention Rollout - Multi-Head Attention Propagation"]
        heatmaps["Radiometric Overlays - Jet and Magma Color Overlays on Slices"]
    end

    %% Deployment & Infrastructure
    subgraph DEPLOY["Production Deployment and CI/CD Automation"]
        docker_compose["Docker Compose Orchestration - Dual Service api 8000 + web 8501"]
        dockerfile["Multi-Stage Dockerfile - Python 3.11-slim + PyTorch CPU/CUDA"]
        github_ci["GitHub Actions CI Matrix - Automated Testing on Python 3.11 and 3.12"]
    end

    %% Flow Connections
    falah_data --> falah_loader
    oasis_data --> oasis_loader
    oasis_loader --> oasis_manifest
    falah_loader --> falah_leakage_warn

    falah_loader --> norm_engine
    oasis_manifest --> norm_engine
    norm_engine --> aug_engine
    falah_loader --> weight_engine

    aug_engine --> trainer
    weight_engine --> loss_fn
    loss_fn --> trainer
    optimizer --> trainer

    cnn_model --> trainer
    resnet_model --> trainer
    vit_model --> trainer
    sim_engine --> trainer

    trainer --> checkpoints
    checkpoints --> eval_engine
    checkpoints --> gradcam
    checkpoints --> vit_rollout

    eval_engine --> roc_engine
    gradcam --> heatmaps
    vit_rollout --> heatmaps

    heatmaps --> streamlit_app
    heatmaps --> fastapi_server
    roc_engine --> streamlit_app
    checkpoints --> fastapi_server

    fastapi_server --> docker_compose
    streamlit_app --> docker_compose
    dockerfile --> docker_compose
    docker_compose --> github_ci
"""

    st.markdown(f"```mermaid\n{flowchart_mermaid}\n```")

    st.subheader("Subsystem Architectural Specifications")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Ingestion & Provenance")
        st.markdown(
            "- **Falah 2D Arrow:** 6,400 axial slices (5,120 train / 1,280 test).\n"
            "- **OASIS-1 3D Volumes:** 434 sessions across 416 unique subjects.\n"
            "- **Zero-Leakage Splitting:** Partitioned strictly by `participant_id` (70/15/15).\n"
            "- **Clinical Mapping:** Validated CDR 0.0, 0.5, 1.0, and 2.0 scores."
        )

    with c2:
        st.markdown("#### 2. Models & Optimization")
        st.markdown(
            "- **Custom Medical CNN:** 4-block deep CNN with LeakyReLU (~1.2M params).\n"
            "- **Medical ResNets:** ResNet-18 (~11.5M) and ResNet-50 (~24.5M).\n"
            "- **Vision Transformer:** $16\\times 16$ patch self-attention (~9.8M).\n"
            "- **Loss Balancing:** Inverse-frequency weighted Cross-Entropy & Focal loss."
        )

    with c3:
        st.markdown("#### 3. Explainability & Serving")
        st.markdown(
            "- **Grad-CAM:** Gradient backpropagation through final conv layers.\n"
            "- **ViT Rollout:** Multi-head self-attention token propagation.\n"
            "- **FastAPI Backend:** OpenAPI documentation at `http://localhost:8000/docs`.\n"
            "- **Containerization:** Production Dockerfile & dual-service Compose."
        )
