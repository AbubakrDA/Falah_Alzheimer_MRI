import sys
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd

# Refresh any stale src.data or src.config modules cached in the running Streamlit process
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src.data") or mod_name.startswith("src.config"):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import (
    OASIS1_RAW_DIR,
    OASIS1_EXTRACTED_DIR,
    OASIS1_METADATA_DIR,
    OASIS1_PROCESSED_DIR,
    DISCLAIMER_TEXT,
)
from src.data.dataset_manager import DatasetManager
from src.data.falah_loader import FalahDatasetLoader
from src.data.oasis1_loader import Oasis1DatasetLoader
from src.data.oasis1_validator import Oasis1Validator
from src.data.oasis1_metadata import Oasis1MetadataManager
from src.data.oasis1_label_mapper import Oasis1LabelMapper
from src.data.oasis1_manifest import Oasis1ManifestBuilder

st.set_page_config(page_title="Dataset Management - NeuroVision", page_icon="📊", layout="wide")

st.title("📊 Dataset Management & Provenance")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

dataset_choice = st.radio(
    "Active Research Dataset",
    options=["Falah Alzheimer MRI", "OASIS-1"],
    horizontal=True
)

st.markdown("---")

if dataset_choice == "Falah Alzheimer MRI":
    st.subheader("Falah Alzheimer MRI Dataset (Arrow Table)")
    
    with st.spinner("Loading local Falah dataset summary..."):
        loader = FalahDatasetLoader()
        summary = loader.get_summary()

    c1, c2, c3, c4 = st.columns(4)
    counts = summary["split_counts"]
    c1.metric("Total Slices", f"{summary['total_samples']:,}")
    c2.metric("Train Split", f"{counts['train']:,}")
    c3.metric("Validation Split", f"{counts['val']:,}")
    c4.metric("Test Split (Original)", f"{counts['test']:,}")

    st.markdown("### Class Distributions")
    dist = summary["class_distributions"]
    dist_df = pd.DataFrame(dist).T
    st.dataframe(dist_df, use_container_width=True)

    st.warning("⚠️ **Severe Imbalance Warning:** `Moderate_Demented` represents only ~0.96% of training samples (49 images). Balanced class weighting is automatically computed and enforced during model training.")

    st.info(f"ℹ️ **Provenance & Methodological Limitation:** {summary['provenance_limitation']}")

    st.markdown("### Sample Image Inspection (Train, Val, & Test Splits)")
    c_split, c_filter, c_cnt = st.columns(3)
    with c_split:
        split_to_view = st.selectbox("Split to preview", ["train", "val", "test"])
    with c_filter:
        class_filter = st.selectbox("Class Filter", ["All"] + loader.class_names)
    with c_cnt:
        count_to_view = st.slider("Sample Count", 4, 16, 8, step=4)

    samples = loader.get_sample_images(split=split_to_view, count=count_to_view, class_filter=class_filter)
    if samples:
        cols = st.columns(4)
        for i, s in enumerate(samples):
            col = cols[i % 4]
            with col:
                col.image(s["image"], caption=f"Idx {s['index']}: {s['class_name']}", use_container_width=True)
    else:
        st.info("No samples match the selected filter.")

else:
    st.subheader("OASIS-1 Cohort Management (Manual Ingestion & Manifest)")
    oasis_loader = Oasis1DatasetLoader()
    status = oasis_loader.get_status()

    st.markdown(f"""
    **Official Source:** [{status['official_source']}]({status['official_source']})  
    *OASIS-1 data must be authorized and downloaded manually by researchers. The platform will never download restricted data automatically.*
    """)

    tabs = st.tabs(["1. Raw Archives", "2. Extracted Scans", "3. Metadata Matching", "4. Manifest & Export"])

    with tabs[0]:
        st.markdown("#### Directory Detection & Archive Validation")
        st.write(f"Raw directory: `{status['raw_dir']}`")
        raw_info = Oasis1Validator.inspect_raw_archives(Path(status["raw_dir"]))

        if raw_info["archives_found"] == 0:
            st.info("No raw archives (.tar, .tar.gz, .zip) currently found in `data/OASIS-1/raw/`.")
            st.caption("Place your authorized OASIS-1 download archives (e.g., `oasis_cross-sectional_disc01.tar.gz`) in `data/OASIS-1/raw/`.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Archives Found", raw_info["archives_found"])
            c2.metric("Valid Archives", raw_info["valid_count"])
            c3.metric("Corrupted Archives", raw_info["corrupted_count"])

            st.dataframe(pd.DataFrame(raw_info["archives"]), use_container_width=True)

            st.markdown("##### Explicit User-Controlled Extraction")
            selected_archive = st.selectbox("Select archive to extract", [a["path"] for a in raw_info["archives"] if a["is_valid"]])
            if st.button("Extract Selected Archive to extracted/"):
                with st.spinner("Extracting archive safely..."):
                    res = Oasis1Validator.extract_archive(Path(selected_archive), Path(status["extracted_dir"]))
                    if res["success"]:
                        st.success(f"Successfully extracted {res['extracted_count']} files to {res['destination']}")
                    else:
                        st.error(f"Extraction error: {res['error']}")

    with tabs[1]:
        st.markdown("#### Scan Discovery (.nii, .nii.gz, Analyze 7.5 .hdr/.img)")
        st.write(f"Extracted directory: `{status['extracted_dir']}`")
        extracted_info = Oasis1Validator.scan_extracted_images(Path(status["extracted_dir"]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", extracted_info["images_found"])
        c2.metric("Valid Volumes", extracted_info["valid_images"])
        c3.metric("Orphaned Headers", extracted_info["orphaned_headers"])
        c4.metric("Orphaned Images", extracted_info["orphaned_images"])

        if extracted_info["records"]:
            st.dataframe(pd.DataFrame(extracted_info["records"]), use_container_width=True)
        else:
            st.info("No extracted scans found yet in `data/OASIS-1/extracted/`.")

    with tabs[2]:
        st.markdown("#### Metadata Matching & Label Mapping")
        st.write(f"Metadata directory: `{status['metadata_dir']}`")
        meta_files = list(Path(status["metadata_dir"]).glob("*.csv")) + list(Path(status["metadata_dir"]).glob("*.xlsx"))

        if not meta_files:
            st.info("Place `oasis_cross-sectional.csv` or `.xlsx` into `data/OASIS-1/metadata/`.")
        else:
            meta_path = st.selectbox("Selected metadata spreadsheet", meta_files, format_func=lambda p: p.name)
            meta_mgr = Oasis1MetadataManager(meta_path)
            st.dataframe(meta_mgr.normalized_df.head(10), use_container_width=True)

            task_type = st.radio("Clinical Research Task", ["multiclass", "binary"], horizontal=True)
            if extracted_info["records"]:
                matching_res = meta_mgr.match_with_mri_records(extracted_info["records"])
                st.write(f"Matched MRIs: **{matching_res['matched_mris']}** | Unmatched MRIs: **{matching_res['unmatched_mris']}** | Missing CDR: **{matching_res['missing_cdr_count']}**")

    with tabs[3]:
        st.markdown("#### Participant-Isolated Manifest & Export")
        if oasis_loader.manifest_df is not None and not oasis_loader.manifest_df.empty:
            st.success(f"Active Manifest Loaded: **{len(oasis_loader.manifest_df)}** records.")
            st.dataframe(oasis_loader.manifest_df.head(20), use_container_width=True)

            if st.button("Export Standardized Research CSV Reports"):
                builder = Oasis1ManifestBuilder(oasis_loader.manifest_df.to_dict(orient="records"))
                exported = builder.export_reports(Path(status["processed_dir"]))
                st.success("Successfully generated and exported research reports:")
                for k, v in exported.items():
                    st.write(f"- `{k}`: {v}")
        else:
            st.info("No manifest built yet. Once extracted scans and metadata are loaded, build your manifest with participant-level split isolation.")
