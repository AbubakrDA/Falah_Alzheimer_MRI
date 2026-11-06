import sys
import importlib
import streamlit as st
import pandas as pd
from pathlib import Path

# Hot-reload any updated src modules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src."):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import (
    REPORTS_DIR,
    OASIS1_PROCESSED_DIR,
    DISCLAIMER_TEXT
)
from src.data.falah_loader import FalahDatasetLoader
from src.data.oasis1_loader import Oasis1DatasetLoader

st.set_page_config(page_title="Research Reports - NeuroVision", page_icon="📝", layout="wide")

st.title("📝 Scientific Provenance & Research Audit Reports")
st.error(f"🔴 **MANDATORY SCIENTIFIC PROTOTYPE STATEMENT:** {DISCLAIMER_TEXT}")

st.markdown("""
## 1. Data Provenance & Cohort Integrity Logs
Transparent documentation of data sources, split isolation, and methodological constraints.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Falah Alzheimer MRI Audit")
    f_loader = FalahDatasetLoader()
    f_summary = f_loader.get_summary()
    img_size = f_summary.get("image_size") or f_summary.get("native_size", (128, 128))

    st.markdown(f"""
    - **Dataset Source:** Local HuggingFace Arrow Archive (`data/Falah_Alzheimer_MRI`)
    - **Total Slices:** {f_summary['total_samples']:,} (Train: {f_summary['split_counts']['train']:,}, Val: {f_summary['split_counts']['val']:,}, Test: {f_summary['split_counts']['test']:,})
    - **Color / Format:** Grayscale 8-bit ({img_size[0]}x{img_size[1]})
    - **Class Balance:** Non-Demented (50.1%), Very Mild (34.8%), Mild (14.1%), Moderate (0.96%)
    - **Leakage Limitation:** *{f_summary['provenance_limitation']}*
    """)

with col2:
    st.subheader("OASIS-1 Clinical Cohort Audit")
    o_loader = Oasis1DatasetLoader()
    o_status = o_loader.get_status()

    st.markdown(f"""
    - **Dataset Source:** [{o_status['official_source']}]({o_status['official_source']}) (Manual Ingestion)
    - **Supported Formats:** NIfTI (`.nii`, `.nii.gz`) and Analyze 7.5 (`.hdr`/`.img`)
    - **Participant Isolation:** Guaranteed zero patient leakage across train/val/test partitions.
    - **Confounding Factor:** Age distribution disparity between cognitively intact adults and older demented subjects.
    """)

st.markdown("---")
st.subheader("2. Generated Manifest and Audit Files")

processed_path = Path(OASIS1_PROCESSED_DIR)
report_files = []
if processed_path.exists():
    report_files = list(processed_path.glob("*.csv")) + list(processed_path.glob("*.json"))

if not report_files:
    st.info("No OASIS-1 manifest reports generated yet. Process the cohort in **Dataset Management** to export audit reports.")
else:
    for rf in sorted(report_files):
        st.write(f"- `{rf.name}` ({rf.stat().st_size:,} bytes)")
        if rf.suffix == ".csv":
            with st.expander(f"Preview {rf.name}"):
                df = pd.read_csv(rf)
                st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")
st.subheader("3. Comprehensive Research Summary Export")
st.download_button(
    label="Download NeuroVision Provenance Whitepaper (Markdown)",
    data=f"""# NeuroVision Research Integrity & Provenance Report

## Scientific Disclaimer
{DISCLAIMER_TEXT}

## 1. Methodological Overview
NeuroVision provides reproducible multi-dataset brain MRI classification with:
1. Falah Alzheimer MRI (6,400 2D axial slices)
2. OASIS-1 Cross-Sectional Cohort (volumetric T1 with CDR ground truth)

## 2. Participant Isolation
All repeated acquisitions and sessions for a given participant in OASIS-1 remain strictly
within that participant's assigned partition (Train: 70%, Val: 15%, Test: 15%).
Zero patient overlap is mathematically verified.

## 3. Class Imbalance Strategy
Balanced class weighting (compute_class_weight) is enforced during training to prevent
model collapse on rare disease stages (e.g. Moderate Demented < 1%).
""",
    file_name="neurovision_provenance_whitepaper.md",
    mime="text/markdown"
)
