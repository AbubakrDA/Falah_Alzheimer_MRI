import streamlit as st
from src.config.settings import DISCLAIMER_TEXT

st.set_page_config(page_title="Research Overview - NeuroVision", page_icon="📖", layout="wide")

st.title("📖 NeuroVision Research Overview")
st.warning(f"⚠️ **BIOMEDICAL AI DISCLAIMER:** {DISCLAIMER_TEXT}")

st.markdown("""
## 1. Scientific Role and Research Context
NeuroVision is designed to investigate deep-learning methodologies for classifying Alzheimer's Disease (AD)
and related neurodegenerative stages from structural Magnetic Resonance Imaging (MRI).

In translational neuroimaging, accurate stratification across early stages—particularly separating
**Non-Demented (CDR 0)** from **Very Mild Cognitive Impairment (CDR 0.5)**—is crucial for clinical trials
and therapeutic intervention windows.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cohort 1: Falah Alzheimer MRI")
    st.markdown("""
    - **Format:** Hugging Face Arrow table (`train` and `test` splits).
    - **Total Records:** 6,400 2D axial MRI slices (5,120 train, 1,280 test).
    - **Resolution:** 128 x 128 grayscale (`L`).
    - **Target Classes:** 4 (`Mild_Demented`, `Moderate_Demented`, `Non_Demented`, `Very_Mild_Demented`).
    - **Key Limitation:** No participant identifiers are recorded in the Arrow table.
      Potential patient-level cross-split contamination may exist; cannot claim subject-independent generalization.
    """)

with col2:
    st.subheader("Cohort 2: OASIS-1 (Washington University ADRC)")
    st.markdown("""
    - **Official Source:** [OASIS Brains Project](https://sites.wustl.edu/oasisbrains/oasis-1/)
    - **Format:** Volumetric 3D MRI (NIfTI `.nii`/`.nii.gz` and Analyze 7.5 `.hdr`/`.img`).
    - **Cohort Size:** 416 subjects aged 18–96 across 434 imaging sessions.
    - **Clinical Ground Truth:** Clinical Dementia Rating (CDR 0, 0.5, 1, 2) and MMSE.
    - **Provenance Guarantee:** Rigorous participant-level splitting (zero patient leakage across train/val/test).
    """)

st.markdown("""
---
## 2. Platform Architecture
```
                        +---------------------------------------------+
                        |        NeuroVision Unified Platform         |
                        +---------------------------------------------+
                                       |
                +----------------------+----------------------+
                |                                             |
   +------------------------+                    +------------------------+
   |   Falah Alzheimer MRI  |                    |    OASIS-1 Ingestion   |
   |   (Local Arrow Tables) |                    | (NIfTI & Analyze 7.5)  |
   +------------------------+                    +------------------------+
                |                                             |
   +------------------------+                    +------------------------+
   | Stratified Val Split   |                    | Participant Isolation  |
   | (From Train Only)      |                    | Zero Patient Leakage   |
   +------------------------+                    +------------------------+
                \                                             /
                 \                                           /
                  +-----------------------------------------+
                  |         Deep Learning Engine            |
                  |   Custom CNN | ResNet50 | Transformer   |
                  +-----------------------------------------+
                                       |
                  +-----------------------------------------+
                  |    Evaluation & Explainability Suite    |
                  |  Macro-F1 | Balanced Acc | Grad-CAM/ViT |
                  +-----------------------------------------+
```

---
## 3. Confounding Factors and Methodological Rigor
- **Age as a Potential Confounder:** In OASIS-1, healthy controls span young, middle-aged, and older adults,
  whereas cognitively impaired subjects are predominantly older adults. Models may erroneously learn age-related
  atrophy rather than Alzheimer-specific pathology if age distributions are not accounted for.
- **Metric Incomparability:** Metrics obtained on Falah (slice-level) and OASIS-1 (participant-isolated volume slices)
  must **never** be directly compared as equivalent benchmarks.
""")
