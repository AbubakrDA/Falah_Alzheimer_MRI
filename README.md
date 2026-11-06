# 🧠 NeuroVision 2.0: Multi-Dataset Brain MRI Staging & Research Platform

[![CI Pipeline](https://github.com/your-username/NeuroVision/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/NeuroVision/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroVision 2.0** is an enterprise-grade biomedical AI platform and research suite designed for automated brain MRI cognitive impairment staging, rigorous neuroimaging data provenance, multi-model benchmarking, and visual explainability.

The platform provides a unified data governance layer seamlessly integrating both the **Falah Alzheimer MRI** dataset (2D axial Arrow slices) and the **OASIS-1** cross-sectional dataset (3D volumetric T1-weighted NIfTI & Analyze 7.5 scans with clinical dementia evaluations), ensuring **strict participant-level isolation** with **zero patient leakage**.

---

## 📑 Table of Contents
1. [Key Highlights & Capabilities](#-key-highlights--capabilities)
2. [Application Flowchart & System Architecture](#-application-flowchart--system-architecture)
3. [Representative Train & Test MRI Image Gallery](#-representative-train--test-mri-image-gallery)
4. [Cohort Provenance & Dataset Comparison](#-cohort-provenance--dataset-comparison)
5. [Clinical Dementia Rating (CDR) Mapping](#-clinical-dementia-rating-cdr-mapping)
6. [Participant-Level Splitting & Leakage Prevention](#-participant-level-splitting--leakage-prevention)
7. [Supported Deep Learning Backbones](#-supported-deep-learning-backbones)
8. [Clinical Explainability Engines](#-clinical-explainability-engines)
9. [Interactive Streamlit Multi-Page Interface](#-interactive-streamlit-multi-page-interface)
10. [FastAPI REST Service & Documentation](#-fastapi-rest-service--documentation)
11. [Docker & Containerized Deployment](#-docker--containerized-deployment)
12. [Installation & Local Setup](#-installation--local-setup)
13. [Testing & Verification](#-testing--verification)
14. [GitHub Repository & CI/CD](#-github-repository--cicd)
15. [Medical & Ethical Prototype Disclaimer](#-medical--ethical-prototype-disclaimer)

---

## 🌟 Key Highlights & Capabilities

- **Unified Multi-Cohort Data Layer:** Coordinates disparate data structures (Hugging Face Arrow tables, Analyze 7.5 `.hdr`/`.img`, NIfTI `.nii`/`.nii.gz`, and CSV/Excel clinical tables) through a single provenance-aware API ([`DatasetManager`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/src/data/dataset_manager.py)).
- **Zero Patient Leakage Guarantee:** Enforces mathematical participant-level disjoint partitioning (`participant_id`) on clinical scans, preventing slice-level data leakage between training and testing holdouts.
- **Multi-Backbone Model Suite:** Modular implementations of **Custom Medical CNN**, **ResNet-18**, **ResNet-50**, and **Medical Vision Transformers (ViT)**, alongside a zero-dependency **Simulation Engine**.
- **Dual Framework Resilience:** Native accelerated **PyTorch** pipeline with graceful fallbacks for **TensorFlow/Keras** and CPU environments, eliminating Windows Python 3.12 C++ DLL crashes.
- **Biomedical Explainability:** Full support for **Grad-CAM** (convolutional backbones) and **ViT Attention Rollout** (transformer self-attention heads) with radiometric colormap overlays.
- **Full RESTful FastAPI Backend:** Production-grade REST API providing real-time inference, Base64 visual explainability, and dataset metadata with interactive Swagger docs at `/docs`.
- **Streamlit 10-Page Research Studio:** Full-featured laboratory workbench spanning dataset management, 3D orthogonal slicing, preprocessing, training, ROC curves, benchmarking, and whitepaper reporting.
- **Containerized Orchestration:** Production `Dockerfile` and `docker-compose.yml` for unified deployment of the API and UI.

---

## 🏛️ System Architecture

```
Falah_Alzheimer_MRI/
├── app.py                              # Main Streamlit App: Diagnostic Assistant & Dataset Explorer
├── Dockerfile                          # Multi-stage production container build
├── docker-compose.yml                  # Unified API + Streamlit service orchestration
├── requirements.txt                    # Project dependencies (PyTorch, FastAPI, Streamlit, etc.)
├── README.md                           # System documentation & technical specifications
├── .gitignore                          # Version control exclusions
├── .dockerignore                       # Docker build exclusions
│
├── .github/                            # GITHUB INFRASTRUCTURE & CI/CD
│   ├── workflows/
│   │   └── ci.yml                      # Automated testing, linting, and Docker validation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md               # Structured issue template for clinical bugs
│   │   └── feature_request.md          # Architectural proposal template
│   └── pull_request_template.md        # Medical AI code review & data governance checklist
│
├── pages/                              # STREAMLIT RESEARCH STUDIO (10 INTERACTIVE PAGES)
│   ├── 2_Dataset_Management.py         # Arrow & OASIS-1 inspection, indexing, & manifest export
│   ├── 3_MRI_Viewer.py                 # Multi-planar orthogonal slicer (Axial, Coronal, Sagittal)
│   ├── 4_Image_Preprocessing.py        # Radiometric scaling (Min-Max/Z-Score) & Augmentation
│   ├── 5_Model_Training.py             # Live training, loss/accuracy curves, & class weighting
│   ├── 6_Model_Evaluation.py           # Test evaluation, Confusion Matrix, & Multi-Class ROC
│   ├── 7_Explainable_AI.py             # Grad-CAM & ViT Attention Rollout visualizer
│   ├── 8_Model_Comparison.py           # Experiment benchmarking & leaderboard
│   ├── 9_Prediction.py                 # Dedicated staging, probability spectrum, & heatmaps
│   └── 10_Research_Reports.py          # Data provenance whitepaper & audit log export
│
├── src/                                # CORE PYTHON SOURCE CODE
│   ├── config/
│   │   └── settings.py                 # Central hyperparameter & path configurations
│   │
│   ├── data/                           # UNIFIED DATASET MANAGEMENT LAYER
│   │   ├── dataset_manager.py          # DatasetManager: format-agnostic abstraction
│   │   ├── falah_loader.py             # Falah Arrow loader, integrity, & duplicate detection
│   │   ├── oasis1_validator.py         # NIfTI & Analyze 7.5 pair validator & safe extractor
│   │   ├── oasis1_metadata.py          # CSV/Excel metadata parser & ID matcher
│   │   ├── oasis1_label_mapper.py      # CDR clinical score mapping (Binary & Multiclass)
│   │   ├── oasis1_manifest.py          # Participant-level disjoint splitter (70/15/15)
│   │   ├── oasis1_loader.py            # OASIS-1 dataset loader & 2D/3D slice extractor
│   │   └── nifti_analyze_parser.py     # Pure Python binary reader for .nii, .nii.gz, .hdr/.img
│   │
│   ├── models/                         # DEEP LEARNING ARCHITECTURES
│   │   ├── __init__.py                 # Model factory (build_model)
│   │   ├── cnn.py                      # Custom 4-block Medical CNN (PyTorch)
│   │   ├── resnet.py                   # Medical ResNet-18 / ResNet-50 (PyTorch)
│   │   ├── vit.py                      # Vision Transformer with patch projections (PyTorch)
│   │   ├── cnn_model.py                # Custom CNN (TensorFlow / Keras)
│   │   ├── resnet_model.py             # ResNet-50 Transfer (TensorFlow / Keras)
│   │   └── vit_model.py                # Vision Transformer (TensorFlow / Keras)
│   │
│   ├── explainability/                 # VISUAL INTERPRETABILITY ENGINES
│   │   ├── gradcam.py                  # Dual PyTorch & Keras Grad-CAM engine
│   │   ├── vit_rollout.py              # ViT Attention Rollout engine (PyTorch)
│   │   └── vit_attention.py            # ViT Patch Saliency engine (TensorFlow)
│   │
│   ├── training/                       # TRAINING ENGINE & EXPERIMENT TRACKING
│   │   ├── metrics.py                  # Clinical metrics (Sens, Spec, BalAcc, F1, AUC)
│   │   └── trainer.py                  # UnifiedMRITrainer & ModelTrainer (PyTorch, TF, Sim)
│   │
│   ├── evaluation/                     # EVALUATION ENGINE
│   │   └── metrics.py                  # Bootstrap 95% CI, Confusion Matrix, ROC-AUC
│   │
│   ├── predict.py                      # NeuroVisionPredictor: multi-framework inference
│   │
│   └── api/                            # FASTAPI REST SERVICE
│       ├── __init__.py
│       ├── main.py                     # FastAPI app, CORS, lifespan, documentation
│       ├── routes.py                   # Health, model list, predict, explain, dataset routes
│       └── schemas.py                  # Pydantic V2 request & response schemas
│
├── data/                               # DATA DROPZONES (SEPARATE PROVENANCE)
│   ├── Falah_Alzheimer_MRI/            # Pre-downloaded Arrow dataset (unmodified)
│   └── oasis1/                         # OASIS-1 dropzone (raw/, extracted/, metadata/)
│
├── models/                             # SAVED CHECKPOINTS (.pt, .pth, .keras, .json)
├── experiments/                        # EXPERIMENT AUDIT LOGS (JSON)
│
└── tests/                              # UNIT & SYNTHETIC INTEGRATION TESTS
    ├── test_falah_loader.py            # Falah Arrow loading, splits, and duplicate tests
    ├── test_oasis1_integration.py      # OASIS-1 mock validation, matching, & zero-leakage tests
    └── test_api.py                     # FastAPI TestClient endpoint verification
```

---

## 📊 Cohort Provenance & Dataset Comparison

NeuroVision maintains strict architectural separation between cohorts to protect research integrity:

| Feature Dimension | Falah Alzheimer MRI | OASIS-1 Cross-Sectional |
| :--- | :--- | :--- |
| **Official Source** | Kaggle / Hugging Face re-host | Washington University in St. Louis / NIH ([oasisbrains.org](https://sites.wustl.edu/oasisbrains/oasis-1/)) |
| **Local Status** | **Locally Available** (`data/Falah_Alzheimer_MRI`) | **Manual Ingestion** (`data/oasis1/`) |
| **Subject Scope** | ~150 subjects (estimated) | **416 participants** (434 imaging sessions) |
| **Imaging Modality** | Pre-sliced 2D axial views | **Full 3D Volumetric T1 Structural MRI** |
| **Format** | Hugging Face Arrow shards | Analyze 7.5 (`.hdr`/`.img`) & NIfTI (`.nii`, `.nii.gz`) |
| **Total Slices / Scans** | 6,400 slices (5,120 train / 1,280 test) | 434 sessions / 416 unique subjects |
| **Participant Identifiers** | ❌ **Absent** | ✅ **Standardized** (`OAS1_xxxx`) |
| **Partitioning Strategy** | Random 2D slice split | **Strict Participant-Level Disjoint Partitioning** |
| **Patient Leakage Risk** | ⚠️ **High Risk** (axial slice leakage) | 🛡️ **Zero Patient Leakage** (mathematically verified) |
| **Clinical Labels** | 4 categorical stages | **Clinical Dementia Rating (CDR 0.0, 0.5, 1.0, 2.0)** |
| **Clinical Metadata** | ❌ None | ✅ Age, Sex, Educ, SES, MMSE, eTIV, nWBV, ASF |
| **Class Imbalance** | Severe (`Moderate_Demented` = 0.96%) | Mild to Moderate dementia over-represented in seniors |

> [!WARNING]
> **Slice Leakage Limitation in Falah:** Because original participant IDs are stripped in the Falah dataset, 2D slices from the same individual patient can appear in both the training and test splits. Results on Falah must **never** be cited as patient-independent clinical validation.

---

## 🔬 Clinical Dementia Rating (CDR) Mapping

OASIS-1 cognitive evaluations are structured around the validated Clinical Dementia Rating (CDR) scale. NeuroVision supports two standardized research tasks:

```
                      ┌───────────────────────────────────────────────┐
                      │              OASIS-1 Metadata Ingestion       │
                      │               CDR Assessment (0.0 to 2.0)     │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ Multiclass Task: 4 Stages ]                   [ Binary Task: 2 Classes ]
         ├── CDR 0.0: Non-Demented                       ├── CDR 0.0: Non-Demented
         ├── CDR 0.5: Very Mild Dementia                 └── CDR > 0: Demented
         ├── CDR 1.0: Mild Dementia                      
         ├── CDR 2.0: Moderate Dementia                  [ Control Flag: Age < 60 yrs ]
         └── Null/Blank: Unrated (<60 yrs controls)      └── Flagged as Unrated/Excluded
```

---

## 🛡️ Participant-Level Splitting & Leakage Prevention

In biomedical imaging, splitting scans at the image or slice level causes artificial performance inflation because correlated anatomical features of the same patient bleed across partitions. 

NeuroVision's [`OASIS1ManifestManager`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/src/data/oasis1_manifest.py) enforces strict participant-level partitioning:
1. **Disjoint Partitioning:** All scans from a unique `participant_id` (e.g. `OAS1_0001`) are grouped exclusively into either **Train (70%)**, **Validation (15%)**, or **Test (15%)**.
2. **Repeat Reliability Scans:** Longitudinal and reliability sessions (e.g. `OAS1_0001_MR1` and `OAS1_0001_MR2`) remain strictly within the same split.
3. **Standardized Manifest Specification:** Generates `oasis1_manifest.csv` containing all 21 clinical, structural, and audit fields:
   `participant_id`, `session_id`, `mri_id`, `file_format`, `image_path`, `header_path`, `split`, `age`, `sex`, `education`, `ses`, `mmse`, `cdr`, `etiv`, `nwbv`, `asf`, `research_label`, `research_class_idx`, `is_unrated_control`, `volume_shape`, `voxel_spacing`.

---

## 🧠 Supported Deep Learning Backbones

NeuroVision features a modular architecture library in both PyTorch and TensorFlow:

| Architecture | Paradigm | Parameter Count | Key Strength |
| :--- | :--- | :--- | :--- |
| **Custom Medical CNN** | Custom Deep CNN | ~1.2M | Fast, lightweight, LeakyReLU activations, spatial dropout, transparent feature maps. |
| **ResNet-18 Transfer** | Deep Residual | ~11.5M | Residual skip connections prevent vanishing gradients; pre-trained on ImageNet. |
| **ResNet-50 Transfer** | Deep Residual | ~24.5M | High capacity bottleneck residual blocks for subtle hippocampal and ventricular atrophy. |
| **Medical Vision Transformer** | Self-Attention (ViT) | ~9.8M | Non-overlapping $16 \times 16$ patch embeddings, multi-head self-attention, global spatial context. |
| **Simulation Baseline Engine** | Pure NumPy / SciPy | 0 | Zero-dependency baseline for rapid UI testing, CI testing, and non-GPU environments. |

---

## 🔍 Clinical Explainability Engines

Explainable AI is critical for clinical adoption to verify that models focus on relevant neuroanatomy (e.g. hippocampal atrophy, enlarged lateral ventricles) rather than image noise.

### 1. Grad-CAM (Convolutional Backbones)
Computes the gradients of the target class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

### 2. Vision Transformer Attention Rollout
Recursively propagates multi-head self-attention matrices across all transformer encoder blocks to trace global information flow from the `[CLS]` token to spatial patch tokens:

$$A_{\text{rollout}} = \prod_{l=1}^{L} \left(0.5 \cdot \bar{A}^{(l)} + 0.5 \cdot I\right)$$

---

## 🖥️ Interactive Streamlit Multi-Page Interface

The platform provides a 10-page research workbench:

| Page | Script | Core Functionality |
| :---: | :--- | :--- |
| **Home** | [`app.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/app.py) | Platform overview, quick MRI diagnosis upload, stage probability spectrum, and live Grad-CAM overlay. |
| **02** | [`pages/2_Dataset_Management.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/2_Dataset_Management.py) | Inspects Falah Arrow partitions and manages OASIS-1 directory scanning, CDR mapping, and manifest generation. |
| **03** | [`pages/3_MRI_Viewer.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/3_MRI_Viewer.py) | Interactive multi-planar 3D volume orthogonal slicer (Axial, Coronal, Sagittal) and 2D Falah slice preview. |
| **04** | [`pages/4_Image_Preprocessing.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/4_Image_Preprocessing.py) | Radiometric normalization (Min-Max, Z-score) and stochastic data augmentation (rotation, zoom, noise). |
| **05** | [`pages/5_Model_Training.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/5_Model_Training.py) | Experiment setup, framework selector (PyTorch/Keras/Simulation), live epoch loss/accuracy curves, and class weighting. |
| **06** | [`pages/6_Model_Evaluation.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/6_Model_Evaluation.py) | Independent test evaluation, Confusion Matrix, Macro-F1 with 95% Bootstrap CI, and Multi-Class ROC curves. |
| **07** | [`pages/7_Explainable_AI.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/7_Explainable_AI.py) | Detailed visual attribution, side-by-side raw heatmaps, anatomical overlays, and Jet/Magma colormaps. |
| **08** | [`pages/8_Model_Comparison.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/8_Model_Comparison.py) | Experiment leaderboard benchmarking architectural accuracy, validation loss, and training duration. |
| **09** | [`pages/9_Prediction.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/9_Prediction.py) | Dedicated clinical inference engine supporting custom file uploads and test cohort indexing. |
| **10** | [`pages/10_Research_Reports.py`](file:///c:/Users/Abo%20Sajid/data/Falah_Alzheimer_MRI/pages/10_Research_Reports.py) | Data provenance whitepaper generation and audit log export in Markdown/CSV. |

---

## ⚡ FastAPI REST Service & Documentation

NeuroVision includes a high-performance, asynchronous REST API powered by FastAPI.

### Interactive API Documentation
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### REST Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, active device (CPU/CUDA), and framework availability report. |
| `GET` | `/api/v1/models` | List all discovered checkpoints in `models/` with framework badges. |
| `GET` | `/api/v1/datasets` | List registered neuroimaging cohorts and capabilities. |
| `GET` | `/api/v1/datasets/{name}/summary` | Retrieve split counts, class distribution, and provenance notes. |
| `GET` | `/api/v1/datasets/oasis1/manifest` | Query or download the complete 21-attribute `oasis1_manifest.csv`. |
| `POST` | `/api/v1/predict` | Multipart image upload for cognitive stage inference and probability spectrum. |
| `POST` | `/api/v1/explain` | Multipart image upload returning Base64-encoded Grad-CAM or ViT rollout heatmaps. |

### Example API Usage (Python)

```python
import requests

# 1. Check Service Health
health = requests.get("http://localhost:8000/api/v1/health").json()
print("Health Status:", health["status"], "| Active Device:", health["active_device"])

# 2. Run Inference on a Brain MRI Slice
with open("sample_mri.png", "rb") as img_file:
    files = {"image": ("sample_mri.png", img_file, "image/png")}
    response = requests.post("http://localhost:8000/api/v1/predict", files=files)
    result = response.json()

print(f"Predicted Stage: {result['predicted_class']} ({result['confidence']}%)")
print("Probabilities:", result["probabilities"])
```

### Example API Usage (cURL)

```bash
# Diagnose an MRI Slice
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@sample_mri.png;type=image/png"
```

---

## 🐳 Docker & Containerized Deployment

Deploy the entire platform with one command using Docker Compose:

### 1. Launch with Docker Compose (API + Streamlit Dashboard)
```bash
docker compose up --build
```
- **Streamlit Web Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Standalone Container Build & Execution
```bash
# Build the Docker image
docker build -t neurovision:2.0 .

# Run the FastAPI REST server (Port 8000)
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models neurovision:2.0

# Or run the Streamlit Dashboard (Port 8501)
docker run -d -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
       neurovision:2.0 streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

---

## 📦 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/NeuroVision.git
cd NeuroVision
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Deep Learning Framework (PyTorch)
Choose the installation option suited for your hardware:

- **Option A: Fast CPU-Only Installation (Recommended for rapid testing, ~200MB):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **Option B: NVIDIA GPU Acceleration (CUDA 12.1+, ~2.5GB):**
  ```bash
  pip install torch torchvision
  ```

### 4. Install Application Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
- **Start Streamlit Dashboard:**
  ```bash
  streamlit run app.py
  ```
- **Start FastAPI REST Backend:**
  ```bash
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  ```

---

## 🧪 Testing & Verification

The repository includes a comprehensive synthetic test suite that verifies data loading, split isolation, patient leakage prevention, and API operations **without requiring large external downloads**:

```bash
# Run all unit and integration test suites:
python -m unittest discover tests/
```

Individual test suites:
```bash
# 1. Verify Falah Arrow dataset loading, splits, and duplicate detection:
python -m unittest tests/test_falah_loader.py

# 2. Verify OASIS-1 synthetic validation, CDR mapping, and zero patient leakage:
python -m unittest tests/test_oasis1_integration.py

# 3. Verify FastAPI endpoints (health, dataset summaries, prediction, explainability):
python -m unittest tests/test_api.py
```

---

## 🛠️ GitHub Repository & CI/CD

The project includes production-ready GitHub templates and workflows:

- **`.github/workflows/ci.yml`**: Continuous Integration matrix (Python 3.11 & 3.12) running test suites, verifying zero patient leakage on mock data, and validating Docker build integrity.
- **`.github/ISSUE_TEMPLATE/bug_report.md`**: Specialized bug tracking template capturing clinical component context.
- **`.github/ISSUE_TEMPLATE/feature_request.md`**: Feature request template with neuroimaging clinical rationale.
- **`.github/pull_request_template.md`**: Mandatory medical AI checklist enforcing participant-level splitting, test isolation, and clinical metric reporting.

---

## ⚠️ Medical & Ethical Prototype Disclaimer

> [!CAUTION]
> **MANDATORY MEDICAL RESEARCH NOTICE:**  
> **NeuroVision 2.0 is an academic and biomedical research software prototype.** It has **not** been cleared, approved, or evaluated by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory authority.  
> 
> - It is **NOT** a diagnostic medical device.
> - It must **NOT** be used as a primary diagnostic tool, clinical decision-making aid, or treatment planning instrument for human patients.
> - Visual explainability maps (Grad-CAM and ViT Attention Rollout) indicate model gradient sensitivity and **do not** prove biological neuropathology, neurofibrillary tangles, or histological tissue atrophy.
> - All diagnoses must be established through comprehensive clinical evaluations by certified medical professionals, neuroimaging radiologists, and cognitive neurologists.
