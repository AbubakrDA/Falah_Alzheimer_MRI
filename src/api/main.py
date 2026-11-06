"""NeuroVision 2.0 FastAPI Application Entrypoint

Initializes the FastAPI REST service with:
- OpenAPI / Swagger documentation
- CORS configuration for frontend and dashboard integration
- Health probes, dataset inspection, inference, and explainability endpoints
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.config.settings import MODELS_DIR
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensures dropzone and model directories exist upon server startup."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/oasis1/raw").mkdir(parents=True, exist_ok=True)
    Path("data/oasis1/extracted").mkdir(parents=True, exist_ok=True)
    Path("data/oasis1/metadata").mkdir(parents=True, exist_ok=True)
    Path("data/oasis1/processed").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="NeuroVision 2.0 Biomedical AI API",
    description="""
## Advanced Brain MRI Alzheimer's Staging & Provenance-Aware Neuroimaging Platform

NeuroVision provides clinical-grade deep learning classification for cognitive impairment staging:
- **Datasets:** Falah Alzheimer MRI (6,400 2D slices) & OASIS-1 (3D T1 NIfTI/Analyze 7.5 with CDR ground truth)
- **Deep Learning Architectures:** Custom Medical CNN, ResNet-18, ResNet-50, and Vision Transformer (ViT)
- **Explainability:** Grad-CAM convolutional overlays and ViT Attention Rollout spatial maps
- **Zero-Patient-Leakage:** Strict participant-level disjoint partitioning on volumetric cohorts

### Interactive Documentation:
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Health Probe:** `/api/v1/health`
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    """Redirects root visitors to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

