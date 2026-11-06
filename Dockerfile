# ==============================================================================
# NeuroVision 2.0 - Production Biomedical AI Container
# Multi-Target: FastAPI REST API (Port 8000) & Streamlit Dashboard (Port 8501)
# ==============================================================================

FROM python:3.11-slim

# Maintainer / Metadata
LABEL maintainer="NeuroVision Research Team"
LABEL description="Brain MRI Cognitive Impairment Staging & Dataset Management Platform"
LABEL version="2.0.0"

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    API_PORT=8000

# Install system dependencies required for OpenCV, image processing, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Upgrade pip and packaging tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU first (compact ~180MB wheel, avoids downloading 2.5GB CUDA packages)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy dependency manifest and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create standard persistent data and model directory dropzones
RUN mkdir -p /app/data/Falah_Alzheimer_MRI \
    /app/data/oasis1/raw \
    /app/data/oasis1/extracted \
    /app/data/oasis1/metadata \
    /app/data/oasis1/processed \
    /app/models \
    /app/experiments

# Copy application codebase
COPY . /app/

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Health check probe against FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: Start FastAPI REST API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

