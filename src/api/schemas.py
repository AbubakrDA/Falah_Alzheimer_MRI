"""NeuroVision API Pydantic Schemas

Defines request and response data models for the NeuroVision REST API.
Includes biomedical diagnosis results, explainability heatmaps, dataset summaries,
and model metadata.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System and environment runtime health status."""
    status: str = Field("healthy", description="Service health state")
    service: str = Field("NeuroVision 2.0 Biomedical AI API")
    version: str = Field("2.0.0", description="API semantic version")
    pytorch_available: bool = Field(..., description="Whether PyTorch is loaded and operational")
    tensorflow_available: bool = Field(..., description="Whether TensorFlow C++ runtime is operational")
    active_device: str = Field(..., description="Active compute device (e.g. cpu, cuda:0)")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class DatasetSummaryResponse(BaseModel):
    """Structured dataset provenance and metadata summary."""
    dataset_name: str
    total_samples: int
    split_counts: Dict[str, int]
    class_names: List[str]
    provenance_limitation: str
    class_distributions: Optional[Dict[str, Any]] = None
    native_dimensions: Optional[List[int]] = None


class ModelCheckpointInfo(BaseModel):
    """Metadata regarding a saved deep learning checkpoint."""
    name: str
    filename: str
    framework: str = Field(..., description="PyTorch, Keras, or Simulation")
    relative_path: str
    size_bytes: int


class ModelsListResponse(BaseModel):
    """List of all available model checkpoints and supported backbones."""
    total_checkpoints: int
    checkpoints: List[ModelCheckpointInfo]
    supported_architectures: List[str] = [
        "Custom Medical CNN",
        "ResNet-18 Transfer Learning",
        "ResNet-50 Transfer Learning",
        "Medical Vision Transformer (ViT)",
        "Simulation Baseline Engine",
    ]


class PredictionResult(BaseModel):
    """Clinical staging and probability output from an inference request."""
    predicted_class: str = Field(..., description="Predicted Alzheimer's cognitive stage")
    predicted_index: int = Field(..., description="Numerical class index")
    confidence: float = Field(..., description="Confidence percentage (0.0 to 100.0%)")
    probabilities: Dict[str, float] = Field(..., description="Full stage probability distribution")
    model_name: str = Field(..., description="Identifier of the model used for inference")
    inference_time_ms: float = Field(..., description="Elapsed inference time in milliseconds")
    disclaimer: str = Field(..., description="Mandatory medical research disclaimer")


class ExplainabilityResult(BaseModel):
    """Visual attention attribution output including Grad-CAM or ViT rollout heatmaps."""
    predicted_class: str
    confidence: float
    method: str = Field(..., description="Grad-CAM, ViT Attention Rollout, or Simulation Saliency")
    heatmap_base64: str = Field(..., description="Base64-encoded PNG of the 2D raw attention heatmap")
    overlay_base64: str = Field(..., description="Base64-encoded PNG of the RGB anatomical overlay")
    probabilities: Dict[str, float]
    disclaimer: str

