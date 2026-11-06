"""NeuroVision Models Package

Exports:
- CustomMedicalCNN
- MedicalResNet
- MedicalVisionTransformer
- build_model factory helper
"""

from __future__ import annotations
from typing import Union, Any, Optional

try:
    import torch
    import torch.nn as nn
    from src.models.cnn import CustomMedicalCNN
    from src.models.resnet import MedicalResNet
    from src.models.vit import MedicalVisionTransformer
except Exception:
    torch = None
    nn = None
    CustomMedicalCNN = None
    MedicalResNet = None
    MedicalVisionTransformer = None


def build_model(
    model_type: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> Any:
    """Builds and returns a deep learning architecture for Alzheimer's diagnosis."""
    if torch is None or CustomMedicalCNN is None:
        raise RuntimeError("PyTorch is not available. Please install torch and torchvision.")

    m_type = model_type.lower()

    if m_type in ["cnn", "custom_cnn", "medical_cnn"]:
        return CustomMedicalCNN(num_classes=num_classes, dropout_rate=dropout_rate)

    elif m_type in ["resnet18", "resnet50"]:
        return MedicalResNet(
            backbone_type=m_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

    elif m_type in ["vit", "vision_transformer"]:
        return MedicalVisionTransformer(num_classes=num_classes, dropout=dropout_rate)

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose from 'cnn', 'resnet18', 'resnet50', 'vit'."
        )


__all__ = [
    "CustomMedicalCNN",
    "MedicalResNet",
    "MedicalVisionTransformer",
    "build_model",
]
