"""NeuroVision 2.0 Model Architectures

Defines deep learning architectures for Alzheimer MRI stage classification.
Supports EfficientNet, ResNet, and ConvNeXt transfer learning backbones
with custom clinical classification heads.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, DEFAULT_BACKBONE


class NeuroVisionClassifier(nn.Module):
    """Deep learning classifier for Alzheimer's MRI multi-class diagnosis."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        # Load backbone
        if self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_layer = self.backbone.features[-1]

        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_layer = self.backbone.layer4[-1]

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_layer = self.backbone.layer4[-1]

        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
            self.target_layer = self.backbone.features[-1]

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Choose from: 'efficientnet_b0', 'resnet18', 'resnet50', 'convnext_tiny'."
            )

        # Freeze backbone parameters if requested (useful for initial warm-up)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom medical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def unfreeze_backbone(self) -> None:
        """Unfreezes all backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the final convolutional feature layer for Grad-CAM visualization."""
        return self.target_layer


def build_model(
    model_name: str = DEFAULT_BACKBONE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
) -> NeuroVisionClassifier:
    """Factory helper to build a NeuroVisionClassifier instance."""
    return NeuroVisionClassifier(
        backbone_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )
