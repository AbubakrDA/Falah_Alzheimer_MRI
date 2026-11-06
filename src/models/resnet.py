"""NeuroVision ResNet Architecture

Provides standard and pre-trained ResNet backbones (ResNet-18, ResNet-50)
with clinical classification heads adaptable to Binary (2-class) and Multiclass (4-class) tasks.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class MedicalResNet(nn.Module):
    """ResNet architecture adapted for brain MRI cognitive impairment staging."""

    def __init__(
        self,
        backbone_type: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.num_classes = num_classes

        if self.backbone_type == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.target_layer = self.model.layer4[-1]
        elif self.backbone_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.target_layer = self.model.layer4[-1]
        else:
            raise ValueError(f"Unsupported ResNet type: {backbone_type}. Choose 'resnet18' or 'resnet50'.")

        # Replace final fully connected layer with clinical classification head
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM."""
        return self.target_layer

