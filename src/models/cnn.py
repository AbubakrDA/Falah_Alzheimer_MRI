"""NeuroVision Custom Medical CNN Architecture

A deep convolutional neural network designed specifically for 2D brain MRI slices.
Includes Conv2D blocks with Batch Normalization, LeakyReLU, MaxPool, and Spatial Dropout.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMedicalCNN(nn.Module):
    """Custom multi-scale CNN for Alzheimer's MRI stage classification."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4 - Target layer for Grad-CAM
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = self.pool1(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.pool2(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = self.pool3(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        x = self.pool4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM."""
        return self.conv4

