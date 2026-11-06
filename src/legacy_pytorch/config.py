"""NeuroVision 2.0 Configuration Module

Defines configuration settings, hyperparameters, file paths, and class labels
for Alzheimer's MRI classification.
"""

from pathlib import Path
import torch

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Dataset paths
DEFAULT_DATASET_PATH = BASE_DIR if (BASE_DIR / "dataset_dict.json").exists() else BASE_DIR / "Falah_Alzheimer_MRI"

# Class definitions (matches Falah/Alzheimer_MRI metadata)
CLASSES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# Image specifications
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
VAL_SPLIT = 0.15
SEED = 42
DEFAULT_BACKBONE = "efficientnet_b0"

# Compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Safe default on Windows to avoid multiprocessing issues
