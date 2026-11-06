"""NeuroVision Configuration Package

Self-contained module exporting all configuration constants, paths, and hyperparameters.
"""

import os
import sys
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
REPORTS_DIR = BASE_DIR / "reports"
LOCAL_KERAS_DIR = BASE_DIR / ".keras"

for directory in [MODELS_DIR, OUTPUTS_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LOCAL_KERAS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment configuration to guarantee clean sandbox execution
os.environ["USERPROFILE"] = str(BASE_DIR)
os.environ["HOME"] = str(BASE_DIR)
os.environ["KERAS_HOME"] = str(LOCAL_KERAS_DIR)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Dataset paths
if (DATA_DIR / "Falah_Alzheimer_MRI" / "dataset_dict.json").exists():
    FALAH_DATASET_PATH = DATA_DIR / "Falah_Alzheimer_MRI"
elif (BASE_DIR / "dataset_dict.json").exists():
    FALAH_DATASET_PATH = BASE_DIR
else:
    FALAH_DATASET_PATH = BASE_DIR / "Falah_Alzheimer_MRI"

DEFAULT_DATASET_PATH = FALAH_DATASET_PATH

# OASIS-1 directory dropzones
OASIS1_BASE_DIR = DATA_DIR / "oasis1"
OASIS1_RAW_DIR = OASIS1_BASE_DIR / "raw"
OASIS1_EXTRACTED_DIR = OASIS1_BASE_DIR / "extracted"
OASIS1_METADATA_DIR = OASIS1_BASE_DIR / "metadata"
OASIS1_PROCESSED_DIR = OASIS1_BASE_DIR / "processed"

for directory in [OASIS1_RAW_DIR, OASIS1_EXTRACTED_DIR, OASIS1_METADATA_DIR, OASIS1_PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Class mappings for Falah & Multi-class research
FALAH_CLASSES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]
CLASSES = FALAH_CLASSES
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# Random seeds
DEFAULT_RANDOM_SEED = 42
SEED = DEFAULT_RANDOM_SEED

# Image defaults & normalization statistics
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGE_SIZE = DEFAULT_IMAGE_SIZE
DEFAULT_CHANNELS = 3
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Splitting defaults
DEFAULT_VAL_SPLIT_RATIO = 0.15
DEFAULT_TEST_SPLIT_RATIO = 0.15
VAL_SPLIT = DEFAULT_VAL_SPLIT_RATIO

# Training defaults
DEFAULT_BATCH_SIZE = 32
BATCH_SIZE = DEFAULT_BATCH_SIZE
DEFAULT_LEARNING_RATE = 1e-4
LEARNING_RATE = DEFAULT_LEARNING_RATE
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 15
EPOCHS = DEFAULT_EPOCHS
DEFAULT_BACKBONE = "resnet18"
NUM_WORKERS = 0

# Safe device detection (PyTorch or fallback)
DEVICE = "cpu"
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu"

DISCLAIMER_TEXT = (
    "NeuroVision is a biomedical AI research prototype developed for academic and translational exploration. "
    "It is NOT a clinically validated medical device and MUST NOT be used for direct clinical diagnosis, patient management, "
    "or medical decision-making."
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "EXPERIMENTS_DIR",
    "REPORTS_DIR",
    "LOCAL_KERAS_DIR",
    "FALAH_DATASET_PATH",
    "DEFAULT_DATASET_PATH",
    "OASIS1_BASE_DIR",
    "OASIS1_RAW_DIR",
    "OASIS1_EXTRACTED_DIR",
    "OASIS1_METADATA_DIR",
    "OASIS1_PROCESSED_DIR",
    "FALAH_CLASSES",
    "CLASSES",
    "NUM_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "DEFAULT_RANDOM_SEED",
    "SEED",
    "DEFAULT_IMAGE_SIZE",
    "IMAGE_SIZE",
    "DEFAULT_CHANNELS",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "DEFAULT_VAL_SPLIT_RATIO",
    "DEFAULT_TEST_SPLIT_RATIO",
    "VAL_SPLIT",
    "DEFAULT_BATCH_SIZE",
    "BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "DEFAULT_EPOCHS",
    "EPOCHS",
    "DEFAULT_BACKBONE",
    "NUM_WORKERS",
    "DEVICE",
    "DISCLAIMER_TEXT",
]
