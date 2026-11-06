"""NeuroVision Training & Evaluation Package

Exports:
- compute_clinical_metrics
- UnifiedMRITrainer
- ModelTrainer
"""

from src.training.metrics import compute_clinical_metrics
from src.training.trainer import UnifiedMRITrainer, ModelTrainer

__all__ = ["compute_clinical_metrics", "UnifiedMRITrainer", "ModelTrainer"]
