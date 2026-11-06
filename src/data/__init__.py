"""NeuroVision Data Package

Contains loaders, validators, label mappers, and manifest managers for:
- Falah Alzheimer MRI (2D Arrow slice dataset)
- OASIS-1 (3D/2D NIfTI, Analyze 7.5, and Clinical Metadata)
"""

from src.data.falah_loader import FalahDatasetLoader, FALAH_CLASSES, resolve_falah_path
from src.data.oasis1_validator import OASIS1Validator, Oasis1Validator
from src.data.oasis1_metadata import OASIS1MetadataManager, Oasis1MetadataManager, extract_ids_from_session_string
from src.data.oasis1_label_mapper import OASIS1LabelMapper, Oasis1LabelMapper, BINARY_CLASSES, MULTICLASS_CLASSES
from src.data.oasis1_manifest import OASIS1ManifestManager, Oasis1ManifestManager, Oasis1ManifestBuilder
from src.data.oasis1_loader import OASIS1DatasetLoader, Oasis1DatasetLoader
from src.data.dataset_manager import DatasetManager

__all__ = [
    "FalahDatasetLoader",
    "FALAH_CLASSES",
    "resolve_falah_path",
    "OASIS1Validator",
    "OASIS1MetadataManager",
    "extract_ids_from_session_string",
    "OASIS1LabelMapper",
    "BINARY_CLASSES",
    "MULTICLASS_CLASSES",
    "OASIS1ManifestManager",
    "OASIS1DatasetLoader",
    "Oasis1Validator",
    "Oasis1MetadataManager",
    "Oasis1LabelMapper",
    "Oasis1ManifestManager",
    "Oasis1ManifestBuilder",
    "Oasis1DatasetLoader",
    "DatasetManager",
]
