"""NeuroVision Unified Dataset Management Layer

Maintains separate provenance for Falah Alzheimer MRI and OASIS-1.
Guarantees datasets are never mixed silently.
Includes graceful fallbacks when native TensorFlow C++ runtime is unavailable.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Safe lazy import for TensorFlow to prevent Windows DLL load failure
try:
    import tensorflow as tf
except Exception:
    tf = None

from src.config.settings import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VAL_SPLIT_RATIO,
    DEFAULT_BATCH_SIZE,
    DISCLAIMER_TEXT,
)
from src.data.falah_loader import FalahDatasetLoader
try:
    from src.data.oasis1_loader import OASIS1DatasetLoader, Oasis1DatasetLoader
except ImportError:
    from src.data.oasis1_loader import OASIS1DatasetLoader
    Oasis1DatasetLoader = OASIS1DatasetLoader


class DatasetManager:
    """Unified Dataset Management layer for NeuroVision.
    
    Maintains separate provenance for Falah Alzheimer MRI and OASIS-1.
    Guarantees datasets are never mixed silently.
    """

    SUPPORTED_DATASETS = ["Falah_Alzheimer_MRI", "OASIS-1"]

    def __init__(
        self,
        default_dataset: str = "Falah_Alzheimer_MRI",
        oasis_task: str = "multiclass",
        seed: int = DEFAULT_RANDOM_SEED,
    ):
        if default_dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {default_dataset}. Supported: {self.SUPPORTED_DATASETS}")

        self.active_dataset_name = default_dataset
        self.oasis_task = oasis_task
        self.seed = seed

        self.falah_loader: Optional[FalahDatasetLoader] = None
        self.oasis1_loader: Optional[OASIS1DatasetLoader] = None

        self._init_active_loader()

    def set_active_dataset(self, dataset_name: str, oasis_task: Optional[str] = None):
        """Switches the active dataset."""
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {self.SUPPORTED_DATASETS}")

        self.active_dataset_name = dataset_name
        if oasis_task:
            self.oasis_task = oasis_task
        self._init_active_loader()

    def _init_active_loader(self):
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            if self.falah_loader is None:
                self.falah_loader = FalahDatasetLoader(seed=self.seed)
        elif self.active_dataset_name == "OASIS-1":
            if self.oasis1_loader is None or getattr(self.oasis1_loader, "task", None) != self.oasis_task:
                self.oasis1_loader = OASIS1DatasetLoader(task=self.oasis_task, split_seed=self.seed)

    def get_classes(self) -> List[str]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            return self.falah_loader.class_names
        else:
            return self.oasis1_loader.label_mapper.target_classes

    def get_summary(self) -> Dict[str, Any]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            summary = self.falah_loader.get_summary()
            summary["disclaimer"] = DISCLAIMER_TEXT
            return summary
        else:
            summary = self.oasis1_loader.get_status()
            summary["disclaimer"] = DISCLAIMER_TEXT
            summary["research_task"] = self.oasis_task
            return summary

    def get_split_counts(self) -> Dict[str, int]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            return self.falah_loader.get_split_counts()
        else:
            if hasattr(self.oasis1_loader, "manifest_df") and self.oasis1_loader.manifest_df is not None and not self.oasis1_loader.manifest_df.empty:
                counts = self.oasis1_loader.manifest_df["split"].value_counts().to_dict()
                counts["total"] = len(self.oasis1_loader.manifest_df)
                return counts
            return {"train": 0, "val": 0, "test": 0, "total": 0}

    def get_class_weights(self) -> Dict[int, float]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            weights_dict = self.falah_loader.compute_class_weights()
            return {i: weights_dict.get(name, 1.0) for i, name in enumerate(self.falah_loader.class_names)}
        else:
            return {i: 1.0 for i in range(len(self.oasis1_loader.label_mapper.target_classes))}

    def get_sample_images(self, split: str = "train", count: int = 8) -> List[Dict[str, Any]]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            return self.falah_loader.get_sample_images(split=split, count=count)
        else:
            return []

    def get_numpy_data(
        self,
        split: str,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        channels: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            return self.falah_loader.to_numpy_arrays(split=split, target_size=target_size, channels=channels)
        else:
            return np.empty((0, target_size[0], target_size[1], channels)), np.empty((0,))

    def get_torch_dataloader(
        self,
        split: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        channels: int = 1,
        shuffle: bool = False,
    ):
        """Builds a PyTorch DataLoader for the active dataset without requiring TensorFlow."""
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
        except Exception:
            raise RuntimeError("PyTorch runtime is not available on this system. Install torch with: pip install torch")

        x, y = self.get_numpy_data(split=split, target_size=target_size, channels=channels)
        if len(x) == 0:
            return None

        # Permute (N, H, W, C) -> (N, C, H, W) for PyTorch Conv2D / ViT
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_tf_dataset(
        self,
        split: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        channels: int = 1,
        shuffle: bool = False,
        augment: bool = False,
    ):
        if tf is None:
            raise RuntimeError(
                "TensorFlow runtime is not available on this system (DLL initialization failed). "
                "Use get_numpy_data() or PyTorch DataLoaders instead."
            )
        if self.active_dataset_name == "Falah_Alzheimer_MRI":
            return self.falah_loader.to_tf_dataset(
                split=split,
                batch_size=batch_size,
                target_size=target_size,
                channels=channels,
                shuffle=shuffle,
                augment=augment,
            )
        else:
            x, y = self.get_numpy_data(split=split, target_size=target_size, channels=channels)
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=self.seed)
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds
