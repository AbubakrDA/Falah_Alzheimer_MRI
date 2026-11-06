"""NeuroVision Falah Alzheimer MRI Dataset Loader

Loads and validates the locally stored Falah/Alzheimer_MRI Hugging Face Arrow dataset.
Handles image integrity verification, train/validation/test splitting,
duplicate/near-duplicate detection between splits, class weight calculation,
and export to both PyTorch and TensorFlow pipeline formats.

PROVENANCE & METHODOLOGICAL WARNING:
This dataset contains 2D axial MRI slices without participant/subject identifiers.
Do NOT claim participant-level independence for this dataset: slice-level shuffling
between train and test splits likely leads to patient data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
import hashlib

import numpy as np
from PIL import Image
from datasets import load_from_disk, Dataset, DatasetDict


# Canonical class names in Falah/Alzheimer_MRI
FALAH_CLASSES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]


def resolve_falah_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Intelligently locates the Falah dataset directory across various environments."""
    if custom_path is not None:
        p = Path(custom_path)
        if (p / "dataset_dict.json").exists():
            return p
        if (p / "data" / "Falah_Alzheimer_MRI" / "dataset_dict.json").exists():
            return p / "data" / "Falah_Alzheimer_MRI"

    # Common location candidates
    candidates = [
        Path("data/Falah_Alzheimer_MRI"),
        Path("Falah_Alzheimer_MRI"),
        Path(__file__).resolve().parent.parent.parent,  # Root directory if dataset is at root
        Path(__file__).resolve().parent.parent.parent / "data" / "Falah_Alzheimer_MRI",
    ]

    for cand in candidates:
        if (cand / "dataset_dict.json").exists():
            return cand.resolve()

    # Fallback to current working directory if dataset_dict.json is present
    cwd = Path.cwd()
    if (cwd / "dataset_dict.json").exists():
        return cwd.resolve()
    if (cwd / "data" / "Falah_Alzheimer_MRI" / "dataset_dict.json").exists():
        return (cwd / "data" / "Falah_Alzheimer_MRI").resolve()

    raise FileNotFoundError(
        "Could not locate Falah Alzheimer MRI dataset. Expected 'dataset_dict.json' in "
        "data/Falah_Alzheimer_MRI or project root."
    )


class FalahDatasetLoader:
    """Robust loader and auditor for the Falah Alzheimer MRI Arrow dataset."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (224, 224),
        val_split: float = 0.15,
        color_mode: str = "RGB",
        seed: int = 42,
    ):
        self.dataset_path = resolve_falah_path(dataset_path)
        self.target_size = target_size
        self.val_split = val_split
        self.color_mode = color_mode.upper()
        self.seed = seed

        self.dataset_dict: Optional[DatasetDict] = None
        self.raw_train: Optional[Dataset] = None
        self.raw_test: Optional[Dataset] = None
        self.class_names: List[str] = FALAH_CLASSES

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Loads Hugging Face Arrow tables from disk without altering original files."""
        self.dataset_dict = load_from_disk(str(self.dataset_path))
        self.raw_train = self.dataset_dict["train"]
        self.raw_test = self.dataset_dict["test"]

        # Extract label names if present in Arrow schema
        label_feature = self.raw_train.features.get("label")
        if hasattr(label_feature, "names") and label_feature.names:
            self.class_names = list(label_feature.names)
        else:
            self.class_names = FALAH_CLASSES

    def get_dataset_info(self) -> Dict[str, Any]:
        """Returns structured metadata, split counts, and class distributions."""
        train_labels = [int(x) for x in self.raw_train["label"]]
        test_labels = [int(x) for x in self.raw_test["label"]]

        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)

        # Inspect first image for native dimensions and mode
        sample_img = self.raw_train[0]["image"]

        info = {
            "dataset_name": "Falah_Alzheimer_MRI",
            "dataset_path": str(self.dataset_path),
            "format": "Hugging Face Arrow",
            "total_samples": len(self.raw_train) + len(self.raw_test),
            "train_samples": len(self.raw_train),
            "test_samples": len(self.raw_test),
            "native_image_size": sample_img.size,
            "native_image_mode": sample_img.mode,
            "target_image_size": self.target_size,
            "color_mode": self.color_mode,
            "class_names": self.class_names,
            "train_distribution": {self.class_names[k]: train_counts[k] for k in sorted(train_counts)},
            "test_distribution": {self.class_names[k]: test_counts[k] for k in sorted(test_counts)},
            "total_distribution": {
                self.class_names[k]: train_counts[k] + test_counts.get(k, 0)
                for k in range(len(self.class_names))
            },
        }
        return info

    def check_image_integrity(self, max_check: Optional[int] = None) -> Dict[str, Any]:
        """Verifies that all images can be successfully read and converted without corruption."""
        corrupted = []
        checked_count = 0

        for split_name, ds in [("train", self.raw_train), ("test", self.raw_test)]:
            limit = min(len(ds), max_check) if max_check else len(ds)
            for idx in range(limit):
                checked_count += 1
                try:
                    img = ds[idx]["image"]
                    if not isinstance(img, Image.Image):
                        corrupted.append((split_name, idx, "Not a valid PIL Image"))
                    else:
                        img.verify()  # Verify integrity
                except Exception as e:
                    corrupted.append((split_name, idx, str(e)))

        return {
            "total_checked": checked_count,
            "corrupted_count": len(corrupted),
            "is_valid": len(corrupted) == 0,
            "corrupted_records": corrupted,
        }

    def detect_duplicates(self, max_samples: Optional[int] = 500) -> Dict[str, Any]:
        """Investigates exact and perceptual near-duplicate images between train and test splits.
        
        Uses MD5 hashing for byte-level duplicates and a 64-bit difference hash (dHash)
        for perceptual near-duplicate slice detection.
        """
        def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
            """Computes a difference hash (dHash) for visual perceptual matching."""
            resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized)
            difference = pixels[:, 1:] > pixels[:, :-1]
            return sum([2 ** i for (i, v) in enumerate(difference.flatten()) if v])

        train_limit = min(len(self.raw_train), max_samples) if max_samples else len(self.raw_train)
        test_limit = min(len(self.raw_test), max_samples) if max_samples else len(self.raw_test)

        train_md5_map: Dict[str, List[int]] = {}
        train_dhash_map: Dict[int, List[int]] = {}

        # Hash training images
        for i in range(train_limit):
            img = self.raw_train[i]["image"]
            md5_hash = hashlib.md5(img.tobytes()).hexdigest()
            dh = compute_dhash(img)

            train_md5_map.setdefault(md5_hash, []).append(i)
            train_dhash_map.setdefault(dh, []).append(i)

        exact_cross_duplicates = []
        near_cross_duplicates = []

        # Compare test images against training hashes
        for j in range(test_limit):
            img = self.raw_test[j]["image"]
            md5_hash = hashlib.md5(img.tobytes()).hexdigest()
            dh = compute_dhash(img)

            if md5_hash in train_md5_map:
                exact_cross_duplicates.append({
                    "test_index": j,
                    "train_indices": train_md5_map[md5_hash],
                    "hash": md5_hash,
                })

            if dh in train_dhash_map:
                near_cross_duplicates.append({
                    "test_index": j,
                    "train_indices": train_dhash_map[dh],
                    "dhash": hex(dh),
                })

        return {
            "samples_checked_train": train_limit,
            "samples_checked_test": test_limit,
            "exact_duplicate_count": len(exact_cross_duplicates),
            "near_duplicate_count": len(near_cross_duplicates),
            "exact_cross_duplicates": exact_cross_duplicates[:10],
            "near_cross_duplicates": near_cross_duplicates[:10],
            "methodological_note": (
                "High near-duplicate similarity between train and test splits indicates "
                "that 2D slices from the same original subject/scan are present in both splits. "
                "This patient leakage occurs because original participant IDs are absent."
            ),
        }

    def compute_class_weights(self) -> Dict[str, float]:
        """Computes inverse-frequency balanced class weights for loss functions.
        
        Formula: total_samples / (num_classes * class_count)
        """
        train_labels = [int(x) for x in self.raw_train["label"]]
        counts = Counter(train_labels)
        total = len(train_labels)
        n_classes = len(self.class_names)

        weights = {}
        for c_idx, c_name in enumerate(self.class_names):
            cnt = counts.get(c_idx, 1)
            weights[c_name] = total / (n_classes * cnt)

        return weights

    def get_splits(
        self,
        val_split: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Splits training data into train and validation sets, keeping test data untouched.
        
        Returns:
            (train_subset, val_subset, test_set)
        """
        split_ratio = val_split if val_split is not None else self.val_split
        s = seed if seed is not None else self.seed

        if split_ratio <= 0.0 or split_ratio >= 1.0:
            return self.raw_train, None, self.raw_test

        total_train = len(self.raw_train)
        val_size = int(total_train * split_ratio)
        train_size = total_train - val_size

        rng = np.random.RandomState(s)
        indices = rng.permutation(total_train)

        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()

        train_subset = self.raw_train.select(train_indices)
        val_subset = self.raw_train.select(val_indices)

        return train_subset, val_subset, self.raw_test

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """Converts PIL image to target resolution and normalized numpy array [0, 1]."""
        img = pil_image.convert(self.color_mode)
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.color_mode == "L" and arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        return arr

    def generate_summary_report(self) -> str:
        """Generates a comprehensive scientific summary report of the Falah dataset."""
        info = self.get_dataset_info()
        weights = self.compute_class_weights()

        report = [
            "==================================================",
            "   NEUROVISION: FALAH ALZHEIMER MRI AUDIT REPORT   ",
            "==================================================",
            f"Dataset Path:       {info['dataset_path']}",
            f"Storage Format:     {info['format']}",
            f"Total Slices:       {info['total_samples']} images",
            f"  - Training Split: {info['train_samples']} images (80%)",
            f"  - Testing Split:  {info['test_samples']} images (20%)",
            f"Native Resolution:  {info['native_image_size']} ({info['native_image_mode']})",
            f"Target Resolution:  {info['target_image_size']} ({info['color_mode']})",
            "",
            "CLASS DISTRIBUTION BREAKDOWN:",
            f"{'Class Name':<22} | {'Train':<7} | {'Test':<7} | {'Total':<7} | {'% Dataset':<10} | {'Class Weight':<12}",
            "-" * 78,
        ]

        total_samples = info["total_samples"]
        for c_name in info["class_names"]:
            tr = info["train_distribution"].get(c_name, 0)
            te = info["test_distribution"].get(c_name, 0)
            tot = tr + te
            pct = (tot / total_samples) * 100
            wt = weights.get(c_name, 1.0)
            report.append(f"{c_name:<22} | {tr:<7} | {te:<7} | {tot:<7} | {pct:>6.2f}%    | {wt:>10.4f}")

        report.extend([
            "-" * 78,
            "",
            "CRITICAL METHODOLOGICAL NOTES & LIMITATIONS:",
            "1. Participant Independence: CANNOT BE CLAIMED.",
            "   The original dataset does not provide participant or session IDs.",
            "   Pre-slicing into 2D axial views without subject-level partitioning",
            "   leads to axial slice leakage across train and test splits.",
            "2. Extreme Class Imbalance:",
            "   'Moderate_Demented' constitutes only 1.0% of the dataset (64 images total).",
            "   Weighted cross-entropy loss or focal loss is mandatory.",
            "3. Clinical Status:",
            "   This is a synthetic 2D research dataset derived from Kaggle re-hosting.",
            "   Do not use for primary clinical diagnosis without 3D volumetric validation.",
            "==================================================",
        ])

        return "\n".join(report)

    @property
    def raw_dataset(self) -> Optional[DatasetDict]:
        """Provides direct access to the underlying Hugging Face DatasetDict."""
        return self.dataset_dict

    @raw_dataset.setter
    def raw_dataset(self, value: Optional[DatasetDict]) -> None:
        self.dataset_dict = value

    def get_split_counts(self) -> Dict[str, int]:
        """Returns sample counts for train, validation, and test splits."""
        total_train = len(self.raw_train)
        val_size = int(total_train * self.val_split)
        train_size = total_train - val_size
        return {
            "train": train_size,
            "val": val_size,
            "test": len(self.raw_test),
            "total": total_train + len(self.raw_test),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Provides structured summary for Streamlit dataset management interfaces."""
        info = self.get_dataset_info()
        counts = self.get_split_counts()
        return {
            "total_samples": info["total_samples"],
            "split_counts": counts,
            "class_distributions": {
                c: {
                    "Train": info["train_distribution"].get(c, 0),
                    "Test": info["test_distribution"].get(c, 0),
                    "Total": info["total_distribution"].get(c, 0),
                }
                for c in self.class_names
            },
            "provenance_limitation": (
                "Participant-level independence cannot be claimed for Falah. "
                "The dataset lacks participant IDs and slice-level shuffling creates axial slice leakage."
            ),
            "native_size": info["native_image_size"],
            "image_size": info["native_image_size"],
            "native_mode": info["native_image_mode"],
        }

    def get_sample_images(
        self,
        split: str = "train",
        count: int = 8,
        class_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves a sample batch of PIL images with labels for preview."""
        if split == "test":
            ds = self.raw_test
        elif split == "val":
            _, val_sub, _ = self.get_splits()
            ds = val_sub
        else:
            train_sub, _, _ = self.get_splits()
            ds = train_sub

        samples = []
        target_idx = None
        if class_filter and class_filter != "All" and class_filter in self.class_names:
            target_idx = self.class_names.index(class_filter)

        for i in range(len(ds)):
            item = ds[i]
            lbl = int(item["label"])
            if target_idx is not None and lbl != target_idx:
                continue

            samples.append({
                "index": i,
                "image": item["image"],
                "label": lbl,
                "class_name": self.class_names[lbl],
            })
            if len(samples) >= count:
                break

        return samples

    def to_numpy_arrays(
        self,
        split: str = "train",
        target_size: Tuple[int, int] = (128, 128),
        channels: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts a split into numpy arrays (X, y) without requiring TensorFlow."""
        if split == "test":
            ds = self.raw_test
        elif split == "val":
            _, val_sub, _ = self.get_splits()
            ds = val_sub
        else:
            train_sub, _, _ = self.get_splits()
            ds = train_sub

        x_list = []
        y_list = []
        mode = "RGB" if channels == 3 else "L"

        for item in ds:
            img = item["image"].convert(mode).resize(target_size, Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if channels == 1 and arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            x_list.append(arr)
            y_list.append(int(item["label"]))

        return np.array(x_list), np.array(y_list)

    def to_tf_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        target_size: Tuple[int, int] = (128, 128),
        channels: int = 1,
        shuffle: bool = False,
        augment: bool = False,
    ):
        """Converts split to a tf.data.Dataset (requires working TensorFlow)."""
        try:
            import tensorflow as tf
        except Exception:
            raise RuntimeError("TensorFlow is not available on this platform (DLL initialization failed).")

        x, y = self.to_numpy_arrays(split=split, target_size=target_size, channels=channels)
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=self.seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
