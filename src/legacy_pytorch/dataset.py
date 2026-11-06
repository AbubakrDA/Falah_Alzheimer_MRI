"""NeuroVision 2.0 Dataset and Preprocessing Module

Handles loading the Alzheimer MRI Hugging Face dataset from disk,
data augmentations, train/val/test splits, and PyTorch DataLoader creation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from datasets import load_from_disk

from src.config import (
    DEFAULT_DATASET_PATH,
    CLASSES,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    SEED,
)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test torchvision image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    return train_transform, eval_transform


class AlzheimerMRIDataset(Dataset):
    """PyTorch Dataset wrapper around Hugging Face Alzheimer MRI dataset."""

    def __init__(self, hf_split_dataset, transform=None):
        self.dataset = hf_split_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_class_weights(dataset_split) -> torch.Tensor:
    """Computes balanced inverse-frequency class weights for CrossEntropyLoss.
    
    Helps alleviate heavy class imbalances (e.g., underrepresented Moderate_Demented).
    """
    labels = [int(item["label"]) for item in dataset_split]
    counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(CLASSES)

    weights = []
    for c in range(num_classes):
        class_count = counts.get(c, 1)
        # Standard balanced weight formula: total / (num_classes * class_count)
        weight = total_samples / (num_classes * class_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    dataset_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Loads dataset from disk and constructs train, val, and test DataLoaders.
    
    Returns:
        train_loader: DataLoader for model training
        val_loader: DataLoader for validation during training
        test_loader: DataLoader for final evaluation
        class_weights: Tensor of class weights for loss function
    """
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    dataset_dict = load_from_disk(str(path))
    raw_train = dataset_dict["train"]
    raw_test = dataset_dict["test"]

    train_tf, eval_tf = get_transforms()

    # Calculate class weights from training distribution
    class_weights = compute_class_weights(raw_train)

    # Stratified or randomized train-validation split
    total_train = len(raw_train)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(total_train)
    train_indices = shuffled_indices[:train_size].tolist()
    val_indices = shuffled_indices[train_size:].tolist()

    # Wrap training subset with train augmentations
    full_train_ds = AlzheimerMRIDataset(raw_train, transform=train_tf)
    full_val_ds = AlzheimerMRIDataset(raw_train, transform=eval_tf)
    test_ds = AlzheimerMRIDataset(raw_test, transform=eval_tf)

    train_subset = Subset(full_train_ds, train_indices)
    val_subset = Subset(full_val_ds, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights
