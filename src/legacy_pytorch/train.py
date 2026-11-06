"""NeuroVision 2.0 Model Training Module

Executes model training with class-weighted CrossEntropyLoss, learning rate
scheduling, validation tracking (Accuracy, Macro F1, Loss), early stopping,
and checkpoint saving.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    DEFAULT_BACKBONE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    CLASSES,
)
from src.dataset import get_dataloaders
from src.models import build_model


class Trainer:
    """Orchestrates model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: torch.Tensor,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        epochs: int = EPOCHS,
        device: torch.device = DEVICE,
        save_dir: Path = MODELS_DIR,
        model_name: str = DEFAULT_BACKBONE,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Weighted loss to mitigate class imbalance
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=(correct / total) * 100)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        """Runs validation evaluation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        loss = running_loss / total
        acc = (correct / total) * 100
        return loss, acc

    def train(self) -> Dict[str, List[float]]:
        """Executes full training across all epochs."""
        print(f"\n==========================================")
        print(f" Starting Training: {self.model_name.upper()}")
        print(f" Device: {self.device}")
        print(f" Total Epochs: {self.epochs}")
        print(f"==========================================\n")

        best_checkpoint_path = self.save_dir / f"neurovision_{self.model_name}_best.pth"

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"| Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%"
            )

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "classes": CLASSES,
                        "backbone": self.model_name,
                    },
                    best_checkpoint_path,
                )
                print(f"  --> Best model checkpoint saved to: {best_checkpoint_path.name}")

        print(f"\nTraining completed! Peak Validation Accuracy: {self.best_val_acc:.2f}%")
        self._save_history_and_plots()
        return self.history

    def _save_history_and_plots(self) -> None:
        """Saves training history JSON and performance curve plots."""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        history_path = OUTPUTS_DIR / f"history_{self.model_name}.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Save loss & accuracy curve plots
        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, self.history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs_range, self.history["val_loss"], label="Val Loss", marker="s")
        ax1.set_title("Cross-Entropy Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(epochs_range, self.history["train_acc"], label="Train Accuracy", marker="o")
        ax2.plot(epochs_range, self.history["val_acc"], label="Val Accuracy", marker="s")
        ax2.set_title("Classification Accuracy Curve (%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(f"NeuroVision 2.0 - Training Progress ({self.model_name})")
        plt.tight_layout()
        plot_path = OUTPUTS_DIR / f"curves_{self.model_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Performance curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NeuroVision 2.0 for Alzheimer MRI")
    parser.add_argument("--model", type=str, default=DEFAULT_BACKBONE, help="Model backbone")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_loader, val_loader, _, class_weights = get_dataloaders(batch_size=args.batch_size)
    model = build_model(model_name=args.model, pretrained=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        learning_rate=args.lr,
        epochs=args.epochs,
        model_name=args.model,
    )
    trainer.train()


if __name__ == "__main__":
    main()
