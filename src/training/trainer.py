"""NeuroVision Unified Multi-Dataset Trainer

Orchestrates model training across Falah Alzheimer MRI and OASIS-1 datasets,
supporting CNN, ResNet, and Vision Transformer architectures with clinical metrics.
Provides dual backends: PyTorch (native, high performance, robust) and TensorFlow/Keras.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import datetime
import numpy as np

# Safe PyTorch lazy import
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from src.models import build_model as build_torch_model
except Exception:
    torch = None
    nn = None
    AdamW = None
    CosineAnnealingLR = None
    DataLoader = None
    Dataset = None
    TensorDataset = None
    build_torch_model = None

# Safe TensorFlow lazy import
try:
    import tensorflow as tf
except Exception:
    tf = None
from src.training.metrics import compute_clinical_metrics
from src.data.falah_loader import FalahDatasetLoader
from src.data.oasis1_loader import OASIS1DatasetLoader
from src.config.settings import MODELS_DIR


class UnifiedMRITrainer:
    """Unified training engine supporting Falah and OASIS-1 with CNN, ResNet, and ViT (PyTorch)."""

    def __init__(
        self,
        dataset_name: str = "falah",
        model_type: str = "resnet18",
        task: str = "multiclass",
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 15,
        device: Optional[Any] = None,
        save_dir: Optional[Path] = None,
    ):
        if torch is None or build_torch_model is None:
            raise RuntimeError(
                "UnifiedMRITrainer requires PyTorch, but PyTorch is not available. "
                "Install PyTorch with: pip install torch torchvision"
            )

        self.dataset_name = dataset_name.lower()
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else None)
        self.save_dir = Path(save_dir) if save_dir else Path(MODELS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Configure dataset and class mapping
        if "falah" in self.dataset_name:
            self.loader = FalahDatasetLoader()
            self.classes = self.loader.class_names
            self.num_classes = len(self.classes)
        elif "oasis" in self.dataset_name:
            self.loader = OASIS1DatasetLoader(task=self.task)
            self.classes = self.loader.label_mapper.target_classes
            self.num_classes = len(self.classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'falah' or 'oasis1'.")

        # Normalize model type name for build_torch_model
        arch = self.model_type
        if "resnet" in arch:
            arch = "resnet50" if "50" in arch else "resnet18"
        elif "transformer" in arch or "vit" in arch:
            arch = "vit"
        else:
            arch = "cnn"

        self.model = build_torch_model(
            model_type=arch,
            num_classes=self.num_classes,
            pretrained=True,
        )
        if hasattr(self.model, "to") and self.device is not None:
            self.model = self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss()

    def get_model(self) -> Any:
        return self.model

    def train(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: Optional[int] = None,
        class_weights: Optional[Dict[int, float]] = None,
        use_early_stopping: bool = True,
        patience: int = 3,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Trains the PyTorch model with validation tracking, early stopping, and callbacks."""
        num_epochs = epochs or self.epochs

        if class_weights is not None:
            weights_arr = [class_weights.get(i, 1.0) for i in range(self.num_classes)]
            weights_tensor = torch.tensor(weights_arr, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total_samples = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

            self.scheduler.step()

            epoch_loss = total_loss / max(total_samples, 1)
            epoch_acc = correct / max(total_samples, 1)
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for x_v, y_v in val_loader:
                        x_v = x_v.to(self.device)
                        y_v = y_v.to(self.device)
                        logits_v = self.model(x_v)
                        loss_v = self.criterion(logits_v, y_v)
                        val_loss += loss_v.item() * len(y_v)
                        preds_v = logits_v.argmax(dim=1)
                        val_correct += (preds_v == y_v).sum().item()
                        val_total += len(y_v)

                val_loss = val_loss / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

            if progress_callback:
                progress_callback(epoch, {
                    "loss": epoch_loss,
                    "accuracy": epoch_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                })

            # Early stopping check
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.save_dir / f"{self.dataset_name}_{self.model_type}_{ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = exp_dir / f"{self.model_type}.pt"

        torch.save({
            "state_dict": self.model.state_dict(),
            "architecture": self.model_type,
            "classes": self.classes,
            "task": self.task,
            "history": history,
            "dataset_name": self.dataset_name,
        }, checkpoint_path)

        return {
            "checkpoint_path": str(checkpoint_path),
            "history": history,
            "experiment_name": exp_dir.name,
            "framework": "torch",
        }


class ModelTrainer:
    """Unified experiment trainer for Streamlit, supporting both PyTorch and Keras."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        dataset_name: str = "Falah_Alzheimer_MRI",
        task: str = "multiclass",
        framework: str = "torch",
    ):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{dataset_name}_{task}_{ts}"
        self.dataset_name = dataset_name
        self.task = task
        self.framework = framework.lower()
        self.model = None
        self.architecture_name = "custom_cnn"
        self.num_classes = 4
        self.classes: List[str] = []

    def build_model(
        self,
        architecture: str,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        classes: Optional[List[str]] = None,
    ):
        self.architecture_name = architecture.replace(" ", "_").lower()
        self.num_classes = num_classes
        self.classes = classes or []
        self.learning_rate = learning_rate
        arch_lower = architecture.lower()

        if "torch" in self.framework:
            if torch is None or build_torch_model is None:
                raise RuntimeError(
                    "PyTorch framework selected, but PyTorch is not installed or could not be loaded. "
                    "Install PyTorch with: pip install torch torchvision"
                )
            arch_key = "cnn"
            if "resnet" in arch_lower:
                arch_key = "resnet50" if "50" in arch_lower else "resnet18"
            elif "transformer" in arch_lower or "vit" in arch_lower:
                arch_key = "vit"

            self.model = build_torch_model(
                model_type=arch_key,
                num_classes=num_classes,
                pretrained=True,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            self.criterion = nn.CrossEntropyLoss()
        elif "simulation" in self.framework or "demo" in self.framework:
            self.model = {
                "name": architecture,
                "architecture": self.architecture_name,
                "num_classes": num_classes,
                "classes": self.classes,
                "framework": "simulation",
            }
        else:
            if tf is None:
                raise RuntimeError(
                    "TensorFlow is required for Keras backend but failed to load native runtime (DLL failed). "
                    "Select PyTorch framework instead."
                )
            if "resnet" in arch_lower:
                from src.models.resnet_model import build_resnet_model
                self.model = build_resnet_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                )
            elif "transformer" in arch_lower or "vit" in arch_lower:
                from src.models.vit_model import build_vit_model
                self.model = build_vit_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                )
            else:
                from src.models.cnn_model import build_custom_cnn
                self.model = build_custom_cnn(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                )
        return self.model

    def train(
        self,
        train_data,
        val_data,
        epochs: int = 5,
        batch_size: int = 32,
        class_weights: Optional[Dict[int, float]] = None,
        use_early_stopping: bool = True,
        patience: int = 3,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        checkpoint_dir = Path(MODELS_DIR) / self.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if "torch" in self.framework:
            # PyTorch training
            if isinstance(train_data, tuple):
                x_tr, y_tr = train_data
                x_tr_t = torch.tensor(x_tr, dtype=torch.float32).permute(0, 3, 1, 2)
                y_tr_t = torch.tensor(y_tr, dtype=torch.long)
                train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
            else:
                train_loader = train_data

            val_loader = None
            if val_data is not None:
                if isinstance(val_data, tuple):
                    x_va, y_va = val_data
                    x_va_t = torch.tensor(x_va, dtype=torch.float32).permute(0, 3, 1, 2)
                    y_va_t = torch.tensor(y_va, dtype=torch.long)
                    val_loader = DataLoader(TensorDataset(x_va_t, y_va_t), batch_size=batch_size, shuffle=False)
                else:
                    val_loader = val_data

            if class_weights is not None:
                w_arr = [class_weights.get(i, 1.0) for i in range(self.num_classes)]
                w_t = torch.tensor(w_arr, dtype=torch.float32, device=self.device)
                self.criterion = nn.CrossEntropyLoss(weight=w_t)

            best_val_loss = float("inf")
            patience_cnt = 0
            best_state = None
            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            for ep in range(1, epochs + 1):
                self.model.train()
                tr_loss, tr_corr, tr_tot = 0.0, 0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    loss.backward()
                    self.optimizer.step()

                    tr_loss += loss.item() * len(yb)
                    tr_corr += (out.argmax(dim=1) == yb).sum().item()
                    tr_tot += len(yb)

                ep_loss = tr_loss / max(tr_tot, 1)
                ep_acc = tr_corr / max(tr_tot, 1)
                history["loss"].append(ep_loss)
                history["accuracy"].append(ep_acc)

                val_loss, val_corr, val_tot = 0.0, 0, 0
                if val_loader:
                    self.model.eval()
                    with torch.no_grad():
                        for xv, yv in val_loader:
                            xv, yv = xv.to(self.device), yv.to(self.device)
                            outv = self.model(xv)
                            lv = self.criterion(outv, yv)
                            val_loss += lv.item() * len(yv)
                            val_corr += (outv.argmax(dim=1) == yv).sum().item()
                            val_tot += len(yv)
                    val_loss = val_loss / max(val_tot, 1)
                    val_acc = val_corr / max(val_tot, 1)
                else:
                    val_loss, val_acc = ep_loss, ep_acc

                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if progress_callback:
                    progress_callback(ep, {
                        "loss": ep_loss,
                        "accuracy": ep_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    })

                if use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_cnt = 0
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

            if best_state is not None:
                self.model.load_state_dict(best_state)

            checkpoint_path = checkpoint_dir / f"{self.architecture_name}.pt"
            torch.save({
                "state_dict": self.model.state_dict(),
                "architecture": self.architecture_name,
                "num_classes": self.num_classes,
                "classes": self.classes,
                "task": self.task,
                "dataset_name": self.dataset_name,
                "history": history,
            }, checkpoint_path)

            return {
                "checkpoint_path": str(checkpoint_path),
                "history": history,
                "experiment_name": self.experiment_name,
                "framework": "torch",
            }

        elif "simulation" in self.framework or "demo" in self.framework:
            # Lightweight simulation engine using pure NumPy
            import time
            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            for ep in range(1, epochs + 1):
                time.sleep(0.15)
                decay = np.exp(-0.35 * (ep - 1))
                ep_loss = float(max(0.12, 1.25 * decay + np.random.uniform(-0.02, 0.02)))
                ep_acc = float(min(0.96, 0.52 + 0.10 * (ep - 1) + np.random.uniform(-0.01, 0.01)))
                val_loss = float(ep_loss + np.random.uniform(0.04, 0.08))
                val_acc = float(max(0.48, ep_acc - np.random.uniform(0.02, 0.04)))

                history["loss"].append(round(ep_loss, 4))
                history["accuracy"].append(round(ep_acc, 4))
                history["val_loss"].append(round(val_loss, 4))
                history["val_accuracy"].append(round(val_acc, 4))

                if progress_callback:
                    progress_callback(ep, {
                        "loss": ep_loss,
                        "accuracy": ep_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    })

            checkpoint_path = checkpoint_dir / f"{self.architecture_name}.pt"
            meta_save = {
                "architecture": self.architecture_name,
                "num_classes": self.num_classes,
                "classes": self.classes,
                "task": self.task,
                "dataset_name": self.dataset_name,
                "history": history,
                "framework": "simulation",
            }
            if torch is not None:
                torch.save(meta_save, checkpoint_path)
            else:
                with open(checkpoint_path.with_suffix(".json"), "w") as fh:
                    json.dump(meta_save, fh, indent=2)

            return {
                "checkpoint_path": str(checkpoint_path),
                "history": history,
                "experiment_name": self.experiment_name,
                "framework": "simulation",
            }

        else:
            # TensorFlow/Keras training
            if tf is None:
                raise RuntimeError("TensorFlow is required for Keras ModelTrainer but failed to load.")

            callbacks = []
            if use_early_stopping:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=patience,
                        restore_best_weights=True,
                    )
                )

            if progress_callback:
                class CallbackBridge(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress_callback(epoch + 1, logs or {})

                callbacks.append(CallbackBridge())

            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )

            checkpoint_path = checkpoint_dir / f"{self.architecture_name}.keras"
            self.model.save(str(checkpoint_path))

            return {
                "checkpoint_path": str(checkpoint_path),
                "history": history.history,
                "experiment_name": self.experiment_name,
                "framework": "tensorflow",
            }
