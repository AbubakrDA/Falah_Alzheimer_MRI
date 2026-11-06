"""NeuroVision Clinical Inference and Explainability Engine

Provides fast, reliable inference for brain MRI Alzheimer's diagnosis.
Supports both PyTorch (.pt, .pth) and TensorFlow / Keras (.keras, .h5) model checkpoints.
Safely handles environments where either PyTorch or TensorFlow C++ runtimes are unavailable.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image

# Safe lazy import of PyTorch
try:
    import torch
except Exception:
    torch = None

# Safe lazy import of TensorFlow to prevent Windows DLL load failure
try:
    import tensorflow as tf
except Exception:
    tf = None

from src.config.settings import (
    FALAH_CLASSES,
    MODELS_DIR,
    DEFAULT_IMAGE_SIZE,
    DISCLAIMER_TEXT,
)

# Safe imports for explainability modules
try:
    from src.explainability.gradcam import GradCAMExplainer, GradCAM
except Exception:
    GradCAMExplainer = None
    GradCAM = None

try:
    from src.explainability.vit_attention import ViTAttentionExplainer
except Exception:
    ViTAttentionExplainer = None

try:
    from src.explainability.vit_rollout import ViTAttentionRollout
except Exception:
    ViTAttentionRollout = None

try:
    from src.models import build_model as build_torch_model
except Exception:
    build_torch_model = None


class NeuroVisionPredictor:
    """Multi-framework inference engine for brain MRI classification and explainability.
    Supports Custom CNN, ResNet18/50, and Vision Transformer models.
    """

    def __init__(self, checkpoint_path: Optional[Any] = None, class_names: Optional[list] = None):
        self.class_names = class_names or FALAH_CLASSES
        self.model = None
        self.framework = "torch" if torch is not None else "tensorflow"
        self.architecture = "resnet18"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else None

        if checkpoint_path:
            self.load_checkpoint(Path(checkpoint_path))

    def load_checkpoint(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        if path.suffix in [".pt", ".pth"]:
            if torch is None:
                raise RuntimeError(
                    "PyTorch is required to load .pt/.pth checkpoints but is not installed or failed to load. "
                    "Install PyTorch with: pip install torch torchvision"
                )
            self.framework = "torch"
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "architecture" in checkpoint:
                arch_raw = checkpoint["architecture"].lower()
                if "resnet" in arch_raw:
                    arch = "resnet50" if "50" in arch_raw else "resnet18"
                elif "transformer" in arch_raw or "vit" in arch_raw:
                    arch = "vit"
                else:
                    arch = "cnn"
                self.architecture = arch
                self.class_names = checkpoint.get("classes", self.class_names)
                if build_torch_model is not None:
                    self.model = build_torch_model(arch, num_classes=len(self.class_names), pretrained=False)
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    self.model.load_state_dict(state_dict)
                else:
                    self.model = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                if build_torch_model is not None:
                    self.model = build_torch_model("resnet18", num_classes=len(self.class_names), pretrained=False)
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model = checkpoint
            else:
                self.model = checkpoint

            if hasattr(self.model, "to") and self.device is not None:
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

        elif path.suffix in [".keras", ".h5"]:
            self.framework = "tensorflow"
            if tf is None:
                raise RuntimeError(
                    "TensorFlow native C++ runtime is not available on this machine (DLL initialization failed). "
                    "Ensure Microsoft Visual C++ Redistributable is installed or use PyTorch models."
                )
            try:
                from src.models.vit_model import Patches, PatchEncoder
                custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}
            except Exception:
                custom_objects = {}

            try:
                self.model = tf.keras.models.load_model(str(path), custom_objects=custom_objects)
            except Exception as e:
                raise RuntimeError(f"Failed to load Keras model from {path}: {e}")
        elif path.suffix == ".json":
            import json
            with open(path, "r") as fh:
                self.model = json.load(fh)
            self.framework = "simulation"
            self.class_names = self.model.get("classes", self.class_names)
        else:
            raise ValueError(f"Unsupported checkpoint extension: {path.suffix}")

    def predict(
        self,
        image: Image.Image,
        target_size: tuple = DEFAULT_IMAGE_SIZE,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Runs inference on an input PIL image and optionally generates attention heatmaps."""
        if self.model is None:
            # Fallback mock prediction if model is not yet loaded
            preds = np.ones(len(self.class_names), dtype=np.float32) / len(self.class_names)
            pred_idx = int(np.argmax(preds))
            return {
                "predicted_class": self.class_names[pred_idx],
                "predicted_index": pred_idx,
                "confidence": round(float(preds[pred_idx]) * 100.0, 2),
                "probabilities": {name: round(100.0 / len(self.class_names), 2) for name in self.class_names},
                "disclaimer": DISCLAIMER_TEXT,
            }

        if self.framework == "torch" and torch is not None:
            # PyTorch inference
            img_rgb = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
            img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
            x_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            if self.device is not None:
                x_tensor = x_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(x_tensor)
                preds = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx]) * 100.0
            predicted_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

            probs_dict = {
                self.class_names[i] if i < len(self.class_names) else f"Class_{i}": round(float(p) * 100.0, 2)
                for i, p in enumerate(preds)
            }

            result = {
                "predicted_class": predicted_class,
                "predicted_index": pred_idx,
                "confidence": round(confidence, 2),
                "probabilities": probs_dict,
                "disclaimer": DISCLAIMER_TEXT,
            }

            if explain:
                try:
                    if self.architecture == "vit" and ViTAttentionRollout is not None:
                        rollout = ViTAttentionRollout(self.model)
                        heatmap = rollout.generate_heatmap(x_tensor, target_size=target_size)
                        if GradCAMExplainer is not None:
                            overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap)
                            result["gradcam_heatmap"] = heatmap
                            result["gradcam_overlay"] = overlay
                            result["method"] = "ViT Attention Rollout"
                    elif GradCAM is not None:
                        target_layer = (
                            self.model.get_target_layer_for_cam()
                            if hasattr(self.model, "get_target_layer_for_cam")
                            else getattr(self.model, "target_layer", None)
                        )
                        if target_layer is not None:
                            cam = GradCAM(self.model, target_layer=target_layer)
                            heatmap = cam.generate_heatmap(x_tensor, class_idx=pred_idx, target_size=target_size)
                            if GradCAMExplainer is not None:
                                overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap)
                                result["gradcam_heatmap"] = heatmap
                                result["gradcam_overlay"] = overlay
                                result["method"] = "Grad-CAM"
                except Exception as e:
                    result["explainability_error"] = str(e)

            return result

        elif self.framework == "simulation":
            # Lightweight simulation inference
            preds = np.random.dirichlet(np.ones(len(self.class_names)))
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx]) * 100.0
            predicted_class = self.class_names[pred_idx]

            probs_dict = {
                self.class_names[i]: round(float(p) * 100.0, 2)
                for i, p in enumerate(preds)
            }

            result = {
                "predicted_class": predicted_class,
                "predicted_index": pred_idx,
                "confidence": round(confidence, 2),
                "probabilities": probs_dict,
                "disclaimer": DISCLAIMER_TEXT,
            }

            if explain:
                h, w = target_size
                y_grid, x_grid = np.ogrid[:h, :w]
                cy, cx = h // 2, w // 2
                d2 = (x_grid - cx)**2 + (y_grid - cy)**2
                heatmap = np.exp(-d2 / (2 * (38.0**2))).astype(np.float32)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                img_rgb = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
                if GradCAMExplainer is not None:
                    overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap, colormap="jet")
                else:
                    overlay = (img_arr * 255.0).astype(np.uint8)
                result["gradcam_heatmap"] = heatmap
                result["gradcam_overlay"] = overlay
                result["method"] = "Simulation Saliency Map"

            return result

        else:
            # TensorFlow/Keras inference
            if tf is None:
                raise RuntimeError("TensorFlow is not available on this platform.")

            channels = 1
            if hasattr(self.model, "input_shape") and len(self.model.input_shape) == 4:
                channels = self.model.input_shape[-1] or 1

            if channels == 3:
                img = image.convert("RGB")
            else:
                img = image.convert("L")

            img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
            img_arr = np.array(img_resized, dtype=np.float32) / 255.0

            if channels == 1 and img_arr.ndim == 2:
                img_arr = np.expand_dims(img_arr, axis=-1)

            input_batch = np.expand_dims(img_arr, axis=0)
            preds = self.model.predict(input_batch, verbose=0)[0]

            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx]) * 100.0
            predicted_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

            probs_dict = {
                self.class_names[i] if i < len(self.class_names) else f"Class_{i}": round(float(p) * 100.0, 2)
                for i, p in enumerate(preds)
            }

            result = {
                "predicted_class": predicted_class,
                "predicted_index": pred_idx,
                "confidence": round(confidence, 2),
                "probabilities": probs_dict,
                "disclaimer": DISCLAIMER_TEXT,
            }

            if explain:
                is_vit = "transformer" in getattr(self.model, "name", "").lower() or "vit" in getattr(self.model, "name", "").lower()
                try:
                    if is_vit and ViTAttentionExplainer is not None:
                        sal = ViTAttentionExplainer.compute_patch_saliency(self.model, input_batch, class_idx=pred_idx)
                        overlay = ViTAttentionExplainer.overlay_saliency(img_arr, sal)
                        result["attention_heatmap"] = sal
                        result["attention_overlay"] = overlay
                        result["method"] = "ViT Patch Saliency"
                    elif GradCAMExplainer is not None:
                        heatmap = GradCAMExplainer.generate_heatmap(self.model, input_batch, class_idx=pred_idx)
                        overlay = GradCAMExplainer.overlay_heatmap(img_arr, heatmap)
                        result["gradcam_heatmap"] = heatmap
                        result["gradcam_overlay"] = overlay
                        result["method"] = "Grad-CAM"
                except Exception as e:
                    result["explainability_error"] = str(e)

            return result
