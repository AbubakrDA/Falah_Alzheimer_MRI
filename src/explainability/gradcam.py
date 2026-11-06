"""NeuroVision Grad-CAM Explainability Module

Generates Gradient-weighted Class Activation Maps (Grad-CAM) for both:
1. TensorFlow / Keras models (GradCAMExplainer)
2. PyTorch models (GradCAM)
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
import numpy as np
import cv2
from PIL import Image

# -----------------------------------------------------------------------------
# 1. TENSORFLOW / KERAS GRAD-CAM EXPLAINER
# -----------------------------------------------------------------------------
class GradCAMExplainer:
    """Computes Grad-CAM attention heatmaps for TensorFlow / Keras models."""

    @classmethod
    def find_last_conv_layer(cls, model) -> Optional[str]:
        """Finds the name of the last convolutional layer in the model."""
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower() or isinstance(layer, (getattr(model, "layers", []))):
                if hasattr(layer, "output") and len(layer.output_shape) == 4:
                    return layer.name
        # Fallback search for any 4D activation layer
        for layer in reversed(model.layers):
            if len(getattr(layer, "output_shape", [])) == 4:
                return layer.name
        return None

    @classmethod
    def generate_heatmap(
        cls,
        model,
        img_array: np.ndarray,
        class_idx: Optional[int] = None,
        target_layer_name: Optional[str] = None,
    ) -> np.ndarray:
        """Computes normalized Grad-CAM heatmap [0, 1] for a TensorFlow model."""
        import tensorflow as tf

        layer_name = target_layer_name or cls.find_last_conv_layer(model)
        if layer_name is None:
            # If no conv layer found, return neutral heatmap
            return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

        try:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.output],
            )
        except Exception:
            return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0.0)

        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()
        h, w = img_array.shape[1], img_array.shape[2]
        resized = cv2.resize(heatmap_np, (w, h))
        return np.maximum(resized, 0.0)

    @classmethod
    def overlay_heatmap(
        cls,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Overlays Grad-CAM heatmap on top of the original MRI slice."""
        if original_img.ndim == 3 and original_img.shape[-1] == 1:
            base = np.squeeze(original_img, axis=-1)
        elif original_img.ndim == 2:
            base = original_img
        else:
            base = original_img[:, :, 0]

        base_uint8 = np.uint8(255 * np.clip(base, 0, 1))
        base_rgb = np.stack([base_uint8] * 3, axis=-1)

        colored_cam = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
        colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(alpha * colored_cam_rgb + (1.0 - alpha) * base_rgb)
        return overlay


# -----------------------------------------------------------------------------
# 2. PYTORCH GRAD-CAM EXPLAINER
# -----------------------------------------------------------------------------
class GradCAM:
    """Computes Grad-CAM heatmaps for PyTorch models from convolutional feature layers."""

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer_for_cam"):
            self.target_layer = model.get_target_layer_for_cam()
        else:
            raise ValueError("Target layer must be specified or model must implement get_target_layer_for_cam().")

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor,
        target_class: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224),
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Computes normalized [0, 1] Grad-CAM heatmap for PyTorch."""
        import torch
        import torch.nn.functional as F

        if target_class is None and class_idx is not None:
            target_class = class_idx

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, target_size)
        return cam_resized


__all__ = ["GradCAMExplainer", "GradCAM"]
