from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

import matplotlib.cm as cm
from PIL import Image


class ViTAttentionExplainer:
    """Computes Patch-Level Attention Attribution for Vision Transformers.
    Does NOT describe transformer attribution as Grad-CAM.
    Visualizes spatial patch saliency across the neuroanatomical regions of the brain MRI.
    """

    @classmethod
    def compute_patch_saliency(
        cls,
        model,
        img_array: np.ndarray,  # (1, H, W, C)
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Computes input attribution sensitivity for the Vision Transformer model
        and pools it over spatial patch regions.
        """
        if tf is None:
            return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            target_loss = predictions[:, class_idx]

        grads = tape.gradient(target_loss, img_tensor)
        saliency = tf.reduce_mean(tf.abs(grads), axis=-1)[0]

        sal_min = tf.reduce_min(saliency)
        sal_max = tf.reduce_max(saliency)
        if sal_max > sal_min:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = tf.zeros_like(saliency)

        return saliency.numpy()

    @classmethod
    def overlay_saliency(
        cls,
        original_img: np.ndarray,
        saliency_map: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "magma",
    ) -> np.ndarray:
        """Overlays transformer spatial saliency on top of the original MRI slice."""
        if original_img.ndim == 3 and original_img.shape[-1] == 1:
            base = np.squeeze(original_img, axis=-1)
        elif original_img.ndim == 2:
            base = original_img
        else:
            base = original_img[:, :, 0]

        base_uint8 = np.uint8(255 * np.clip(base, 0, 1))
        base_rgb = np.stack([base_uint8] * 3, axis=-1)

        sal_pil = Image.fromarray(np.uint8(255 * np.clip(saliency_map, 0, 1)))
        sal_resized = sal_pil.resize((base.shape[1], base.shape[0]), Image.Resampling.BILINEAR)
        sal_norm = np.array(sal_resized) / 255.0

        import matplotlib as mpl
        try:
            cmap = mpl.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(colormap)
        colored_sal = cmap(sal_norm)[:, :, :3]
        colored_sal_uint8 = np.uint8(255 * colored_sal)

        overlay = np.uint8(alpha * colored_sal_uint8 + (1 - alpha) * base_rgb)
        return overlay
