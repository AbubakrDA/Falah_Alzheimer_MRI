"""NeuroVision Vision Transformer Attention Rollout Module

Computes attention rollout maps across transformer encoder layers,
visualizing which anatomical regions the ViT self-attention heads focus on.
"""

from __future__ import annotations
from typing import List, Optional, Any
import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None


class ViTAttentionRollout:
    """Computes attention rollout attribution for Vision Transformers."""

    def __init__(self, model: Any, head_fusion: str = "mean", discard_ratio: float = 0.9):
        if torch is None:
            raise RuntimeError("PyTorch is required for ViTAttentionRollout.")
        self.model = model.eval() if hasattr(model, "eval") else model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def generate_heatmap(
        self,
        input_tensor: Any,
        target_size: tuple = (224, 224),
    ) -> np.ndarray:
        """Computes 2D attention heatmap via attention rollout."""
        if torch is None:
            return np.zeros(target_size, dtype=np.float32)

        self.model.zero_grad()
        # Forward pass with attention recording enabled
        _ = self.model(input_tensor, record_attention=True)
        attn_weights = self.model.get_attention_weights()

        if not attn_weights:
            return np.zeros(target_size, dtype=np.float32)

        # Start with identity matrix: [Tokens, Tokens]
        # Number of tokens = 1 (CLS) + n_patches
        num_tokens = attn_weights[0].shape[-1]
        result = torch.eye(num_tokens).to(input_tensor.device)

        with torch.no_grad():
            for attention in attn_weights:
                # attention shape: [Batch, Heads, Tokens, Tokens]
                if self.head_fusion == "mean":
                    attn_fused = torch.mean(attention, dim=1).squeeze(0)
                elif self.head_fusion == "max":
                    attn_fused = torch.max(attention, dim=1)[0].squeeze(0)
                else:
                    attn_fused = torch.mean(attention, dim=1).squeeze(0)

                # Add identity to account for residual connections
                identity = torch.eye(num_tokens).to(input_tensor.device)
                a = 0.5 * attn_fused + 0.5 * identity
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        # Extract CLS token attention to patches: ignore CLS to CLS (index 0)
        cls_attention = result[0, 1:].cpu().numpy()

        # Reshape to 2D patch grid (e.g. 14x14 for 224x224 with 16x16 patches)
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        heatmap_grid = cls_attention.reshape(grid_size, grid_size)

        # Normalize
        if heatmap_grid.max() > 0:
            heatmap_norm = (heatmap_grid - heatmap_grid.min()) / (heatmap_grid.max() - heatmap_grid.min() + 1e-8)
        else:
            heatmap_norm = np.zeros_like(heatmap_grid)

        # Upsample to image size
        cam_resized = cv2.resize(heatmap_norm, target_size, interpolation=cv2.INTER_CUBIC)
        return cam_resized
