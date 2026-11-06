"""NeuroVision 2.0 Evaluation & Clinical Interpretability Module

Computes test set performance metrics, confusion matrices, and Grad-CAM
heatmaps for visualizing anatomical brain regions relevant to Alzheimer diagnosis.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    DEFAULT_BACKBONE,
    CLASSES,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
)
from src.dataset import get_dataloaders
from src.models import build_model, NeuroVisionClassifier


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN/ConvNeXt backbones."""

    def __init__(self, model: NeuroVisionClassifier):
        self.model = model.eval()
        self.target_layer = model.get_target_layer_for_cam()
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

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Computes Grad-CAM heatmap for a single input tensor [1, 3, H, W]."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()

        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, IMAGE_SIZE)
        return cam_resized


def evaluate_test_set(
    checkpoint_path: Optional[Path] = None,
    model_name: str = DEFAULT_BACKBONE,
    batch_size: int = 32,
) -> None:
    """Evaluates the trained model on test data and exports reports & plots."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    chk_path = checkpoint_path or (MODELS_DIR / f"neurovision_{model_name}_best.pth")

    if not chk_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {chk_path}")

    print(f"Loading checkpoint: {chk_path.name}")
    checkpoint = torch.load(chk_path, map_location=DEVICE)

    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _, _, test_loader, _ = get_dataloaders(batch_size=batch_size)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    print("\n========== NEUROVISION 2.0 TEST SET REPORT ==========")
    report = classification_report(all_targets, all_preds, target_names=CLASSES, digits=4)
    print(report)

    # Save classification report to text file
    report_file = OUTPUTS_DIR / f"classification_report_{model_name}.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title=f"Confusion Matrix (Normalized) - {model_name.upper()}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    # Annotate counts and percentages in each cell
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    cm_plot_path = OUTPUTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot saved to: {cm_plot_path}")


def generate_gradcam_visualizations(
    checkpoint_path: Optional[Path] = None,
    model_name: str = DEFAULT_BACKBONE,
    num_samples: int = 4,
) -> None:
    """Generates and saves Grad-CAM heatmaps overlaid on sample test MRI scans."""
    chk_path = checkpoint_path or (MODELS_DIR / f"neurovision_{model_name}_best.pth")
    if not chk_path.exists():
        print(f"Skipping Grad-CAM: checkpoint not found at {chk_path}")
        return

    checkpoint = torch.load(chk_path, map_location=DEVICE)
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    gradcam = GradCAM(model)
    _, _, test_loader, _ = get_dataloaders(batch_size=num_samples)

    images, labels = next(iter(test_loader))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        img_tensor = images[i : i + 1].to(DEVICE)
        true_label = CLASSES[labels[i].item()]

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]
            confidence = probs[pred_idx] * 100

        heatmap = gradcam.generate_heatmap(img_tensor, target_class=pred_idx)

        # Denormalize image for display
        raw_img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        raw_img = raw_img * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)
        raw_img = np.clip(raw_img, 0, 1)

        # Apply colormap to heatmap
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Create overlay
        overlay = 0.6 * raw_img + 0.4 * colored_heatmap
        overlay = np.clip(overlay, 0, 1)

        # Column 1: Original MRI
        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"True: {true_label}")
        axes[i, 0].axis("off")

        # Column 2: Grad-CAM Activation Heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        # Column 3: Anatomical Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Pred: {pred_label} ({confidence:.1f}%)")
        axes[i, 2].axis("off")

    plt.suptitle("NeuroVision 2.0 - Anatomical Attention & Explainability (Grad-CAM)")
    plt.tight_layout()
    gradcam_path = OUTPUTS_DIR / f"gradcam_samples_{model_name}.png"
    plt.savefig(gradcam_path, dpi=300)
    plt.close()
    print(f"Grad-CAM visualizations saved to: {gradcam_path}")


if __name__ == "__main__":
    evaluate_test_set()
    generate_gradcam_visualizations()
