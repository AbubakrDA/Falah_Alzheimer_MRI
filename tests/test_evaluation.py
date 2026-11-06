import pytest
import numpy as np
import tensorflow as tf

from src.evaluation.metrics import EvaluationMetrics
from src.models.cnn_model import build_custom_cnn
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.vit_attention import ViTAttentionExplainer
from src.models.vit_model import build_vit_model

def test_evaluation_metrics_multiclass():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 2, 3])
    # Create synthetic probabilities with high confidence for ground truth
    y_prob = np.zeros((10, 4), dtype=np.float32)
    for i, t in enumerate(y_true):
        y_prob[i, t] = 0.85
        y_prob[i, (t + 1) % 4] = 0.15

    class_names = ["Mild", "Moderate", "Non", "Very_Mild"]
    results = EvaluationMetrics.evaluate(y_true, y_prob, class_names, n_bootstraps=50)

    assert results["accuracy"] == 1.0
    assert results["f1_macro"] == 1.0
    assert results["balanced_accuracy"] == 1.0
    assert len(results["per_class"]) == 4
    assert len(results["confusion_matrix"]) == 4

def test_gradcam_generation():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4, conv_filters=[16, 32])
    dummy_img = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 1)).astype(np.float32)
    
    heatmap = GradCAMExplainer.generate_heatmap(model, dummy_img)
    assert heatmap.ndim == 2
    assert heatmap.max() <= 1.0 and heatmap.min() >= 0.0

    overlay = GradCAMExplainer.overlay_heatmap(dummy_img[0], heatmap)
    assert overlay.shape == (64, 64, 3)

def test_vit_attention_generation():
    model = build_vit_model(
        input_shape=(64, 64, 1),
        num_classes=4,
        patch_size=16,
        projection_dim=32,
        num_heads=2,
        transformer_layers=2,
        mlp_dim=64
    )
    dummy_img = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 1)).astype(np.float32)
    
    saliency = ViTAttentionExplainer.compute_patch_saliency(model, dummy_img)
    assert saliency.ndim == 2
    assert saliency.shape == (64, 64)

    overlay = ViTAttentionExplainer.overlay_saliency(dummy_img[0], saliency)
    assert overlay.shape == (64, 64, 3)
