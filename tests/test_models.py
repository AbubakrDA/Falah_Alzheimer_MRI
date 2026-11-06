import pytest
import numpy as np
import tensorflow as tf

from src.models.cnn_model import build_custom_cnn
from src.models.resnet_model import build_resnet_model
from src.models.vit_model import build_vit_model

def test_custom_cnn_forward_pass():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4, conv_filters=[16, 32])
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 64, 64, 1)).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (2, 4)
    # Assert probabilities sum to ~1.0
    np.testing.assert_allclose(np.sum(output.numpy(), axis=-1), np.ones(2), rtol=1e-5)

def test_resnet_transfer_binary_and_multiclass():
    # Test binary (2 classes)
    model_bin = build_resnet_model(input_shape=(64, 64, 1), num_classes=2, pretrained=False)
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 64, 64, 1)).astype(np.float32)
    out_bin = model_bin(dummy_input)
    assert out_bin.shape == (2, 2)

    # Test multiclass (4 classes)
    model_multi = build_resnet_model(input_shape=(64, 64, 1), num_classes=4, pretrained=False)
    out_multi = model_multi(dummy_input)
    assert out_multi.shape == (2, 4)

def test_vision_transformer_patch_embeddings():
    model = build_vit_model(
        input_shape=(64, 64, 1),
        num_classes=4,
        patch_size=16,
        projection_dim=32,
        num_heads=2,
        transformer_layers=2,
        mlp_dim=64
    )
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 64, 64, 1)).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (2, 4)
    np.testing.assert_allclose(np.sum(output.numpy(), axis=-1), np.ones(2), rtol=1e-5)
