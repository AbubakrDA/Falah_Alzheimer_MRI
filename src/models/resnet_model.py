from typing import Tuple, Optional
from src.config.settings import DEFAULT_IMAGE_SIZE

try:
    import tensorflow as tf
except Exception:
    tf = None


def build_resnet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 4,
    freeze_backbone: bool = True,
    unfreeze_top_n_layers: int = 0,
    dense_units: int = 256,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-4,
    pretrained: bool = True
):
    """Builds a Transfer Learning ResNet50 model adapted for brain MRI in TensorFlow/Keras."""
    if tf is None:
        raise RuntimeError("TensorFlow native runtime is not available on this platform.")

    inputs = tf.keras.Input(shape=input_shape, name="mri_input")
    x = inputs

    if input_shape[-1] == 1:
        x = tf.keras.layers.Conv2D(3, (1, 1), padding="same", name="channel_adapter_conv")(x)

    weights = "imagenet" if pretrained else None
    try:
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=x,
            pooling=None
        )
    except Exception:
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling=None
        )

    if freeze_backbone:
        backbone.trainable = False
        if unfreeze_top_n_layers > 0:
            for layer in backbone.layers[-unfreeze_top_n_layers:]:
                layer.trainable = True
    else:
        backbone.trainable = True

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="resnet_gap")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="resnet_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="resnet_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="resnet_dropout")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NeuroVision_ResNet50_Transfer")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
