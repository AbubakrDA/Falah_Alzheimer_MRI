from typing import Tuple, List, Optional
from src.config.settings import DEFAULT_IMAGE_SIZE

try:
    import tensorflow as tf
except Exception:
    tf = None


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 4,
    conv_filters: List[int] = [32, 64, 128, 256],
    kernel_size: Tuple[int, int] = (3, 3),
    dropout_rate: float = 0.3,
    dense_units: int = 128,
    learning_rate: float = 1e-4
):
    """Builds a modular biomedical CNN for brain MRI classification in TensorFlow/Keras."""
    if tf is None:
        raise RuntimeError("TensorFlow native runtime is not available on this platform.")

    inputs = tf.keras.Input(shape=input_shape, name="mri_input")
    x = inputs

    for i, filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            name=f"conv2d_block_{i+1}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_block_{i+1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_block_{i+1}")(x)
        
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            name=f"conv2d_block_{i+1}_sub"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_block_{i+1}_sub")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_block_{i+1}_sub")(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool_{i+1}")(x)
        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(dropout_rate / 2.0, name=f"dropout_conv_{i+1}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NeuroVision_Custom_CNN")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
