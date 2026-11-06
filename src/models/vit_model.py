from typing import Tuple, List, Optional
from src.config.settings import DEFAULT_IMAGE_SIZE

try:
    import tensorflow as tf
    _layer_base = tf.keras.layers.Layer
except Exception:
    tf = None
    _layer_base = object


class Patches(_layer_base):
    """Extracts non-overlapping rectangular patches from images."""
    def __init__(self, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        if tf is None:
            raise RuntimeError("TensorFlow is not available.")
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = tf.shape(patches)[-1]
        num_patches = (tf.shape(images)[1] // self.patch_size) * (tf.shape(images)[2] // self.patch_size)
        return tf.reshape(patches, [batch_size, num_patches, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(_layer_base):
    """Linearly projects patches and injects learnable positional embeddings."""
    def __init__(self, num_patches: int = 64, projection_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        if tf is not None:
            self.projection = tf.keras.layers.Dense(units=projection_dim, name="patch_projection")
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_patches,
                output_dim=projection_dim,
                name="pos_embedding"
            )

    def call(self, patch):
        if tf is None:
            raise RuntimeError("TensorFlow is not available.")
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


def build_vit_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 4,
    patch_size: int = 16,
    projection_dim: int = 64,
    num_heads: int = 4,
    transformer_layers: int = 4,
    mlp_dim: int = 128,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-4
):
    """Builds a Vision Transformer (ViT) architecture in TensorFlow/Keras tailored for brain MRI."""
    if tf is None:
        raise RuntimeError("TensorFlow native runtime is not available on this platform.")

    if input_shape[0] % patch_size != 0 or input_shape[1] % patch_size != 0:
        raise ValueError(f"Image dimensions {input_shape[:2]} must be divisible by patch_size {patch_size}")

    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    inputs = tf.keras.Input(shape=input_shape, name="mri_input")

    patches = Patches(patch_size=patch_size, name="patch_extractor")(inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, name="patch_encoder")(patches)

    x = encoded_patches
    for i in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_block_{i+1}")(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout_rate,
            name=f"mha_block_{i+1}"
        )(x1, x1)
        x2 = tf.keras.layers.Add(name=f"add1_block_{i+1}")([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_block_{i+1}")(x2)
        mlp = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu, name=f"mlp_dense1_block_{i+1}")(x3)
        mlp = tf.keras.layers.Dropout(dropout_rate, name=f"mlp_drop1_block_{i+1}")(mlp)
        mlp = tf.keras.layers.Dense(projection_dim, name=f"mlp_dense2_block_{i+1}")(mlp)
        mlp = tf.keras.layers.Dropout(dropout_rate, name=f"mlp_drop2_block_{i+1}")(mlp)
        x = tf.keras.layers.Add(name=f"add2_block_{i+1}")([mlp, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    representation = tf.keras.layers.GlobalAveragePooling1D(name="gap_vit")(representation)
    representation = tf.keras.layers.Dropout(dropout_rate, name="gap_dropout")(representation)

    features = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu, name="features_dense")(representation)
    features = tf.keras.layers.Dropout(dropout_rate, name="features_dropout")(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NeuroVision_Vision_Transformer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
