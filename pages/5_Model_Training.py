import sys
import importlib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Hot-reload any updated src modules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src."):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DISCLAIMER_TEXT
)
from src.data.dataset_manager import DatasetManager
from src.training.trainer import ModelTrainer

# Safe check for deep learning runtimes availability
try:
    import torch
except Exception:
    torch = None

try:
    import tensorflow as tf
except Exception:
    tf = None

st.set_page_config(page_title="Model Training - NeuroVision", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Training & Experiment Tracking")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("1. Framework & Architecture Selection")
    
    framework_options = []
    if torch is not None:
        framework_options.append("PyTorch (Native & GPU/CPU Accelerated)")
    if tf is not None:
        framework_options.append("TensorFlow / Keras")

    # Simulation engine is always available as a zero-dependency fallback
    framework_options.append("🔬 Simulation Engine (NumPy Baseline — No GPU/Torch Required)")

    if torch is None and tf is None:
        st.info("""
        💡 **Neither PyTorch nor TensorFlow is currently installed in this Python environment.**

        To enable native deep learning GPU/CPU training, run this in your terminal:
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ```
        *(Or for full NVIDIA CUDA GPU support, run: `pip install torch torchvision`)*

        **In the meantime, you can use the Simulation Engine below to test the complete training, metrics, and visualization pipeline immediately!**
        """)
        
    selected_framework = st.selectbox("Deep Learning Framework", framework_options, index=0)
    framework_code = (
        "torch" if "PyTorch" in selected_framework
        else ("tensorflow" if "TensorFlow" in selected_framework else "simulation")
    )

    active_dataset = st.selectbox("Training Dataset", ["Falah_Alzheimer_MRI", "OASIS-1"])
    task = "multiclass"
    if active_dataset == "OASIS-1":
        task = st.radio("OASIS-1 Task", ["multiclass", "binary"])
        
    architecture = st.selectbox(
        "Model Architecture",
        ["Custom CNN", "ResNet18 Transfer", "ResNet50 Transfer", "Vision Transformer"]
    )
    
    st.subheader("2. Hyperparameters")
    epochs = st.slider("Epochs", 1, 30, 5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    learning_rate = st.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
    use_weights = st.checkbox("Enforce Balanced Class Weighting", value=True, help="Crucial for minority classes like Moderate Demented (<1%)")
    early_stop = st.checkbox("Early Stopping on Validation Loss", value=True)
    patience = st.slider("Early Stopping Patience", 2, 8, 3) if early_stop else 3

with c2:
    st.subheader("3. Execution & Experiment Metadata")
    exp_name = st.text_input("Experiment Run Name (Optional)", value="")
    
    st.info("""
    **Dataset Provenance Rule:**
    - Falah models are trained strictly on Falah training data and validated on a stratified holdout from training only.
    - Test data is NEVER used during training or hyperparameter tuning.
    - All experiment parameters and checkpoints are stored under `models/`.
    """)

    if st.button("🚀 Start Training Experiment", type="primary"):
        with st.spinner("Preparing dataset and building model..."):
            mgr = DatasetManager(default_dataset=active_dataset, oasis_task=task)
            class_names = mgr.get_classes()
            num_classes = len(class_names)
            channels = 3
            
            trainer = ModelTrainer(
                experiment_name=exp_name if exp_name.strip() else None,
                dataset_name=active_dataset,
                task=task,
                framework=framework_code,
            )
            
            model = trainer.build_model(
                architecture=architecture,
                input_shape=(128, 128, channels),
                num_classes=num_classes,
                learning_rate=learning_rate,
                classes=class_names,
            )
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def on_epoch_progress(epoch, logs):
                pct = epoch / epochs
                progress_bar.progress(pct)
                val_acc = logs.get("val_accuracy", 0.0) * 100
                train_loss = logs.get("loss", 0.0)
                status_text.text(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
                
            if framework_code == "torch":
                train_ds = mgr.get_torch_dataloader("train", batch_size=batch_size, channels=channels, shuffle=True)
                val_ds = mgr.get_torch_dataloader("val", batch_size=batch_size, channels=channels, shuffle=False)
            elif framework_code == "tensorflow":
                train_ds = mgr.get_tf_dataset("train", batch_size=batch_size, channels=channels, shuffle=True, augment=True)
                val_ds = mgr.get_tf_dataset("val", batch_size=batch_size, channels=channels, shuffle=False)
            else:
                train_ds = mgr.get_numpy_data("train", channels=channels)
                val_ds = mgr.get_numpy_data("val", channels=channels)
                
            class_weights = mgr.get_class_weights() if use_weights else None

            # Train
            meta = trainer.train(
                train_data=train_ds,
                val_data=val_ds,
                epochs=epochs,
                batch_size=batch_size,
                class_weights=class_weights,
                use_early_stopping=early_stop,
                patience=patience,
                progress_callback=on_epoch_progress
            )
            
            st.success(f"🎉 Training complete! Checkpoint saved to: `{meta['checkpoint_path']}`")
            
            # Plot curves
            hist = meta["history"]
            if "loss" in hist and "val_loss" in hist:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.plot(hist["loss"], label="Train Loss", marker="o")
                ax1.plot(hist["val_loss"], label="Val Loss", marker="s")
                ax1.set_title("Cross-Entropy Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

                ax2.plot(hist["accuracy"], label="Train Accuracy", marker="o")
                ax2.plot(hist["val_accuracy"], label="Val Accuracy", marker="s")
                ax2.set_title("Classification Accuracy")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
