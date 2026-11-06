import sys
import importlib
import streamlit as st
import json
import pandas as pd
from pathlib import Path

# Hot-reload any updated src modules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src."):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.config.settings import EXPERIMENTS_DIR, DISCLAIMER_TEXT

st.set_page_config(page_title="Model Comparison - NeuroVision", page_icon="⚖️", layout="wide")

st.title("⚖️ Comparative Model Benchmarking")
st.warning(f"⚠️ **PROTOTYPE NOTICE:** {DISCLAIMER_TEXT}")

st.markdown("""
Compares architectural performance across Custom CNN, ResNet50 Transfer, and Vision Transformer.
**Methodological Integrity Rule:** Never compare metrics across Falah and OASIS-1 as directly equivalent.
""")

exp_files = list(EXPERIMENTS_DIR.glob("**/experiment.json")) if EXPERIMENTS_DIR.exists() else []

if not exp_files:
    st.info("No experiment logs found in `experiments/`. Run model training runs first in the **Model Training** page.")
else:
    records = []
    for f in exp_files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                records.append({
                    "Experiment ID": data.get("experiment_id", f.parent.name),
                    "Dataset": data.get("dataset_name", "Unknown"),
                    "Architecture": data.get("architecture", "Unknown"),
                    "Epochs": data.get("epochs_completed", 0),
                    "Best Val Acc (%)": round(data.get("best_val_accuracy", 0.0) * 100, 2) if data.get("best_val_accuracy") else None,
                    "Best Val Loss": round(data.get("best_val_loss", 0.0), 4) if data.get("best_val_loss") else None,
                    "Training Time (s)": data.get("training_time_seconds", 0.0),
                    "Learning Rate": data.get("learning_rate", 1e-4)
                })
        except Exception:
            pass

    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Performance Comparison Chart")
        valid_df = df.dropna(subset=["Best Val Acc (%)"])
        if not valid_df.empty:
            chart_df = valid_df.set_index("Architecture")[["Best Val Acc (%)"]]
            st.bar_chart(chart_df)
    else:
        st.info("No valid experiment records could be parsed.")
