"""NeuroVision Sample Image Extraction Script

Extracts authentic representative MRI slices for all 4 cognitive impairment stages
(Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented) from both the
TRAIN and TEST splits of the Falah Alzheimer MRI Arrow dataset.

Saves individual sample PNGs and a combined comparison grid into docs/assets/samples/
for direct inclusion in README.md, Streamlit UI, and research reports.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import Falah loader
try:
    from src.data.falah_loader import FalahDatasetLoader, FALAH_CLASSES
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.falah_loader import FalahDatasetLoader, FALAH_CLASSES


def extract_and_save_samples(output_dir: Path = None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "docs" / "assets" / "samples"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[NeuroVision] Initializing Falah Dataset Loader...")
    loader = FalahDatasetLoader()
    
    class_names = loader.class_names
    print(f"[NeuroVision] Found {len(class_names)} classes: {class_names}")

    # Canonical order for medical staging
    staging_order = [
        "Non_Demented",
        "Very_Mild_Demented",
        "Mild_Demented",
        "Moderate_Demented"
    ]

    saved_files = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        ds = loader.raw_train if split == "train" else loader.raw_test
        print(f"[NeuroVision] Scanning {split} split ({len(ds)} images)...")
        
        found_classes = set()
        for idx in range(len(ds)):
            item = ds[idx]
            lbl = int(item["label"])
            c_name = class_names[lbl]
            
            if c_name not in found_classes and c_name in staging_order:
                img = item["image"].convert("RGB")
                
                # File naming
                slug = c_name.lower()
                filename = f"{split}_{slug}.png"
                filepath = output_dir / filename
                
                # Save native resolution PNG
                img.save(filepath, format="PNG")
                saved_files[split][c_name] = {
                    "path": str(filepath),
                    "filename": filename,
                    "index": idx,
                    "size": img.size,
                    "image": img
                }
                found_classes.add(c_name)
                print(f"  -> Saved {split.upper()} [{c_name}] (index {idx}, size {img.size}) to {filename}")
            
            if len(found_classes) == len(staging_order):
                break

    # Build an aggregated 2x4 visual comparison grid (Row 1: Train, Row 2: Test)
    grid_w, grid_h = 224, 224
    header_h = 40
    gutter = 12
    margin = 20

    total_w = margin * 2 + (grid_w * 4) + (gutter * 3)
    total_h = margin * 2 + (grid_h * 2) + (gutter * 1) + (header_h * 2) + 60

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Title
    title_text = "NeuroVision 2.0 - Representative Brain MRI Staging Gallery (Train vs. Test)"
    draw.text((margin, 15), title_text, fill=(240, 245, 255))

    splits_to_draw = [
        ("Training Cohort (80% Partition)", "train", 55),
        ("Testing Cohort (20% Evaluation Partition)", "test", 55 + grid_h + header_h + gutter)
    ]

    for label_split, split_key, y_start in splits_to_draw:
        draw.text((margin, y_start), label_split, fill=(100, 200, 255))
        
        for c_idx, c_name in enumerate(staging_order):
            x_pos = margin + c_idx * (grid_w + gutter)
            y_img = y_start + 25
            
            info = saved_files[split_key].get(c_name)
            if info:
                resized_img = info["image"].resize((grid_w, grid_h), Image.Resampling.LANCZOS)
                canvas.paste(resized_img, (x_pos, y_img))
                
                # Class label banner below image
                display_label = c_name.replace("_", " ")
                draw.rectangle([x_pos, y_img, x_pos + grid_w, y_img + 22], fill=(0, 0, 0, 180))
                draw.text((x_pos + 6, y_img + 4), display_label, fill=(255, 255, 255))

    grid_path = output_dir / "falah_mri_samples_grid.png"
    canvas.save(grid_path, format="PNG")
    print(f"\n[NeuroVision] Aggregated gallery saved to: {grid_path}")
    print("[NeuroVision] Extraction complete! All 8 individual PNGs + 1 master gallery created successfully.")
    return saved_files


if __name__ == "__main__":
    extract_and_save_samples()

