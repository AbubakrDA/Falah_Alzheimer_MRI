"""NeuroVision OASIS-1 Dataset Loader

Coordinates directory detection, image discovery, metadata ingestion, clinical label mapping,
and slice extraction for the OASIS-1 (Open Access Series of Imaging Studies) dataset.
Provides normalized 2D/3D tensor generation and clinical age/demographic analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image

from src.data.oasis1_validator import OASIS1Validator
from src.data.oasis1_metadata import OASIS1MetadataManager
from src.data.oasis1_label_mapper import OASIS1LabelMapper
from src.data.oasis1_manifest import OASIS1ManifestManager


class OASIS1DatasetLoader:
    """Unified manager and loader for the OASIS-1 neuroimaging dataset."""

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        task: str = "multiclass",
        target_size: Tuple[int, int] = (224, 224),
        slice_orientation: str = "axial",
        split_seed: int = 42,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path("data/oasis1")
        self.task = task.lower()
        self.target_size = target_size
        self.slice_orientation = slice_orientation.lower()
        self.split_seed = split_seed

        # Initialize sub-modules
        self.validator = OASIS1Validator(self.base_dir)
        self.metadata_mgr = OASIS1MetadataManager(self.base_dir / "metadata")
        self.label_mapper = OASIS1LabelMapper(task=self.task)
        self.manifest_mgr = OASIS1ManifestManager(output_dir=self.base_dir, split_seed=self.split_seed)

        self.discovered_scans: List[Dict[str, Any]] = []
        self.matched_records: List[Dict[str, Any]] = []
        self.annotated_records: List[Dict[str, Any]] = []
        self.participant_split_map: Dict[str, str] = {}

    def scan_and_index(self) -> Dict[str, Any]:
        """Discovers physical images on disk, parses clinical metadata, and links them."""
        # Ensure directory dropzones exist
        self.validator.ensure_directories()

        # Discover NIfTI and Analyze 7.5 scans
        scan_discovery = self.validator.discover_images()
        all_scans = []
        all_scans.extend(scan_discovery["nifti_images"])
        all_scans.extend(scan_discovery["analyze_pairs"])
        self.discovered_scans = all_scans

        # Ingest clinical metadata
        meta_records = self.metadata_mgr.load_metadata()

        # Match scans with metadata
        match_res = self.metadata_mgr.match_scans_to_metadata(self.discovered_scans)
        self.matched_records = match_res["matched_records"]

        # Map clinical CDR to research labels
        self.annotated_records = self.label_mapper.map_dataset(self.matched_records)

        # Compute participant-level split
        self.participant_split_map = self.manifest_mgr.perform_participant_split(
            self.annotated_records, seed=self.split_seed
        )

        return {
            "total_scans_found": len(self.discovered_scans),
            "nifti_count": len(scan_discovery["nifti_images"]),
            "analyze_count": len(scan_discovery["analyze_pairs"]),
            "orphaned_analyze_count": len(scan_discovery["orphaned_analyze"]),
            "total_metadata_rows": len(meta_records),
            "matched_records_count": len(self.matched_records),
            "research_task": self.task,
            "classes": self.label_mapper.target_classes,
        }

    def generate_manifest(self) -> Path:
        """Exports the complete oasis1_manifest.csv file."""
        if not self.annotated_records:
            self.scan_and_index()
        return self.manifest_mgr.generate_manifest(
            self.annotated_records, self.participant_split_map
        )

    def extract_slice_from_volume(
        self,
        volume_data: np.ndarray,
        slice_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Extracts and normalizes a 2D slice from a 3D MRI volume array.
        
        Applies 1st-99th percentile intensity clipping and min-max scaling to [0, 1].
        """
        if volume_data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got array with shape {volume_data.shape}")

        if self.slice_orientation == "axial":
            # Axial slice along z-axis (axis 2 or 0 depending on storage)
            idx = slice_idx if slice_idx is not None else volume_data.shape[2] // 2
            slice_2d = volume_data[:, :, idx]
        elif self.slice_orientation == "coronal":
            idx = slice_idx if slice_idx is not None else volume_data.shape[1] // 2
            slice_2d = volume_data[:, idx, :]
        elif self.slice_orientation == "sagittal":
            idx = slice_idx if slice_idx is not None else volume_data.shape[0] // 2
            slice_2d = volume_data[idx, :, :]
        else:
            raise ValueError(f"Unknown slice orientation: {self.slice_orientation}")

        # Robust medical intensity normalization
        p1, p99 = np.percentile(slice_2d, (1.0, 99.0))
        clipped = np.clip(slice_2d, p1, p99)

        if p99 > p1:
            norm_slice = (clipped - p1) / (p99 - p1)
        else:
            norm_slice = np.zeros_like(clipped, dtype=np.float32)

        # Convert to PIL for standardized resizing
        pil_img = Image.fromarray((norm_slice * 255.0).astype(np.uint8))
        if pil_img.size != self.target_size:
            pil_img = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to 3-channel RGB float array [0, 1]
        rgb_img = pil_img.convert("RGB")
        return np.asarray(rgb_img, dtype=np.float32) / 255.0

    def analyze_demographics_and_age(self) -> Dict[str, Any]:
        """Analyzes age and demographic distributions across clinical stages.
        
        Critical for avoiding confounding age-related brain atrophy with Alzheimer's pathology.
        """
        if not self.annotated_records:
            return {"error": "No indexed records available"}

        age_by_class: Dict[str, List[float]] = {}
        sex_by_class: Dict[str, Dict[str, int]] = {}

        for rec in self.annotated_records:
            meta = rec.get("metadata", {})
            labeling = rec.get("clinical_labeling", {})
            lbl = labeling.get("research_label", "Unrated")

            age = meta.get("age")
            if age is not None:
                age_by_class.setdefault(lbl, []).append(float(age))

            sex = str(meta.get("sex", "U")).upper()
            sex_by_class.setdefault(lbl, {})
            sex_by_class[lbl][sex] = sex_by_class[lbl].get(sex, 0) + 1

        summary = {}
        for lbl, ages in age_by_class.items():
            summary[lbl] = {
                "count": len(ages),
                "mean_age": round(float(np.mean(ages)), 2),
                "std_age": round(float(np.std(ages)), 2),
                "min_age": round(float(np.min(ages)), 2),
                "max_age": round(float(np.max(ages)), 2),
                "sex_distribution": sex_by_class.get(lbl, {}),
            }

        return {
            "demographic_summary": summary,
            "scientific_note": (
                "In OASIS-1, demented subjects are substantially older on average than "
                "the non-demented cohort. Machine learning models can inadvertently learn "
                "normal age-related ventricular enlargement rather than dementia-specific "
                "hippocampal atrophy if age is not controlled for."
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns directory dropzones and pipeline status for Streamlit UI."""
        raw_p = self.base_dir / "raw"
        ext_p = self.base_dir / "extracted"
        meta_p = self.base_dir / "metadata"
        proc_p = self.base_dir / "processed"
        for p in [raw_p, ext_p, meta_p, proc_p]:
            p.mkdir(parents=True, exist_ok=True)

        return {
            "official_source": "https://sites.wustl.edu/oasisbrains/oasis-1/",
            "raw_dir": str(raw_p),
            "extracted_dir": str(ext_p),
            "metadata_dir": str(meta_p),
            "processed_dir": str(proc_p),
            "task": self.task,
            "classes": self.label_mapper.target_classes,
        }

    @property
    def manifest_df(self):
        """Returns the active manifest DataFrame or loads existing manifest file."""
        import pandas as pd
        manifest_p = self.base_dir / "oasis1_manifest.csv"
        proc_manifest_p = self.base_dir / "processed" / "oasis1_manifest.csv"
        for p in [manifest_p, proc_manifest_p]:
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame()


# Case-insensitive alias
Oasis1DatasetLoader = OASIS1DatasetLoader

__all__ = [
    "OASIS1DatasetLoader",
    "Oasis1DatasetLoader",
]
