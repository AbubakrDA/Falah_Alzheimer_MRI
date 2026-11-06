"""NeuroVision OASIS-1 Clinical Label Mapping Engine

Maps Clinical Dementia Rating (CDR) scores to standardized research labels
for both Binary and Multiclass cognitive impairment tasks.
Preserves original clinical scores, audit fields, and provenance notes.
"""

from typing import Dict, List, Optional, Tuple, Any, Union

# Versioning for reproducible label mapping
LABEL_MAPPING_VERSION = "1.0.0"

# Canonical research label sets
BINARY_CLASSES = ["Non-Demented", "Demented"]
MULTICLASS_CLASSES = [
    "Non-Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]


class OASIS1LabelMapper:
    """Standardizes CDR scores into research labels for binary and multiclass tasks."""

    def __init__(self, task: str = "multiclass", mapping_version: str = LABEL_MAPPING_VERSION):
        self.task = task.lower()
        if self.task not in ["binary", "multiclass"]:
            raise ValueError(f"Unsupported research task: '{task}'. Choose 'binary' or 'multiclass'.")
        self.mapping_version = mapping_version

    @property
    def target_classes(self) -> List[str]:
        """Returns the active list of clinical class names."""
        return BINARY_CLASSES if self.task == "binary" else MULTICLASS_CLASSES

    def map_record(self, metadata_record: Dict[str, Any]) -> Dict[str, Any]:
        """Maps an individual clinical metadata record into a research diagnosis.
        
        Preserves original values and returns structured audit fields.
        """
        raw_cdr = metadata_record.get("cdr")
        original_cdr_str = metadata_record.get("original_cdr")
        original_diag = metadata_record.get("original_diagnosis", "Clinical_CDR_Assessment")
        age = metadata_record.get("age")

        # Handle missing or unassessed CDR values
        if raw_cdr is None:
            reason = "CDR not evaluated (common in younger control participants <60 yrs)" if (age and age < 60) else "Missing CDR score in metadata"
            return {
                "original_cdr": original_cdr_str,
                "original_diagnosis": original_diag,
                "research_label": "Unrated / Excluded",
                "label_id": -1,
                "research_task": self.task,
                "label_mapping_version": self.mapping_version,
                "label_mapping_reason": reason,
                "is_usable_for_training": False,
            }

        cdr_val = float(raw_cdr)

        # 1. Binary Task Mapping
        if self.task == "binary":
            if cdr_val == 0.0:
                label = "Non-Demented"
                label_id = 0
                reason = "CDR 0 indicates cognitively normal"
            elif cdr_val in [0.5, 1.0, 2.0]:
                label = "Demented"
                label_id = 1
                reason = f"CDR {cdr_val} indicates clinical cognitive impairment"
            else:
                return {
                    "original_cdr": original_cdr_str,
                    "original_diagnosis": original_diag,
                    "research_label": "Unsupported CDR Value",
                    "label_id": -1,
                    "research_task": self.task,
                    "label_mapping_version": self.mapping_version,
                    "label_mapping_reason": f"Unexpected CDR score: {cdr_val}",
                    "is_usable_for_training": False,
                }

        # 2. Multiclass Task Mapping
        else:
            if cdr_val == 0.0:
                label = "Non-Demented"
                label_id = 0
                reason = "CDR 0: No dementia (normal cognition)"
            elif cdr_val == 0.5:
                label = "Very Mild Dementia"
                label_id = 1
                reason = "CDR 0.5: Questionable / Very Mild Dementia"
            elif cdr_val == 1.0:
                label = "Mild Dementia"
                label_id = 2
                reason = "CDR 1: Mild Dementia"
            elif cdr_val == 2.0:
                label = "Moderate Dementia"
                label_id = 3
                reason = "CDR 2: Moderate Dementia"
            else:
                return {
                    "original_cdr": original_cdr_str,
                    "original_diagnosis": original_diag,
                    "research_label": "Unsupported CDR Value",
                    "label_id": -1,
                    "research_task": self.task,
                    "label_mapping_version": self.mapping_version,
                    "label_mapping_reason": f"Unexpected CDR score: {cdr_val}",
                    "is_usable_for_training": False,
                }

        return {
            "original_cdr": original_cdr_str,
            "original_diagnosis": original_diag,
            "research_label": label,
            "label_id": label_id,
            "research_task": self.task,
            "label_mapping_version": self.mapping_version,
            "label_mapping_reason": reason,
            "is_usable_for_training": True,
        }

    def map_dataset(self, matched_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Maps an entire list of matched scan-metadata records."""
        annotated = []
        for rec in matched_records:
            meta = rec.get("metadata", rec)
            mapping_result = self.map_record(meta)
            merged = dict(rec)
            merged["clinical_labeling"] = mapping_result
            annotated.append(merged)
        return annotated


# Case-insensitive alias
Oasis1LabelMapper = OASIS1LabelMapper
