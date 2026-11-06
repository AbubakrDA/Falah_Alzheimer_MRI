"""NeuroVision OASIS-1 Clinical Metadata Parser & Matcher

Parses CSV and Excel clinical metadata for the OASIS-1 cross-sectional study.
Standardizes clinical variables (ID, Age, M/F, CDR, MMSE, eTIV, nWBV, ASF),
extracts participant/session IDs, and matches physical MRI scans to clinical records.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import csv
import pandas as pd


# Known column aliases across different OASIS-1 releases & distributions
COLUMN_ALIASES = {
    "session_id": ["ID", "Subject_ID", "Subject ID", "OASIS_ID", "session_id", "session"],
    "sex": ["M/F", "Gender", "Sex", "gender", "sex"],
    "hand": ["Hand", "Handedness", "hand"],
    "age": ["Age", "AGE", "age"],
    "educ": ["Educ", "Education", "EDUC", "education"],
    "ses": ["SES", "Socioeconomic_Status", "ses"],
    "mmse": ["MMSE", "Mini_Mental", "mmse"],
    "cdr": ["CDR", "Clinical_Dementia_Rating", "cdr"],
    "etiv": ["eTIV", "ETIV", "Estimated_Total_Intracranial_Volume", "etiv"],
    "nwbv": ["nWBV", "NWBV", "Normalized_Whole_Brain_Volume", "nwbv"],
    "asf": ["ASF", "Atlas_Scaling_Factor", "asf"],
    "delay": ["Delay", "delay"],
}


def extract_ids_from_session_string(raw_id: str) -> Tuple[str, str]:
    """Extracts standardized participant_id and session_id from OASIS strings.
    
    Examples:
        'OAS1_0001_MR1' -> participant_id: 'OAS1_0001', session_id: 'OAS1_0001_MR1'
        'OASIS1_0001_MR1' -> participant_id: 'OASIS1_0001', session_id: 'OASIS1_0001_MR1'
    """
    clean_id = str(raw_id).strip()
    match = re.match(r"(OAS(?:IS)?1_\d+)(?:_MR(\d+))?", clean_id, re.IGNORECASE)
    if match:
        part_id = match.group(1).upper()
        sess_num = match.group(2) if match.group(2) else "1"
        sess_id = f"{part_id}_MR{sess_num}"
        return part_id, sess_id

    parts = clean_id.split("_")
    if len(parts) >= 2:
        part_id = "_".join(parts[:2])
        return part_id, clean_id

    return clean_id, clean_id


class OASIS1MetadataManager:
    """Imports, cleans, and matches OASIS-1 clinical records with physical scans."""

    def __init__(self, metadata_path_or_dir: Optional[Union[str, Path]] = None):
        p = Path(metadata_path_or_dir) if metadata_path_or_dir else Path("data/oasis1/metadata")
        if p.is_file():
            self.metadata_file = p
            self.metadata_dir = p.parent
        else:
            self.metadata_file = None
            self.metadata_dir = p

        self.records: List[Dict[str, Any]] = []
        self.raw_columns: List[str] = []

        if self.metadata_file and self.metadata_file.exists():
            self.load_metadata(self.metadata_file)

    @property
    def normalized_df(self) -> pd.DataFrame:
        """Returns records as a normalized pandas DataFrame for Streamlit display."""
        if not self.records and self.metadata_file:
            self.load_metadata(self.metadata_file)
        if not self.records:
            return pd.DataFrame()
        # Strip out private dictionary keys
        clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.records]
        return pd.DataFrame(clean_rows)

    def discover_metadata_files(self) -> List[Path]:
        """Discovers CSV and Excel metadata files in the metadata directory."""
        if not self.metadata_dir.exists():
            return []

        found = []
        for ext in ["*.csv", "*.xlsx", "*.xls"]:
            found.extend(list(self.metadata_dir.glob(ext)))
        return sorted(found)

    def load_metadata(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Parses clinical metadata from CSV or Excel file."""
        target_file = Path(file_path) if file_path else self.metadata_file
        if target_file is None:
            discovered = self.discover_metadata_files()
            if not discovered:
                return []
            target_file = discovered[0]

        if not target_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {target_file}")

        self.metadata_file = target_file
        ext = target_file.suffix.lower()
        rows_data = []

        if ext == ".csv":
            with open(target_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self.raw_columns = reader.fieldnames or []
                for row in reader:
                    rows_data.append(dict(row))

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(target_file)
            self.raw_columns = list(df.columns)
            rows_data = df.to_dict(orient="records")

        # Standardize records
        self.records = [self._standardize_row(r) for r in rows_data]
        return self.records

    def _standardize_row(self, raw_row: Dict[str, Any]) -> Dict[str, Any]:
        """Maps diverse column headers to canonical clinical variable names."""
        canonical: Dict[str, Any] = {}

        col_to_canonical = {}
        for canon, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                col_to_canonical[alias.lower()] = canon

        for k, v in raw_row.items():
            canon_name = col_to_canonical.get(str(k).strip().lower())
            if canon_name:
                canonical[canon_name] = v

        raw_sess = str(canonical.get("session_id", "")).strip()
        part_id, sess_id = extract_ids_from_session_string(raw_sess)
        canonical["participant_id"] = part_id
        canonical["session_id"] = sess_id

        raw_cdr = canonical.get("cdr")
        canonical["original_cdr"] = raw_cdr
        if raw_cdr is not None and str(raw_cdr).strip() not in ["", "NA", "NaN", "None"]:
            try:
                canonical["cdr"] = float(raw_cdr)
            except ValueError:
                canonical["cdr"] = None
        else:
            canonical["cdr"] = None

        raw_age = canonical.get("age")
        if raw_age is not None and str(raw_age).strip() not in ["", "NA", "NaN", "None"]:
            try:
                canonical["age"] = float(raw_age)
            except ValueError:
                canonical["age"] = None

        raw_mmse = canonical.get("mmse")
        if raw_mmse is not None and str(raw_mmse).strip() not in ["", "NA", "NaN", "None"]:
            try:
                canonical["mmse"] = float(raw_mmse)
            except ValueError:
                canonical["mmse"] = None

        canonical["_raw_row"] = raw_row
        return canonical

    def match_with_mri_records(self, extracted_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Method interface for matching extracted scan records against metadata."""
        res = self.match_scans_to_metadata(extracted_records)
        missing_cdr = sum(1 for r in self.records if r.get("cdr") is None)
        return {
            "matched_mris": res["matched_count"],
            "unmatched_mris": res["unmatched_scans_count"],
            "missing_cdr_count": missing_cdr,
            "matched_records": res["matched_records"],
            "unmatched_scans": res["unmatched_scans"],
            "unmatched_metadata": res["unmatched_metadata"],
        }

    def match_scans_to_metadata(
        self,
        discovered_scans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Matches physical image scans to metadata rows using participant & session IDs."""
        metadata_by_session: Dict[str, Dict[str, Any]] = {}
        metadata_by_participant: Dict[str, List[Dict[str, Any]]] = {}

        for rec in self.records:
            sess_id = rec.get("session_id")
            part_id = rec.get("participant_id")

            if sess_id:
                metadata_by_session[sess_id.upper()] = rec
            if part_id:
                metadata_by_participant.setdefault(part_id.upper(), []).append(rec)

        matched_scans = []
        unmatched_scans = []
        matched_session_ids = set()

        for scan in discovered_scans:
            path_str = scan.get("file_path", "") or scan.get("header_path", "") or scan.get("stem", "")
            part_id, sess_id = extract_ids_from_session_string(path_str)

            meta_match = metadata_by_session.get(sess_id.upper())
            if not meta_match and part_id.upper() in metadata_by_participant:
                meta_match = metadata_by_participant[part_id.upper()][0]

            if meta_match:
                matched_session_ids.add(meta_match["session_id"].upper())
                matched_scans.append({
                    "scan_info": scan,
                    "metadata": meta_match,
                    "match_status": "matched",
                })
            else:
                unmatched_scans.append({
                    "scan_info": scan,
                    "extracted_participant_id": part_id,
                    "extracted_session_id": sess_id,
                    "match_status": "unmatched_scan",
                })

        unmatched_metadata = [
            rec for rec in self.records
            if rec.get("session_id", "").upper() not in matched_session_ids
        ]

        return {
            "total_metadata_records": len(self.records),
            "total_scans_provided": len(discovered_scans),
            "matched_count": len(matched_scans),
            "unmatched_scans_count": len(unmatched_scans),
            "unmatched_metadata_count": len(unmatched_metadata),
            "matched_records": matched_scans,
            "unmatched_scans": unmatched_scans,
            "unmatched_metadata": unmatched_metadata,
        }


# Case-insensitive alias
Oasis1MetadataManager = OASIS1MetadataManager
