"""Synthetic Integration and Unit Tests for OASIS-1 Pipeline

Validates directory detection, archive validation, manual extraction safeguards,
Analyze 7.5 pair discovery, NIfTI discovery, CSV/Excel metadata import,
ID extraction, MRI-to-metadata matching, CDR label mapping, participant-level splitting,
leakage prevention, and manifest export using synthetic mock data.
Does NOT require real OASIS downloads.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import tarfile
import zipfile
import csv
import numpy as np

from src.data.oasis1_validator import OASIS1Validator
from src.data.oasis1_metadata import OASIS1MetadataManager, extract_ids_from_session_string
from src.data.oasis1_label_mapper import OASIS1LabelMapper
from src.data.oasis1_manifest import OASIS1ManifestManager
from src.data.oasis1_loader import OASIS1DatasetLoader


class TestOASIS1SyntheticIntegration(unittest.TestCase):
    """End-to-end synthetic verification suite for OASIS-1 modules."""

    def setUp(self):
        # Create temporary working directory for synthetic testing
        self.temp_dir = tempfile.mkdtemp(prefix="neurovision_oasis_test_")
        self.base_dir = Path(self.temp_dir)
        self.raw_dir = self.base_dir / "raw"
        self.extracted_dir = self.base_dir / "extracted"
        self.metadata_dir = self.base_dir / "metadata"

        self.raw_dir.mkdir(parents=True)
        self.extracted_dir.mkdir(parents=True)
        self.metadata_dir.mkdir(parents=True)

        self._create_synthetic_environment()

    def tearDown(self):
        # Clean up temporary test directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_environment(self):
        """Builds synthetic archives, images, and clinical metadata tables."""
        # 1. Synthetic Archive (.tar.gz and .zip)
        dummy_file = self.base_dir / "dummy.txt"
        dummy_file.write_text("synthetic neuroimaging payload")

        self.tar_archive = self.raw_dir / "oasis_synthetic_disc1.tar.gz"
        with tarfile.open(self.tar_archive, "w:gz") as tar:
            tar.add(dummy_file, arcname="disc1/dummy.txt")

        self.zip_archive = self.raw_dir / "oasis_synthetic_disc2.zip"
        with zipfile.ZipFile(self.zip_archive, "w") as zf:
            zf.write(dummy_file, arcname="disc2/dummy.txt")

        dummy_file.unlink()

        # 2. Synthetic NIfTI files
        sess1_dir = self.extracted_dir / "OAS1_0001_MR1"
        sess1_dir.mkdir(parents=True)
        (sess1_dir / "mpr-1.nii.gz").write_bytes(b"\x1f\x8b\x08dummy_nii_gz")

        sess1_rep_dir = self.extracted_dir / "OAS1_0001_MR2"  # Repeat reliability scan
        sess1_rep_dir.mkdir(parents=True)
        (sess1_rep_dir / "mpr-1.nii").write_bytes(b"dummy_nii")

        # 3. Synthetic Analyze 7.5 Paired & Orphaned Files
        sess2_dir = self.extracted_dir / "OAS1_0002_MR1"
        sess2_dir.mkdir(parents=True)
        (sess2_dir / "mpr-1.hdr").write_bytes(b"\x00" * 348)  # Standard 348-byte header
        (sess2_dir / "mpr-1.img").write_bytes(b"\x00" * 1024)

        sess3_dir = self.extracted_dir / "OAS1_0003_MR1"
        sess3_dir.mkdir(parents=True)
        (sess3_dir / "mpr-1.hdr").write_bytes(b"\x00" * 348)
        (sess3_dir / "mpr-1.img").write_bytes(b"\x00" * 1024)

        sess4_dir = self.extracted_dir / "OAS1_0004_MR1"
        sess4_dir.mkdir(parents=True)
        (sess4_dir / "mpr-1.hdr").write_bytes(b"\x00" * 348)
        (sess4_dir / "mpr-1.img").write_bytes(b"\x00" * 1024)

        sess5_dir = self.extracted_dir / "OAS1_0005_MR1"
        sess5_dir.mkdir(parents=True)
        (sess5_dir / "mpr-1.hdr").write_bytes(b"\x00" * 348)
        (sess5_dir / "mpr-1.img").write_bytes(b"\x00" * 1024)

        # Orphaned Analyze file (header without image)
        sess_orph = self.extracted_dir / "OAS1_9999_MR1"
        sess_orph.mkdir(parents=True)
        (sess_orph / "orphan.hdr").write_bytes(b"\x00" * 348)

        # 4. Synthetic Clinical Metadata CSV
        self.csv_file = self.metadata_dir / "oasis_cross-sectional.csv"
        fieldnames = ["ID", "M/F", "Hand", "Age", "Educ", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
        rows = [
            # Participant 1: Session 1 & Session 2 (Reliability scan)
            {"ID": "OAS1_0001_MR1", "M/F": "F", "Hand": "R", "Age": "74", "Educ": "2", "SES": "3", "MMSE": "29", "CDR": "0", "eTIV": "1344", "nWBV": "0.743", "ASF": "1.306"},
            {"ID": "OAS1_0001_MR2", "M/F": "F", "Hand": "R", "Age": "74", "Educ": "2", "SES": "3", "MMSE": "30", "CDR": "0", "eTIV": "1344", "nWBV": "0.743", "ASF": "1.306"},
            # Participant 2: Very Mild Dementia (CDR 0.5)
            {"ID": "OAS1_0002_MR1", "M/F": "M", "Hand": "R", "Age": "55", "Educ": "4", "SES": "1", "MMSE": "28", "CDR": "0.5", "eTIV": "1145", "nWBV": "0.731", "ASF": "1.533"},
            # Participant 3: Mild Dementia (CDR 1.0)
            {"ID": "OAS1_0003_MR1", "M/F": "F", "Hand": "R", "Age": "73", "Educ": "4", "SES": "3", "MMSE": "27", "CDR": "1", "eTIV": "1454", "nWBV": "0.708", "ASF": "1.207"},
            # Participant 4: Moderate Dementia (CDR 2.0)
            {"ID": "OAS1_0004_MR1", "M/F": "M", "Hand": "R", "Age": "80", "Educ": "3", "SES": "2", "MMSE": "20", "CDR": "2", "eTIV": "1500", "nWBV": "0.680", "ASF": "1.170"},
            # Participant 5: Young control without CDR assessment
            {"ID": "OAS1_0005_MR1", "M/F": "F", "Hand": "R", "Age": "25", "Educ": "3", "SES": "2", "MMSE": "30", "CDR": "", "eTIV": "1400", "nWBV": "0.820", "ASF": "1.250"},
            # Participant 6: Metadata exists but MRI image missing on disk (unmatched metadata)
            {"ID": "OAS1_0006_MR1", "M/F": "M", "Hand": "R", "Age": "68", "Educ": "2", "SES": "4", "MMSE": "29", "CDR": "0", "eTIV": "1350", "nWBV": "0.750", "ASF": "1.300"},
        ]

        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_directory_detection(self):
        """A. Verifies detection of base, raw, extracted, and metadata paths."""
        validator = OASIS1Validator(self.base_dir)
        det = validator.detect_directories()
        self.assertTrue(det["base_dir_exists"])
        self.assertTrue(det["raw_dir_exists"])
        self.assertTrue(det["extracted_dir_exists"])
        self.assertTrue(det["metadata_dir_exists"])

    def test_raw_archive_validation(self):
        """B. Verifies discovery and integrity check of user-provided archive files."""
        validator = OASIS1Validator(self.base_dir)
        res = validator.validate_raw_archives()
        self.assertEqual(res["total_archives"], 2)
        self.assertTrue(res["all_valid"])

    def test_manual_archive_extraction_safeguard(self):
        """C. Verifies that extraction fails without explicit confirm=True and succeeds with it."""
        validator = OASIS1Validator(self.base_dir)

        # Unconfirmed extraction must fail
        unconf = validator.extract_archive("oasis_synthetic_disc1.tar.gz", confirm=False)
        self.assertFalse(unconf["extracted"])
        self.assertIn("confirm=True is required", unconf["message"])

        # Confirmed extraction must succeed
        dest = self.base_dir / "test_extract"
        conf = validator.extract_archive("oasis_synthetic_disc1.tar.gz", destination_dir=dest, confirm=True)
        self.assertTrue(conf["extracted"])
        self.assertTrue((dest / "disc1" / "dummy.txt").exists())

    def test_image_discovery(self):
        """D & E. Verifies discovery of NIfTI files and Analyze 7.5 paired files, catching orphans."""
        validator = OASIS1Validator(self.base_dir)
        discovery = validator.discover_images()

        self.assertEqual(len(discovery["nifti_images"]), 2)  # .nii.gz and .nii
        self.assertEqual(len(discovery["analyze_pairs"]), 4)  # 4 valid pairs
        self.assertEqual(len(discovery["orphaned_analyze"]), 1)  # 1 orphan
        self.assertEqual(discovery["total_valid_scans"], 6)

    def test_participant_and_session_id_extraction(self):
        """G & H. Verifies reliable extraction of participant IDs and session IDs."""
        part, sess = extract_ids_from_session_string("OAS1_0001_MR1")
        self.assertEqual(part, "OAS1_0001")
        self.assertEqual(sess, "OAS1_0001_MR1")

        part_rep, sess_rep = extract_ids_from_session_string("OAS1_0001_MR2")
        self.assertEqual(part_rep, "OAS1_0001")
        self.assertEqual(sess_rep, "OAS1_0001_MR2")

    def test_metadata_import_and_matching(self):
        """F, I & J. Verifies metadata loading, scan matching, and detection of unmatched records."""
        validator = OASIS1Validator(self.base_dir)
        discovery = validator.discover_images()
        all_scans = discovery["nifti_images"] + discovery["analyze_pairs"]

        meta_mgr = OASIS1MetadataManager(self.metadata_dir)
        records = meta_mgr.load_metadata(self.csv_file)
        self.assertEqual(len(records), 7)

        matched = meta_mgr.match_scans_to_metadata(all_scans)
        self.assertEqual(matched["matched_count"], 6)
        self.assertEqual(matched["unmatched_scans_count"], 0)
        self.assertEqual(matched["unmatched_metadata_count"], 1)  # OAS1_0006_MR1 has no scan on disk

    def test_clinical_label_mapping(self):
        """L. Verifies CDR mapping for both Binary and Multiclass tasks, and handles missing CDR."""
        # 1. Multiclass Mapper
        multi_mapper = OASIS1LabelMapper(task="multiclass")
        self.assertEqual(multi_mapper.map_record({"cdr": 0.0})["research_label"], "Non-Demented")
        self.assertEqual(multi_mapper.map_record({"cdr": 0.5})["research_label"], "Very Mild Dementia")
        self.assertEqual(multi_mapper.map_record({"cdr": 1.0})["research_label"], "Mild Dementia")
        self.assertEqual(multi_mapper.map_record({"cdr": 2.0})["research_label"], "Moderate Dementia")

        # Missing CDR (young control)
        young_res = multi_mapper.map_record({"cdr": None, "age": 25.0})
        self.assertEqual(young_res["research_label"], "Unrated / Excluded")
        self.assertFalse(young_res["is_usable_for_training"])

        # 2. Binary Mapper
        bin_mapper = OASIS1LabelMapper(task="binary")
        self.assertEqual(bin_mapper.map_record({"cdr": 0.0})["research_label"], "Non-Demented")
        self.assertEqual(bin_mapper.map_record({"cdr": 0.5})["research_label"], "Demented")
        self.assertEqual(bin_mapper.map_record({"cdr": 2.0})["research_label"], "Demented")

    def test_participant_level_splitting_and_leakage_prevention(self):
        """M. CRITICAL TEST: Verifies zero patient leakage across train, val, and test splits."""
        loader = OASIS1DatasetLoader(base_dir=self.base_dir, task="multiclass")
        index_res = loader.scan_and_index()
        self.assertEqual(index_res["matched_records_count"], 6)

        split_map = loader.participant_split_map

        # Participant OAS1_0001 has two scans: OAS1_0001_MR1 and OAS1_0001_MR2.
        # Both must reside in the exact same split!
        sess1_split = split_map["OAS1_0001"]
        self.assertIn(sess1_split, ["train", "val", "test"])

        # Check in the generated manifest rows
        manifest_path = loader.generate_manifest()
        self.assertTrue(manifest_path.exists())

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            p1_splits = [row["split"] for row in reader if row["participant_id"] == "OAS1_0001"]

        self.assertEqual(len(p1_splits), 2)
        # Slices/sessions from participant 1 must be in the identical split
        self.assertEqual(p1_splits[0], p1_splits[1], "Patient leakage detected! Scans for OAS1_0001 split across partitions.")

    def test_manifest_export_and_audit_fields(self):
        """N & O. Verifies export of oasis1_manifest.csv and all required audit columns."""
        loader = OASIS1DatasetLoader(base_dir=self.base_dir, task="multiclass")
        manifest_path = loader.generate_manifest()

        required_cols = [
            "participant_id", "session_id", "image_path", "diagnosis_source",
            "CDR", "MMSE", "age", "sex", "label", "dataset_name", "split",
            "original_diagnosis", "image_format", "image_validation_status",
            "metadata_match_status", "research_task", "label_mapping_version",
            "split_seed", "source_archive", "import_timestamp", "provenance_notes",
        ]

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in required_cols:
                self.assertIn(col, reader.fieldnames)

            rows = list(reader)
            self.assertEqual(len(rows), 6)

    def test_demographic_and_age_analysis(self):
        """Verifies age distribution and clinical demographic analysis."""
        loader = OASIS1DatasetLoader(base_dir=self.base_dir, task="multiclass")
        loader.scan_and_index()
        demo = loader.analyze_demographics_and_age()

        self.assertIn("demographic_summary", demo)
        self.assertIn("scientific_note", demo)


if __name__ == "__main__":
    unittest.main()
