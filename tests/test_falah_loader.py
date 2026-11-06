"""Unit and Integration Tests for Falah Alzheimer MRI Loader

Tests dataset discovery, Arrow loading, train/val/test isolation,
preprocessing consistency, duplicate analysis, and summary reporting.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.falah_loader import FalahDatasetLoader, FALAH_CLASSES, resolve_falah_path


class TestFalahLoader(unittest.TestCase):
    """Test suite for Falah dataset loading, splitting, and verification."""

    @classmethod
    def setUpClass(cls):
        # Initialize loader with actual local Arrow dataset
        cls.loader = FalahDatasetLoader(val_split=0.15, seed=42)

    def test_path_resolution(self):
        """Verifies that the dataset path resolves to an existing dataset_dict.json."""
        resolved = resolve_falah_path()
        self.assertTrue((resolved / "dataset_dict.json").exists())

    def test_dataset_counts(self):
        """Verifies exact split counts from the Arrow dataset."""
        info = self.loader.get_dataset_info()
        self.assertEqual(info["train_samples"], 5120)
        self.assertEqual(info["test_samples"], 1280)
        self.assertEqual(info["total_samples"], 6400)

    def test_class_names(self):
        """Verifies the four canonical Alzheimer's disease stages."""
        info = self.loader.get_dataset_info()
        self.assertEqual(len(info["class_names"]), 4)
        for expected_class in FALAH_CLASSES:
            self.assertIn(expected_class, info["class_names"])

    def test_class_imbalance_weights(self):
        """Verifies that class weights are strictly positive and Moderate_Demented has max weight."""
        weights = self.loader.compute_class_weights()
        self.assertEqual(len(weights), 4)

        # Moderate_Demented is ~1% of dataset, so its inverse weight must be the largest
        max_class = max(weights, key=weights.get)
        self.assertEqual(max_class, "Moderate_Demented")
        self.assertGreater(weights["Moderate_Demented"], weights["Non_Demented"])

    def test_train_val_test_isolation(self):
        """Verifies that validation data is carved strictly from train, leaving test untouched."""
        val_ratio = 0.20
        train_sub, val_sub, test_set = self.loader.get_splits(val_split=val_ratio, seed=123)

        # Test set must remain exactly 1,280 samples
        self.assertEqual(len(test_set), 1280)

        # Train + Val must equal original 5,120 train samples
        self.assertEqual(len(train_sub) + len(val_sub), 5120)
        self.assertEqual(len(val_sub), int(5120 * val_ratio))

    def test_image_preprocessing(self):
        """Verifies that preprocessing converts PIL images to normalized float32 arrays."""
        sample_img = Image.new("L", (128, 128), color=128)
        processed = self.loader.preprocess_image(sample_img)

        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.all(processed >= 0.0) and np.all(processed <= 1.0))

    def test_image_integrity_check(self):
        """Verifies image integrity verification on a sample batch."""
        res = self.loader.check_image_integrity(max_check=50)
        self.assertEqual(res["total_checked"], 100)  # 50 train + 50 test
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["corrupted_count"], 0)

    def test_duplicate_detection_structure(self):
        """Verifies that cross-split duplicate detection returns valid metrics."""
        dup_res = self.loader.detect_duplicates(max_samples=20)
        self.assertIn("exact_duplicate_count", dup_res)
        self.assertIn("near_duplicate_count", dup_res)
        self.assertIn("methodological_note", dup_res)

    def test_summary_report(self):
        """Verifies that the formatted summary report generates with key sections."""
        report = self.loader.generate_summary_report()
        self.assertIn("NEUROVISION: FALAH ALZHEIMER MRI AUDIT REPORT", report)
        self.assertIn("Moderate_Demented", report)
        self.assertIn("Participant Independence: CANNOT BE CLAIMED", report)

    def test_get_summary_dict(self):
        """Verifies structured summary dict for Streamlit Dataset Management."""
        summary = self.loader.get_summary()
        self.assertIn("total_samples", summary)
        self.assertIn("split_counts", summary)
        self.assertIn("class_distributions", summary)
        self.assertEqual(summary["total_samples"], 6400)

    def test_get_sample_images(self):
        """Verifies retrieving sample image batches with and without class filter."""
        samples = self.loader.get_sample_images(split="train", count=4)
        self.assertEqual(len(samples), 4)
        self.assertIn("image", samples[0])
        self.assertIn("class_name", samples[0])

        # Test filtering by a specific class
        filtered = self.loader.get_sample_images(split="train", count=2, class_filter="Mild_Demented")
        self.assertTrue(len(filtered) > 0)
        self.assertEqual(filtered[0]["class_name"], "Mild_Demented")

    def test_raw_dataset_property(self):
        """Verifies raw_dataset property returns the underlying DatasetDict without setter errors."""
        self.assertIsNotNone(self.loader.raw_dataset)
        self.assertIn("train", self.loader.raw_dataset)
        self.assertIn("test", self.loader.raw_dataset)


if __name__ == "__main__":
    unittest.main()
