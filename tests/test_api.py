"""Unit & Integration Tests for NeuroVision FastAPI Backend

Verifies health probes, dataset metadata endpoints, checkpoint queries,
image inference, and visual explainability outputs using FastAPI TestClient.
"""

import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import FALAH_CLASSES


class TestNeuroVisionAPI(unittest.TestCase):
    """Test suite for the REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Generate synthetic in-memory test image
        img = Image.new("RGB", (128, 128), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.synthetic_png_bytes = buf.getvalue()

    def test_root_redirect(self):
        """Verifies root route redirects to Swagger /docs."""
        response = self.client.get("/", follow_redirects=False)
        self.assertIn(response.status_code, [302, 307])
        self.assertEqual(response.headers["location"], "/docs")

    def test_health_endpoint(self):
        """Verifies health probe returns healthy status and runtime report."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "2.0.0")
        self.assertIn("active_device", data)
        self.assertIn("pytorch_available", data)

    def test_models_endpoint(self):
        """Verifies model listing returns supported architectures list."""
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_checkpoints", data)
        self.assertIn("supported_architectures", data)
        self.assertTrue(len(data["supported_architectures"]) >= 4)

    def test_datasets_list(self):
        """Verifies supported datasets list returns Falah and OASIS-1."""
        response = self.client.get("/api/v1/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        ids = [d["id"] for d in data["supported_datasets"]]
        self.assertIn("Falah_Alzheimer_MRI", ids)
        self.assertIn("OASIS-1", ids)

    def test_dataset_summary_falah(self):
        """Verifies Falah dataset summary returns exact 6,400 sample count."""
        response = self.client.get("/api/v1/datasets/Falah_Alzheimer_MRI/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_samples"], 6400)
        self.assertEqual(data["dataset_name"], "Falah_Alzheimer_MRI")
        self.assertIn("Mild_Demented", data["class_names"])

    def test_predict_endpoint(self):
        """Verifies single image inference returns valid Alzheimer stage classification."""
        files = {
            "image": ("test_slice.png", self.synthetic_png_bytes, "image/png")
        }
        response = self.client.post("/api/v1/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIn("inference_time_ms", data)
        self.assertIn(data["predicted_class"], FALAH_CLASSES)
        self.assertTrue(0.0 <= data["confidence"] <= 100.0)

    def test_explain_endpoint(self):
        """Verifies explainability endpoint returns Base64 encoded attention heatmaps."""
        files = {
            "image": ("test_slice.png", self.synthetic_png_bytes, "image/png")
        }
        response = self.client.post("/api/v1/explain", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("heatmap_base64", data)
        self.assertIn("overlay_base64", data)
        self.assertIn("method", data)
        self.assertTrue(len(data["heatmap_base64"]) > 100)
        self.assertTrue(len(data["overlay_base64"]) > 100)


if __name__ == "__main__":
    unittest.main()

