import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temporary_directory.name)
        cls.settings = Settings(
            model_path=Path("artifacts/quality_model.joblib"),
            database_path=cls.temp_path / "analyses.db",
            upload_dir=cls.temp_path / "uploads",
            max_upload_bytes=10 * 1024 * 1024,
        )
        cls.client = TestClient(create_app(cls.settings))
        cls.client.__enter__()
        cls.sample_path = next(Path("Data/images/test").glob("*.jpg"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls.temporary_directory.cleanup()

    def upload_sample(self):
        return self.client.post(
            "/api/v1/analyses",
            files={
                "image": (
                    self.sample_path.name,
                    self.sample_path.read_bytes(),
                    "image/jpeg",
                )
            },
        )

    def test_health_returns_model_version(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "model_version": "1.0.0"})

    def test_upload_returns_and_persists_analysis(self) -> None:
        response = self.upload_sample()

        self.assertEqual(response.status_code, 201)
        result = response.json()
        self.assertIn(result["quality_label"], {"ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"})
        self.assertGreaterEqual(result["quality_score"], 0)
        self.assertLessEqual(result["quality_score"], 100)
        self.assertEqual(len(result["statistics"]), 12)

        saved = self.client.get(f"/api/v1/analyses/{result['id']}")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["id"], result["id"])

        uploaded_image = self.client.get(result["image_url"])
        self.assertEqual(uploaded_image.status_code, 200)

    def test_history_supports_pagination(self) -> None:
        self.upload_sample()

        response = self.client.get("/api/v1/analyses?limit=1&offset=0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["limit"], 1)
        self.assertEqual(len(response.json()["items"]), 1)
        self.assertEqual(self.client.get("/api/v1/analyses?limit=0").status_code, 422)

    def test_missing_analysis_returns_404(self) -> None:
        response = self.client.get("/api/v1/analyses/999999")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Analysis result not found.")

    def test_unsupported_file_returns_415(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            files={"image": ("notes.txt", b"text", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)

    def test_unreadable_image_returns_400(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            files={"image": ("broken.jpg", b"not an image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)

    def test_oversized_image_returns_413(self) -> None:
        small_settings = Settings(
            model_path=self.settings.model_path,
            database_path=self.temp_path / "small.db",
            upload_dir=self.temp_path / "small-uploads",
            max_upload_bytes=4,
        )
        with TestClient(create_app(small_settings)) as client:
            response = client.post(
                "/api/v1/analyses",
                files={"image": ("large.jpg", b"12345", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 413)

    def test_analysis_failure_returns_500(self) -> None:
        analyzer = self.client.app.state.analyzer
        uploads_before = set(self.settings.upload_dir.iterdir())

        class FailingAnalyzer:
            def analyze(self, image):
                raise RuntimeError("test failure")

        self.client.app.state.analyzer = FailingAnalyzer()
        try:
            response = self.upload_sample()
        finally:
            self.client.app.state.analyzer = analyzer

        self.assertEqual(response.status_code, 500)
        self.assertEqual(set(self.settings.upload_dir.iterdir()), uploads_before)


if __name__ == "__main__":
    unittest.main()
