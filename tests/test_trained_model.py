import unittest
from pathlib import Path

import cv2
import joblib
import numpy as np

from scripts.extract_features import FEATURE_FIELDS, extract_image_features


class TrainedModelTests(unittest.TestCase):
    def test_model_predicts_valid_result(self) -> None:
        model_path = Path("artifacts/quality_model.joblib")
        image_path = next(Path("Data/images/test").glob("*.jpg"))
        bundle = joblib.load(model_path)
        image = cv2.imread(str(image_path))
        features = extract_image_features(image)
        values = np.array([[features[name] for name in FEATURE_FIELDS]])

        probabilities = bundle["classifier"].predict_proba(values)[0]
        quality_score = float(bundle["regressor"].predict(values)[0])

        self.assertEqual(bundle["feature_names"], list(FEATURE_FIELDS))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 100)


if __name__ == "__main__":
    unittest.main()
