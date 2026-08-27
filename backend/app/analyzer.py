"""Model-backed image quality analysis service."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.extract_features import extract_image_features


class ImageAnalyzer:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.bundle = joblib.load(model_path)
        self.classifier = self.bundle["classifier"]
        self.regressor = self.bundle["regressor"]
        self.feature_names = self.bundle["feature_names"]
        self.model_version = self.bundle["model_version"]

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        if image.size == 0:
            raise ValueError("Image is empty")

        statistics = extract_image_features(image)
        values = np.array(
            [[statistics[name] for name in self.feature_names]],
            dtype=np.float32,
        )
        probabilities = self.classifier.predict_proba(values)[0]
        class_index = int(np.argmax(probabilities))
        predicted_issue = str(self.classifier.classes_[class_index])
        confidence = float(probabilities[class_index])
        quality_score = float(np.clip(self.regressor.predict(values)[0], 0, 100))

        if quality_score < 40:
            quality_label = "POTENTIALLY_DEFECTIVE"
        elif predicted_issue != "acceptable" or quality_score < 75:
            quality_label = "DEGRADED"
        else:
            quality_label = "ACCEPTABLE"

        issues = []
        if predicted_issue != "acceptable":
            if quality_score >= 75:
                severity = "low"
            elif quality_score >= 50:
                severity = "medium"
            else:
                severity = "high"
            issues.append(
                {
                    "type": predicted_issue,
                    "severity": severity,
                    "confidence": round(confidence, 4),
                }
            )

        height, width = image.shape[:2]
        return {
            "width": width,
            "height": height,
            "quality_score": round(quality_score, 2),
            "quality_label": quality_label,
            "issues": issues,
            "statistics": {
                name: round(float(value), 4) for name, value in statistics.items()
            },
        }
