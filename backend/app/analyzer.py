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
        self.compression_severity_classifier = self.bundle[
            "compression_severity_classifier"
        ]
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
        class_probabilities = {
            str(name): float(probability)
            for name, probability in zip(self.classifier.classes_, probabilities)
        }
        class_index = int(np.argmax(probabilities))
        predicted_issue = str(self.classifier.classes_[class_index])
        confidence = float(probabilities[class_index])
        quality_score = float(np.clip(self.regressor.predict(values)[0], 0, 100))

        issues = []
        bright_pixel_ratio = statistics["bright_pixel_ratio"]
        if bright_pixel_ratio >= 0.12:
            if bright_pixel_ratio >= 0.40:
                exposure_severity = "high"
                quality_score = min(quality_score, 35)
            elif bright_pixel_ratio >= 0.22:
                exposure_severity = "medium"
                quality_score = min(quality_score, 60)
            else:
                exposure_severity = "low"
                quality_score = min(quality_score, 80)
            exposure_confidence = max(
                class_probabilities.get("overexposure", 0),
                min(0.99, bright_pixel_ratio / 0.60),
            )
            issues.append(
                {
                    "type": "overexposure",
                    "severity": exposure_severity,
                    "confidence": round(exposure_confidence, 4),
                }
            )

        severity = None
        severity_confidence = None
        if predicted_issue == "blur":
            edge_density = statistics["edge_density"]
            gradient_strength = statistics["gradient_strength"]
            if edge_density <= 0.005 or gradient_strength <= 22:
                severity = "high"
                quality_score = min(quality_score, 35)
            elif edge_density <= 0.025 or gradient_strength <= 38:
                severity = "medium"
                quality_score = min(quality_score, 60)
            else:
                severity = "low"
                quality_score = min(quality_score, 80)
        elif predicted_issue == "compression":
            severity_probabilities = self.compression_severity_classifier.predict_proba(
                values
            )[0]
            severity_index = int(np.argmax(severity_probabilities))
            severity = str(self.compression_severity_classifier.classes_[severity_index])
            severity_confidence = float(severity_probabilities[severity_index])
            score_limits = {"high": 35, "medium": 60, "low": 80}
            quality_score = min(quality_score, score_limits[severity])

        if quality_score < 40:
            quality_label = "POTENTIALLY_DEFECTIVE"
        elif issues or predicted_issue != "acceptable" or quality_score < 75:
            quality_label = "DEGRADED"
        else:
            quality_label = "ACCEPTABLE"

        existing_issue_types = {issue["type"] for issue in issues}
        if predicted_issue != "acceptable" and predicted_issue not in existing_issue_types:
            if severity is None:
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
                    "severity_confidence": (
                        round(severity_confidence, 4)
                        if severity_confidence is not None
                        else None
                    ),
                }
            )

        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda issue: (severity_order[issue["severity"]], -issue["confidence"]))

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
