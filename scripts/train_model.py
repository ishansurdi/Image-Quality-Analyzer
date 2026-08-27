"""Train and evaluate image-quality classification and regression models."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from extract_features import FEATURE_FIELDS


QUALITY_SCORES = {"none": 100, "low": 80, "medium": 55, "high": 30}


def load_features(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def prepare_split(rows: list[dict[str, str]], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["split"] == split]
    features = np.array(
        [[float(row[field]) for field in FEATURE_FIELDS] for row in selected],
        dtype=np.float32,
    )
    issues = np.array([row["issue"] for row in selected])
    scores = np.array([QUALITY_SCORES[row["severity"]] for row in selected], dtype=np.float32)
    return features, issues, scores


def classification_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(expected, predicted)), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "per_class": classification_report(
            expected,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def regression_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(expected, predicted)), 4),
        "rmse": round(float(mean_squared_error(expected, predicted) ** 0.5), 4),
        "r2": round(float(r2_score(expected, predicted)), 4),
    }


def train_models(rows: list[dict[str, str]], seed: int) -> tuple[dict, dict]:
    train_x, train_issues, train_scores = prepare_split(rows, "train")
    validation_x, validation_issues, validation_scores = prepare_split(rows, "val")
    test_x, test_issues, test_scores = prepare_split(rows, "test")

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    regressor = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    classifier.fit(train_x, train_issues)
    regressor.fit(train_x, train_scores)

    labels = sorted(classifier.classes_.tolist())
    importance = sorted(
        zip(FEATURE_FIELDS, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "dataset": {
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
            "test_rows": len(test_x),
        },
        "labels": labels,
        "classification": {
            "validation": classification_metrics(
                validation_issues, classifier.predict(validation_x), labels
            ),
            "test": classification_metrics(test_issues, classifier.predict(test_x), labels),
        },
        "quality_regression": {
            "validation": regression_metrics(validation_scores, regressor.predict(validation_x)),
            "test": regression_metrics(test_scores, regressor.predict(test_x)),
        },
        "feature_importance": {
            name: round(float(value), 6) for name, value in importance
        },
    }
    bundle = {
        "model_version": "1.0.0",
        "feature_names": list(FEATURE_FIELDS),
        "quality_scores": QUALITY_SCORES,
        "classifier": classifier,
        "regressor": regressor,
    }
    return bundle, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/quality_model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/model_evaluation.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_features(args.features)
    bundle, report = train_models(rows, args.seed)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    test_metrics = report["classification"]["test"]
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Saved model to {args.model}")
    print(f"Saved evaluation to {args.report}")


if __name__ == "__main__":
    main()
