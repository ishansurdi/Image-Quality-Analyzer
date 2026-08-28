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
SYNTHETIC_CLASSIFICATION_WEIGHT = 6.0
SYNTHETIC_REGRESSION_WEIGHT = 2.0
MODEL_TREES = 150


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
    scores = np.array(
        [float(row.get("quality_score") or QUALITY_SCORES[row["severity"]]) for row in selected],
        dtype=np.float32,
    )
    return features, issues, scores


def combine_splits(*splits: tuple[np.ndarray, np.ndarray, np.ndarray]):
    return tuple(np.concatenate(values) for values in zip(*splits))


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


def train_models(
    rows: list[dict[str, str]],
    kadid_rows: list[dict[str, str]],
    seed: int,
) -> tuple[dict, dict]:
    synthetic_train = prepare_split(rows, "train")
    kadid_train = prepare_split(kadid_rows, "train")
    train_x, train_issues, train_scores = combine_splits(synthetic_train, kadid_train)

    synthetic_validation = prepare_split(rows, "val")
    synthetic_test = prepare_split(rows, "test")
    kadid_validation = prepare_split(kadid_rows, "val")
    kadid_test = prepare_split(kadid_rows, "test")

    classifier = RandomForestClassifier(
        n_estimators=MODEL_TREES,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    regressor = RandomForestRegressor(
        n_estimators=MODEL_TREES,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    synthetic_count = len(synthetic_train[0])
    kadid_count = len(kadid_train[0])
    classifier_weights = np.concatenate(
        (
            np.full(synthetic_count, SYNTHETIC_CLASSIFICATION_WEIGHT),
            np.ones(kadid_count),
        )
    )
    regression_weights = np.concatenate(
        (
            np.full(synthetic_count, SYNTHETIC_REGRESSION_WEIGHT),
            np.ones(kadid_count),
        )
    )
    classifier.fit(train_x, train_issues, sample_weight=classifier_weights)
    regressor.fit(train_x, train_scores, sample_weight=regression_weights)

    labels = sorted(classifier.classes_.tolist())
    importance = sorted(
        zip(FEATURE_FIELDS, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "dataset": {
            "train_rows": len(train_x),
            "synthetic_train_rows": len(synthetic_train[0]),
            "kadid_train_rows": len(kadid_train[0]),
            "synthetic_validation_rows": len(synthetic_validation[0]),
            "synthetic_test_rows": len(synthetic_test[0]),
            "kadid_validation_rows": len(kadid_validation[0]),
            "kadid_test_rows": len(kadid_test[0]),
            "synthetic_classification_weight": SYNTHETIC_CLASSIFICATION_WEIGHT,
            "synthetic_regression_weight": SYNTHETIC_REGRESSION_WEIGHT,
            "model_trees": MODEL_TREES,
        },
        "labels": labels,
        "classification": {
            "synthetic_validation": classification_metrics(
                synthetic_validation[1], classifier.predict(synthetic_validation[0]), labels
            ),
            "synthetic_test": classification_metrics(
                synthetic_test[1], classifier.predict(synthetic_test[0]), labels
            ),
            "kadid_validation": classification_metrics(
                kadid_validation[1], classifier.predict(kadid_validation[0]), labels
            ),
            "kadid_test": classification_metrics(
                kadid_test[1], classifier.predict(kadid_test[0]), labels
            ),
        },
        "quality_regression": {
            "synthetic_validation": regression_metrics(
                synthetic_validation[2], regressor.predict(synthetic_validation[0])
            ),
            "synthetic_test": regression_metrics(
                synthetic_test[2], regressor.predict(synthetic_test[0])
            ),
            "kadid_validation": regression_metrics(
                kadid_validation[2], regressor.predict(kadid_validation[0])
            ),
            "kadid_test": regression_metrics(
                kadid_test[2], regressor.predict(kadid_test[0])
            ),
        },
        "feature_importance": {
            name: round(float(value), 6) for name, value in importance
        },
    }
    bundle = {
        "model_version": "2.0.0",
        "feature_names": list(FEATURE_FIELDS),
        "quality_scores": QUALITY_SCORES,
        "classifier": classifier,
        "regressor": regressor,
    }
    return bundle, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument(
        "--kadid-features",
        type=Path,
        default=Path("artifacts/kadid_features.csv"),
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/quality_model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/model_evaluation.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_features(args.features)
    kadid_rows = load_features(args.kadid_features)
    bundle, report = train_models(rows, kadid_rows, args.seed)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    synthetic_metrics = report["classification"]["synthetic_test"]
    kadid_metrics = report["classification"]["kadid_test"]
    print(f"Synthetic test macro F1: {synthetic_metrics['macro_f1']:.4f}")
    print(f"KADID test macro F1: {kadid_metrics['macro_f1']:.4f}")
    print(f"Saved model to {args.model}")
    print(f"Saved evaluation to {args.report}")


if __name__ == "__main__":
    main()
