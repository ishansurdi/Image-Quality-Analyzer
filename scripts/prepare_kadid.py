"""Prepare supported KADID-10k distortions for model training."""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2

from extract_features import FEATURE_FIELDS, extract_image_features


DISTORTION_CLASSES = {
    1: "blur",
    2: "blur",
    3: "blur",
    9: "compression",
    10: "compression",
    11: "noise",
    12: "noise",
    13: "noise",
    14: "noise",
    16: "overexposure",
    17: "underexposure",
}
OUTPUT_FIELDS = (
    "file_path",
    "source_image",
    "split",
    "issue",
    "severity",
    "level",
    "quality_score",
    *FEATURE_FIELDS,
)


def build_reference_splits(reference_names: set[str], seed: int) -> dict[str, str]:
    """Create a deterministic 80/10/10 split of reference images."""
    ordered = sorted(
        reference_names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest(),
    )
    train_end = round(len(ordered) * 0.8)
    validation_end = round(len(ordered) * 0.9)
    return {
        name: "train" if index < train_end else "val" if index < validation_end else "test"
        for index, name in enumerate(ordered)
    }


def severity_from_level(level: int) -> str:
    if level <= 2:
        return "low"
    if level == 3:
        return "medium"
    return "high"


def prepare_kadid(dataset_dir: Path, output_path: Path, seed: int) -> int:
    metadata_path = dataset_dir / "dmos.csv"
    image_dir = dataset_dir / "images"
    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        metadata = list(csv.DictReader(file))
    reference_splits = build_reference_splits(
        {row["ref_img"] for row in metadata},
        seed,
    )

    rows = []
    for row in metadata:
        parts = Path(row["dist_img"]).stem.split("_")
        distortion = int(parts[1])
        if distortion not in DISTORTION_CLASSES:
            continue

        image_path = image_dir / row["dist_img"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")

        level = int(parts[2])
        rows.append(
            {
                "file_path": str(image_path),
                "source_image": row["ref_img"],
                "split": reference_splits[row["ref_img"]],
                "issue": DISTORTION_CLASSES[distortion],
                "severity": severity_from_level(level),
                "level": level,
                "quality_score": round(float(row["dmos"]) * 20, 4),
                **extract_image_features(image),
            }
        )
        if len(rows) % 500 == 0:
            print(f"Processed {len(rows)} supported images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Data/kadid10k"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/kadid_features.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare_kadid(args.dataset, args.output, args.seed)
    print(f"Saved {count} KADID rows to {args.output}")


if __name__ == "__main__":
    main()
