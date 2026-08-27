"""Validate generated images, metadata, class balance, and split isolation."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_dataset(dataset_dir: Path) -> dict:
    rows = load_metadata(dataset_dir / "metadata.csv")
    class_counts = Counter(row["issue"] for row in rows)
    severity_counts = Counter(
        f'{row["issue"]}:{row["severity"]}' for row in rows if row["severity"] != "none"
    )
    split_counts = Counter(row["split"] for row in rows)
    split_sources: dict[str, set[str]] = defaultdict(set)
    missing_files = []
    unreadable_files = []
    dimensions = Counter()

    for row in rows:
        split_sources[row["split"]].add(row["source_image"])
        image_path = dataset_dir / row["file_path"]
        if not image_path.is_file():
            missing_files.append(row["file_path"])
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_files.append(row["file_path"])
            continue
        dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1

    split_names = sorted(split_sources)
    overlap = {}
    for index, first_split in enumerate(split_names):
        for second_split in split_names[index + 1 :]:
            key = f"{first_split}:{second_split}"
            overlap[key] = len(split_sources[first_split] & split_sources[second_split])

    valid = not missing_files and not unreadable_files and not any(overlap.values())
    return {
        "valid": valid,
        "total_images": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "split_overlap": overlap,
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Data/generated"))
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset)
    output = json.dumps(report, indent=2)
    print(output)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
