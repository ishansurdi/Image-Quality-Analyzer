"""Extract explainable image-quality features into a compact CSV file."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


METADATA_FIELDS = ("file_path", "source_image", "split", "issue", "severity", "level")
FEATURE_FIELDS = (
    "brightness_mean",
    "brightness_std",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "laplacian_variance",
    "gradient_strength",
    "edge_density",
    "noise_estimate",
    "entropy",
    "saturation_mean",
    "saturation_std",
    "blockiness",
)


def calculate_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def calculate_blockiness(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    vertical_boundaries = np.arange(8, image.shape[1], 8)
    horizontal_boundaries = np.arange(8, image.shape[0], 8)
    vertical = 0.0
    if vertical_boundaries.size:
        vertical = np.abs(
            image[:, vertical_boundaries] - image[:, vertical_boundaries - 1]
        ).mean()

    horizontal = 0.0
    if horizontal_boundaries.size:
        horizontal = np.abs(
            image[horizontal_boundaries, :] - image[horizontal_boundaries - 1, :]
        ).mean()
    return float((vertical + horizontal) / 2)


def extract_image_features(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape
    if height >= 20 and width >= 20:
        margin_y = max(1, int(height * 0.05))
        margin_x = max(1, int(width * 0.05))
        detail_region = gray[margin_y:-margin_y, margin_x:-margin_x]
    else:
        detail_region = gray

    laplacian = cv2.Laplacian(detail_region, cv2.CV_64F)
    gradient_x = cv2.Sobel(detail_region, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(detail_region, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    edges = cv2.Canny(detail_region, 100, 200)
    residual = detail_region.astype(np.float32) - cv2.GaussianBlur(
        detail_region, (3, 3), 0
    )
    saturation = hsv[:, :, 1]

    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "dark_pixel_ratio": float(np.mean(gray < 40)),
        "bright_pixel_ratio": float(np.mean(gray > 215)),
        "laplacian_variance": float(laplacian.var()),
        "gradient_strength": float(gradient.mean()),
        "edge_density": float(np.mean(edges > 0)),
        "noise_estimate": float(residual.std()),
        "entropy": calculate_entropy(gray),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "blockiness": calculate_blockiness(detail_region),
    }


def extract_dataset_features(dataset_dir: Path, output_path: Path) -> int:
    metadata_path = dataset_dir / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(*METADATA_FIELDS, *FEATURE_FIELDS))
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            image_path = dataset_dir / row["file_path"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")

            writer.writerow({**row, **extract_image_features(image)})
            if index % 500 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} images")

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Data/quality_dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_dataset_features(args.dataset, args.output)
    print(f"Saved {count} feature rows to {args.output}")


if __name__ == "__main__":
    main()
