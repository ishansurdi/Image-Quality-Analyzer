"""Generate labelled image-quality degradations from clean images."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("acceptable", "blur", "noise", "underexposure", "overexposure", "compression")
SEVERITIES = ("low", "medium", "high")
CSV_FIELDS = ("file_path", "source_image", "split", "issue", "severity", "level")


def stable_seed(seed: int, *values: str) -> int:
    """Return the same random seed for the same input values."""
    text = ":".join((str(seed), *values))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def apply_degradation(
    image: np.ndarray,
    issue: str,
    level: int,
    random_seed: int,
) -> np.ndarray:
    """Apply one degradation at a severity level from 1 to 3."""
    if issue == "blur":
        kernel_sizes = (5, 11, 21)
        kernel_size = kernel_sizes[level - 1]
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    if issue == "noise":
        noise_levels = (8, 18, 32)
        generator = np.random.default_rng(random_seed)
        noise = generator.normal(0, noise_levels[level - 1], image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if issue == "underexposure":
        factors = (0.75, 0.50, 0.30)
        return cv2.convertScaleAbs(image, alpha=factors[level - 1], beta=0)

    if issue == "overexposure":
        factors = (1.20, 1.50, 2.00)
        return cv2.convertScaleAbs(image, alpha=factors[level - 1], beta=0)

    raise ValueError(f"Unsupported degradation: {issue}")


def write_image(image: np.ndarray, output_path: Path, issue: str, level: int) -> None:
    """Write an image once, using JPEG quality to create compression artifacts."""
    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = []
    if issue == "compression":
        jpeg_quality = (70, 40, 15)[level - 1]
        parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    if not cv2.imwrite(str(output_path), image, parameters):
        raise OSError(f"Could not write image: {output_path}")


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    max_images_per_split: int | None = None,
    seed: int = 42,
) -> list[dict[str, str | int]]:
    """Generate one deterministic class example per source image."""
    rows: list[dict[str, str | int]] = []

    for split_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        images = sorted(split_dir.glob("*.jpg"))
        if max_images_per_split is not None:
            images = images[:max_images_per_split]

        for source_path in images:
            image = cv2.imread(str(source_path))
            if image is None:
                raise ValueError(f"Unreadable source image: {source_path}")

            for issue in CLASSES:
                if issue == "acceptable":
                    severity = "none"
                    level = 0
                else:
                    issue_seed = stable_seed(seed, split_dir.name, source_path.stem, issue)
                    level = issue_seed % 3 + 1
                    severity = SEVERITIES[level - 1]

                if issue == "acceptable":
                    output_name = source_path.name
                else:
                    output_name = f"{source_path.stem}_s{seed}_l{level}.jpg"

                relative_path = Path(split_dir.name) / issue / output_name
                output_path = output_dir / relative_path

                if issue == "acceptable":
                    if not output_path.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, output_path)
                elif issue == "compression":
                    write_image(image, output_path, issue, level)
                else:
                    degraded = apply_degradation(image, issue, level, issue_seed)
                    write_image(degraded, output_path, issue, level)

                rows.append(
                    {
                        "file_path": relative_path.as_posix(),
                        "source_image": source_path.name,
                        "split": split_dir.name,
                        "issue": issue,
                        "severity": severity,
                        "level": level,
                    }
                )

    return rows


def write_metadata(rows: list[dict[str, str | int]], output_path: Path) -> None:
    """Write metadata for every generated image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Data/images"))
    parser.add_argument("--output", type=Path, default=Path("Data/generated"))
    parser.add_argument("--max-images-per-split", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")
    if args.max_images_per_split is not None and args.max_images_per_split < 1:
        raise ValueError("--max-images-per-split must be at least 1")

    rows = generate_dataset(
        source_dir=args.source,
        output_dir=args.output,
        max_images_per_split=args.max_images_per_split,
        seed=args.seed,
    )
    write_metadata(rows, args.output / "metadata.csv")
    print(f"Generated {len(rows)} labelled images in {args.output}")


if __name__ == "__main__":
    main()
