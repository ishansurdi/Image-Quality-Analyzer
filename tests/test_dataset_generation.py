import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.generate_dataset import CLASSES, generate_dataset, write_metadata


class DatasetGenerationTests(unittest.TestCase):
    def test_generates_every_class_for_each_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            output_dir = root / "output"
            image = np.full((32, 32, 3), 120, dtype=np.uint8)

            for split in ("train", "val", "test"):
                split_dir = source_dir / split
                split_dir.mkdir(parents=True)
                cv2.imwrite(str(split_dir / "sample.jpg"), image)

            rows = generate_dataset(source_dir, output_dir, seed=42)
            write_metadata(rows, output_dir / "metadata.csv")

            self.assertEqual(len(rows), 3 * len(CLASSES))
            self.assertTrue((output_dir / "metadata.csv").is_file())
            for row in rows:
                self.assertTrue((output_dir / str(row["file_path"])).is_file())


if __name__ == "__main__":
    unittest.main()
