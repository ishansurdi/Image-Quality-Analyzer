"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    database_path: Path
    upload_dir: Path
    max_upload_bytes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
        return cls(
            model_path=Path(
                os.getenv("MODEL_PATH", PROJECT_DIR / "artifacts/quality_model.joblib")
            ),
            database_path=Path(
                os.getenv("DATABASE_PATH", PROJECT_DIR / "backend/data/analyses.db")
            ),
            upload_dir=Path(os.getenv("UPLOAD_DIR", PROJECT_DIR / "backend/uploads")),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
        )


settings = Settings.from_environment()
