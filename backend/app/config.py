"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


@dataclass(frozen=True)
class Settings:
    model_path: Path
    database_path: Path
    upload_dir: Path
    max_upload_bytes: int
    frontend_origins: tuple[str, ...] = LOCAL_FRONTEND_ORIGINS

    @classmethod
    def from_environment(cls) -> "Settings":
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
        configured_origins = os.getenv("FRONTEND_ORIGINS", "")
        frontend_origins = tuple(
            origin.strip() for origin in configured_origins.split(",") if origin.strip()
        )
        return cls(
            model_path=Path(
                os.getenv("MODEL_PATH", PROJECT_DIR / "artifacts/quality_model.joblib")
            ),
            database_path=Path(
                os.getenv("DATABASE_PATH", PROJECT_DIR / "backend/data/analyses.db")
            ),
            upload_dir=Path(os.getenv("UPLOAD_DIR", PROJECT_DIR / "backend/uploads")),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            frontend_origins=frontend_origins or LOCAL_FRONTEND_ORIGINS,
        )


settings = Settings.from_environment()
