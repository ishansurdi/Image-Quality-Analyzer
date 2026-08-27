"""FastAPI application for image-quality analysis."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.analyzer import ImageAnalyzer
from backend.app.config import Settings, settings
from backend.app.database import AnalysisRepository
from backend.app.schemas import AnalysisHistoryResponse, AnalysisResponse, HealthResponse


ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def format_response(record: dict) -> dict:
    result = dict(record)
    result["image_url"] = f'/uploads/{result.pop("stored_filename")}'
    return result


def create_app(app_settings: Settings = settings) -> FastAPI:
    repository = AnalysisRepository(app_settings.database_path)
    app_settings.upload_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.initialize()
        app.state.repository = repository
        app.state.analyzer = ImageAnalyzer(app_settings.model_path)
        yield

    app = FastAPI(
        title="SmartCity Image Quality API",
        version="1.0.0",
        description="Analyze image quality and retrieve previous results.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/uploads", StaticFiles(directory=app_settings.upload_dir), name="uploads")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> dict:
        return {
            "status": "healthy",
            "model_version": request.app.state.analyzer.model_version,
        }

    @app.post(
        "/api/v1/analyses",
        response_model=AnalysisResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["analyses"],
    )
    async def create_analysis(request: Request, image: UploadFile = File(...)) -> dict:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Supported image types are JPEG, PNG, and WebP.",
            )

        content = await image.read(app_settings.max_upload_bytes + 1)
        if len(content) > app_settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image must be at most {app_settings.max_upload_bytes // 1048576} MB.",
            )

        decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is not a readable image.",
            )

        extension = ALLOWED_IMAGE_TYPES[image.content_type]
        stored_filename = f"{uuid4().hex}{extension}"
        stored_path = app_settings.upload_dir / stored_filename
        stored_path.write_bytes(content)

        try:
            result = request.app.state.analyzer.analyze(decoded)
            record = request.app.state.repository.create(
                {
                    **result,
                    "original_filename": Path(image.filename or "image").name,
                    "stored_filename": stored_filename,
                    "content_type": image.content_type,
                }
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image analysis failed.",
            )
        return format_response(record)

    @app.get(
        "/api/v1/analyses",
        response_model=AnalysisHistoryResponse,
        tags=["analyses"],
    )
    def list_analyses(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        records = request.app.state.repository.list(limit, offset)
        return {
            "items": [format_response(record) for record in records],
            "limit": limit,
            "offset": offset,
        }

    @app.get(
        "/api/v1/analyses/{analysis_id}",
        response_model=AnalysisResponse,
        tags=["analyses"],
    )
    def get_analysis(request: Request, analysis_id: int) -> dict:
        record = request.app.state.repository.get(analysis_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis result not found.",
            )
        return format_response(record)

    return app


app = create_app()
