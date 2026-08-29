"""Pydantic schemas used in API responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal["low", "medium", "high"]
QualityLabel = Literal["ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"]


class DetectedIssue(BaseModel):
    type: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    severity_confidence: float | None = Field(default=None, ge=0, le=1)


class ImageStatistics(BaseModel):
    brightness_mean: float
    brightness_std: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    laplacian_variance: float
    gradient_strength: float
    edge_density: float
    noise_estimate: float
    entropy: float
    saturation_mean: float
    saturation_std: float
    blockiness: float


class AnalysisResponse(BaseModel):
    id: int
    original_filename: str
    image_url: str
    content_type: str
    width: int
    height: int
    quality_score: float = Field(ge=0, le=100)
    quality_label: QualityLabel
    issues: list[DetectedIssue]
    statistics: ImageStatistics
    created_at: datetime


class AnalysisHistoryResponse(BaseModel):
    items: list[AnalysisResponse]
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_version: str
