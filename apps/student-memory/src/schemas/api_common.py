"""Pydantic schemas for standardized API error responses and common envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid
from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Specific field-level or contextual error detail."""

    model_config = ConfigDict(extra="ignore")

    field: str | None = Field(default=None, description="Path to invalid field e.g. body.student_id.")
    message: str = Field(..., description="Human-readable error description.")
    error_type: str | None = Field(default=None, description="Error classification string.")


class ErrorResponse(BaseModel):
    """Standardized error envelope returned across all 4xx and 5xx API responses."""

    model_config = ConfigDict(extra="ignore")

    error_code: str = Field(
        ...,
        description="High-level error code e.g. VALIDATION_ERROR, NOT_FOUND, BAD_REQUEST, INTERNAL_ERROR.",
    )
    message: str = Field(..., description="Primary human-readable error summary.")
    detail: str | None = Field(
        default=None,
        description="Backward-compatible detail string mirroring primary error message.",
    )
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Optional list of granular field or validation error details.",
    )
    request_id: str = Field(..., description="Unique request tracing ID.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the error event.",
    )


class HealthResponse(BaseModel):
    """Lightweight liveness probe response."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ok", description="Service liveness state.")
    service: str = Field(
        default="student-personalization-memory",
        description="Canonical service identifier.",
    )


class ReadinessResponse(BaseModel):
    """Deep readiness probe response reporting core dependency statuses."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ready", description="Overall service readiness state.")
    service: str = Field(
        default="student-personalization-memory",
        description="Canonical service identifier.",
    )
    database: str = Field(default="ready", description="Database connection health.")
    migrations: str = Field(default="ready", description="Database schema migration status.")
    topic_extractor: str = Field(default="ready", description="Topic extraction model artifact status.")
    learning_state_model: str = Field(default="ready", description="Learning state classification model artifact status.")

