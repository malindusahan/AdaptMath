"""Pydantic schemas for the Support Preference Evidence service."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class SupportStrategySummary(BaseModel):
    """Aggregated empirical evidence for a single repair action strategy."""

    repair_action: str
    observation_count: int
    successful_count: int
    partial_count: int = 0
    failed_count: int = 0
    success_rate: float
    average_score: float | None = None


class SupportPreferenceResponse(BaseModel):
    """Estimated support style preference based purely on historical repair efficacy."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    skill_id: str | None = None
    preferred_support_style: str | None = None
    status: str = Field(
        ...,
        description="Evidence status: 'SUPPORTED_BY_HISTORY' or 'INSUFFICIENT_EVIDENCE'",
    )
    evidence_count: int | None = None
    success_rate: float | None = None
    strategies: list[SupportStrategySummary] = Field(default_factory=list)
