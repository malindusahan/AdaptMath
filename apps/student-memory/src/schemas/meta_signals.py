"""Pydantic schemas for Meta-Agent Evidence Signals emitted by the Memory subsystem."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class MetaSignalItem(BaseModel):
    """An individual auditable evidence signal emitted for the Meta-Agent."""

    model_config = ConfigDict(extra="ignore")

    signal_type: str = Field(
        ...,
        description="Signal type: 'correct_answer', 'incorrect_answer', 'confusion', 'clarification_request', 'repeated_misunderstanding'",
    )
    student_id: str
    session_id: str | None = None
    skill_id: str | None = None
    interaction_id: str | None = None
    timestamp: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class MetaSignalsResponse(BaseModel):
    """Collection of evidence signals generated for Meta-Agent analysis."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    session_id: str | None = None
    skill_id: str | None = None
    total_signals: int
    signals: list[MetaSignalItem] = Field(default_factory=list)
