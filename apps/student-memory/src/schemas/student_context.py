"""Pydantic schemas for the unified Student Memory Context service."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class RecentInteractionItem(BaseModel):
    """Summarized raw interaction log record."""

    interaction_id: str
    session_id: str
    canonical_skill_id: str | None = None
    student_utterance: str | None = None
    identified_error: str | None = None
    is_correct: bool | None = None
    attempt_count: int | None = None
    hint_count: int | None = None
    response_time_ms: float | None = None
    created_at: str


class RepairOutcomeItem(BaseModel):
    """Recorded pedagogical repair outcome."""

    repair_outcome_id: str
    session_id: str
    canonical_skill_id: str
    interaction_id: str | None = None
    repair_action: str
    outcome: str
    score: float | None = None
    notes: str | None = None
    created_at: str


class StudentContextResponse(BaseModel):
    """Unified full student memory context response."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    session_id: str | None = None
    skill_id: str | None = None

    short_term_memory: dict[str, Any] | None = None
    long_term_memory: dict[str, Any] | None = None
    concept_memory: dict[str, Any] | None = None

    learning_state: dict[str, Any] | None = None
    misconceptions: list[dict[str, Any]] = Field(default_factory=list)
    recent_interactions: list[RecentInteractionItem] = Field(default_factory=list)
    recent_repairs: list[RepairOutcomeItem] = Field(default_factory=list)
