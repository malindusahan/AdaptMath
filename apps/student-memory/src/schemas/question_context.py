"""Pydantic schemas for Question Context API and live memory resolution."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionContextRequest(BaseModel):
    """Request payload for resolving free-text student question to canonical skill and memory context."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "student_id": "student_101",
                "session_id": "session_303",
                "question": "How do I solve 3x + 5 = 20?",
            }
        },
    )

    student_id: str = Field(..., description="Unique student identifier (external or UUID).")
    session_id: str = Field(..., description="Learning session identifier.")
    question: str = Field(..., description="Raw natural language question text from the student.")

    @field_validator("student_id", "session_id", "question")
    @classmethod
    def require_non_blank(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be blank.")
        return value.strip()


class UIQuestionContextRequest(BaseModel):
    """Frontend request payload where student_id is optional and auto-populated from authentication."""

    model_config = ConfigDict(extra="ignore")

    student_id: str | None = Field(default=None, description="Optional student ID; auto-resolved from auth.")
    session_id: str = Field(..., description="Learning session identifier.")
    question: str = Field(..., description="Raw natural language question text.")

    @field_validator("session_id", "question")
    @classmethod
    def require_non_blank_ui(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be blank.")
        return value.strip()


class TopicContextItem(BaseModel):
    """Extracted canonical topic details and confidence metadata."""

    skill_id: str | None = None
    skill_code: str | None = None
    canonical_skill_name: str | None = None
    display_name: str | None = None
    confidence: float = 0.0
    method: str
    needs_review: bool
    top_candidates: list[dict[str, Any]] = Field(default_factory=list)
    model_version: str = "phase14-minilm-ft-epoch4"


class MisconceptionItem(BaseModel):
    """Recorded student misconception on the resolved canonical skill."""

    misconception_id: str
    normalized_error: str
    display_error: str
    occurrence_count: int
    last_seen_at: str


class LearningStateItem(BaseModel):
    """Current dynamic learning state for the student on the resolved canonical skill."""

    learning_state: str
    evidence_level: str
    evidence_strength: str | float
    behavioural_coverage: str | float
    model_used: bool | str | None = None
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    updated_at: str


class QuestionContextResponse(BaseModel):
    """Full question context response combining topic extraction and personalized memory."""

    student_id: str
    session_id: str
    question: str
    topic: TopicContextItem
    short_term_memory: dict[str, Any] | None = None
    long_term_memory: dict[str, Any] | None = None
    concept_memory: dict[str, Any] | None = None
    learning_state: LearningStateItem | None = None
    misconceptions: list[MisconceptionItem] = Field(default_factory=list)
