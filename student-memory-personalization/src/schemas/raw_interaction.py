"""Validated write and read contracts for raw interaction evidence."""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RawInteractionCreate(BaseModel):
    """Input required to append one source-of-truth interaction."""

    model_config = ConfigDict(str_strip_whitespace=True)

    student_id: uuid.UUID
    session_id: uuid.UUID
    canonical_skill_id: uuid.UUID | None = None

    source: str = Field(min_length=1, max_length=100)
    external_interaction_id: str = Field(min_length=1, max_length=255)
    problem_id: str | None = Field(default=None, max_length=255)
    question_id: str | None = Field(default=None, max_length=255)

    student_utterance: str | None = None
    tutor_response: str | None = None
    student_answer: str | None = None
    expected_answer: str | None = None

    is_correct: bool | None = None
    identified_error: str | None = None

    attempt_count: int | None = Field(default=None, ge=0)
    hint_count: int | None = Field(default=None, ge=0)
    hint_total: int | None = Field(default=None, ge=0)
    response_time_ms: float | None = Field(default=None, ge=0)
    created_at: datetime | None = None

    @field_validator(
        "problem_id",
        "question_id",
        "student_utterance",
        "tutor_response",
        "student_answer",
        "expected_answer",
        "identified_error",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def validate_hint_usage(self):
        if (
            self.hint_count is not None
            and self.hint_total is not None
            and self.hint_count > self.hint_total
        ):
            raise ValueError("hint_count must not exceed hint_total.")
        return self


class RawInteractionRecord(BaseModel):
    """Storage-neutral raw interaction returned by repository queries."""

    model_config = ConfigDict(from_attributes=True)

    interaction_id: uuid.UUID
    student_id: uuid.UUID
    session_id: uuid.UUID
    canonical_skill_id: uuid.UUID | None

    source: str
    external_interaction_id: str
    problem_id: str | None
    question_id: str | None

    student_utterance: str | None
    tutor_response: str | None
    student_answer: str | None
    expected_answer: str | None

    is_correct: bool | None
    identified_error: str | None

    attempt_count: int | None
    hint_count: int | None
    hint_total: int | None
    response_time_ms: float | None
    created_at: datetime


