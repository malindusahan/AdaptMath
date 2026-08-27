"""Validated persistence contracts for derived memory projections."""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceProjection(BaseModel):
    """Shared evidence counts and sums retained by every projection."""

    interaction_count: int = Field(default=0, ge=0)
    correct_count: int = Field(default=0, ge=0)
    incorrect_count: int = Field(default=0, ge=0)

    attempt_observation_count: int = Field(default=0, ge=0)
    attempt_sum: int = Field(default=0, ge=0)
    hint_observation_count: int = Field(default=0, ge=0)
    hint_sum: int = Field(default=0, ge=0)
    response_time_observation_count: int = Field(default=0, ge=0)
    response_time_sum_ms: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_evidence_consistency(self):
        if self.correct_count + self.incorrect_count > self.interaction_count:
            raise ValueError(
                "correct_count + incorrect_count must not exceed interaction_count."
            )
        for count_name, sum_name in (
            ("attempt_observation_count", "attempt_sum"),
            ("hint_observation_count", "hint_sum"),
            ("response_time_observation_count", "response_time_sum_ms"),
        ):
            if getattr(self, count_name) == 0 and getattr(self, sum_name) != 0:
                raise ValueError(f"{sum_name} must be zero when {count_name} is zero.")
        return self

    @property
    def average_attempts(self) -> float | None:
        if self.attempt_observation_count == 0:
            return None
        return self.attempt_sum / self.attempt_observation_count

    @property
    def average_hints(self) -> float | None:
        if self.hint_observation_count == 0:
            return None
        return self.hint_sum / self.hint_observation_count

    @property
    def average_response_time_ms(self) -> float | None:
        if self.response_time_observation_count == 0:
            return None
        return self.response_time_sum_ms / self.response_time_observation_count

    @property
    def evidence_accuracy(self) -> float | None:
        evaluated_count = self.correct_count + self.incorrect_count
        if evaluated_count == 0:
            return None
        return self.correct_count / evaluated_count


class ShortTermMemoryUpsert(EvidenceProjection):
    student_id: uuid.UUID
    session_id: uuid.UUID
    student_external_id: str | None = None
    current_skill_id: uuid.UUID | None = None
    skill_name: str | None = None
    recent_accuracy: float | None = Field(default=None, ge=0, le=1)
    last_interaction_at: datetime | None = None

    @model_validator(mode="after")
    def validate_zero_accuracy(self):
        if self.interaction_count == 0 and self.recent_accuracy is not None:
            raise ValueError("recent_accuracy must be None when interaction_count is zero.")
        return self


class LongTermMemoryUpsert(EvidenceProjection):
    student_id: uuid.UUID
    student_external_id: str | None = None
    total_sessions: int = Field(default=0, ge=0)
    overall_accuracy: float | None = Field(default=None, ge=0, le=1)
    concept_count: int = Field(default=0, ge=0)
    last_interaction_at: datetime | None = None

    @model_validator(mode="after")
    def validate_zero_accuracy(self):
        if self.interaction_count == 0 and self.overall_accuracy is not None:
            raise ValueError("overall_accuracy must be None when interaction_count is zero.")
        return self


class ConceptMemoryUpsert(EvidenceProjection):
    student_id: uuid.UUID
    canonical_skill_id: uuid.UUID
    student_external_id: str | None = None
    skill_name: str | None = None
    accuracy: float | None = Field(default=None, ge=0, le=1)
    first_interaction_at: datetime | None = None
    last_interaction_at: datetime | None = None

    @model_validator(mode="after")
    def validate_concept_timing_and_accuracy(self):
        if self.interaction_count == 0 and self.accuracy is not None:
            raise ValueError("accuracy must be None when interaction_count is zero.")
        if (
            self.first_interaction_at is not None
            and self.last_interaction_at is not None
            and self.first_interaction_at > self.last_interaction_at
        ):
            raise ValueError("first_interaction_at must not exceed last_interaction_at.")
        return self


class ShortTermMemoryRecord(ShortTermMemoryUpsert):
    model_config = ConfigDict(from_attributes=True)
    updated_at: datetime


class LongTermMemoryRecord(LongTermMemoryUpsert):
    model_config = ConfigDict(from_attributes=True)
    updated_at: datetime


class ConceptMemoryRecord(ConceptMemoryUpsert):
    model_config = ConfigDict(from_attributes=True)
    updated_at: datetime
