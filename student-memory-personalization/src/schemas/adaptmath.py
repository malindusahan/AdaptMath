"""Strict, evidence-only integration contract for AdaptMath completed attempts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AdaptMathQuestionEvidence(BaseModel):
    """One objectively evaluated assessment question."""

    model_config = ConfigDict(extra="forbid")

    external_interaction_id: str = Field(min_length=1, max_length=255)
    question_id: str = Field(min_length=1, max_length=255)
    problem_id: str = Field(min_length=1, max_length=255)
    student_answer: str | None = Field(default=None, max_length=4000)
    expected_answer: str | None = Field(default=None, max_length=4000)
    is_correct: bool
    identified_error: str | None = Field(default=None, max_length=1000)

    @field_validator(
        "external_interaction_id",
        "question_id",
        "problem_id",
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Identifier fields must not be blank.")
        return value

    @field_validator("student_answer", "expected_answer", "identified_error")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class AdaptMathCompletedAttemptRequest(BaseModel):
    """Stable, transactional request emitted after authoritative BKT completion."""

    model_config = ConfigDict(extra="forbid")

    external_student_id: str = Field(min_length=1, max_length=255)
    external_session_id: str = Field(min_length=1, max_length=255)
    external_attempt_id: str = Field(min_length=1, max_length=255)
    source: Literal["adaptmath"] = "adaptmath"
    target_skill: str = Field(min_length=1, max_length=255)
    session_status: Literal["ACTIVE", "COMPLETED"]
    questions: list[AdaptMathQuestionEvidence] = Field(min_length=1, max_length=20)

    @field_validator(
        "external_student_id",
        "external_session_id",
        "external_attempt_id",
        "target_skill",
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Identifier fields must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_stable_question_keys(self):
        question_ids = [item.question_id for item in self.questions]
        interaction_ids = [item.external_interaction_id for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question_id values must be unique within an attempt.")
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError(
                "external_interaction_id values must be unique within an attempt."
            )
        for item in self.questions:
            expected_interaction_id = f"{self.external_attempt_id}:{item.question_id}"
            if item.external_interaction_id != expected_interaction_id:
                raise ValueError(
                    "external_interaction_id must equal "
                    "'<external_attempt_id>:<question_id>'."
                )
            if item.problem_id != self.external_session_id:
                raise ValueError("problem_id must equal external_session_id.")
        return self


class AdaptMathCompletedAttemptResponse(BaseModel):
    """Evidence-only acknowledgement; deliberately excludes learning-state fields."""

    model_config = ConfigDict(extra="forbid")

    receipt_id: str
    external_student_id: str
    external_session_id: str
    external_attempt_id: str
    canonical_skill_id: str
    source: Literal["adaptmath"]
    status: Literal["STORED", "ALREADY_STORED"]
    session_status: Literal["ACTIVE", "COMPLETED"]
    question_count: int
    correct_count: int
    incorrect_count: int
    created_at: str
