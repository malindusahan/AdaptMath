from typing import Any

from pydantic import BaseModel, Field


class TeachingStrategy(BaseModel):
    strategy_id: str | None = None

    strategy_name: str = Field(
        ...,
        min_length=1,
    )

    strategy_description: str = Field(
        ...,
        min_length=1,
    )

    teaching_guidance: list[str] = Field(
        default_factory=list,
    )


class StrategyInput(BaseModel):
    """
    Evidence supplied to the Strategy Agent when a reteaching
    attempt has been requested.

    The schema is deliberately independent of the current
    implementation so the same contract can later be mapped to
    the teammate's real Strategy API without changing Tutor
    behavior.
    """

    student_id: str = Field(
        ...,
        min_length=1,
    )

    question: str = Field(
        ...,
        min_length=1,
    )

    topic: str = Field(
        ...,
        min_length=1,
    )

    subtopic: str | None = None

    student_age: int = Field(
        ...,
        ge=8,
        le=18,
    )

    complexity_score: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
    )

    relevant_history: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    correct_answers: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    wrong_answers: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
    )

    identified_errors: list[str] = Field(
        default_factory=list,
    )

    previous_strategies: list[str] = Field(
        default_factory=list,
    )
