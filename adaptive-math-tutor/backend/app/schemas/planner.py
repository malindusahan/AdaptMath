from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.input_validation import (
    MAX_CONTEXT_ITEMS,
    QUESTION_MAX_LENGTH,
    SUBTOPIC_MAX_LENGTH,
    TOPIC_MAX_LENGTH,
    clean_optional_text,
    clean_required_text,
    clean_text_list,
)


class PlannerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LENGTH)
    topic: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)
    subtopic: str | None = Field(default=None, max_length=SUBTOPIC_MAX_LENGTH)
    student_age: int = Field(..., ge=8, le=18)
    complexity_score: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description=(
            "Continuous mathematical complexity score produced by the trained "
            "Complexity Model."
        ),
    )
    previous_errors: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )
    previous_strategies: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )

    @field_validator("question", "topic")
    @classmethod
    def normalize_required_fields(cls, value: str, info) -> str:
        return clean_required_text(
            value,
            label=info.field_name.replace("_", " ").title(),
        )

    @field_validator("subtopic")
    @classmethod
    def normalize_subtopic(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Subtopic")

    @field_validator("previous_errors", "previous_strategies")
    @classmethod
    def normalize_context_lists(cls, values: list[str], info) -> list[str]:
        return clean_text_list(
            values,
            label=info.field_name.replace("_", " ").title(),
        )


class PlannerOutput(BaseModel):
    learning_goal: str = Field(..., min_length=1)
    required_concepts: list[str] = Field(default_factory=list)
    teaching_sequence: list[str] = Field(default_factory=list)
    anticipated_difficulties: list[str] = Field(default_factory=list)
    tutor_guidance: list[str] = Field(default_factory=list)
