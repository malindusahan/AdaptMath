from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


from app.core.input_validation import (
    EVENT_TYPE_MAX_LENGTH,
    MAX_CONTEXT_ITEMS,
    MAX_HISTORY_ITEMS,
    STUDENT_ID_MAX_LENGTH,
    SUBTOPIC_MAX_LENGTH,
    TOPIC_MAX_LENGTH,
    clean_optional_text,
    clean_required_text,
    clean_text_list,
)


class LearnerHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str | None = Field(default=None, max_length=TOPIC_MAX_LENGTH)
    event_type: str = Field(..., min_length=1, max_length=EVENT_TYPE_MAX_LENGTH)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="History topic")

    @field_validator("event_type")
    @classmethod
    def normalize_event_type(cls, value: str) -> str:
        return clean_required_text(value, label="Event type")


class MemoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(
        ...,
        min_length=1,
        max_length=STUDENT_ID_MAX_LENGTH,
    )
    topic: str = Field(
        ...,
        min_length=1,
        max_length=TOPIC_MAX_LENGTH,
    )
    subtopic: str | None = Field(
        default=None,
        max_length=SUBTOPIC_MAX_LENGTH,
    )

    relevant_history: list[LearnerHistoryItem] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
    )

    previous_errors: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )

    previous_strategies: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )

    @field_validator("student_id", "topic")
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
