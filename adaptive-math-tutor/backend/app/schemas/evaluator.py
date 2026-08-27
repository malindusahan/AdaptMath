from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.input_validation import (
    ANSWER_MAX_LENGTH,
    QUESTION_ID_MAX_LENGTH,
    clean_required_text,
)
from app.schemas.assessment import AssessmentQuestion


class StudentAnswer(BaseModel):
    """One validated answer submitted by the learner."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1, max_length=QUESTION_ID_MAX_LENGTH)
    answer: str = Field(..., min_length=1, max_length=ANSWER_MAX_LENGTH)

    @field_validator("question_id", "answer")
    @classmethod
    def normalize_required_fields(cls, value: str, info) -> str:
        return clean_required_text(
            value,
            label=info.field_name.replace("_", " ").title(),
        )


class EvaluatedAnswer(BaseModel):
    """Evaluator judgment for one assessment question."""

    question_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    is_correct: bool
    feedback: str = Field(..., min_length=1)
    identified_error: str | None = None


class EvaluatorInput(BaseModel):
    """Information supplied to the Evaluator after the 3-question check."""

    original_question: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    subtopic: str | None = None
    assessment_questions: list[AssessmentQuestion] = Field(min_length=1)
    student_answers: list[StudentAnswer] = Field(min_length=1)


class EvaluatorOutput(BaseModel):
    """Structured evidence used by Memory and the adaptive reteaching loop."""

    correct_answers: list[EvaluatedAnswer] = Field(default_factory=list)
    wrong_answers: list[EvaluatedAnswer] = Field(default_factory=list)
    identified_errors: list[str] = Field(default_factory=list)
    needs_reteaching: bool
    overall_feedback: str = Field(..., min_length=1)
