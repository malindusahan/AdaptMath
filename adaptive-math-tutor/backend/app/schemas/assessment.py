from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.dialogue import DialogueTurn
from app.schemas.planner import PlannerOutput


class AssessmentQuestion(BaseModel):
    """
    One post-teaching assessment item.

    expected_answer is private backend evidence and must never be exposed to the
    learner before the learner answers.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)

    @field_validator("expected_answer", mode="before")
    @classmethod
    def normalize_expected_answer(cls, value):
        if isinstance(value, (int, float, bool)):
            return str(value)
        return value


class AssessmentGenerationInput(BaseModel):
    """Context used by the Evaluator to generate the 3-question check."""

    model_config = ConfigDict(extra="forbid")

    original_question: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    subtopic: str | None = None
    student_age: int = Field(..., ge=8, le=18)
    conversation_history: list[DialogueTurn] = Field(min_length=2)
    planner_output: PlannerOutput | None = None
    previous_errors: list[str] = Field(default_factory=list)
    reteaching: bool = False


class AssessmentGenerationOutput(BaseModel):
    """Exactly three assessment questions generated after teaching completes."""

    model_config = ConfigDict(extra="forbid")

    assessment_questions: list[AssessmentQuestion] = Field(
        ..., min_length=3, max_length=3
    )

    @model_validator(mode="after")
    def require_unique_question_ids(self):
        ids = [question.question_id for question in self.assessment_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Assessment question IDs must be unique.")
        return self
