from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.dialogue import DialogueTurn
from app.schemas.planner import PlannerOutput


TeachingProgressStatus = Literal[
    "continue_teaching",
    "ready_for_assessment",
]

StudentResponseCorrectness = Literal[
    "correct",
    "partial",
    "incorrect",
    "unknown",
]

StudentResponseEvidenceCategory = Literal[
    "mathematical_evidence",
    "knowledge_state_evidence",
    "interaction_management",
    "acknowledgement",
    "unclear_no_evidence",
]

OriginalProblemCompletion = Literal[
    "not_yet",
    "complete",
    "not_applicable_reteaching",
]


class TeachingProgressInput(BaseModel):
    """Evidence supplied to the teaching-progress judge after a student turn."""

    model_config = ConfigDict(extra="forbid")

    original_question: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    subtopic: str | None = None
    student_age: int = Field(..., ge=8, le=18)
    conversation_history: list[DialogueTurn] = Field(min_length=2)
    planner_output: PlannerOutput | None = None
    previous_errors: list[str] = Field(default_factory=list)
    reteaching: bool = False
    verified_math_evidence: list[dict[str, str]] = Field(default_factory=list)


class TeachingProgressOutput(BaseModel):
    """GenAI judgment about whether interactive teaching should continue."""

    model_config = ConfigDict(extra="forbid")

    status: TeachingProgressStatus
    reason: str = Field(..., min_length=1)
    latest_response_correctness: StudentResponseCorrectness
    correctness_confidence: float = Field(..., ge=0.0, le=1.0)
    correctness_reason: str = Field(..., min_length=1)
    latest_response_evidence_category: StudentResponseEvidenceCategory = (
        "mathematical_evidence"
    )
    original_problem_completion: OriginalProblemCompletion
    reasoning_sufficient_for_assessment: bool
    readiness_evidence: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def enforce_assessment_readiness(self):
        if self.status != "ready_for_assessment":
            return self

        if self.original_problem_completion == "not_yet":
            raise ValueError(
                "Assessment cannot start before the learner completes the "
                "original problem."
            )
        if not self.reasoning_sufficient_for_assessment:
            raise ValueError(
                "Assessment cannot start without sufficient learner reasoning."
            )
        if self.latest_response_correctness != "correct":
            raise ValueError(
                "Assessment readiness requires a correct latest mathematical "
                "response."
            )
        if self.latest_response_evidence_category != "mathematical_evidence":
            raise ValueError(
                "Assessment readiness requires current mathematical evidence."
            )
        return self
