from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.dialogue import DialogueTurn
from app.schemas.planner import PlannerOutput


TeachingProgressStatus = Literal[
    "continue_teaching",
    "ready_for_assessment",
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


class TeachingProgressOutput(BaseModel):
    """GenAI judgment about whether interactive teaching should continue."""

    model_config = ConfigDict(extra="forbid")

    status: TeachingProgressStatus
    reason: str = Field(..., min_length=1)
