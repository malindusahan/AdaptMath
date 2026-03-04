from pydantic import BaseModel, ConfigDict, Field

from app.schemas.dialogue import DialogueTurn
from app.schemas.pedagogical_move import PedagogicalMove
from app.schemas.planner import PlannerOutput


class TutorMathInput(BaseModel):
    """Problem context used once to prepare verified mathematical evidence."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    subtopic: str | None = None
    student_age: int = Field(..., ge=8, le=18)
    complexity_score: float = Field(..., gt=0.0, lt=1.0)
    planner_output: PlannerOutput | None = None
    previous_errors: list[str] = Field(default_factory=list)
    reteaching: bool = False


class TutorInput(TutorMathInput):
    """
    Context for one learner-facing teacher turn.

    pedagogical_move is selected externally by Omash's component. The Tutor
    executes that move for this turn and must not choose a different one.
    """

    pedagogical_move: PedagogicalMove
    conversation_history: list[DialogueTurn] = Field(default_factory=list)


class TutorOutput(BaseModel):
    """Exactly one learner-facing teacher turn; no assessment is generated here."""

    model_config = ConfigDict(extra="forbid")

    teaching_response: str = Field(..., min_length=1)
    fallback_used: bool = False
