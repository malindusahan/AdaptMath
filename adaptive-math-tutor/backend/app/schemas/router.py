from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.memory import MemoryContext
from app.schemas.profile import StudentProfile


class RouterInput(BaseModel):
    """
    Context supplied to the GenAI Router.
    """

    question: str = Field(
        ...,
        min_length=1,
    )

    profile: StudentProfile

    memory: MemoryContext

    complexity_score: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description=(
            "Continuous mathematical problem complexity "
            "score produced by the trained Complexity Model."
        ),
    )


class RouterOutput(BaseModel):
    """
    Structured pre-tutoring workflow selected by the GenAI Router.

    direct_tutor:
        Tutor directly.

    planned_tutor:
        Planner -> Tutor.

    Assessment is universal after tutoring and is therefore not
    selected by the Router.
    """

    route: Literal[
        "direct_tutor",
        "planned_tutor",
    ]

    reason: str = Field(
        ...,
        min_length=1,
        description=(
            "Brief evidence-based justification "
            "for whether a separate planning stage is useful."
        ),
    )
