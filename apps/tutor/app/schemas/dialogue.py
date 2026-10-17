from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.pedagogical_move import PedagogicalMove


class DialogueTurn(BaseModel):
    """One learner-facing turn in the active tutoring dialogue."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["teacher", "student"]
    content: str = Field(..., min_length=1, max_length=5000)
    pedagogical_move: PedagogicalMove | None = None

    @model_validator(mode="after")
    def validate_move_ownership(self):
        if self.role == "student" and self.pedagogical_move is not None:
            raise ValueError(
                "Student dialogue turns cannot contain a pedagogical move."
            )
        return self
