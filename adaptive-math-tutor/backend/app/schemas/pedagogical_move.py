from typing import Literal

from pydantic import BaseModel, ConfigDict


PedagogicalMove = Literal[
    "telling",
    "focus",
    "generic",
    "probing",
]


class PedagogicalMoveSelection(BaseModel):
    """
    Pedagogical move selected by Omash's external move-selector component.

    AdaptMath consumes this decision. It does not choose or override the move.
    """

    model_config = ConfigDict(extra="forbid")

    move: PedagogicalMove
