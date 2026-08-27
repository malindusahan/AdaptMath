from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.input_validation import QUESTION_MAX_LENGTH, clean_required_text
from app.schemas.memory import MemoryContext
from app.schemas.profile import StudentProfile
from app.schemas.pedagogical_move import PedagogicalMove


class TutorRequestContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=QUESTION_MAX_LENGTH,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return clean_required_text(value, label="Question")

    profile: StudentProfile

    memory: MemoryContext

    pedagogical_move: PedagogicalMove