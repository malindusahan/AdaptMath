from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.input_validation import STUDENT_ID_MAX_LENGTH, clean_required_text


class StudentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(
        ...,
        min_length=1,
        max_length=STUDENT_ID_MAX_LENGTH,
    )
    age: int = Field(..., ge=8, le=18)

    @field_validator("student_id")
    @classmethod
    def normalize_student_id(cls, value: str) -> str:
        return clean_required_text(value, label="Student id")
