from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.input_validation import QUESTION_MAX_LENGTH, clean_required_text

from app.agents.complexity.service import (
    get_complexity_service,
)


router = APIRouter(
    prefix="/complexity",
    tags=["complexity"],
)


class ComplexityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=QUESTION_MAX_LENGTH,
        description="Mathematics question to analyse.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        return clean_required_text(
            value,
            label="Question",
        )


class ComplexityResponse(BaseModel):
    complexity_score: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Continuous predicted mathematical "
            "difficulty score."
        ),
    )


@router.post(
    "/predict",
    response_model=ComplexityResponse,
)
def predict_complexity(
    request: ComplexityRequest,
) -> ComplexityResponse:
    """
    Predict mathematical problem complexity.

    The endpoint returns the continuous score
    produced by the trained ML model.

    It does not create Easy/Medium/Hard labels
    or apply pedagogical thresholds.
    """

    service = get_complexity_service()

    try:
        score = service.predict(
            request.question
        )

    except (
        ValueError,
        TypeError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Complexity prediction failed."
            ),
        ) from exc

    return ComplexityResponse(
        complexity_score=score
    )