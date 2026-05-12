"""API routes for dedicated Topic Classification."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.api.auth import verify_service_api_key
from src.schemas.topic import TopicClassifyRequest, TopicClassifyResponse
from src.topic_extraction.topic_extractor import get_topic_extractor

router = APIRouter(
    prefix="/topic",
    tags=["Topic Classification"],
)


@router.post(
    "/classify",
    response_model=TopicClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Math Question to Canonical Topic",
    description=(
        "Classify a natural-language student question into one of the 111 Canonical Math Topics "
        "using the pure fine-tuned Sentence Transformer neural model. "
        "Returns topic=null and is_math=false for off-topic greetings or non-math input."
    ),
    dependencies=[Depends(verify_service_api_key)],
)
def classify_topic(request: TopicClassifyRequest) -> TopicClassifyResponse:
    """Classify student question into a single canonical topic."""
    extractor = get_topic_extractor()
    result = extractor.extract(request.question)

    is_math = not result.is_abstain and result.display_name is not None

    return TopicClassifyResponse(
        topic=result.display_name if is_math else None,
        skill_id=result.skill_code if is_math else None,
        confidence=round(float(result.confidence), 4),
        is_math=is_math,
        model_version=result.model_version,
    )
