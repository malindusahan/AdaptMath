"""Schemas for dedicated Topic Classification API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicClassifyRequest(BaseModel):
    """Request payload for topic classification."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural language math question or conversational student input.",
        examples=["Solve 3x + 5 = 20."],
    )


class TopicClassifyResponse(BaseModel):
    """Response payload for topic classification."""

    topic: str | None = Field(
        default=None,
        description="Canonical display name of the extracted topic (1 of 111 canonical values), or null if non-math.",
        examples=["Linear Equations"],
    )
    skill_id: str | None = Field(
        default=None,
        description="Canonical skill identifier code (e.g., SKILL_193), or null if non-math.",
        examples=["SKILL_193"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Neural cosine similarity confidence score (0.0 to 1.0).",
        examples=[0.94],
    )
    is_math: bool = Field(
        ...,
        description="Whether the input question was classified as a valid mathematical topic (confidence >= threshold).",
        examples=[True],
    )
    model_version: str = Field(
        ...,
        description="Version string of the active topic extractor model.",
        examples=["phase16-minilm-ft-v2"],
    )
