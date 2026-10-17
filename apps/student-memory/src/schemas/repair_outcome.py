"""Pydantic schemas for storing and returning Repair Outcome events."""

from __future__ import annotations

import uuid
from pydantic import BaseModel, ConfigDict, Field


class RepairOutcomeCreateRequest(BaseModel):
    """Request payload from FAPR-LB reporting a pedagogical repair intervention result."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "student_id": "student_101",
                "session_id": "session_303",
                "skill_id": "14ff089f-5134-4dc5-a624-6a20125a6b1f",
                "interaction_id": "2f63827b-5cf0-4a31-8ae7-8560eac6fe9a",
                "repair_action": "inverse_operation_prompt",
                "outcome": "RESOLVED",
                "score": 1.0,
                "notes": "Student applied inverse operations correctly after prompt.",
            }
        },
    )

    student_id: str = Field(..., min_length=1, description="Student external identifier or UUID.")
    session_id: str = Field(..., min_length=1, description="Session external identifier or UUID.")
    skill_id: str = Field(..., min_length=1, description="Canonical skill identifier or UUID.")
    interaction_id: str | None = Field(default=None, description="Optional interaction UUID tied to repair.")
    repair_action: str = Field(..., min_length=1, description="The pedagogical repair action applied.")
    outcome: str = Field(..., min_length=1, description="Result outcome e.g. RESOLVED, PARTIALLY_RESOLVED, UNRESOLVED.")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Normalized score [0.0, 1.0].")
    notes: str | None = Field(default=None, description="Optional diagnostic notes or observations.")


class RepairOutcomeResponse(BaseModel):
    """Stored repair outcome record returned to client."""

    model_config = ConfigDict(extra="ignore")

    repair_outcome_id: str
    student_id: str
    session_id: str
    canonical_skill_id: str
    interaction_id: str | None = None
    repair_action: str
    outcome: str
    score: float | None = None
    notes: str | None = None
    created_at: str
