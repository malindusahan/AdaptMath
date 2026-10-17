"""Schemas for Learning Session management in student chat interface."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SessionItemResponse(BaseModel):
    """Session item summary returned to chat sidebar."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    external_session_id: str
    student_id: str
    status: str
    started_at: str
