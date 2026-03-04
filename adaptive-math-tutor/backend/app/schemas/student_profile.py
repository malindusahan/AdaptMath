"""Learner-facing, read-only student profile contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProfileSkill(BaseModel):
    skill: str
    mastery_probability: float = Field(ge=0.0, le=1.0)
    mastery_status: Literal["weak", "partial", "strong"]
    previous_mastery_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    trend: Literal["improving", "declining", "steady", "not_available"]
    last_practiced_at: str | None = None


class ProfileRecentSession(BaseModel):
    completed_at: str
    skills: list[str]
    questions_answered: int = Field(ge=0)
    correct_answers: int = Field(ge=0)
    incorrect_answers: int = Field(ge=0)


class ProfileWeeklySummary(BaseModel):
    window_days: Literal[7] = 7
    sessions: int = Field(ge=0)
    skills_practiced: int = Field(ge=0)
    questions_answered: int = Field(ge=0)
    correct_answers: int = Field(ge=0)


class ProfileFocusNext(BaseModel):
    skill: str
    reason: str
    source: Literal["student_model_learning_path"] = "student_model_learning_path"


class StudentProfileResponse(BaseModel):
    username: str
    age: int | None = None
    recent_session: ProfileRecentSession | None = None
    weekly_summary: ProfileWeeklySummary
    focus_next: ProfileFocusNext | None = None
    skills: list[ProfileSkill]
    total_practiced_skills: int = Field(ge=0)
