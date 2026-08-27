"""Pydantic schemas for the focused Tutor Personalization Context."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class BehaviouralEvidenceSummary(BaseModel):
    """Aggregated behavioral observation counts and averages."""

    observation_count: int = 0
    total_value: float = 0.0
    average_value: float | None = None


class TutorMisconceptionSummary(BaseModel):
    """Active or repeated misconception context for the tutor."""

    misconception_id: str
    normalized_error: str
    display_error: str
    occurrence_count: int
    last_seen_at: str


class TutorInteractionSummary(BaseModel):
    """Recent interaction evidence item."""

    interaction_id: str
    session_id: str
    canonical_skill_id: str | None = None
    is_correct: bool | None = None
    attempt_count: int | None = None
    hint_count: int | None = None
    response_time_ms: float | None = None
    created_at: str


class TutorRepairSummary(BaseModel):
    """Past pedagogical repair history for the tutor to adapt future interventions."""

    repair_outcome_id: str
    session_id: str
    canonical_skill_id: str
    repair_action: str
    outcome: str
    score: float | None = None
    notes: str | None = None
    created_at: str


class TutorContextResponse(BaseModel):
    """Focused student personalization context tailored specifically for the Tutor agent."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    session_id: str | None = None
    skill_id: str | None = None

    # Learning state & evidence reliability
    current_learning_state: str | None = None
    evidence_strength: str | None = None
    behavioural_coverage: str | None = None

    # Performance counts
    recent_accuracy: float | None = None
    recent_correct_count: int = 0
    recent_incorrect_count: int = 0

    # Structured behavioral evidence
    attempt_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    hint_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    response_time_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)

    # Conceptual context
    misconceptions: list[TutorMisconceptionSummary] = Field(default_factory=list)
    recent_interactions: list[TutorInteractionSummary] = Field(default_factory=list)
    recent_repairs: list[TutorRepairSummary] = Field(default_factory=list)
