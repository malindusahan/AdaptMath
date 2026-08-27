"""Pydantic schemas for the focused FAPR-LB (Failure-Aware Pedagogical Repair) Context."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class BehaviouralEvidenceSummary(BaseModel):
    """Aggregated behavioral observation counts and averages."""

    observation_count: int = 0
    total_value: float = 0.0
    average_value: float | None = None


class FAPRMisconceptionSummary(BaseModel):
    """Active student misconception evidence for repair selection."""

    misconception_id: str
    normalized_error: str
    display_error: str
    occurrence_count: int
    last_seen_at: str


class FAPRInteractionSummary(BaseModel):
    """Recent interaction evidence item including student utterance and detected error."""

    interaction_id: str
    session_id: str
    canonical_skill_id: str | None = None
    student_utterance: str | None = None
    identified_error: str | None = None
    is_correct: bool
    attempt_count: int
    hint_count: int
    response_time_ms: float | None = None
    created_at: str


class FAPRRepairSummary(BaseModel):
    """Past pedagogical repair history and outcomes."""

    repair_outcome_id: str
    session_id: str
    canonical_skill_id: str
    repair_action: str
    outcome: str
    score: float | None = None
    notes: str | None = None
    created_at: str


class FAPRContextResponse(BaseModel):
    """Focused student repair evidence tailored specifically for the FAPR-LB repair selector."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    session_id: str | None = None
    skill_id: str | None = None

    # Learning state & reliability markers
    current_learning_state: str | None = None
    evidence_strength: str | None = None
    behavioural_coverage: str | None = None

    # Error & performance counts
    recent_accuracy: float | None = None
    recent_incorrect_count: int = 0

    # Structured behavioral evidence
    attempt_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    hint_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    response_time_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)

    # Conceptual context & repair history
    misconceptions: list[FAPRMisconceptionSummary] = Field(default_factory=list)
    recent_interactions: list[FAPRInteractionSummary] = Field(default_factory=list)
    previous_repairs: list[FAPRRepairSummary] = Field(default_factory=list)

    # Latest student utterance
    latest_student_utterance: str | None = None
