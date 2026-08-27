"""Pydantic schemas for the focused Planner Personalization Context."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class BehaviouralEvidenceSummary(BaseModel):
    """Aggregated behavioral observation counts and averages."""

    observation_count: int = 0
    total_value: float = 0.0
    average_value: float | None = None


class PlannerMisconceptionSummary(BaseModel):
    """Active misconception summary relevant for curriculum progression and remediation planning."""

    misconception_id: str
    normalized_error: str
    display_error: str
    occurrence_count: int
    last_seen_at: str


class PlannerInteractionSummary(BaseModel):
    """Recent interaction evidence item."""

    interaction_id: str
    session_id: str
    canonical_skill_id: str | None = None
    is_correct: bool
    attempt_count: int
    hint_count: int
    response_time_ms: float | None = None
    created_at: str


class PlannerContextResponse(BaseModel):
    """Focused student performance and longitudinal evidence tailored specifically for the Planner agent."""

    model_config = ConfigDict(extra="ignore")

    student_id: str
    session_id: str | None = None
    skill_id: str | None = None

    # Session-level performance
    session_interaction_count: int = 0
    session_accuracy: float | None = None
    session_correct_count: int = 0
    session_incorrect_count: int = 0

    # Behavioral evidence sums and averages
    attempt_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    hint_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)
    response_time_evidence: BehaviouralEvidenceSummary = Field(default_factory=BehaviouralEvidenceSummary)

    # Learning state & reliability indicators
    current_learning_state: str | None = None
    evidence_strength: str | None = None
    behavioural_coverage: str | None = None

    # Concept-level performance (scoped to requested skill)
    concept_interaction_count: int = 0
    concept_accuracy: float | None = None
    concept_correct_count: int = 0
    concept_incorrect_count: int = 0

    # Longitudinal / Long-term metrics
    long_term_interaction_count: int = 0
    overall_accuracy: float | None = None
    total_sessions: int = 0
    concept_count: int = 0

    # Conceptual context & chronological history
    misconceptions: list[PlannerMisconceptionSummary] = Field(default_factory=list)
    recent_interactions: list[PlannerInteractionSummary] = Field(default_factory=list)
