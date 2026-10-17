"""Classify genuinely observed production behavioural-data coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.database.history_repository import HistoricalInteraction


BehaviouralCoverageLevel = Literal[
    "FULL_BEHAVIOURAL_COVERAGE",
    "PARTIAL_BEHAVIOURAL_COVERAGE",
    "CORRECTNESS_ONLY_COVERAGE",
]


@dataclass(frozen=True)
class BehaviouralCoverage:
    level: BehaviouralCoverageLevel
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    has_attempt_evidence: bool
    has_hint_evidence: bool
    has_response_time_evidence: bool


def classify_behavioural_coverage(
    overall_history: list[HistoricalInteraction],
) -> BehaviouralCoverage:
    """Classify coverage from genuine measurements in the latest ten rows."""

    recent = overall_history[-10:]
    attempt_count = sum(
        1
        for item in recent
        if item.attempt_data_available and item.attempt_count is not None
    )
    hint_count = sum(1 for item in recent if item.hint_data_available)
    response_count = sum(
        1
        for item in recent
        if item.response_time_available and item.response_time_ms is not None
    )

    has_attempt = attempt_count > 0
    has_hint = hint_count > 0
    has_response = response_count > 0
    available_categories = sum([has_attempt, has_hint, has_response])

    if available_categories == 3:
        level = "FULL_BEHAVIOURAL_COVERAGE"
    elif available_categories == 0:
        level = "CORRECTNESS_ONLY_COVERAGE"
    else:
        level = "PARTIAL_BEHAVIOURAL_COVERAGE"

    return BehaviouralCoverage(
        level=level,
        recent_interaction_count=len(recent),
        attempt_observation_count=attempt_count,
        hint_observation_count=hint_count,
        response_time_observation_count=response_count,
        has_attempt_evidence=has_attempt,
        has_hint_evidence=has_hint,
        has_response_time_evidence=has_response,
    )

