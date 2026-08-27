"""Build the deployed model's historical feature contract from stored history."""

from __future__ import annotations

import math
from statistics import median

from src.database.history_repository import HistoricalInteraction


MODEL_FEATURES = [
    "previous_interaction_count",
    "previous_skill_interaction_count",
    "recent_accuracy_change_5_vs_10",
    "recent_attempt_log_mean_10",
    "recent_multi_attempt_rate_10",
    "recent_hint_usage_rate_10",
    "recent_hint_available_rate_10",
    "recent_response_log_mean_10",
    "recent_response_median_ms_10",
    "recent_skill_accuracy_change_3_vs_previous_2",
    "has_full_skill_window_5",
    "has_recent_hint_usage_evidence",
]


def _accuracy(interactions: list[HistoricalInteraction]) -> float | None:
    if not interactions:
        return None
    return sum(int(item.is_correct) for item in interactions) / len(interactions)


def _recent_accuracy_change(history: list[HistoricalInteraction]) -> float:
    """Return latest-five accuracy minus latest-ten accuracy."""

    if not history:
        return 0.0
    accuracy_5 = _accuracy(history[-5:])
    accuracy_10 = _accuracy(history[-10:])
    return float(accuracy_5 - accuracy_10)


def _attempt_features(
    history: list[HistoricalInteraction],
) -> tuple[float | None, float | None]:
    """Use only measured attempts among the latest ten interactions."""

    attempts = [
        item.attempt_count
        for item in history[-10:]
        if item.attempt_data_available and item.attempt_count is not None
    ]
    if not attempts:
        return None, None

    log_mean = sum(math.log1p(value) for value in attempts) / len(attempts)
    multi_attempt_rate = sum(value > 1 for value in attempts) / len(attempts)
    return float(log_mean), float(multi_attempt_rate)


def _hint_features(
    history: list[HistoricalInteraction],
) -> tuple[float | None, float, int]:
    """Keep recent hint availability separate from measurable usage."""

    recent = history[-10:]
    if not recent:
        return None, 0.0, 0

    hint_available = [item for item in recent if item.hint_data_available]
    available_rate = len(hint_available) / len(recent)
    usage_values = [
        item.hint_count / item.hint_total
        for item in hint_available
        if (
            item.hint_count is not None
            and item.hint_total is not None
            and item.hint_total > 0
        )
    ]

    if not usage_values:
        return None, float(available_rate), 0

    return (
        float(sum(usage_values) / len(usage_values)),
        float(available_rate),
        1,
    )


def _response_features(
    history: list[HistoricalInteraction],
) -> tuple[float | None, float | None]:
    """Use only measured response times among the latest ten interactions."""

    values = [
        item.response_time_ms
        for item in history[-10:]
        if item.response_time_available and item.response_time_ms is not None
    ]
    if not values:
        return None, None

    log_mean = sum(math.log1p(value) for value in values) / len(values)
    return float(log_mean), float(median(values))


def _skill_accuracy_change(
    skill_history: list[HistoricalInteraction],
) -> float | None:
    """Return latest-three minus preceding-two skill accuracy."""

    if len(skill_history) < 5:
        return None

    recent_5 = skill_history[-5:]
    previous_accuracy = _accuracy(recent_5[:2])
    recent_accuracy = _accuracy(recent_5[2:])
    return float(recent_accuracy - previous_accuracy)


def build_production_features(
    overall_history: list[HistoricalInteraction],
    skill_history: list[HistoricalInteraction],
) -> dict[str, float | int | None]:
    """Build the exact 12-feature input contract from previous interactions."""

    attempt_log_mean, multi_attempt_rate = _attempt_features(overall_history)
    hint_usage_rate, hint_available_rate, has_hint_evidence = _hint_features(
        overall_history
    )
    response_log_mean, response_median = _response_features(overall_history)

    features = {
        "previous_interaction_count": len(overall_history),
        "previous_skill_interaction_count": len(skill_history),
        "recent_accuracy_change_5_vs_10": _recent_accuracy_change(overall_history),
        "recent_attempt_log_mean_10": attempt_log_mean,
        "recent_multi_attempt_rate_10": multi_attempt_rate,
        "recent_hint_usage_rate_10": hint_usage_rate,
        "recent_hint_available_rate_10": hint_available_rate,
        "recent_response_log_mean_10": response_log_mean,
        "recent_response_median_ms_10": response_median,
        "recent_skill_accuracy_change_3_vs_previous_2": _skill_accuracy_change(
            skill_history
        ),
        "has_full_skill_window_5": int(len(skill_history) >= 5),
        "has_recent_hint_usage_evidence": has_hint_evidence,
    }

    assert list(features.keys()) == MODEL_FEATURES
    return features

