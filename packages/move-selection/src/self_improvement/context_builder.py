"""Canonical SI7-A candidate context implementation.

This module implements the candidate 12-dimensional context used by the
MRB1-informed self-improvement architecture. It enforces temporal causality:
current-attempt outcomes cannot be decision features. MD6 probabilities
describe the current frozen-selector state, while MRB1 features describe the
previous completed tutoring attempt. The scientific necessity of all twelve
dimensions is not assumed and will be tested later through context ablation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Final

import numpy as np


MOVE_ORDER: Final[tuple[str, ...]] = (
    "generic",
    "probing",
    "focus",
    "telling",
)

MRB1_LABELS: Final[tuple[str, ...]] = (
    "No",
    "To some extent",
    "Yes",
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "mastery_before",
    "prev_evaluator_rate",
    "prev_mastery_delta",
    "prev_mistake_identification",
    "prev_mistake_location",
    "prev_providing_guidance",
    "prev_actionability",
    "has_previous_feedback",
)

_PROBABILITY_SUM_ATOL: Final[float] = 1e-5


def _finite_float(name: str, value: object) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")

    return result


def _unit_interval(name: str, value: object) -> float:
    """Validate and return a finite value in the closed interval [0, 1]."""

    result = _finite_float(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}.")
    return result


def _evaluator_score(value: object) -> int:
    """Validate and return a completed-attempt evaluator score."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            "evaluator_score must be an integer in {0, 1, 2, 3}, "
            f"got {value!r}."
        )

    numeric = int(value)
    if numeric not in {0, 1, 2, 3}:
        raise ValueError(
            "evaluator_score must be one of {0, 1, 2, 3}, "
            f"got {value!r}."
        )
    return numeric


def _probability_vector(
    probabilities: Mapping[str, object],
    expected_labels: tuple[str, ...],
    source_name: str,
) -> np.ndarray:
    """Validate an exact probability mapping and return canonical ordering."""

    if not isinstance(probabilities, Mapping):
        raise TypeError(f"{source_name} probabilities must be a mapping.")

    actual_labels = set(probabilities)
    required_labels = set(expected_labels)
    if actual_labels != required_labels:
        missing = sorted(required_labels - actual_labels, key=repr)
        extra = sorted(actual_labels - required_labels, key=repr)
        raise ValueError(
            f"{source_name} probabilities must contain exactly "
            f"{list(expected_labels)}; missing={missing}, extra={extra}."
        )

    ordered = np.asarray(
        [
            _finite_float(
                f"{source_name} probability {label!r}",
                probabilities[label],
            )
            for label in expected_labels
        ],
        dtype=np.float64,
    )

    if np.any(ordered < 0.0):
        raise ValueError(f"{source_name} probabilities cannot be negative.")
    if np.any(ordered > 1.0):
        raise ValueError(f"{source_name} probabilities cannot exceed 1.")

    total = float(ordered.sum())
    if not np.isclose(
        total,
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_SUM_ATOL,
    ):
        raise ValueError(
            f"{source_name} probabilities must sum approximately to 1; "
            f"got {total}."
        )

    return ordered


def validate_md6_probabilities(
    probabilities: Mapping[str, float],
) -> np.ndarray:
    """Return validated MD6 probabilities in canonical move order.

    The returned order is ``generic``, ``probing``, ``focus``, ``telling``.
    The input must contain exactly those keys with finite, nonnegative values
    whose sum is approximately one.
    """

    return _probability_vector(
        probabilities=probabilities,
        expected_labels=MOVE_ORDER,
        source_name="MD6",
    )


def mrb1_expected_score(probabilities: Mapping[str, float]) -> float:
    """Convert MRB1 class probabilities to a continuous ordinal score.

    The full ordinal interpretation is::

        0.0 * P(No) + 0.5 * P(To some extent) + 1.0 * P(Yes)

    This expectation preserves all three probabilities and deliberately does
    not reduce them to a hard argmax label.
    """

    ordered = _probability_vector(
        probabilities=probabilities,
        expected_labels=MRB1_LABELS,
        source_name="MRB1",
    )
    score = float(0.0 * ordered[0] + 0.5 * ordered[1] + ordered[2])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("Validated MRB1 expectation fell outside [0, 1].")
    return score


@dataclass(frozen=True, slots=True)
class PreviousAttemptFeedback:
    """Validated outcomes from one completed previous tutoring attempt.

    Validation guarantees the value domains. The caller remains responsible
    for supplying feedback from the genuinely preceding completed attempt.
    """

    evaluator_score: int
    mastery_before: float
    mastery_after: float
    mistake_identification: float
    mistake_location: float
    providing_guidance: float
    actionability: float

    def __post_init__(self) -> None:
        """Validate and canonicalize all stored feedback values."""

        object.__setattr__(
            self,
            "evaluator_score",
            _evaluator_score(self.evaluator_score),
        )
        for field_name in (
            "mastery_before",
            "mastery_after",
            "mistake_identification",
            "mistake_location",
            "providing_guidance",
            "actionability",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_interval(field_name, getattr(self, field_name)),
            )


def build_lints_context(
    md6_probabilities: Mapping[str, float],
    mastery_before: float,
    previous_feedback: PreviousAttemptFeedback | None = None,
) -> np.ndarray:
    """Build the causally valid SI7-A context vector.

    Only current frozen-MD6 probabilities, current ``mastery_before``, and
    outcomes from a completed previous attempt are accepted. Current evaluator,
    mastery-after, MRB1, student-response, and future information are excluded
    from the API because they do not exist when the current decision is made.

    Returns:
        A finite ``numpy.ndarray`` with dtype ``float64`` and shape ``(12,)``
        in exactly :data:`FEATURE_NAMES` order.
    """

    md6 = validate_md6_probabilities(md6_probabilities)
    current_mastery = _unit_interval("mastery_before", mastery_before)

    if previous_feedback is None:
        historical = np.zeros(7, dtype=np.float64)
    else:
        if not isinstance(previous_feedback, PreviousAttemptFeedback):
            raise TypeError(
                "previous_feedback must be PreviousAttemptFeedback or None."
            )

        previous_mastery_delta = float(
            np.clip(
                previous_feedback.mastery_after
                - previous_feedback.mastery_before,
                -1.0,
                1.0,
            )
        )
        historical = np.asarray(
            [
                previous_feedback.evaluator_score / 3.0,
                previous_mastery_delta,
                previous_feedback.mistake_identification,
                previous_feedback.mistake_location,
                previous_feedback.providing_guidance,
                previous_feedback.actionability,
                1.0,
            ],
            dtype=np.float64,
        )

    feature_values = {
        "md6_p_generic": md6[0],
        "md6_p_probing": md6[1],
        "md6_p_focus": md6[2],
        "md6_p_telling": md6[3],
        "mastery_before": current_mastery,
        "prev_evaluator_rate": historical[0],
        "prev_mastery_delta": historical[1],
        "prev_mistake_identification": historical[2],
        "prev_mistake_location": historical[3],
        "prev_providing_guidance": historical[4],
        "prev_actionability": historical[5],
        "has_previous_feedback": historical[6],
    }

    if set(feature_values) != set(FEATURE_NAMES):
        raise RuntimeError(
            "Internal SI7-A feature names do not match the canonical contract."
        )

    context = np.asarray(
        [feature_values[name] for name in FEATURE_NAMES],
        dtype=np.float64,
    )

    if context.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            "Internal SI7-A context assembly violated the feature contract."
        )
    if context.dtype != np.float64:
        raise RuntimeError("SI7-A context must have dtype float64.")
    if not np.isfinite(context).all():
        raise RuntimeError("SI7-A context contains a non-finite value.")

    return context


__all__ = (
    "FEATURE_NAMES",
    "MOVE_ORDER",
    "MRB1_LABELS",
    "PreviousAttemptFeedback",
    "build_lints_context",
    "mrb1_expected_score",
    "validate_md6_probabilities",
)
