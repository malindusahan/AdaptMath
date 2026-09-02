"""Immutable adaptive-attempt boundary models.

The adaptive reward unit is one completed tutoring attempt/session.  These
models deliberately contain no resolved-event identifiers: individual BKT
observations contribute to the session result but never receive that reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import Any


FLOAT_TOLERANCE = 1e-9


def _normalise_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalise_number(
    value: Real,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if not minimum <= numeric <= maximum:
        raise ValueError(
            f"{field_name} must be in [{minimum}, {maximum}]"
        )
    return numeric


@dataclass(frozen=True, slots=True)
class AdaptiveAttemptContext:
    """Authoritative mastery snapshot captured before tutoring starts."""

    student_id: str
    attempt_id: str
    skill: str
    mastery_before: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "student_id",
            _normalise_identifier(self.student_id, "student_id"),
        )
        object.__setattr__(
            self,
            "attempt_id",
            _normalise_identifier(self.attempt_id, "attempt_id"),
        )
        object.__setattr__(
            self,
            "skill",
            _normalise_identifier(self.skill, "skill"),
        )
        object.__setattr__(
            self,
            "mastery_before",
            _normalise_number(
                self.mastery_before,
                "mastery_before",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class AttemptLearningOutcome:
    """One BKT-derived learning outcome for one completed attempt/session."""

    attempt_id: str
    skill: str
    mastery_before: float
    mastery_after: float
    delta_mastery: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_id",
            _normalise_identifier(self.attempt_id, "attempt_id"),
        )
        object.__setattr__(
            self,
            "skill",
            _normalise_identifier(self.skill, "skill"),
        )

        before = _normalise_number(
            self.mastery_before,
            "mastery_before",
            minimum=0.0,
            maximum=1.0,
        )
        after = _normalise_number(
            self.mastery_after,
            "mastery_after",
            minimum=0.0,
            maximum=1.0,
        )
        delta = _normalise_number(
            self.delta_mastery,
            "delta_mastery",
            minimum=-1.0,
            maximum=1.0,
        )

        if not isclose(
            delta,
            after - before,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "delta_mastery must equal mastery_after - mastery_before"
            )

        object.__setattr__(self, "mastery_before", before)
        object.__setattr__(self, "mastery_after", after)
        object.__setattr__(self, "delta_mastery", delta)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible pipeline boundary representation."""
        return {
            "attempt_id": self.attempt_id,
            "skill": self.skill,
            "mastery_before": self.mastery_before,
            "mastery_after": self.mastery_after,
            "delta_mastery": self.delta_mastery,
        }
