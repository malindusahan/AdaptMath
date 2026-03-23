"""Final-candidate context for turn-level delayed-feedback LinTS.

The nine-dimensional C3 context combines current-turn frozen-MD6
probabilities with running MRB1 quality from completed earlier tutor responses
in the same attempt. Mastery remains learner-state metadata and never enters
this numerical policy context. Current-turn MRB1 scores, terminal evaluator
reward, ``mastery_after``, and future information are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Final

import numpy as np

from .context_builder import MOVE_ORDER, validate_md6_probabilities


MRB1_TASKS: Final[tuple[str, ...]] = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)

TURN_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "running_mistake_identification",
    "running_mistake_location",
    "running_providing_guidance",
    "running_actionability",
    "has_within_attempt_quality",
)


def _finite_unit_interval(name: str, value: object) -> float:
    """Return a finite, non-boolean real value in ``[0, 1]``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a non-boolean real number, got {value!r}."
        )
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} could not be represented as a float.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}.")
    return result


def validate_mastery_probability(value: object) -> float:
    """Return a strict attempt-start or attempt-end mastery probability."""

    return _finite_unit_interval("mastery probability", value)


def validate_mrb1_scores(
    scores: Mapping[str, float],
) -> dict[str, float]:
    """Validate one completed tutor response's four MRB1 expected scores."""

    if not isinstance(scores, Mapping):
        raise TypeError("MRB1 scores must be a mapping.")

    expected = set(MRB1_TASKS)
    actual = set(scores)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(
            "MRB1 scores must contain exactly "
            f"{list(MRB1_TASKS)}; missing={missing}, extra={extra}."
        )

    return {
        task: _finite_unit_interval(f"MRB1 score {task!r}", scores[task])
        for task in MRB1_TASKS
    }


def _strict_md6_probabilities(
    probabilities: Mapping[str, float],
) -> np.ndarray:
    """Return canonical MD6 probabilities with strict semantic typing."""

    if not isinstance(probabilities, Mapping):
        raise TypeError("md6_probabilities must be a mapping.")

    expected = set(MOVE_ORDER)
    actual = set(probabilities)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(
            "md6_probabilities must contain exactly "
            f"{list(MOVE_ORDER)}; missing={missing}, extra={extra}."
        )

    for move in MOVE_ORDER:
        value = probabilities[move]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"MD6 probability {move!r} must be a non-boolean real "
                f"number, got {value!r}."
            )

    return validate_md6_probabilities(probabilities)


@dataclass(frozen=True, slots=True)
class TutorQualitySnapshot:
    """Immutable running-quality state after zero or more scored responses."""

    count: int
    mistake_identification: float
    mistake_location: float
    providing_guidance: float
    actionability: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, Integral):
            raise TypeError("quality count must be a non-boolean integer.")
        count = int(self.count)
        if count < 0:
            raise ValueError("quality count cannot be negative.")
        object.__setattr__(self, "count", count)

        for field_name in (
            "mistake_identification",
            "mistake_location",
            "providing_guidance",
            "actionability",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_unit_interval(field_name, getattr(self, field_name)),
            )

        if count == 0 and any(value != 0.0 for value in self.feature_values):
            raise ValueError("An empty quality snapshot must have zero means.")

    @property
    def feature_values(self) -> tuple[float, float, float, float]:
        """Return running means in turn-context feature order."""

        return (
            self.mistake_identification,
            self.mistake_location,
            self.providing_guidance,
            self.actionability,
        )

    @property
    def has_within_attempt_quality(self) -> float:
        return float(self.count > 0)

    def as_task_mapping(self) -> dict[str, float]:
        """Return an independent task-keyed snapshot for reporting."""

        return dict(zip(MRB1_TASKS, self.feature_values, strict=True))


class RunningTutorQuality:
    """Accumulate running means from completed earlier tutor responses.

    The object starts empty. ``add_scores`` validates and incorporates exactly
    one completed response atomically. Its current immutable snapshot is safe
    to pass to the next turn's context builder; scores for a response currently
    being selected or generated cannot enter through this API.
    """

    __slots__ = ("_count", "_sums")

    def __init__(self) -> None:
        self._count = 0
        self._sums = {task: 0.0 for task in MRB1_TASKS}

    @property
    def count(self) -> int:
        return self._count

    def add_scores(
        self,
        scores: Mapping[str, float],
    ) -> TutorQualitySnapshot:
        """Add one validated completed-response observation."""

        validated = validate_mrb1_scores(scores)
        candidate_sums = {
            task: self._sums[task] + validated[task] for task in MRB1_TASKS
        }
        if not all(isfinite(value) for value in candidate_sums.values()):
            raise ValueError("Running MRB1 quality sums became non-finite.")

        self._sums = candidate_sums
        self._count += 1
        return self.snapshot()

    def snapshot(self) -> TutorQualitySnapshot:
        """Return the current immutable count and arithmetic means."""

        if self._count == 0:
            means = (0.0, 0.0, 0.0, 0.0)
        else:
            means = tuple(
                float(self._sums[task] / self._count) for task in MRB1_TASKS
            )

        return TutorQualitySnapshot(
            count=self._count,
            mistake_identification=means[0],
            mistake_location=means[1],
            providing_guidance=means[2],
            actionability=means[3],
        )


def build_turn_context(
    md6_probabilities: Mapping[str, float],
    running_quality: RunningTutorQuality | TutorQualitySnapshot,
) -> np.ndarray:
    """Build the causally valid nine-dimensional C3 current-turn context.

    ``running_quality`` must contain only MRB1 scores from tutor responses that
    completed before this turn's selection. The API has no current evaluator,
    mastery value, current-turn MRB1, response text, or future inputs.
    """

    md6 = _strict_md6_probabilities(md6_probabilities)

    if isinstance(running_quality, RunningTutorQuality):
        quality = running_quality.snapshot()
    elif isinstance(running_quality, TutorQualitySnapshot):
        quality = running_quality
    else:
        raise TypeError(
            "running_quality must be RunningTutorQuality or "
            "TutorQualitySnapshot."
        )

    context = np.asarray(
        [
            *md6,
            *quality.feature_values,
            quality.has_within_attempt_quality,
        ],
        dtype=np.float64,
    )

    expected_shape = (len(TURN_FEATURE_NAMES),)
    if context.shape != expected_shape:
        raise RuntimeError(
            f"Internal turn context must have shape {expected_shape}, "
            f"got {context.shape}."
        )
    if context.dtype != np.float64:
        raise RuntimeError("Turn context must have dtype float64.")
    if not np.isfinite(context).all():
        raise RuntimeError("Turn context contains a non-finite value.")
    return context


__all__ = (
    "MRB1_TASKS",
    "MOVE_ORDER",
    "RunningTutorQuality",
    "TURN_FEATURE_NAMES",
    "TutorQualitySnapshot",
    "build_turn_context",
    "validate_mastery_probability",
    "validate_mrb1_scores",
)
