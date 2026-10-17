"""Reward contracts for historical experiments and the active v3 runtime.

``primary_reward`` preserves the archived success-only evaluator mapping used
by earlier experiments.  The active turn-level v3 policy instead consumes an
already-produced, validated learning outcome and uses its raw signed mastery
change.  MRB1 and optional diagnostic signals never enter either computation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isclose, isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Final


REWARD_NAME: Final[str] = "mastery_delta"
LEARNING_OUTCOME_FIELDS: Final[tuple[str, ...]] = (
    "skill",
    "mastery_before",
    "mastery_after",
    "delta_mastery",
)
OPTIONAL_DIAGNOSTIC_FIELDS: Final[tuple[str, ...]] = (
    "evaluator_result",
    "reasoning",
    "uncertainty",
    "clarification",
    "repeated_misunderstanding",
)


def _finite_real(name: str, value: object) -> float:
    """Return a finite, non-boolean real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-boolean real number.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} could not be represented as a float.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _bounded_real(name: str, value: object, lower: float, upper: float) -> float:
    result = _finite_real(name, value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {result}.")
    return result


def validate_signed_reward(value: object) -> float:
    """Return a finite non-boolean v3 reward in ``[-1, 1]``."""

    return _bounded_real("reward", value, -1.0, 1.0)


def _json_compatible(value: object, path: str) -> object:
    """Copy optional diagnostics into finite JSON-native values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _finite_real(path, value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings.")
            result[key] = _json_compatible(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_compatible(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} is not JSON-compatible: {type(value).__name__}.")


def validate_optional_diagnostics(
    diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate already-supplied Malindu diagnostics without interpreting them."""

    if diagnostics is None:
        return {}
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping or None.")
    unknown = set(diagnostics) - set(OPTIONAL_DIAGNOSTIC_FIELDS)
    if unknown:
        raise ValueError(
            "diagnostics contains unsupported fields: "
            f"{sorted(unknown, key=repr)}."
        )
    return {
        name: _json_compatible(diagnostics[name], f"diagnostics.{name}")
        for name in OPTIONAL_DIAGNOSTIC_FIELDS
        if name in diagnostics
    }


@dataclass(frozen=True, slots=True)
class LearningOutcome:
    """Validated v3 learning outcome supplied by the host integration."""

    skill: str
    mastery_before: float
    mastery_after: float
    delta_mastery: float
    diagnostics: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.skill, str):
            raise TypeError("skill must be a non-empty string.")
        if not self.skill.strip():
            raise ValueError("skill must be a non-empty string.")

        before = _bounded_real("mastery_before", self.mastery_before, 0.0, 1.0)
        after = _bounded_real("mastery_after", self.mastery_after, 0.0, 1.0)
        delta = _bounded_real("delta_mastery", self.delta_mastery, -1.0, 1.0)
        expected_delta = float(after - before)
        if not isclose(delta, expected_delta, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "delta_mastery must equal mastery_after - mastery_before "
                "within abs_tol=1e-9."
            )

        validated_diagnostics = validate_optional_diagnostics(self.diagnostics)
        object.__setattr__(self, "mastery_before", before)
        object.__setattr__(self, "mastery_after", after)
        object.__setattr__(self, "delta_mastery", delta)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(validated_diagnostics),
        )

    def as_mapping(self) -> dict[str, object]:
        """Return the four required scientific fields as an independent mapping."""

        return {
            "skill": self.skill,
            "mastery_before": self.mastery_before,
            "mastery_after": self.mastery_after,
            "delta_mastery": self.delta_mastery,
        }

    def diagnostics_mapping(self) -> dict[str, object]:
        """Return only supplied optional diagnostics as a JSON-compatible copy."""

        return {
            name: _json_compatible(value, f"diagnostics.{name}")
            for name, value in self.diagnostics.items()
        }


def validate_learning_outcome(
    learning_outcome: LearningOutcome | Mapping[str, object],
    *,
    diagnostics: Mapping[str, object] | None = None,
) -> LearningOutcome:
    """Validate the required Malindu output contract for the v3 runtime.

    Optional diagnostic fields may be included directly in the input mapping,
    nested under ``diagnostics``, or supplied through the keyword argument.
    Only one diagnostic source is accepted so retry records cannot be merged or
    reinterpreted silently.
    """

    if isinstance(learning_outcome, LearningOutcome):
        if diagnostics is not None:
            raise ValueError(
                "diagnostics cannot be supplied with a validated LearningOutcome."
            )
        return LearningOutcome(
            **learning_outcome.as_mapping(),
            diagnostics=learning_outcome.diagnostics_mapping(),
        )
    if not isinstance(learning_outcome, Mapping):
        raise TypeError("learning_outcome must be a mapping or LearningOutcome.")

    required = set(LEARNING_OUTCOME_FIELDS)
    missing = required - set(learning_outcome)
    if missing:
        raise ValueError(
            f"learning_outcome is missing required fields: {sorted(missing)}."
        )

    optional = set(OPTIONAL_DIAGNOSTIC_FIELDS)
    allowed = required | optional | {"diagnostics"}
    extra = set(learning_outcome) - allowed
    if extra:
        raise ValueError(
            f"learning_outcome contains unsupported fields: {sorted(extra)}."
        )

    direct_diagnostics = {
        name: learning_outcome[name]
        for name in OPTIONAL_DIAGNOSTIC_FIELDS
        if name in learning_outcome
    }
    nested_diagnostics = learning_outcome.get("diagnostics")
    diagnostic_sources = sum(
        source is not None and source != {}
        for source in (direct_diagnostics, nested_diagnostics, diagnostics)
    )
    if diagnostic_sources > 1:
        raise ValueError("Optional diagnostics must be supplied through one source.")

    selected_diagnostics: Mapping[str, object] | None
    if direct_diagnostics:
        selected_diagnostics = direct_diagnostics
    elif nested_diagnostics is not None:
        if not isinstance(nested_diagnostics, Mapping):
            raise TypeError("learning_outcome.diagnostics must be a mapping.")
        selected_diagnostics = nested_diagnostics
    else:
        selected_diagnostics = diagnostics

    return LearningOutcome(
        skill=learning_outcome["skill"],  # type: ignore[arg-type]
        mastery_before=learning_outcome["mastery_before"],  # type: ignore[arg-type]
        mastery_after=learning_outcome["mastery_after"],  # type: ignore[arg-type]
        delta_mastery=learning_outcome["delta_mastery"],  # type: ignore[arg-type]
        diagnostics=validate_optional_diagnostics(selected_diagnostics),
    )


def mastery_delta_reward(
    learning_outcome: LearningOutcome | Mapping[str, object],
) -> float:
    """Return the validated raw signed ``delta_mastery`` without normalization."""

    return validate_learning_outcome(learning_outcome).delta_mastery


def primary_reward(evaluator_score: int) -> float:
    """Preserve the historical success-only evaluator reward mapping."""

    if type(evaluator_score) is not int:
        raise TypeError(
            "evaluator_score must be a non-boolean integer in {0, 1, 2, 3}; "
            f"got {evaluator_score!r}."
        )
    if evaluator_score not in {0, 1, 2, 3}:
        raise ValueError(
            "evaluator_score must be one of {0, 1, 2, 3}; "
            f"got {evaluator_score}."
        )
    return float(evaluator_score == 3)


__all__ = (
    "LEARNING_OUTCOME_FIELDS",
    "OPTIONAL_DIAGNOSTIC_FIELDS",
    "REWARD_NAME",
    "LearningOutcome",
    "mastery_delta_reward",
    "primary_reward",
    "validate_learning_outcome",
    "validate_optional_diagnostics",
    "validate_signed_reward",
)
