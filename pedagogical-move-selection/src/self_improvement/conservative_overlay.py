"""Canonical conservative MD6 overlay mechanics for the current SI7 design.

Frozen MD6 remains the base pedagogical-move selector. Linear Thompson
Sampling supplies only an attempt-level bias arm, and this module
conservatively reranks the frozen MD6 output. A bias may act only when its
target move is sufficiently close to the MD6 argmax.

The default probability-gap threshold of 0.10 is the current candidate
configuration. It is not scientifically established as optimal; later
controlled sensitivity analysis will determine whether it should remain
0.10. This module validates overlay mechanics and makes no claim about
educational benefit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Final

from .context_builder import MOVE_ORDER, validate_md6_probabilities
from .lints_policy import ARMS


DEFAULT_GAP_THRESHOLD: Final[float] = 0.10

BIAS_TARGET: Final[Mapping[str, str | None]] = MappingProxyType(
    {
        "baseline": None,
        "generic_bias": "generic",
        "probing_bias": "probing",
        "focus_bias": "focus",
        "telling_bias": "telling",
    }
)


@dataclass(frozen=True, slots=True)
class OverlayDecision:
    """Immutable diagnostics for one conservative overlay decision.

    For the baseline arm, ``target_move`` and ``target_probability`` are
    ``None`` and ``gap`` is consistently ``0.0``. For every other arm,
    probability fields and the gap are ordinary Python ``float`` values.
    """

    arm: str
    base_move: str
    final_move: str
    target_move: str | None
    overridden: bool
    gap: float
    gap_threshold: float
    base_probability: float
    target_probability: float | None


def _validate_arm(arm: object) -> str:
    """Return a canonical overlay arm or reject an unknown value."""

    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError(f"Unknown overlay arm: {arm!r}.")
    return arm


def _validate_gap_threshold(value: object) -> float:
    """Return a finite, non-boolean threshold in the interval [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "gap_threshold must be a real number in [0, 1], "
            f"got {value!r}."
        )

    try:
        threshold = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"gap_threshold could not be represented as a float: {value!r}."
        ) from exc

    if not isfinite(threshold):
        raise ValueError(f"gap_threshold must be finite, got {threshold!r}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"gap_threshold must be in [0, 1], got {threshold}."
        )
    return threshold


def _validated_probability_mapping(
    probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Return strict MD6 probabilities keyed in canonical move order.

    The public context-builder validator supplies the canonical exact-key,
    range, finiteness, and approximate-unit-sum contract. This overlay adds a
    semantic type guard before calling it: booleans and float-coercible strings
    are not MD6 probability values.
    """

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

    ordered = validate_md6_probabilities(probabilities)
    return {
        move: float(probability)
        for move, probability in zip(MOVE_ORDER, ordered, strict=True)
    }


def apply_conservative_overlay(
    arm: str,
    md6_probabilities: Mapping[str, float],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> OverlayDecision:
    """Apply one selected bias arm to already-computed frozen-MD6 output.

    The MD6 base move is the maximum probability in :data:`MOVE_ORDER`.
    Because ``max`` retains the first encountered maximum, exact ties are
    resolved deterministically as ``generic``, ``probing``, ``focus``, then
    ``telling``.

    ``baseline`` always returns that base move. A bias whose target is already
    the base also leaves it unchanged. Otherwise, the target overrides only
    when ``P(base) - P(target) <= gap_threshold``. The comparison is literally
    inclusive; invalid probabilities and thresholds are rejected rather than
    normalized or clipped.
    """

    validated_arm = _validate_arm(arm)
    threshold = _validate_gap_threshold(gap_threshold)
    probabilities = _validated_probability_mapping(md6_probabilities)

    base_move = max(MOVE_ORDER, key=probabilities.__getitem__)
    base_probability = float(probabilities[base_move])
    target_move = BIAS_TARGET[validated_arm]

    if target_move is None:
        return OverlayDecision(
            arm=validated_arm,
            base_move=base_move,
            final_move=base_move,
            target_move=None,
            overridden=False,
            gap=0.0,
            gap_threshold=threshold,
            base_probability=base_probability,
            target_probability=None,
        )

    target_probability = float(probabilities[target_move])
    if target_move == base_move:
        return OverlayDecision(
            arm=validated_arm,
            base_move=base_move,
            final_move=base_move,
            target_move=target_move,
            overridden=False,
            gap=0.0,
            gap_threshold=threshold,
            base_probability=base_probability,
            target_probability=target_probability,
        )

    gap = float(base_probability - target_probability)
    overridden = gap <= threshold
    final_move = target_move if overridden else base_move

    return OverlayDecision(
        arm=validated_arm,
        base_move=base_move,
        final_move=final_move,
        target_move=target_move,
        overridden=overridden,
        gap=gap,
        gap_threshold=threshold,
        base_probability=base_probability,
        target_probability=target_probability,
    )


def eligible_arms(
    md6_probabilities: Mapping[str, float],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> tuple[str, ...]:
    """Return canonical arms capable of changing the current MD6 decision.

    ``baseline`` is always eligible. A bias arm is eligible only when its
    target differs from the deterministic MD6 argmax and its literal
    probability gap is at most ``gap_threshold``. The returned tuple always
    preserves canonical :data:`ARMS` ordering. This helper applies the current
    candidate threshold mechanics; it does not select or scientifically tune
    the threshold.
    """

    threshold = _validate_gap_threshold(gap_threshold)
    probabilities = _validated_probability_mapping(md6_probabilities)
    base_move = max(MOVE_ORDER, key=probabilities.__getitem__)
    base_probability = float(probabilities[base_move])

    eligible = ["baseline"]
    for arm in ARMS[1:]:
        target_move = BIAS_TARGET[arm]
        if target_move is None:
            raise RuntimeError(f"Bias arm {arm!r} has no target move.")
        if target_move == base_move:
            continue
        gap = float(base_probability - probabilities[target_move])
        if gap <= threshold:
            eligible.append(arm)

    return tuple(eligible)


__all__ = (
    "ARMS",
    "BIAS_TARGET",
    "DEFAULT_GAP_THRESHOLD",
    "MOVE_ORDER",
    "OverlayDecision",
    "apply_conservative_overlay",
    "eligible_arms",
)
