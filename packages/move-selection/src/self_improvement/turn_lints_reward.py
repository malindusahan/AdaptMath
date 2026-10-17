"""Turn-level reward calculators for the direct-arm Turn-LinTS lineage."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Final


REWARD_MODES: Final[tuple[str, ...]] = ("raw_delta", "headroom_normalized")


def _mastery(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number.")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return result


def turn_rewards(mastery_before: object, mastery_after: object) -> dict[str, float]:
    """Return raw and symmetric available-headroom normalized mastery deltas."""

    before = _mastery("mastery_before", mastery_before)
    after = _mastery("mastery_after", mastery_after)
    delta = float(after - before)
    if delta > 0.0:
        denominator = 1.0 - before
        headroom = delta / denominator if denominator > 0.0 else 0.0
    elif delta < 0.0:
        headroom = delta / before if before > 0.0 else 0.0
    else:
        headroom = 0.0
    if not -1.0 - 1e-12 <= headroom <= 1.0 + 1e-12:
        raise RuntimeError("Headroom-normalized reward escaped [-1, 1].")
    headroom = max(-1.0, min(1.0, float(headroom)))
    return {"raw_delta": delta, "headroom_normalized": headroom}


def configured_turn_reward(
    mastery_before: object,
    mastery_after: object,
    reward_mode: str,
) -> float:
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
    return turn_rewards(mastery_before, mastery_after)[reward_mode]


__all__ = ("REWARD_MODES", "configured_turn_reward", "turn_rewards")
