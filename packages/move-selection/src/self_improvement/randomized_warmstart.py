"""Raw frozen-MD7 probability-proportional treatment assignment."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Final, Protocol

import numpy as np

from .context_builder import MOVE_ORDER


BEHAVIOR_POLICY_ID: Final[str] = "md7_r2_probability_proportional_v1"
WARMSTART_LINEAGE_SCHEMA: Final[str] = "adaptmath_randomized_warmstart_lineage_v1"


class RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class RandomizedAssignment:
    selected_move: str
    behavior_propensity: float
    full_probability_vector: Mapping[str, float]
    behavior_policy: str
    randomized_assignment: bool
    treatment_assignment_source: str
    assignment_rng_source: str
    assignment_seed: int | None
    assignment_random_draw: float


class MD7ProportionalAssigner:
    """Sample all four direct moves without thresholds or eligibility rules."""

    def __init__(
        self,
        *,
        rng: RandomSource | None = None,
        seed: int | None = None,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Supply either rng or seed, not both.")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("Injected assignment seed must be nonnegative.")
            self._rng: RandomSource = np.random.default_rng(int(seed))
            self.rng_source = "injected_deterministic_numpy_generator"
            self.seed: int | None = int(seed)
        elif rng is not None:
            if not callable(getattr(rng, "random", None)):
                raise TypeError("Injected assignment RNG must provide random().")
            self._rng = rng
            self.rng_source = "injected_random_source"
            self.seed = None
        else:
            # SystemRandom draws from operating-system entropy and has no reusable
            # global seed. The exact draw is logged with each assignment.
            self._rng = secrets.SystemRandom()
            self.rng_source = "system_entropy_secrets_systemrandom"
            self.seed = None

    @staticmethod
    def _probabilities(values: Mapping[str, object]) -> dict[str, float]:
        if not isinstance(values, Mapping) or set(values) != set(MOVE_ORDER):
            raise ValueError("MD7 assignment needs exactly the four direct moves.")
        vector: list[float] = []
        for move in MOVE_ORDER:
            value = values[move]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("MD7 assignment probabilities must be real numbers.")
            probability = float(value)
            if not isfinite(probability) or probability < 0.0:
                raise ValueError("MD7 assignment probabilities must be finite and nonnegative.")
            vector.append(probability)
        total = sum(vector)
        if not isfinite(total) or abs(total - 1.0) > 1e-5:
            raise ValueError("MD7 assignment probabilities must sum to one.")
        if total <= 0.0:
            raise ValueError("MD7 assignment distribution cannot be empty.")
        # Floating-point normalization only. No probability floor, threshold,
        # arm removal, or subset renormalization is applied.
        return {
            move: vector[index] / total for index, move in enumerate(MOVE_ORDER)
        }

    def assign(self, probabilities: Mapping[str, object]) -> RandomizedAssignment:
        normalized = self._probabilities(probabilities)
        draw = float(self._rng.random())
        if not isfinite(draw) or not 0.0 <= draw < 1.0:
            raise ValueError("Assignment RNG must return a finite value in [0, 1).")
        cumulative = 0.0
        selected = MOVE_ORDER[-1]
        for move in MOVE_ORDER:
            cumulative += normalized[move]
            if draw < cumulative:
                selected = move
                break
        return RandomizedAssignment(
            selected_move=selected,
            behavior_propensity=normalized[selected],
            full_probability_vector=normalized,
            behavior_policy=BEHAVIOR_POLICY_ID,
            randomized_assignment=True,
            treatment_assignment_source="raw_md7_probability_sample",
            assignment_rng_source=self.rng_source,
            assignment_seed=self.seed,
            assignment_random_draw=draw,
        )


__all__ = (
    "BEHAVIOR_POLICY_ID",
    "MD7ProportionalAssigner",
    "RandomizedAssignment",
    "WARMSTART_LINEAGE_SCHEMA",
)
