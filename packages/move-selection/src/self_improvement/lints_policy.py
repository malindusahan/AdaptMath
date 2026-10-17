"""Canonical v3 true disjoint Linear Thompson Sampling mechanics.

Each canonical overlay arm owns an independent linear posterior, and every arm
decision is made from a stochastic posterior parameter sample. This is true
Linear Thompson Sampling, not LinUCB or an optimistic confidence-bound rule.

The mathematical policy keeps ``context_dim`` configurable for reusable
mechanics. The final turn-level candidate uses ``ridge_lambda=1.0`` and
``exploration_scale=0.20``.

Simulation-learned posterior state must never seed real learning. A real pilot
must begin with fresh ``ridge_lambda * I`` precision matrices, zero reward
vectors, and ``data_mode="real"``. This module validates mechanics only and
makes no claim about educational efficacy.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Final, TypedDict

import numpy as np

from .reward import validate_signed_reward


ARMS: Final[tuple[str, ...]] = (
    "baseline",
    "generic_bias",
    "probing_bias",
    "focus_bias",
    "telling_bias",
)

DEFAULT_RIDGE_LAMBDA: Final[float] = 1.0
DEFAULT_EXPLORATION_SCALE: Final[float] = 0.20

_DATA_MODES: Final[frozenset[str]] = frozenset({"synthetic", "real"})
LINTS_STATE_SCHEMA_VERSION: Final[str] = "true_disjoint_lints_v3"
_STATE_SCHEMA: Final[str] = LINTS_STATE_SCHEMA_VERSION


class LinTSDecision(TypedDict):
    """Observable diagnostics from one posterior-sampling decision."""

    selected_arm: str
    sampled_scores: dict[str, float]
    sampled_thetas: dict[str, list[float]]


def _positive_integer(name: str, value: object) -> int:
    """Return a strictly positive, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero, got {result}.")
    return result


def _nonnegative_integer(name: str, value: object) -> int:
    """Return a nonnegative, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}.")
    return result


def _positive_float(name: str, value: object) -> float:
    """Return a finite, strictly positive real number as ``float``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number, got {value!r}.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} could not be represented as a float.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}.")
    return result


def _validate_data_mode(value: object, name: str = "data_mode") -> str:
    """Return an exact supported data mode."""

    if not isinstance(value, str) or value not in _DATA_MODES:
        raise ValueError(f"{name} must be exactly 'synthetic' or 'real'.")
    return value


def _real_float64_array(name: str, value: object) -> np.ndarray:
    """Convert a real numeric array-like value to a finite float64 copy."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array-like value.") from exc

    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain only real numeric values.")

    try:
        result = np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} could not be represented as float64.") from exc

    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _json_compatible(value: object) -> object:
    """Recursively convert NumPy values into JSON-compatible Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


class TrueDisjointLinTS:
    """True disjoint Linear Thompson Sampling over canonical overlay arms.

    The class owns one ``A`` precision matrix and one ``b`` reward vector per
    arm. It does not compute rewards, apply overlays, manage tutoring attempts,
    aggregate MRB1, or perform disk persistence.
    """

    def __init__(
        self,
        context_dim: int,
        ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
        exploration_scale: float = DEFAULT_EXPLORATION_SCALE,
        seed: int = 42,
        data_mode: str = "synthetic",
    ) -> None:
        self.context_dim = _positive_integer("context_dim", context_dim)
        self.ridge_lambda = _positive_float("ridge_lambda", ridge_lambda)
        self.exploration_scale = _positive_float(
            "exploration_scale",
            exploration_scale,
        )
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError(f"seed must be a nonnegative integer, got {seed!r}.")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError(f"seed must be nonnegative, got {self.seed}.")

        self.data_mode = _validate_data_mode(data_mode)
        self.arms = ARMS
        self.rng = np.random.default_rng(self.seed)

        self.A = {
            arm: self.ridge_lambda
            * np.eye(self.context_dim, dtype=np.float64)
            for arm in self.arms
        }
        self.b = {
            arm: np.zeros(self.context_dim, dtype=np.float64)
            for arm in self.arms
        }
        self.arm_update_counts = {arm: 0 for arm in self.arms}
        self.total_updates = 0

    def _validate_arm(self, arm: object) -> str:
        if not isinstance(arm, str) or arm not in self.arms:
            raise ValueError(f"Unknown LinTS arm: {arm!r}.")
        return arm

    def _validate_context(self, context: object) -> np.ndarray:
        result = _real_float64_array("context", context)
        if result.ndim != 1:
            raise ValueError(
                f"context must be one-dimensional, got {result.ndim} dimensions."
            )
        if result.shape != (self.context_dim,):
            raise ValueError(
                f"context must have shape {(self.context_dim,)}, "
                f"got {result.shape}."
            )
        return result

    @staticmethod
    def _validate_reward(reward: object) -> float:
        return validate_signed_reward(reward)

    @staticmethod
    def _validate_sample_weight(sample_weight: object) -> float:
        """Return a finite, non-boolean update weight in ``(0, 1]``."""

        if isinstance(sample_weight, bool) or not isinstance(sample_weight, Real):
            raise TypeError(
                "sample_weight must be a non-boolean real number, "
                f"got {sample_weight!r}."
            )
        try:
            result = float(sample_weight)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "sample_weight could not be represented as a float."
            ) from exc
        if not isfinite(result):
            raise ValueError(f"sample_weight must be finite, got {result!r}.")
        if not 0.0 < result <= 1.0:
            raise ValueError(f"sample_weight must be in (0, 1], got {result}.")
        return result

    def _validate_eligible_arms(
        self,
        eligible_arms: Iterable[str],
    ) -> tuple[str, ...]:
        """Validate an arm subset and return it in canonical arm order."""

        if isinstance(eligible_arms, (str, bytes)):
            raise TypeError("eligible_arms must be a non-empty iterable of arms.")
        try:
            supplied = tuple(eligible_arms)
        except TypeError as exc:
            raise TypeError(
                "eligible_arms must be a non-empty iterable of arms."
            ) from exc
        if not supplied:
            raise ValueError("eligible_arms cannot be empty.")

        validated = tuple(self._validate_arm(arm) for arm in supplied)
        if len(set(validated)) != len(validated):
            raise ValueError("eligible_arms cannot contain duplicates.")

        requested = set(validated)
        return tuple(arm for arm in self.arms if arm in requested)

    def posterior_for_arm(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and covariance for one arm."""

        validated_arm = self._validate_arm(arm)
        precision = self.A[validated_arm]
        reward_vector = self.b[validated_arm]

        posterior_mean = np.linalg.solve(precision, reward_vector)
        precision_inverse = np.linalg.solve(
            precision,
            np.eye(self.context_dim, dtype=np.float64),
        )
        posterior_covariance = (
            self.exploration_scale**2 * precision_inverse
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.T
        )
        return posterior_mean, posterior_covariance

    def sample_arm_scores(
        self,
        context: object,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Draw one posterior parameter per arm and score the context."""

        x = self._validate_context(context)
        sampled_scores: dict[str, float] = {}
        sampled_thetas: dict[str, np.ndarray] = {}

        for arm in self.arms:
            posterior_mean, posterior_covariance = self.posterior_for_arm(arm)
            sampled_theta = self.rng.multivariate_normal(
                mean=posterior_mean,
                cov=posterior_covariance,
            )
            sampled_thetas[arm] = sampled_theta
            sampled_scores[arm] = float(x @ sampled_theta)

        return sampled_scores, sampled_thetas

    def select_arm(
        self,
        context: object,
        eligible_arms: Iterable[str] | None = None,
    ) -> LinTSDecision:
        """Select the largest sampled score among eligible canonical arms.

        With ``eligible_arms=None``, all canonical arms participate exactly as
        in the original interface. A supplied subset is validated and ordered
        canonically before the sampled-score argmax is restricted to it.
        Posterior sampling remains ordinary true Thompson sampling and never
        updates policy state.
        """

        candidates = (
            self.arms
            if eligible_arms is None
            else self._validate_eligible_arms(eligible_arms)
        )
        sampled_scores, sampled_thetas = self.sample_arm_scores(context)
        selected_arm = max(
            candidates,
            key=lambda arm: sampled_scores[arm],
        )
        return {
            "selected_arm": selected_arm,
            "sampled_scores": sampled_scores,
            "sampled_thetas": {
                arm: sampled_thetas[arm].tolist() for arm in self.arms
            },
        }

    def update(
        self,
        arm: str,
        context: object,
        reward: object,
        sample_weight: object = 1.0,
    ) -> None:
        """Apply one weighted observation to the explicitly selected arm.

        ``sample_weight=1.0`` preserves the original update exactly. Each call,
        including a fractional delayed-credit contribution, increments the
        existing arm and total update counters once; those counters continue
        to represent low-level update calls rather than completed attempts.
        """

        validated_arm = self._validate_arm(arm)
        x = self._validate_context(context)
        validated_reward = self._validate_reward(reward)
        validated_weight = self._validate_sample_weight(sample_weight)

        try:
            with np.errstate(over="raise", invalid="raise"):
                candidate_A = (
                    self.A[validated_arm]
                    + validated_weight * np.outer(x, x)
                )
                candidate_b = (
                    self.b[validated_arm]
                    + validated_weight * validated_reward * x
                )
        except FloatingPointError as exc:
            raise ValueError("LinTS update produced numeric overflow.") from exc

        if not np.isfinite(candidate_A).all() or not np.isfinite(candidate_b).all():
            raise ValueError("LinTS update produced non-finite posterior state.")

        self.A[validated_arm] = candidate_A
        self.b[validated_arm] = candidate_b
        self.arm_update_counts[validated_arm] += 1
        self.total_updates += 1

    def state_dict(self) -> dict[str, object]:
        """Return complete JSON-compatible posterior and RNG state."""

        state = {
            "schema_version": _STATE_SCHEMA,
            "arms": list(self.arms),
            "context_dim": self.context_dim,
            "ridge_lambda": self.ridge_lambda,
            "exploration_scale": self.exploration_scale,
            "data_mode": self.data_mode,
            "A": {arm: self.A[arm].tolist() for arm in self.arms},
            "b": {arm: self.b[arm].tolist() for arm in self.arms},
            "arm_update_counts": {
                arm: int(self.arm_update_counts[arm]) for arm in self.arms
            },
            "total_updates": int(self.total_updates),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }
        compatible = _json_compatible(state)
        if not isinstance(compatible, dict):
            raise RuntimeError("Internal LinTS state serialization failed.")
        return compatible

    @staticmethod
    def _ordered_arm_mapping(
        state: Mapping[str, object],
        field_name: str,
    ) -> Mapping[str, object]:
        value = state[field_name]
        if not isinstance(value, Mapping):
            raise TypeError(f"state.{field_name} must be a mapping.")
        if tuple(value.keys()) != ARMS:
            raise ValueError(
                f"state.{field_name} keys must preserve canonical arm order."
            )
        return value

    def load_state_dict(
        self,
        state: Mapping[str, object],
        expected_data_mode: str | None = None,
    ) -> None:
        """Validate a complete state and atomically replace policy state.

        Simulation-learned matrices cannot cross into a real policy, nor can
        real posterior state silently enter a synthetic policy. All validation
        is completed using temporary objects before this instance is mutated.
        """

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")

        required_fields = frozenset(
            {
                "schema_version",
                "arms",
                "context_dim",
                "ridge_lambda",
                "exploration_scale",
                "data_mode",
                "A",
                "b",
                "arm_update_counts",
                "total_updates",
                "rng_state",
            }
        )
        missing = required_fields - set(state)
        if missing:
            raise ValueError(f"LinTS state is missing fields: {sorted(missing)}.")

        if state["schema_version"] != _STATE_SCHEMA:
            raise ValueError("Unsupported LinTS state schema.")

        state_arms = state["arms"]
        if (
            not isinstance(state_arms, Sequence)
            or isinstance(state_arms, (str, bytes))
            or tuple(state_arms) != ARMS
        ):
            raise ValueError("LinTS state arms do not match canonical arm order.")

        loaded_context_dim = _positive_integer(
            "state.context_dim",
            state["context_dim"],
        )
        if loaded_context_dim != self.context_dim:
            raise ValueError(
                f"LinTS context_dim mismatch: state={loaded_context_dim}, "
                f"policy={self.context_dim}."
            )

        loaded_ridge = _positive_float(
            "state.ridge_lambda",
            state["ridge_lambda"],
        )
        if loaded_ridge != self.ridge_lambda:
            raise ValueError(
                f"LinTS ridge_lambda mismatch: state={loaded_ridge}, "
                f"policy={self.ridge_lambda}."
            )

        loaded_exploration = _positive_float(
            "state.exploration_scale",
            state["exploration_scale"],
        )
        if loaded_exploration != self.exploration_scale:
            raise ValueError(
                "LinTS exploration_scale mismatch: "
                f"state={loaded_exploration}, policy={self.exploration_scale}."
            )

        loaded_data_mode = _validate_data_mode(
            state["data_mode"],
            "state.data_mode",
        )
        if expected_data_mode is not None:
            validated_expected_mode = _validate_data_mode(
                expected_data_mode,
                "expected_data_mode",
            )
            if validated_expected_mode != self.data_mode:
                raise ValueError(
                    "expected_data_mode does not match the receiving policy."
                )
        else:
            validated_expected_mode = self.data_mode

        if loaded_data_mode != self.data_mode or (
            loaded_data_mode != validated_expected_mode
        ):
            raise ValueError(
                "Synthetic/real LinTS state barrier triggered: "
                f"state={loaded_data_mode}, policy={self.data_mode}, "
                f"expected={validated_expected_mode}."
            )

        state_A = self._ordered_arm_mapping(state, "A")
        state_b = self._ordered_arm_mapping(state, "b")
        state_counts = self._ordered_arm_mapping(state, "arm_update_counts")

        candidate_A: dict[str, np.ndarray] = {}
        candidate_b: dict[str, np.ndarray] = {}
        candidate_counts: dict[str, int] = {}

        for arm in self.arms:
            matrix = _real_float64_array(f"state.A[{arm!r}]", state_A[arm])
            if matrix.shape != (self.context_dim, self.context_dim):
                raise ValueError(
                    f"state.A[{arm!r}] must have shape "
                    f"{(self.context_dim, self.context_dim)}, got {matrix.shape}."
                )
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
                raise ValueError(f"state.A[{arm!r}] must be symmetric.")
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"state.A[{arm!r}] must be positive definite."
                ) from exc

            vector = _real_float64_array(f"state.b[{arm!r}]", state_b[arm])
            if vector.shape != (self.context_dim,):
                raise ValueError(
                    f"state.b[{arm!r}] must have shape {(self.context_dim,)}, "
                    f"got {vector.shape}."
                )

            count = _nonnegative_integer(
                f"state.arm_update_counts[{arm!r}]",
                state_counts[arm],
            )
            candidate_A[arm] = matrix
            candidate_b[arm] = vector
            candidate_counts[arm] = count

        candidate_total_updates = _nonnegative_integer(
            "state.total_updates",
            state["total_updates"],
        )
        if candidate_total_updates != sum(candidate_counts.values()):
            raise ValueError(
                "state.total_updates must equal the sum of arm update counts."
            )

        candidate_rng_state = copy.deepcopy(state["rng_state"])
        try:
            json.dumps(candidate_rng_state)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("state.rng_state must be JSON-compatible.") from exc

        candidate_rng = np.random.default_rng(0)
        try:
            candidate_rng.bit_generator.state = candidate_rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("state.rng_state is invalid.") from exc

        # Atomic commit: every validation above succeeded before mutation.
        self.A = candidate_A
        self.b = candidate_b
        self.arm_update_counts = candidate_counts
        self.total_updates = candidate_total_updates
        self.rng = candidate_rng


class LinTSPolicy:
    """Deprecated adapter for older in-repository true-LinTS experiments.

    New code must use :class:`TrueDisjointLinTS`. This adapter preserves the
    earlier constructor and string-returning ``select_arm`` interface for old
    scripts; it does not expose persistence or define the canonical state.
    """

    def __init__(
        self,
        n_features: int,
        arms: Sequence[str],
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        supplied_arms = tuple(arms)
        if len(supplied_arms) != len(ARMS) or set(supplied_arms) != set(ARMS):
            raise ValueError(
                "Legacy LinTSPolicy supports only the five canonical arm names."
            )
        self.n_features = _positive_integer("n_features", n_features)
        self.arms = list(supplied_arms)
        self.alpha = _positive_float("alpha", alpha)
        legacy_seed = 42 if seed is None else seed
        self._policy = TrueDisjointLinTS(
            context_dim=self.n_features,
            ridge_lambda=1.0,
            exploration_scale=self.alpha,
            seed=legacy_seed,
            data_mode="synthetic",
        )

    @property
    def A(self) -> dict[str, np.ndarray]:
        return self._policy.A

    @property
    def b(self) -> dict[str, np.ndarray]:
        return self._policy.b

    @property
    def rng(self) -> np.random.Generator:
        return self._policy.rng

    def _validate_context(self, context: object) -> np.ndarray:
        return self._policy._validate_context(context)

    def arm_posterior(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        return self._policy.posterior_for_arm(arm)

    def sample_arm_scores(self, context: object) -> dict[str, float]:
        scores, _ = self._policy.sample_arm_scores(context)
        return scores

    def select_arm(self, context: object) -> str:
        return self._policy.select_arm(context)["selected_arm"]

    def update(self, arm: str, context: object, reward: object) -> None:
        self._policy.update(arm, context, reward)


__all__ = (
    "ARMS",
    "DEFAULT_EXPLORATION_SCALE",
    "DEFAULT_RIDGE_LAMBDA",
    "LINTS_STATE_SCHEMA_VERSION",
    "LinTSDecision",
    "LinTSPolicy",
    "TrueDisjointLinTS",
)
