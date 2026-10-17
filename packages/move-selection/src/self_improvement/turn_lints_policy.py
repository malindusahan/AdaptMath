"""Fresh direct-move disjoint Linear Thompson Sampling policy and state I/O."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite, log
from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import numpy as np

from .context_builder import MOVE_ORDER
from .turn_lints_reward import REWARD_MODES


DIRECT_ARMS: Final[tuple[str, ...]] = MOVE_ORDER
ALGORITHM_VERSION: Final[str] = "direct_disjoint_turn_lints_v1"
POLICY_STATE_SCHEMA_VERSION: Final[str] = "adaptmath_turn_lints_state_v2"
LEGACY_POLICY_STATE_SCHEMA_VERSION: Final[str] = "adaptmath_turn_lints_state_v1"
DEFAULT_RIDGE_LAMBDA: Final[float] = 1.0
DEFAULT_EXPLORATION_SCALE: Final[float] = 0.20
ANCHOR_MODES: Final[tuple[str, ...]] = ("none", "md7_logprob_anchor")
ANCHOR_PROBABILITY_FLOOR: Final[float] = 1e-12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a nonnegative real number.")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


class DirectTurnLinTS:
    """One independent ``A``/``b`` posterior for each pedagogical move."""

    def __init__(
        self,
        *,
        context_schema_id: str,
        enabled_context_blocks: Sequence[str],
        feature_names: Sequence[str],
        selector_version: str,
        selector_sha256: str,
        reward_mode: str,
        anchor_mode: str = "none",
        anchor_gamma: float = 0.0,
        anchor_selector_version: str | None = None,
        anchor_selector_sha256: str | None = None,
        ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
        exploration_scale: float = DEFAULT_EXPLORATION_SCALE,
        seed: int = 42,
        data_mode: str = "real",
        created_at: str | None = None,
    ) -> None:
        self.context_schema_id = str(context_schema_id)
        self.enabled_context_blocks = tuple(str(v) for v in enabled_context_blocks)
        self.feature_names = tuple(str(v) for v in feature_names)
        if not self.context_schema_id or not self.enabled_context_blocks:
            raise ValueError("Context schema and enabled blocks are required.")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Ordered feature names must be non-empty and unique.")
        self.context_dim = len(self.feature_names)
        self.selector_version = str(selector_version)
        self.selector_sha256 = str(selector_sha256).lower()
        if not self.selector_version or len(self.selector_sha256) != 64:
            raise ValueError("Selector version and lowercase SHA-256 are required.")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
        self.reward_mode = reward_mode
        if anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}.")
        self.anchor_mode = anchor_mode
        self.anchor_gamma = _nonnegative("anchor_gamma", anchor_gamma)
        if self.anchor_mode == "none" and self.anchor_gamma != 0.0:
            raise ValueError("anchor_gamma must be zero when anchor_mode is none.")
        self.anchor_selector_version = str(
            anchor_selector_version or self.selector_version
        )
        self.anchor_selector_sha256 = str(
            anchor_selector_sha256 or self.selector_sha256
        ).lower()
        if (
            not self.anchor_selector_version
            or len(self.anchor_selector_sha256) != 64
        ):
            raise ValueError("Anchor selector version and SHA-256 are required.")
        self.ridge_lambda = _positive("ridge_lambda", ridge_lambda)
        self.exploration_scale = _positive("exploration_scale", exploration_scale)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.seed = int(seed)
        if data_mode not in {"real", "synthetic"}:
            raise ValueError("data_mode must be exactly 'real' or 'synthetic'.")
        self.data_mode = data_mode
        self.created_at = created_at or _utc_now()
        self.rng = np.random.default_rng(self.seed)
        self.A = {
            arm: self.ridge_lambda * np.eye(self.context_dim, dtype=np.float64)
            for arm in DIRECT_ARMS
        }
        self.b = {arm: np.zeros(self.context_dim, dtype=np.float64) for arm in DIRECT_ARMS}
        self.arm_update_counts = {arm: 0 for arm in DIRECT_ARMS}
        self.total_updates = 0
        self.applied_update_ids: list[str] = []
        self._applied_update_set: set[str] = set()

    def _context(self, value: object) -> np.ndarray:
        raw = np.asarray(value)
        if raw.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("context must contain only real numeric values.")
        result = np.asarray(raw, dtype=np.float64)
        if result.shape != (self.context_dim,) or not np.isfinite(result).all():
            raise ValueError(f"context must be finite with shape {(self.context_dim,)}.")
        return result

    def _anchor_scores(
        self,
        selector_probabilities: Mapping[str, object] | None,
    ) -> dict[str, float]:
        """Return a separate frozen-selector action anchor for every arm."""

        if self.anchor_mode == "none":
            return {arm: 0.0 for arm in DIRECT_ARMS}
        if not isinstance(selector_probabilities, Mapping):
            raise ValueError(
                "md7_logprob_anchor requires current selector probabilities."
            )
        if set(selector_probabilities) != set(DIRECT_ARMS):
            raise ValueError("Selector anchor needs exactly the four direct arms.")
        probabilities: dict[str, float] = {}
        for arm in DIRECT_ARMS:
            value = selector_probabilities[arm]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("Selector probabilities must be real numbers.")
            probability = float(value)
            if not isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Selector probabilities must be finite and in [0, 1].")
            probabilities[arm] = probability
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("Selector probabilities must sum to one.")
        return {
            arm: self.anchor_gamma
            * log(max(probabilities[arm], ANCHOR_PROBABILITY_FLOOR))
            for arm in DIRECT_ARMS
        }

    def select_arm(
        self,
        context: object,
        *,
        selector_probabilities: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Sample all reward arms, then add the separate action anchor."""

        x = self._context(context)
        reward_scores: dict[str, float] = {}
        thetas: dict[str, list[float]] = {}
        for arm in DIRECT_ARMS:
            mean = np.linalg.solve(self.A[arm], self.b[arm])
            inverse = np.linalg.solve(self.A[arm], np.eye(self.context_dim))
            covariance = self.exploration_scale**2 * 0.5 * (inverse + inverse.T)
            theta = self.rng.multivariate_normal(mean, covariance)
            reward_scores[arm] = float(x @ theta)
            thetas[arm] = theta.tolist()
        anchor_scores = self._anchor_scores(selector_probabilities)
        policy_scores = {
            arm: reward_scores[arm] + anchor_scores[arm] for arm in DIRECT_ARMS
        }
        p0_selected = max(DIRECT_ARMS, key=reward_scores.__getitem__)
        selected = max(DIRECT_ARMS, key=policy_scores.__getitem__)
        return {
            "selected_arm": selected,
            "p0_selected_arm": p0_selected,
            "p1_selected_arm": (
                selected if self.anchor_mode == "md7_logprob_anchor" else None
            ),
            "sampled_scores": policy_scores,
            "contextual_ts_reward_scores": reward_scores,
            "anchor_scores": anchor_scores,
            "policy_scores": policy_scores,
            "sampled_thetas": thetas,
        }

    def update_once(
        self,
        *,
        update_id: str,
        arm: str,
        context: object,
        reward: object,
        weight: object = 1.0,
    ) -> bool:
        """Apply one weighted observation exactly once in the posterior.

        Runtime observations retain unit weight.  The explicit ``weight``
        parameter exists for the audited synthetic warm-start lineage only;
        it does not change the update-count or idempotency semantics.
        """

        identity = str(update_id).strip()
        if not identity:
            raise ValueError("update_id must be non-empty.")
        if identity in self._applied_update_set:
            return False
        if arm not in DIRECT_ARMS:
            raise ValueError(f"Unknown direct arm: {arm!r}.")
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("reward must be a non-boolean real number.")
        r = float(reward)
        if not isfinite(r) or not -1.0 <= r <= 1.0:
            raise ValueError("reward must be finite and in [-1, 1].")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("weight must be a non-boolean real number.")
        w = float(weight)
        if not isfinite(w) or w <= 0.0:
            raise ValueError("weight must be finite and strictly positive.")
        x = self._context(context)
        candidate_A = self.A[arm] + w * np.outer(x, x)
        candidate_b = self.b[arm] + w * r * x
        if not np.isfinite(candidate_A).all() or not np.isfinite(candidate_b).all():
            raise ValueError("Turn-LinTS update produced non-finite state.")
        self.A[arm] = candidate_A
        self.b[arm] = candidate_b
        self.arm_update_counts[arm] += 1
        self.total_updates += 1
        self.applied_update_ids.append(identity)
        self._applied_update_set.add(identity)
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": POLICY_STATE_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "direct_arms": list(DIRECT_ARMS),
            "selector_version": self.selector_version,
            "selector_sha256": self.selector_sha256,
            "context_schema_id": self.context_schema_id,
            "enabled_context_blocks": list(self.enabled_context_blocks),
            "ordered_feature_names": list(self.feature_names),
            "context_dimension": self.context_dim,
            "reward_mode": self.reward_mode,
            "anchor_mode": self.anchor_mode,
            "anchor_gamma": self.anchor_gamma,
            "anchor_selector_version": self.anchor_selector_version,
            "anchor_selector_sha256": self.anchor_selector_sha256,
            "anchor_probability_floor": ANCHOR_PROBABILITY_FLOOR,
            "posterior_hyperparameters": {
                "ridge_lambda": self.ridge_lambda,
                "exploration_scale": self.exploration_scale,
                "seed": self.seed,
            },
            "data_mode": self.data_mode,
            "policy_created_at": self.created_at,
            "A": {arm: self.A[arm].tolist() for arm in DIRECT_ARMS},
            "b": {arm: self.b[arm].tolist() for arm in DIRECT_ARMS},
            "arm_update_counts": dict(self.arm_update_counts),
            "update_count": self.total_updates,
            "applied_update_ids": list(self.applied_update_ids),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Fail closed on every lineage, selector, context, or reward mismatch."""

        state_schema = state.get("schema_version")
        if state_schema not in {
            LEGACY_POLICY_STATE_SCHEMA_VERSION,
            POLICY_STATE_SCHEMA_VERSION,
        }:
            raise ValueError("Unsupported Turn-LinTS policy-state schema.")
        if state_schema == LEGACY_POLICY_STATE_SCHEMA_VERSION and (
            self.anchor_mode != "none" or self.anchor_gamma != 0.0
        ):
            raise ValueError("Turn-LinTS state mismatch for anchor_mode.")
        expected = {
            "algorithm_version": ALGORITHM_VERSION,
            "direct_arms": list(DIRECT_ARMS),
            "selector_version": self.selector_version,
            "selector_sha256": self.selector_sha256,
            "context_schema_id": self.context_schema_id,
            "enabled_context_blocks": list(self.enabled_context_blocks),
            "ordered_feature_names": list(self.feature_names),
            "context_dimension": self.context_dim,
            "reward_mode": self.reward_mode,
            "data_mode": self.data_mode,
        }
        if state_schema == POLICY_STATE_SCHEMA_VERSION:
            expected.update(
                {
                    "anchor_mode": self.anchor_mode,
                    "anchor_gamma": self.anchor_gamma,
                    "anchor_selector_version": self.anchor_selector_version,
                    "anchor_selector_sha256": self.anchor_selector_sha256,
                    "anchor_probability_floor": ANCHOR_PROBABILITY_FLOOR,
                }
            )
        version_label_fields = {"selector_version", "anchor_selector_version"}
        for field, value in expected.items():
            actual = state.get(field)
            if field in version_label_fields:
                matches = (
                    isinstance(actual, str)
                    and actual.casefold() == str(value).casefold()
                )
            else:
                matches = actual == value
            if not matches:
                raise ValueError(f"Turn-LinTS state mismatch for {field}.")
        hyper = state.get("posterior_hyperparameters")
        expected_hyper = {"ridge_lambda": self.ridge_lambda,
                          "exploration_scale": self.exploration_scale,
                          "seed": self.seed}
        if hyper != expected_hyper:
            raise ValueError("Turn-LinTS posterior hyperparameter mismatch.")
        raw_A, raw_b = state.get("A"), state.get("b")
        raw_counts = state.get("arm_update_counts")
        if not all(isinstance(v, Mapping) for v in (raw_A, raw_b, raw_counts)):
            raise TypeError("Turn-LinTS posterior mappings are missing.")
        if any(tuple(v.keys()) != DIRECT_ARMS for v in (raw_A, raw_b, raw_counts)):
            raise ValueError("Turn-LinTS posterior arm order mismatch.")
        candidate_A: dict[str, np.ndarray] = {}
        candidate_b: dict[str, np.ndarray] = {}
        candidate_counts: dict[str, int] = {}
        for arm in DIRECT_ARMS:
            matrix = np.asarray(raw_A[arm], dtype=np.float64)  # type: ignore[index]
            vector = np.asarray(raw_b[arm], dtype=np.float64)  # type: ignore[index]
            if matrix.shape != (self.context_dim, self.context_dim):
                raise ValueError(f"A[{arm}] has incompatible shape.")
            if vector.shape != (self.context_dim,):
                raise ValueError(f"b[{arm}] has incompatible shape.")
            if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
                raise ValueError("Turn-LinTS posterior contains non-finite values.")
            if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
                raise ValueError(f"A[{arm}] is not symmetric.")
            np.linalg.cholesky(matrix)
            count = raw_counts[arm]  # type: ignore[index]
            if isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 0:
                raise ValueError("Invalid Turn-LinTS arm update count.")
            candidate_A[arm], candidate_b[arm] = matrix, vector
            candidate_counts[arm] = int(count)
        total = state.get("update_count")
        if isinstance(total, bool) or not isinstance(total, Integral):
            raise ValueError("Invalid Turn-LinTS update count.")
        if int(total) != sum(candidate_counts.values()):
            raise ValueError("Turn-LinTS update counts are inconsistent.")
        ids = state.get("applied_update_ids")
        if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("Invalid applied update identities.")
        if len(ids) != len(set(ids)) or len(ids) != int(total):
            raise ValueError("Applied update identities are inconsistent.")
        created = state.get("policy_created_at")
        if not isinstance(created, str) or not created:
            raise ValueError("Policy creation timestamp is missing.")
        candidate_rng = np.random.default_rng(self.seed)
        candidate_rng.bit_generator.state = copy.deepcopy(state.get("rng_state"))
        self.A, self.b = candidate_A, candidate_b
        self.arm_update_counts, self.total_updates = candidate_counts, int(total)
        self.applied_update_ids, self._applied_update_set = list(ids), set(ids)
        self.created_at, self.rng = created, candidate_rng


def save_turn_lints_state(policy: DirectTurnLinTS, path: str | Path) -> Path:
    destination = Path(path)
    from .postgres_repository import (
        policy_key_for_path,
        postgres_enabled,
        save_policy_snapshot,
    )
    if postgres_enabled():
        save_policy_snapshot(
            policy_key=policy_key_for_path(destination),
            payload=policy.state_dict(),
            policy_class=type(policy).__name__,
        )
        return destination
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing_schema = existing.get("schema_version")
        if existing_schema not in {
            LEGACY_POLICY_STATE_SCHEMA_VERSION,
            POLICY_STATE_SCHEMA_VERSION,
        }:
            raise ValueError("Refusing to overwrite non-Turn-LinTS state.")
        if (
            existing_schema == LEGACY_POLICY_STATE_SCHEMA_VERSION
            and policy.anchor_mode != "none"
        ):
            raise ValueError(
                "Refusing implicit anchored migration of a v1 Turn-LinTS state."
            )
    serialized = json.dumps(policy.state_dict(), ensure_ascii=False, allow_nan=False,
                            indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                               dir=destination.parent, prefix=f".{destination.name}.",
                               suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_turn_lints_state(policy: DirectTurnLinTS, path: str | Path) -> None:
    source = Path(path)
    from .postgres_repository import (
        load_policy_snapshot,
        policy_key_for_path,
        postgres_enabled,
    )
    raw = (
        load_policy_snapshot(policy_key_for_path(source))
        if postgres_enabled()
        else json.loads(source.read_text(encoding="utf-8"))
    )
    if not isinstance(raw, Mapping):
        raise ValueError("Turn-LinTS state must contain one JSON object.")
    policy.load_state_dict(raw)


__all__ = (
    "ALGORITHM_VERSION", "ANCHOR_MODES", "ANCHOR_PROBABILITY_FLOOR",
    "DEFAULT_EXPLORATION_SCALE", "DEFAULT_RIDGE_LAMBDA", "DIRECT_ARMS",
    "DirectTurnLinTS", "LEGACY_POLICY_STATE_SCHEMA_VERSION",
    "POLICY_STATE_SCHEMA_VERSION", "load_turn_lints_state", "save_turn_lints_state",
)
