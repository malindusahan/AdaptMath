"""Pre-specified Stage B gap-threshold sensitivity experiment.

This module varies only the conservative MD6 probability-gap threshold while
holding the C3 turn context, synthetic worlds, LinTS mechanics, reward, and
delayed-credit rule fixed. It reports neutral performance and intervention
statistics and deliberately does not select or recommend a threshold.

Run deterministic checks only::

    python -m src.self_improvement.experiments.threshold_sensitivity --sanity-only

Run the full pre-specified experiment::

    python -m src.self_improvement.experiments.threshold_sensitivity --run-full
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.self_improvement.conservative_overlay import (
    BIAS_TARGET,
    apply_conservative_overlay,
    eligible_arms,
)
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.experiments import context_ablation as base
from src.self_improvement.lints_policy import ARMS, TrueDisjointLinTS
from src.self_improvement.reward import primary_reward
from src.self_improvement.turn_context_builder import MRB1_TASKS, RunningTutorQuality


EXPERIMENT_NAME: Final[str] = "threshold_sensitivity_v1"
SELECTED_CONTEXT: Final[str] = "C3"
SELECTED_CONTEXT_FEATURES: Final[tuple[str, ...]] = base.CONTEXT_FEATURES[
    SELECTED_CONTEXT
]
THRESHOLDS: Final[tuple[float, ...]] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30)
REFERENCE_THRESHOLD: Final[float] = 0.10
ENVIRONMENTS: Final[tuple[str, ...]] = (
    "E0_NO_CONTEXT",
    "E3_QUALITY",
    "E4_MIXED",
)
ENVIRONMENT_INDEX: Final[dict[str, int]] = {
    environment: base.ENVIRONMENTS.index(environment) for environment in ENVIRONMENTS
}
SEEDS: Final[tuple[int, ...]] = base.SEEDS
NUM_SEEDS: Final[int] = len(SEEDS)
ATTEMPTS_PER_SEED: Final[int] = base.ATTEMPTS_PER_SEED
TURNS_PER_ATTEMPT: Final[int] = base.TURNS_PER_ATTEMPT
CHECKPOINTS: Final[tuple[int, ...]] = base.CHECKPOINTS
RIDGE_LAMBDA: Final[float] = base.RIDGE_LAMBDA
EXPLORATION_SCALE: Final[float] = base.EXPLORATION_SCALE
BASE_SEED: Final[int] = base.BASE_SEED
NUMERIC_TOLERANCE: Final[float] = base.NUMERIC_TOLERANCE
BOOTSTRAP_SEED: Final[int] = 482_711
PERMUTATION_SEED: Final[int] = 931_027
NUM_BOOTSTRAP_RESAMPLES: Final[int] = 2_000
NUM_PERMUTATIONS: Final[int] = 50_000

# Explicit comparisons against the current candidate threshold. Adjacent
# comparisons were optional in the protocol and are not added to this family.
THRESHOLD_COMPARISONS: Final[tuple[tuple[float, float], ...]] = (
    (0.05, 0.10),
    (0.15, 0.10),
    (0.20, 0.10),
    (0.30, 0.10),
    (0.00, 0.10),
)
PAIRED_METRICS: Final[tuple[str, ...]] = (
    "cumulative_regret",
    "mean_reward",
    "optimal_arm_rate",
    "at_least_one_alternative_rate",
    "override_rate",
)
SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "cumulative_regret",
    "mean_reward",
    "optimal_arm_rate",
    "mean_eligible_arm_count",
    "baseline_only_rate",
    "at_least_one_alternative_rate",
    "at_least_two_alternatives_rate",
    "override_rate",
    "bias_arm_selection_rate",
    "baseline_selection_rate",
    "mean_override_gap",
    "median_override_gap",
    "p90_override_gap",
    "max_override_gap",
)

CANONICAL_HASH_FILES: Final[tuple[str, ...]] = (
    "lints_policy.py",
    "conservative_overlay.py",
    "turn_context_builder.py",
    "turn_level_controller.py",
    "reward.py",
)

ATTEMPT_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "threshold",
    "seed",
    "attempt_index",
    "attempt_regret",
    "cumulative_regret",
    "p_policy",
    "p_oracle",
    "evaluator_score",
    "reward",
    "optimal_arm_rate_for_attempt",
    "mean_eligible_arm_count_for_attempt",
    "baseline_only_rate_for_attempt",
    "at_least_one_alternative_rate_for_attempt",
    "at_least_two_alternatives_rate_for_attempt",
    "override_rate_for_attempt",
    "bias_arm_selection_rate_for_attempt",
    "baseline_selection_rate_for_attempt",
    "mean_override_gap_for_attempt",
    "median_override_gap_for_attempt",
    "p90_override_gap_for_attempt",
    "max_override_gap_for_attempt",
    "selected_arms",
    "eligible_arm_counts",
)

TURN_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "threshold",
    "seed",
    "attempt_index",
    "turn_index",
    "base_move",
    "selected_arm",
    "oracle_arm",
    "selected_is_oracle",
    "final_move",
    "target_move",
    "overridden",
    "eligible_arm_count",
    "eligible_arms",
    "base_probability",
    "target_probability",
    "gap",
    "override_gap",
    "hidden_selected_utility",
    "hidden_oracle_utility",
    "turn_regret",
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

CHECKPOINT_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "threshold",
    "seed",
    "checkpoint",
    "cumulative_regret",
    "mean_regret",
    "mean_reward",
    "optimal_arm_rate",
    "mean_eligible_arm_count",
    "baseline_only_rate",
    "at_least_one_alternative_rate",
    "at_least_two_alternatives_rate",
    "override_rate",
    "bias_arm_selection_rate",
    "baseline_selection_rate",
    "mean_override_gap",
    "median_override_gap",
    "p90_override_gap",
    "max_override_gap",
)


def _md6_mapping(probabilities: Sequence[float]) -> dict[str, float]:
    return {
        move: float(probability)
        for move, probability in zip(MOVE_ORDER, probabilities, strict=True)
    }


def _policy_seed(environment_index: int, seed: int) -> int:
    """Common LinTS stream shared by thresholds for an env/seed pair."""

    return base._policy_seed(  # noqa: SLF001 - exact experiment reuse is deliberate
        environment_index,
        seed,
        base.CONTEXTS.index(SELECTED_CONTEXT),
    )


def _gap_summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        nan = float("nan")
        return nan, nan, nan, nan
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.median(array)),
        float(np.percentile(array, 90)),
        float(np.max(array)),
    )


def simulate_run(
    environment: str,
    environment_index: int,
    threshold: float,
    seed: int,
    hidden: base.HiddenEnvironment,
    world: base.ExogenousWorld,
    attempts: int,
    *,
    assert_invariants: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Simulate one fixed-C3 environment/threshold/seed run."""

    policy = TrueDisjointLinTS(
        context_dim=len(SELECTED_CONTEXT_FEATURES),
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=_policy_seed(environment_index, seed),
        data_mode="synthetic",
    )
    attempt_rows: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []

    cumulative_regret = 0.0
    cumulative_reward = 0.0
    cumulative_optimal_turns = 0
    cumulative_eligible_count = 0
    cumulative_baseline_only = 0
    cumulative_one_alternative = 0
    cumulative_two_alternatives = 0
    cumulative_bias_selections = 0
    cumulative_overrides = 0
    cumulative_baseline_selections = 0
    cumulative_override_gaps: list[float] = []

    for attempt_zero in range(attempts):
        attempt_index = attempt_zero + 1
        mastery = float(world.mastery_before[attempt_zero])
        running_quality = RunningTutorQuality()
        updates_at_attempt_start = policy.total_updates
        pending_updates: list[tuple[str, np.ndarray]] = []
        selected_utilities: list[float] = []
        oracle_utilities: list[float] = []
        selected_arms: list[str] = []
        optimal_indicators: list[int] = []
        eligible_counts: list[int] = []
        attempt_override_gaps: list[float] = []
        overrides: list[int] = []
        bias_selections: list[int] = []
        baseline_selections: list[int] = []

        for turn_zero in range(TURNS_PER_ATTEMPT):
            turn_index = turn_zero + 1
            md6_array = world.md6_probabilities[attempt_zero, turn_zero]
            md6 = _md6_mapping(md6_array)
            quality_before = running_quality.snapshot()
            context = base.context_vector(
                SELECTED_CONTEXT, md6, mastery, quality_before
            )
            candidates = eligible_arms(md6, threshold)
            decision = policy.select_arm(context, candidates)
            selected_arm = decision["selected_arm"]
            if assert_invariants:
                assert selected_arm in candidates
                assert policy.total_updates == updates_at_attempt_start

            utilities = {
                arm: base.hidden_arm_utility(
                    hidden, arm, md6_array, mastery, quality_before
                )
                for arm in candidates
            }
            oracle_arm = max(candidates, key=utilities.__getitem__)
            if assert_invariants:
                assert oracle_arm in candidates
            selected_utility = float(utilities[selected_arm])
            oracle_utility = float(utilities[oracle_arm])
            turn_regret = oracle_utility - selected_utility
            if turn_regret < -NUMERIC_TOLERANCE:
                raise AssertionError("Eligible-oracle regret became negative.")
            turn_regret = max(0.0, float(turn_regret))

            overlay = apply_conservative_overlay(selected_arm, md6, threshold)
            is_bias = int(selected_arm != "baseline")
            is_override = int(overlay.overridden)
            is_baseline = int(selected_arm == "baseline")
            override_gap = float(overlay.gap) if overlay.overridden else float("nan")
            if overlay.overridden:
                attempt_override_gaps.append(float(overlay.gap))

            selected_utilities.append(selected_utility)
            oracle_utilities.append(oracle_utility)
            selected_arms.append(selected_arm)
            optimal_indicators.append(int(selected_arm == oracle_arm))
            eligible_counts.append(len(candidates))
            overrides.append(is_override)
            bias_selections.append(is_bias)
            baseline_selections.append(is_baseline)
            pending_updates.append((selected_arm, context.copy()))

            turn_rows.append(
                {
                    "environment": environment,
                    "threshold": threshold,
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "turn_index": turn_index,
                    "base_move": overlay.base_move,
                    "selected_arm": selected_arm,
                    "oracle_arm": oracle_arm,
                    "selected_is_oracle": int(selected_arm == oracle_arm),
                    "final_move": overlay.final_move,
                    "target_move": overlay.target_move,
                    "overridden": is_override,
                    "eligible_arm_count": len(candidates),
                    "eligible_arms": "|".join(candidates),
                    "base_probability": overlay.base_probability,
                    "target_probability": overlay.target_probability,
                    "gap": overlay.gap,
                    "override_gap": override_gap,
                    "hidden_selected_utility": selected_utility,
                    "hidden_oracle_utility": oracle_utility,
                    "turn_regret": turn_regret,
                    "md6_p_generic": float(md6_array[0]),
                    "md6_p_probing": float(md6_array[1]),
                    "md6_p_focus": float(md6_array[2]),
                    "md6_p_telling": float(md6_array[3]),
                    "running_mistake_identification": quality_before.feature_values[0],
                    "running_mistake_location": quality_before.feature_values[1],
                    "running_providing_guidance": quality_before.feature_values[2],
                    "running_actionability": quality_before.feature_values[3],
                    "has_within_attempt_quality": quality_before.has_within_attempt_quality,
                }
            )

            scores = world.mrb1_scores[attempt_zero, turn_zero]
            running_quality.add_scores(
                {
                    task: float(score)
                    for task, score in zip(MRB1_TASKS, scores, strict=True)
                }
            )
            if assert_invariants:
                assert policy.total_updates == updates_at_attempt_start

        p_policy = float(np.mean(selected_utilities))
        p_oracle = float(np.mean(oracle_utilities))
        attempt_regret = p_oracle - p_policy
        if attempt_regret < -NUMERIC_TOLERANCE:
            raise AssertionError("Attempt eligible-oracle regret became negative.")
        attempt_regret = max(0.0, float(attempt_regret))
        evaluator_score = int(
            np.count_nonzero(world.evaluator_uniforms[attempt_zero] < p_policy)
        )
        reward = primary_reward(evaluator_score)
        if assert_invariants:
            assert evaluator_score in {0, 1, 2, 3}
            assert reward in {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
            assert policy.total_updates == updates_at_attempt_start

        sample_weight = 1.0 / TURNS_PER_ATTEMPT
        if not math.isclose(
            sample_weight * len(pending_updates), 1.0, abs_tol=NUMERIC_TOLERANCE
        ):
            raise AssertionError("Delayed turn weights do not total one.")
        for arm, context in pending_updates:
            policy.update(arm, context, reward, sample_weight=sample_weight)
        if assert_invariants:
            assert policy.total_updates == updates_at_attempt_start + TURNS_PER_ATTEMPT

        cumulative_regret += attempt_regret
        cumulative_reward += reward
        cumulative_optimal_turns += sum(optimal_indicators)
        cumulative_eligible_count += sum(eligible_counts)
        cumulative_baseline_only += sum(count == 1 for count in eligible_counts)
        cumulative_one_alternative += sum(count >= 2 for count in eligible_counts)
        cumulative_two_alternatives += sum(count >= 3 for count in eligible_counts)
        cumulative_bias_selections += sum(bias_selections)
        cumulative_overrides += sum(overrides)
        cumulative_baseline_selections += sum(baseline_selections)
        cumulative_override_gaps.extend(attempt_override_gaps)

        gap_mean, gap_median, gap_p90, gap_max = _gap_summary(
            attempt_override_gaps
        )
        attempt_rows.append(
            {
                "environment": environment,
                "threshold": threshold,
                "seed": seed,
                "attempt_index": attempt_index,
                "attempt_regret": attempt_regret,
                "cumulative_regret": cumulative_regret,
                "p_policy": p_policy,
                "p_oracle": p_oracle,
                "evaluator_score": evaluator_score,
                "reward": reward,
                "optimal_arm_rate_for_attempt": float(np.mean(optimal_indicators)),
                "mean_eligible_arm_count_for_attempt": float(np.mean(eligible_counts)),
                "baseline_only_rate_for_attempt": float(
                    np.mean(np.asarray(eligible_counts) == 1)
                ),
                "at_least_one_alternative_rate_for_attempt": float(
                    np.mean(np.asarray(eligible_counts) >= 2)
                ),
                "at_least_two_alternatives_rate_for_attempt": float(
                    np.mean(np.asarray(eligible_counts) >= 3)
                ),
                "override_rate_for_attempt": float(np.mean(overrides)),
                "bias_arm_selection_rate_for_attempt": float(
                    np.mean(bias_selections)
                ),
                "baseline_selection_rate_for_attempt": float(
                    np.mean(baseline_selections)
                ),
                "mean_override_gap_for_attempt": gap_mean,
                "median_override_gap_for_attempt": gap_median,
                "p90_override_gap_for_attempt": gap_p90,
                "max_override_gap_for_attempt": gap_max,
                "selected_arms": "|".join(selected_arms),
                "eligible_arm_counts": "|".join(str(x) for x in eligible_counts),
            }
        )

        if attempt_index in CHECKPOINTS:
            turns_seen = attempt_index * TURNS_PER_ATTEMPT
            all_gap_mean, all_gap_median, all_gap_p90, all_gap_max = _gap_summary(
                cumulative_override_gaps
            )
            checkpoint_rows.append(
                {
                    "environment": environment,
                    "threshold": threshold,
                    "seed": seed,
                    "checkpoint": attempt_index,
                    "cumulative_regret": cumulative_regret,
                    "mean_regret": cumulative_regret / attempt_index,
                    "mean_reward": cumulative_reward / attempt_index,
                    "optimal_arm_rate": cumulative_optimal_turns / turns_seen,
                    "mean_eligible_arm_count": cumulative_eligible_count / turns_seen,
                    "baseline_only_rate": cumulative_baseline_only / turns_seen,
                    "at_least_one_alternative_rate": (
                        cumulative_one_alternative / turns_seen
                    ),
                    "at_least_two_alternatives_rate": (
                        cumulative_two_alternatives / turns_seen
                    ),
                    "override_rate": cumulative_overrides / turns_seen,
                    "bias_arm_selection_rate": cumulative_bias_selections / turns_seen,
                    "baseline_selection_rate": (
                        cumulative_baseline_selections / turns_seen
                    ),
                    "mean_override_gap": all_gap_mean,
                    "median_override_gap": all_gap_median,
                    "p90_override_gap": all_gap_p90,
                    "max_override_gap": all_gap_max,
                }
            )

    return attempt_rows, turn_rows, checkpoint_rows


def run_sanity_checks() -> list[dict[str, str]]:
    """Run all pre-specified deterministic checks before full execution."""

    checks: list[dict[str, str]] = []

    def passed(name: str) -> None:
        checks.append({"check": name, "status": "PASS"})

    md6 = _md6_mapping([0.31, 0.28, 0.23, 0.18])
    empty_quality = RunningTutorQuality()
    context_low = base.context_vector(SELECTED_CONTEXT, md6, 0.1, empty_quality)
    context_high = base.context_vector(SELECTED_CONTEXT, md6, 0.9, empty_quality)
    assert context_low.shape == (9,)
    assert len(SELECTED_CONTEXT_FEATURES) == 9
    passed("selected context dimension equals 9")
    assert "mastery_before" not in SELECTED_CONTEXT_FEATURES
    np.testing.assert_array_equal(context_low, context_high)
    passed("mastery is absent from the threshold experiment context")

    # Fixed-state eligibility must be monotonic, and zero admits no continuous
    # non-ties in this deterministic sample.
    audit_world = base.generate_exogenous_world(ENVIRONMENT_INDEX["E4_MIXED"], 2, 80)
    for md6_array in audit_world.md6_probabilities.reshape(-1, len(MOVE_ORDER)):
        mapping = _md6_mapping(md6_array)
        counts = [len(eligible_arms(mapping, threshold)) for threshold in THRESHOLDS]
        assert all(left <= right for left, right in zip(counts, counts[1:]))
    passed("increasing threshold never reduces fixed-state eligible-arm count")
    zero_counts = [
        len(eligible_arms(_md6_mapping(row), 0.0))
        for row in audit_world.md6_probabilities.reshape(-1, len(MOVE_ORDER))
    ]
    assert float(np.mean(np.asarray(zero_counts) == 1)) >= 0.999
    passed("threshold zero is a near-baseline-only control")

    environment = "E4_MIXED"
    environment_index = ENVIRONMENT_INDEX[environment]
    hidden = base.generate_hidden_environment(environment, environment_index, 3)
    world = base.generate_exogenous_world(environment_index, 3, 6)
    first = simulate_run(
        environment, environment_index, 0.10, 3, hidden, world, 6
    )
    attempts, turns, checkpoints = first
    assert all(str(row["oracle_arm"]) in str(row["eligible_arms"]).split("|") for row in turns)
    passed("oracle arm always eligible")
    assert all(str(row["selected_arm"]) in str(row["eligible_arms"]).split("|") for row in turns)
    passed("policy selected arm always eligible")
    assert all(float(row["attempt_regret"]) >= -NUMERIC_TOLERANCE for row in attempts)
    assert all(float(row["turn_regret"]) >= -NUMERIC_TOLERANCE for row in turns)
    passed("regret is nonnegative within tolerance")
    allowed = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
    assert all(float(row["reward"]) in allowed for row in attempts)
    passed("evaluator reward has the prescribed support")
    # Timing and weighting assertions are embedded in every simulate_run turn.
    passed("no within-attempt posterior update")
    passed("delayed total turn weight per attempt equals one")
    assert all(
        float(row["has_within_attempt_quality"])
        == (0.0 if int(row["turn_index"]) == 1 else 1.0)
        for row in turns
    )
    passed("current-turn MRB1 is unavailable before action selection")
    second = simulate_run(
        environment, environment_index, 0.10, 3, hidden, world, 6
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    passed("deterministic rerun reproduces results")
    assert checkpoints == []
    return checks


def _bootstrap_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(NUM_BOOTSTRAP_RESAMPLES, n))
    weights = np.zeros((NUM_BOOTSTRAP_RESAMPLES, n), dtype=np.float64)
    rows = np.repeat(np.arange(NUM_BOOTSTRAP_RESAMPLES), n)
    np.add.at(weights, (rows, indices.ravel()), 1.0)
    return weights / n


def _bootstrap_mean_ci(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).all():
        return float("nan"), float("nan")
    if np.isnan(array).any():
        raise ValueError("Partially missing seed-level metric cannot be bootstrapped.")
    boot = weights @ array
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def _paired_permutation_p(values: np.ndarray, stream_index: int) -> float:
    differences = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(
        np.random.SeedSequence([PERMUTATION_SEED, stream_index])
    )
    exceed = 0
    completed = 0
    batch_size = 2_000
    while completed < NUM_PERMUTATIONS:
        batch = min(batch_size, NUM_PERMUTATIONS - completed)
        signs = rng.integers(0, 2, size=(batch, differences.size), dtype=np.int8)
        randomized = np.mean(
            (signs.astype(np.float64) * 2.0 - 1.0) * differences[None, :], axis=1
        )
        exceed += int(np.count_nonzero(np.abs(randomized) >= observed - 1e-15))
        completed += batch
    return float((exceed + 1) / (NUM_PERMUTATIONS + 1))


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float((total - rank) * values[index])))
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def compute_summary_statistics(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (environment, threshold, checkpoint), group in checkpoint_df.groupby(
        ["environment", "threshold", "checkpoint"], sort=False
    ):
        ordered = group.sort_values("seed")
        if len(ordered) != NUM_SEEDS:
            raise RuntimeError("Summary group does not contain all 50 seeds.")
        row: dict[str, object] = {
            "environment": environment,
            "threshold": float(threshold),
            "checkpoint": int(checkpoint),
            "n_seeds": len(ordered),
        }
        for metric in SUMMARY_METRICS:
            all_values = ordered[metric].to_numpy(dtype=np.float64)
            values = all_values[np.isfinite(all_values)]
            n_valid = int(values.size)
            row[f"{metric}_n_valid_seeds"] = n_valid
            if n_valid:
                low, high = _bootstrap_mean_ci(
                    values, _bootstrap_weights(n_valid)
                )
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = (
                    float(np.std(values, ddof=1))
                    if n_valid > 1
                    else float("nan")
                )
                row[f"{metric}_median"] = float(np.median(values))
            else:
                low, high = float("nan"), float("nan")
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_sd"] = float("nan")
                row[f"{metric}_median"] = float("nan")
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_comparisons(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired evidence with difference defined as threshold_a - b."""

    weights = _bootstrap_weights(NUM_SEEDS)
    rows: list[dict[str, object]] = []
    stream_index = 0
    for environment in ENVIRONMENTS:
        for checkpoint in CHECKPOINTS:
            subset = checkpoint_df[
                (checkpoint_df["environment"] == environment)
                & (checkpoint_df["checkpoint"] == checkpoint)
            ]
            for metric in PAIRED_METRICS:
                family_indices: list[int] = []
                pivot = subset.pivot(index="seed", columns="threshold", values=metric).sort_index()
                if len(pivot) != NUM_SEEDS:
                    raise RuntimeError("Incomplete seed pairing.")
                for threshold_a, threshold_b in THRESHOLD_COMPARISONS:
                    values_a = pivot[threshold_a].to_numpy(dtype=np.float64)
                    values_b = pivot[threshold_b].to_numpy(dtype=np.float64)
                    differences = values_a - values_b
                    low, high = _bootstrap_mean_ci(differences, weights)
                    p_value = _paired_permutation_p(differences, stream_index)
                    stream_index += 1
                    family_indices.append(len(rows))
                    rows.append(
                        {
                            "environment": environment,
                            "comparison": f"{threshold_a:.2f}-{threshold_b:.2f}",
                            "checkpoint": checkpoint,
                            "n_pairs": len(differences),
                            "metric_name": metric,
                            "threshold_a": threshold_a,
                            "threshold_b": threshold_b,
                            "mean_a": float(np.mean(values_a)),
                            "mean_b": float(np.mean(values_b)),
                            "mean_paired_difference": float(np.mean(differences)),
                            "median_paired_difference": float(np.median(differences)),
                            "sd_paired_difference": float(np.std(differences, ddof=1)),
                            "bootstrap_ci_low": low,
                            "bootstrap_ci_high": high,
                            "permutation_p": p_value,
                            "holm_adjusted_p": float("nan"),
                        }
                    )
                adjusted = _holm_adjust(rows[index]["permutation_p"] for index in family_indices)
                for index, value in zip(family_indices, adjusted, strict=True):
                    rows[index]["holm_adjusted_p"] = value
    return pd.DataFrame(rows)


def compute_tradeoff_summary(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    endpoint = checkpoint_df[checkpoint_df["checkpoint"] == ATTEMPTS_PER_SEED]
    metrics = {
        "cumulative_regret": "mean_cumulative_regret",
        "mean_reward": "mean_reward",
        "mean_eligible_arm_count": "mean_eligible_arm_count",
        "at_least_one_alternative_rate": "at_least_one_alternative_rate",
        "override_rate": "override_rate",
        "mean_override_gap": "mean_override_gap",
    }
    grouped = (
        endpoint.groupby(["environment", "threshold"], sort=False)[list(metrics)]
        .mean()
        .rename(columns=metrics)
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for environment in ENVIRONMENTS:
        env_rows = grouped[grouped["environment"] == environment].set_index("threshold")
        reference = env_rows.loc[REFERENCE_THRESHOLD]
        for threshold in THRESHOLDS:
            current = env_rows.loc[threshold]
            delta_regret = float(current["mean_cumulative_regret"] - reference["mean_cumulative_regret"])
            delta_reward = float(current["mean_reward"] - reference["mean_reward"])
            delta_alt = float(current["at_least_one_alternative_rate"] - reference["at_least_one_alternative_rate"])
            delta_override = float(current["override_rate"] - reference["override_rate"])
            delta_gap = float(current["mean_override_gap"] - reference["mean_override_gap"])
            rows.append(
                {
                    "environment": environment,
                    "threshold": threshold,
                    "checkpoint": ATTEMPTS_PER_SEED,
                    "mean_cumulative_regret": float(current["mean_cumulative_regret"]),
                    "mean_reward": float(current["mean_reward"]),
                    "mean_eligible_arm_count": float(current["mean_eligible_arm_count"]),
                    "at_least_one_alternative_rate": float(current["at_least_one_alternative_rate"]),
                    "override_rate": float(current["override_rate"]),
                    "mean_override_gap": float(current["mean_override_gap"]),
                    "delta_regret_vs_010": delta_regret,
                    "delta_reward_vs_010": delta_reward,
                    "delta_alt_rate_vs_010": delta_alt,
                    "delta_override_rate_vs_010": delta_override,
                    "delta_override_gap_vs_010": delta_gap,
                    "regret_change_per_additional_override_opportunity": (
                        delta_regret / delta_alt if abs(delta_alt) > NUMERIC_TOLERANCE else float("nan")
                    ),
                    "regret_change_per_additional_override": (
                        delta_regret / delta_override if abs(delta_override) > NUMERIC_TOLERANCE else float("nan")
                    ),
                    "reward_change_per_additional_override": (
                        delta_reward / delta_override if abs(delta_override) > NUMERIC_TOLERANCE else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_distributions(
    selected_counts: Mapping[tuple[str, float, int, str], int],
    eligible_counts: Mapping[tuple[str, float, int, int], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    turns_per_run = ATTEMPTS_PER_SEED * TURNS_PER_ATTEMPT
    selected_rows: list[dict[str, object]] = []
    eligible_rows: list[dict[str, object]] = []
    for environment in ENVIRONMENTS:
        for threshold in THRESHOLDS:
            for seed in SEEDS:
                for arm in ARMS:
                    count = int(selected_counts.get((environment, threshold, seed, arm), 0))
                    selected_rows.append(
                        {
                            "environment": environment,
                            "threshold": threshold,
                            "seed": seed,
                            "selected_arm": arm,
                            "count": count,
                            "rate": count / turns_per_run,
                        }
                    )
                for count_value in range(1, 5):
                    count = int(
                        eligible_counts.get(
                            (environment, threshold, seed, count_value), 0
                        )
                    )
                    eligible_rows.append(
                        {
                            "environment": environment,
                            "threshold": threshold,
                            "seed": seed,
                            "eligible_arm_count": count_value,
                            "count": count,
                            "rate": count / turns_per_run,
                        }
                    )
    return pd.DataFrame(selected_rows), pd.DataFrame(eligible_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _manifest(
    project_root: Path,
    output_dir: Path,
    sanity_checks: list[dict[str, str]],
    elapsed_seconds: float,
) -> dict[str, object]:
    source_dir = project_root / "src" / "self_improvement"
    script_path = Path(__file__).resolve()
    stage_a_manifest = (
        project_root
        / "results"
        / "self_improvement"
        / "md6_gap_threshold_audit"
        / "audit_manifest.json"
    )
    result_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    return {
        "experiment_name": EXPERIMENT_NAME,
        "stage": "B_gap_threshold_performance_sensitivity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_timezone": "UTC",
        "git_commit": _git_commit(project_root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "base_seed": BASE_SEED,
        "seed_list": list(SEEDS),
        "threshold_grid": list(THRESHOLDS),
        "reference_threshold_for_explicit_comparisons": REFERENCE_THRESHOLD,
        "environments": {
            environment: base.ENVIRONMENT_DEFINITIONS[environment]
            for environment in ENVIRONMENTS
        },
        "attempts_per_seed": ATTEMPTS_PER_SEED,
        "turns_per_attempt": TURNS_PER_ATTEMPT,
        "checkpoints": list(CHECKPOINTS),
        "selected_context": {
            "name": SELECTED_CONTEXT,
            "dimension": len(SELECTED_CONTEXT_FEATURES),
            "features": list(SELECTED_CONTEXT_FEATURES),
            "mastery_before_observed_by_policy": False,
        },
        "fixed_ridge_lambda": RIDGE_LAMBDA,
        "fixed_exploration_scale": EXPLORATION_SCALE,
        "fixed_reward_definition": "evaluator_score / 3.0; evaluator_score in {0,1,2,3}",
        "fixed_delayed_credit_scheme": (
            "All three selected turns update only after attempt completion, "
            "each with sample_weight=1/3; total attempt weight=1."
        ),
        "oracle_definition": (
            "Canonical-order argmax hidden utility over the current "
            "threshold-specific eligible-arm set only."
        ),
        "eligibility_definition": (
            "baseline plus each non-base target with P(base)-P(target) <= threshold; "
            "a bias targeting the current base move is excluded."
        ),
        "intervention_metric_definitions": {
            "baseline_only_rate": "Fraction of turns with exactly one eligible arm.",
            "at_least_one_alternative_rate": "Fraction of turns with at least two eligible arms.",
            "at_least_two_alternatives_rate": "Fraction of turns with at least three eligible arms.",
            "bias_arm_selection_rate": "Fraction of turns selecting a non-baseline LinTS arm.",
            "actual_override_rate": "Fraction of turns where the selected overlay changes MD6's base move.",
            "override_gap": "P(base)-P(selected bias target), only on actual override turns; unavailable otherwise.",
        },
        "bootstrap_method": (
            "Nonparametric percentile bootstrap across the 50 paired seed units; "
            "attempts and turns are never resampled as independent units. For "
            "override-gap summaries, seeds with no observed override have an "
            "unavailable value and the bootstrap uses the reported metric-specific "
            "number of valid seed units; unavailable values are never replaced by zero."
        ),
        "bootstrap_resamples": NUM_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_method": "Two-sided Monte Carlo paired sign-flip test of seed-level mean differences with +1 correction.",
        "permutation_resamples": NUM_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "paired_difference_sign_convention": (
            "metric_at_threshold_a - metric_at_threshold_b for every metric. "
            "For cumulative regret, a positive value means threshold_b has lower regret."
        ),
        "holm_correction_scope": (
            "The five explicit reference comparisons within each environment, "
            "checkpoint, and metric family."
        ),
        "tradeoff_delta_definition": "threshold metric minus the corresponding 0.10-reference mean.",
        "tradeoff_ratio_definitions": {
            "regret_change_per_additional_override_opportunity": "delta cumulative regret / delta at-least-one-alternative rate",
            "regret_change_per_additional_override": "delta cumulative regret / delta actual override rate",
            "reward_change_per_additional_override": "delta mean reward / delta actual override rate",
        },
        "rng_strategy": {
            "synthetic_framework": "Exact generators imported from context_ablation.py.",
            "environment_and_world_streams": (
                "Original environment indices and SeedSequence logic are retained; "
                "each environment/seed hidden coefficients, MD6, mastery, MRB1, and "
                "evaluator uniforms are generated once and shared across thresholds."
            ),
            "policy_streams": (
                "The deterministic C3 LinTS seed is identical across thresholds "
                "within an environment/seed pair. LinTS samples all five canonical "
                "arms every turn, providing common random numbers while posterior "
                "and decision trajectories may diverge."
            ),
        },
        "context_ablation_script_sha256": _sha256(
            source_dir / "experiments" / "context_ablation.py"
        ),
        "canonical_source_sha256": {
            name: _sha256(source_dir / name) for name in CANONICAL_HASH_FILES
        },
        "threshold_sensitivity_script_sha256": _sha256(script_path),
        "stage_a_audit_manifest": str(stage_a_manifest.relative_to(project_root)).replace("\\", "/"),
        "stage_a_audit_manifest_sha256": _sha256(stage_a_manifest),
        "sanity_checks": sanity_checks,
        "elapsed_seconds": elapsed_seconds,
        "result_file_sha256": result_hashes,
    }


def run_full_experiment(project_root: Path, output_dir: Path) -> None:
    sanity_checks = run_sanity_checks()
    if not all(item["status"] == "PASS" for item in sanity_checks):
        raise RuntimeError("A sanity check failed; full experiment was not started.")

    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output_dir}")
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging results: {stage_dir}")
    stage_dir.mkdir(parents=True)
    (stage_dir / "figures").mkdir()

    raw_path = stage_dir / "raw_attempt_metrics.csv"
    turn_path = stage_dir / "raw_turn_metrics.csv.gz"
    checkpoint_rows_all: list[dict[str, object]] = []
    selected_counts: dict[tuple[str, float, int, str], int] = {}
    eligible_counts: dict[tuple[str, float, int, int], int] = {}
    total_runs = len(ENVIRONMENTS) * len(THRESHOLDS) * len(SEEDS)
    completed_runs = 0
    start = time.perf_counter()

    with raw_path.open("w", encoding="utf-8", newline="") as raw_handle, gzip.open(
        turn_path, "wt", encoding="utf-8", newline="", compresslevel=6
    ) as turn_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=ATTEMPT_COLUMNS)
        turn_writer = csv.DictWriter(turn_handle, fieldnames=TURN_COLUMNS)
        raw_writer.writeheader()
        turn_writer.writeheader()

        for environment in ENVIRONMENTS:
            environment_index = ENVIRONMENT_INDEX[environment]
            for seed in SEEDS:
                hidden = base.generate_hidden_environment(
                    environment, environment_index, seed
                )
                world = base.generate_exogenous_world(
                    environment_index, seed, ATTEMPTS_PER_SEED
                )
                for threshold in THRESHOLDS:
                    attempts, turns, checkpoints = simulate_run(
                        environment,
                        environment_index,
                        threshold,
                        seed,
                        hidden,
                        world,
                        ATTEMPTS_PER_SEED,
                    )
                    raw_writer.writerows(attempts)
                    turn_writer.writerows(turns)
                    checkpoint_rows_all.extend(checkpoints)
                    for row in turns:
                        selected_key = (
                            environment,
                            threshold,
                            seed,
                            str(row["selected_arm"]),
                        )
                        selected_counts[selected_key] = selected_counts.get(selected_key, 0) + 1
                        eligible_key = (
                            environment,
                            threshold,
                            seed,
                            int(row["eligible_arm_count"]),
                        )
                        eligible_counts[eligible_key] = eligible_counts.get(eligible_key, 0) + 1
                    completed_runs += 1
                    if completed_runs % 25 == 0 or completed_runs == total_runs:
                        elapsed = time.perf_counter() - start
                        projected = elapsed * total_runs / completed_runs
                        print(
                            f"progress {completed_runs}/{total_runs}; "
                            f"elapsed={elapsed / 60:.1f} min; "
                            f"projected={projected / 60:.1f} min",
                            flush=True,
                        )

    checkpoint_df = pd.DataFrame(checkpoint_rows_all, columns=CHECKPOINT_COLUMNS)
    checkpoint_df.to_csv(stage_dir / "checkpoint_metrics.csv", index=False)
    summary_df = compute_summary_statistics(checkpoint_df)
    summary_df.to_csv(stage_dir / "summary_statistics.csv", index=False)
    paired_df = compute_paired_comparisons(checkpoint_df)
    paired_df.to_csv(stage_dir / "paired_threshold_comparisons.csv", index=False)
    tradeoff_df = compute_tradeoff_summary(checkpoint_df)
    tradeoff_df.to_csv(stage_dir / "tradeoff_summary.csv", index=False)
    selected_df, eligible_df = compute_distributions(selected_counts, eligible_counts)
    selected_df.to_csv(stage_dir / "selected_arm_distribution.csv", index=False)
    eligible_df.to_csv(stage_dir / "eligible_arm_count_distribution.csv", index=False)

    elapsed_seconds = time.perf_counter() - start
    manifest = _manifest(project_root, stage_dir, sanity_checks, elapsed_seconds)
    with (stage_dir / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    stage_dir.replace(output_dir)
    print(
        f"completed {total_runs} runs in {elapsed_seconds / 60:.1f} minutes; "
        f"results={output_dir}",
        flush=True,
    )


def _count_csv_rows(path: Path, *, compression: str | None = None) -> int:
    total = 0
    for chunk in pd.read_csv(
        path,
        usecols=["environment"],
        chunksize=200_000,
        compression=compression,
    ):
        total += len(chunk)
    return total


def _distribution_counts_from_turn_csv(
    path: Path,
) -> tuple[
    dict[tuple[str, float, int, str], int],
    dict[tuple[str, float, int, int], int],
]:
    selected_counts: dict[tuple[str, float, int, str], int] = {}
    eligible_counts: dict[tuple[str, float, int, int], int] = {}
    columns = [
        "environment",
        "threshold",
        "seed",
        "selected_arm",
        "eligible_arm_count",
    ]
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=200_000,
        compression="gzip",
    ):
        for key, count in chunk.groupby(
            ["environment", "threshold", "seed", "selected_arm"]
        ).size().items():
            normalized = (str(key[0]), float(key[1]), int(key[2]), str(key[3]))
            selected_counts[normalized] = selected_counts.get(normalized, 0) + int(count)
        for key, count in chunk.groupby(
            ["environment", "threshold", "seed", "eligible_arm_count"]
        ).size().items():
            normalized = (str(key[0]), float(key[1]), int(key[2]), int(key[3]))
            eligible_counts[normalized] = eligible_counts.get(normalized, 0) + int(count)
    return selected_counts, eligible_counts


def finalize_staged_experiment(project_root: Path, output_dir: Path) -> None:
    """Finish statistics after a completed simulation-stage interruption."""

    sanity_checks = run_sanity_checks()
    if not all(item["status"] == "PASS" for item in sanity_checks):
        raise RuntimeError("A sanity check failed; staged results were not finalized.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output_dir}")
    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    required = (
        stage_dir / "raw_attempt_metrics.csv",
        stage_dir / "raw_turn_metrics.csv.gz",
        stage_dir / "checkpoint_metrics.csv",
    )
    if not stage_dir.is_dir() or not all(path.is_file() for path in required):
        raise FileNotFoundError("Complete staged raw/checkpoint files were not found.")

    expected_attempt_rows = len(ENVIRONMENTS) * len(THRESHOLDS) * NUM_SEEDS * ATTEMPTS_PER_SEED
    expected_turn_rows = expected_attempt_rows * TURNS_PER_ATTEMPT
    attempt_rows = _count_csv_rows(required[0])
    turn_rows = _count_csv_rows(required[1], compression="gzip")
    if attempt_rows != expected_attempt_rows or turn_rows != expected_turn_rows:
        raise RuntimeError(
            f"Staged raw row count mismatch: attempts={attempt_rows}, turns={turn_rows}."
        )

    checkpoint_df = pd.read_csv(required[2])
    expected_checkpoints = len(ENVIRONMENTS) * len(THRESHOLDS) * NUM_SEEDS * len(CHECKPOINTS)
    if len(checkpoint_df) != expected_checkpoints:
        raise RuntimeError("Staged checkpoint row count mismatch.")
    summary_df = compute_summary_statistics(checkpoint_df)
    summary_df.to_csv(stage_dir / "summary_statistics.csv", index=False)
    paired_df = compute_paired_comparisons(checkpoint_df)
    paired_df.to_csv(stage_dir / "paired_threshold_comparisons.csv", index=False)
    tradeoff_df = compute_tradeoff_summary(checkpoint_df)
    tradeoff_df.to_csv(stage_dir / "tradeoff_summary.csv", index=False)
    selected_counts, eligible_counts = _distribution_counts_from_turn_csv(required[1])
    selected_df, eligible_df = compute_distributions(selected_counts, eligible_counts)
    selected_df.to_csv(stage_dir / "selected_arm_distribution.csv", index=False)
    eligible_df.to_csv(stage_dir / "eligible_arm_count_distribution.csv", index=False)

    start_time = min(path.stat().st_mtime for path in required)
    end_time = max(path.stat().st_mtime for path in required)
    elapsed_seconds = max(0.0, end_time - start_time)
    manifest = _manifest(project_root, stage_dir, sanity_checks, elapsed_seconds)
    manifest["completion_mode"] = "resumed_statistics_from_complete_staged_raw_outputs"
    manifest["raw_row_validation"] = {
        "attempt_rows": attempt_rows,
        "expected_attempt_rows": expected_attempt_rows,
        "turn_rows": turn_rows,
        "expected_turn_rows": expected_turn_rows,
        "checkpoint_rows": len(checkpoint_df),
        "expected_checkpoint_rows": expected_checkpoints,
    }
    with (stage_dir / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    stage_dir.replace(output_dir)
    print(
        f"finalized complete staged simulation results at {output_dir}", flush=True
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity-only", action="store_true")
    mode.add_argument("--run-full", action="store_true")
    mode.add_argument("--resume-staged-analysis", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = _default_project_root()
    output_dir = args.output_dir or (
        project_root / "results" / "self_improvement" / EXPERIMENT_NAME
    )
    try:
        checks = run_sanity_checks()
        print("THRESHOLD SENSITIVITY SANITY CHECKS")
        for item in checks:
            print(f"- {item['check']}: {item['status']}")
        print("OVERALL: PASS")
        if args.run_full:
            run_full_experiment(project_root, output_dir)
        elif args.resume_staged_analysis:
            finalize_staged_experiment(project_root, output_dir)
    except Exception as exc:
        print(f"OVERALL: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
