"""Final simple turn-level LinTS sensitivity suite.

The suite runs three independent, pre-specified studies: reward definition,
delayed-credit allocation, and local one-factor-at-a-time LinTS settings. It
reuses the synthetic generators from ``context_ablation.py`` and deliberately
does not select or recommend a runtime configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.experiments import context_ablation as base
from src.self_improvement.lints_policy import TrueDisjointLinTS
from src.self_improvement.reward import primary_reward
from src.self_improvement.turn_context_builder import MRB1_TASKS, RunningTutorQuality


EXPERIMENT_NAME: Final[str] = "final_sensitivity_suite_v1"
CONTEXT_NAME: Final[str] = "C3"
CONTEXT_FEATURES: Final[tuple[str, ...]] = base.CONTEXT_FEATURES[CONTEXT_NAME]
GAP_THRESHOLD: Final[float] = 0.10
ENVIRONMENTS: Final[tuple[str, ...]] = ("E3_QUALITY", "E4_MIXED")
ENVIRONMENT_INDEX: Final[dict[str, int]] = {
    environment: base.ENVIRONMENTS.index(environment) for environment in ENVIRONMENTS
}
SEEDS: Final[tuple[int, ...]] = base.SEEDS
NUM_SEEDS: Final[int] = len(SEEDS)
ATTEMPTS_PER_SEED: Final[int] = 500
TURNS_PER_ATTEMPT: Final[int] = 3
CHECKPOINTS: Final[tuple[int, ...]] = (100, 500)
NUMERIC_TOLERANCE: Final[float] = 1e-12
BOOTSTRAP_SEED: Final[int] = 314_159
PERMUTATION_SEED: Final[int] = 271_828
NUM_BOOTSTRAP_RESAMPLES: Final[int] = 2_000
NUM_PERMUTATIONS: Final[int] = 50_000

R_LINEAR: Final[str] = "R_LINEAR"
R_SUCCESS_ONLY: Final[str] = "R_SUCCESS_ONLY"
C_EQUAL: Final[str] = "C_EQUAL"
C_RECENCY: Final[str] = "C_RECENCY"
H_BASE: Final[str] = "H_BASE"
H_RIDGE_LOW: Final[str] = "H_RIDGE_LOW"
H_RIDGE_HIGH: Final[str] = "H_RIDGE_HIGH"
H_EXP_LOW: Final[str] = "H_EXP_LOW"
H_EXP_HIGH: Final[str] = "H_EXP_HIGH"

EQUAL_CREDIT: Final[tuple[float, ...]] = (1.0 / 3.0,) * 3
RECENCY_CREDIT: Final[tuple[float, ...]] = (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    study: str
    condition: str
    reward_kind: str
    credit_weights: tuple[float, ...]
    ridge_lambda: float
    exploration_scale: float


STUDY_CONDITIONS: Final[dict[str, tuple[ConditionSpec, ...]]] = {
    "reward": (
        ConditionSpec("reward", R_LINEAR, "linear", EQUAL_CREDIT, 1.0, 0.10),
        ConditionSpec(
            "reward", R_SUCCESS_ONLY, "success_only", EQUAL_CREDIT, 1.0, 0.10
        ),
    ),
    "credit": (
        ConditionSpec("credit", C_EQUAL, "linear", EQUAL_CREDIT, 1.0, 0.10),
        ConditionSpec(
            "credit", C_RECENCY, "linear", RECENCY_CREDIT, 1.0, 0.10
        ),
    ),
    "lints": (
        ConditionSpec("lints", H_BASE, "linear", EQUAL_CREDIT, 1.0, 0.10),
        ConditionSpec(
            "lints", H_RIDGE_LOW, "linear", EQUAL_CREDIT, 0.5, 0.10
        ),
        ConditionSpec(
            "lints", H_RIDGE_HIGH, "linear", EQUAL_CREDIT, 2.0, 0.10
        ),
        ConditionSpec("lints", H_EXP_LOW, "linear", EQUAL_CREDIT, 1.0, 0.05),
        ConditionSpec("lints", H_EXP_HIGH, "linear", EQUAL_CREDIT, 1.0, 0.20),
    ),
}
STUDIES: Final[tuple[str, ...]] = tuple(STUDY_CONDITIONS)

PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "cumulative_regret",
    "mean_evaluator_fraction",
    "optimal_arm_rate",
)

RAW_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "condition",
    "seed",
    "attempt_index",
    "evaluator_score",
    "evaluator_fraction",
    "internal_reward",
    "attempt_regret",
    "cumulative_regret",
    "optimal_arm_rate",
    "p_policy",
    "p_oracle",
    "ridge_lambda",
    "exploration_scale",
    "credit_weights",
)

CHECKPOINT_COLUMNS: Final[tuple[str, ...]] = (
    "study",
    "environment",
    "condition",
    "seed",
    "checkpoint",
    "cumulative_regret",
    "mean_evaluator_fraction",
    "optimal_arm_rate",
)

CANONICAL_HASH_FILES: Final[tuple[str, ...]] = (
    "lints_policy.py",
    "conservative_overlay.py",
    "turn_context_builder.py",
    "turn_level_controller.py",
    "reward.py",
)


def _md6_mapping(probabilities: Sequence[float]) -> dict[str, float]:
    return {
        move: float(probability)
        for move, probability in zip(MOVE_ORDER, probabilities, strict=True)
    }


def _policy_seed(environment_index: int, seed: int, study_index: int) -> int:
    state = base._seed_sequence(  # noqa: SLF001 - exact base seed logic is reused
        environment_index, seed, 67, study_index
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def internal_reward(evaluator_score: int, reward_kind: str) -> float:
    if reward_kind == "linear":
        return primary_reward(evaluator_score)
    if reward_kind == "success_only":
        if evaluator_score not in {0, 1, 2, 3}:
            raise ValueError("evaluator_score must be in {0,1,2,3}.")
        return float(evaluator_score == 3)
    raise ValueError(f"Unknown reward kind: {reward_kind}")


def simulate_run(
    spec: ConditionSpec,
    study_index: int,
    environment: str,
    environment_index: int,
    seed: int,
    hidden: base.HiddenEnvironment,
    world: base.ExogenousWorld,
    attempts: int,
    *,
    assert_invariants: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Simulate one study condition using fixed C3 and threshold 0.10."""

    policy = TrueDisjointLinTS(
        context_dim=len(CONTEXT_FEATURES),
        ridge_lambda=spec.ridge_lambda,
        exploration_scale=spec.exploration_scale,
        seed=_policy_seed(environment_index, seed, study_index),
        data_mode="synthetic",
    )
    raw_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    cumulative_regret = 0.0
    cumulative_evaluator_fraction = 0.0
    cumulative_optimal_turns = 0

    for attempt_zero in range(attempts):
        attempt_index = attempt_zero + 1
        mastery = float(world.mastery_before[attempt_zero])
        running_quality = RunningTutorQuality()
        updates_at_start = policy.total_updates
        pending_updates: list[tuple[str, np.ndarray, float]] = []
        selected_utilities: list[float] = []
        oracle_utilities: list[float] = []
        optimal_indicators: list[int] = []

        for turn_zero in range(TURNS_PER_ATTEMPT):
            md6_array = world.md6_probabilities[attempt_zero, turn_zero]
            md6 = _md6_mapping(md6_array)
            quality_before = running_quality.snapshot()
            context = base.context_vector(CONTEXT_NAME, md6, mastery, quality_before)
            candidates = eligible_arms(md6, GAP_THRESHOLD)
            decision = policy.select_arm(context, candidates)
            selected_arm = decision["selected_arm"]
            if assert_invariants:
                assert selected_arm in candidates
                assert policy.total_updates == updates_at_start

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
            if oracle_utility - selected_utility < -NUMERIC_TOLERANCE:
                raise AssertionError("Eligible-oracle turn regret became negative.")
            selected_utilities.append(selected_utility)
            oracle_utilities.append(oracle_utility)
            optimal_indicators.append(int(selected_arm == oracle_arm))
            pending_updates.append(
                (selected_arm, context.copy(), spec.credit_weights[turn_zero])
            )

            scores = world.mrb1_scores[attempt_zero, turn_zero]
            running_quality.add_scores(
                {
                    task: float(score)
                    for task, score in zip(MRB1_TASKS, scores, strict=True)
                }
            )
            if assert_invariants:
                assert policy.total_updates == updates_at_start

        p_policy = float(np.mean(selected_utilities))
        p_oracle = float(np.mean(oracle_utilities))
        attempt_regret = p_oracle - p_policy
        if attempt_regret < -NUMERIC_TOLERANCE:
            raise AssertionError("Eligible-oracle attempt regret became negative.")
        attempt_regret = max(0.0, float(attempt_regret))
        evaluator_score = int(
            np.count_nonzero(world.evaluator_uniforms[attempt_zero] < p_policy)
        )
        evaluator_fraction = primary_reward(evaluator_score)
        update_reward = internal_reward(evaluator_score, spec.reward_kind)
        if assert_invariants:
            assert evaluator_score in {0, 1, 2, 3}
            assert evaluator_fraction in {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
            assert policy.total_updates == updates_at_start
            assert math.isclose(
                sum(weight for _, _, weight in pending_updates),
                1.0,
                abs_tol=NUMERIC_TOLERANCE,
            )
        for arm, context, sample_weight in pending_updates:
            policy.update(
                arm, context, update_reward, sample_weight=sample_weight
            )
        if assert_invariants:
            assert policy.total_updates == updates_at_start + TURNS_PER_ATTEMPT

        cumulative_regret += attempt_regret
        cumulative_evaluator_fraction += evaluator_fraction
        cumulative_optimal_turns += sum(optimal_indicators)
        attempt_optimal_rate = float(np.mean(optimal_indicators))
        raw_rows.append(
            {
                "environment": environment,
                "condition": spec.condition,
                "seed": seed,
                "attempt_index": attempt_index,
                "evaluator_score": evaluator_score,
                "evaluator_fraction": evaluator_fraction,
                "internal_reward": update_reward,
                "attempt_regret": attempt_regret,
                "cumulative_regret": cumulative_regret,
                "optimal_arm_rate": attempt_optimal_rate,
                "p_policy": p_policy,
                "p_oracle": p_oracle,
                "ridge_lambda": spec.ridge_lambda,
                "exploration_scale": spec.exploration_scale,
                "credit_weights": "|".join(
                    f"{weight:.17g}" for weight in spec.credit_weights
                ),
            }
        )
        if attempt_index in CHECKPOINTS:
            checkpoint_rows.append(
                {
                    "study": spec.study,
                    "environment": environment,
                    "condition": spec.condition,
                    "seed": seed,
                    "checkpoint": attempt_index,
                    "cumulative_regret": cumulative_regret,
                    "mean_evaluator_fraction": (
                        cumulative_evaluator_fraction / attempt_index
                    ),
                    "optimal_arm_rate": (
                        cumulative_optimal_turns
                        / (attempt_index * TURNS_PER_ATTEMPT)
                    ),
                }
            )
    return raw_rows, checkpoint_rows


def run_sanity_checks() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def passed(name: str) -> None:
        checks.append({"check": name, "status": "PASS"})

    md6 = _md6_mapping([0.31, 0.28, 0.23, 0.18])
    empty = RunningTutorQuality()
    low = base.context_vector(CONTEXT_NAME, md6, 0.1, empty)
    high = base.context_vector(CONTEXT_NAME, md6, 0.9, empty)
    assert low.shape == (9,) and len(CONTEXT_FEATURES) == 9
    passed("context dimension equals 9")
    assert "mastery_before" not in CONTEXT_FEATURES
    np.testing.assert_array_equal(low, high)
    passed("mastery absent from policy context")
    assert GAP_THRESHOLD == 0.10
    passed("gap threshold fixed at 0.10")

    assert [internal_reward(score, "linear") for score in range(4)] == [
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0,
    ]
    passed("R_LINEAR mapping exact")
    assert [internal_reward(score, "success_only") for score in range(4)] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    passed("R_SUCCESS_ONLY mapping exact")
    assert EQUAL_CREDIT == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    assert math.isclose(sum(EQUAL_CREDIT), 1.0, abs_tol=NUMERIC_TOLERANCE)
    passed("equal credit weights sum to one")
    assert RECENCY_CREDIT == (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    assert math.isclose(sum(RECENCY_CREDIT), 1.0, abs_tol=NUMERIC_TOLERANCE)
    passed("recency credit weights exact and sum to one")

    expected_hyperparameters = {
        H_BASE: (1.0, 0.10),
        H_RIDGE_LOW: (0.5, 0.10),
        H_RIDGE_HIGH: (2.0, 0.10),
        H_EXP_LOW: (1.0, 0.05),
        H_EXP_HIGH: (1.0, 0.20),
    }
    observed_hyperparameters = {
        spec.condition: (spec.ridge_lambda, spec.exploration_scale)
        for spec in STUDY_CONDITIONS["lints"]
    }
    assert observed_hyperparameters == expected_hyperparameters
    passed("hyperparameter configurations exactly match specification")

    environment = "E4_MIXED"
    environment_index = ENVIRONMENT_INDEX[environment]
    hidden = base.generate_hidden_environment(environment, environment_index, 7)
    world = base.generate_exogenous_world(environment_index, 7, 6)
    spec = STUDY_CONDITIONS["reward"][0]
    first = simulate_run(
        spec, 0, environment, environment_index, 7, hidden, world, 6
    )
    raw_rows, checkpoints = first
    # Eligibility and update-timing assertions execute at every simulated turn.
    passed("policy selected arm always eligible")
    passed("oracle arm always eligible")
    assert all(float(row["attempt_regret"]) >= -NUMERIC_TOLERANCE for row in raw_rows)
    passed("regret nonnegative")
    assert all(int(row["evaluator_score"]) in {0, 1, 2, 3} for row in raw_rows)
    passed("evaluator score support exact")
    assert all(
        float(row["evaluator_fraction"])
        in {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
        for row in raw_rows
    )
    passed("evaluator fraction support exact")
    passed("no posterior update before attempt completion")
    second = simulate_run(
        spec, 0, environment, environment_index, 7, hidden, world, 6
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    passed("deterministic rerun succeeds")
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
    boot = weights @ np.asarray(values, dtype=np.float64)
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
    while completed < NUM_PERMUTATIONS:
        batch = min(2_000, NUM_PERMUTATIONS - completed)
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


def compute_summary(checkpoints: pd.DataFrame, study: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = _bootstrap_weights(NUM_SEEDS)
    subset = checkpoints[checkpoints["study"] == study]
    for (environment, condition, checkpoint), group in subset.groupby(
        ["environment", "condition", "checkpoint"], sort=False
    ):
        ordered = group.sort_values("seed")
        if len(ordered) != NUM_SEEDS:
            raise RuntimeError("Summary group does not contain all paired seeds.")
        row: dict[str, object] = {
            "environment": environment,
            "condition": condition,
            "checkpoint": int(checkpoint),
            "n_seeds": len(ordered),
        }
        for metric in PRIMARY_METRICS:
            values = ordered[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(values, weights)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_pairs(study: str) -> tuple[tuple[str, str], ...]:
    if study == "reward":
        return ((R_LINEAR, R_SUCCESS_ONLY),)
    if study == "credit":
        return ((C_EQUAL, C_RECENCY),)
    return (
        (H_RIDGE_LOW, H_BASE),
        (H_RIDGE_HIGH, H_BASE),
        (H_EXP_LOW, H_BASE),
        (H_EXP_HIGH, H_BASE),
    )


def compute_paired_comparisons(
    checkpoints: pd.DataFrame, study: str
) -> pd.DataFrame:
    endpoint = checkpoints[
        (checkpoints["study"] == study)
        & (checkpoints["checkpoint"] == ATTEMPTS_PER_SEED)
    ]
    weights = _bootstrap_weights(NUM_SEEDS)
    rows: list[dict[str, object]] = []
    stream_index = STUDIES.index(study) * 100
    for environment in ENVIRONMENTS:
        env_rows = endpoint[endpoint["environment"] == environment]
        for metric in PRIMARY_METRICS:
            family_indices: list[int] = []
            pivot = env_rows.pivot(
                index="seed", columns="condition", values=metric
            ).sort_index()
            for condition_a, condition_b in _comparison_pairs(study):
                values_a = pivot[condition_a].to_numpy(dtype=np.float64)
                values_b = pivot[condition_b].to_numpy(dtype=np.float64)
                differences = values_a - values_b
                low, high = _bootstrap_mean_ci(differences, weights)
                p_value = _paired_permutation_p(differences, stream_index)
                stream_index += 1
                family_indices.append(len(rows))
                rows.append(
                    {
                        "environment": environment,
                        "comparison": f"{condition_a}-{condition_b}",
                        "checkpoint": ATTEMPTS_PER_SEED,
                        "metric_name": metric,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "n_pairs": len(differences),
                        "mean_a": float(np.mean(values_a)),
                        "mean_b": float(np.mean(values_b)),
                        "mean_paired_difference": float(np.mean(differences)),
                        "median_paired_difference": float(np.median(differences)),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "permutation_p": p_value,
                        "holm_adjusted_p": float("nan"),
                    }
                )
            if study == "lints":
                adjusted = _holm_adjust(
                    rows[index]["permutation_p"] for index in family_indices
                )
                for index, value in zip(family_indices, adjusted, strict=True):
                    rows[index]["holm_adjusted_p"] = value
    return pd.DataFrame(rows)


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
    experiment_script = Path(__file__).resolve()
    plot_script = source_dir / "analysis" / "plot_final_sensitivity_suite.py"
    result_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    return {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_timezone": "UTC",
        "git_commit": _git_commit(project_root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "base_seed": base.BASE_SEED,
        "seed_list": list(SEEDS),
        "environments": list(ENVIRONMENTS),
        "attempts_per_seed": ATTEMPTS_PER_SEED,
        "turns_per_attempt": TURNS_PER_ATTEMPT,
        "checkpoints": list(CHECKPOINTS),
        "total_runs": sum(len(values) for values in STUDY_CONDITIONS.values())
        * len(ENVIRONMENTS)
        * NUM_SEEDS,
        "total_attempts": 450_000,
        "total_turns": 1_350_000,
        "context": {
            "name": CONTEXT_NAME,
            "dimension": len(CONTEXT_FEATURES),
            "features": list(CONTEXT_FEATURES),
            "mastery_before_observed_by_policy": False,
        },
        "gap_threshold": GAP_THRESHOLD,
        "reward_conditions": {
            R_LINEAR: "evaluator_score / 3",
            R_SUCCESS_ONLY: "1 if evaluator_score == 3 else 0",
        },
        "credit_conditions": {
            C_EQUAL: list(EQUAL_CREDIT),
            C_RECENCY: list(RECENCY_CREDIT),
        },
        "lints_conditions": {
            spec.condition: {
                "ridge_lambda": spec.ridge_lambda,
                "exploration_scale": spec.exploration_scale,
            }
            for spec in STUDY_CONDITIONS["lints"]
        },
        "external_evaluation_metrics": list(PRIMARY_METRICS),
        "bootstrap_method": (
            "Nonparametric percentile bootstrap across the 50 paired seed units; "
            "attempts are never resampled as independent units."
        ),
        "bootstrap_resamples": NUM_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_method": (
            "Two-sided Monte Carlo paired sign-flip test of the seed-level mean "
            "difference, with +1 correction."
        ),
        "permutation_resamples": NUM_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "paired_difference_definition": "metric(condition_a) - metric(condition_b)",
        "holm_correction_scope": (
            "Study C only: four pre-specified alternative-minus-H_BASE "
            "comparisons within each environment and metric."
        ),
        "rng_strategy": {
            "worlds": (
                "Exact context_ablation.py hidden-environment and exogenous-world "
                "generators with original environment indices and seeds; each "
                "environment/seed world is shared across all nine suite conditions."
            ),
            "policy": (
                "One deterministic SeedSequence-derived LinTS stream per study, "
                "environment, and seed, shared across conditions within that study."
            ),
        },
        "sanity_checks": sanity_checks,
        "elapsed_seconds": elapsed_seconds,
        "canonical_source_sha256": {
            name: _sha256(source_dir / name) for name in CANONICAL_HASH_FILES
        },
        "reused_context_ablation_sha256": _sha256(
            source_dir / "experiments" / "context_ablation.py"
        ),
        "experiment_source_sha256": {
            "final_sensitivity_suite.py": _sha256(experiment_script),
            "plot_final_sensitivity_suite.py": _sha256(plot_script),
        },
        "result_file_sha256": result_hashes,
    }


def run_full_experiment(project_root: Path, output_dir: Path) -> None:
    sanity_checks = run_sanity_checks()
    if not all(item["status"] == "PASS" for item in sanity_checks):
        raise RuntimeError("A sanity check failed; full suite was not started.")
    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    if output_dir.exists() or stage_dir.exists():
        raise FileExistsError("Refusing to overwrite existing or staged suite results.")
    for study in STUDIES:
        (stage_dir / study / "figures").mkdir(parents=True)

    checkpoint_rows: list[dict[str, object]] = []
    total_runs = 900
    completed_runs = 0
    start = time.perf_counter()
    with ExitStack() as stack:
        writers: dict[str, csv.DictWriter] = {}
        for study in STUDIES:
            handle = stack.enter_context(
                (stage_dir / study / "raw_attempt_metrics.csv").open(
                    "w", encoding="utf-8", newline=""
                )
            )
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            writers[study] = writer

        for environment in ENVIRONMENTS:
            environment_index = ENVIRONMENT_INDEX[environment]
            for seed in SEEDS:
                hidden = base.generate_hidden_environment(
                    environment, environment_index, seed
                )
                world = base.generate_exogenous_world(
                    environment_index, seed, ATTEMPTS_PER_SEED
                )
                for study_index, study in enumerate(STUDIES):
                    for spec in STUDY_CONDITIONS[study]:
                        raw_rows, checkpoints = simulate_run(
                            spec,
                            study_index,
                            environment,
                            environment_index,
                            seed,
                            hidden,
                            world,
                            ATTEMPTS_PER_SEED,
                        )
                        writers[study].writerows(raw_rows)
                        checkpoint_rows.extend(checkpoints)
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

    checkpoint_df = pd.DataFrame(checkpoint_rows, columns=CHECKPOINT_COLUMNS)
    for study in STUDIES:
        compute_summary(checkpoint_df, study).to_csv(
            stage_dir / study / "summary.csv", index=False
        )
        comparison_name = (
            "paired_comparisons.csv" if study == "lints" else "paired_comparison.csv"
        )
        compute_paired_comparisons(checkpoint_df, study).to_csv(
            stage_dir / study / comparison_name, index=False
        )

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


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity-only", action="store_true")
    mode.add_argument("--run-full", action="store_true")
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
        print("FINAL SENSITIVITY SUITE SANITY CHECKS")
        for item in checks:
            print(f"- {item['check']}: {item['status']}")
        print("OVERALL: PASS")
        if args.run_full:
            run_full_experiment(project_root, output_dir)
    except Exception as exc:
        print(f"OVERALL: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
