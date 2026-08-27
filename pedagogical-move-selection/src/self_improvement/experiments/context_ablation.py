"""Pre-specified turn-level LinTS context ablation (context_ablation_v1).

This module creates synthetic decision environments, runs the fixed C0--C4
ablation, and writes raw and neutral statistical outputs.  It deliberately
does not select, recommend, or name a preferred context representation.

The canonical runtime modules are imported and reused but never modified.
Run the inexpensive deterministic checks with::

    python -m src.self_improvement.experiments.context_ablation --sanity-only

Run the complete pre-specified experiment with::

    python -m src.self_improvement.experiments.context_ablation --run-full
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.self_improvement.conservative_overlay import (
    DEFAULT_GAP_THRESHOLD,
    apply_conservative_overlay,
    eligible_arms,
)
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.lints_policy import ARMS, TrueDisjointLinTS
from src.self_improvement.reward import primary_reward
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    RunningTutorQuality,
    TutorQualitySnapshot,
    build_turn_context,
)


EXPERIMENT_NAME: Final[str] = "context_ablation_v1"
BASE_SEED: Final[int] = 20260823
BOOTSTRAP_SEED: Final[int] = 814730
PERMUTATION_SEED: Final[int] = 991827
NUM_BOOTSTRAP_RESAMPLES: Final[int] = 2_000
NUM_PERMUTATIONS: Final[int] = 50_000
NUM_SEEDS: Final[int] = 50
SEEDS: Final[tuple[int, ...]] = tuple(range(NUM_SEEDS))
ATTEMPTS_PER_SEED: Final[int] = 500
TURNS_PER_ATTEMPT: Final[int] = 3
CHECKPOINTS: Final[tuple[int, ...]] = (25, 50, 100, 250, 500)
RIDGE_LAMBDA: Final[float] = 1.0
EXPLORATION_SCALE: Final[float] = 0.10
GAP_THRESHOLD: Final[float] = DEFAULT_GAP_THRESHOLD
NUMERIC_TOLERANCE: Final[float] = 1e-12

CONTEXT_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "C0": ("constant",),
    "C1": (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
    ),
    "C2": (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
        "mastery_before",
    ),
    "C3": (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
        "running_mistake_identification",
        "running_mistake_location",
        "running_providing_guidance",
        "running_actionability",
        "has_within_attempt_quality",
    ),
    "C4": (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
        "mastery_before",
        "running_mistake_identification",
        "running_mistake_location",
        "running_providing_guidance",
        "running_actionability",
        "has_within_attempt_quality",
    ),
}
CONTEXTS: Final[tuple[str, ...]] = tuple(CONTEXT_FEATURES)

ENVIRONMENT_DEFINITIONS: Final[dict[str, str]] = {
    "E0_NO_CONTEXT": "Arm identity only; all observed state is irrelevant.",
    "E1_MD6": "Arm identity and current MD6 probability state only.",
    "E2_MASTERY": "Arm identity, current MD6 state, and mastery_before.",
    "E3_QUALITY": (
        "Arm identity, current MD6 state, and prior-turn running MRB1 quality."
    ),
    "E4_MIXED": (
        "Arm identity, current MD6 state, mastery_before, and prior-turn "
        "running MRB1 quality, with weak sparse interactions."
    ),
}
ENVIRONMENTS: Final[tuple[str, ...]] = tuple(ENVIRONMENT_DEFINITIONS)

# Pre-specified feature-source comparisons.  Each tuple is
# (environment, label, simpler, richer).
PRIMARY_COMPARISONS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("E2_MASTERY", "C2-C1", "C1", "C2"),
    ("E4_MIXED", "C2-C1", "C1", "C2"),
    ("E3_QUALITY", "C3-C1", "C1", "C3"),
    ("E4_MIXED", "C3-C1", "C1", "C3"),
    ("E3_QUALITY", "C4-C2", "C2", "C4"),
    ("E4_MIXED", "C4-C2", "C2", "C4"),
    ("E2_MASTERY", "C4-C3", "C3", "C4"),
    ("E4_MIXED", "C4-C3", "C3", "C4"),
)

CANONICAL_HASH_FILES: Final[tuple[str, ...]] = (
    "lints_policy.py",
    "conservative_overlay.py",
    "turn_context_builder.py",
    "turn_level_controller.py",
    "reward.py",
)


@dataclass(frozen=True, slots=True)
class HiddenEnvironment:
    """Hidden, seed-specific bounded arm utility parameters."""

    name: str
    arm_intercept: np.ndarray
    md6_coefficients: np.ndarray
    mastery_coefficients: np.ndarray
    quality_coefficients: np.ndarray
    interaction_coefficients: np.ndarray


@dataclass(frozen=True, slots=True)
class ExogenousWorld:
    """Common random-number world shared by C0--C4 for one env/seed."""

    md6_probabilities: np.ndarray
    mastery_before: np.ndarray
    mrb1_scores: np.ndarray
    evaluator_uniforms: np.ndarray


ATTEMPT_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "context_condition",
    "seed",
    "attempt_index",
    "attempt_regret",
    "cumulative_regret",
    "p_policy",
    "p_oracle",
    "evaluator_score",
    "reward",
    "optimal_arm_rate_for_attempt",
    "oracle_arm_selection_indicator",
    "eligible_arm_mean_count",
    "eligible_arm_counts",
    "override_opportunity_count",
    "override_opportunity_rate",
    "policy_intervention_count",
    "policy_intervention_rate",
    "selected_arm",
    "selected_arm_turn_1",
    "selected_arm_turn_2",
    "selected_arm_turn_3",
    "oracle_arm_turn_1",
    "oracle_arm_turn_2",
    "oracle_arm_turn_3",
    "selected_final_moves",
)

TURN_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "context_condition",
    "seed",
    "attempt_index",
    "turn_index",
    "mastery_before",
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "running_mistake_identification",
    "running_mistake_location",
    "running_providing_guidance",
    "running_actionability",
    "has_within_attempt_quality",
    "selected_arm",
    "oracle_arm",
    "selected_is_oracle",
    "selected_hidden_utility",
    "oracle_hidden_utility",
    "turn_regret",
    "eligible_arms",
    "eligible_arm_count",
    "override_opportunity",
    "selected_final_move",
    "selected_overlay_overridden",
    "mrb1_mistake_identification_after_turn",
    "mrb1_mistake_location_after_turn",
    "mrb1_providing_guidance_after_turn",
    "mrb1_actionability_after_turn",
)

CHECKPOINT_COLUMNS: Final[tuple[str, ...]] = (
    "environment",
    "context_condition",
    "seed",
    "checkpoint",
    "cumulative_regret",
    "mean_regret",
    "mean_reward",
    "optimal_arm_rate",
    "eligible_arm_mean_count",
    "override_opportunity_rate",
    "policy_intervention_rate",
)


def _seed_sequence(*components: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([BASE_SEED, *components])


def _policy_seed(environment_index: int, seed: int, context_index: int) -> int:
    state = _seed_sequence(environment_index, seed, 91, context_index).generate_state(
        1, dtype=np.uint32
    )
    return int(state[0])


def _sigmoid_tanh(latent: float) -> float:
    """Map a moderate latent score strictly inside [0, 1]."""

    return float(0.5 + 0.24 * np.tanh(latent))


def generate_hidden_environment(
    environment: str,
    environment_index: int,
    seed: int,
) -> HiddenEnvironment:
    """Generate fixed bounded arm-specific parameters before any policy run.

    Coefficients are generated once for an environment/seed and shared by all
    contexts.  Sparse zero entries and weak interactions keep E4 from being a
    mechanical direct sum of all C4 dimensions.
    """

    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    rng = np.random.default_rng(
        _seed_sequence(environment_index, seed, 11)
    )

    # Baseline remains competitive while at least one bias has a larger fixed
    # identity component.  Random permutation prevents a fixed favored bias.
    bias_template = rng.permutation(np.asarray([-0.08, -0.01, 0.06, 0.13]))
    intercept = np.concatenate(
        ([rng.uniform(-0.035, 0.035)], bias_template + rng.uniform(-0.015, 0.015, 4))
    )
    intercept = np.clip(intercept, -0.15, 0.15)

    md6 = rng.uniform(-0.62, 0.62, size=(len(ARMS), len(MOVE_ORDER)))
    md6 -= md6.mean(axis=1, keepdims=True)
    for arm_index in range(len(ARMS)):
        md6[arm_index, (arm_index + 2) % len(MOVE_ORDER)] = 0.0
    # Modest target-aligned structure plus bounded random heterogeneity.
    for arm_index in range(1, len(ARMS)):
        target_index = arm_index - 1
        md6[arm_index, target_index] += 0.28
    md6 = np.clip(md6, -0.75, 0.75)

    mastery = rng.uniform(-0.55, 0.55, size=len(ARMS))
    mastery[0] = 0.0
    mastery[1 + (seed % 4)] = 0.0

    quality = rng.uniform(-0.48, 0.48, size=(len(ARMS), len(MRB1_TASKS)))
    sparse_mask = np.fromfunction(
        lambda arm, task: ((arm + 2 * task + seed) % 4) == 0,
        quality.shape,
        dtype=int,
    )
    quality[sparse_mask] = 0.0

    interactions = rng.uniform(-0.30, 0.30, size=(len(ARMS), 2))
    interactions[::2, 1] = 0.0

    return HiddenEnvironment(
        name=environment,
        arm_intercept=intercept.astype(np.float64),
        md6_coefficients=md6.astype(np.float64),
        mastery_coefficients=mastery.astype(np.float64),
        quality_coefficients=quality.astype(np.float64),
        interaction_coefficients=interactions.astype(np.float64),
    )


def generate_exogenous_world(
    environment_index: int,
    seed: int,
    attempts: int,
) -> ExogenousWorld:
    """Generate controlled independent exogenous streams for one world."""

    stream_root = _seed_sequence(environment_index, seed, 23)
    md6_ss, mastery_ss, quality_base_ss, quality_noise_ss, evaluator_ss = (
        stream_root.spawn(5)
    )
    md6_rng = np.random.default_rng(md6_ss)
    mastery_rng = np.random.default_rng(mastery_ss)
    quality_base_rng = np.random.default_rng(quality_base_ss)
    quality_noise_rng = np.random.default_rng(quality_noise_ss)
    evaluator_rng = np.random.default_rng(evaluator_ss)

    # Fixed before execution and never tuned from outcomes.
    dirichlet_alpha = np.asarray([4.5, 5.0, 5.5, 6.0], dtype=np.float64)
    md6 = md6_rng.dirichlet(
        dirichlet_alpha, size=attempts * TURNS_PER_ATTEMPT
    ).reshape(attempts, TURNS_PER_ATTEMPT, len(MOVE_ORDER))
    mastery = mastery_rng.beta(2.2, 2.2, size=attempts).astype(np.float64)

    attempt_quality = quality_base_rng.beta(
        2.4, 2.2, size=(attempts, len(MRB1_TASKS))
    )
    turn_quality = quality_noise_rng.beta(
        2.0,
        2.0,
        size=(attempts, TURNS_PER_ATTEMPT, len(MRB1_TASKS)),
    )
    task_offsets = np.asarray([-0.035, 0.015, 0.035, -0.015])
    mrb1 = (
        0.64 * attempt_quality[:, None, :]
        + 0.36 * turn_quality
        + task_offsets[None, None, :]
    )
    mrb1 = np.clip(mrb1, 0.0, 1.0).astype(np.float64)

    # Three shared Bernoulli uniforms make sum(U_i < p) exactly Binomial(3,p)
    # while coupling evaluator noise across context conditions.
    evaluator_uniforms = evaluator_rng.random(
        (attempts, 3), dtype=np.float64
    )
    return ExogenousWorld(
        md6_probabilities=md6.astype(np.float64),
        mastery_before=mastery,
        mrb1_scores=mrb1,
        evaluator_uniforms=evaluator_uniforms,
    )


def context_vector(
    condition: str,
    md6_mapping: Mapping[str, float],
    mastery_before: float,
    running_quality: RunningTutorQuality | TutorQualitySnapshot,
) -> np.ndarray:
    """Project the canonical 10-D context into one pre-specified condition."""

    if condition not in CONTEXT_FEATURES:
        raise ValueError(f"Unknown context condition: {condition}")
    full = build_turn_context(md6_mapping, mastery_before, running_quality)
    if condition == "C0":
        result = np.asarray([1.0], dtype=np.float64)
    elif condition == "C1":
        result = full[:4].copy()
    elif condition == "C2":
        result = full[:5].copy()
    elif condition == "C3":
        result = np.concatenate((full[:4], full[5:])).astype(
            np.float64, copy=False
        )
    else:
        result = full.copy()
    expected = (len(CONTEXT_FEATURES[condition]),)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(
            f"Invalid {condition} context: shape={result.shape}, expected={expected}."
        )
    return result


def hidden_arm_utility(
    hidden: HiddenEnvironment,
    arm: str,
    md6_probabilities: Sequence[float],
    mastery_before: float,
    running_quality: TutorQualitySnapshot,
) -> float:
    """Return simulator/oracle-only expected utility for one eligible arm."""

    arm_index = ARMS.index(arm)
    md6 = np.asarray(md6_probabilities, dtype=np.float64)
    centered_md6 = md6 - 0.25
    mastery_centered = float(mastery_before - 0.5)
    quality = np.asarray(running_quality.feature_values, dtype=np.float64)
    availability = running_quality.has_within_attempt_quality
    centered_quality = quality - 0.5

    latent = float(hidden.arm_intercept[arm_index])
    if hidden.name != "E0_NO_CONTEXT":
        latent += float(hidden.md6_coefficients[arm_index] @ centered_md6)
    if hidden.name in {"E2_MASTERY", "E4_MIXED"}:
        latent += float(
            hidden.mastery_coefficients[arm_index] * mastery_centered
        )
    if hidden.name in {"E3_QUALITY", "E4_MIXED"}:
        latent += float(
            availability
            * (hidden.quality_coefficients[arm_index] @ centered_quality)
        )
    if hidden.name == "E4_MIXED":
        # Weak, sparse interactions prevent exact linear alignment with C4.
        md6_contrast = float(md6[0] - md6[3])
        quality_contrast = float(quality[0] - quality[3])
        latent += float(
            0.14
            * hidden.interaction_coefficients[arm_index, 0]
            * mastery_centered
            * md6_contrast
        )
        latent += float(
            0.12
            * hidden.interaction_coefficients[arm_index, 1]
            * availability
            * quality_contrast
            * md6_contrast
        )

    utility = _sigmoid_tanh(latent)
    if not 0.0 <= utility <= 1.0 or not math.isfinite(utility):
        raise RuntimeError("Hidden utility left [0, 1] or became non-finite.")
    return utility


def _md6_mapping(probabilities: Sequence[float]) -> dict[str, float]:
    return {
        move: float(probability)
        for move, probability in zip(MOVE_ORDER, probabilities, strict=True)
    }


def simulate_run(
    environment: str,
    environment_index: int,
    condition: str,
    seed: int,
    hidden: HiddenEnvironment,
    world: ExogenousWorld,
    attempts: int,
    *,
    assert_invariants: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Simulate one environment/context/seed run using canonical LinTS."""

    policy = TrueDisjointLinTS(
        context_dim=len(CONTEXT_FEATURES[condition]),
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=_policy_seed(environment_index, seed, CONTEXTS.index(condition)),
        data_mode="synthetic",
    )
    attempt_rows: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    cumulative_regret = 0.0
    cumulative_reward = 0.0
    cumulative_optimal_turns = 0
    cumulative_eligible_count = 0
    cumulative_override_opportunities = 0
    cumulative_policy_interventions = 0

    for attempt_zero in range(attempts):
        attempt_index = attempt_zero + 1
        mastery = float(world.mastery_before[attempt_zero])
        running_quality = RunningTutorQuality()
        updates_at_attempt_start = policy.total_updates
        pending_updates: list[tuple[str, np.ndarray]] = []
        selected_utilities: list[float] = []
        oracle_utilities: list[float] = []
        selected_arms: list[str] = []
        oracle_arms: list[str] = []
        selected_final_moves: list[str] = []
        optimal_indicators: list[int] = []
        eligible_counts: list[int] = []
        override_opportunities: list[int] = []
        interventions: list[int] = []

        for turn_zero in range(TURNS_PER_ATTEMPT):
            turn_index = turn_zero + 1
            md6_array = world.md6_probabilities[attempt_zero, turn_zero]
            md6 = _md6_mapping(md6_array)
            quality_before = running_quality.snapshot()
            x = context_vector(condition, md6, mastery, quality_before)
            candidates = eligible_arms(md6, GAP_THRESHOLD)
            decision = policy.select_arm(x, candidates)
            selected_arm = decision["selected_arm"]
            if assert_invariants:
                assert selected_arm in candidates
                assert policy.total_updates == updates_at_attempt_start

            utilities = {
                arm: hidden_arm_utility(
                    hidden, arm, md6_array, mastery, quality_before
                )
                for arm in candidates
            }
            # candidates are in canonical order, so max has deterministic ties.
            oracle_arm = max(candidates, key=utilities.__getitem__)
            selected_utility = float(utilities[selected_arm])
            oracle_utility = float(utilities[oracle_arm])
            turn_regret = oracle_utility - selected_utility
            if turn_regret < -NUMERIC_TOLERANCE:
                raise AssertionError("Eligible oracle regret became negative.")
            turn_regret = max(0.0, float(turn_regret))

            overlay = apply_conservative_overlay(
                selected_arm, md6, GAP_THRESHOLD
            )
            current_scores = world.mrb1_scores[attempt_zero, turn_zero]
            score_mapping = {
                task: float(score)
                for task, score in zip(MRB1_TASKS, current_scores, strict=True)
            }

            selected_arms.append(selected_arm)
            oracle_arms.append(oracle_arm)
            selected_final_moves.append(overlay.final_move)
            selected_utilities.append(selected_utility)
            oracle_utilities.append(oracle_utility)
            optimal_indicators.append(int(selected_arm == oracle_arm))
            eligible_counts.append(len(candidates))
            override_opportunities.append(int(len(candidates) > 1))
            interventions.append(int(overlay.overridden))
            pending_updates.append((selected_arm, x.copy()))

            turn_rows.append(
                {
                    "environment": environment,
                    "context_condition": condition,
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "turn_index": turn_index,
                    "mastery_before": mastery,
                    "md6_p_generic": float(md6_array[0]),
                    "md6_p_probing": float(md6_array[1]),
                    "md6_p_focus": float(md6_array[2]),
                    "md6_p_telling": float(md6_array[3]),
                    "running_mistake_identification": quality_before.feature_values[0],
                    "running_mistake_location": quality_before.feature_values[1],
                    "running_providing_guidance": quality_before.feature_values[2],
                    "running_actionability": quality_before.feature_values[3],
                    "has_within_attempt_quality": quality_before.has_within_attempt_quality,
                    "selected_arm": selected_arm,
                    "oracle_arm": oracle_arm,
                    "selected_is_oracle": int(selected_arm == oracle_arm),
                    "selected_hidden_utility": selected_utility,
                    "oracle_hidden_utility": oracle_utility,
                    "turn_regret": turn_regret,
                    "eligible_arms": "|".join(candidates),
                    "eligible_arm_count": len(candidates),
                    "override_opportunity": int(len(candidates) > 1),
                    "selected_final_move": overlay.final_move,
                    "selected_overlay_overridden": int(overlay.overridden),
                    "mrb1_mistake_identification_after_turn": float(current_scores[0]),
                    "mrb1_mistake_location_after_turn": float(current_scores[1]),
                    "mrb1_providing_guidance_after_turn": float(current_scores[2]),
                    "mrb1_actionability_after_turn": float(current_scores[3]),
                }
            )

            # Current response quality becomes available only after selection.
            running_quality.add_scores(score_mapping)
            if assert_invariants:
                assert policy.total_updates == updates_at_attempt_start

        p_policy = float(np.mean(selected_utilities))
        p_oracle = float(np.mean(oracle_utilities))
        attempt_regret = p_oracle - p_policy
        if attempt_regret < -NUMERIC_TOLERANCE:
            raise AssertionError("Attempt oracle regret became negative.")
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
            raise AssertionError("Delayed sample weights do not total one.")
        for arm, x in pending_updates:
            policy.update(arm, x, reward, sample_weight=sample_weight)
        if assert_invariants:
            assert policy.total_updates == (
                updates_at_attempt_start + TURNS_PER_ATTEMPT
            )

        cumulative_regret += attempt_regret
        cumulative_reward += reward
        cumulative_optimal_turns += sum(optimal_indicators)
        cumulative_eligible_count += sum(eligible_counts)
        cumulative_override_opportunities += sum(override_opportunities)
        cumulative_policy_interventions += sum(interventions)
        opportunity_count = sum(override_opportunities)
        intervention_count = sum(interventions)

        attempt_rows.append(
            {
                "environment": environment,
                "context_condition": condition,
                "seed": seed,
                "attempt_index": attempt_index,
                "attempt_regret": attempt_regret,
                "cumulative_regret": cumulative_regret,
                "p_policy": p_policy,
                "p_oracle": p_oracle,
                "evaluator_score": evaluator_score,
                "reward": reward,
                "optimal_arm_rate_for_attempt": float(
                    np.mean(optimal_indicators)
                ),
                "oracle_arm_selection_indicator": "|".join(
                    str(value) for value in optimal_indicators
                ),
                "eligible_arm_mean_count": float(np.mean(eligible_counts)),
                "eligible_arm_counts": "|".join(
                    str(value) for value in eligible_counts
                ),
                "override_opportunity_count": opportunity_count,
                "override_opportunity_rate": opportunity_count / TURNS_PER_ATTEMPT,
                "policy_intervention_count": intervention_count,
                "policy_intervention_rate": intervention_count / TURNS_PER_ATTEMPT,
                "selected_arm": "|".join(selected_arms),
                "selected_arm_turn_1": selected_arms[0],
                "selected_arm_turn_2": selected_arms[1],
                "selected_arm_turn_3": selected_arms[2],
                "oracle_arm_turn_1": oracle_arms[0],
                "oracle_arm_turn_2": oracle_arms[1],
                "oracle_arm_turn_3": oracle_arms[2],
                "selected_final_moves": "|".join(selected_final_moves),
            }
        )

        if attempt_index in CHECKPOINTS:
            turns_seen = attempt_index * TURNS_PER_ATTEMPT
            checkpoint_rows.append(
                {
                    "environment": environment,
                    "context_condition": condition,
                    "seed": seed,
                    "checkpoint": attempt_index,
                    "cumulative_regret": cumulative_regret,
                    "mean_regret": cumulative_regret / attempt_index,
                    "mean_reward": cumulative_reward / attempt_index,
                    "optimal_arm_rate": cumulative_optimal_turns / turns_seen,
                    "eligible_arm_mean_count": cumulative_eligible_count / turns_seen,
                    "override_opportunity_rate": (
                        cumulative_override_opportunities / turns_seen
                    ),
                    "policy_intervention_rate": (
                        cumulative_policy_interventions / turns_seen
                    ),
                }
            )

    return attempt_rows, turn_rows, checkpoint_rows


def _assert_hidden_invariances() -> None:
    md6 = np.asarray([0.31, 0.28, 0.23, 0.18], dtype=np.float64)
    empty = RunningTutorQuality().snapshot()
    quality_a = TutorQualitySnapshot(1, 0.2, 0.4, 0.6, 0.8)
    quality_b = TutorQualitySnapshot(1, 0.8, 0.6, 0.4, 0.2)
    for environment_index, environment in enumerate(ENVIRONMENTS):
        hidden = generate_hidden_environment(environment, environment_index, 7)
        for arm in ARMS:
            low_mastery = hidden_arm_utility(
                hidden, arm, md6, 0.2, quality_a
            )
            high_mastery = hidden_arm_utility(
                hidden, arm, md6, 0.8, quality_a
            )
            if environment in {"E0_NO_CONTEXT", "E1_MD6", "E3_QUALITY"}:
                assert low_mastery == high_mastery
            quality_u_a = hidden_arm_utility(
                hidden, arm, md6, 0.4, quality_a
            )
            quality_u_b = hidden_arm_utility(
                hidden, arm, md6, 0.4, quality_b
            )
            if environment in {"E0_NO_CONTEXT", "E1_MD6", "E2_MASTERY"}:
                assert quality_u_a == quality_u_b
            assert 0.0 <= hidden_arm_utility(
                hidden, arm, md6, 0.4, empty
            ) <= 1.0


def run_sanity_checks() -> list[dict[str, str]]:
    """Execute every pre-specified deterministic check before a full run."""

    checks: list[dict[str, str]] = []

    def passed(name: str) -> None:
        checks.append({"check": name, "status": "PASS"})

    # Context dimensions, constant baseline, and exact projection order.
    md6 = _md6_mapping([0.29, 0.27, 0.24, 0.20])
    quality = RunningTutorQuality()
    for condition in CONTEXTS:
        x = context_vector(condition, md6, 0.61, quality)
        assert x.shape == (len(CONTEXT_FEATURES[condition]),)
    assert np.array_equal(context_vector("C0", md6, 0.61, quality), [1.0])
    passed("C0 context always constant")
    passed("correct dimensions for C0-C4")

    _assert_hidden_invariances()
    passed("MRB1 excluded from E0/E1/E2 hidden utility")
    passed("mastery excluded from E0/E1/E3 hidden utility")

    # Structural diversity check on a fixed pre-outcome state sample.  This is
    # not tuned from policy results and only verifies that the utility
    # construction permits state-dependent oracle-arm variation.
    for environment_index, environment in enumerate(ENVIRONMENTS):
        audit_hidden = generate_hidden_environment(
            environment, environment_index, 0
        )
        audit_world = generate_exogenous_world(
            environment_index, 0, attempts=40
        )
        oracle_arms_seen: set[str] = set()
        for attempt_zero in range(40):
            audit_quality = RunningTutorQuality()
            for turn_zero in range(TURNS_PER_ATTEMPT):
                md6_array = audit_world.md6_probabilities[
                    attempt_zero, turn_zero
                ]
                md6_mapping = _md6_mapping(md6_array)
                candidates = eligible_arms(md6_mapping, GAP_THRESHOLD)
                snapshot = audit_quality.snapshot()
                utilities = {
                    arm: hidden_arm_utility(
                        audit_hidden,
                        arm,
                        md6_array,
                        float(audit_world.mastery_before[attempt_zero]),
                        snapshot,
                    )
                    for arm in candidates
                }
                oracle_arms_seen.add(max(candidates, key=utilities.__getitem__))
                audit_quality.add_scores(
                    dict(
                        zip(
                            MRB1_TASKS,
                            audit_world.mrb1_scores[attempt_zero, turn_zero],
                            strict=True,
                        )
                    )
                )
        assert len(oracle_arms_seen) >= 2
    passed("multiple arms can be oracle-optimal across generated states")

    # A small complete run exercises eligibility, regret, evaluator support,
    # delayed timing, total attempt weight, and MRB1 timing assertions embedded
    # in simulate_run.
    hidden = generate_hidden_environment("E4_MIXED", 4, 3)
    world = generate_exogenous_world(4, 3, attempts=6)
    first = simulate_run(
        "E4_MIXED", 4, "C4", 3, hidden, world, attempts=6
    )
    attempts_a, turns_a, checkpoints_a = first
    assert all(float(row["attempt_regret"]) >= 0.0 for row in attempts_a)
    passed("regret nonnegative")
    assert all(
        str(row["oracle_arm"]) in str(row["eligible_arms"]).split("|")
        for row in turns_a
    )
    passed("oracle arm always eligible")
    assert all(
        str(row["selected_arm"]) in str(row["eligible_arms"]).split("|")
        for row in turns_a
    )
    passed("policy arm always eligible")
    allowed_rewards = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
    assert all(float(row["reward"]) in allowed_rewards for row in attempts_a)
    passed("evaluator reward support")
    assert all(
        float(row["has_within_attempt_quality"]) == (0.0 if row["turn_index"] == 1 else 1.0)
        for row in turns_a
    )
    assert all(
        float(row["running_mistake_identification"]) == 0.0
        for row in turns_a
        if row["turn_index"] == 1
    )
    # Turn 3 must carry the arithmetic mean of post-turn-1 and post-turn-2.
    grouped = {}
    for row in turns_a:
        grouped.setdefault(int(row["attempt_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["turn_index"]))
        expected = 0.5 * (
            float(rows[0]["mrb1_mistake_identification_after_turn"])
            + float(rows[1]["mrb1_mistake_identification_after_turn"])
        )
        assert math.isclose(
            float(rows[2]["running_mistake_identification"]),
            expected,
            rel_tol=0.0,
            abs_tol=NUMERIC_TOLERANCE,
        )
    passed("no current-turn MRB1 leakage and running arithmetic means")
    passed("no LinTS update before attempt completion")
    passed("delayed sample weights total one")

    second = simulate_run(
        "E4_MIXED", 4, "C4", 3, hidden, world, attempts=6
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    passed("deterministic rerun reproducibility")

    # Validate exogenous world contracts directly.
    assert np.all(world.md6_probabilities >= 0.0)
    np.testing.assert_allclose(
        world.md6_probabilities.sum(axis=2), 1.0, rtol=0.0, atol=1e-12
    )
    assert np.all((world.mrb1_scores >= 0.0) & (world.mrb1_scores <= 1.0))
    assert np.all((world.mastery_before >= 0.0) & (world.mastery_before <= 1.0))
    passed("MD6, MRB1, and mastery state bounds")

    # The short run may not hit a checkpoint, but its result shape is stable.
    assert checkpoints_a == []
    return checks


def _bootstrap_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(NUM_BOOTSTRAP_RESAMPLES, n))
    weights = np.zeros((NUM_BOOTSTRAP_RESAMPLES, n), dtype=np.float64)
    rows = np.repeat(np.arange(NUM_BOOTSTRAP_RESAMPLES), n)
    np.add.at(weights, (rows, indices.ravel()), 1.0)
    return weights / n


def _bootstrap_mean_ci(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    boot = weights @ np.asarray(values, dtype=np.float64)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def _paired_permutation_p(values: np.ndarray, stream_index: int) -> float:
    """Two-sided Monte Carlo paired sign-flip test for a mean difference."""

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
        signs = signs.astype(np.float64) * 2.0 - 1.0
        randomized = np.mean(signs * differences[None, :], axis=1)
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
        candidate = min(1.0, float((total - rank) * values[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def _effect_band(relative_reduction: float) -> str:
    magnitude = abs(relative_reduction)
    if magnitude < 0.02:
        return "negligible"
    if magnitude < 0.05:
        return "small"
    if magnitude <= 0.10:
        return "moderate"
    return "large"


def compute_summary_statistics(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level checkpoints with seed-bootstrap mean CIs."""

    rows: list[dict[str, object]] = []
    metrics = ("cumulative_regret", "mean_reward", "optimal_arm_rate")
    weights = _bootstrap_weights(NUM_SEEDS)
    for (environment, context, checkpoint), group in checkpoint_df.groupby(
        ["environment", "context_condition", "checkpoint"], sort=False
    ):
        ordered = group.sort_values("seed")
        if len(ordered) != NUM_SEEDS:
            raise RuntimeError("Summary group does not contain all paired seeds.")
        row: dict[str, object] = {
            "environment": environment,
            "context_condition": context,
            "checkpoint": int(checkpoint),
            "n_seeds": len(ordered),
        }
        for metric in metrics:
            values = ordered[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(values, weights)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_comparisons(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the pre-specified paired evidence at every checkpoint."""

    weights = _bootstrap_weights(NUM_SEEDS)
    rows: list[dict[str, object]] = []
    permutation_stream = 0
    for checkpoint in CHECKPOINTS:
        checkpoint_rows_start = len(rows)
        at_checkpoint = checkpoint_df[checkpoint_df["checkpoint"] == checkpoint]
        for environment, comparison, simpler, richer in PRIMARY_COMPARISONS:
            subset = at_checkpoint[at_checkpoint["environment"] == environment]
            pivot = subset.pivot(
                index="seed",
                columns="context_condition",
                values="cumulative_regret",
            ).sort_index()
            if simpler not in pivot or richer not in pivot or len(pivot) != NUM_SEEDS:
                raise RuntimeError("Incomplete seed pairing for primary comparison.")
            simpler_values = pivot[simpler].to_numpy(dtype=np.float64)
            richer_values = pivot[richer].to_numpy(dtype=np.float64)
            differences = simpler_values - richer_values
            simple_mean = float(np.mean(simpler_values))
            rich_mean = float(np.mean(richer_values))
            mean_improvement = float(np.mean(differences))
            relative = (
                float(mean_improvement / simple_mean)
                if simple_mean > 0.0
                else float("nan")
            )
            low, high = _bootstrap_mean_ci(differences, weights)
            permutation_p = _paired_permutation_p(
                differences, permutation_stream
            )
            permutation_stream += 1
            rows.append(
                {
                    "environment": environment,
                    "comparison": comparison,
                    "simpler_context": simpler,
                    "richer_context": richer,
                    "checkpoint": checkpoint,
                    "n_pairs": len(differences),
                    "simpler_mean_regret": simple_mean,
                    "richer_mean_regret": rich_mean,
                    "mean_paired_improvement": mean_improvement,
                    "median_paired_improvement": float(np.median(differences)),
                    "sd_paired_improvement": float(np.std(differences, ddof=1)),
                    "relative_regret_reduction": relative,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "permutation_p": permutation_p,
                    "holm_adjusted_p": float("nan"),
                    "descriptive_effect_band": _effect_band(relative),
                }
            )
        checkpoint_indices = range(checkpoint_rows_start, len(rows))
        adjusted = _holm_adjust(rows[index]["permutation_p"] for index in checkpoint_indices)
        for index, value in zip(checkpoint_indices, adjusted, strict=True):
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
    canonical_dir = project_root / "src" / "self_improvement"
    canonical_hashes = {
        name: _sha256(canonical_dir / name) for name in CANONICAL_HASH_FILES
    }
    result_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
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
        "contexts": {
            condition: {
                "dimension": len(features),
                "features": list(features),
            }
            for condition, features in CONTEXT_FEATURES.items()
        },
        "environments": ENVIRONMENT_DEFINITIONS,
        "hidden_utility_generator": {
            "form": "0.5 + 0.24*tanh(latent_arm_score)",
            "arm_intercept_bounds": [-0.15, 0.15],
            "md6_coefficient_bounds": [-0.75, 0.75],
            "mastery_coefficient_bounds": [-0.55, 0.55],
            "quality_coefficient_bounds": [-0.48, 0.48],
            "interaction_coefficient_bounds": [-0.30, 0.30],
            "sparsity": (
                "Deterministic seed-index masks set selected mastery, quality, "
                "and interaction coefficients to zero."
            ),
            "e4_non_circularity": (
                "Bounded arm-specific parameters, sparse zero coefficients, "
                "tanh nonlinearity, and weak mastery/quality interactions; "
                "parameters are generated before policy outcomes."
            ),
        },
        "state_generators": {
            "md6": "Dirichlet(alpha=[4.5,5.0,5.5,6.0]) independently per turn.",
            "mastery_before": "Beta(2.2,2.2) independently per attempt.",
            "mrb1": (
                "clip(0.64*attempt Beta(2.4,2.2) + 0.36*turn Beta(2,2) "
                "+ fixed task offset, 0, 1); observed only after response."
            ),
            "evaluator": (
                "Three shared independent Uniform(0,1) draws; score=sum(U<p_policy), "
                "which is Binomial(n=3,p=p_policy)."
            ),
        },
        "num_seeds": NUM_SEEDS,
        "seed_list": list(SEEDS),
        "attempts_per_seed": ATTEMPTS_PER_SEED,
        "turns_per_attempt": TURNS_PER_ATTEMPT,
        "checkpoints": list(CHECKPOINTS),
        "gap_threshold": GAP_THRESHOLD,
        "ridge_lambda": RIDGE_LAMBDA,
        "exploration_scale": EXPLORATION_SCALE,
        "reward_definition": "evaluator_score / 3.0; evaluator_score in {0,1,2,3}",
        "credit_scheme": (
            "All three selected turns updated only after attempt completion, "
            "each with sample_weight=1/3; total attempt weight=1."
        ),
        "oracle_definition": (
            "Deterministic canonical-order argmax of hidden expected utility "
            "over the current canonical eligible-arm set only."
        ),
        "optimal_arm_indicator_definition": (
            "1 when selected arm equals the deterministic eligible oracle arm."
        ),
        "bootstrap_method": (
            "Nonparametric percentile bootstrap of seed units using common "
            "multinomial resamples; individual attempts are never resampled."
        ),
        "bootstrap_resamples": NUM_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_confidence_level": 0.95,
        "permutation_method": (
            "Two-sided Monte Carlo paired sign-flip test of the mean seed-level "
            "cumulative-regret difference; +1 correction."
        ),
        "permutation_resamples": NUM_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "multiple_comparison_correction": (
            "Holm correction across the eight pre-specified comparisons "
            "within each checkpoint; attempt 500 is the primary endpoint."
        ),
        "paired_improvement_definition": (
            "cumulative_regret_simpler - cumulative_regret_richer; positive "
            "means lower observed regret for the richer representation."
        ),
        "relative_regret_reduction_definition": (
            "mean_paired_improvement / simpler_mean_regret; stored as a fraction."
        ),
        "descriptive_effect_band_definition": (
            "Absolute relative reduction magnitude: <2% negligible, 2%-<5% "
            "small, 5%-10% moderate, >10% large; descriptive only."
        ),
        "rng_strategy": {
            "base_seed": BASE_SEED,
            "construction": "NumPy SeedSequence with explicit stream components.",
            "coefficient_stream": "Independent per environment and numerical seed.",
            "exogenous_streams": (
                "Independent spawned MD6, mastery, MRB1 attempt, MRB1 turn, and "
                "evaluator streams, shared exactly across C0-C4."
            ),
            "policy_streams": (
                "Independent deterministic LinTS stream per environment, seed, "
                "and context condition; posterior sampling advances once per turn."
            ),
        },
        "sanity_checks": sanity_checks,
        "elapsed_seconds": elapsed_seconds,
        "canonical_source_sha256": canonical_hashes,
        "result_file_sha256": result_hashes,
    }


def run_full_experiment(project_root: Path, output_dir: Path) -> None:
    sanity_checks = run_sanity_checks()
    if not all(item["status"] == "PASS" for item in sanity_checks):
        raise RuntimeError("A sanity check failed; full experiment was not started.")

    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result directory: {output_dir}"
        )
    if stage_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete staging directory: {stage_dir}"
        )
    stage_dir.mkdir(parents=True)
    (stage_dir / "figures").mkdir()

    raw_path = stage_dir / "raw_attempt_metrics.csv"
    turn_path = stage_dir / "raw_turn_metrics.csv.gz"
    checkpoint_path = stage_dir / "checkpoint_metrics.csv"
    checkpoint_rows_all: list[dict[str, object]] = []
    total_runs = len(ENVIRONMENTS) * len(CONTEXTS) * len(SEEDS)
    completed_runs = 0
    start = time.perf_counter()

    with raw_path.open("w", encoding="utf-8", newline="") as raw_handle, gzip.open(
        turn_path, "wt", encoding="utf-8", newline="", compresslevel=6
    ) as turn_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=ATTEMPT_COLUMNS)
        turn_writer = csv.DictWriter(turn_handle, fieldnames=TURN_COLUMNS)
        raw_writer.writeheader()
        turn_writer.writeheader()

        for environment_index, environment in enumerate(ENVIRONMENTS):
            for seed in SEEDS:
                hidden = generate_hidden_environment(
                    environment, environment_index, seed
                )
                world = generate_exogenous_world(
                    environment_index, seed, ATTEMPTS_PER_SEED
                )
                for condition in CONTEXTS:
                    attempts, turns, checkpoints = simulate_run(
                        environment,
                        environment_index,
                        condition,
                        seed,
                        hidden,
                        world,
                        ATTEMPTS_PER_SEED,
                    )
                    raw_writer.writerows(attempts)
                    turn_writer.writerows(turns)
                    checkpoint_rows_all.extend(checkpoints)
                    completed_runs += 1
                    if completed_runs % 25 == 0 or completed_runs == total_runs:
                        elapsed = time.perf_counter() - start
                        projected = elapsed * total_runs / completed_runs
                        print(
                            f"progress {completed_runs}/{total_runs} runs; "
                            f"elapsed={elapsed / 60:.1f} min; "
                            f"projected={projected / 60:.1f} min",
                            flush=True,
                        )

    checkpoint_df = pd.DataFrame(
        checkpoint_rows_all, columns=CHECKPOINT_COLUMNS
    )
    checkpoint_df.to_csv(checkpoint_path, index=False)
    summary_df = compute_summary_statistics(checkpoint_df)
    summary_df.to_csv(stage_dir / "summary_statistics.csv", index=False)
    paired_df = compute_paired_comparisons(checkpoint_df)
    paired_df.to_csv(stage_dir / "paired_comparisons.csv", index=False)

    elapsed_seconds = time.perf_counter() - start
    manifest = _manifest(
        project_root, stage_dir, sanity_checks, elapsed_seconds
    )
    with (stage_dir / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/self_improvement/context_ablation_v1.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = _default_project_root()
    output_dir = args.output_dir or (
        project_root / "results" / "self_improvement" / EXPERIMENT_NAME
    )
    try:
        checks = run_sanity_checks()
        print("CONTEXT ABLATION SANITY CHECKS")
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
