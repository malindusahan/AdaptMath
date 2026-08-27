from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Run from repository root with:
#   python experiments/personalized_context_v1.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.lints_policy import (
    ARMS,
    TrueDisjointLinTS,
)
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    RunningTutorQuality,
    build_turn_context,
)

# ============================================================
# Fixed scientific design
# ============================================================

EXPERIMENT_NAME = "personalized_context_v1"

CONTEXTS = ("C3_9D", "P14_14D")
ENVIRONMENTS = ("E0_NO_PERSONAL_SIGNAL", "E1_PERSONAL_SIGNAL", "E2_MIXED")

N_SEEDS = 50
N_ATTEMPTS = 500
TURNS_PER_ATTEMPT = 3

GAP_THRESHOLD = 0.10
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20
SAMPLE_WEIGHT = 1.0 / TURNS_PER_ATTEMPT

WORLD_SEED_BASE = 2026082400
POLICY_SEED_BASE = 2026082500
ENV_COEFFICIENT_SEED = 2026082600

RESULT_DIR = REPO_ROOT / "results" / "self_improvement" / EXPERIMENT_NAME
FIGURE_DIR = RESULT_DIR / "figures"

C3_FEATURE_NAMES = (
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

PERSONAL_FEATURE_NAMES = (
    "mastery_before",
    "reasoning",
    "uncertainty",
    "clarification",
    "repeated_misunderstanding",
)

P14_FEATURE_NAMES = C3_FEATURE_NAMES + PERSONAL_FEATURE_NAMES

# The planted environment remains exactly linear in the relevant
# context block. Coefficients are neutral random directions rather
# than hand-written educational rules.
#
# L1 scaling limits reward-probability extremes while keeping enough
# action contrast for a 500-attempt capacity test.
BLOCK_L1 = 0.60

# Center values are synthetic distribution means / natural baselines.
C3_CENTER = np.asarray(
    [0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50, 0.50, 0.50],
    dtype=np.float64,
)
PERSONAL_CENTER = np.asarray(
    [0.50, 0.50, 0.50, 0.50, 0.30],
    dtype=np.float64,
)


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_rows_l1(matrix: np.ndarray, target_l1: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.sum(np.abs(matrix), axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("Coefficient row unexpectedly has zero L1 norm.")
    return matrix * (target_l1 / norms)


def build_environment_coefficients() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ENV_COEFFICIENT_SEED)

    base = rng.normal(size=(len(ARMS), len(C3_FEATURE_NAMES)))
    personal = rng.normal(size=(len(ARMS), len(PERSONAL_FEATURE_NAMES)))

    # Remove feature-wise common components, because a coefficient
    # shared by every arm cannot affect action preference.
    base = base - base.mean(axis=0, keepdims=True)
    personal = personal - personal.mean(axis=0, keepdims=True)

    base = _normalize_rows_l1(base, BLOCK_L1)
    personal = _normalize_rows_l1(personal, BLOCK_L1)
    return base, personal


BASE_COEF, PERSONAL_COEF = build_environment_coefficients()
ARM_TO_INDEX = {arm: i for i, arm in enumerate(ARMS)}


@dataclass(frozen=True)
class TurnWorld:
    md6: dict[str, float]
    mrb1_after_turn: dict[str, float]
    personal: np.ndarray


@dataclass(frozen=True)
class AttemptWorld:
    turns: tuple[TurnWorld, ...]
    terminal_uniform: float


def make_world(seed_index: int) -> tuple[AttemptWorld, ...]:
    rng = np.random.default_rng(WORLD_SEED_BASE + seed_index)

    attempts: list[AttemptWorld] = []

    for _ in range(N_ATTEMPTS):
        turns: list[TurnWorld] = []

        for _turn in range(TURNS_PER_ATTEMPT):
            # Dirichlet(4,4,4,4) yields moderately competitive MD6
            # distributions, so the frozen 0.10 overlay has genuine
            # intervention opportunities without forcing them.
            md6_values = rng.dirichlet(np.full(4, 4.0))
            md6 = {
                "generic": float(md6_values[0]),
                "probing": float(md6_values[1]),
                "focus": float(md6_values[2]),
                "telling": float(md6_values[3]),
            }

            # Synthetic tutor-quality outputs for the completed response.
            mrb1_values = rng.beta(2.5, 2.5, size=4)
            mrb1 = {
                task: float(value)
                for task, value in zip(MRB1_TASKS, mrb1_values, strict=True)
            }

            # Only the previously agreed Malindu-output family is represented.
            # No API-only/internal signal is used.
            personal = np.asarray(
                [
                    rng.beta(2.0, 2.0),  # mastery_before
                    rng.beta(2.0, 2.0),  # reasoning
                    rng.beta(2.0, 2.0),  # uncertainty
                    rng.beta(2.0, 2.0),  # clarification
                    float(rng.random() < 0.30),  # repeated misunderstanding
                ],
                dtype=np.float64,
            )

            turns.append(
                TurnWorld(
                    md6=md6,
                    mrb1_after_turn=mrb1,
                    personal=personal,
                )
            )

        # Common random number used by both C3 and P14 for paired outcome noise.
        attempts.append(
            AttemptWorld(
                turns=tuple(turns),
                terminal_uniform=float(rng.random()),
            )
        )

    return tuple(attempts)


def expected_reward_probability(
    environment: str,
    arm: str,
    c3_context: np.ndarray,
    personal: np.ndarray,
) -> tuple[float, bool]:
    arm_i = ARM_TO_INDEX[arm]

    c3_centered = np.asarray(c3_context, dtype=np.float64) - C3_CENTER
    personal_centered = np.asarray(personal, dtype=np.float64) - PERSONAL_CENTER

    base_signal = float(BASE_COEF[arm_i] @ c3_centered)
    personal_signal = float(PERSONAL_COEF[arm_i] @ personal_centered)

    if environment == "E0_NO_PERSONAL_SIGNAL":
        signal = base_signal
    elif environment == "E1_PERSONAL_SIGNAL":
        signal = personal_signal
    elif environment == "E2_MIXED":
        # Equal block contribution without increasing total planted scale.
        signal = 0.5 * base_signal + 0.5 * personal_signal
    else:
        raise ValueError(f"Unknown environment: {environment}")

    raw = 0.5 + signal
    clipped = not 0.05 <= raw <= 0.95
    probability = float(np.clip(raw, 0.05, 0.95))
    return probability, clipped


def build_policy_context(
    context_name: str,
    c3_context: np.ndarray,
    personal: np.ndarray,
) -> np.ndarray:
    if context_name == "C3_9D":
        context = np.asarray(c3_context, dtype=np.float64)
    elif context_name == "P14_14D":
        context = np.concatenate(
            [
                np.asarray(c3_context, dtype=np.float64),
                np.asarray(personal, dtype=np.float64),
            ]
        )
    else:
        raise ValueError(f"Unknown context: {context_name}")

    expected_dim = 9 if context_name == "C3_9D" else 14

    if context.shape != (expected_dim,):
        raise RuntimeError(
            f"{context_name} context should have shape {(expected_dim,)}, "
            f"got {context.shape}."
        )
    if not np.isfinite(context).all():
        raise RuntimeError(f"{context_name} context contains non-finite values.")

    return context


# ============================================================
# One paired policy run
# ============================================================

def run_policy_on_world(
    *,
    seed_index: int,
    environment: str,
    context_name: str,
    world: tuple[AttemptWorld, ...],
) -> tuple[list[dict], list[dict]]:
    context_dim = 9 if context_name == "C3_9D" else 14

    policy = TrueDisjointLinTS(
        context_dim=context_dim,
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=POLICY_SEED_BASE + seed_index,
        data_mode="synthetic",
    )

    attempt_rows: list[dict] = []
    turn_rows: list[dict] = []

    for attempt_index, attempt in enumerate(world, start=1):
        running_quality = RunningTutorQuality()

        chosen_turns: list[tuple[str, np.ndarray]] = []
        selected_probabilities: list[float] = []
        attempt_regret = 0.0
        attempt_oracle_probability = 0.0
        attempt_override_count = 0
        attempt_alt_opportunity_count = 0
        attempt_optimal_count = 0
        attempt_clip_count = 0

        for turn_index, turn in enumerate(attempt.turns, start=1):
            c3_context = build_turn_context(
                md6_probabilities=turn.md6,
                running_quality=running_quality,
            )
            if c3_context.shape != (9,):
                raise RuntimeError("Canonical C3 builder did not return 9-D.")

            policy_context = build_policy_context(
                context_name,
                c3_context,
                turn.personal,
            )

            candidates = eligible_arms(
                turn.md6,
                gap_threshold=GAP_THRESHOLD,
            )
            if len(candidates) > 1:
                attempt_alt_opportunity_count += 1

            arm_expected: dict[str, float] = {}
            clipped_any = False
            for arm in candidates:
                p, clipped = expected_reward_probability(
                    environment,
                    arm,
                    c3_context,
                    turn.personal,
                )
                arm_expected[arm] = p
                clipped_any = clipped_any or clipped

            if clipped_any:
                attempt_clip_count += 1

            oracle_arm = max(
                candidates,
                key=lambda arm: arm_expected[arm],
            )
            oracle_p = float(arm_expected[oracle_arm])

            selection = policy.select_arm(
                policy_context,
                eligible_arms=candidates,
            )
            selected_arm = selection["selected_arm"]
            if selected_arm not in candidates:
                raise RuntimeError("LinTS selected an ineligible arm.")

            selected_p = float(arm_expected[selected_arm])
            regret = float(oracle_p - selected_p)

            if regret < -1e-12:
                raise RuntimeError("Synthetic regret became negative.")

            if selected_arm != "baseline":
                attempt_override_count += 1
            if selected_arm == oracle_arm:
                attempt_optimal_count += 1

            attempt_regret += regret
            attempt_oracle_probability += oracle_p
            selected_probabilities.append(selected_p)
            chosen_turns.append((selected_arm, policy_context.copy()))

            turn_rows.append(
                {
                    "seed": seed_index,
                    "environment": environment,
                    "context": context_name,
                    "attempt": attempt_index,
                    "turn": turn_index,
                    "eligible_count": len(candidates),
                    "eligible_arms": "|".join(candidates),
                    "selected_arm": selected_arm,
                    "oracle_arm": oracle_arm,
                    "selected_expected_success": selected_p,
                    "oracle_expected_success": oracle_p,
                    "eligible_oracle_regret": regret,
                    "overridden": int(selected_arm != "baseline"),
                    "alternative_opportunity": int(len(candidates) > 1),
                    "optimal_arm_selected": int(selected_arm == oracle_arm),
                    "reward_probability_clipped": int(clipped_any),
                    **{
                        name: float(value)
                        for name, value in zip(
                            C3_FEATURE_NAMES,
                            c3_context,
                            strict=True,
                        )
                    },
                    **{
                        name: float(value)
                        for name, value in zip(
                            PERSONAL_FEATURE_NAMES,
                            turn.personal,
                            strict=True,
                        )
                    },
                }
            )

            # Causal timing: the current tutor-response MRB1 scores become
            # available only after selection, therefore only next turn sees them.
            running_quality.add_scores(turn.mrb1_after_turn)

        attempt_success_probability = float(np.mean(selected_probabilities))
        oracle_attempt_success_probability = float(
            attempt_oracle_probability / TURNS_PER_ATTEMPT
        )

        # Binary evaluator-success proxy: exactly the current 0/1 reward form.
        reward = float(attempt.terminal_uniform < attempt_success_probability)

        # No within-attempt updates. All three weighted updates happen now.
        updates_before = policy.total_updates
        for selected_arm, policy_context in chosen_turns:
            policy.update(
                selected_arm,
                policy_context,
                reward,
                sample_weight=SAMPLE_WEIGHT,
            )
        if policy.total_updates != updates_before + TURNS_PER_ATTEMPT:
            raise RuntimeError("Delayed update count is inconsistent.")

        attempt_rows.append(
            {
                "seed": seed_index,
                "environment": environment,
                "context": context_name,
                "attempt": attempt_index,
                "reward": reward,
                "attempt_success_probability": attempt_success_probability,
                "oracle_attempt_success_probability": (
                    oracle_attempt_success_probability
                ),
                "eligible_oracle_regret": attempt_regret,
                "mean_turn_regret": attempt_regret / TURNS_PER_ATTEMPT,
                "override_rate": (
                    attempt_override_count / TURNS_PER_ATTEMPT
                ),
                "alternative_opportunity_rate": (
                    attempt_alt_opportunity_count / TURNS_PER_ATTEMPT
                ),
                "optimal_arm_rate": (
                    attempt_optimal_count / TURNS_PER_ATTEMPT
                ),
                "reward_probability_clip_rate": (
                    attempt_clip_count / TURNS_PER_ATTEMPT
                ),
            }
        )

    return attempt_rows, turn_rows


# ============================================================
# Statistics
# ============================================================

def percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(values),
        size=(n_bootstrap, len(values)),
    )
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_sign_flip_pvalue(
    values: np.ndarray,
    *,
    seed: int,
    n_permutations: int = 50000,
) -> float:
    """Two-sided paired randomization p-value for a mean difference."""
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))

    rng = np.random.default_rng(seed)
    extreme = 0

    # Chunking avoids a large 50k x 50 temporary matrix at once.
    remaining = n_permutations
    while remaining > 0:
        n = min(5000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(n, len(values)),
        )
        perm_means = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(perm_means >= observed))
        remaining -= n

    # Plus-one correction.
    return float((extreme + 1) / (n_permutations + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)

    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)

    return adjusted.tolist()


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    all_attempt_rows: list[dict] = []
    all_turn_rows: list[dict] = []

    print("=" * 78)
    print("PERSONALIZED CONTEXT V1")
    print("=" * 78)
    print("Contexts:", CONTEXTS)
    print("Environments:", ENVIRONMENTS)
    print("Seeds:", N_SEEDS)
    print("Attempts per seed:", N_ATTEMPTS)
    print("Turns per attempt:", TURNS_PER_ATTEMPT)
    print("Gap threshold:", GAP_THRESHOLD)
    print("Ridge lambda:", RIDGE_LAMBDA)
    print("Exploration:", EXPLORATION_SCALE)
    print("Reward: binary attempt success")
    print("Delayed credit: equal 1/T")
    print()

    for seed_index in range(N_SEEDS):
        world = make_world(seed_index)

        for environment in ENVIRONMENTS:
            for context_name in CONTEXTS:
                print(
                    f"seed={seed_index:02d} "
                    f"env={environment} "
                    f"context={context_name}"
                )

                attempt_rows, turn_rows = run_policy_on_world(
                    seed_index=seed_index,
                    environment=environment,
                    context_name=context_name,
                    world=world,
                )
                all_attempt_rows.extend(attempt_rows)
                all_turn_rows.extend(turn_rows)

    attempt_df = pd.DataFrame(all_attempt_rows)
    turn_df = pd.DataFrame(all_turn_rows)

    attempt_path = RESULT_DIR / "raw_attempt_results.csv.gz"
    turn_path = RESULT_DIR / "raw_turn_results.csv.gz"

    attempt_df.to_csv(
        attempt_path,
        index=False,
        compression="gzip",
    )
    turn_df.to_csv(
        turn_path,
        index=False,
        compression="gzip",
    )

    # One row per seed/environment/context.
    seed_summary = (
        attempt_df.groupby(
            ["seed", "environment", "context"],
            as_index=False,
        )
        .agg(
            cumulative_regret=("eligible_oracle_regret", "sum"),
            mean_turn_regret=("mean_turn_regret", "mean"),
            mean_reward=("reward", "mean"),
            mean_expected_success=("attempt_success_probability", "mean"),
            mean_oracle_expected_success=(
                "oracle_attempt_success_probability",
                "mean",
            ),
            override_rate=("override_rate", "mean"),
            alternative_opportunity_rate=(
                "alternative_opportunity_rate",
                "mean",
            ),
            optimal_arm_rate=("optimal_arm_rate", "mean"),
            reward_probability_clip_rate=(
                "reward_probability_clip_rate",
                "mean",
            ),
        )
    )

    seed_summary.to_csv(
        RESULT_DIR / "seed_summary.csv",
        index=False,
    )

    # Paired P14 - C3 comparisons.
    paired_rows: list[dict] = []

    for env_i, environment in enumerate(ENVIRONMENTS):
        env = seed_summary[
            seed_summary["environment"] == environment
        ]

        c3 = (
            env[env["context"] == "C3_9D"]
            .set_index("seed")
            .sort_index()
        )
        p14 = (
            env[env["context"] == "P14_14D"]
            .set_index("seed")
            .sort_index()
        )

        if not c3.index.equals(p14.index):
            raise RuntimeError("Paired seeds are inconsistent.")

        regret_diff = (
            p14["cumulative_regret"].to_numpy()
            - c3["cumulative_regret"].to_numpy()
        )
        reward_diff = (
            p14["mean_reward"].to_numpy()
            - c3["mean_reward"].to_numpy()
        )
        expected_diff = (
            p14["mean_expected_success"].to_numpy()
            - c3["mean_expected_success"].to_numpy()
        )
        optimal_diff = (
            p14["optimal_arm_rate"].to_numpy()
            - c3["optimal_arm_rate"].to_numpy()
        )

        regret_ci = percentile_bootstrap_ci(
            regret_diff,
            seed=2026082700 + env_i,
        )
        reward_ci = percentile_bootstrap_ci(
            reward_diff,
            seed=2026082800 + env_i,
        )

        regret_p = paired_sign_flip_pvalue(
            regret_diff,
            seed=2026082900 + env_i,
        )
        reward_p = paired_sign_flip_pvalue(
            reward_diff,
            seed=2026083000 + env_i,
        )

        c3_regret_mean = float(c3["cumulative_regret"].mean())
        p14_regret_mean = float(p14["cumulative_regret"].mean())

        if c3_regret_mean > 0.0:
            regret_reduction_pct = (
                100.0
                * (c3_regret_mean - p14_regret_mean)
                / c3_regret_mean
            )
        else:
            regret_reduction_pct = float("nan")

        paired_rows.append(
            {
                "environment": environment,
                "c3_mean_cumulative_regret": c3_regret_mean,
                "p14_mean_cumulative_regret": p14_regret_mean,
                "p14_minus_c3_regret": float(regret_diff.mean()),
                "regret_diff_ci95_low": regret_ci[0],
                "regret_diff_ci95_high": regret_ci[1],
                "regret_reduction_pct": regret_reduction_pct,
                "regret_randomization_p": regret_p,
                "c3_mean_reward": float(c3["mean_reward"].mean()),
                "p14_mean_reward": float(p14["mean_reward"].mean()),
                "p14_minus_c3_reward": float(reward_diff.mean()),
                "reward_diff_ci95_low": reward_ci[0],
                "reward_diff_ci95_high": reward_ci[1],
                "reward_randomization_p": reward_p,
                "p14_minus_c3_expected_success": float(
                    expected_diff.mean()
                ),
                "p14_minus_c3_optimal_arm_rate": float(
                    optimal_diff.mean()
                ),
                "mean_alternative_opportunity_rate": float(
                    c3["alternative_opportunity_rate"].mean()
                ),
                "c3_override_rate": float(c3["override_rate"].mean()),
                "p14_override_rate": float(p14["override_rate"].mean()),
                "c3_clip_rate": float(
                    c3["reward_probability_clip_rate"].mean()
                ),
                "p14_clip_rate": float(
                    p14["reward_probability_clip_rate"].mean()
                ),
            }
        )

    paired_stats = pd.DataFrame(paired_rows)

    paired_stats["regret_holm_p"] = holm_adjust(
        paired_stats["regret_randomization_p"].tolist()
    )
    paired_stats["reward_holm_p"] = holm_adjust(
        paired_stats["reward_randomization_p"].tolist()
    )

    paired_stats.to_csv(
        RESULT_DIR / "paired_statistics.csv",
        index=False,
    )

    # ========================================================
    # Figures from raw seed summaries
    # ========================================================

    mean_plot = (
        seed_summary.groupby(
            ["environment", "context"],
            as_index=False,
        )
        .agg(
            mean_cumulative_regret=("cumulative_regret", "mean"),
            se_cumulative_regret=(
                "cumulative_regret",
                lambda x: x.std(ddof=1) / math.sqrt(len(x)),
            ),
            mean_reward=("mean_reward", "mean"),
            se_reward=(
                "mean_reward",
                lambda x: x.std(ddof=1) / math.sqrt(len(x)),
            ),
        )
    )

    x = np.arange(len(ENVIRONMENTS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, context_name in [
        (-width / 2, "C3_9D"),
        (width / 2, "P14_14D"),
    ]:
        rows = (
            mean_plot[
                mean_plot["context"] == context_name
            ]
            .set_index("environment")
            .loc[list(ENVIRONMENTS)]
        )
        ax.bar(
            x + offset,
            rows["mean_cumulative_regret"],
            width,
            yerr=1.96 * rows["se_cumulative_regret"],
            capsize=4,
            label=context_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["E0 no personal", "E1 personal", "E2 mixed"]
    )
    ax.set_ylabel("Mean cumulative eligible-oracle regret")
    ax.set_title("C3 vs P14 personalized context")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "cumulative_regret.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, context_name in [
        (-width / 2, "C3_9D"),
        (width / 2, "P14_14D"),
    ]:
        rows = (
            mean_plot[
                mean_plot["context"] == context_name
            ]
            .set_index("environment")
            .loc[list(ENVIRONMENTS)]
        )
        ax.bar(
            x + offset,
            rows["mean_reward"],
            width,
            yerr=1.96 * rows["se_reward"],
            capsize=4,
            label=context_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["E0 no personal", "E1 personal", "E2 mixed"]
    )
    ax.set_ylabel("Mean binary attempt reward")
    ax.set_title("Binary evaluator-success proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "mean_reward.png",
        dpi=180,
    )
    plt.close(fig)

    # ========================================================
    # Manifest
    # ========================================================

    source_paths = {
        "experiment_script": Path(__file__).resolve(),
        "lints_policy": REPO_ROOT / "src/self_improvement/lints_policy.py",
        "conservative_overlay": (
            REPO_ROOT / "src/self_improvement/conservative_overlay.py"
        ),
        "turn_context_builder": (
            REPO_ROOT / "src/self_improvement/turn_context_builder.py"
        ),
    }

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Synthetic capacity/robustness test of adding five dynamic "
            "student-state features to the frozen 9-D C3 LinTS context."
        ),
        "claim_boundary": (
            "This synthetic experiment does not establish real educational "
            "benefit. It tests whether the bandit can exploit planted "
            "student-state signal and whether irrelevant added dimensions "
            "materially harm learning."
        ),
        "contexts": {
            "C3_9D": list(C3_FEATURE_NAMES),
            "P14_14D": list(P14_FEATURE_NAMES),
        },
        "malindu_outputs_used_for_personalization": [
            "mastery_before",
            "reasoning",
            "uncertainty",
            "clarification",
            "repeated_misunderstanding",
        ],
        "explicitly_not_used_as_context": [
            "skill raw string",
            "mastery_after",
            "delta_mastery",
            "evaluator_result",
        ],
        "environments": {
            "E0_NO_PERSONAL_SIGNAL": (
                "Expected reward depends on C3 only; personalized block "
                "is irrelevant."
            ),
            "E1_PERSONAL_SIGNAL": (
                "Expected reward depends on personalized block only."
            ),
            "E2_MIXED": (
                "Expected reward receives equal planted contribution "
                "from C3 and personalized blocks."
            ),
        },
        "n_seeds": N_SEEDS,
        "attempts_per_seed": N_ATTEMPTS,
        "turns_per_attempt": TURNS_PER_ATTEMPT,
        "gap_threshold": GAP_THRESHOLD,
        "ridge_lambda": RIDGE_LAMBDA,
        "exploration_scale": EXPLORATION_SCALE,
        "reward": (
            "Binary terminal attempt-success proxy; unchanged 0/1 form."
        ),
        "credit": "Equal normalized delayed credit 1/T.",
        "within_attempt_policy_updates": False,
        "environment_coefficient_seed": ENV_COEFFICIENT_SEED,
        "world_seed_base": WORLD_SEED_BASE,
        "policy_seed_base": POLICY_SEED_BASE,
        "coefficient_block_l1": BLOCK_L1,
        "source_sha256": {
            name: sha256_file(path)
            for name, path in source_paths.items()
        },
        "result_files": {
            "raw_attempt_results": str(attempt_path.relative_to(REPO_ROOT)),
            "raw_turn_results": str(turn_path.relative_to(REPO_ROOT)),
            "seed_summary": str(
                (RESULT_DIR / "seed_summary.csv").relative_to(REPO_ROOT)
            ),
            "paired_statistics": str(
                (RESULT_DIR / "paired_statistics.csv").relative_to(REPO_ROOT)
            ),
            "figures": [
                str(
                    (FIGURE_DIR / "cumulative_regret.png").relative_to(
                        REPO_ROOT
                    )
                ),
                str(
                    (FIGURE_DIR / "mean_reward.png").relative_to(REPO_ROOT)
                ),
            ],
        },
    }

    manifest_path = RESULT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 78)
    print("PAIRED RESULTS")
    print("=" * 78)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(paired_stats.to_string(index=False))

    print()
    print("Saved:")
    print(" ", attempt_path)
    print(" ", turn_path)
    print(" ", RESULT_DIR / "seed_summary.csv")
    print(" ", RESULT_DIR / "paired_statistics.csv")
    print(" ", manifest_path)
    print(" ", FIGURE_DIR / "cumulative_regret.png")
    print(" ", FIGURE_DIR / "mean_reward.png")


if __name__ == "__main__":
    main()
