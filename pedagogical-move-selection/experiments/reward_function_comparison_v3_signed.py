from __future__ import annotations

import hashlib
import json
import math
import sys
from math import isfinite
from numbers import Real
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Run from repository root:
#   python experiments/reward_function_comparison_v1.py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.lints_policy import ARMS, TrueDisjointLinTS
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    RunningTutorQuality,
    build_turn_context,
)


class SignedRewardLinTS(TrueDisjointLinTS):
    """Experiment-only LinTS variant allowing signed mastery-gain rewards.

    The canonical implementation is unchanged. This subclass overrides only
    reward validation; posterior sampling, A updates, b updates, arm selection,
    delayed weighting, and RNG behavior are inherited unchanged.

    Because mastery_before and mastery_after are probabilities in [0,1],
    delta_mastery = mastery_after - mastery_before is theoretically bounded
    in [-1,1].
    """

    @staticmethod
    def _validate_reward(reward: object) -> float:
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError(
                f"reward must be a non-boolean real number, got {reward!r}."
            )
        try:
            result = float(reward)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "reward could not be represented as a float."
            ) from exc
        if not isfinite(result):
            raise ValueError(f"reward must be finite, got {result!r}.")
        if not -1.0 <= result <= 1.0:
            raise ValueError(
                f"signed mastery-gain reward must be in [-1, 1], got {result}."
            )
        return result


# ============================================================
# Fixed scientific design
# ============================================================

EXPERIMENT_NAME = "reward_function_comparison_v3_signed"

REWARD_MODES = (
    "R0_BINARY_SUCCESS",
    "R1_RAW_DELTA_MASTERY",
)

N_SEEDS = 50
N_ATTEMPTS = 500
TURNS_PER_ATTEMPT = 3
LATE_WINDOW = 100

GAP_THRESHOLD = 0.10
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20
SAMPLE_WEIGHT = 1.0 / TURNS_PER_ATTEMPT

# Synthetic world / policy seeds.
WORLD_SEED_BASE = 2026083100
POLICY_SEED_BASE = 2026090100
ENV_COEFFICIENT_SEED = 2026090200

# Planted expected-success environment.
# Coefficients are neutral random directions, not educational claims.
BLOCK_L1 = 0.60

# Raw signed mastery-gain reward scale.
#
# This is deliberately NOT claimed to be the real Malindu distribution.
# It is only a pre-integration scale stress-test using the rough magnitude
# already observed in the small example set supplied for this project.
#
# E[R1 | p] = DELTA_GAIN_SCALE * (p - 0.5)
# so R0 and R1 have exactly the same expected arm ordering.
DELTA_GAIN_SCALE = 0.14
DELTA_NOISE_STD = 0.012
DELTA_CLIP = 1.0

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

C3_CENTER = np.asarray(
    [0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50, 0.50, 0.50],
    dtype=np.float64,
)

ARM_TO_INDEX = {arm: i for i, arm in enumerate(ARMS)}


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_rows_l1(matrix: np.ndarray, target_l1: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.sum(np.abs(matrix), axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("Synthetic coefficient row has zero L1 norm.")
    return matrix * (target_l1 / norms)


def build_environment_coefficients() -> np.ndarray:
    rng = np.random.default_rng(ENV_COEFFICIENT_SEED)
    coef = rng.normal(size=(len(ARMS), len(C3_FEATURE_NAMES)))

    # Remove feature-wise arm-common components because they cannot change
    # action preference.
    coef = coef - coef.mean(axis=0, keepdims=True)
    return normalize_rows_l1(coef, BLOCK_L1)


ENV_COEF = build_environment_coefficients()


@dataclass(frozen=True)
class TurnWorld:
    md6: dict[str, float]
    mrb1_after_turn: dict[str, float]


@dataclass(frozen=True)
class AttemptWorld:
    turns: tuple[TurnWorld, ...]
    binary_uniform: float
    delta_normal: float


def make_world(seed_index: int) -> tuple[AttemptWorld, ...]:
    rng = np.random.default_rng(WORLD_SEED_BASE + seed_index)
    attempts: list[AttemptWorld] = []

    for _ in range(N_ATTEMPTS):
        turns: list[TurnWorld] = []

        for _turn in range(TURNS_PER_ATTEMPT):
            # Moderately competitive MD6 distributions create a realistic
            # mixture of baseline-only and alternative-eligible turns under
            # the frozen 0.10 probability-gap rule.
            md6_values = rng.dirichlet(np.full(4, 4.0))
            md6 = {
                "generic": float(md6_values[0]),
                "probing": float(md6_values[1]),
                "focus": float(md6_values[2]),
                "telling": float(md6_values[3]),
            }

            # Synthetic completed-response MRB1 scores. They enter only the
            # next turn through the canonical RunningTutorQuality machinery.
            mrb1_values = rng.beta(2.5, 2.5, size=4)
            mrb1 = {
                task: float(value)
                for task, value in zip(MRB1_TASKS, mrb1_values, strict=True)
            }

            turns.append(
                TurnWorld(
                    md6=md6,
                    mrb1_after_turn=mrb1,
                )
            )

        attempts.append(
            AttemptWorld(
                turns=tuple(turns),
                binary_uniform=float(rng.random()),
                delta_normal=float(rng.normal()),
            )
        )

    return tuple(attempts)


def expected_success_probability(
    arm: str,
    c3_context: np.ndarray,
) -> tuple[float, bool]:
    arm_i = ARM_TO_INDEX[arm]
    centered = np.asarray(c3_context, dtype=np.float64) - C3_CENTER
    signal = float(ENV_COEF[arm_i] @ centered)

    raw = 0.5 + signal
    clipped = not 0.05 <= raw <= 0.95
    return float(np.clip(raw, 0.05, 0.95)), clipped


def terminal_reward(
    reward_mode: str,
    attempt_success_probability: float,
    attempt_world: AttemptWorld,
) -> tuple[float, bool]:
    if reward_mode == "R0_BINARY_SUCCESS":
        return (
            float(
                attempt_world.binary_uniform
                < attempt_success_probability
            ),
            False,
        )

    if reward_mode == "R1_RAW_DELTA_MASTERY":
        raw = (
            DELTA_GAIN_SCALE
            * (attempt_success_probability - 0.5)
            + DELTA_NOISE_STD * attempt_world.delta_normal
        )
        clipped = not -DELTA_CLIP <= raw <= DELTA_CLIP
        return (
            float(np.clip(raw, -DELTA_CLIP, DELTA_CLIP)),
            clipped,
        )

    raise ValueError(f"Unknown reward mode: {reward_mode}")


# ============================================================
# One policy run
# ============================================================

def run_policy_on_world(
    *,
    seed_index: int,
    reward_mode: str,
    world: tuple[AttemptWorld, ...],
) -> tuple[list[dict], list[dict]]:
    policy_class = (
        TrueDisjointLinTS
        if reward_mode == "R0_BINARY_SUCCESS"
        else SignedRewardLinTS
    )
    policy = policy_class(
        context_dim=9,
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=POLICY_SEED_BASE + seed_index,
        data_mode="synthetic",
    )

    attempt_rows: list[dict] = []
    turn_rows: list[dict] = []

    for attempt_index, attempt in enumerate(world, start=1):
        running_quality = RunningTutorQuality()

        delayed_updates: list[tuple[str, np.ndarray]] = []
        selected_probabilities: list[float] = []

        cumulative_attempt_regret = 0.0
        override_count = 0
        alternative_count = 0
        optimal_count = 0
        probability_clip_count = 0

        for turn_index, turn in enumerate(attempt.turns, start=1):
            c3_context = build_turn_context(
                md6_probabilities=turn.md6,
                running_quality=running_quality,
            )

            if c3_context.shape != (9,):
                raise RuntimeError(
                    f"Canonical C3 builder returned {c3_context.shape}, not (9,)."
                )

            candidates = eligible_arms(
                turn.md6,
                gap_threshold=GAP_THRESHOLD,
            )
            if len(candidates) > 1:
                alternative_count += 1

            expected = {}
            any_clip = False
            for arm in candidates:
                p, clipped = expected_success_probability(
                    arm,
                    c3_context,
                )
                expected[arm] = p
                any_clip = any_clip or clipped

            if any_clip:
                probability_clip_count += 1

            oracle_arm = max(
                candidates,
                key=lambda arm: expected[arm],
            )
            oracle_p = float(expected[oracle_arm])

            selection = policy.select_arm(
                c3_context,
                eligible_arms=candidates,
            )
            selected_arm = selection["selected_arm"]
            if selected_arm not in candidates:
                raise RuntimeError("LinTS selected an ineligible arm.")

            selected_p = float(expected[selected_arm])
            regret = float(oracle_p - selected_p)
            if regret < -1e-12:
                raise RuntimeError("Eligible-oracle regret became negative.")

            cumulative_attempt_regret += regret
            selected_probabilities.append(selected_p)
            delayed_updates.append((selected_arm, c3_context.copy()))

            if selected_arm != "baseline":
                override_count += 1
            if selected_arm == oracle_arm:
                optimal_count += 1

            turn_rows.append(
                {
                    "seed": seed_index,
                    "reward_mode": reward_mode,
                    "attempt": attempt_index,
                    "turn": turn_index,
                    "selected_arm": selected_arm,
                    "oracle_arm": oracle_arm,
                    "eligible_count": len(candidates),
                    "eligible_arms": "|".join(candidates),
                    "selected_expected_success": selected_p,
                    "oracle_expected_success": oracle_p,
                    "eligible_oracle_regret": regret,
                    "overridden": int(selected_arm != "baseline"),
                    "alternative_opportunity": int(len(candidates) > 1),
                    "optimal_arm_selected": int(selected_arm == oracle_arm),
                    "expected_success_probability_clipped": int(any_clip),
                    **{
                        name: float(value)
                        for name, value in zip(
                            C3_FEATURE_NAMES,
                            c3_context,
                            strict=True,
                        )
                    },
                }
            )

            # Causal timing: current response quality is added only after
            # selection and therefore affects only a later tutor turn.
            running_quality.add_scores(turn.mrb1_after_turn)

        attempt_success_probability = float(
            np.mean(selected_probabilities)
        )

        reward, reward_clipped = terminal_reward(
            reward_mode,
            attempt_success_probability,
            attempt,
        )

        # Delayed attempt-end updates only, equal normalized credit 1/T.
        updates_before = policy.total_updates
        for selected_arm, context in delayed_updates:
            policy.update(
                selected_arm,
                context,
                reward,
                sample_weight=SAMPLE_WEIGHT,
            )

        if policy.total_updates != updates_before + TURNS_PER_ATTEMPT:
            raise RuntimeError("Delayed update count is inconsistent.")

        attempt_rows.append(
            {
                "seed": seed_index,
                "reward_mode": reward_mode,
                "attempt": attempt_index,
                "reward": reward,
                "reward_clipped": int(reward_clipped),
                "attempt_success_probability": attempt_success_probability,
                "eligible_oracle_regret": cumulative_attempt_regret,
                "mean_turn_regret": (
                    cumulative_attempt_regret / TURNS_PER_ATTEMPT
                ),
                "override_rate": override_count / TURNS_PER_ATTEMPT,
                "alternative_opportunity_rate": (
                    alternative_count / TURNS_PER_ATTEMPT
                ),
                "optimal_arm_rate": optimal_count / TURNS_PER_ATTEMPT,
                "expected_success_probability_clip_rate": (
                    probability_clip_count / TURNS_PER_ATTEMPT
                ),
            }
        )

    return attempt_rows, turn_rows


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(
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
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)

    extreme = 0
    remaining = n_permutations

    while remaining > 0:
        n = min(5000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(n, len(values)),
        )
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        remaining -= n

    return float((extreme + 1) / (n_permutations + 1))


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    all_attempt_rows: list[dict] = []
    all_turn_rows: list[dict] = []

    print("=" * 78)
    print("REWARD FUNCTION COMPARISON V3 — SIGNED DELTA MASTERY")
    print("=" * 78)
    print("Context: frozen C3 9-D")
    print("Reward modes:", REWARD_MODES)
    print("Seeds:", N_SEEDS)
    print("Attempts per seed:", N_ATTEMPTS)
    print("Turns per attempt:", TURNS_PER_ATTEMPT)
    print("Gap threshold:", GAP_THRESHOLD)
    print("Ridge lambda:", RIDGE_LAMBDA)
    print("Exploration:", EXPLORATION_SCALE)
    print("Delayed credit: equal 1/T")
    print(
        "R1 scale:",
        f"E[delta|p]={DELTA_GAIN_SCALE}*(p-0.5), "
        f"noise_sd={DELTA_NOISE_STD}, clip=±{DELTA_CLIP}",
    )
    print()

    for seed_index in range(N_SEEDS):
        world = make_world(seed_index)

        for reward_mode in REWARD_MODES:
            print(
                f"seed={seed_index:02d} "
                f"reward={reward_mode}"
            )

            attempt_rows, turn_rows = run_policy_on_world(
                seed_index=seed_index,
                reward_mode=reward_mode,
                world=world,
            )
            all_attempt_rows.extend(attempt_rows)
            all_turn_rows.extend(turn_rows)

    attempt_df = pd.DataFrame(all_attempt_rows)
    turn_df = pd.DataFrame(all_turn_rows)

    raw_attempt_path = RESULT_DIR / "raw_attempt_results.csv.gz"
    raw_turn_path = RESULT_DIR / "raw_turn_results.csv.gz"

    attempt_df.to_csv(
        raw_attempt_path,
        index=False,
        compression="gzip",
    )
    turn_df.to_csv(
        raw_turn_path,
        index=False,
        compression="gzip",
    )

    # Seed-level summaries are the statistical research units.
    seed_rows: list[dict] = []

    for (seed, reward_mode), group in attempt_df.groupby(
        ["seed", "reward_mode"],
        sort=True,
    ):
        late = group[group["attempt"] > N_ATTEMPTS - LATE_WINDOW]

        seed_rows.append(
            {
                "seed": int(seed),
                "reward_mode": reward_mode,
                "cumulative_regret": float(
                    group["eligible_oracle_regret"].sum()
                ),
                "late_window_cumulative_regret": float(
                    late["eligible_oracle_regret"].sum()
                ),
                "mean_turn_regret": float(
                    group["mean_turn_regret"].mean()
                ),
                "mean_expected_success": float(
                    group["attempt_success_probability"].mean()
                ),
                "mean_raw_reward": float(group["reward"].mean()),
                "raw_reward_std": float(group["reward"].std(ddof=1)),
                "override_rate": float(group["override_rate"].mean()),
                "alternative_opportunity_rate": float(
                    group["alternative_opportunity_rate"].mean()
                ),
                "optimal_arm_rate": float(
                    group["optimal_arm_rate"].mean()
                ),
                "reward_clip_rate": float(
                    group["reward_clipped"].mean()
                ),
                "expected_success_probability_clip_rate": float(
                    group[
                        "expected_success_probability_clip_rate"
                    ].mean()
                ),
            }
        )

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(
        RESULT_DIR / "seed_summary.csv",
        index=False,
    )

    r0 = (
        seed_summary[
            seed_summary["reward_mode"] == "R0_BINARY_SUCCESS"
        ]
        .set_index("seed")
        .sort_index()
    )
    r1 = (
        seed_summary[
            seed_summary["reward_mode"] == "R1_RAW_DELTA_MASTERY"
        ]
        .set_index("seed")
        .sort_index()
    )

    if not r0.index.equals(r1.index):
        raise RuntimeError("Reward-mode seed pairing is inconsistent.")

    regret_diff = (
        r1["cumulative_regret"].to_numpy()
        - r0["cumulative_regret"].to_numpy()
    )
    late_regret_diff = (
        r1["late_window_cumulative_regret"].to_numpy()
        - r0["late_window_cumulative_regret"].to_numpy()
    )
    expected_success_diff = (
        r1["mean_expected_success"].to_numpy()
        - r0["mean_expected_success"].to_numpy()
    )
    optimal_rate_diff = (
        r1["optimal_arm_rate"].to_numpy()
        - r0["optimal_arm_rate"].to_numpy()
    )
    override_diff = (
        r1["override_rate"].to_numpy()
        - r0["override_rate"].to_numpy()
    )

    regret_ci = bootstrap_mean_ci(
        regret_diff,
        seed=2026090301,
    )
    late_regret_ci = bootstrap_mean_ci(
        late_regret_diff,
        seed=2026090302,
    )
    expected_ci = bootstrap_mean_ci(
        expected_success_diff,
        seed=2026090303,
    )

    stats = {
        "r0_mean_cumulative_regret": float(
            r0["cumulative_regret"].mean()
        ),
        "r1_mean_cumulative_regret": float(
            r1["cumulative_regret"].mean()
        ),
        "r1_minus_r0_cumulative_regret": float(regret_diff.mean()),
        "regret_diff_ci95": list(regret_ci),
        "regret_randomization_p": paired_sign_flip_pvalue(
            regret_diff,
            seed=2026090401,
        ),
        "r0_mean_late_window_regret": float(
            r0["late_window_cumulative_regret"].mean()
        ),
        "r1_mean_late_window_regret": float(
            r1["late_window_cumulative_regret"].mean()
        ),
        "r1_minus_r0_late_window_regret": float(
            late_regret_diff.mean()
        ),
        "late_regret_diff_ci95": list(late_regret_ci),
        "late_regret_randomization_p": paired_sign_flip_pvalue(
            late_regret_diff,
            seed=2026090402,
        ),
        "r0_mean_expected_success": float(
            r0["mean_expected_success"].mean()
        ),
        "r1_mean_expected_success": float(
            r1["mean_expected_success"].mean()
        ),
        "r1_minus_r0_expected_success": float(
            expected_success_diff.mean()
        ),
        "expected_success_diff_ci95": list(expected_ci),
        "r1_minus_r0_optimal_arm_rate": float(
            optimal_rate_diff.mean()
        ),
        "r0_override_rate": float(r0["override_rate"].mean()),
        "r1_override_rate": float(r1["override_rate"].mean()),
        "r1_minus_r0_override_rate": float(override_diff.mean()),
        "mean_alternative_opportunity_rate": float(
            r0["alternative_opportunity_rate"].mean()
        ),
        "r0_mean_raw_reward": float(r0["mean_raw_reward"].mean()),
        "r1_mean_raw_reward": float(r1["mean_raw_reward"].mean()),
        "r0_raw_reward_std_mean": float(r0["raw_reward_std"].mean()),
        "r1_raw_reward_std_mean": float(r1["raw_reward_std"].mean()),
        "r1_reward_clip_rate": float(r1["reward_clip_rate"].mean()),
        "expected_success_probability_clip_rate": float(
            r0["expected_success_probability_clip_rate"].mean()
        ),
    }

    paired_stats = pd.DataFrame([stats])
    paired_stats.to_csv(
        RESULT_DIR / "paired_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Learning-curve figure from raw attempt data.
    # --------------------------------------------------------
    trajectory = (
        attempt_df.groupby(
            ["reward_mode", "attempt"],
            as_index=False,
        )
        .agg(
            mean_attempt_regret=("eligible_oracle_regret", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for reward_mode in REWARD_MODES:
        rows = trajectory[
            trajectory["reward_mode"] == reward_mode
        ].sort_values("attempt")

        cumulative = rows["mean_attempt_regret"].cumsum()
        ax.plot(
            rows["attempt"],
            cumulative,
            label=reward_mode,
        )

    ax.set_xlabel("Attempt")
    ax.set_ylabel("Mean cumulative eligible-oracle regret")
    ax.set_title(
        "Frozen 9-D C3: binary success vs raw signed mastery-gain reward"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "cumulative_regret_trajectory.png",
        dpi=180,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Raw reward distributions kept separate because their scales differ.
    # --------------------------------------------------------
    for reward_mode in REWARD_MODES:
        values = attempt_df.loc[
            attempt_df["reward_mode"] == reward_mode,
            "reward",
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=30)
        ax.set_xlabel("Observed terminal reward")
        ax.set_ylabel("Count")
        ax.set_title(reward_mode)
        fig.tight_layout()

        safe_name = reward_mode.lower()
        fig.savefig(
            FIGURE_DIR / f"{safe_name}_reward_distribution.png",
            dpi=180,
        )
        plt.close(fig)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------
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
        "scientific_question": (
            "With the current 9-D C3 context and frozen LinTS settings, "
            "can the policy learn effectively from a small signed continuous "
            "mastery-gain reward compared with the current binary reward?"
        ),
        "claim_boundary": (
            "This is a pre-integration synthetic signed-reward capacity test. "
            "It does not establish the empirical distribution or educational "
            "validity of Malindu delta_mastery."
        ),
        "context": {
            "name": "C3_9D",
            "features": list(C3_FEATURE_NAMES),
        },
        "reward_modes": {
            "R0_BINARY_SUCCESS": (
                "Bernoulli terminal success with probability equal to the "
                "attempt's mean selected expected-success value."
            ),
            "R1_RAW_DELTA_MASTERY": (
                f"Signed continuous proxy: {DELTA_GAIN_SCALE}*(p-0.5) "
                f"+ Normal(0,{DELTA_NOISE_STD}); only the theoretical "
                "mastery-difference bound [-1,+1] is enforced."
            ),
        },
        "important_equivalence": (
            "Both rewards are generated from the same latent expected-success "
            "probability, so their expected arm ordering is identical. The "
            "experiment isolates the effect of reward representation/scale "
            "under the frozen LinTS hyperparameters."
        ),
        "signed_reward_implementation": (
            "R0 uses canonical TrueDisjointLinTS. R1 uses an experiment-only "
            "subclass that overrides only _validate_reward to permit finite "
            "rewards in [-1,1]. The inherited posterior update remains "
            "A <- A + w*x*x^T and b <- b + w*r*x."
        ),
        "n_seeds": N_SEEDS,
        "attempts_per_seed": N_ATTEMPTS,
        "turns_per_attempt": TURNS_PER_ATTEMPT,
        "late_window_attempts": LATE_WINDOW,
        "gap_threshold": GAP_THRESHOLD,
        "ridge_lambda": RIDGE_LAMBDA,
        "exploration_scale": EXPLORATION_SCALE,
        "delayed_credit": "Equal normalized 1/T; no within-attempt updates.",
        "environment_coefficient_seed": ENV_COEFFICIENT_SEED,
        "world_seed_base": WORLD_SEED_BASE,
        "policy_seed_base": POLICY_SEED_BASE,
        "coefficient_block_l1": BLOCK_L1,
        "source_sha256": {
            name: sha256_file(path)
            for name, path in source_paths.items()
        },
        "result_files": {
            "raw_attempt_results": str(
                raw_attempt_path.relative_to(REPO_ROOT)
            ),
            "raw_turn_results": str(
                raw_turn_path.relative_to(REPO_ROOT)
            ),
            "seed_summary": str(
                (RESULT_DIR / "seed_summary.csv").relative_to(REPO_ROOT)
            ),
            "paired_statistics": str(
                (RESULT_DIR / "paired_statistics.csv").relative_to(REPO_ROOT)
            ),
            "figures": [
                str(
                    (
                        FIGURE_DIR
                        / "cumulative_regret_trajectory.png"
                    ).relative_to(REPO_ROOT)
                ),
                str(
                    (
                        FIGURE_DIR
                        / "r0_binary_success_reward_distribution.png"
                    ).relative_to(REPO_ROOT)
                ),
                str(
                    (
                        FIGURE_DIR
                        / "r1_raw_delta_mastery_reward_distribution.png"
                    ).relative_to(REPO_ROOT)
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
        240,
    ):
        print(paired_stats.to_string(index=False))

    print()
    print("Saved:")
    print(" ", raw_attempt_path)
    print(" ", raw_turn_path)
    print(" ", RESULT_DIR / "seed_summary.csv")
    print(" ", RESULT_DIR / "paired_statistics.csv")
    print(" ", manifest_path)
    print(" ", FIGURE_DIR / "cumulative_regret_trajectory.png")


if __name__ == "__main__":
    main()
