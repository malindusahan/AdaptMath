import json
from pathlib import Path

import numpy as np

from src.self_improvement.lints_policy import LinTSPolicy


# ============================================================
# CONFIGURATION
# ============================================================

ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias",
]


CONTEXT_GROUPS = [
    "fraction_low",
    "fraction_high",
    "word_low",
    "word_high",
    "division_low",
    "division_high",
]


N_RUNS = 10

ATTEMPTS_PER_RUN = 3000

EARLY_WINDOW = 600

LATE_WINDOW = 600


# Thompson Sampling exploration scale.
#
# This is a synthetic algorithm-validation setting.
# It is NOT a frozen production hyperparameter.
ALPHA = 0.35


# Small observation noise so reward is not deterministic.
REWARD_NOISE_STD = 0.05


RESULT_PATH = Path(
    "results/self_improvement/"
    "si6e_lints_convergence.json"
)


# ============================================================
# PLANTED SYNTHETIC REWARD ENVIRONMENT
# ============================================================

"""
Each context has a known expected reward for each bandit arm.

The values are synthetic.

They are NOT claims about real tutoring effectiveness.

The purpose is to create a controlled environment where we
know the correct answer in advance and can determine whether
LinTS learns it.
"""


EXPECTED_REWARDS = {

    "fraction_low": {
        "baseline": 0.55,
        "probing_bias": 0.78,
        "focus_bias": 0.62,
        "telling_bias": 0.50,
        "generic_bias": 0.48,
    },

    "fraction_high": {
        "baseline": 0.58,
        "probing_bias": 0.62,
        "focus_bias": 0.80,
        "telling_bias": 0.50,
        "generic_bias": 0.48,
    },

    "word_low": {
        "baseline": 0.54,
        "probing_bias": 0.60,
        "focus_bias": 0.79,
        "telling_bias": 0.52,
        "generic_bias": 0.47,
    },

    "word_high": {
        "baseline": 0.57,
        "probing_bias": 0.79,
        "focus_bias": 0.63,
        "telling_bias": 0.50,
        "generic_bias": 0.48,
    },

    "division_low": {
        "baseline": 0.53,
        "probing_bias": 0.60,
        "focus_bias": 0.58,
        "telling_bias": 0.79,
        "generic_bias": 0.47,
    },

    "division_high": {
        "baseline": 0.58,
        "probing_bias": 0.80,
        "focus_bias": 0.62,
        "telling_bias": 0.52,
        "generic_bias": 0.48,
    },
}


OPTIMAL_ARMS = {

    context_name:
        max(
            arm_rewards,
            key=arm_rewards.get,
        )

    for context_name, arm_rewards
    in EXPECTED_REWARDS.items()
}


# ============================================================
# CONTEXT REPRESENTATION
# ============================================================

def build_context_vector(
    context_name,
):
    """
    Synthetic one-hot context representation.

    There are six possible context groups.

    Using one-hot context features here is deliberate:
    the planted reward function becomes exactly representable
    by the disjoint linear bandit.

    Therefore this experiment tests LinTS convergence itself,
    rather than testing whether some arbitrary feature
    representation can approximate the reward function.
    """

    if context_name not in CONTEXT_GROUPS:
        raise ValueError(
            f"Unknown context: {context_name}"
        )

    vector = np.zeros(
        len(CONTEXT_GROUPS),
        dtype=np.float64,
    )

    index = CONTEXT_GROUPS.index(
        context_name
    )

    vector[index] = 1.0

    return vector


# ============================================================
# SYNTHETIC REWARD
# ============================================================

def sample_reward(
    rng,
    context_name,
    selected_arm,
):
    """
    Generate noisy realized reward.

    Expected reward comes from the planted reward table.

    Noise is intentionally small so learning remains possible,
    but the environment is not perfectly deterministic.
    """

    expected_reward = (
        EXPECTED_REWARDS[
            context_name
        ][
            selected_arm
        ]
    )

    realized_reward = rng.normal(
        loc=expected_reward,
        scale=REWARD_NOISE_STD,
    )

    realized_reward = float(
        np.clip(
            realized_reward,
            0.0,
            1.0,
        )
    )

    return (
        expected_reward,
        realized_reward,
    )


# ============================================================
# CONTEXT SCHEDULE
# ============================================================

def build_balanced_schedule(
    rng,
):
    """
    Build a balanced context schedule.

    Every synthetic context appears approximately equally often.

    Since 3000 is divisible by 6, every context occurs exactly
    500 times per run before shuffling.
    """

    repetitions = (
        ATTEMPTS_PER_RUN
        //
        len(CONTEXT_GROUPS)
    )

    schedule = (
        CONTEXT_GROUPS
        *
        repetitions
    )

    remainder = (
        ATTEMPTS_PER_RUN
        -
        len(schedule)
    )

    if remainder > 0:
        schedule.extend(
            CONTEXT_GROUPS[
                :remainder
            ]
        )

    schedule = np.array(
        schedule,
        dtype=object,
    )

    rng.shuffle(
        schedule
    )

    return schedule.tolist()


# ============================================================
# ONE BANDIT RUN
# ============================================================

def run_single_simulation(
    run_index,
):
    """
    Run one independent LinTS learning experiment.
    """

    policy_seed = (
        10000
        +
        run_index
    )

    environment_seed = (
        20000
        +
        run_index
    )

    environment_rng = (
        np.random.default_rng(
            environment_seed
        )
    )

    policy = LinTSPolicy(
        n_features=len(
            CONTEXT_GROUPS
        ),
        arms=ARMS,
        alpha=ALPHA,
        seed=policy_seed,
    )

    schedule = build_balanced_schedule(
        environment_rng
    )

    records = []

    for attempt_index, context_name in enumerate(
        schedule
    ):

        context_vector = (
            build_context_vector(
                context_name
            )
        )

        selected_arm = (
            policy.select_arm(
                context_vector
            )
        )

        (
            selected_expected_reward,
            realized_reward,
        ) = sample_reward(
            rng=environment_rng,
            context_name=context_name,
            selected_arm=selected_arm,
        )

        optimal_arm = (
            OPTIMAL_ARMS[
                context_name
            ]
        )

        optimal_expected_reward = (
            EXPECTED_REWARDS[
                context_name
            ][
                optimal_arm
            ]
        )

        baseline_expected_reward = (
            EXPECTED_REWARDS[
                context_name
            ][
                "baseline"
            ]
        )

        expected_regret = (
            optimal_expected_reward
            -
            selected_expected_reward
        )

        policy.update(
            arm=selected_arm,
            context=context_vector,
            reward=realized_reward,
        )

        records.append(
            {
                "attempt_index":
                    attempt_index,

                "context":
                    context_name,

                "selected_arm":
                    selected_arm,

                "optimal_arm":
                    optimal_arm,

                "selected_expected_reward":
                    float(
                        selected_expected_reward
                    ),

                "baseline_expected_reward":
                    float(
                        baseline_expected_reward
                    ),

                "optimal_expected_reward":
                    float(
                        optimal_expected_reward
                    ),

                "realized_reward":
                    realized_reward,

                "expected_regret":
                    float(
                        expected_regret
                    ),

                "optimal_selected":
                    bool(
                        selected_arm
                        ==
                        optimal_arm
                    ),
            }
        )

    return records


# ============================================================
# METRICS
# ============================================================

def mean_value(
    values,
):
    return float(
        np.mean(
            values
        )
    )


def summarize_window(
    records,
):
    """
    Summarize one collection of attempts.
    """

    optimal_rate = mean_value(
        [
            float(
                record[
                    "optimal_selected"
                ]
            )

            for record
            in records
        ]
    )

    expected_regret = mean_value(
        [
            record[
                "expected_regret"
            ]

            for record
            in records
        ]
    )

    realized_reward = mean_value(
        [
            record[
                "realized_reward"
            ]

            for record
            in records
        ]
    )

    selected_expected_reward = (
        mean_value(
            [
                record[
                    "selected_expected_reward"
                ]

                for record
                in records
            ]
        )
    )

    baseline_expected_reward = (
        mean_value(
            [
                record[
                    "baseline_expected_reward"
                ]

                for record
                in records
            ]
        )
    )

    return {
        "optimal_arm_rate":
            optimal_rate,

        "mean_expected_regret":
            expected_regret,

        "mean_realized_reward":
            realized_reward,

        "mean_selected_expected_reward":
            selected_expected_reward,

        "mean_baseline_expected_reward":
            baseline_expected_reward,

        "expected_uplift_vs_baseline":
            (
                selected_expected_reward
                -
                baseline_expected_reward
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "SI6-E TRUE LinTS CONVERGENCE EXPERIMENT"
    )
    print("=" * 72)

    print()

    print(
        "Synthetic experiment only."
    )

    print(
        "No real educational claims are made."
    )

    print()

    print(
        "Runs:",
        N_RUNS,
    )

    print(
        "Attempts per run:",
        ATTEMPTS_PER_RUN,
    )

    print(
        "Total decisions:",
        (
            N_RUNS
            *
            ATTEMPTS_PER_RUN
        ),
    )

    print(
        "LinTS alpha:",
        ALPHA,
    )

    print(
        "Reward noise std:",
        REWARD_NOISE_STD,
    )

    print()

    print(
        "Planted optimal policy:"
    )

    for context_name in CONTEXT_GROUPS:

        print(
            f"  {context_name:<20}"
            f" -> "
            f"{OPTIMAL_ARMS[context_name]}"
        )

    print()

    all_runs = []

    total_updates = 0

    for run_index in range(
        N_RUNS
    ):

        records = (
            run_single_simulation(
                run_index
            )
        )

        all_runs.append(
            records
        )

        total_updates += len(
            records
        )

        print(
            f"Completed run "
            f"{run_index + 1}"
            f"/"
            f"{N_RUNS}"
        )

    print()

    expected_updates = (
        N_RUNS
        *
        ATTEMPTS_PER_RUN
    )

    assert (
        total_updates
        ==
        expected_updates
    )

    print(
        "Update count:",
        total_updates,
    )

    print(
        "Update timing/count: PASSED"
    )

    # --------------------------------------------------------
    # Aggregate early and late windows across all runs.
    # --------------------------------------------------------

    early_records = []

    late_records = []

    all_records = []

    for records in all_runs:

        early_records.extend(
            records[
                :EARLY_WINDOW
            ]
        )

        late_records.extend(
            records[
                -LATE_WINDOW:
            ]
        )

        all_records.extend(
            records
        )

    early_summary = (
        summarize_window(
            early_records
        )
    )

    late_summary = (
        summarize_window(
            late_records
        )
    )

    overall_summary = (
        summarize_window(
            all_records
        )
    )

    cumulative_expected_regret = float(
        sum(
            record[
                "expected_regret"
            ]

            for record
            in all_records
        )
    )

    # --------------------------------------------------------
    # Late performance by context.
    # --------------------------------------------------------

    context_results = {}

    for context_name in CONTEXT_GROUPS:

        relevant = [
            record

            for record
            in late_records

            if (
                record[
                    "context"
                ]
                ==
                context_name
            )
        ]

        optimal_rate = mean_value(
            [
                float(
                    record[
                        "optimal_selected"
                    ]
                )

                for record
                in relevant
            ]
        )

        context_results[
            context_name
        ] = {
            "planted_optimal_arm":
                OPTIMAL_ARMS[
                    context_name
                ],

            "late_examples":
                len(
                    relevant
                ),

            "late_optimal_arm_rate":
                optimal_rate,
        }

    # --------------------------------------------------------
    # Print convergence metrics.
    # --------------------------------------------------------

    print()
    print("-" * 72)
    print("EARLY VS LATE")
    print("-" * 72)

    print(
        "Early optimal-arm rate:",
        round(
            early_summary[
                "optimal_arm_rate"
            ],
            4,
        ),
    )

    print(
        "Late optimal-arm rate:",
        round(
            late_summary[
                "optimal_arm_rate"
            ],
            4,
        ),
    )

    print()

    print(
        "Early mean expected regret:",
        round(
            early_summary[
                "mean_expected_regret"
            ],
            6,
        ),
    )

    print(
        "Late mean expected regret:",
        round(
            late_summary[
                "mean_expected_regret"
            ],
            6,
        ),
    )

    print()

    print(
        "Early realized reward:",
        round(
            early_summary[
                "mean_realized_reward"
            ],
            4,
        ),
    )

    print(
        "Late realized reward:",
        round(
            late_summary[
                "mean_realized_reward"
            ],
            4,
        ),
    )

    print()

    print(
        "Late expected reward:",
        round(
            late_summary[
                "mean_selected_expected_reward"
            ],
            4,
        ),
    )

    print(
        "Late baseline expected reward:",
        round(
            late_summary[
                "mean_baseline_expected_reward"
            ],
            4,
        ),
    )

    print(
        "Late expected uplift vs baseline:",
        round(
            late_summary[
                "expected_uplift_vs_baseline"
            ],
            4,
        ),
    )

    print()

    print(
        "Cumulative expected regret:",
        round(
            cumulative_expected_regret,
            4,
        ),
    )

    # --------------------------------------------------------
    # Context sensitivity
    # --------------------------------------------------------

    print()
    print("-" * 72)
    print(
        "LATE OPTIMAL-ARM RATE BY CONTEXT"
    )
    print("-" * 72)

    for context_name in CONTEXT_GROUPS:

        result = (
            context_results[
                context_name
            ]
        )

        print(
            f"{context_name:<20}"
            f" target="
            f"{result['planted_optimal_arm']:<15}"
            f" rate="
            f"{result['late_optimal_arm_rate']:.4f}"
        )

    # --------------------------------------------------------
    # Validation criteria
    # --------------------------------------------------------

    checks = {}

    checks[
        "late_optimal_rate_above_85_percent"
    ] = bool(
        late_summary[
            "optimal_arm_rate"
        ]
        >=
        0.85
    )

    checks[
        "optimal_rate_improves_by_at_least_10_points"
    ] = bool(
        late_summary[
            "optimal_arm_rate"
        ]
        >=
        (
            early_summary[
                "optimal_arm_rate"
            ]
            +
            0.10
        )
    )

    checks[
        "late_regret_lower_than_early_regret"
    ] = bool(
        late_summary[
            "mean_expected_regret"
        ]
        <
        early_summary[
            "mean_expected_regret"
        ]
    )

    checks[
        "late_policy_beats_baseline_expectation"
    ] = bool(
        late_summary[
            "mean_selected_expected_reward"
        ]
        >
        late_summary[
            "mean_baseline_expected_reward"
        ]
    )

    checks[
        "all_contexts_above_75_percent_optimal"
    ] = bool(
        all(
            context_results[
                context_name
            ][
                "late_optimal_arm_rate"
            ]
            >=
            0.75

            for context_name
            in CONTEXT_GROUPS
        )
    )

    print()
    print("-" * 72)
    print("VALIDATION CHECKS")
    print("-" * 72)

    for check_name, passed in checks.items():

        print(
            f"{check_name:<50}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    overall_passed = bool(
        all(
            checks.values()
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_payload = {

        "experiment":
            "si6e_true_lints_convergence",

        "data_origin":
            "synthetic",

        "educational_claim_allowed":
            False,

        "configuration": {
            "n_runs":
                N_RUNS,

            "attempts_per_run":
                ATTEMPTS_PER_RUN,

            "total_decisions":
                expected_updates,

            "early_window":
                EARLY_WINDOW,

            "late_window":
                LATE_WINDOW,

            "alpha":
                ALPHA,

            "reward_noise_std":
                REWARD_NOISE_STD,

            "arms":
                ARMS,

            "contexts":
                CONTEXT_GROUPS,
        },

        "planted_optimal_policy":
            OPTIMAL_ARMS,

        "early":
            early_summary,

        "late":
            late_summary,

        "overall":
            overall_summary,

        "cumulative_expected_regret":
            cumulative_expected_regret,

        "late_context_results":
            context_results,

        "validation_checks":
            checks,

        "validation_passed":
            overall_passed,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_payload,
            f,
            indent=2,
        )

    print()

    print(
        "Saved:",
        RESULT_PATH,
    )

    print()

    print("=" * 72)

    if overall_passed:

        print(
            "SI6-E CONVERGENCE VALIDATION PASSED"
        )

    else:

        print(
            "SI6-E CONVERGENCE VALIDATION FAILED"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()