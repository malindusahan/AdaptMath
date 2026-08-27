import json
from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# SI5-B
#
# SYNTHETIC SIMULATION ONLY.
#
# The invented utilities below are NOT educational claims.
# They are used only to test whether the contextual bandit
# can learn context-dependent policy overlays when outcomes
# depend on REALIZED move changes.
#
# This learned state must NEVER be used in the real system.
# ============================================================


SEED = 42
rng = np.random.default_rng(SEED)


RESULT_PATH = Path(
    "results/self_improvement/"
    "si5b_realized_policy_effect_simulation.json"
)


MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias",
]


ARM_TO_MOVE = {
    "baseline": None,
    "probing_bias": "probing",
    "focus_bias": "focus",
    "telling_bias": "telling",
    "generic_bias": "generic",
}


SKILLS = [
    "fraction_multiplication",
    "word_problem_reasoning",
    "basic_division",
]


NUM_ATTEMPTS = 4500

ALPHA = 0.80


# Simulation-only value.
#
# The real value must later be calibrated using
# integrated validation data.
#
MAX_PROBABILITY_GAP = 0.15


# Context:
#
# [bias,
#  mastery_before,
#  attempt_index_normalized,
#  previous_evaluator_rate]
#
CONTEXT_DIM = 4


def build_context(
    mastery_before,
    attempt_index,
    previous_evaluator_rate,
):

    return np.array(
        [
            1.0,

            float(
                mastery_before
            ),

            min(
                float(
                    attempt_index
                ),
                3.0,
            )
            / 3.0,

            float(
                previous_evaluator_rate
            ),
        ],
        dtype=np.float64,
    )


# ============================================================
# DISJOINT LinUCB
# ============================================================


class DisjointLinUCB:

    def __init__(
        self,
        arms,
        context_dim,
        alpha,
    ):

        self.arms = list(
            arms
        )

        self.context_dim = int(
            context_dim
        )

        self.alpha = float(
            alpha
        )

        self.A = {
            arm: np.eye(
                self.context_dim,
                dtype=np.float64,
            )
            for arm in self.arms
        }

        self.b = {
            arm: np.zeros(
                self.context_dim,
                dtype=np.float64,
            )
            for arm in self.arms
        }

        self.counts = Counter()

        self.total_updates = 0


    def score_arm(
        self,
        arm,
        context,
    ):

        A_inv = np.linalg.inv(
            self.A[
                arm
            ]
        )

        theta = (
            A_inv
            @
            self.b[
                arm
            ]
        )

        expected_reward = float(
            theta
            @
            context
        )

        uncertainty = float(
            np.sqrt(
                context
                @
                A_inv
                @
                context
            )
        )

        ucb_score = (
            expected_reward
            +
            self.alpha
            * uncertainty
        )

        return {
            "expected_reward":
                expected_reward,

            "uncertainty":
                uncertainty,

            "ucb_score":
                ucb_score,
        }


    def select_arm(
        self,
        context,
    ):

        scores = {
            arm: self.score_arm(
                arm,
                context,
            )
            for arm in self.arms
        }

        selected_arm = max(
            self.arms,
            key=lambda arm: (
                scores[
                    arm
                ][
                    "ucb_score"
                ],
                -self.arms.index(
                    arm
                ),
            ),
        )

        return (
            selected_arm,
            scores,
        )


    def update(
        self,
        arm,
        context,
        reward,
    ):

        self.A[
            arm
        ] += np.outer(
            context,
            context,
        )

        self.b[
            arm
        ] += (
            float(
                reward
            )
            *
            context
        )

        self.counts[
            arm
        ] += 1

        self.total_updates += 1


# ============================================================
# ONE BANDIT PER SKILL
#
# This is only the SI5-B prototype.
# ============================================================


bandits = {
    skill:
        DisjointLinUCB(
            arms=ARMS,
            context_dim=CONTEXT_DIM,
            alpha=ALPHA,
        )

    for skill in SKILLS
}


# ============================================================
# SYNTHETIC FROZEN SELECTOR
# ============================================================


def simulate_base_probs():

    values = rng.dirichlet(
        np.array(
            [
                1.6,
                4.6,
                4.8,
                3.0,
            ],
            dtype=np.float64,
        )
    )

    return {
        move: float(
            value
        )

        for move, value
        in zip(
            MOVES,
            values,
        )
    }


# ============================================================
# CONSERVATIVE OVERLAY
# ============================================================


def apply_policy_overlay(
    base_probs,
    arm,
):

    base_move = max(
        MOVES,
        key=lambda move:
            base_probs[
                move
            ],
    )

    preferred_move = (
        ARM_TO_MOVE[
            arm
        ]
    )

    # Baseline means no modification.
    if preferred_move is None:

        return (
            base_move,
            False,
            base_move,
        )


    # Already chosen by frozen selector.
    if preferred_move == base_move:

        return (
            base_move,
            False,
            base_move,
        )


    probability_gap = (
        base_probs[
            base_move
        ]
        -
        base_probs[
            preferred_move
        ]
    )


    # Frozen model is sufficiently more confident
    # in its original decision.
    if (
        probability_gap
        >
        MAX_PROBABILITY_GAP
    ):

        return (
            base_move,
            False,
            base_move,
        )


    # Conservative reranking is allowed.
    return (
        preferred_move,
        True,
        base_move,
    )


# ============================================================
# HIDDEN SYNTHETIC MOVE UTILITIES
#
# IMPORTANT:
#
# These values are INVENTED.
#
# Their only purpose is to create a simulated world
# where different pedagogical preferences work in
# different contexts.
#
# The bandit does not see these values.
# ============================================================


def hidden_move_utility(
    skill,
    mastery,
    move,
):

    if (
        skill
        ==
        "fraction_multiplication"
    ):

        values = {
            "probing":
                1.05
                -
                0.75
                * mastery,

            "focus":
                0.25
                +
                0.78
                * mastery,

            "telling":
                0.30,

            "generic":
                0.15,
        }


    elif (
        skill
        ==
        "word_problem_reasoning"
    ):

        values = {
            "focus":
                1.00
                -
                0.65
                * mastery,

            "probing":
                0.25
                +
                0.70
                * mastery,

            "telling":
                0.25,

            "generic":
                0.10,
        }


    elif (
        skill
        ==
        "basic_division"
    ):

        values = {
            "telling":
                1.10
                -
                1.05
                * mastery,

            "probing":
                0.08
                +
                0.92
                * mastery,

            "focus":
                0.20
                +
                0.25
                * mastery,

            "generic":
                0.08,
        }


    else:

        raise ValueError(
            f"Unknown skill: {skill}"
        )


    return float(
        np.clip(
            values[
                move
            ],
            0.05,
            0.95,
        )
    )


# ============================================================
# SYNTHETIC EXTERNAL FEEDBACK
#
# CRITICAL DIFFERENCE FROM SI5-A:
#
# This function receives the REALIZED final moves.
#
# It does NOT receive selected_arm.
#
# Therefore an overlay gets no invented advantage
# simply because it was selected.
# ============================================================


def simulate_external_feedback(
    skill,
    mastery_before,
    baseline_moves,
    final_moves,
):

    baseline_utility = float(
        np.mean(
            [
                hidden_move_utility(
                    skill,
                    mastery_before,
                    move,
                )

                for move
                in baseline_moves
            ]
        )
    )


    realized_utility = float(
        np.mean(
            [
                hidden_move_utility(
                    skill,
                    mastery_before,
                    move,
                )

                for move
                in final_moves
            ]
        )
    )


    realized_policy_lift = (
        realized_utility
        -
        baseline_utility
    )


    evaluator_probability = float(
        np.clip(
            0.08
            +
            0.88
            * realized_utility,

            0.05,
            0.95,
        )
    )


    evaluator_score = int(
        rng.binomial(
            n=3,
            p=evaluator_probability,
        )
    )


    evaluator_rate = (
        evaluator_score
        / 3.0
    )


    # Synthetic mastery signal.
    #
    # Again, based on realized behavior,
    # NOT on arm identity.
    expected_mastery_gain = (
        0.015
        +
        0.16
        *
        (
            realized_utility
            -
            0.45
        )
    )


    mastery_delta = float(
        expected_mastery_gain
        +
        rng.normal(
            0.0,
            0.02,
        )
    )


    mastery_after = float(
        np.clip(
            mastery_before
            +
            mastery_delta,

            0.0,
            1.0,
        )
    )


    mastery_delta = (
        mastery_after
        -
        mastery_before
    )


    return {
        "baseline_utility":
            baseline_utility,

        "realized_utility":
            realized_utility,

        "realized_policy_lift":
            realized_policy_lift,

        "evaluator_probability":
            evaluator_probability,

        "evaluator_score":
            evaluator_score,

        "evaluator_rate":
            evaluator_rate,

        "mastery_after":
            mastery_after,

        "mastery_delta":
            mastery_delta,
    }


# ============================================================
# SIMULATION
# ============================================================


records = []

mid_attempt_updates = 0

policy_freeze_errors = 0

zero_override_nonzero_lift = 0

override_count = 0

total_turns = 0


previous_state = {
    skill: {
        "attempt_index":
            0,

        "evaluator_rate":
            0.0,
    }

    for skill
    in SKILLS
}


print(
    "=" * 70
)

print(
    "SI5-B - REALIZED POLICY EFFECT BANDIT SIMULATION"
)

print(
    "=" * 70
)

print(
    "\nSIMULATION ONLY - NO REAL TRAINING DATA"
)


for global_attempt in range(
    NUM_ATTEMPTS
):

    skill = SKILLS[
        global_attempt
        %
        len(
            SKILLS
        )
    ]


    mastery_before = float(
        rng.uniform(
            0.15,
            0.85,
        )
    )


    previous = (
        previous_state[
            skill
        ]
    )


    context = build_context(
        mastery_before=
            mastery_before,

        attempt_index=
            previous[
                "attempt_index"
            ],

        previous_evaluator_rate=
            previous[
                "evaluator_rate"
            ],
    )


    bandit = bandits[
        skill
    ]


    # ========================================================
    # SELECT ONE ARM BEFORE ATTEMPT
    # ========================================================

    selected_arm, arm_scores = (
        bandit.select_arm(
            context
        )
    )


    updates_before_attempt = (
        bandit.total_updates
    )


    # ========================================================
    # RUN COMPLETE ATTEMPT
    # ========================================================

    number_of_turns = int(
        rng.integers(
            4,
            9,
        )
    )


    baseline_moves = []

    final_moves = []

    turns = []

    attempt_overrides = 0


    for turn_index in range(
        number_of_turns
    ):

        base_probs = (
            simulate_base_probs()
        )


        (
            final_move,
            overridden,
            base_move,
        ) = apply_policy_overlay(
            base_probs=
                base_probs,

            arm=
                selected_arm,
        )


        baseline_moves.append(
            base_move
        )

        final_moves.append(
            final_move
        )


        if overridden:

            attempt_overrides += 1

            override_count += 1


        total_turns += 1


        turns.append({
            "turn_index":
                turn_index,

            "base_move":
                base_move,

            "final_move":
                final_move,

            "overlay_arm":
                selected_arm,

            "overridden":
                bool(
                    overridden
                ),
        })


        # No learning is allowed here.
        if (
            bandit.total_updates
            !=
            updates_before_attempt
        ):

            mid_attempt_updates += 1


        # Same overlay must be used
        # throughout this attempt.
        if (
            turns[
                -1
            ][
                "overlay_arm"
            ]
            !=
            selected_arm
        ):

            policy_freeze_errors += 1


    # ========================================================
    # ATTEMPT ENDS
    #
    # Only now do evaluator/mastery arrive.
    # ========================================================

    feedback = (
        simulate_external_feedback(
            skill=
                skill,

            mastery_before=
                mastery_before,

            baseline_moves=
                baseline_moves,

            final_moves=
                final_moves,
        )
    )


    # --------------------------------------------------------
    # Critical integrity test:
    #
    # If overlay changed zero moves, realized
    # behavior must exactly equal baseline behavior.
    # --------------------------------------------------------

    if (
        attempt_overrides
        == 0
        and
        abs(
            feedback[
                "realized_policy_lift"
            ]
        )
        >
        1e-12
    ):

        zero_override_nonzero_lift += 1


    # ========================================================
    # BANDIT REWARD
    #
    # External evaluator only.
    # ========================================================

    reward = (
        feedback[
            "evaluator_rate"
        ]
    )


    # ========================================================
    # ONE UPDATE AFTER ATTEMPT
    # ========================================================

    bandit.update(
        arm=
            selected_arm,

        context=
            context,

        reward=
            reward,
    )


    assert (
        bandit.total_updates
        ==
        updates_before_attempt
        + 1
    )


    if (
        feedback[
            "evaluator_score"
        ]
        < 3
    ):

        next_attempt_index = (
            previous[
                "attempt_index"
            ]
            + 1
        )

    else:

        next_attempt_index = 0


    previous_state[
        skill
    ] = {
        "attempt_index":
            next_attempt_index,

        "evaluator_rate":
            feedback[
                "evaluator_rate"
            ],
    }


    records.append({
        "global_attempt":
            global_attempt,

        "skill_id":
            skill,

        "mastery_before":
            mastery_before,

        "attempt_index":
            previous[
                "attempt_index"
            ],

        "selected_arm":
            selected_arm,

        "num_turns":
            number_of_turns,

        "num_overrides":
            attempt_overrides,

        "reward":
            reward,

        "evaluator_score":
            feedback[
                "evaluator_score"
            ],

        "mastery_delta":
            feedback[
                "mastery_delta"
            ],

        "realized_policy_lift":
            feedback[
                "realized_policy_lift"
            ],

        "turns":
            turns,
    })


# ============================================================
# VALIDATION 1 - UPDATE TIMING
# ============================================================


actual_updates = sum(
    bandit.total_updates

    for bandit
    in bandits.values()
)


timing_passed = (
    actual_updates
    ==
    NUM_ATTEMPTS

    and

    mid_attempt_updates
    ==
    0
)


print(
    "\n"
    + "=" * 70
)

print(
    "UPDATE-TIMING VALIDATION"
)

print(
    "=" * 70
)

print(
    "\nAttempts:",
    NUM_ATTEMPTS
)

print(
    "Expected updates:",
    NUM_ATTEMPTS
)

print(
    "Observed updates:",
    actual_updates
)

print(
    "Mid-attempt updates:",
    mid_attempt_updates
)

print(
    "Validation:",
    (
        "PASSED"
        if timing_passed
        else "FAILED"
    )
)


# ============================================================
# VALIDATION 2 - POLICY FREEZE
# ============================================================


print(
    "\n"
    + "=" * 70
)

print(
    "WITHIN-ATTEMPT POLICY FREEZE"
)

print(
    "=" * 70
)

print(
    "\nOverlay-change errors:",
    policy_freeze_errors
)

print(
    "Validation:",
    (
        "PASSED"
        if policy_freeze_errors == 0
        else "FAILED"
    )
)


# ============================================================
# VALIDATION 3 - REALIZED EFFECT
# ============================================================


attempts_with_override = sum(
    int(
        row[
            "num_overrides"
        ]
        >
        0
    )

    for row
    in records
)


override_rate = (
    override_count
    /
    total_turns
)


print(
    "\n"
    + "=" * 70
)

print(
    "REALIZED-POLICY-EFFECT VALIDATION"
)

print(
    "=" * 70
)

print(
    "\nTotal turns:",
    total_turns
)

print(
    "Total overrides:",
    override_count
)

print(
    "Turn override rate:",
    f"{override_rate * 100:.2f}%"
)

print(
    "Attempts with >= 1 override:",
    attempts_with_override
)

print(
    "Zero-override attempts with non-zero policy lift:",
    zero_override_nonzero_lift
)

realized_effect_passed = (
    zero_override_nonzero_lift
    ==
    0
)

print(
    "Validation:",
    (
        "PASSED"
        if realized_effect_passed
        else "FAILED"
    )
)


# ============================================================
# LEARNING SANITY CHECK
# ============================================================


WINDOW = 600


def summarize(
    rows
):

    return {
        "mean_reward":
            float(
                np.mean(
                    [
                        row[
                            "reward"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

        "success_3_of_3_rate":
            float(
                np.mean(
                    [
                        row[
                            "evaluator_score"
                        ]
                        ==
                        3

                        for row
                        in rows
                    ]
                )
            ),

        "mean_mastery_delta":
            float(
                np.mean(
                    [
                        row[
                            "mastery_delta"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

        "mean_realized_policy_lift":
            float(
                np.mean(
                    [
                        row[
                            "realized_policy_lift"
                        ]

                        for row
                        in rows
                    ]
                )
            ),
    }


early_summary = summarize(
    records[
        :WINDOW
    ]
)


late_summary = summarize(
    records[
        -WINDOW:
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "SIMULATION LEARNING SANITY CHECK"
)

print(
    "=" * 70
)

print(
    f"\nEarly window: first {WINDOW} attempts"
)

print(
    "  Mean evaluator reward:",
    f"{early_summary['mean_reward']:.4f}"
)

print(
    "  3/3 success rate:",
    f"{early_summary['success_3_of_3_rate'] * 100:.2f}%"
)

print(
    "  Mean mastery delta:",
    f"{early_summary['mean_mastery_delta']:+.4f}"
)

print(
    "  Mean realized policy lift:",
    f"{early_summary['mean_realized_policy_lift']:+.4f}"
)


print(
    f"\nLate window: last {WINDOW} attempts"
)

print(
    "  Mean evaluator reward:",
    f"{late_summary['mean_reward']:.4f}"
)

print(
    "  3/3 success rate:",
    f"{late_summary['success_3_of_3_rate'] * 100:.2f}%"
)

print(
    "  Mean mastery delta:",
    f"{late_summary['mean_mastery_delta']:+.4f}"
)

print(
    "  Mean realized policy lift:",
    f"{late_summary['mean_realized_policy_lift']:+.4f}"
)


# ============================================================
# CONTEXT-SENSITIVITY AUDIT
#
# Look only at recent observations from clear
# low/high mastery regions.
# ============================================================


EXPECTED = {
    (
        "fraction_multiplication",
        "low",
    ):
        "probing_bias",

    (
        "fraction_multiplication",
        "high",
    ):
        "focus_bias",

    (
        "word_problem_reasoning",
        "low",
    ):
        "focus_bias",

    (
        "word_problem_reasoning",
        "high",
    ):
        "probing_bias",

    (
        "basic_division",
        "low",
    ):
        "telling_bias",

    (
        "basic_division",
        "high",
    ):
        "probing_bias",
}


context_results = {}

context_checks_passed = True


print(
    "\n"
    + "=" * 70
)

print(
    "CONTEXT-SENSITIVITY AUDIT"
)

print(
    "=" * 70
)


for skill in SKILLS:

    for mastery_group in [
        "low",
        "high",
    ]:

        if mastery_group == "low":

            candidates = [
                row

                for row
                in records

                if (
                    row[
                        "skill_id"
                    ]
                    ==
                    skill

                    and

                    row[
                        "mastery_before"
                    ]
                    <
                    0.40
                )
            ]

        else:

            candidates = [
                row

                for row
                in records

                if (
                    row[
                        "skill_id"
                    ]
                    ==
                    skill

                    and

                    row[
                        "mastery_before"
                    ]
                    >
                    0.60
                )
            ]


        recent = candidates[
            -200:
        ]


        counts = Counter(
            row[
                "selected_arm"
            ]

            for row
            in recent
        )


        most_selected = max(
            ARMS,
            key=lambda arm:
                counts[
                    arm
                ],
        )


        expected_arm = EXPECTED[
            (
                skill,
                mastery_group,
            )
        ]


        passed = (
            most_selected
            ==
            expected_arm
        )


        if not passed:

            context_checks_passed = False


        key = (
            f"{skill}"
            f"__{mastery_group}"
        )


        context_results[
            key
        ] = {
            "n":
                len(
                    recent
                ),

            "expected_arm":
                expected_arm,

            "most_selected_arm":
                most_selected,

            "passed":
                bool(
                    passed
                ),

            "counts": {
                arm:
                    int(
                        counts[
                            arm
                        ]
                    )

                for arm
                in ARMS
            },
        }


        print(
            f"\n{skill} / {mastery_group} mastery"
        )

        print(
            "  Expected:",
            expected_arm
        )

        print(
            "  Most selected:",
            most_selected
        )

        for arm in ARMS:

            print(
                f"  {arm:15s}: "
                f"{counts[arm]}"
            )

        print(
            "  Check:",
            (
                "PASSED"
                if passed
                else "FAILED"
            )
        )


# ============================================================
# SAVE
# ============================================================


all_passed = (
    timing_passed
    and
    policy_freeze_errors
    ==
    0
    and
    realized_effect_passed
    and
    context_checks_passed
)


result = {
    "experiment":
        "SI5-B_realized_policy_effect_simulation",

    "simulation_only":
        True,

    "real_training_data_used":
        False,

    "algorithm":
        "disjoint_LinUCB",

    "num_attempts":
        NUM_ATTEMPTS,

    "alpha":
        ALPHA,

    "max_probability_gap":
        MAX_PROBABILITY_GAP,

    "primary_reward":
        "evaluator_score / 3",

    "mastery_in_reward":
        False,

    "outcome_depends_on_selected_arm_directly":
        False,

    "update_timing":
        "after_attempt_only",

    "policy_fixed_within_attempt":
        True,

    "timing_validation_passed":
        bool(
            timing_passed
        ),

    "policy_freeze_errors":
        int(
            policy_freeze_errors
        ),

    "zero_override_nonzero_lift":
        int(
            zero_override_nonzero_lift
        ),

    "total_turns":
        int(
            total_turns
        ),

    "total_overrides":
        int(
            override_count
        ),

    "override_rate":
        float(
            override_rate
        ),

    "early_summary":
        early_summary,

    "late_summary":
        late_summary,

    "context_sensitivity":
        context_results,

    "context_checks_passed":
        bool(
            context_checks_passed
        ),

    "overall_validation_passed":
        bool(
            all_passed
        ),

    "research_warning":
        (
            "All outcomes and hidden utilities are synthetic "
            "and are used only for algorithm validation. "
            "Reset all bandit state before real integration."
        ),
}


RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with RESULT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print(
    "\n"
    + "=" * 70
)

print(
    "SI5-B STATUS"
)

print(
    "=" * 70
)

print(
    "\nOVERALL VALIDATION:",
    (
        "PASSED"
        if all_passed
        else "FAILED"
    )
)

print(
    "\nIMPORTANT:"
)

print(
    "This simulation does NOT demonstrate educational improvement."
)

print(
    "It only validates the contextual-bandit mechanics."
)

print(
    "The learned synthetic bandit state must be discarded "
    "before real integration."
)

print(
    "\nSaved:"
)

print(
    RESULT_PATH
)
