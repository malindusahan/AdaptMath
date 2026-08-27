import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


# ============================================================
# IMPORTANT
#
# THIS SCRIPT USES SYNTHETIC / SIMULATED OUTCOMES ONLY.
#
# Purpose:
#   - validate bandit mechanics
#   - validate update timing
#   - validate policy freezing within attempts
#   - validate exploration / learning behavior
#
# It is NOT research training data.
# It is NOT evidence of educational improvement.
#
# The real research bandit must be RESET before integration.
# ============================================================


SEED = 42

rng = np.random.default_rng(
    SEED
)


RESULT_PATH = Path(
    "results/self_improvement/"
    "si5_attempt_level_bandit_simulation.json"
)


# ============================================================
# CONFIG
# ============================================================

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


NUM_ATTEMPTS = 900


# LinUCB exploration coefficient.
#
# Simulation value only.
#
ALPHA = 0.75


# Conservative overlay rule:
#
# An arm can only replace the frozen selector's
# top move when its preferred move is within this
# absolute probability gap.
#
# Simulation value only.
#
MAX_PROBABILITY_GAP = 0.10


# ============================================================
# CONTEXT
#
# Separate bandit instance per skill_id.
#
# Numeric context:
#
# 1. bias
# 2. mastery_before
# 3. attempt_index / 3
# 4. previous evaluator rate
# 5. previous mastery delta
# ============================================================

CONTEXT_DIM = 5


def build_context(
    mastery_before,
    attempt_index,
    previous_evaluator_rate,
    previous_mastery_delta,
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

            float(
                np.clip(
                    previous_mastery_delta,
                    -1.0,
                    1.0,
                )
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

            arm:
                np.eye(
                    self.context_dim,
                    dtype=np.float64,
                )

            for arm in self.arms
        }


        self.b = {

            arm:
                np.zeros(
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

            arm:
                self.score_arm(
                    arm,
                    context,
                )

            for arm in self.arms
        }


        # Deterministic tie-breaking uses ARMS order.
        selected = max(

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
            selected,
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
# CONSERVATIVE FROZEN-SELECTOR OVERLAY
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


    # Baseline arm:
    # frozen selector decides directly.
    if preferred_move is None:

        return (
            base_move,
            False,
        )


    # If the frozen selector already prefers
    # this move, no actual override occurred.
    if preferred_move == base_move:

        return (
            base_move,
            False,
        )


    gap = (

        base_probs[
            base_move
        ]

        -

        base_probs[
            preferred_move
        ]
    )


    # Conservative fallback.
    if (
        gap
        >
        MAX_PROBABILITY_GAP
    ):

        return (
            base_move,
            False,
        )


    # Preferred move is plausible according
    # to frozen selector, so bounded reranking
    # is allowed.
    return (
        preferred_move,
        True,
    )


# ============================================================
# SYNTHETIC FROZEN SELECTOR
#
# Only used to test overlay mechanics.
# ============================================================

def simulate_base_probs():

    values = rng.dirichlet(
        np.array(
            [
                1.6,
                2.8,
                3.3,
                2.0,
            ],
            dtype=np.float64,
        )
    )


    return {

        move:
            float(
                value
            )

        for move, value
        in zip(
            MOVES,
            values,
        )
    }


# ============================================================
# SYNTHETIC ENVIRONMENT
#
# Hidden educational outcome function.
#
# IMPORTANT:
# These relationships are intentionally invented.
# They exist ONLY to check whether LinUCB can learn
# a known simulated structure.
#
# They are NOT research assumptions.
# ============================================================

def simulated_arm_quality(
    skill,
    mastery,
    arm,
):

    quality = 0.48


    # --------------------------------------------------------
    # Baseline is reasonably competent.
    # --------------------------------------------------------

    if arm == "baseline":

        quality += 0.08


    # --------------------------------------------------------
    # Invented context-dependent patterns.
    # --------------------------------------------------------

    if (
        skill
        == "fraction_multiplication"
    ):

        if mastery < 0.45:

            if arm == "probing_bias":
                quality += 0.25

            if arm == "focus_bias":
                quality += 0.15

        else:

            if arm == "focus_bias":
                quality += 0.23

            if arm == "probing_bias":
                quality += 0.08


    elif (
        skill
        == "word_problem_reasoning"
    ):

        if mastery < 0.55:

            if arm == "focus_bias":
                quality += 0.26

            if arm == "probing_bias":
                quality += 0.12

        else:

            if arm == "probing_bias":
                quality += 0.20

            if arm == "baseline":
                quality += 0.05


    elif (
        skill
        == "basic_division"
    ):

        if mastery < 0.35:

            if arm == "telling_bias":
                quality += 0.18

            if arm == "focus_bias":
                quality += 0.12

        else:

            if arm == "probing_bias":
                quality += 0.18

            if arm == "focus_bias":
                quality += 0.11


    # Generic bias intentionally weaker
    # in this synthetic world.
    if arm == "generic_bias":

        quality -= 0.08


    return float(
        np.clip(
            quality,
            0.05,
            0.95,
        )
    )


def simulate_external_feedback(
    skill,
    mastery_before,
    arm,
):

    quality = simulated_arm_quality(
        skill,
        mastery_before,
        arm,
    )


    # --------------------------------------------------------
    # Evaluator score 0â€“3
    #
    # Generate 3 Bernoulli "questions".
    # --------------------------------------------------------

    evaluator_score = int(
        rng.binomial(
            n=3,
            p=quality,
        )
    )


    evaluator_rate = (
        evaluator_score
        / 3.0
    )


    # --------------------------------------------------------
    # Synthetic mastery_after
    #
    # Again: purely for algorithm testing.
    # --------------------------------------------------------

    expected_gain = (

        0.02

        +

        0.16
        * (
            quality
            - 0.40
        )
    )


    mastery_delta = float(
        expected_gain
        +
        rng.normal(
            0.0,
            0.025,
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


    actual_delta = (

        mastery_after
        -
        mastery_before
    )


    return {
        "evaluator_score":
            evaluator_score,

        "evaluator_rate":
            evaluator_rate,

        "mastery_after":
            mastery_after,

        "mastery_delta":
            actual_delta,

        "hidden_quality":
            quality,
    }


# ============================================================
# SIMULATE COMPLETED ATTEMPTS
# ============================================================

records = []

mid_attempt_updates = 0

override_count = 0

total_turns = 0


previous_state = {

    skill: {
        "evaluator_rate":
            0.0,

        "mastery_delta":
            0.0,

        "attempt_index":
            0,
    }

    for skill in SKILLS
}


print("=" * 70)

print(
    "SI5-A â€” ATTEMPT-LEVEL CONTEXTUAL BANDIT SIMULATION"
)

print("=" * 70)


print(
    "\nSIMULATION ONLY â€” "
    "NO REAL TRAINING DATA"
)


for global_attempt in range(
    NUM_ATTEMPTS
):

    # --------------------------------------------------------
    # Simulated incoming learner / skill.
    # --------------------------------------------------------

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


    state = previous_state[
        skill
    ]


    attempt_index = int(
        state[
            "attempt_index"
        ]
    )


    context = build_context(

        mastery_before=
            mastery_before,

        attempt_index=
            attempt_index,

        previous_evaluator_rate=
            state[
                "evaluator_rate"
            ],

        previous_mastery_delta=
            state[
                "mastery_delta"
            ],
    )


    bandit = bandits[
        skill
    ]


    # ========================================================
    # SELECT POLICY ONCE AT ATTEMPT START
    # ========================================================

    selected_arm, arm_scores = (
        bandit.select_arm(
            context
        )
    )


    updates_at_attempt_start = (
        bandit.total_updates
    )


    # ========================================================
    # SIMULATE ENTIRE TUTORING ATTEMPT
    #
    # Same selected arm is used for ALL turns.
    #
    # Bandit is NOT updated here.
    # ========================================================

    num_turns = int(
        rng.integers(
            3,
            8,
        )
    )


    turn_records = []


    for turn_index in range(
        num_turns
    ):

        base_probs = (
            simulate_base_probs()
        )


        final_move, overridden = (
            apply_policy_overlay(
                base_probs,
                selected_arm,
            )
        )


        if overridden:

            override_count += 1


        total_turns += 1


        turn_records.append({

            "turn_index":
                turn_index,

            "base_probs":
                base_probs,

            "final_move":
                final_move,

            "overlay_arm":
                selected_arm,

            "overridden":
                bool(
                    overridden
                ),
        })


        # --------------------------------------------
        # Critical update-timing assertion.
        # --------------------------------------------

        if (
            bandit.total_updates
            !=
            updates_at_attempt_start
        ):

            mid_attempt_updates += 1


    # ========================================================
    # ATTEMPT FINISHES
    #
    # External components now return feedback.
    # ========================================================

    feedback = (
        simulate_external_feedback(
            skill,
            mastery_before,
            selected_arm,
        )
    )


    # ========================================================
    # PRIMARY BANDIT REWARD
    #
    # evaluator score only.
    #
    # mastery is retained separately.
    # ========================================================

    reward = feedback[
        "evaluator_rate"
    ]


    # ========================================================
    # UPDATE ONCE â€” AFTER ATTEMPT
    # ========================================================

    bandit.update(
        selected_arm,
        context,
        reward,
    )


    # Must be exactly one new update.
    assert (
        bandit.total_updates
        ==
        updates_at_attempt_start
        + 1
    )


    # --------------------------------------------------------
    # Simulate reteach context.
    #
    # If evaluator < 3, next same-skill occurrence
    # gets incremented attempt index.
    #
    # If evaluator == 3, reset to a future learner/session.
    # --------------------------------------------------------

    if (
        feedback[
            "evaluator_score"
        ]
        < 3
    ):

        next_attempt_index = (
            attempt_index
            + 1
        )

    else:

        next_attempt_index = 0


    previous_state[
        skill
    ] = {

        "evaluator_rate":
            feedback[
                "evaluator_rate"
            ],

        "mastery_delta":
            feedback[
                "mastery_delta"
            ],

        "attempt_index":
            next_attempt_index,
    }


    records.append({

        "global_attempt":
            global_attempt,

        "skill_id":
            skill,

        "mastery_before":
            mastery_before,

        "attempt_index":
            attempt_index,

        "selected_arm":
            selected_arm,

        "reward":
            reward,

        "evaluator_score":
            feedback[
                "evaluator_score"
            ],

        "mastery_after":
            feedback[
                "mastery_after"
            ],

        "mastery_delta":
            feedback[
                "mastery_delta"
            ],

        "hidden_simulated_quality":
            feedback[
                "hidden_quality"
            ],

        "num_turns":
            num_turns,

        "turns":
            turn_records,
    })


# ============================================================
# VALIDATE UPDATE TIMING
# ============================================================

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
    "\nSimulated attempts:",
    len(records),
)


print(
    "Bandit updates expected:",
    len(records),
)


actual_updates = sum(

    bandit.total_updates

    for bandit
    in bandits.values()
)


print(
    "Bandit updates observed:",
    actual_updates,
)


print(
    "Mid-attempt updates:",
    mid_attempt_updates,
)


timing_passed = (

    actual_updates
    == len(records)

    and

    mid_attempt_updates
    == 0
)


print(
    "\nTiming validation:",
    (
        "PASSED"
        if timing_passed
        else "FAILED"
    ),
)


# ============================================================
# POLICY-FREEZE VALIDATION
# ============================================================

freeze_errors = 0


for record in records:

    selected_arm = record[
        "selected_arm"
    ]


    for turn in record[
        "turns"
    ]:

        if (
            turn[
                "overlay_arm"
            ]
            != selected_arm
        ):

            freeze_errors += 1


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
    freeze_errors,
)


print(
    "Validation:",
    (
        "PASSED"
        if freeze_errors == 0
        else "FAILED"
    ),
)


# ============================================================
# CONSERVATIVE OVERLAY AUDIT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CONSERVATIVE OVERLAY AUDIT"
)

print(
    "=" * 70
)


override_rate = (

    override_count
    /
    total_turns

    if total_turns
    else 0.0
)


print(
    "\nTotal tutoring turns:",
    total_turns,
)


print(
    "Policy overrides:",
    override_count,
)


print(
    "Override rate:",
    f"{override_rate * 100:.2f}%",
)


print(
    "Max probability gap:",
    MAX_PROBABILITY_GAP,
)


# ============================================================
# EARLY VS LATE LEARNING AUDIT
#
# Simulation sanity check only.
# ============================================================

window = 150


early = records[
    :window
]

late = records[
    -window:
]


def summarize_window(
    rows
):

    rewards = np.asarray(
        [
            row[
                "reward"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    evaluator_scores = np.asarray(
        [
            row[
                "evaluator_score"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    mastery_deltas = np.asarray(
        [
            row[
                "mastery_delta"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    return {
        "mean_reward":
            float(
                np.mean(
                    rewards
                )
            ),

        "success_3_of_3_rate":
            float(
                np.mean(
                    evaluator_scores
                    == 3
                )
            ),

        "mean_mastery_delta":
            float(
                np.mean(
                    mastery_deltas
                )
            ),
    }


early_summary = summarize_window(
    early
)

late_summary = summarize_window(
    late
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
    f"\nEarly window: first {window} attempts"
)


print(
    "  Mean evaluator reward:",
    f"{early_summary['mean_reward']:.4f}",
)


print(
    "  3/3 success rate:",
    f"{early_summary['success_3_of_3_rate'] * 100:.2f}%",
)


print(
    "  Mean mastery delta:",
    f"{early_summary['mean_mastery_delta']:+.4f}",
)


print(
    f"\nLate window: last {window} attempts"
)


print(
    "  Mean evaluator reward:",
    f"{late_summary['mean_reward']:.4f}",
)


print(
    "  3/3 success rate:",
    f"{late_summary['success_3_of_3_rate'] * 100:.2f}%",
)


print(
    "  Mean mastery delta:",
    f"{late_summary['mean_mastery_delta']:+.4f}",
)


# ============================================================
# ARM USAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ARM-SELECTION AUDIT"
)

print(
    "=" * 70
)


for skill in SKILLS:

    skill_records = [

        row

        for row in records

        if row[
            "skill_id"
        ]
        == skill
    ]


    early_skill = skill_records[
        :50
    ]

    late_skill = skill_records[
        -50:
    ]


    early_counts = Counter(

        row[
            "selected_arm"
        ]

        for row
        in early_skill
    )


    late_counts = Counter(

        row[
            "selected_arm"
        ]

        for row
        in late_skill
    )


    print(
        f"\n{skill}"
    )


    print(
        "  First 50:"
    )


    for arm in ARMS:

        print(
            f"    {arm:15s}: "
            f"{early_counts[arm]}"
        )


    print(
        "  Last 50:"
    )


    for arm in ARMS:

        print(
            f"    {arm:15s}: "
            f"{late_counts[arm]}"
        )


# ============================================================
# BANDIT STATE SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL BANDIT UPDATE COUNTS"
)

print(
    "=" * 70
)


bandit_counts = {}


for skill in SKILLS:

    counts = {

        arm:
            int(
                bandits[
                    skill
                ].counts[
                    arm
                ]
            )

        for arm in ARMS
    }


    bandit_counts[
        skill
    ] = counts


    print(
        f"\n{skill}"
    )


    for arm in ARMS:

        print(
            f"  {arm:15s}: "
            f"{counts[arm]}"
        )


# ============================================================
# SAVE SIMULATION SUMMARY ONLY
# ============================================================

result = {

    "experiment":
        "SI5-A_attempt_level_contextual_bandit_simulation",

    "simulation_only":
        True,

    "real_training_data_used":
        False,

    "algorithm":
        "disjoint_LinUCB",

    "arms":
        ARMS,

    "primary_reward":
        "evaluator_score / 3",

    "mastery_in_primary_reward":
        False,

    "update_timing":
        "once_after_attempt",

    "policy_fixed_within_attempt":
        True,

    "attempts":
        len(records),

    "timing_validation_passed":
        bool(
            timing_passed
        ),

    "mid_attempt_updates":
        int(
            mid_attempt_updates
        ),

    "policy_freeze_errors":
        int(
            freeze_errors
        ),

    "total_turns":
        int(
            total_turns
        ),

    "overrides":
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

    "bandit_update_counts":
        bandit_counts,

    "research_warning":
        (
            "Synthetic outcomes are for implementation "
            "validation only. Reset bandit before real "
            "integrated tutoring experiments."
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
    "SIMULATION STATUS"
)

print(
    "=" * 70
)


if (
    timing_passed
    and
    freeze_errors == 0
):

    print(
        "\nIMPLEMENTATION VALIDATION: PASSED"
    )

else:

    print(
        "\nIMPLEMENTATION VALIDATION: FAILED"
    )


print(
    "\nIMPORTANT:"
)

print(
    "These simulated learning numbers are NOT "
    "research performance results."
)

print(
    "The real bandit must be reset before "
    "integrated data collection."
)


print(
    "\nSaved:"
)

print(
    RESULT_PATH
)
