import inspect
import json
from pathlib import Path

import numpy as np

from src.self_improvement.attempt_level_bandit import (
    ARMS,
    ARM_TO_MOVE,
    CONTEXT_FEATURE_NAMES,
    MOVES,
    AttemptLevelBanditPolicy,
)


SPEC_PATH = Path(
    "results/self_improvement/"
    "frozen_adaptive_policy_spec.json"
)


SI6E_PATH = Path(
    "results/self_improvement/"
    "si6e_lints_convergence.json"
)

SI7A_PATH = Path(
    "results/self_improvement/"
    "si7a_frozen_selector_recovery_validation.json"
)

SI7C_PATH = Path(
    "results/self_improvement/"
    "si7c_real_frozen_overlay_validation.json"
)

SI7D_PATH = Path(
    "results/self_improvement/"
    "si7d_recovered_selector_attempt_lints_integration.json"
)


EXPECTED_MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


EXPECTED_ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias",
]


EXPECTED_ARM_TO_MOVE = {
    "baseline": None,
    "probing_bias": "probing",
    "focus_bias": "focus",
    "telling_bias": "telling",
    "generic_bias": "generic",
}


EXPECTED_CONTEXT_FEATURES = [
    "bias",
    "mastery_before",
    "attempt_index_normalized",
    "has_previous_feedback",
    "previous_evaluator_rate",
    "previous_mastery_delta",
]


EXPECTED_MAX_GAP = 0.15

EXPECTED_EXPLORATION_SCALE = 0.50


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def main():

    print("=" * 78)
    print(
        "SI8-A FROZEN ADAPTIVE POLICY SPECIFICATION VALIDATION"
    )
    print("=" * 78)
    print()

    # ========================================================
    # TEST 1
    # SPEC EXISTS
    # ========================================================

    if not SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Frozen specification not found: {SPEC_PATH}"
        )

    spec = load_json(
        SPEC_PATH
    )

    assert (
        spec[
            "status"
        ]
        ==
        "frozen_primary_architecture"
    )

    print(
        "Frozen specification found: PASSED"
    )

    # ========================================================
    # TEST 2
    # BASE SELECTOR SPEC
    # ========================================================

    base_selector = (
        spec[
            "base_selector"
        ]
    )

    assert (
        base_selector[
            "architecture"
        ]
        ==
        "context_routed_hybrid"
    )

    assert (
        base_selector[
            "move_space"
        ]
        ==
        EXPECTED_MOVES
    )

    routing = (
        base_selector[
            "routing"
        ]
    )

    assert (
        routing[
            "routing_tokenizer_source"
        ]
        ==
        "MD4_DistilRoBERTa"
    )

    assert (
        routing[
            "threshold_tokens"
        ]
        ==
        512
    )

    validation = (
        base_selector[
            "validation"
        ]
    )

    assert (
        validation[
            "examples"
        ]
        ==
        1850
    )

    assert (
        validation[
            "md4_route_examples"
        ]
        ==
        1507
    )

    assert (
        validation[
            "md3_route_examples"
        ]
        ==
        343
    )

    assert np.isclose(
        validation[
            "accuracy"
        ],
        0.5005405405405405,
        atol=1e-12,
    )

    assert np.isclose(
        validation[
            "macro_f1"
        ],
        0.48676777751987244,
        atol=1e-12,
    )

    print(
        "Frozen selector specification: PASSED"
    )

    # ========================================================
    # TEST 3
    # ACTION SPACE MATCHES IMPLEMENTATION
    # ========================================================

    assert list(MOVES) == EXPECTED_MOVES

    assert list(ARMS) == EXPECTED_ARMS

    assert dict(
        ARM_TO_MOVE
    ) == EXPECTED_ARM_TO_MOVE

    adaptive = (
        spec[
            "adaptive_policy"
        ]
    )

    assert (
        adaptive[
            "action_space"
        ]
        ==
        EXPECTED_ARMS
    )

    assert (
        adaptive[
            "arm_to_move"
        ]
        ==
        EXPECTED_ARM_TO_MOVE
    )

    assert (
        adaptive[
            "algorithm"
        ]
        ==
        "disjoint_linear_thompson_sampling"
    )

    assert (
        adaptive[
            "level"
        ]
        ==
        "attempt"
    )

    print(
        "Action-space implementation match: PASSED"
    )

    # ========================================================
    # TEST 4
    # CONTEXT MATCHES IMPLEMENTATION
    # ========================================================

    assert (
        list(
            CONTEXT_FEATURE_NAMES
        )
        ==
        EXPECTED_CONTEXT_FEATURES
    )

    assert (
        spec[
            "attempt_context"
        ][
            "feature_names"
        ]
        ==
        EXPECTED_CONTEXT_FEATURES
    )

    print(
        "Attempt context implementation match: PASSED"
    )

    # ========================================================
    # TEST 5
    # CONSTRUCTOR DEFAULTS MATCH FROZEN SPEC
    # ========================================================

    signature = inspect.signature(
        AttemptLevelBanditPolicy.__init__
    )

    default_max_gap = (
        signature.parameters[
            "max_probability_gap"
        ].default
    )

    default_lints_scale = (
        signature.parameters[
            "lints_exploration_scale"
        ].default
    )

    assert np.isclose(
        float(
            default_max_gap
        ),
        EXPECTED_MAX_GAP,
        atol=1e-12,
    )

    assert np.isclose(
        float(
            default_lints_scale
        ),
        EXPECTED_EXPLORATION_SCALE,
        atol=1e-12,
    )

    assert np.isclose(
        adaptive[
            "conservative_overlay"
        ][
            "max_probability_gap"
        ],
        EXPECTED_MAX_GAP,
        atol=1e-12,
    )

    assert np.isclose(
        adaptive[
            "lints"
        ][
            "exploration_scale"
        ],
        EXPECTED_EXPLORATION_SCALE,
        atol=1e-12,
    )

    print(
        "Primary hyperparameter defaults: PASSED"
    )

    # ========================================================
    # TEST 6
    # LIFECYCLE + REWARD FORMULA
    #
    # Fresh policy:
    # one attempt
    # multiple turns
    # evaluator score 2
    #
    # Expected reward = 2/3.
    # ========================================================

    policy = AttemptLevelBanditPolicy(
        algorithm="lints",
        seed=2026,
        data_origin="synthetic",
    )

    start = policy.start_attempt(
        skill_id="si8a_validation_skill",
        mastery_before=0.40,
        attempt_index=0,
    )

    selected_arm = (
        start[
            "selected_arm"
        ]
    )

    assert selected_arm in EXPECTED_ARMS

    bandit = (
        policy.bandits[
            "si8a_validation_skill"
        ]
    )

    context = (
        policy.active_attempt[
            "context"
        ].copy()
    )

    assert (
        bandit.total_updates
        ==
        0
    )

    initial_A = {
        arm:
            bandit.A[
                arm
            ].copy()

        for arm in EXPECTED_ARMS
    }

    initial_b = {
        arm:
            bandit.b[
                arm
            ].copy()

        for arm in EXPECTED_ARMS
    }

    base_probs = {
        "generic": 0.30,
        "probing": 0.28,
        "focus": 0.25,
        "telling": 0.17,
    }

    # Multiple tutoring turns.
    for _ in range(3):

        active_arm_before = (
            policy.active_attempt[
                "selected_arm"
            ]
        )

        assert (
            active_arm_before
            ==
            selected_arm
        )

        updates_before = (
            bandit.total_updates
        )

        policy.apply_overlay(
            base_probs
        )

        assert (
            policy.active_attempt[
                "selected_arm"
            ]
            ==
            selected_arm
        )

        # Absolutely no turn-level learning.
        assert (
            bandit.total_updates
            ==
            updates_before
        )

    assert (
        bandit.total_updates
        ==
        0
    )

    # Terminal feedback.
    policy.finish_attempt(
        evaluator_score=2,
        mastery_after=0.90,
    )

    assert (
        bandit.total_updates
        ==
        1
    )

    assert (
        policy.active_attempt
        is None
    )

    expected_reward = (
        2.0
        /
        3.0
    )

    expected_A = (
        initial_A[
            selected_arm
        ]
        +
        np.outer(
            context,
            context,
        )
    )

    expected_b = (
        initial_b[
            selected_arm
        ]
        +
        expected_reward
        *
        context
    )

    assert np.allclose(
        bandit.A[
            selected_arm
        ],
        expected_A,
    )

    assert np.allclose(
        bandit.b[
            selected_arm
        ],
        expected_b,
    )

    for arm in EXPECTED_ARMS:

        if arm == selected_arm:
            continue

        assert np.allclose(
            bandit.A[
                arm
            ],
            initial_A[
                arm
            ],
        )

        assert np.allclose(
            bandit.b[
                arm
            ],
            initial_b[
                arm
            ],
        )

    print(
        "One-update attempt lifecycle: PASSED"
    )

    print(
        "Evaluator-score reward formula: PASSED"
    )

    # ========================================================
    # TEST 7
    # MASTERY_AFTER DOES NOT CHANGE PRIMARY REWARD
    #
    # Two identical policies receive:
    # same context
    # same selected arm
    # same evaluator score
    #
    # but very different mastery_after.
    #
    # Their LinTS update must be identical.
    # ========================================================

    policy_a = AttemptLevelBanditPolicy(
        algorithm="lints",
        seed=777,
        data_origin="synthetic",
    )

    policy_b = AttemptLevelBanditPolicy(
        algorithm="lints",
        seed=777,
        data_origin="synthetic",
    )

    start_a = policy_a.start_attempt(
        skill_id="mastery_reward_check",
        mastery_before=0.40,
        attempt_index=0,
    )

    start_b = policy_b.start_attempt(
        skill_id="mastery_reward_check",
        mastery_before=0.40,
        attempt_index=0,
    )

    assert (
        start_a[
            "selected_arm"
        ]
        ==
        start_b[
            "selected_arm"
        ]
    )

    policy_a.finish_attempt(
        evaluator_score=2,
        mastery_after=0.41,
    )

    policy_b.finish_attempt(
        evaluator_score=2,
        mastery_after=0.95,
    )

    bandit_a = (
        policy_a.bandits[
            "mastery_reward_check"
        ]
    )

    bandit_b = (
        policy_b.bandits[
            "mastery_reward_check"
        ]
    )

    assert (
        bandit_a.total_updates
        ==
        1
    )

    assert (
        bandit_b.total_updates
        ==
        1
    )

    for arm in EXPECTED_ARMS:

        assert np.allclose(
            bandit_a.A[
                arm
            ],
            bandit_b.A[
                arm
            ],
        )

        assert np.allclose(
            bandit_a.b[
                arm
            ],
            bandit_b.b[
                arm
            ],
        )

    assert (
        spec[
            "reward"
        ][
            "mastery_in_primary_reward"
        ]
        is False
    )

    print(
        "Mastery excluded from primary reward: PASSED"
    )

    # ========================================================
    # TEST 8
    # STRUCTURAL FROZEN DECISIONS
    # ========================================================

    assert (
        adaptive[
            "arm_selection_timing"
        ]
        ==
        "once_at_attempt_start"
    )

    assert (
        adaptive[
            "arm_fixed_during_attempt"
        ]
        is True
    )

    assert (
        adaptive[
            "turn_level_arm_reselection"
        ]
        is False
    )

    assert (
        adaptive[
            "turn_level_bandit_updates"
        ]
        is False
    )

    assert (
        adaptive[
            "updates_per_completed_attempt"
        ]
        ==
        1
    )

    assert (
        spec[
            "reward"
        ][
            "credit_assignment_level"
        ]
        ==
        "attempt"
    )

    print(
        "Frozen temporal-credit architecture: PASSED"
    )

    # ========================================================
    # TEST 9
    # NON-PRIMARY ALTERNATIVES ARE EXPLICITLY EXCLUDED
    # ========================================================

    excluded = (
        spec[
            "not_in_primary_architecture"
        ]
    )

    assert (
        excluded[
            "weighted_evaluator_mastery_reward"
        ][
            "status"
        ]
        ==
        "provisional_experiment_not_used"
    )

    assert (
        excluded[
            "direct_move_bandit_arms"
        ][
            "status"
        ]
        ==
        "not_primary"
    )

    assert (
        excluded[
            "turn_level_pedagogical_reward_model"
        ][
            "used_in_primary_policy"
        ]
        is False
    )

    assert (
        excluded[
            "turn_level_lints_updates"
        ][
            "used_in_primary_policy"
        ]
        is False
    )

    print(
        "Alternative-design exclusions: PASSED"
    )

    # ========================================================
    # TEST 10
    # PRIOR VALIDATION EVIDENCE
    # ========================================================

    required_reports = [
        SI6E_PATH,
        SI7A_PATH,
        SI7C_PATH,
        SI7D_PATH,
    ]

    for path in required_reports:

        if not path.exists():

            raise FileNotFoundError(
                f"Required validation evidence "
                f"is missing: {path}"
            )

    si6e = load_json(
        SI6E_PATH
    )

    si7a = load_json(
        SI7A_PATH
    )

    si7c = load_json(
        SI7C_PATH
    )

    si7d = load_json(
        SI7D_PATH
    )

    assert (
        si6e[
            "validation_passed"
        ]
        is True
    )

    assert (
        si7a[
            "status"
        ]
        ==
        "passed"
    )

    assert (
        si7c[
            "validation_passed"
        ]
        is True
    )

    assert (
        si7d[
            "validation_passed"
        ]
        is True
    )

    assert np.isclose(
        si7a[
            "recovered_hybrid_metrics"
        ][
            "accuracy"
        ],
        0.5005405405405405,
        atol=1e-12,
    )

    assert np.isclose(
        si7a[
            "recovered_hybrid_metrics"
        ][
            "macro_f1"
        ],
        0.48676777751987244,
        atol=1e-12,
    )

    assert (
        si7d[
            "architecture_under_test"
        ][
            "adaptive_level"
        ]
        ==
        "attempt"
    )

    assert (
        si7d[
            "architecture_under_test"
        ][
            "reward"
        ]
        ==
        "evaluator_score_divided_by_3"
    )

    assert (
        si7d[
            "total_updates"
        ]
        ==
        2
    )

    print(
        "Prior validation evidence: PASSED"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("-" * 78)

    print(
        "Frozen architecture:"
    )

    print(
        "  Base selector : context-routed MD4 / MD3 hybrid"
    )

    print(
        "  Adaptation    : attempt-level Disjoint LinTS"
    )

    print(
        "  Arms          : residual bias arms + baseline"
    )

    print(
        "  Safety gate   : probability gap <= 0.15"
    )

    print(
        "  Reward        : evaluator_score / 3"
    )

    print(
        "  Mastery reward: NO"
    )

    print(
        "  Updates       : exactly one per completed attempt"
    )

    print("-" * 78)

    print()

    print("=" * 78)
    print(
        "SI8-A FROZEN ADAPTIVE POLICY SPECIFICATION VALIDATION PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()