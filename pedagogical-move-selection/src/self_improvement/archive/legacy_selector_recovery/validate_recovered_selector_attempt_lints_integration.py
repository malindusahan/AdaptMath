import json
from collections import Counter, defaultdict
from pathlib import Path

from src.self_improvement.attempt_level_bandit import (
    AttemptLevelBanditPolicy,
    ARMS,
)
from src.self_improvement.frozen_selector_replay import (
    FrozenSelectorReplay,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_PATH = Path(
    "results/self_improvement/"
    "si7d_recovered_selector_attempt_lints_integration.json"
)


SKILL_ID = "si7d_synthetic_integration_skill"


# Existing reusable SI5 defaults.
LINTS_EXPLORATION_SCALE = 0.50

MAX_PROBABILITY_GAP = 0.15

SEED = 2026


# ============================================================
# SELECT A REAL REPLAY TRAJECTORY
# ============================================================

def find_mixed_route_qid(
    selector,
):
    """
    Find one validation dialogue/qid that contains both:

        short_md4
        long_md3

    This lets one synthetic integration attempt exercise both
    branches of the recovered frozen selector.

    We are NOT treating this dialogue as a learning-outcome
    dataset. We only replay its real selector probabilities.
    """

    by_qid = defaultdict(list)

    for example_id in selector.example_ids():

        prediction = selector.predict(
            example_id
        )

        by_qid[
            prediction["qid"]
        ].append(
            {
                "example_id":
                    example_id,

                "prediction":
                    prediction,
            }
        )

    for qid, examples in by_qid.items():

        routes = {
            item[
                "prediction"
            ][
                "route"
            ]
            for item in examples
        }

        if (
            "short_md4" in routes
            and
            "long_md3" in routes
            and
            len(examples) >= 3
        ):
            return (
                qid,
                examples,
            )

    raise RuntimeError(
        "Could not find a validation qid containing "
        "both MD4 and MD3 routing branches."
    )


# ============================================================
# RUN ONE ATTEMPT
# ============================================================

def run_attempt(
    policy,
    selector_examples,
    attempt_index,
    mastery_before,
    evaluator_score,
    mastery_after,
    previous_evaluator_score=None,
    previous_mastery_delta=None,
):
    """
    Run one synthetic attempt lifecycle while replaying
    REAL frozen-selector validation probabilities.

    No tutor response is generated here.

    This test isolates:

        attempt start
        -> one LinTS arm
        -> real frozen probabilities
        -> conservative overlay
        -> no mid-attempt learning
        -> one terminal update
    """

    start_result = policy.start_attempt(
        skill_id=SKILL_ID,
        mastery_before=mastery_before,
        attempt_index=attempt_index,
        previous_evaluator_score=
            previous_evaluator_score,
        previous_mastery_delta=
            previous_mastery_delta,
    )

    selected_arm = (
        start_result[
            "selected_arm"
        ]
    )

    assert selected_arm in ARMS

    assert (
        policy.active_attempt[
            "selected_arm"
        ]
        ==
        selected_arm
    )

    bandit = (
        policy.bandits[
            SKILL_ID
        ]
    )

    updates_at_start = (
        bandit.total_updates
    )

    active_context = (
        policy.active_attempt[
            "context"
        ].copy()
    )

    overlay_records = []

    route_counts = Counter()

    override_count = 0

    # --------------------------------------------------------
    # Every replayed turn receives the SAME selected arm.
    #
    # That is the crucial attempt-level property.
    # --------------------------------------------------------

    for turn_number, item in enumerate(
        selector_examples
    ):

        example_id = (
            item[
                "example_id"
            ]
        )

        frozen_prediction = (
            item[
                "prediction"
            ]
        )

        assert (
            policy.active_attempt[
                "selected_arm"
            ]
            ==
            selected_arm
        )

        updates_before_turn = (
            bandit.total_updates
        )

        overlay = (
            policy.apply_overlay(
                frozen_prediction[
                    "probabilities"
                ]
            )
        )

        # apply_overlay() must never update LinTS.
        assert (
            bandit.total_updates
            ==
            updates_before_turn
        )

        assert (
            bandit.total_updates
            ==
            updates_at_start
        )

        # Arm must remain fixed across every turn.
        assert (
            policy.active_attempt[
                "selected_arm"
            ]
            ==
            selected_arm
        )

        route_counts[
            frozen_prediction[
                "route"
            ]
        ] += 1

        if overlay["overridden"]:
            override_count += 1

        overlay_records.append(
            {
                "turn_number":
                    turn_number,

                "example_id":
                    example_id,

                "qid":
                    frozen_prediction[
                        "qid"
                    ],

                "route":
                    frozen_prediction[
                        "route"
                    ],

                "source_model":
                    frozen_prediction[
                        "source_model"
                    ],

                "selected_arm":
                    selected_arm,

                "frozen_move":
                    frozen_prediction[
                        "predicted_move"
                    ],

                "base_move":
                    overlay[
                        "base_move"
                    ],

                "final_move":
                    overlay[
                        "final_move"
                    ],

                "overridden":
                    bool(
                        overlay[
                            "overridden"
                        ]
                    ),

                "probability_gap":
                    overlay[
                        "probability_gap"
                    ],
            }
        )

    # --------------------------------------------------------
    # Still zero new updates before finish_attempt().
    # --------------------------------------------------------

    assert (
        bandit.total_updates
        ==
        updates_at_start
    )

    turn_count_before_finish = (
        policy.active_attempt[
            "turn_count"
        ]
    )

    override_count_before_finish = (
        policy.active_attempt[
            "override_count"
        ]
    )

    assert (
        turn_count_before_finish
        ==
        len(
            selector_examples
        )
    )

    assert (
        override_count_before_finish
        ==
        override_count
    )

    # ========================================================
    # TERMINAL ATTEMPT FEEDBACK
    #
    # This is the ONE AND ONLY learning point.
    #
    # evaluator_score and mastery_after are synthetic here.
    # ========================================================

    policy.finish_attempt(
        evaluator_score=
            evaluator_score,

        mastery_after=
            mastery_after,
    )

    # Exactly one update should have occurred.
    assert (
        bandit.total_updates
        ==
        updates_at_start
        +
        1
    )

    # Attempt must be closed.
    assert (
        policy.active_attempt
        is None
    )

    expected_reward = (
        evaluator_score
        /
        3.0
    )

    mastery_delta = (
        mastery_after
        -
        mastery_before
    )

    return {
        "attempt_index":
            attempt_index,

        "selected_arm":
            selected_arm,

        "context":
            active_context.tolist(),

        "turn_count":
            len(
                selector_examples
            ),

        "override_count":
            override_count,

        "override_rate":
            (
                override_count
                /
                len(
                    selector_examples
                )
            ),

        "route_counts":
            dict(
                route_counts
            ),

        "evaluator_score":
            evaluator_score,

        "reward":
            expected_reward,

        "mastery_before":
            mastery_before,

        "mastery_after":
            mastery_after,

        "mastery_delta":
            mastery_delta,

        "updates_before":
            updates_at_start,

        "updates_after":
            bandit.total_updates,

        "overlays":
            overlay_records,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SI7-D RECOVERED FROZEN SELECTOR + ATTEMPT-LEVEL TRUE LinTS"
    )
    print("=" * 78)

    print()

    print(
        "Selector probabilities:"
        " REAL saved validation predictions"
    )

    print(
        "Bandit feedback:"
        " SYNTHETIC integration-test outcomes"
    )

    print(
        "Bandit state:"
        " disposable; not persisted"
    )

    print(
        "Educational-effectiveness claim:"
        " NONE"
    )

    print()

    # ========================================================
    # LOAD REAL RECOVERED SELECTOR
    # ========================================================

    selector = (
        FrozenSelectorReplay()
    )

    assert len(selector) == 1850

    (
        selected_qid,
        selector_examples,
    ) = find_mixed_route_qid(
        selector
    )

    print(
        "Selected replay qid:",
        selected_qid,
    )

    print(
        "Turns in replay attempt:",
        len(
            selector_examples
        ),
    )

    replay_route_counts = Counter(
        item[
            "prediction"
        ][
            "route"
        ]
        for item
        in selector_examples
    )

    print(
        "Replay route counts:",
        dict(
            replay_route_counts
        ),
    )

    assert (
        replay_route_counts[
            "short_md4"
        ]
        >
        0
    )

    assert (
        replay_route_counts[
            "long_md3"
        ]
        >
        0
    )

    print(
        "Both frozen routing branches available: PASSED"
    )

    # ========================================================
    # CREATE EXISTING RESEARCH-HARDENED POLICY
    # ========================================================

    policy = (
        AttemptLevelBanditPolicy(
            algorithm="lints",

            lints_exploration_scale=
                LINTS_EXPLORATION_SCALE,

            max_probability_gap=
                MAX_PROBABILITY_GAP,

            seed=SEED,

            # IMPORTANT:
            #
            # Although selector probabilities come from real
            # validation artifacts, terminal rewards below are
            # synthetic.
            #
            # Therefore this bandit state MUST remain synthetic.
            data_origin="synthetic",
        )
    )

    print()

    print(
        "Policy algorithm:",
        policy.algorithm,
    )

    assert (
        policy.algorithm
        ==
        "lints"
    )

    print(
        "True LinTS policy construction: PASSED"
    )

    # ========================================================
    # ATTEMPT 0
    # ========================================================

    print()
    print("-" * 78)
    print(
        "ATTEMPT 0"
    )
    print("-" * 78)

    attempt_0 = run_attempt(
        policy=policy,

        selector_examples=
            selector_examples,

        attempt_index=0,

        mastery_before=0.40,

        evaluator_score=2,

        mastery_after=0.55,
    )

    print(
        "Selected arm:",
        attempt_0[
            "selected_arm"
        ],
    )

    print(
        "Turns:",
        attempt_0[
            "turn_count"
        ],
    )

    print(
        "Overrides:",
        attempt_0[
            "override_count"
        ],
    )

    print(
        "Reward:",
        attempt_0[
            "reward"
        ],
    )

    print(
        "Updates:",
        attempt_0[
            "updates_before"
        ],
        "->",
        attempt_0[
            "updates_after"
        ],
    )

    assert (
        attempt_0[
            "updates_after"
        ]
        -
        attempt_0[
            "updates_before"
        ]
        ==
        1
    )

    print(
        "Attempt 0 one-update lifecycle: PASSED"
    )

    # ========================================================
    # ATTEMPT 1 / RETEACH
    #
    # Previous evaluator/mastery feedback is supplied only
    # when the next attempt begins.
    # ========================================================

    print()
    print("-" * 78)
    print(
        "ATTEMPT 1 / RETEACH"
    )
    print("-" * 78)

    attempt_1 = run_attempt(
        policy=policy,

        selector_examples=
            selector_examples,

        attempt_index=1,

        mastery_before=0.55,

        previous_evaluator_score=2,

        previous_mastery_delta=0.15,

        evaluator_score=3,

        mastery_after=0.65,
    )

    print(
        "Selected arm:",
        attempt_1[
            "selected_arm"
        ],
    )

    print(
        "Turns:",
        attempt_1[
            "turn_count"
        ],
    )

    print(
        "Overrides:",
        attempt_1[
            "override_count"
        ],
    )

    print(
        "Reward:",
        attempt_1[
            "reward"
        ],
    )

    print(
        "Updates:",
        attempt_1[
            "updates_before"
        ],
        "->",
        attempt_1[
            "updates_after"
        ],
    )

    assert (
        attempt_1[
            "updates_after"
        ]
        -
        attempt_1[
            "updates_before"
        ]
        ==
        1
    )

    # Two completed attempts total.
    final_bandit = (
        policy.bandits[
            SKILL_ID
        ]
    )

    assert (
        final_bandit.total_updates
        ==
        2
    )

    print(
        "Attempt 1 one-update lifecycle: PASSED"
    )

    print(
        "Total bandit updates after two attempts:",
        final_bandit.total_updates,
    )

    # ========================================================
    # CRITICAL INTEGRATION CHECKS
    # ========================================================

    print()
    print("-" * 78)
    print(
        "INTEGRATION CHECKS"
    )
    print("-" * 78)

    # All replay probabilities came from the real recovered
    # hybrid rather than dummy selector outputs.
    assert (
        attempt_0[
            "route_counts"
        ][
            "short_md4"
        ]
        >
        0
    )

    assert (
        attempt_0[
            "route_counts"
        ][
            "long_md3"
        ]
        >
        0
    )

    print(
        "Real MD4/MD3 probabilities consumed: PASSED"
    )

    # Same selected arm must be present on every turn.
    for attempt in [
        attempt_0,
        attempt_1,
    ]:

        selected_arm = (
            attempt[
                "selected_arm"
            ]
        )

        for overlay in attempt[
            "overlays"
        ]:

            assert (
                overlay[
                    "selected_arm"
                ]
                ==
                selected_arm
            )

    print(
        "One fixed arm per attempt: PASSED"
    )

    # We already checked updates remain unchanged after every
    # apply_overlay() call inside run_attempt().
    print(
        "No turn-level bandit updates: PASSED"
    )

    assert (
        final_bandit.total_updates
        ==
        2
    )

    print(
        "Exactly one update per completed attempt: PASSED"
    )

    # ========================================================
    # SAVE ONLY THE VALIDATION REPORT.
    #
    # DO NOT save the learned bandit state:
    # its rewards were synthetic.
    # ========================================================

    payload = {
        "experiment":
            "si7d_recovered_selector_attempt_lints_integration",

        "validation_passed":
            True,

        "educational_claim_allowed":
            False,

        "selector_data_origin":
            "real_saved_validation_predictions",

        "bandit_feedback_origin":
            "synthetic_integration_test",

        "bandit_state_persisted":
            False,

        "architecture_under_test": {
            "selector":
                "recovered_context_routed_MD4_MD3_hybrid",

            "adaptive_algorithm":
                "existing_DisjointLinTS",

            "adaptive_level":
                "attempt",

            "arm_space":
                list(
                    ARMS
                ),

            "max_probability_gap":
                MAX_PROBABILITY_GAP,

            "reward":
                "evaluator_score_divided_by_3",

            "mastery_in_primary_reward":
                False,
        },

        "replay_qid":
            selected_qid,

        "replay_turns":
            len(
                selector_examples
            ),

        "attempts": [
            attempt_0,
            attempt_1,
        ],

        "total_updates":
            final_bandit.total_updates,
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
            payload,
            f,
            indent=2,
        )

    print()

    print(
        "Saved validation report:"
    )

    print(
        " ",
        RESULT_PATH,
    )

    print()

    print("=" * 78)
    print(
        "SI7-D ATTEMPT-LEVEL LinTS INTEGRATION VALIDATION PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()