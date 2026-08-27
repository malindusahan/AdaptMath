import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.self_improvement.frozen_selector_replay import (
    FrozenSelectorReplay,
    LABELS,
)


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


ARM_TO_MOVE = {
    "baseline": None,
    "probing_bias": "probing",
    "focus_bias": "focus",
    "telling_bias": "telling",
    "generic_bias": "generic",
}


# Exact default from the reusable SI5 AttemptLevelBanditPolicy.
MAX_PROBABILITY_GAP = 0.15


RESULT_PATH = Path(
    "results/self_improvement/"
    "si7c_real_frozen_overlay_validation.json"
)


# ============================================================
# CONSERVATIVE OVERLAY
# ============================================================

def apply_conservative_overlay(
    base_move_probs,
    selected_arm,
    max_probability_gap=MAX_PROBABILITY_GAP,
):
    """
    Reproduce the already-implemented SI5 overlay semantics.

    IMPORTANT:
    This function does NOT learn and does NOT update LinTS.

    It only answers:

        Given a frozen selector probability distribution
        and an already-selected residual arm,
        what final pedagogical move would be executed?
    """

    if selected_arm not in ARM_TO_MOVE:
        raise ValueError(
            f"Unknown arm: {selected_arm}"
        )

    missing_moves = [
        move
        for move in LABELS
        if move not in base_move_probs
    ]

    if missing_moves:
        raise ValueError(
            "Missing move probabilities: "
            + ", ".join(missing_moves)
        )

    values = {
        move: float(
            base_move_probs[move]
        )
        for move in LABELS
    }

    if any(
        not np.isfinite(value)
        for value in values.values()
    ):
        raise ValueError(
            "Move probabilities must be finite."
        )

    if any(
        value < 0.0
        for value in values.values()
    ):
        raise ValueError(
            "Move probabilities cannot be negative."
        )

    total_probability = sum(
        values.values()
    )

    if total_probability <= 0.0:
        raise ValueError(
            "Move probabilities must have positive total mass."
        )

    # Same defensive normalization as SI5.
    normalized = {
        move:
            value / total_probability
        for move, value
        in values.items()
    }

    base_move = max(
        LABELS,
        key=lambda move:
            normalized[move],
    )

    preferred_move = (
        ARM_TO_MOVE[
            selected_arm
        ]
    )

    overridden = False

    probability_gap = None

    final_move = base_move

    # baseline maps to None, so it never changes the base move.
    if preferred_move is not None:

        # If the frozen selector already picked the preferred move,
        # there is no actual override.
        if preferred_move != base_move:

            probability_gap = (
                normalized[
                    base_move
                ]
                -
                normalized[
                    preferred_move
                ]
            )

            # Exact conservative SI5 rule.
            if (
                probability_gap
                <=
                max_probability_gap
            ):
                final_move = (
                    preferred_move
                )

                overridden = True

    return {
        "selected_arm":
            selected_arm,

        "preferred_move":
            preferred_move,

        "base_move":
            base_move,

        "final_move":
            final_move,

        "overridden":
            overridden,

        "probability_gap":
            probability_gap,

        "base_move_probs":
            normalized,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "SI7-C REAL FROZEN SELECTOR + CONSERVATIVE OVERLAY"
    )
    print("=" * 76)

    print()

    print(
        "Offline validation replay only."
    )

    print(
        "No LinTS updates are performed."
    )

    print(
        "No educational-effectiveness claim is made."
    )

    print()

    print(
        "Maximum probability gap:",
        MAX_PROBABILITY_GAP,
    )

    print()

    selector = FrozenSelectorReplay()

    example_ids = (
        selector.example_ids()
    )

    assert len(example_ids) == 1850

    print(
        "Recovered frozen examples:",
        len(example_ids),
    )

    print()

    results_by_arm = {}

    # ========================================================
    # Evaluate every possible residual arm against the exact
    # recovered frozen-selector probabilities.
    # ========================================================

    for selected_arm in ARMS:

        override_count = 0

        unchanged_count = 0

        already_preferred_count = 0

        blocked_by_gap_count = 0

        final_move_counts = Counter()

        route_override_counts = Counter()

        gaps_when_overridden = []

        gaps_when_blocked = []

        examples_with_override = []

        for example_id in example_ids:

            prediction = (
                selector.predict(
                    example_id
                )
            )

            overlay = (
                apply_conservative_overlay(
                    base_move_probs=
                        prediction[
                            "probabilities"
                        ],

                    selected_arm=
                        selected_arm,

                    max_probability_gap=
                        MAX_PROBABILITY_GAP,
                )
            )

            base_move = (
                overlay[
                    "base_move"
                ]
            )

            final_move = (
                overlay[
                    "final_move"
                ]
            )

            preferred_move = (
                overlay[
                    "preferred_move"
                ]
            )

            final_move_counts[
                final_move
            ] += 1

            if (
                overlay[
                    "overridden"
                ]
            ):

                override_count += 1

                route_override_counts[
                    prediction[
                        "route"
                    ]
                ] += 1

                gaps_when_overridden.append(
                    float(
                        overlay[
                            "probability_gap"
                        ]
                    )
                )

                # Keep only a few examples for diagnostics.
                if (
                    len(
                        examples_with_override
                    )
                    <
                    5
                ):

                    examples_with_override.append(
                        {
                            "example_id":
                                example_id,

                            "route":
                                prediction[
                                    "route"
                                ],

                            "base_move":
                                base_move,

                            "preferred_move":
                                preferred_move,

                            "final_move":
                                final_move,

                            "probability_gap":
                                float(
                                    overlay[
                                        "probability_gap"
                                    ]
                                ),

                            "probabilities":
                                overlay[
                                    "base_move_probs"
                                ],
                        }
                    )

            else:

                unchanged_count += 1

                if (
                    preferred_move
                    is not None
                    and
                    preferred_move
                    ==
                    base_move
                ):

                    already_preferred_count += 1

                elif (
                    preferred_move
                    is not None
                    and
                    preferred_move
                    !=
                    base_move
                    and
                    overlay[
                        "probability_gap"
                    ]
                    is not None
                ):

                    if (
                        overlay[
                            "probability_gap"
                        ]
                        >
                        MAX_PROBABILITY_GAP
                    ):

                        blocked_by_gap_count += 1

                        gaps_when_blocked.append(
                            float(
                                overlay[
                                    "probability_gap"
                                ]
                            )
                        )

        override_rate = (
            override_count
            /
            len(example_ids)
        )

        results_by_arm[
            selected_arm
        ] = {
            "examples":
                len(example_ids),

            "override_count":
                override_count,

            "override_rate":
                override_rate,

            "unchanged_count":
                unchanged_count,

            "already_preferred_count":
                already_preferred_count,

            "blocked_by_gap_count":
                blocked_by_gap_count,

            "final_move_counts":
                dict(
                    final_move_counts
                ),

            "route_override_counts":
                dict(
                    route_override_counts
                ),

            "mean_gap_when_overridden":
                (
                    float(
                        np.mean(
                            gaps_when_overridden
                        )
                    )
                    if gaps_when_overridden
                    else None
                ),

            "max_gap_when_overridden":
                (
                    float(
                        np.max(
                            gaps_when_overridden
                        )
                    )
                    if gaps_when_overridden
                    else None
                ),

            "mean_gap_when_blocked":
                (
                    float(
                        np.mean(
                            gaps_when_blocked
                        )
                    )
                    if gaps_when_blocked
                    else None
                ),

            "sample_overrides":
                examples_with_override,
        }

    # ========================================================
    # VALIDATION CONDITIONS
    # ========================================================

    print("-" * 76)
    print(
        "OVERLAY EFFECT BY FORCED ARM"
    )
    print("-" * 76)

    print()

    for arm in ARMS:

        result = (
            results_by_arm[
                arm
            ]
        )

        print(
            f"{arm:<15}"
            f" overrides="
            f"{result['override_count']:>4}"
            f"/"
            f"{result['examples']}"
            f"  "
            f"rate="
            f"{result['override_rate']:.4f}"
            f"  "
            f"already_preferred="
            f"{result['already_preferred_count']:>4}"
            f"  "
            f"blocked="
            f"{result['blocked_by_gap_count']:>4}"
        )

    print()

    # --------------------------------------------------------
    # Baseline must NEVER alter the frozen selector.
    # --------------------------------------------------------

    baseline_result = (
        results_by_arm[
            "baseline"
        ]
    )

    assert (
        baseline_result[
            "override_count"
        ]
        ==
        0
    )

    assert (
        baseline_result[
            "unchanged_count"
        ]
        ==
        1850
    )

    print(
        "Baseline preservation: PASSED"
    )

    # --------------------------------------------------------
    # Every non-baseline arm should be capable of affecting
    # at least some real frozen-selector examples.
    # --------------------------------------------------------

    for arm in ARMS:

        if arm == "baseline":
            continue

        assert (
            results_by_arm[
                arm
            ][
                "override_count"
            ]
            >
            0
        )

    print(
        "All bias arms have realizable effects: PASSED"
    )

    # --------------------------------------------------------
    # Every override must satisfy the frozen threshold.
    # --------------------------------------------------------

    for arm in ARMS:

        max_gap = (
            results_by_arm[
                arm
            ][
                "max_gap_when_overridden"
            ]
        )

        if max_gap is None:
            continue

        assert (
            max_gap
            <=
            MAX_PROBABILITY_GAP
            +
            1e-12
        )

    print(
        "Probability-gap safety rule: PASSED"
    )

    # --------------------------------------------------------
    # The conservative policy should block at least some
    # proposed changes on these real frozen probabilities.
    #
    # Otherwise the threshold would effectively do nothing.
    # --------------------------------------------------------

    total_blocked = sum(
        results_by_arm[
            arm
        ][
            "blocked_by_gap_count"
        ]

        for arm in ARMS
        if arm != "baseline"
    )

    assert (
        total_blocked
        >
        0
    )

    print(
        "Conservative blocking is active: PASSED"
    )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    payload = {
        "experiment":
            "si7c_real_frozen_overlay_validation",

        "data_origin":
            "real_saved_validation_predictions",

        "learning_performed":
            False,

        "educational_claim_allowed":
            False,

        "frozen_selector":
            "context_routed_MD4_MD3_hybrid",

        "examples":
            len(
                example_ids
            ),

        "max_probability_gap":
            MAX_PROBABILITY_GAP,

        "arm_to_move":
            ARM_TO_MOVE,

        "results_by_arm":
            results_by_arm,

        "total_blocked_proposals":
            total_blocked,

        "validation_passed":
            True,
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
        "Saved:",
        RESULT_PATH,
    )

    print()

    print("=" * 76)
    print(
        "SI7-C REAL FROZEN OVERLAY VALIDATION PASSED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()