from collections import Counter

from src.self_improvement.frozen_selector_replay import (
    FrozenSelectorReplay,
    LABELS,
)


EXPECTED_EXAMPLES = 1850

EXPECTED_MD4 = 1507

EXPECTED_MD3 = 343


def main():

    print("=" * 76)
    print(
        "SI7-B FROZEN SELECTOR REPLAY VALIDATION"
    )
    print("=" * 76)
    print()

    selector = FrozenSelectorReplay()

    # --------------------------------------------------
    # TEST 1
    # Correct number of recovered examples
    # --------------------------------------------------

    print(
        "Loaded examples:",
        len(selector),
    )

    assert (
        len(selector)
        ==
        EXPECTED_EXAMPLES
    )

    print(
        "Example count: PASSED"
    )

    # --------------------------------------------------
    # TEST 2
    # Inspect all records through the public interface
    # --------------------------------------------------

    route_counts = Counter()

    source_counts = Counter()

    prediction_counts = Counter()

    for example_id in selector.example_ids():

        prediction = selector.predict(
            example_id
        )

        probabilities = prediction[
            "probabilities"
        ]

        assert (
            set(
                probabilities.keys()
            )
            ==
            set(
                LABELS
            )
        )

        probability_sum = sum(
            probabilities.values()
        )

        assert abs(
            probability_sum
            -
            1.0
        ) < 1e-5

        argmax_move = max(
            LABELS,
            key=lambda label:
                probabilities[label],
        )

        assert (
            argmax_move
            ==
            prediction[
                "predicted_move"
            ]
        )

        route_counts[
            prediction[
                "route"
            ]
        ] += 1

        source_counts[
            prediction[
                "source_model"
            ]
        ] += 1

        prediction_counts[
            prediction[
                "predicted_move"
            ]
        ] += 1

    print(
        "Public prediction interface: PASSED"
    )

    # --------------------------------------------------
    # TEST 3
    # Exact routing counts
    # --------------------------------------------------

    print()
    print(
        "Route counts:"
    )

    for route, count in sorted(
        route_counts.items()
    ):
        print(
            f"  {route:<15}: {count}"
        )

    print()
    print(
        "Source model counts:"
    )

    for source, count in sorted(
        source_counts.items()
    ):
        print(
            f"  {source:<5}: {count}"
        )

    assert (
        source_counts[
            "MD4"
        ]
        ==
        EXPECTED_MD4
    )

    assert (
        source_counts[
            "MD3"
        ]
        ==
        EXPECTED_MD3
    )

    print(
        "Routing counts: PASSED"
    )

    # --------------------------------------------------
    # TEST 4
    # Confirm both routes are accessible
    # --------------------------------------------------

    md4_example = None

    md3_example = None

    for example_id in selector.example_ids():

        prediction = selector.predict(
            example_id
        )

        if (
            prediction[
                "source_model"
            ]
            ==
            "MD4"
            and
            md4_example is None
        ):
            md4_example = prediction

        if (
            prediction[
                "source_model"
            ]
            ==
            "MD3"
            and
            md3_example is None
        ):
            md3_example = prediction

        if (
            md4_example is not None
            and
            md3_example is not None
        ):
            break

    assert md4_example is not None

    assert md3_example is not None

    print()
    print(
        "Example MD4 route:"
    )

    print(
        "  example_id:",
        md4_example[
            "example_id"
        ],
    )

    print(
        "  predicted_move:",
        md4_example[
            "predicted_move"
        ],
    )

    print(
        "  confidence:",
        round(
            md4_example[
                "confidence"
            ],
            6,
        ),
    )

    print()

    print(
        "Example MD3 route:"
    )

    print(
        "  example_id:",
        md3_example[
            "example_id"
        ],
    )

    print(
        "  predicted_move:",
        md3_example[
            "predicted_move"
        ],
    )

    print(
        "  confidence:",
        round(
            md3_example[
                "confidence"
            ],
            6,
        ),
    )

    print()

    print(
        "Both routing branches accessible: PASSED"
    )

    # --------------------------------------------------
    # TEST 5
    # Unknown examples must fail clearly
    # --------------------------------------------------

    try:

        selector.predict(
            "not_a_real_example"
        )

    except KeyError:

        print(
            "Unknown example protection: PASSED"
        )

    else:

        raise AssertionError(
            "Unknown example did not raise KeyError."
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        "Predicted move distribution:"
    )

    for label in LABELS:

        print(
            f"  {label:<8}: "
            f"{prediction_counts[label]}"
        )

    print()
    print("=" * 76)
    print(
        "SI7-B FROZEN SELECTOR REPLAY VALIDATION PASSED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()