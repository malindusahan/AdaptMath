import json
from pathlib import Path
from collections import defaultdict

import numpy as np


TRAIN_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_train.jsonl"
)

VAL_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_validation.jsonl"
)


MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def normalize_text(text):
    return " ".join(
        str(text).lower().split()
    )


def word_count(text):
    return len(
        str(text).split()
    )


train_rows = load_jsonl(
    TRAIN_PATH
)

val_rows = load_jsonl(
    VAL_PATH
)


assert len(train_rows) == 9989
assert len(val_rows) == 2509


def build_pools(rows):
    pools = defaultdict(list)

    for index, row in enumerate(rows):

        pools[
            row["teacher_move"]
        ].append({
            "index":
                index,

            "qid":
                str(
                    row["qid"]
                ),

            "source":
                int(
                    row["source_row_index"]
                ),

            "response":
                row["student_reply"],

            "normalized":
                normalize_text(
                    row["student_reply"]
                ),

            "words":
                word_count(
                    row["student_reply"]
                ),
        })

    return pools


def eligible_without_length(
    row,
    row_index,
    pools,
):

    positive_text = normalize_text(
        row["student_reply"]
    )

    result = []

    for candidate in pools[
        row["teacher_move"]
    ]:

        if candidate["index"] == row_index:
            continue

        if (
            candidate["qid"]
            == str(row["qid"])
        ):
            continue

        if (
            candidate["source"]
            == int(
                row["source_row_index"]
            )
        ):
            continue

        if (
            candidate["normalized"]
            == positive_text
        ):
            continue

        result.append(
            candidate
        )

    return result


def count_in_window(
    candidates,
    positive_words,
    low_ratio,
    high_ratio,
):

    minimum = max(
        1,
        int(
            np.floor(
                positive_words
                * low_ratio
            )
        ),
    )

    maximum = max(
        minimum,
        int(
            np.ceil(
                positive_words
                * high_ratio
            )
        ),
    )

    matches = [
        candidate
        for candidate in candidates
        if (
            minimum
            <= candidate["words"]
            <= maximum
        )
    ]

    return (
        len(matches),
        minimum,
        maximum,
        matches,
    )


WINDOWS = [
    (
        "strict_70_130",
        0.70,
        1.30,
    ),
    (
        "moderate_50_150",
        0.50,
        1.50,
    ),
    (
        "broad_33_200",
        0.33,
        2.00,
    ),
]


def inspect_split(
    name,
    rows,
):

    pools = build_pools(
        rows
    )

    problematic = []

    for index, row in enumerate(rows):

        candidates = (
            eligible_without_length(
                row,
                index,
                pools,
            )
        )

        positive_words = word_count(
            row["student_reply"]
        )

        strict_count, _, _, _ = (
            count_in_window(
                candidates,
                positive_words,
                0.70,
                1.30,
            )
        )

        if strict_count < 3:

            problematic.append(
                (
                    index,
                    row,
                    candidates,
                    positive_words,
                )
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{name} EDGE CASES"
    )

    print(
        "=" * 70
    )

    print(
        "\nProblematic contexts:",
        len(problematic),
    )

    for (
        index,
        row,
        candidates,
        positive_words,
    ) in problematic:

        print(
            "\n"
            + "-" * 70
        )

        print(
            "Row index:",
            index,
        )

        print(
            "Move:",
            row["teacher_move"],
        )

        print(
            "QID:",
            row["qid"],
        )

        print(
            "Source:",
            row[
                "source_row_index"
            ],
        )

        print(
            "Positive response words:",
            positive_words,
        )

        print(
            "\nStudent response:"
        )

        print(
            row["student_reply"]
        )

        print(
            "\nCandidate availability:"
        )

        for (
            window_name,
            low_ratio,
            high_ratio,
        ) in WINDOWS:

            (
                count,
                minimum,
                maximum,
                matches,
            ) = count_in_window(
                candidates,
                positive_words,
                low_ratio,
                high_ratio,
            )

            print(
                f"  {window_name}: "
                f"{count} candidates "
                f"(word range "
                f"{minimum}-{maximum})"
            )

        print(
            "  no_length_constraint:",
            len(candidates),
        )

        # --------------------------------------------
        # Show nearest candidates by word-count
        # distance.
        # --------------------------------------------

        nearest = sorted(
            candidates,
            key=lambda c: abs(
                c["words"]
                - positive_words
            ),
        )[:5]

        print(
            "\nNearest candidate lengths:"
        )

        for candidate in nearest:

            print(
                f"  words={candidate['words']} "
                f"qid={candidate['qid']} "
                f"source={candidate['source']}"
            )

            preview = (
                candidate["response"]
                .replace("\n", " ")
            )

            print(
                "   ",
                preview[:220],
            )


print("=" * 70)

print(
    "SI3-A1 â€” CONTRASTIVE EDGE-CASE AUDIT"
)

print("=" * 70)


inspect_split(
    "SI TRAIN",
    train_rows,
)

inspect_split(
    "SI VALIDATION",
    val_rows,
)


print(
    "\n"
    + "=" * 70
)

print(
    "EDGE-CASE AUDIT COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nDo not change the sampling rule yet."
)
