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
        str(text)
        .lower()
        .split()
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


print("=" * 70)
print("SI3-C â€” SAME-PROBLEM HARD-NEGATIVE FEASIBILITY")
print("=" * 70)


# ============================================================
# BUILD POOLS
# ============================================================

def build_indices(rows):

    by_qid = defaultdict(list)
    by_group = defaultdict(list)

    for index, row in enumerate(rows):

        item = {
            "index":
                index,

            "qid":
                str(row["qid"]),

            "group_id":
                str(row["group_id"]),

            "source":
                int(
                    row["source_row_index"]
                ),

            "move":
                row["teacher_move"],

            "response":
                row["student_reply"],

            "normalized_response":
                normalize_text(
                    row["student_reply"]
                ),

            "words":
                word_count(
                    row["student_reply"]
                ),
        }

        by_qid[
            item["qid"]
        ].append(item)

        by_group[
            item["group_id"]
        ].append(item)

    return by_qid, by_group


# ============================================================
# CANDIDATE COUNT
# ============================================================

def get_candidates(
    row,
    row_index,
    pool,
    require_same_move,
    low_ratio,
    high_ratio,
):

    positive_text = normalize_text(
        row["student_reply"]
    )

    positive_words = word_count(
        row["student_reply"]
    )

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

    candidates = []

    for candidate in pool:

        # Not itself
        if candidate["index"] == row_index:
            continue

        # Must come from another dialogue
        if (
            candidate["source"]
            == int(
                row["source_row_index"]
            )
        ):
            continue

        # Optional same pedagogical move
        if (
            require_same_move
            and
            candidate["move"]
            != row["teacher_move"]
        ):
            continue

        # Do not use identical response text
        if (
            candidate["normalized_response"]
            == positive_text
        ):
            continue

        # Similar response length
        if not (
            minimum
            <= candidate["words"]
            <= maximum
        ):
            continue

        candidates.append(
            candidate
        )

    return candidates


# ============================================================
# AUDIT ONE SAMPLING SCHEME
# ============================================================

def audit_scheme(
    rows,
    pool_type,
    require_same_move,
    low_ratio,
    high_ratio,
):

    by_qid, by_group = build_indices(
        rows
    )

    counts = []

    examples = []

    for index, row in enumerate(rows):

        if pool_type == "qid":

            pool = by_qid[
                str(row["qid"])
            ]

        elif pool_type == "group":

            pool = by_group[
                str(row["group_id"])
            ]

        else:

            raise ValueError(
                pool_type
            )

        candidates = get_candidates(
            row,
            index,
            pool,
            require_same_move,
            low_ratio,
            high_ratio,
        )

        count = len(
            candidates
        )

        counts.append(
            count
        )

        if (
            count >= 3
            and
            len(examples) < 3
        ):

            examples.append({
                "row":
                    row,

                "candidates":
                    candidates[:3],
            })

    values = np.array(
        counts,
        dtype=np.int64,
    )

    return {
        "minimum":
            int(values.min()),

        "median":
            float(
                np.median(values)
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90,
                )
            ),

        "at_least_1":
            int(
                np.sum(values >= 1)
            ),

        "at_least_3":
            int(
                np.sum(values >= 3)
            ),

        "total":
            len(rows),

        "examples":
            examples,
    }


# ============================================================
# SCHEMES
# ============================================================

SCHEMES = [
    {
        "name":
            "A_same_qid_same_move_70_130",

        "pool_type":
            "qid",

        "same_move":
            True,

        "low":
            0.70,

        "high":
            1.30,
    },
    {
        "name":
            "B_same_group_same_move_70_130",

        "pool_type":
            "group",

        "same_move":
            True,

        "low":
            0.70,

        "high":
            1.30,
    },
    {
        "name":
            "C_same_group_same_move_50_150",

        "pool_type":
            "group",

        "same_move":
            True,

        "low":
            0.50,

        "high":
            1.50,
    },
    {
        "name":
            "D_same_group_any_move_70_130",

        "pool_type":
            "group",

        "same_move":
            False,

        "low":
            0.70,

        "high":
            1.30,
    },
]


# ============================================================
# RUN
# ============================================================

all_results = {}


for split_name, rows in [
    ("SI TRAIN", train_rows),
    ("SI VALIDATION", val_rows),
]:

    print(
        "\n"
        + "=" * 70
    )

    print(
        split_name
    )

    print(
        "=" * 70
    )

    split_results = {}

    for scheme in SCHEMES:

        result = audit_scheme(
            rows,
            scheme["pool_type"],
            scheme["same_move"],
            scheme["low"],
            scheme["high"],
        )

        split_results[
            scheme["name"]
        ] = result

        total = result[
            "total"
        ]

        n1 = result[
            "at_least_1"
        ]

        n3 = result[
            "at_least_3"
        ]

        print(
            f"\n{scheme['name']}"
        )

        print(
            "  Minimum candidates:",
            result["minimum"],
        )

        print(
            "  Median candidates:",
            result["median"],
        )

        print(
            "  P90 candidates:",
            result["p90"],
        )

        print(
            "  Contexts with >=1 candidate:",
            f"{n1}/{total} "
            f"({n1 / total * 100:.2f}%)",
        )

        print(
            "  Contexts with >=3 candidates:",
            f"{n3}/{total} "
            f"({n3 / total * 100:.2f}%)",
        )

    all_results[
        split_name
    ] = split_results


# ============================================================
# MANUAL EXAMPLES
#
# Show examples from the strongest strict scheme first.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SAME-PROBLEM HARD-NEGATIVE EXAMPLES"
)

print(
    "=" * 70
)


for split_name in [
    "SI TRAIN",
    "SI VALIDATION",
]:

    result = all_results[
        split_name
    ][
        "B_same_group_same_move_70_130"
    ]

    print(
        f"\n### {split_name}"
    )

    if not result[
        "examples"
    ]:

        print(
            "No contexts with >=3 candidates."
        )

        continue

    for example in result[
        "examples"
    ]:

        row = example[
            "row"
        ]

        print(
            "\n"
            + "-" * 70
        )

        print(
            "QID:",
            row["qid"],
        )

        print(
            "Group:",
            row["group_id"],
        )

        print(
            "Move:",
            row["teacher_move"],
        )

        print(
            "\nTeacher:"
        )

        print(
            row["teacher_text"]
        )

        print(
            "\nACTUAL NEXT STUDENT RESPONSE:"
        )

        print(
            row["student_reply"]
        )

        for i, candidate in enumerate(
            example["candidates"],
            start=1,
        ):

            print(
                f"\nHARD NEGATIVE {i}:"
            )

            print(
                candidate[
                    "response"
                ]
            )


# ============================================================
# STOP
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SI3-C FEASIBILITY AUDIT COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nDo not construct the replacement dataset yet."
)
