import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_train.jsonl"
)

VAL_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_validation.jsonl"
)

RESULT_PATH = Path(
    "results/self_improvement/"
    "si3_contrastive_pair_audit.json"
)


MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


NEGATIVES_PER_POSITIVE = 3

LENGTH_RATIO_MIN = 0.70
LENGTH_RATIO_MAX = 1.30


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# LOAD
# ============================================================

train_rows = load_jsonl(
    TRAIN_PATH
)

val_rows = load_jsonl(
    VAL_PATH
)


assert len(train_rows) == 9989
assert len(val_rows) == 2509


print("=" * 70)

print(
    "SI3-A — SELF-SUPERVISED RESPONSE PAIR FEASIBILITY"
)

print("=" * 70)


print(
    "\nSI train transitions:",
    len(train_rows)
)

print(
    "SI validation transitions:",
    len(val_rows)
)


# ============================================================
# SOURCE SPLIT VALIDATION
# ============================================================

def get_set(
    rows,
    field,
):

    return {

        str(
            row[field]
        )

        for row in rows
    }


train_groups = get_set(
    train_rows,
    "group_id",
)

val_groups = get_set(
    val_rows,
    "group_id",
)


train_qids = get_set(
    train_rows,
    "qid",
)

val_qids = get_set(
    val_rows,
    "qid",
)


train_sources = {

    int(
        row[
            "source_row_index"
        ]
    )

    for row in train_rows
}


val_sources = {

    int(
        row[
            "source_row_index"
        ]
    )

    for row in val_rows
}


group_overlap = len(
    train_groups
    &
    val_groups
)

qid_overlap = len(
    train_qids
    &
    val_qids
)

source_overlap = len(
    train_sources
    &
    val_sources
)


print(
    "\n"
    + "=" * 70
)

print(
    "SOURCE SPLIT VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nGroup overlap:",
    group_overlap,
)

print(
    "QID overlap:",
    qid_overlap,
)

print(
    "Source overlap:",
    source_overlap,
)


assert group_overlap == 0
assert qid_overlap == 0
assert source_overlap == 0


# ============================================================
# RESPONSE DUPLICATION
# ============================================================

def duplication_stats(
    rows,
):

    texts = [

        normalize_text(
            row[
                "student_reply"
            ]
        )

        for row in rows
    ]


    counts = Counter(
        texts
    )


    unique = len(
        counts
    )


    repeated_examples = sum(

        count

        for count in counts.values()

        if count > 1
    )


    maximum_frequency = max(
        counts.values()
    )


    return {

        "responses":
            len(rows),

        "unique":
            unique,

        "unique_rate":
            unique
            / len(rows),

        "repeated_examples":
            repeated_examples,

        "maximum_frequency":
            maximum_frequency,
    }


train_dup = duplication_stats(
    train_rows
)

val_dup = duplication_stats(
    val_rows
)


print(
    "\n"
    + "=" * 70
)

print(
    "STUDENT RESPONSE DUPLICATION"
)

print(
    "=" * 70
)


for name, result in [

    (
        "SI TRAIN",
        train_dup,
    ),

    (
        "SI VALIDATION",
        val_dup,
    ),

]:

    print(
        f"\n{name}"
    )

    print(
        "  Responses:",
        result[
            "responses"
        ],
    )

    print(
        "  Unique normalized responses:",
        result[
            "unique"
        ],
    )

    print(
        "  Unique percentage:",
        f"{(
            result[
                'unique_rate'
            ]
            * 100
        ):.2f}%",
    )

    print(
        "  Examples belonging to repeated text:",
        result[
            "repeated_examples"
        ],
    )

    print(
        "  Maximum exact-response frequency:",
        result[
            "maximum_frequency"
        ],
    )


# ============================================================
# BUILD NEGATIVE POOLS
# ============================================================

def build_pools(
    rows,
):

    pools = defaultdict(
        list
    )


    for index, row in enumerate(
        rows
    ):

        move = row[
            "teacher_move"
        ]


        assert move in MOVES


        pools[
            move
        ].append({

            "index":
                index,

            "qid":
                str(
                    row[
                        "qid"
                    ]
                ),

            "source_row_index":
                int(
                    row[
                        "source_row_index"
                    ]
                ),

            "normalized_response":
                normalize_text(
                    row[
                        "student_reply"
                    ]
                ),

            "word_count":
                word_count(
                    row[
                        "student_reply"
                    ]
                ),
        })


    return pools


# ============================================================
# HARD-NEGATIVE CANDIDATES
#
# Candidate must:
#
# - use same teacher move
# - come from another qid
# - come from another source dialogue
# - have different exact response text
# - have similar response length
# ============================================================

def count_candidates(
    row,
    row_index,
    pools,
):

    positive_text = normalize_text(
        row[
            "student_reply"
        ]
    )


    positive_words = word_count(
        row[
            "student_reply"
        ]
    )


    minimum_words = max(

        1,

        int(
            np.floor(
                positive_words
                *
                LENGTH_RATIO_MIN
            )
        ),
    )


    maximum_words = max(

        minimum_words,

        int(
            np.ceil(
                positive_words
                *
                LENGTH_RATIO_MAX
            )
        ),
    )


    count = 0


    for candidate in pools[
        row[
            "teacher_move"
        ]
    ]:

        if (
            candidate[
                "index"
            ]
            == row_index
        ):

            continue


        if (
            candidate[
                "qid"
            ]
            == str(
                row[
                    "qid"
                ]
            )
        ):

            continue


        if (
            candidate[
                "source_row_index"
            ]
            == int(
                row[
                    "source_row_index"
                ]
            )
        ):

            continue


        if (
            candidate[
                "normalized_response"
            ]
            == positive_text
        ):

            continue


        if not (

            minimum_words

            <=

            candidate[
                "word_count"
            ]

            <=

            maximum_words
        ):

            continue


        count += 1


    return count


# ============================================================
# AVAILABILITY AUDIT
# ============================================================

def audit_availability(
    rows,
):

    pools = build_pools(
        rows
    )


    candidate_counts = []

    counts_by_move = defaultdict(
        list
    )


    for index, row in enumerate(
        rows
    ):

        count = count_candidates(
            row,
            index,
            pools,
        )


        candidate_counts.append(
            count
        )


        counts_by_move[
            row[
                "teacher_move"
            ]
        ].append(
            count
        )


    values = np.asarray(
        candidate_counts,
        dtype=np.int64,
    )


    insufficient = int(
        np.sum(
            values
            <
            NEGATIVES_PER_POSITIVE
        )
    )


    by_move = {}


    for move in MOVES:

        move_values = np.asarray(
            counts_by_move[
                move
            ],
            dtype=np.int64,
        )


        by_move[
            move
        ] = {

            "contexts":
                int(
                    len(
                        move_values
                    )
                ),

            "minimum":
                int(
                    move_values.min()
                ),

            "median":
                float(
                    np.median(
                        move_values
                    )
                ),

            "p10":
                float(
                    np.percentile(
                        move_values,
                        10,
                    )
                ),
        }


    return {

        "minimum":
            int(
                values.min()
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p10":
            float(
                np.percentile(
                    values,
                    10,
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90,
                )
            ),

        "insufficient":
            insufficient,

        "by_move":
            by_move,
    }


train_availability = (
    audit_availability(
        train_rows
    )
)

val_availability = (
    audit_availability(
        val_rows
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "HARD-NEGATIVE AVAILABILITY"
)

print(
    "=" * 70
)


for name, result in [

    (
        "SI TRAIN",
        train_availability,
    ),

    (
        "SI VALIDATION",
        val_availability,
    ),

]:

    print(
        f"\n{name}"
    )

    print(
        "  Minimum candidates/context:",
        result[
            "minimum"
        ],
    )

    print(
        "  Median candidates/context:",
        result[
            "median"
        ],
    )

    print(
        "  P10 candidates/context:",
        result[
            "p10"
        ],
    )

    print(
        "  P90 candidates/context:",
        result[
            "p90"
        ],
    )

    print(
        f"  Contexts with fewer than "
        f"{NEGATIVES_PER_POSITIVE} candidates:",
        result[
            "insufficient"
        ],
    )


    print(
        "\n  By move:"
    )


    for move in MOVES:

        move_result = (
            result[
                "by_move"
            ][
                move
            ]
        )


        print(
            f"\n    {move}"
        )

        print(
            "      contexts:",
            move_result[
                "contexts"
            ],
        )

        print(
            "      min candidates:",
            move_result[
                "minimum"
            ],
        )

        print(
            "      median candidates:",
            move_result[
                "median"
            ],
        )

        print(
            "      p10 candidates:",
            move_result[
                "p10"
            ],
        )


# ============================================================
# VALIDATION GATE
# ============================================================

validation_passed = (

    train_availability[
        "insufficient"
    ]
    == 0

    and

    val_availability[
        "insufficient"
    ]
    == 0
)


print(
    "\n"
    + "=" * 70
)

print(
    "PAIR-CONSTRUCTION VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nEvery SI train context has >= "
    f"{NEGATIVES_PER_POSITIVE} candidates:",
    (
        train_availability[
            "insufficient"
        ]
        == 0
    ),
)


print(
    "Every SI validation context has >= "
    f"{NEGATIVES_PER_POSITIVE} candidates:",
    (
        val_availability[
            "insufficient"
        ]
        == 0
    ),
)


if validation_passed:

    print(
        "\nVALIDATION: PASSED"
    )

    print(
        "Hard-negative construction is feasible."
    )

else:

    print(
        "\nVALIDATION: FAILED"
    )

    print(
        "Do not construct or train the "
        "contrastive dataset yet."
    )


# ============================================================
# SAVE AUDIT
# ============================================================

result = {

    "experiment":
        "SI3-A_contrastive_pair_feasibility",

    "negative_sampling": {

        "same_teacher_move":
            True,

        "different_qid":
            True,

        "different_source_dialogue":
            True,

        "exclude_exact_response":
            True,

        "length_ratio_min":
            LENGTH_RATIO_MIN,

        "length_ratio_max":
            LENGTH_RATIO_MAX,

        "negatives_per_positive":
            NEGATIVES_PER_POSITIVE,
    },

    "source_split_overlap": {

        "group":
            group_overlap,

        "qid":
            qid_overlap,

        "source":
            source_overlap,
    },

    "duplication": {

        "train":
            train_dup,

        "validation":
            val_dup,
    },

    "availability": {

        "train":
            train_availability,

        "validation":
            val_availability,
    },

    "validation_passed":
        bool(
            validation_passed
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
    "\nSaved:"
)

print(
    RESULT_PATH
)