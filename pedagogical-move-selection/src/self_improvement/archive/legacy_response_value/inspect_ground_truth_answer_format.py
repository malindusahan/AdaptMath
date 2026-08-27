import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

RAW_TRAIN = Path(
    "data/raw/mathdial/train.jsonl"
)

RAW_TEST = Path(
    "data/raw/mathdial/test.jsonl"
)

PROCESSED_TRAIN = Path(
    "data/processed/mathdial/train.jsonl"
)


# ============================================================
# LOAD
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


official_train = load_jsonl(
    RAW_TRAIN
)

official_test = load_jsonl(
    RAW_TEST
)

processed_train = load_jsonl(
    PROCESSED_TRAIN
)


combined_raw = (
    official_train
    + official_test
)


train_source_indices = sorted({

    int(
        row["source_row_index"]
    )

    for row in processed_train
})


assert len(
    train_source_indices
) == 2295


# ============================================================
# PATTERNS
# ============================================================

PURE_NUMBER = re.compile(

    r"""
    ^\s*
    \$?
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    \s*$
    """,

    flags=re.VERBOSE,
)


LITERAL_HASH = re.compile(

    r"""
    \#\#\#\#
    \s*
    \$?
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    """,

    flags=re.VERBOSE,
)


# ============================================================
# AUDIT STRUCTURE
# ============================================================

stats = Counter()

non_numeric_final_lines = []

empty_ground_truth = []


for source_index in train_source_indices:

    ground_truth = str(
        combined_raw[
            source_index
        ][
            "ground_truth"
        ]
    )


    if not ground_truth.strip():

        empty_ground_truth.append(
            source_index
        )

        continue


    if LITERAL_HASH.search(
        ground_truth
    ):

        stats[
            "contains_literal_hash"
        ] += 1


    # --------------------------------------------
    # Non-empty lines only
    # --------------------------------------------

    lines = [

        line.strip()

        for line in ground_truth.splitlines()

        if line.strip()
    ]


    if not lines:

        empty_ground_truth.append(
            source_index
        )

        continue


    final_line = lines[-1]


    if PURE_NUMBER.fullmatch(
        final_line
    ):

        stats[
            "final_nonempty_line_is_number"
        ] += 1

    else:

        stats[
            "final_nonempty_line_not_number"
        ] += 1


        non_numeric_final_lines.append({

            "source_row_index":
                source_index,

            "final_line":
                final_line,

            "tail":
                ground_truth[-300:],
        })


# ============================================================
# REPORT
# ============================================================

print("=" * 70)

print(
    "MATHDIAL GROUND-TRUTH ANSWER FORMAT AUDIT"
)

print("=" * 70)


print(
    "\nCustom TRAIN dialogues:",
    len(train_source_indices),
)


print(
    "\nContains literal #### answer:",
    stats[
        "contains_literal_hash"
    ],
)


print(
    "Final non-empty line is pure number:",
    stats[
        "final_nonempty_line_is_number"
    ],
)


print(
    "Final non-empty line is NOT pure number:",
    stats[
        "final_nonempty_line_not_number"
    ],
)


print(
    "Empty ground truth:",
    len(
        empty_ground_truth
    ),
)


coverage = (

    stats[
        "final_nonempty_line_is_number"
    ]

    /

    len(
        train_source_indices
    )
)


print(
    f"\nPure-number final-line coverage: "
    f"{coverage * 100:.2f}%"
)


# ============================================================
# SHOW RAW repr() EXAMPLES
#
# repr is important because it exposes hidden newlines,
# spaces, hashes, etc.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RAW ENDING EXAMPLES"
)

print(
    "=" * 70
)


for source_index in (
    train_source_indices[:5]
):

    ground_truth = str(
        combined_raw[
            source_index
        ][
            "ground_truth"
        ]
    )


    lines = [

        line

        for line in ground_truth.splitlines()

        if line.strip()
    ]


    print(
        f"\nSource row: "
        f"{source_index}"
    )

    print(
        "Last 3 non-empty lines:"
    )


    for line in lines[-3:]:

        print(
            repr(line)
        )


# ============================================================
# NON-NUMERIC EXAMPLES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "NON-NUMERIC FINAL-LINE EXAMPLES"
)

print(
    "=" * 70
)


if not non_numeric_final_lines:

    print(
        "\nNone."
    )


else:

    for record in (
        non_numeric_final_lines[:10]
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            "Source:",
            record[
                "source_row_index"
            ],
        )

        print(
            "Final line:",
            repr(
                record[
                    "final_line"
                ]
            ),
        )

        print(
            "Tail:"
        )

        print(
            repr(
                record[
                    "tail"
                ]
            )
        )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FORMAT VALIDATION"
)

print(
    "=" * 70
)


if (
    stats[
        "final_nonempty_line_is_number"
    ]
    ==
    len(
        train_source_indices
    )
):

    print(
        "\nVALIDATION: PASSED"
    )

    print(
        "All 2295 custom-train ground truths "
        "end with a standalone numeric answer."
    )


else:

    print(
        "\nVALIDATION: NEEDS REVIEW"
    )

    print(
        "Do not choose an answer extractor yet."
    )
