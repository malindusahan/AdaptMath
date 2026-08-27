import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


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

OUTPUT_PATH = Path(
    "results/self_improvement/"
    "si0_mathdial_auxiliary_attribute_audit.json"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


MOVE_PATTERN = re.compile(
    r"^\s*Teacher\s*:\s*"
    r"\((generic|probing|focus|telling)\)",
    flags=re.IGNORECASE,
)


# ============================================================
# LOAD JSONL
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


raw_official_train = load_jsonl(
    RAW_TRAIN
)

raw_official_test = load_jsonl(
    RAW_TEST
)

processed_train = load_jsonl(
    PROCESSED_TRAIN
)


# ============================================================
# SAME RAW ORDER USED BY OUR PREPROCESSING
# ============================================================

combined_raw = (
    raw_official_train
    + raw_official_test
)


assert len(combined_raw) == 2861
assert len(processed_train) == 14905


# ============================================================
# IDENTIFY ONLY OUR FROZEN CUSTOM TRAIN DIALOGUES
#
# IMPORTANT:
# We deliberately do NOT inspect custom validation/test
# auxiliary outcomes during reward design.
# ============================================================

train_source_indices = sorted({

    int(
        example[
            "source_row_index"
        ]
    )

    for example
    in processed_train

})


train_raw_rows = [

    combined_raw[index]

    for index
    in train_source_indices

]


print("=" * 70)

print(
    "SI0 â€” MATHDIAL AUXILIARY ATTRIBUTE AUDIT"
)

print("=" * 70)


print(
    "\nCustom TRAIN dialogues:",
    len(train_raw_rows)
)

print(
    "Custom TRAIN supervised turns:",
    len(processed_train)
)


assert len(train_raw_rows) == 2295


# ============================================================
# HELPERS
# ============================================================

def is_missing(value):

    if value is None:
        return True

    if isinstance(
        value,
        str,
    ):

        return (
            len(
                value.strip()
            )
            == 0
        )

    return False


def word_count(value):

    if is_missing(value):
        return 0

    return len(
        str(value).split()
    )


def extract_moves(
    conversation
):

    moves = []


    for segment in conversation.split(
        "|EOM|"
    ):

        segment = segment.strip()

        match = MOVE_PATTERN.match(
            segment
        )


        if match:

            moves.append(
                match.group(1).lower()
            )


    return moves


def normalize_outcome(value):

    if is_missing(value):
        return "MISSING"

    value = (
        str(value)
        .strip()
    )


    lower = value.lower()


    if lower == "yes":

        return "Yes"


    if (
        "reveal"
        in lower
    ):

        return (
            "Yes, but I had "
            "to reveal the answer"
        )


    if lower == "no":

        return "No"


    return value


def normalize_likert(value):

    if value is None:
        return None

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# SCHEMA / MISSINGNESS
# ============================================================

fields = [

    "scenario",
    "question",
    "ground_truth",
    "student_incorrect_solution",
    "student_profile",
    "teacher_described_confusion",
    "self-correctness",
    "self-typical-confusion",
    "self-typical-interactions",
    "conversation",
]


print(
    "\n"
    + "=" * 70
)

print(
    "TRAIN RAW ATTRIBUTE MISSINGNESS"
)

print(
    "=" * 70
)


missingness = {}


for field in fields:

    missing = sum(

        is_missing(
            row.get(
                field
            )
        )

        for row
        in train_raw_rows
    )


    present = (
        len(train_raw_rows)
        - missing
    )


    missingness[
        field
    ] = {

        "present":
            present,

        "missing":
            missing,

        "missing_percentage":
            (
                missing
                / len(train_raw_rows)
                * 100
            ),
    }


    print(
        f"\n{field}"
    )

    print(
        f"  Present: "
        f"{present}"
    )

    print(
        f"  Missing: "
        f"{missing}"
    )

    print(
        f"  Missing %: "
        f"{(
            missing
            / len(train_raw_rows)
            * 100
        ):.2f}%"
    )


# ============================================================
# SELF-CORRECTNESS
# ============================================================

outcomes = [

    normalize_outcome(
        row.get(
            "self-correctness"
        )
    )

    for row
    in train_raw_rows

]


outcome_counts = Counter(
    outcomes
)


outcome_order = [

    "Yes",
    "Yes, but I had to reveal the answer",
    "No",
    "MISSING",
]


print(
    "\n"
    + "=" * 70
)

print(
    "SELF-CORRECTNESS"
)

print(
    "=" * 70
)


for outcome in outcome_order:

    count = outcome_counts[
        outcome
    ]


    print(
        f"\n{outcome}: "
        f"{count}"
    )


    if len(train_raw_rows):

        print(
            f"  Percentage: "
            f"{(
                count
                / len(train_raw_rows)
                * 100
            ):.2f}%"
        )


# Print unexpected values if any.

unexpected_outcomes = {

    key:
        value

    for key, value
    in outcome_counts.items()

    if key
    not in outcome_order
}


print(
    "\nUnexpected outcome values:",
    unexpected_outcomes
)


# ============================================================
# LIKERT QUALITY ANNOTATIONS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "LIKERT QUALITY ANNOTATIONS"
)

print(
    "=" * 70
)


likert_summary = {}


for field in [

    "self-typical-confusion",
    "self-typical-interactions",

]:

    values = [

        normalize_likert(
            row.get(
                field
            )
        )

        for row
        in train_raw_rows
    ]


    valid = [

        value

        for value in values

        if value is not None
    ]


    counts = Counter(
        valid
    )


    print(
        f"\n{field}"
    )

    print(
        "  Valid:",
        len(valid)
    )

    print(
        "  Missing:",
        len(values)
        - len(valid)
    )


    for score in [
        1,
        2,
        3,
        4,
        5,
    ]:

        print(
            f"  Score {score}: "
            f"{counts[float(score)]}"
        )


    if valid:

        print(
            f"  Mean: "
            f"{np.mean(valid):.3f}"
        )

        print(
            f"  Median: "
            f"{np.median(valid):.3f}"
        )


    likert_summary[
        field
    ] = {

        "valid":
            len(valid),

        "missing":
            (
                len(values)
                - len(valid)
            ),

        "distribution": {

            str(score):
                int(
                    counts[
                        float(score)
                    ]
                )

            for score in [
                1,
                2,
                3,
                4,
                5,
            ]
        },

        "mean":
            (
                float(
                    np.mean(valid)
                )

                if valid

                else None
            ),

        "median":
            (
                float(
                    np.median(valid)
                )

                if valid

                else None
            ),
    }


# ============================================================
# MOVE DISTRIBUTION CONDITIONED ON TERMINAL OUTCOME
#
# IMPORTANT:
# This is descriptive correlation only.
# It must NOT be interpreted as causal move effectiveness.
# ============================================================

outcome_move_counts = defaultdict(
    Counter
)

outcome_dialogue_counts = Counter()

outcome_total_moves = Counter()

outcome_dialogues_with_telling = Counter()


for row, outcome in zip(
    train_raw_rows,
    outcomes,
):

    moves = extract_moves(
        row["conversation"]
    )


    outcome_dialogue_counts[
        outcome
    ] += 1


    outcome_total_moves[
        outcome
    ] += len(
        moves
    )


    outcome_move_counts[
        outcome
    ].update(
        moves
    )


    if "telling" in moves:

        outcome_dialogues_with_telling[
            outcome
        ] += 1


print(
    "\n"
    + "=" * 70
)

print(
    "OUTCOME VS TEACHER MOVES"
)

print(
    "=" * 70
)


outcome_move_summary = {}


for outcome in outcome_order:

    dialogues = (
        outcome_dialogue_counts[
            outcome
        ]
    )


    if dialogues == 0:
        continue


    total_moves = (
        outcome_total_moves[
            outcome
        ]
    )


    print(
        f"\n{outcome}"
    )

    print(
        "  Dialogues:",
        dialogues
    )

    print(
        "  Teacher moves:",
        total_moves
    )


    print(
        f"  Mean moves/dialogue: "
        f"{(
            total_moves
            / dialogues
        ):.3f}"
    )


    telling_dialogues = (
        outcome_dialogues_with_telling[
            outcome
        ]
    )


    print(
        f"  Dialogues containing telling: "
        f"{telling_dialogues} "
        f"({(
            telling_dialogues
            / dialogues
            * 100
        ):.2f}%)"
    )


    move_distribution = {}


    for move in LABELS:

        count = (
            outcome_move_counts[
                outcome
            ][
                move
            ]
        )


        proportion = (

            count
            / total_moves

            if total_moves

            else 0.0
        )


        move_distribution[
            move
        ] = {

            "count":
                int(
                    count
                ),

            "proportion":
                float(
                    proportion
                ),
        }


        print(
            f"  {move:10s}: "
            f"{count:5d} "
            f"({proportion * 100:.2f}%)"
        )


    outcome_move_summary[
        outcome
    ] = {

        "dialogues":
            int(
                dialogues
            ),

        "teacher_moves":
            int(
                total_moves
            ),

        "mean_moves_per_dialogue":
            float(
                total_moves
                / dialogues
            ),

        "dialogues_with_telling":
            int(
                telling_dialogues
            ),

        "dialogues_with_telling_rate":
            float(
                telling_dialogues
                / dialogues
            ),

        "move_distribution":
            move_distribution,
    }


# ============================================================
# TYPICALITY BY TERMINAL OUTCOME
#
# If these differ strongly, typicality may be useful as a
# reliability/sample-weight variable rather than reward.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TYPICALITY BY TERMINAL OUTCOME"
)

print(
    "=" * 70
)


typicality_by_outcome = {}


for outcome in outcome_order:

    indices = [

        index

        for index, value
        in enumerate(
            outcomes
        )

        if value == outcome
    ]


    if not indices:
        continue


    outcome_result = {}


    print(
        f"\n{outcome}"
    )


    for field in [

        "self-typical-confusion",
        "self-typical-interactions",

    ]:

        values = []


        for index in indices:

            value = normalize_likert(
                train_raw_rows[
                    index
                ].get(
                    field
                )
            )


            if value is not None:

                values.append(
                    value
                )


        mean_value = (

            float(
                np.mean(values)
            )

            if values

            else None
        )


        median_value = (

            float(
                np.median(values)
            )

            if values

            else None
        )


        outcome_result[
            field
        ] = {

            "n":
                len(values),

            "mean":
                mean_value,

            "median":
                median_value,
        }


        print(
            f"  {field}:"
        )

        print(
            f"    n = "
            f"{len(values)}"
        )

        print(
            f"    mean = "
            f"{mean_value}"
        )

        print(
            f"    median = "
            f"{median_value}"
        )


    typicality_by_outcome[
        outcome
    ] = outcome_result


# ============================================================
# SCENARIO
#
# Scenario = position 1..5 in data collection session.
# We check whether outcome varies substantially by scenario.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SCENARIO VS OUTCOME"
)

print(
    "=" * 70
)


scenario_outcomes = defaultdict(
    Counter
)


for row, outcome in zip(
    train_raw_rows,
    outcomes,
):

    scenario = row.get(
        "scenario"
    )


    scenario_outcomes[
        str(scenario)
    ][
        outcome
    ] += 1


scenario_summary = {}


for scenario in sorted(
    scenario_outcomes,
    key=lambda x:
        (
            int(x)
            if x.isdigit()
            else 999
        ),
):

    counts = (
        scenario_outcomes[
            scenario
        ]
    )


    total = sum(
        counts.values()
    )


    print(
        f"\nScenario {scenario}"
    )

    print(
        "  Dialogues:",
        total
    )


    result = {}


    for outcome in outcome_order:

        count = counts[
            outcome
        ]


        rate = (

            count
            / total

            if total

            else 0.0
        )


        result[
            outcome
        ] = {

            "count":
                int(
                    count
                ),

            "rate":
                float(
                    rate
                ),
        }


        print(
            f"  {outcome}: "
            f"{count} "
            f"({rate * 100:.2f}%)"
        )


    scenario_summary[
        scenario
    ] = result


# ============================================================
# STUDENT PROFILE
#
# Treat this as auxiliary metadata / subgroup information,
# not as observable selector input.
# ============================================================

profiles = [

    (
        row.get(
            "student_profile"
        )
        if not is_missing(
            row.get(
                "student_profile"
            )
        )
        else "MISSING"
    )

    for row
    in train_raw_rows

]


profile_counts = Counter(
    profiles
)


print(
    "\n"
    + "=" * 70
)

print(
    "STUDENT PROFILE"
)

print(
    "=" * 70
)


print(
    "\nUnique profiles:",
    len(
        profile_counts
    )
)


print(
    "\nTop 15 profiles:"
)


for profile, count in (
    profile_counts
    .most_common(15)
):

    print(
        f"\n  Count: "
        f"{count}"
    )

    print(
        f"  Profile: "
        f"{profile}"
    )


# ============================================================
# AUXILIARY TEXT LENGTHS
#
# We only inspect structure/coverage now.
# We do NOT train on these fields yet.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TEXT AUXILIARY FIELDS"
)

print(
    "=" * 70
)


text_fields = [

    "ground_truth",
    "student_incorrect_solution",
    "teacher_described_confusion",
]


text_length_summary = {}


for field in text_fields:

    lengths = [

        word_count(
            row.get(
                field
            )
        )

        for row
        in train_raw_rows

        if not is_missing(
            row.get(
                field
            )
        )
    ]


    lengths_np = np.array(
        lengths,
        dtype=np.int64,
    )


    print(
        f"\n{field}"
    )

    print(
        "  Present:",
        len(lengths)
    )


    if len(lengths):

        summary = {

            "minimum_words":
                int(
                    lengths_np.min()
                ),

            "median_words":
                float(
                    np.median(
                        lengths_np
                    )
                ),

            "p90_words":
                float(
                    np.percentile(
                        lengths_np,
                        90
                    )
                ),

            "p95_words":
                float(
                    np.percentile(
                        lengths_np,
                        95
                    )
                ),

            "maximum_words":
                int(
                    lengths_np.max()
                ),
        }


        for key, value in (
            summary.items()
        ):

            print(
                f"  {key}: "
                f"{value}"
            )


        text_length_summary[
            field
        ] = summary


# ============================================================
# SUMMARY / SAVE
# ============================================================

result = {

    "scope":
        "custom_train_only",

    "train_dialogues":
        len(
            train_raw_rows
        ),

    "train_supervised_turns":
        len(
            processed_train
        ),

    "missingness":
        missingness,

    "self_correctness": {

        key:
            int(value)

        for key, value
        in outcome_counts.items()
    },

    "likert_summary":
        likert_summary,

    "outcome_move_summary":
        outcome_move_summary,

    "typicality_by_outcome":
        typicality_by_outcome,

    "scenario_vs_outcome":
        scenario_summary,

    "student_profile": {

        "unique_profiles":
            len(
                profile_counts
            ),

        "top_15": [

            {
                "profile":
                    str(profile),

                "count":
                    int(count),
            }

            for profile, count
            in profile_counts.most_common(
                15
            )
        ],
    },

    "text_length_summary":
        text_length_summary,
}


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    "\n"
    + "=" * 70
)

print(
    "SI0 AUXILIARY ATTRIBUTE AUDIT COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)
