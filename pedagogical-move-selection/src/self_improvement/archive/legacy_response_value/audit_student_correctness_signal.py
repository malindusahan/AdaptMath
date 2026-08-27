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

TRANSITIONS_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_all.jsonl"
)

OUTPUT_PATH = Path(
    "results/self_improvement/"
    "si2_student_correctness_signal_audit.json"
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

transitions = load_jsonl(
    TRANSITIONS_PATH
)


combined_raw = (
    official_train
    + official_test
)


assert len(combined_raw) == 2861
assert len(processed_train) == 14905
assert len(transitions) == 12498


print("=" * 70)

print(
    "SI2-A â€” STUDENT CORRECTNESS SIGNAL AUDIT"
)

print("=" * 70)


# ============================================================
# CUSTOM TRAIN SOURCE DIALOGUES ONLY
# ============================================================

train_source_indices = sorted({

    int(
        row[
            "source_row_index"
        ]
    )

    for row in processed_train

})


assert len(
    train_source_indices
) == 2295


train_raw = {

    source_index:
        combined_raw[
            source_index
        ]

    for source_index
    in train_source_indices
}


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

NUMBER_PATTERN = re.compile(

    r"""
    (?<![\w.])
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    (?![\w.])
    """,

    flags=re.VERBOSE,
)


def normalize_number(
    text
):

    text = (
        str(text)
        .strip()
        .replace(
            ",",
            "",
        )
    )


    try:

        value = float(
            text
        )

    except ValueError:

        return None


    if abs(
        value
        - round(value)
    ) < 1e-10:

        return str(
            int(
                round(value)
            )
        )


    return (
        f"{value:.12g}"
    )


def extract_numbers(
    text
):

    values = []


    for match in NUMBER_PATTERN.finditer(
        str(text)
    ):

        normalized = (
            normalize_number(
                match.group(0)
            )
        )


        if normalized is not None:

            values.append(
                normalized
            )


    return values


# ============================================================
# FINAL-ANSWER EXTRACTION
#
# We deliberately test several common GSM8K / MathDial
# conventions and record which one succeeds.
# ============================================================

ANSWER_IS_PATTERN = re.compile(

    r"""
    (?:
        answer
        |
        result
        |
        total
    )
    \s*
    (?:is|=|:)
    \s*
    \$?
    ([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    """,

    flags=(
        re.IGNORECASE
        |
        re.VERBOSE
    ),
)


FINAL_LINE_PATTERN = re.compile(

    r"""
    ^\s*
    \$?
    (
        [-+]?
        (?:
            \d{1,3}(?:,\d{3})+
            |
            \d+
        )
        (?:\.\d+)?
    )
    \s*$
    """,

    flags=re.VERBOSE,
)


def extract_final_answer(
    ground_truth
):

    lines = [

        line.strip()

        for line in str(
            ground_truth
        ).splitlines()

        if line.strip()
    ]


    if not lines:

        return (
            None,
            "failed",
        )


    final_line = lines[-1]


    match = FINAL_LINE_PATTERN.fullmatch(
        final_line
    )


    if match is None:

        return (
            None,
            "failed",
        )


    answer = normalize_number(
        match.group(1)
    )


    return (
        answer,
        "final_line",
    )


# ============================================================
# ANSWER EXTRACTION AUDIT
# ============================================================

extraction_methods = Counter()

source_to_answer = {}


for source_index, row in (
    train_raw.items()
):

    answer, method = (
        extract_final_answer(
            row[
                "ground_truth"
            ]
        )
    )


    extraction_methods[
        method
    ] += 1


    source_to_answer[
        source_index
    ] = answer


print(
    "\n"
    + "=" * 70
)

print(
    "GROUND-TRUTH ANSWER EXTRACTION"
)

print(
    "=" * 70
)


print(
    "\nCustom TRAIN dialogues:",
    len(train_raw)
)


for method in [

    "final_line",
    "failed",

]:

    count = extraction_methods[
        method
    ]


    print(
        f"\n{method:15s}: "
        f"{count:4d} "
        f"({(
            count
            / len(train_raw)
            * 100
        ):.2f}%)"
    )


failed_sources = [

    source_index

    for source_index, answer
    in source_to_answer.items()

    if answer is None
]


print(
    "\nFailed extractions:",
    len(
        failed_sources
    )
)


# ============================================================
# PRINT EXAMPLE EXTRACTIONS FOR MANUAL VALIDATION
#
# Deterministic first 3 examples from each extraction method.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ANSWER EXTRACTION EXAMPLES"
)

print(
    "=" * 70
)


examples_printed = Counter()


for source_index in train_source_indices:

    raw_row = train_raw[
        source_index
    ]


    answer, method = (
        extract_final_answer(
            raw_row[
                "ground_truth"
            ]
        )
    )


    if (
        examples_printed[
            method
        ]
        >= 3
    ):

        continue


    examples_printed[
        method
    ] += 1


    ground_truth = str(
        raw_row[
            "ground_truth"
        ]
    )


    tail = ground_truth[
        -250:
    ]


    print(
        f"\nMethod: "
        f"{method}"
    )

    print(
        f"Source row: "
        f"{source_index}"
    )

    print(
        f"Extracted answer: "
        f"{answer}"
    )

    print(
        "Ground-truth tail:"
    )

    print(
        tail
    )


# ============================================================
# IMPORTANT VALIDATION GATE
#
# Do not continue to local-feedback analysis if answer
# extraction itself is unreliable.
# ============================================================

successful_extractions = (

    len(train_raw)
    -
    len(failed_sources)
)


extraction_rate = (

    successful_extractions
    /
    len(train_raw)
)


print(
    "\n"
    + "=" * 70
)

print(
    "ANSWER EXTRACTION VALIDATION"
)

print(
    "=" * 70
)


print(
    f"\nSuccessful extraction: "
    f"{successful_extractions}"
)

print(
    f"Extraction rate: "
    f"{extraction_rate * 100:.2f}%"
)


# We require very high structural coverage.
#
# Manual inspection of the examples is STILL required
# before using this as a correctness label.

if extraction_rate < 0.99:

    print(
        "\nANSWER EXTRACTION VALIDATION: FAILED"
    )

    print(
        "Do not continue to student-response scoring."
    )


else:

    print(
        "\nANSWER EXTRACTION VALIDATION: "
        "STRUCTURALLY PASSED"
    )

    print(
        "Manual review of extraction examples "
        "is required before SI2-B."
    )


# ============================================================
# SAVE ONLY EXTRACTION AUDIT
#
# IMPORTANT:
#
# We intentionally STOP HERE.
#
# We do NOT yet declare replies correct/incorrect.
# That depends on validating that our ground-truth
# final-answer extraction is genuinely correct.
# ============================================================

result = {

    "experiment":
        "SI2-A_ground_truth_answer_extraction_audit",

    "custom_train_dialogues":
        len(
            train_raw
        ),

    "extraction_methods":
        {

            key:
                int(value)

            for key, value
            in extraction_methods.items()
        },

    "successful_extractions":
        int(
            successful_extractions
        ),

    "failed_extractions":
        int(
            len(
                failed_sources
            )
        ),

    "extraction_rate":
        float(
            extraction_rate
        ),
}


with OUTPUT_PATH.open(
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
    OUTPUT_PATH
)
