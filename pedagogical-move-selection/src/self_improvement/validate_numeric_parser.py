import json
import re
from pathlib import Path


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


# ============================================================
# OLD BUGGY PATTERN
#
# Problem:
#
# (?![\w.])
#
# prevents matching:
#
#     52.
#
# because "." immediately follows the number.
# ============================================================

OLD_NUMBER_PATTERN = re.compile(

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


# ============================================================
# CORRECTED PATTERN
#
# We still prohibit a following WORD character,
# but sentence punctuation such as:
#
#     52.
#     52,
#     52?
#
# is allowed.
#
# Decimal numbers such as 4.50 are still consumed
# by (?:\.\d+)?.
# ============================================================

NEW_NUMBER_PATTERN = re.compile(

    r"""
    (?<![\w.])
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    (?!\w)
    """,

    flags=re.VERBOSE,
)


# ============================================================
# GROUND-TRUTH FINAL LINE
# ============================================================

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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_number(
    value
):

    value = (
        str(value)
        .strip()
        .replace(
            ",",
            "",
        )
    )


    try:

        number = float(
            value
        )

    except ValueError:

        return None


    if abs(
        number
        - round(number)
    ) < 1e-10:

        return str(
            int(
                round(number)
            )
        )


    return f"{number:.12g}"


# ============================================================
# EXTRACTION
# ============================================================

def extract_numbers(
    text,
    pattern,
):

    values = []


    for match in pattern.finditer(
        str(text)
    ):

        value = normalize_number(
            match.group(0)
        )


        if value is not None:

            values.append(
                value
            )


    return values


def old_numbers(
    text
):

    return extract_numbers(
        text,
        OLD_NUMBER_PATTERN,
    )


def new_numbers(
    text
):

    return extract_numbers(
        text,
        NEW_NUMBER_PATTERN,
    )


def old_last_number(
    text
):

    values = old_numbers(
        text
    )


    if not values:

        return None


    return values[-1]


def new_last_number(
    text
):

    values = new_numbers(
        text
    )


    if not values:

        return None


    return values[-1]


# ============================================================
# SYNTHETIC UNIT TESTS
# ============================================================

TEST_CASES = [

    (
        "52.",
        [
            "52",
        ],
    ),

    (
        "4x13=52.",
        [
            "4",
            "13",
            "52",
        ],
    ),

    (
        "It costs $4.50 per batch.",
        [
            "4.5",
        ],
    ),

    (
        "The answer is 1,200.",
        [
            "1200",
        ],
    ),

    (
        "The result is 30%.",
        [
            "30",
        ],
    ),

    (
        "James had 90 pairs of socks.",
        [
            "90",
        ],
    ),

    (
        "6x13 = 78, but 4x13=52.",
        [
            "6",
            "13",
            "78",
            "4",
            "13",
            "52",
        ],
    ),

    # --------------------------------------------------------
    # Known limitation:
    #
    # fractions are represented as their component numbers.
    #
    # We record this explicitly rather than pretending
    # "3/4" is parsed as 0.75.
    # --------------------------------------------------------

    (
        "The fraction is 3/4.",
        [
            "3",
            "4",
        ],
    ),
]


print("=" * 70)

print(
    "SI2-C0 â€” NUMERIC PARSER VALIDATION"
)

print("=" * 70)


print(
    "\n"
    + "=" * 70
)

print(
    "SYNTHETIC PARSER TESTS"
)

print(
    "=" * 70
)


synthetic_failures = 0


for text, expected in TEST_CASES:

    actual = new_numbers(
        text
    )


    passed = (
        actual
        == expected
    )


    if not passed:

        synthetic_failures += 1


    print(
        "\nText:"
    )

    print(
        repr(
            text
        )
    )

    print(
        "Expected:",
        expected,
    )

    print(
        "Actual:",
        actual,
    )

    print(
        "Passed:",
        passed,
    )


print(
    "\nSynthetic failures:",
    synthetic_failures,
)


assert synthetic_failures == 0


# ============================================================
# CUSTOM TRAIN SOURCES / ANSWERS
# ============================================================

train_source_indices = sorted({

    int(
        row[
            "source_row_index"
        ]
    )

    for row
    in processed_train
})


assert len(
    train_source_indices
) == 2295


def extract_correct_answer(
    ground_truth
):

    lines = [

        line.strip()

        for line in str(
            ground_truth
        ).splitlines()

        if line.strip()
    ]


    assert lines


    match = (
        FINAL_LINE_PATTERN.fullmatch(
            lines[-1]
        )
    )


    assert match


    return normalize_number(
        match.group(1)
    )


source_to_answer = {}


for source_index in train_source_indices:

    source_to_answer[
        source_index
    ] = extract_correct_answer(

        combined_raw[
            source_index
        ][
            "ground_truth"
        ]
    )


assert len(
    source_to_answer
) == 2295


# ============================================================
# COMPARE OLD VS NEW ON REAL TRANSITIONS
# ============================================================

teacher_last_changed = []

student_last_changed = []


for transition in transitions:

    source_index = int(
        transition[
            "source_row_index"
        ]
    )


    answer = source_to_answer[
        source_index
    ]


    teacher_text = (
        transition[
            "teacher_text"
        ]
    )


    student_text = (
        transition[
            "student_reply"
        ]
    )


    old_teacher = old_last_number(
        teacher_text
    )

    new_teacher = new_last_number(
        teacher_text
    )


    old_student = old_last_number(
        student_text
    )

    new_student = new_last_number(
        student_text
    )


    if old_teacher != new_teacher:

        teacher_last_changed.append({

            "source_row_index":
                source_index,

            "correct_answer":
                answer,

            "old":
                old_teacher,

            "new":
                new_teacher,

            "text":
                teacher_text,
        })


    if old_student != new_student:

        student_last_changed.append({

            "source_row_index":
                source_index,

            "correct_answer":
                answer,

            "old":
                old_student,

            "new":
                new_student,

            "text":
                student_text,
        })


print(
    "\n"
    + "=" * 70
)

print(
    "REAL-DATA OLD VS NEW PARSER"
)

print(
    "=" * 70
)


print(
    "\nTeacher last-number changes:",
    len(
        teacher_last_changed
    ),
)


print(
    "Student last-number changes:",
    len(
        student_last_changed
    ),
)


print(
    f"\nTeacher change rate: "
    f"{(
        len(
            teacher_last_changed
        )
        / len(
            transitions
        )
        * 100
    ):.2f}%"
)


print(
    f"Student change rate: "
    f"{(
        len(
            student_last_changed
        )
        / len(
            transitions
        )
        * 100
    ):.2f}%"
)


# ============================================================
# EFFECT ON TEACHER EXPOSURE LABEL
# ============================================================

old_teacher_exposure_count = 0
new_teacher_exposure_count = 0

teacher_exposure_changed = 0


old_student_correct_count = 0
new_student_correct_count = 0

student_correct_changed = 0


for transition in transitions:

    source_index = int(
        transition[
            "source_row_index"
        ]
    )


    answer = source_to_answer[
        source_index
    ]


    teacher_text = (
        transition[
            "teacher_text"
        ]
    )


    student_text = (
        transition[
            "student_reply"
        ]
    )


    old_teacher_exposes = (

        old_last_number(
            teacher_text
        )

        == answer
    )


    new_teacher_exposes = (

        new_last_number(
            teacher_text
        )

        == answer
    )


    old_student_correct = (

        old_last_number(
            student_text
        )

        == answer
    )


    new_student_correct = (

        new_last_number(
            student_text
        )

        == answer
    )


    old_teacher_exposure_count += int(
        old_teacher_exposes
    )


    new_teacher_exposure_count += int(
        new_teacher_exposes
    )


    old_student_correct_count += int(
        old_student_correct
    )


    new_student_correct_count += int(
        new_student_correct
    )


    teacher_exposure_changed += int(

        old_teacher_exposes
        != new_teacher_exposes
    )


    student_correct_changed += int(

        old_student_correct
        != new_student_correct
    )


print(
    "\n"
    + "=" * 70
)

print(
    "LABEL IMPACT"
)

print(
    "=" * 70
)


print(
    "\nCurrent-teacher exposure:"
)

print(
    "  Old parser:",
    old_teacher_exposure_count,
)

print(
    "  New parser:",
    new_teacher_exposure_count,
)

print(
    "  Changed labels:",
    teacher_exposure_changed,
)


print(
    "\nStudent reply final-answer correct:"
)

print(
    "  Old parser:",
    old_student_correct_count,
)

print(
    "  New parser:",
    new_student_correct_count,
)

print(
    "  Changed labels:",
    student_correct_changed,
)


# ============================================================
# KNOWN MANUAL CASE
#
# Source 6 exposed the parser bug in SI2-C.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SOURCE 6 CHECK"
)

print(
    "=" * 70
)


source6_rows = [

    row

    for row in transitions

    if int(
        row[
            "source_row_index"
        ]
    ) == 6
]


print(
    "\nCorrect answer:",
    source_to_answer[
        6
    ],
)


for row in source6_rows:

    teacher_text = (
        row[
            "teacher_text"
        ]
    )


    if "78" not in teacher_text:
        continue


    print(
        "\nTeacher:"
    )

    print(
        teacher_text
    )


    print(
        "\nOLD numbers:"
    )

    print(
        old_numbers(
            teacher_text
        )
    )


    print(
        "OLD last:",
        old_last_number(
            teacher_text
        ),
    )


    print(
        "\nNEW numbers:"
    )

    print(
        new_numbers(
            teacher_text
        )
    )


    print(
        "NEW last:",
        new_last_number(
            teacher_text
        ),
    )


# ============================================================
# CHANGED REAL EXAMPLES
# ============================================================

def show_examples(
    title,
    rows,
    limit=10,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


    if not rows:

        print(
            "\nNo changed examples."
        )

        return


    for row in rows[
        :limit
    ]:

        print(
            "\n"
            + "-" * 60
        )


        print(
            "Source:",
            row[
                "source_row_index"
            ],
        )

        print(
            "Correct answer:",
            row[
                "correct_answer"
            ],
        )

        print(
            "OLD last:",
            row[
                "old"
            ],
        )

        print(
            "NEW last:",
            row[
                "new"
            ],
        )

        print(
            "Text:"
        )

        print(
            row[
                "text"
            ]
        )


show_examples(

    "CHANGED TEACHER EXAMPLES",

    teacher_last_changed,
)


show_examples(

    "CHANGED STUDENT EXAMPLES",

    student_last_changed,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "NUMERIC PARSER VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nSynthetic tests: PASSED"
)


if (
    len(
        teacher_last_changed
    )
    == 0

    and

    len(
        student_last_changed
    )
    == 0
):

    print(
        "Real-data impact: none"
    )


else:

    print(
        "Real-data impact: CONFIRMED"
    )


print(
    "\nCorrected parser is structurally ready "
    "for review."
)

print(
    "Do not recompute SI2-C anchor precision "
    "until these changed examples are inspected."
)
