import json
import re
from pathlib import Path
from collections import Counter, defaultdict


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
    "si2_final_answer_anchor_audit.json"
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
    "SI2-C â€” FINAL-ANSWER ANCHOR AUDIT"
)

print("=" * 70)


# ============================================================
# CUSTOM TRAIN SOURCES
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
# NUMBER EXTRACTION
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


def extract_numbers(
    text
):

    result = []


    for match in NUMBER_PATTERN.finditer(
        str(text)
    ):

        value = normalize_number(
            match.group(0)
        )


        if value is not None:

            result.append(
                value
            )


    return result


def last_number(
    text
):

    numbers = extract_numbers(
        text
    )


    if not numbers:

        return None


    return numbers[-1]


def contains_number(
    text,
    number
):

    return (
        number
        in extract_numbers(
            text
        )
    )


# ============================================================
# DATASET-VALIDATED GROUND-TRUTH EXTRACTOR
# ============================================================

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


for source_index, raw_row in (
    train_raw.items()
):

    source_to_answer[
        source_index
    ] = extract_correct_answer(
        raw_row[
            "ground_truth"
        ]
    )


assert len(
    source_to_answer
) == 2295


# ============================================================
# COLLISION FLAGS
# ============================================================

answer_in_problem = {}

answer_in_incorrect_solution = {}


for source_index, raw_row in (
    train_raw.items()
):

    answer = source_to_answer[
        source_index
    ]


    answer_in_problem[
        source_index
    ] = contains_number(

        raw_row[
            "question"
        ],

        answer,
    )


    answer_in_incorrect_solution[
        source_index
    ] = contains_number(

        raw_row[
            "student_incorrect_solution"
        ],

        answer,
    )


# ============================================================
# HISTORY HELPERS
# ============================================================

def prior_student_final_answer(
    history,
    correct_answer,
):

    for turn in history:

        if (
            str(
                turn[
                    "user"
                ]
            )
            .strip()
            .lower()
            != "student"
        ):

            continue


        if (
            last_number(
                turn[
                    "text"
                ]
            )
            == correct_answer
        ):

            return True


    return False


def prior_teacher_final_answer(
    history,
    correct_answer,
):

    for turn in history:

        if (
            str(
                turn[
                    "user"
                ]
            )
            .strip()
            .lower()
            != "teacher"
        ):

            continue


        if (
            last_number(
                turn[
                    "text"
                ]
            )
            == correct_answer
        ):

            return True


    return False


# ============================================================
# SCORE TRANSITIONS
# ============================================================

scored = []


for transition in transitions:

    source_index = int(
        transition[
            "source_row_index"
        ]
    )


    correct_answer = (
        source_to_answer[
            source_index
        ]
    )


    pre_history = (
        transition[
            "pre_history"
        ]
    )


    teacher_text = (
        transition[
            "teacher_text"
        ]
    )


    student_reply = (
        transition[
            "student_reply"
        ]
    )


    # --------------------------------------------------------
    # OLD WEAK SIGNAL
    # --------------------------------------------------------

    reply_mentions_answer = (
        contains_number(
            student_reply,
            correct_answer,
        )
    )


    # --------------------------------------------------------
    # NEW STRICTER SIGNAL
    #
    # Student's LAST numeric expression equals
    # the reference final answer.
    # --------------------------------------------------------

    student_last = (
        last_number(
            student_reply
        )
    )


    reply_final_answer_correct = (

        student_last
        == correct_answer
    )


    student_finally_correct_before = (
        prior_student_final_answer(
            pre_history,
            correct_answer,
        )
    )


    first_final_answer_attainment = (

        reply_final_answer_correct

        and

        not student_finally_correct_before
    )


    # --------------------------------------------------------
    # STRONGER TEACHER-EXPOSURE HEURISTIC
    #
    # Instead of any mention of answer number,
    # ask whether teacher's LAST number equals answer.
    # --------------------------------------------------------

    teacher_last = (
        last_number(
            teacher_text
        )
    )


    current_teacher_final_exposes = (

        teacher_last
        == correct_answer
    )


    teacher_final_exposed_before = (
        prior_teacher_final_answer(
            pre_history,
            correct_answer,
        )
    )


    teacher_final_exposed_before_or_now = (

        teacher_final_exposed_before

        or

        current_teacher_final_exposes
    )


    autonomous_final_attainment = (

        first_final_answer_attainment

        and

        not teacher_final_exposed_before_or_now

        and

        not answer_in_problem[
            source_index
        ]
    )


    # Even stricter sensitivity variant:
    #
    # correct answer also never appeared numerically
    # in the original incorrect solution.

    ultra_strict_autonomous = (

        autonomous_final_attainment

        and

        not answer_in_incorrect_solution[
            source_index
        ]
    )


    scored.append({

        "transition_id":
            transition[
                "transition_id"
            ],

        "source_row_index":
            source_index,

        "terminal_outcome":
            transition[
                "terminal_outcome"
            ],

        "teacher_move":
            transition[
                "teacher_move"
            ],

        "remaining_teacher_moves":
            transition[
                "remaining_teacher_moves"
            ],

        "correct_answer":
            correct_answer,

        "student_last_number":
            student_last,

        "teacher_last_number":
            teacher_last,

        "reply_mentions_answer":
            reply_mentions_answer,

        "reply_final_answer_correct":
            reply_final_answer_correct,

        "first_final_answer_attainment":
            first_final_answer_attainment,

        "teacher_final_exposed_before":
            teacher_final_exposed_before,

        "current_teacher_final_exposes":
            current_teacher_final_exposes,

        "teacher_final_exposed_before_or_now":
            teacher_final_exposed_before_or_now,

        "answer_in_problem":
            answer_in_problem[
                source_index
            ],

        "answer_in_incorrect_solution":
            answer_in_incorrect_solution[
                source_index
            ],

        "autonomous_final_attainment":
            autonomous_final_attainment,

        "ultra_strict_autonomous":
            ultra_strict_autonomous,

        "teacher_text":
            teacher_text,

        "student_reply":
            student_reply,
    })


assert len(
    scored
) == 12498


# ============================================================
# OLD VS NEW SIGNAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "OLD MENTION SIGNAL VS FINAL-ANSWER SIGNAL"
)

print(
    "=" * 70
)


flags = [

    "reply_mentions_answer",
    "reply_final_answer_correct",
    "first_final_answer_attainment",
    "autonomous_final_attainment",
    "ultra_strict_autonomous",
]


flag_counts = {}


for flag in flags:

    count = sum(

        row[
            flag
        ]

        for row in scored
    )


    flag_counts[
        flag
    ] = count


    print(
        f"\n{flag}:"
    )

    print(
        f"  Count: "
        f"{count}"
    )

    print(
        f"  Percentage: "
        f"{(
            count
            / len(scored)
            * 100
        ):.2f}%"
    )


# ============================================================
# FINAL-ANSWER ATTAINMENT BY OUTCOME
# ============================================================

OUTCOMES = [

    "clean_success",
    "revealed_success",
    "failure",
]


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL-ANSWER ATTAINMENT BY TERMINAL OUTCOME"
)

print(
    "=" * 70
)


by_outcome = {}


for outcome in OUTCOMES:

    subset = [

        row

        for row in scored

        if row[
            "terminal_outcome"
        ]
        == outcome
    ]


    total = len(
        subset
    )


    first = sum(

        row[
            "first_final_answer_attainment"
        ]

        for row in subset
    )


    autonomous = sum(

        row[
            "autonomous_final_attainment"
        ]

        for row in subset
    )


    ultra = sum(

        row[
            "ultra_strict_autonomous"
        ]

        for row in subset
    )


    by_outcome[
        outcome
    ] = {

        "transitions":
            total,

        "first_final_answer_attainment":
            first,

        "autonomous_final_attainment":
            autonomous,

        "ultra_strict_autonomous":
            ultra,
    }


    print(
        f"\n{outcome}"
    )

    print(
        f"  Transitions: "
        f"{total}"
    )


    print(
        f"  First final-answer attainment: "
        f"{first} "
        f"({(
            first
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Autonomous final attainment: "
        f"{autonomous} "
        f"({(
            autonomous
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Ultra-strict autonomous: "
        f"{ultra} "
        f"({(
            ultra
            / total
            * 100
        ):.2f}%)"
    )


# ============================================================
# POSITIVE ANCHOR PRECISION
#
# How often does each candidate event occur in a
# clean-success trajectory?
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CANDIDATE POSITIVE-ANCHOR PRECISION"
)

print(
    "=" * 70
)


anchor_precision = {}


for flag in [

    "autonomous_final_attainment",
    "ultra_strict_autonomous",

]:

    rows = [

        row

        for row in scored

        if row[
            flag
        ]
    ]


    outcomes = Counter(

        row[
            "terminal_outcome"
        ]

        for row in rows
    )


    total = len(
        rows
    )


    clean = outcomes[
        "clean_success"
    ]


    precision = (

        clean
        / total

        if total

        else 0.0
    )


    anchor_precision[
        flag
    ] = {

        "n":
            total,

        "clean":
            clean,

        "revealed":
            outcomes[
                "revealed_success"
            ],

        "failure":
            outcomes[
                "failure"
            ],

        "clean_precision":
            precision,
    }


    print(
        f"\n{flag}"
    )

    print(
        f"  Events: "
        f"{total}"
    )

    print(
        f"  clean_success: "
        f"{clean}"
    )

    print(
        f"  revealed_success: "
        f"{outcomes['revealed_success']}"
    )

    print(
        f"  failure: "
        f"{outcomes['failure']}"
    )

    print(
        f"  Clean precision: "
        f"{precision * 100:.2f}%"
    )


# ============================================================
# MOVE DISTRIBUTION OF STRICT ANCHOR
#
# DESCRIPTIVE ONLY.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "AUTONOMOUS FINAL ATTAINMENT BY MOVE"
)

print(
    "=" * 70
)


move_counts = Counter(

    row[
        "teacher_move"
    ]

    for row in scored

    if row[
        "autonomous_final_attainment"
    ]
)


for move in [

    "generic",
    "probing",
    "focus",
    "telling",

]:

    print(
        f"\n{move:10s}: "
        f"{move_counts[move]}"
    )


# ============================================================
# HOW MANY OLD NUMERIC EVENTS ARE REMOVED?
# ============================================================

removed_numeric_false_candidates = [

    row

    for row in scored

    if (
        row[
            "reply_mentions_answer"
        ]

        and

        not row[
            "reply_final_answer_correct"
        ]
    )
]


removed_by_outcome = Counter(

    row[
        "terminal_outcome"
    ]

    for row
    in removed_numeric_false_candidates
)


print(
    "\n"
    + "=" * 70
)

print(
    "EVENTS REMOVED BY FINAL-ANSWER REQUIREMENT"
)

print(
    "=" * 70
)


print(
    "\nRemoved events:",
    len(
        removed_numeric_false_candidates
    ),
)


for outcome in OUTCOMES:

    print(
        f"{outcome:18s}: "
        f"{removed_by_outcome[outcome]}"
    )


# ============================================================
# MANUAL AUDIT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MANUAL FINAL-ANSWER EXAMPLES"
)

print(
    "=" * 70
)


def print_examples(
    title,
    rows,
    limit=5,
):

    print(
        f"\n--- {title} ---"
    )


    if not rows:

        print(
            "No examples."
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
            "Outcome:",
            row[
                "terminal_outcome"
            ],
        )

        print(
            "Move:",
            row[
                "teacher_move"
            ],
        )

        print(
            "Correct answer:",
            row[
                "correct_answer"
            ],
        )

        print(
            "Student last number:",
            row[
                "student_last_number"
            ],
        )

        print(
            "Teacher last number:",
            row[
                "teacher_last_number"
            ],
        )

        print(
            "Answer in problem:",
            row[
                "answer_in_problem"
            ],
        )

        print(
            "Answer in incorrect solution:",
            row[
                "answer_in_incorrect_solution"
            ],
        )


        print(
            "\nTeacher:"
        )

        print(
            row[
                "teacher_text"
            ]
        )


        print(
            "\nStudent:"
        )

        print(
            row[
                "student_reply"
            ]
        )


# ------------------------------------------------------------
# Clean strict positives
# ------------------------------------------------------------

clean_anchor = [

    row

    for row in scored

    if (
        row[
            "autonomous_final_attainment"
        ]

        and

        row[
            "terminal_outcome"
        ]
        == "clean_success"
    )
]


# ------------------------------------------------------------
# Remaining FAILURE false positives.
#
# These are now the crucial cases.
# ------------------------------------------------------------

failure_anchor = [

    row

    for row in scored

    if (
        row[
            "autonomous_final_attainment"
        ]

        and

        row[
            "terminal_outcome"
        ]
        == "failure"
    )
]


# ------------------------------------------------------------
# Remaining REVEALED false positives.
# ------------------------------------------------------------

revealed_anchor = [

    row

    for row in scored

    if (
        row[
            "autonomous_final_attainment"
        ]

        and

        row[
            "terminal_outcome"
        ]
        == "revealed_success"
    )
]


# ------------------------------------------------------------
# Examples correctly removed by last-number rule.
# ------------------------------------------------------------

removed_examples = (
    removed_numeric_false_candidates
)


print_examples(

    "CLEAN SUCCESS â€” "
    "AUTONOMOUS FINAL-ANSWER ATTAINMENT",

    clean_anchor,
)


print_examples(

    "FAILURE â€” "
    "REMAINING AUTONOMOUS FINAL-ANSWER EVENTS",

    failure_anchor,
)


print_examples(

    "REVEALED SUCCESS â€” "
    "REMAINING AUTONOMOUS FINAL-ANSWER EVENTS",

    revealed_anchor,
)


print_examples(

    "REMOVED BECAUSE CORRECT NUMBER "
    "WAS NOT STUDENT'S FINAL NUMBER",

    removed_examples,
)


# ============================================================
# SAVE
# ============================================================

result = {

    "experiment":
        "SI2-C_final_answer_anchor",

    "transitions":
        len(
            scored
        ),

    "flag_counts":
        flag_counts,

    "by_outcome":
        by_outcome,

    "anchor_precision":
        anchor_precision,

    "autonomous_move_counts":
        dict(
            move_counts
        ),

    "removed_by_final_answer_requirement": {

        "count":
            len(
                removed_numeric_false_candidates
            ),

        "outcomes":
            dict(
                removed_by_outcome
            ),
    },
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
    "\n"
    + "=" * 70
)

print(
    "SI2-C FINAL-ANSWER ANCHOR AUDIT COMPLETE"
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
