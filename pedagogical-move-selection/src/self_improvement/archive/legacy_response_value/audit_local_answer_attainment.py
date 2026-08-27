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
    "si2_local_answer_attainment_audit.json"
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
    "SI2-B â€” LOCAL ANSWER ATTAINMENT AUDIT"
)

print("=" * 70)


# ============================================================
# CUSTOM TRAIN DIALOGUES
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


    return (
        f"{number:.12g}"
    )


def extract_numbers(
    text
):

    numbers = []


    for match in NUMBER_PATTERN.finditer(
        str(text)
    ):

        normalized = normalize_number(
            match.group(0)
        )


        if normalized is not None:

            numbers.append(
                normalized
            )


    return numbers


def extract_ground_truth_answer(
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

        return None


    final_line = lines[-1]


    match = FINAL_LINE_PATTERN.fullmatch(
        final_line
    )


    if match is None:

        return None


    return normalize_number(
        match.group(1)
    )


def contains_answer(
    text,
    answer
):

    return (
        answer
        in extract_numbers(
            text
        )
    )


# ============================================================
# TERMINAL OUTCOME
# ============================================================

def normalize_outcome(
    value
):

    if value is None:
        return None


    value = str(
        value
    ).strip()


    if not value:
        return None


    lower = (
        value.lower()
    )


    if lower == "yes":

        return (
            "clean_success"
        )


    if "reveal" in lower:

        return (
            "revealed_success"
        )


    if lower == "no":

        return (
            "failure"
        )


    raise ValueError(
        f"Unexpected outcome: {value}"
    )


# ============================================================
# ANSWER MAP
# ============================================================

source_to_answer = {}


for source_index, row in (
    train_raw.items()
):

    answer = (
        extract_ground_truth_answer(
            row[
                "ground_truth"
            ]
        )
    )


    assert answer is not None


    source_to_answer[
        source_index
    ] = answer


assert len(
    source_to_answer
) == 2295


# ============================================================
# AMBIGUITY AUDIT
#
# If the correct answer already occurs numerically in the
# original problem, merely observing that number later is
# weaker evidence of genuine solution attainment.
# ============================================================

answer_in_problem = {}

answer_in_incorrect_solution = {}


for source_index, row in (
    train_raw.items()
):

    answer = source_to_answer[
        source_index
    ]


    answer_in_problem[
        source_index
    ] = contains_answer(

        row[
            "question"
        ],

        answer,
    )


    answer_in_incorrect_solution[
        source_index
    ] = contains_answer(

        row[
            "student_incorrect_solution"
        ],

        answer,
    )


problem_collision_count = sum(
    answer_in_problem.values()
)

incorrect_solution_collision_count = sum(
    answer_in_incorrect_solution.values()
)


print(
    "\n"
    + "=" * 70
)

print(
    "ANSWER-NUMBER AMBIGUITY"
)

print(
    "=" * 70
)


print(
    "\nCorrect answer already appears "
    "in problem:",
    problem_collision_count,
)


print(
    f"Percentage: "
    f"{(
        problem_collision_count
        / len(train_raw)
        * 100
    ):.2f}%"
)


print(
    "\nCorrect answer appears in original "
    "incorrect solution:",
    incorrect_solution_collision_count,
)


print(
    f"Percentage: "
    f"{(
        incorrect_solution_collision_count
        / len(train_raw)
        * 100
    ):.2f}%"
)


# ============================================================
# HISTORY HELPERS
# ============================================================

def student_history_contains_answer(
    history,
    answer,
):

    for turn in history:

        user = (
            str(
                turn[
                    "user"
                ]
            )
            .strip()
            .lower()
        )


        if user != "student":

            continue


        if contains_answer(
            turn[
                "text"
            ],
            answer,
        ):

            return True


    return False


def teacher_history_contains_answer(
    history,
    answer,
):

    for turn in history:

        user = (
            str(
                turn[
                    "user"
                ]
            )
            .strip()
            .lower()
        )


        if user != "teacher":

            continue


        if contains_answer(
            turn[
                "text"
            ],
            answer,
        ):

            return True


    return False


# ============================================================
# SCORE EVERY TRANSITION
# ============================================================

scored_transitions = []


for transition in transitions:

    source_index = int(
        transition[
            "source_row_index"
        ]
    )


    assert (
        source_index
        in source_to_answer
    )


    answer = source_to_answer[
        source_index
    ]


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


    student_had_answer_before = (
        student_history_contains_answer(
            pre_history,
            answer,
        )
    )


    teacher_exposed_before = (
        teacher_history_contains_answer(
            pre_history,
            answer,
        )
    )


    current_teacher_exposes = (
        contains_answer(
            teacher_text,
            answer,
        )
    )


    teacher_exposed_before_or_now = (

        teacher_exposed_before

        or

        current_teacher_exposes
    )


    student_reply_contains = (
        contains_answer(
            student_reply,
            answer,
        )
    )


    # --------------------------------------------------------
    # First observed student attainment:
    #
    # student did not previously state the correct number,
    # but now does.
    # --------------------------------------------------------

    first_student_attainment = (

        student_reply_contains

        and

        not student_had_answer_before
    )


    autonomous_first_attainment = (

        first_student_attainment

        and

        not teacher_exposed_before_or_now

        and

        not answer_in_problem[
            source_index
        ]
    )


    teacher_exposed_first_attainment = (

        first_student_attainment

        and

        teacher_exposed_before_or_now
    )


    scored_transitions.append({

        "transition_id":
            transition[
                "transition_id"
            ],

        "source_row_index":
            source_index,

        "qid":
            transition[
                "qid"
            ],

        "teacher_move":
            transition[
                "teacher_move"
            ],

        "terminal_outcome":
            transition[
                "terminal_outcome"
            ],

        "remaining_teacher_moves":
            int(
                transition[
                    "remaining_teacher_moves"
                ]
            ),

        "correct_answer":
            answer,

        "answer_in_problem":
            bool(
                answer_in_problem[
                    source_index
                ]
            ),

        "answer_in_incorrect_solution":
            bool(
                answer_in_incorrect_solution[
                    source_index
                ]
            ),

        "student_had_answer_before":
            bool(
                student_had_answer_before
            ),

        "teacher_exposed_before":
            bool(
                teacher_exposed_before
            ),

        "current_teacher_exposes_answer":
            bool(
                current_teacher_exposes
            ),

        "teacher_exposed_before_or_now":
            bool(
                teacher_exposed_before_or_now
            ),

        "student_reply_contains_answer":
            bool(
                student_reply_contains
            ),

        "first_student_attainment":
            bool(
                first_student_attainment
            ),

        "autonomous_first_attainment":
            bool(
                autonomous_first_attainment
            ),

        "teacher_exposed_first_attainment":
            bool(
                teacher_exposed_first_attainment
            ),

        "teacher_text":
            teacher_text,

        "student_reply":
            student_reply,
    })


assert len(
    scored_transitions
) == 12498


# ============================================================
# TRANSITION-LEVEL COVERAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRANSITION-LEVEL ANSWER SIGNAL"
)

print(
    "=" * 70
)


transition_flags = [

    "student_reply_contains_answer",
    "first_student_attainment",
    "autonomous_first_attainment",
    "teacher_exposed_first_attainment",
    "current_teacher_exposes_answer",
]


transition_flag_counts = {}


for flag in transition_flags:

    count = sum(

        row[
            flag
        ]

        for row
        in scored_transitions
    )


    rate = (
        count
        / len(
            scored_transitions
        )
    )


    transition_flag_counts[
        flag
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
        f"\n{flag}:"
    )

    print(
        f"  Count: "
        f"{count}"
    )

    print(
        f"  Percentage: "
        f"{rate * 100:.2f}%"
    )


# ============================================================
# FIRST ATTAINMENT BY TERMINAL OUTCOME
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FIRST ATTAINMENT BY TERMINAL OUTCOME"
)

print(
    "=" * 70
)


OUTCOMES = [

    "clean_success",
    "revealed_success",
    "failure",
]


attainment_by_outcome = {}


for outcome in OUTCOMES:

    subset = [

        row

        for row
        in scored_transitions

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
            "first_student_attainment"
        ]

        for row
        in subset
    )


    autonomous = sum(

        row[
            "autonomous_first_attainment"
        ]

        for row
        in subset
    )


    exposed = sum(

        row[
            "teacher_exposed_first_attainment"
        ]

        for row
        in subset
    )


    reply_answer = sum(

        row[
            "student_reply_contains_answer"
        ]

        for row
        in subset
    )


    attainment_by_outcome[
        outcome
    ] = {

        "transitions":
            total,

        "reply_contains_answer":
            int(
                reply_answer
            ),

        "first_attainment":
            int(
                first
            ),

        "autonomous_first_attainment":
            int(
                autonomous
            ),

        "teacher_exposed_first_attainment":
            int(
                exposed
            ),
    }


    print(
        f"\n{outcome}"
    )

    print(
        f"  Transitions: "
        f"{total}"
    )

    print(
        f"  Reply contains answer: "
        f"{reply_answer} "
        f"({(
            reply_answer
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  First student attainment: "
        f"{first} "
        f"({(
            first
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Autonomous first attainment: "
        f"{autonomous} "
        f"({(
            autonomous
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Teacher-exposed first attainment: "
        f"{exposed} "
        f"({(
            exposed
            / total
            * 100
        ):.2f}%)"
    )


# ============================================================
# DIALOGUE-LEVEL SUMMARY
#
# Each dialogue should contribute at most one first attainment.
# ============================================================

by_source = defaultdict(
    list
)


for row in scored_transitions:

    by_source[
        row[
            "source_row_index"
        ]
    ].append(
        row
    )


dialogue_records = []


for source_index in train_source_indices:

    raw_row = train_raw[
        source_index
    ]


    outcome = normalize_outcome(
        raw_row.get(
            "self-correctness"
        )
    )


    if outcome is None:

        continue


    rows = by_source.get(
        source_index,
        [],
    )


    first_attainment_rows = [

        row

        for row in rows

        if row[
            "first_student_attainment"
        ]
    ]


    # Due to "had answer before",
    # normally at most one should exist.

    if len(
        first_attainment_rows
    ) > 1:

        raise AssertionError(
            "Multiple first attainments "
            f"for source {source_index}"
        )


    if first_attainment_rows:

        attainment = (
            first_attainment_rows[
                0
            ]
        )


        attained = True


        autonomous = bool(
            attainment[
                "autonomous_first_attainment"
            ]
        )


        teacher_exposed = bool(
            attainment[
                "teacher_exposed_first_attainment"
            ]
        )


        first_move = (
            attainment[
                "teacher_move"
            ]
        )


        first_remaining = int(
            attainment[
                "remaining_teacher_moves"
            ]
        )


    else:

        attained = False
        autonomous = False
        teacher_exposed = False
        first_move = None
        first_remaining = None


    dialogue_records.append({

        "source_row_index":
            source_index,

        "terminal_outcome":
            outcome,

        "correct_answer":
            source_to_answer[
                source_index
            ],

        "answer_in_problem":
            bool(
                answer_in_problem[
                    source_index
                ]
            ),

        "answer_in_incorrect_solution":
            bool(
                answer_in_incorrect_solution[
                    source_index
                ]
            ),

        "first_student_attainment":
            attained,

        "autonomous_first_attainment":
            autonomous,

        "teacher_exposed_first_attainment":
            teacher_exposed,

        "first_attainment_move":
            first_move,

        "remaining_teacher_moves_at_attainment":
            first_remaining,
    })


print(
    "\n"
    + "=" * 70
)

print(
    "DIALOGUE-LEVEL ATTAINMENT"
)

print(
    "=" * 70
)


dialogue_outcome_summary = {}


for outcome in OUTCOMES:

    subset = [

        row

        for row
        in dialogue_records

        if row[
            "terminal_outcome"
        ]
        == outcome
    ]


    total = len(
        subset
    )


    attained = sum(

        row[
            "first_student_attainment"
        ]

        for row
        in subset
    )


    autonomous = sum(

        row[
            "autonomous_first_attainment"
        ]

        for row
        in subset
    )


    exposed = sum(

        row[
            "teacher_exposed_first_attainment"
        ]

        for row
        in subset
    )


    no_attainment = (
        total
        - attained
    )


    dialogue_outcome_summary[
        outcome
    ] = {

        "dialogues":
            total,

        "attained":
            int(
                attained
            ),

        "autonomous":
            int(
                autonomous
            ),

        "teacher_exposed":
            int(
                exposed
            ),

        "no_attainment":
            int(
                no_attainment
            ),
    }


    print(
        f"\n{outcome}"
    )

    print(
        f"  Dialogues: "
        f"{total}"
    )

    print(
        f"  Student reaches answer: "
        f"{attained} "
        f"({(
            attained
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Autonomous attainment: "
        f"{autonomous} "
        f"({(
            autonomous
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  Teacher-exposed attainment: "
        f"{exposed} "
        f"({(
            exposed
            / total
            * 100
        ):.2f}%)"
    )


    print(
        f"  No detected attainment: "
        f"{no_attainment} "
        f"({(
            no_attainment
            / total
            * 100
        ):.2f}%)"
    )


# ============================================================
# HIGH-CONFIDENCE AUTONOMOUS ATTAINMENT
#
# Exclude:
# - answer already visible in problem
# - any teacher exposure before/current turn
#
# This is our strongest candidate positive event.
# ============================================================

high_confidence = [

    row

    for row
    in scored_transitions

    if (
        row[
            "autonomous_first_attainment"
        ]

        and

        not row[
            "answer_in_problem"
        ]
    )
]


high_confidence_outcomes = Counter(

    row[
        "terminal_outcome"
    ]

    for row
    in high_confidence
)


print(
    "\n"
    + "=" * 70
)

print(
    "HIGH-CONFIDENCE AUTONOMOUS ATTAINMENT"
)

print(
    "=" * 70
)


print(
    "\nTotal events:",
    len(
        high_confidence
    ),
)


for outcome in OUTCOMES:

    count = (
        high_confidence_outcomes[
            outcome
        ]
    )


    print(
        f"\n{outcome:18s}: "
        f"{count}"
    )


    if high_confidence:

        print(
            f"  Percentage: "
            f"{(
                count
                / len(high_confidence)
                * 100
            ):.2f}%"
        )


# ============================================================
# FIRST ATTAINMENT BY TEACHER MOVE
#
# DESCRIPTIVE ONLY.
#
# This is NOT causal evidence that a move causes attainment.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FIRST ATTAINMENT BY OBSERVED MOVE"
)

print(
    "=" * 70
)


first_attainment_rows = [

    row

    for row
    in scored_transitions

    if row[
        "first_student_attainment"
    ]
]


first_move_counts = Counter(

    row[
        "teacher_move"
    ]

    for row
    in first_attainment_rows
)


autonomous_move_counts = Counter(

    row[
        "teacher_move"
    ]

    for row
    in first_attainment_rows

    if row[
        "autonomous_first_attainment"
    ]
)


exposed_move_counts = Counter(

    row[
        "teacher_move"
    ]

    for row
    in first_attainment_rows

    if row[
        "teacher_exposed_first_attainment"
    ]
)


move_summary = {}


for move in [

    "generic",
    "probing",
    "focus",
    "telling",

]:

    total = (
        first_move_counts[
            move
        ]
    )


    autonomous = (
        autonomous_move_counts[
            move
        ]
    )


    exposed = (
        exposed_move_counts[
            move
        ]
    )


    move_summary[
        move
    ] = {

        "first_attainments":
            int(
                total
            ),

        "autonomous":
            int(
                autonomous
            ),

        "teacher_exposed":
            int(
                exposed
            ),
    }


    print(
        f"\n{move}"
    )

    print(
        f"  First attainments: "
        f"{total}"
    )

    print(
        f"  Autonomous: "
        f"{autonomous}"
    )

    print(
        f"  Teacher-exposed: "
        f"{exposed}"
    )


# ============================================================
# POSITION OF FIRST ATTAINMENT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FIRST ATTAINMENT POSITION"
)

print(
    "=" * 70
)


for category, rows in [

    (
        "all_first_attainment",
        first_attainment_rows,
    ),

    (
        "autonomous",
        [
            row
            for row
            in first_attainment_rows
            if row[
                "autonomous_first_attainment"
            ]
        ],
    ),

    (
        "teacher_exposed",
        [
            row
            for row
            in first_attainment_rows
            if row[
                "teacher_exposed_first_attainment"
            ]
        ],
    ),

]:

    values = np.array([

        row[
            "remaining_teacher_moves"
        ]

        for row
        in rows

    ], dtype=np.int64)


    print(
        f"\n{category}"
    )


    if len(values) == 0:

        print(
            "  No examples."
        )

        continue


    print(
        f"  n: "
        f"{len(values)}"
    )

    print(
        f"  mean remaining teacher moves: "
        f"{np.mean(values):.3f}"
    )

    print(
        f"  median remaining teacher moves: "
        f"{np.median(values):.3f}"
    )

    print(
        f"  fraction at terminal teacher move: "
        f"{np.mean(values == 0):.4f}"
    )


# ============================================================
# MANUAL AUDIT EXAMPLES
#
# We need to inspect text before adopting this as a signal.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MANUAL SIGNAL EXAMPLES"
)

print(
    "=" * 70
)


def print_examples(
    title,
    rows,
    limit=3,
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
            "Remaining teacher moves:",
            row[
                "remaining_teacher_moves"
            ],
        )

        print(
            "Teacher exposed before:",
            row[
                "teacher_exposed_before"
            ],
        )

        print(
            "Current teacher exposes:",
            row[
                "current_teacher_exposes_answer"
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
# 1. Strong clean-success autonomous examples
# ------------------------------------------------------------

examples_clean_autonomous = [

    row

    for row
    in scored_transitions

    if (
        row[
            "terminal_outcome"
        ]
        == "clean_success"

        and

        row[
            "autonomous_first_attainment"
        ]
    )
]


# ------------------------------------------------------------
# 2. Revealed-success examples where teacher exposed answer
# ------------------------------------------------------------

examples_revealed_exposed = [

    row

    for row
    in scored_transitions

    if (
        row[
            "terminal_outcome"
        ]
        == "revealed_success"

        and

        row[
            "teacher_exposed_first_attainment"
        ]
    )
]


# ------------------------------------------------------------
# 3. Failure trajectories where correct number nevertheless
#    appears in student's response.
#
# These are especially important possible false positives.
# ------------------------------------------------------------

examples_failure_answer = [

    row

    for row
    in scored_transitions

    if (
        row[
            "terminal_outcome"
        ]
        == "failure"

        and

        row[
            "first_student_attainment"
        ]
    )
]


# ------------------------------------------------------------
# 4. Answer-collision examples where the final answer number
#    already exists in the problem.
# ------------------------------------------------------------

examples_problem_collision = [

    row

    for row
    in scored_transitions

    if (
        row[
            "first_student_attainment"
        ]

        and

        row[
            "answer_in_problem"
        ]
    )
]


print_examples(

    "CLEAN SUCCESS â€” "
    "AUTONOMOUS FIRST ATTAINMENT",

    examples_clean_autonomous,
)


print_examples(

    "REVEALED SUCCESS â€” "
    "TEACHER-EXPOSED FIRST ATTAINMENT",

    examples_revealed_exposed,
)


print_examples(

    "FAILURE â€” "
    "STUDENT STILL MENTIONS CORRECT NUMBER",

    examples_failure_answer,
)


print_examples(

    "AMBIGUOUS â€” "
    "CORRECT NUMBER ALREADY IN PROBLEM",

    examples_problem_collision,
)


# ============================================================
# SAVE
# ============================================================

result = {

    "experiment":
        "SI2-B_local_answer_attainment_audit",

    "scope":
        "custom_train_only",

    "dialogues":
        len(
            dialogue_records
        ),

    "transitions":
        len(
            scored_transitions
        ),

    "ambiguity": {

        "answer_in_problem":
            int(
                problem_collision_count
            ),

        "answer_in_problem_rate":
            float(
                problem_collision_count
                / len(train_raw)
            ),

        "answer_in_incorrect_solution":
            int(
                incorrect_solution_collision_count
            ),

        "answer_in_incorrect_solution_rate":
            float(
                incorrect_solution_collision_count
                / len(train_raw)
            ),
    },

    "transition_flags":
        transition_flag_counts,

    "attainment_by_outcome":
        attainment_by_outcome,

    "dialogue_outcome_summary":
        dialogue_outcome_summary,

    "high_confidence_autonomous": {

        "count":
            len(
                high_confidence
            ),

        "outcomes":
            dict(
                high_confidence_outcomes
            ),
    },

    "first_attainment_by_move":
        move_summary,
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
    "SI2-B LOCAL ANSWER ATTAINMENT AUDIT COMPLETE"
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
