import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from sklearn.model_selection import (
    GroupShuffleSplit,
)


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


OUTPUT_DIR = Path(
    "data/processed/self_improvement"
)

RESULT_DIR = Path(
    "results/self_improvement"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALL_PATH = (
    OUTPUT_DIR
    / "value_transitions_all.jsonl"
)

TRAIN_PATH = (
    OUTPUT_DIR
    / "value_transitions_train.jsonl"
)

VAL_PATH = (
    OUTPUT_DIR
    / "value_transitions_validation.jsonl"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "si1_value_transition_summary.json"
)


# ============================================================
# LABELS
# ============================================================

MOVE_LABELS = {
    "generic",
    "probing",
    "focus",
    "telling",
}


OUTCOME_LABELS = [
    "clean_success",
    "revealed_success",
    "failure",
]


MOVE_PATTERN = re.compile(
    r"^\s*\("
    r"(generic|probing|focus|telling)"
    r"\)\s*",
    flags=re.IGNORECASE,
)


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


def write_jsonl(
    path,
    rows,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def normalize_terminal_outcome(value):

    if value is None:
        return None


    value = str(
        value
    ).strip()


    if not value:
        return None


    lower = value.lower()


    if lower == "yes":

        return "clean_success"


    if "reveal" in lower:

        return "revealed_success"


    if lower == "no":

        return "failure"


    raise ValueError(
        f"Unexpected self-correctness: {value}"
    )


# ============================================================
# PARSE RAW CONVERSATION
# ============================================================

def parse_segment(segment):

    segment = segment.strip()


    if not segment:
        return None


    if ":" not in segment:

        return {
            "speaker":
                "unknown",

            "text":
                segment,
        }


    speaker, text = (
        segment.split(
            ":",
            1,
        )
    )


    speaker = (
        speaker
        .strip()
        .lower()
    )


    text = text.strip()


    if speaker.startswith(
        "teacher"
    ):

        speaker = "teacher"


    elif speaker.startswith(
        "student"
    ):

        speaker = "student"


    return {

        "speaker":
            speaker,

        "text":
            text,
    }


def parse_conversation(
    conversation
):

    parsed = []


    for segment in conversation.split(
        "|EOM|"
    ):

        turn = parse_segment(
            segment
        )


        if turn is not None:

            parsed.append(
                turn
            )


    return parsed


# ============================================================
# TEACHER MOVE
# ============================================================

def extract_move(
    text
):

    match = MOVE_PATTERN.match(
        text
    )


    if match is None:
        return None


    return (
        match.group(1)
        .lower()
    )


def strip_move_label(
    text
):

    return MOVE_PATTERN.sub(
        "",
        text,
        count=1,
    ).strip()


# ============================================================
# OBSERVABLE HISTORY FORMAT
#
# Important:
#
# - no dialog_act
# - no move label
# - no hidden MathDial attributes
# ============================================================

def observable_turn(
    speaker,
    text,
):

    if speaker == "teacher":

        user = "Teacher"

    elif speaker == "student":

        user = "Student"

    else:

        user = speaker


    return {

        "user":
            user,

        "text":
            text,
    }


# ============================================================
# LOAD SOURCE DATA
# ============================================================

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


assert len(combined_raw) == 2861

assert len(processed_train) == 14905


print("=" * 70)

print(
    "SI1-A â€” BUILD VALUE TRANSITIONS"
)

print("=" * 70)


# ============================================================
# MAP CUSTOM-TRAIN SOURCE ROWS TO THEIR LEAKAGE-SAFE GROUP
# ============================================================

source_to_group = {}

source_to_qid = {}


for example in processed_train:

    source_index = int(
        example[
            "source_row_index"
        ]
    )


    group_id = str(
        example[
            "group_id"
        ]
    )


    qid = str(
        example[
            "qid"
        ]
    )


    if source_index in source_to_group:

        assert (
            source_to_group[
                source_index
            ]
            == group_id
        )

        assert (
            source_to_qid[
                source_index
            ]
            == qid
        )

    else:

        source_to_group[
            source_index
        ] = group_id

        source_to_qid[
            source_index
        ] = qid


train_source_indices = sorted(
    source_to_group
)


print(
    "\nCustom TRAIN source dialogues:",
    len(train_source_indices)
)


assert len(
    train_source_indices
) == 2295


# ============================================================
# BUILD OBSERVED TRANSITIONS
# ============================================================

transitions = []


skipped_missing_outcome = 0

skipped_no_student_reply = 0

unlabeled_teacher_turns = 0


for source_index in train_source_indices:

    raw_row = combined_raw[
        source_index
    ]


    outcome = (
        normalize_terminal_outcome(
            raw_row.get(
                "self-correctness"
            )
        )
    )


    if outcome is None:

        skipped_missing_outcome += 1
        continue


    parsed = parse_conversation(
        raw_row[
            "conversation"
        ]
    )


    # ----------------------------------------------
    # Number of future labeled teacher moves.
    #
    # Metadata only.
    # NEVER feed this to the value model.
    # ----------------------------------------------

    teacher_move_positions = []


    for index, turn in enumerate(
        parsed
    ):

        if (
            turn[
                "speaker"
            ]
            == "teacher"
        ):

            move = extract_move(
                turn[
                    "text"
                ]
            )


            if move is not None:

                teacher_move_positions.append(
                    index
                )


    observable_history = []


    for turn_index, turn in enumerate(
        parsed
    ):

        speaker = turn[
            "speaker"
        ]


        raw_text = turn[
            "text"
        ]


        # ==========================================
        # LABELED TEACHER DECISION
        # ==========================================

        if speaker == "teacher":

            move = extract_move(
                raw_text
            )


            if move is None:

                unlabeled_teacher_turns += 1


                observable_history.append(
                    observable_turn(
                        "teacher",
                        raw_text,
                    )
                )

                continue


            assert move in MOVE_LABELS


            teacher_text = (
                strip_move_label(
                    raw_text
                )
            )


            # --------------------------------------
            # We only have immediate feedback if the
            # next parsed turn is a student.
            # --------------------------------------

            if (
                turn_index + 1
                >= len(parsed)
            ):

                skipped_no_student_reply += 1


                observable_history.append(
                    observable_turn(
                        "teacher",
                        teacher_text,
                    )
                )

                continue


            next_turn = parsed[
                turn_index + 1
            ]


            if (
                next_turn[
                    "speaker"
                ]
                != "student"
            ):

                skipped_no_student_reply += 1


                observable_history.append(
                    observable_turn(
                        "teacher",
                        teacher_text,
                    )
                )

                continue


            student_text = (
                next_turn[
                    "text"
                ]
            )


            # --------------------------------------
            # STATE BEFORE ACTION
            # --------------------------------------

            pre_history = [
                dict(x)
                for x in observable_history
            ]


            # --------------------------------------
            # STATE AFTER:
            #
            # teacher move utterance
            # +
            # subsequent student response
            # --------------------------------------

            post_history = (

                pre_history

                +

                [
                    observable_turn(
                        "teacher",
                        teacher_text,
                    ),

                    observable_turn(
                        "student",
                        student_text,
                    ),
                ]
            )


            # --------------------------------------
            # Distance to terminal teacher move
            #
            # For diagnostic/credit-assignment
            # analysis only.
            # --------------------------------------

            current_position_index = (
                teacher_move_positions.index(
                    turn_index
                )
            )


            remaining_teacher_moves = (

                len(
                    teacher_move_positions
                )

                - current_position_index

                - 1
            )


            transition = {

                "transition_id":
                    (
                        f"src{source_index}"
                        f"_turn{turn_index}"
                    ),

                "source_row_index":
                    source_index,

                "qid":
                    source_to_qid[
                        source_index
                    ],

                "group_id":
                    source_to_group[
                        source_index
                    ],

                "scenario":
                    raw_row[
                        "scenario"
                    ],

                # ----------------------------------
                # OBSERVABLE STATE
                # ----------------------------------

                "problem":
                    raw_row[
                        "question"
                    ],

                "pre_history":
                    pre_history,

                "post_history":
                    post_history,

                # ----------------------------------
                # OBSERVED ACTION / RESPONSE
                # ----------------------------------

                "teacher_move":
                    move,

                "teacher_text":
                    teacher_text,

                "student_reply":
                    student_text,

                # ----------------------------------
                # TERMINAL SUPERVISION
                #
                # NOT model input.
                # ----------------------------------

                "terminal_outcome":
                    outcome,

                # ----------------------------------
                # DIAGNOSTIC METADATA ONLY
                # ----------------------------------

                "remaining_teacher_moves":
                    remaining_teacher_moves,

                "self_typical_confusion":
                    raw_row.get(
                        "self-typical-confusion"
                    ),

                "self_typical_interactions":
                    raw_row.get(
                        "self-typical-interactions"
                    ),
            }


            transitions.append(
                transition
            )


        # ==========================================
        # UPDATE OBSERVABLE HISTORY
        #
        # We must avoid adding the student reply here
        # twice. The main loop will encounter it on
        # its normal next iteration.
        # ==========================================

        if speaker == "teacher":

            text_to_add = (
                strip_move_label(
                    raw_text
                )
                if extract_move(
                    raw_text
                )
                else raw_text
            )


            observable_history.append(
                observable_turn(
                    "teacher",
                    text_to_add,
                )
            )


        elif speaker == "student":

            observable_history.append(
                observable_turn(
                    "student",
                    raw_text,
                )
            )


# ============================================================
# BASIC VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRANSITION DATASET"
)

print(
    "=" * 70
)


print(
    "\nUsable transitions:",
    len(transitions)
)

print(
    "Dialogues skipped for missing outcome:",
    skipped_missing_outcome
)

print(
    "Teacher moves without student reply:",
    skipped_no_student_reply
)

print(
    "Unlabeled teacher turns encountered:",
    unlabeled_teacher_turns
)


assert len(
    transitions
) > 10000


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

invalid_history = 0

move_label_leakage = 0

invalid_outcomes = 0


for row in transitions:

    if (
        len(
            row[
                "post_history"
            ]
        )
        !=
        len(
            row[
                "pre_history"
            ]
        )
        + 2
    ):

        invalid_history += 1


    if (
        row[
            "terminal_outcome"
        ]
        not in OUTCOME_LABELS
    ):

        invalid_outcomes += 1


    # No historical turn should contain
    # a "(generic)" etc move annotation.

    for turn in (
        row[
            "pre_history"
        ]
        +
        row[
            "post_history"
        ]
    ):

        if MOVE_PATTERN.search(
            turn[
                "text"
            ]
        ):

            move_label_leakage += 1


print(
    "\nInvalid pre/post histories:",
    invalid_history
)

print(
    "Invalid outcomes:",
    invalid_outcomes
)

print(
    "Historical move-label leakage:",
    move_label_leakage
)


assert invalid_history == 0
assert invalid_outcomes == 0
assert move_label_leakage == 0


# ============================================================
# GLOBAL OUTCOME DISTRIBUTION
# ============================================================

global_outcomes = Counter(

    row[
        "terminal_outcome"
    ]

    for row
    in transitions
)


global_moves = Counter(

    row[
        "teacher_move"
    ]

    for row
    in transitions
)


print(
    "\n"
    + "=" * 70
)

print(
    "GLOBAL DISTRIBUTION"
)

print(
    "=" * 70
)


print(
    "\nTerminal outcomes:"
)


for outcome in OUTCOME_LABELS:

    count = global_outcomes[
        outcome
    ]


    print(
        f"  {outcome:18s}: "
        f"{count:5d} "
        f"({(
            count
            / len(transitions)
            * 100
        ):.2f}%)"
    )


print(
    "\nObserved moves:"
)


for move in [
    "generic",
    "probing",
    "focus",
    "telling",
]:

    count = global_moves[
        move
    ]


    print(
        f"  {move:10s}: "
        f"{count:5d} "
        f"({(
            count
            / len(transitions)
            * 100
        ):.2f}%)"
    )


# ============================================================
# INTERNAL SI TRAIN / VALIDATION SPLIT
#
# IMPORTANT:
#
# This split happens ONLY inside the original
# frozen MathDial TRAIN split.
#
# Groups remain intact.
#
# Main MathDial validation/test remain unused.
# ============================================================

indices = np.arange(
    len(transitions)
)


groups = np.array([

    row[
        "group_id"
    ]

    for row
    in transitions

])


target_val_fraction = 0.20


global_distribution = np.array([

    global_outcomes[
        outcome
    ]
    / len(
        transitions
    )

    for outcome
    in OUTCOME_LABELS

])


best_candidate = None


for random_state in range(
    300
):

    splitter = GroupShuffleSplit(

        n_splits=1,

        test_size=
            target_val_fraction,

        random_state=
            random_state,
    )


    train_idx, val_idx = next(
        splitter.split(
            indices,
            groups=groups,
        )
    )


    train_rows_candidate = [

        transitions[i]

        for i in train_idx
    ]


    val_rows_candidate = [

        transitions[i]

        for i in val_idx
    ]


    val_counts = Counter(

        row[
            "terminal_outcome"
        ]

        for row
        in val_rows_candidate
    )


    val_distribution = np.array([

        val_counts[
            outcome
        ]
        / len(
            val_rows_candidate
        )

        for outcome
        in OUTCOME_LABELS

    ])


    size_error = abs(

        (
            len(
                val_rows_candidate
            )
            / len(
                transitions
            )
        )

        -
        target_val_fraction
    )


    distribution_error = float(
        np.abs(
            val_distribution
            - global_distribution
        ).sum()
    )


    score = (

        size_error
        +
        distribution_error
    )


    candidate = {

        "score":
            score,

        "random_state":
            random_state,

        "train_idx":
            train_idx,

        "val_idx":
            val_idx,
    }


    if (
        best_candidate is None

        or

        score
        <
        best_candidate[
            "score"
        ]
    ):

        best_candidate = (
            candidate
        )


train_idx = (
    best_candidate[
        "train_idx"
    ]
)

val_idx = (
    best_candidate[
        "val_idx"
    ]
)


si_train = [

    transitions[i]

    for i in train_idx
]


si_val = [

    transitions[i]

    for i in val_idx
]


# ============================================================
# SPLIT LEAKAGE VALIDATION
# ============================================================

train_groups = {

    row[
        "group_id"
    ]

    for row
    in si_train
}


val_groups = {

    row[
        "group_id"
    ]

    for row
    in si_val
}


train_qids = {

    row[
        "qid"
    ]

    for row
    in si_train
}


val_qids = {

    row[
        "qid"
    ]

    for row
    in si_val
}


train_sources = {

    row[
        "source_row_index"
    ]

    for row
    in si_train
}


val_sources = {

    row[
        "source_row_index"
    ]

    for row
    in si_val
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
    "INTERNAL SI SPLIT"
)

print(
    "=" * 70
)


print(
    "\nSelected random state:",
    best_candidate[
        "random_state"
    ]
)

print(
    "Balance score:",
    f"{best_candidate['score']:.6f}"
)


print(
    "\nSI TRAIN transitions:",
    len(si_train)
)

print(
    "SI VALIDATION transitions:",
    len(si_val)
)


print(
    "\nSI TRAIN groups:",
    len(train_groups)
)

print(
    "SI VALIDATION groups:",
    len(val_groups)
)


print(
    "\nGroup overlap:",
    group_overlap
)

print(
    "QID overlap:",
    qid_overlap
)

print(
    "Source-dialogue overlap:",
    source_overlap
)


assert group_overlap == 0
assert qid_overlap == 0
assert source_overlap == 0


# ============================================================
# SPLIT DISTRIBUTIONS
# ============================================================

def print_distribution(
    name,
    rows,
):

    counts = Counter(

        row[
            "terminal_outcome"
        ]

        for row in rows
    )


    print(
        f"\n{name}:"
    )


    for outcome in OUTCOME_LABELS:

        count = counts[
            outcome
        ]


        print(
            f"  {outcome:18s}: "
            f"{count:5d} "
            f"({(
                count
                / len(rows)
                * 100
            ):.2f}%)"
        )


    return counts


train_outcomes = (
    print_distribution(
        "SI TRAIN",
        si_train,
    )
)


val_outcomes = (
    print_distribution(
        "SI VALIDATION",
        si_val,
    )
)


# ============================================================
# CREDIT-ASSIGNMENT DIAGNOSTIC
#
# How far are observed transitions from the end?
# ============================================================

remaining = np.array([

    row[
        "remaining_teacher_moves"
    ]

    for row
    in transitions

])


print(
    "\n"
    + "=" * 70
)

print(
    "DISTANCE-TO-TERMINAL DIAGNOSTIC"
)

print(
    "=" * 70
)


print(
    "\nMinimum remaining teacher moves:",
    int(
        remaining.min()
    )
)

print(
    "Median remaining teacher moves:",
    float(
        np.median(
            remaining
        )
    )
)

print(
    "P90 remaining teacher moves:",
    float(
        np.percentile(
            remaining,
            90
        )
    )
)

print(
    "Maximum remaining teacher moves:",
    int(
        remaining.max()
    )
)


for move in [
    "generic",
    "probing",
    "focus",
    "telling",
]:

    values = np.array([

        row[
            "remaining_teacher_moves"
        ]

        for row
        in transitions

        if row[
            "teacher_move"
        ]
        == move
    ])


    print(
        f"\n{move}"
    )

    print(
        f"  count: "
        f"{len(values)}"
    )

    print(
        f"  median remaining: "
        f"{np.median(values):.2f}"
    )

    print(
        f"  mean remaining: "
        f"{np.mean(values):.2f}"
    )


# ============================================================
# SAVE
# ============================================================

write_jsonl(
    ALL_PATH,
    transitions,
)

write_jsonl(
    TRAIN_PATH,
    si_train,
)

write_jsonl(
    VAL_PATH,
    si_val,
)


summary = {

    "scope":
        "frozen_mathdial_train_only",

    "usable_transitions":
        len(
            transitions
        ),

    "skipped_missing_outcome_dialogues":
        skipped_missing_outcome,

    "skipped_teacher_moves_without_student_reply":
        skipped_no_student_reply,

    "global_outcomes":
        dict(
            global_outcomes
        ),

    "global_moves":
        dict(
            global_moves
        ),

    "si_internal_split": {

        "random_state":
            best_candidate[
                "random_state"
            ],

        "balance_score":
            float(
                best_candidate[
                    "score"
                ]
            ),

        "train_transitions":
            len(
                si_train
            ),

        "validation_transitions":
            len(
                si_val
            ),

        "train_groups":
            len(
                train_groups
            ),

        "validation_groups":
            len(
                val_groups
            ),

        "group_overlap":
            group_overlap,

        "qid_overlap":
            qid_overlap,

        "source_overlap":
            source_overlap,

        "train_outcomes":
            dict(
                train_outcomes
            ),

        "validation_outcomes":
            dict(
                val_outcomes
            ),
    },

    "distance_to_terminal": {

        "minimum":
            int(
                remaining.min()
            ),

        "median":
            float(
                np.median(
                    remaining
                )
            ),

        "p90":
            float(
                np.percentile(
                    remaining,
                    90
                )
            ),

        "maximum":
            int(
                remaining.max()
            ),
    },
}


with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\n"
    + "=" * 70
)

print(
    "SI1-A VALUE TRANSITION VALIDATION: PASSED"
)

print(
    "=" * 70
)


print(
    "\nSaved:"
)

print(
    ALL_PATH
)

print(
    TRAIN_PATH
)

print(
    VAL_PATH
)

print(
    SUMMARY_PATH
)
