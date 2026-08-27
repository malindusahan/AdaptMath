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

PROCESSED_ALL = Path(
    "data/processed/mathdial/all_examples.jsonl"
)

OUTPUT_PATH = Path(
    "results/self_improvement/"
    "si0_feedback_transition_audit.json"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


MOVE_PATTERN = re.compile(
    r"^\s*\("
    r"(generic|probing|focus|telling)"
    r"\)\s*",
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


raw_train = load_jsonl(
    RAW_TRAIN
)

raw_test = load_jsonl(
    RAW_TEST
)

processed_examples = load_jsonl(
    PROCESSED_ALL
)


print("=" * 70)

print(
    "SI0 â€” MATHDIAL FEEDBACK TRANSITION AUDIT"
)

print("=" * 70)


print(
    "\nOfficial train dialogues:",
    len(raw_train)
)

print(
    "Official test dialogues:",
    len(raw_test)
)

print(
    "Processed supervised examples:",
    len(processed_examples)
)


assert len(raw_train) == 2262
assert len(raw_test) == 599
assert len(processed_examples) == 18607


# ============================================================
# RECONSTRUCT SAME COMBINED ROW ORDER USED IN PREPROCESSING
#
# source_row_index was created after:
#
# official train
# +
# official test
#
# ============================================================

combined_raw = (
    raw_train
    + raw_test
)


assert len(combined_raw) == 2861


# ============================================================
# MAP EACH SOURCE ROW TO OUR FROZEN CUSTOM SPLIT
#
# All examples from the same source dialogue should belong
# to exactly one custom split.
# ============================================================

source_to_splits = defaultdict(
    set
)


for example in processed_examples:

    source_index = int(
        example["source_row_index"]
    )

    # Infer split from membership below later.
    # all_examples itself does not need a split field.


processed_train = load_jsonl(
    Path(
        "data/processed/mathdial/train.jsonl"
    )
)

processed_val = load_jsonl(
    Path(
        "data/processed/mathdial/validation.jsonl"
    )
)

processed_test = load_jsonl(
    Path(
        "data/processed/mathdial/test.jsonl"
    )
)


for split_name, rows in [

    ("train", processed_train),
    ("validation", processed_val),
    ("test", processed_test),

]:

    for example in rows:

        source_to_splits[
            int(
                example["source_row_index"]
            )
        ].add(
            split_name
        )


split_conflicts = {

    source_index:
        sorted(splits)

    for source_index, splits
    in source_to_splits.items()

    if len(splits) != 1
}


print(
    "\nSource-row split conflicts:",
    len(split_conflicts)
)


assert len(split_conflicts) == 0


# ============================================================
# PARSE CONVERSATION
# ============================================================

def parse_segment(segment):

    segment = segment.strip()

    if not segment:
        return None


    if ":" not in segment:
        return {
            "speaker": "unknown",
            "text": segment,
        }


    speaker, text = segment.split(
        ":",
        1,
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

        normalized_speaker = (
            "teacher"
        )

    elif speaker.startswith(
        "student"
    ):

        normalized_speaker = (
            "student"
        )

    else:

        normalized_speaker = (
            speaker
        )


    return {

        "speaker":
            normalized_speaker,

        "text":
            text,
    }


def parse_conversation(text):

    segments = []


    for raw_segment in text.split(
        "|EOM|"
    ):

        parsed = parse_segment(
            raw_segment
        )

        if parsed is not None:

            segments.append(
                parsed
            )


    return segments


# ============================================================
# EXTRACT TEACHER MOVE
# ============================================================

def extract_teacher_move(text):

    match = MOVE_PATTERN.match(
        text
    )


    if match is None:

        return None


    return (
        match.group(1)
        .lower()
    )


# ============================================================
# AUDIT
# ============================================================

global_stats = Counter()

split_stats = defaultdict(
    Counter
)

move_stats = {

    label:
        Counter()

    for label in LABELS
}


student_reply_lengths = []

student_reply_word_lengths = []


examples_by_split = Counter()


transition_records = []


for source_index, raw_row in enumerate(
    combined_raw
):

    splits = source_to_splits.get(
        source_index,
        set(),
    )


    if len(splits) == 0:

        # A dialogue may contain no labeled
        # teacher move and therefore no
        # supervised examples.
        custom_split = None

    else:

        custom_split = next(
            iter(splits)
        )


    conversation = parse_conversation(
        raw_row["conversation"]
    )


    for turn_index, turn in enumerate(
        conversation
    ):

        if turn["speaker"] != "teacher":
            continue


        global_stats[
            "teacher_segments"
        ] += 1


        move = extract_teacher_move(
            turn["text"]
        )


        if move is None:

            global_stats[
                "unlabeled_teacher_segments"
            ] += 1

            continue


        global_stats[
            "labeled_teacher_moves"
        ] += 1


        if custom_split is not None:

            split_stats[
                custom_split
            ][
                "labeled_teacher_moves"
            ] += 1


        move_stats[
            move
        ][
            "total"
        ] += 1


        # ----------------------------------------------------
        # Find immediate next parsed segment.
        # ----------------------------------------------------

        next_turn = None


        if (
            turn_index + 1
            < len(conversation)
        ):

            next_turn = conversation[
                turn_index + 1
            ]


        has_immediate_student_reply = (

            next_turn is not None

            and

            next_turn[
                "speaker"
            ] == "student"
        )


        if has_immediate_student_reply:

            global_stats[
                "teacher_to_student_transitions"
            ] += 1


            move_stats[
                move
            ][
                "has_student_reply"
            ] += 1


            if custom_split is not None:

                split_stats[
                    custom_split
                ][
                    "teacher_to_student_transitions"
                ] += 1


            student_text = (
                next_turn["text"]
            )


            student_reply_lengths.append(
                len(
                    student_text
                )
            )


            student_reply_word_lengths.append(
                len(
                    student_text.split()
                )
            )


            # -----------------------------------------------
            # Does another teacher move follow the student?
            #
            # This gives us:
            #
            # state -> teacher move -> student response
            #      -> next teacher decision
            # -----------------------------------------------

            following_turn = None


            if (
                turn_index + 2
                < len(conversation)
            ):

                following_turn = (
                    conversation[
                        turn_index + 2
                    ]
                )


            has_following_teacher = (

                following_turn is not None

                and

                following_turn[
                    "speaker"
                ] == "teacher"
            )


            if has_following_teacher:

                next_move = (
                    extract_teacher_move(
                        following_turn[
                            "text"
                        ]
                    )
                )


                if next_move is not None:

                    global_stats[
                        "teacher_student_teacher_chains"
                    ] += 1


                    move_stats[
                        move
                    ][
                        "has_next_teacher_move"
                    ] += 1

                else:

                    next_move = None

            else:

                next_move = None


            transition_records.append({

                "source_row_index":
                    source_index,

                "qid":
                    str(
                        raw_row["qid"]
                    ),

                "custom_split":
                    custom_split,

                "teacher_move":
                    move,

                "teacher_text":
                    MOVE_PATTERN.sub(
                        "",
                        turn["text"],
                        count=1,
                    ),

                "student_reply":
                    student_text,

                "student_reply_words":
                    len(
                        student_text.split()
                    ),

                "next_teacher_move":
                    next_move,
            })


        else:

            global_stats[
                "teacher_without_immediate_student_reply"
            ] += 1


            move_stats[
                move
            ][
                "no_student_reply"
            ] += 1


            if custom_split is not None:

                split_stats[
                    custom_split
                ][
                    "teacher_without_student_reply"
                ] += 1


# ============================================================
# COUNTS MUST MATCH SUPERVISED DATA
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "GLOBAL TRANSITION COVERAGE"
)

print(
    "=" * 70
)


for key in [

    "teacher_segments",
    "labeled_teacher_moves",
    "unlabeled_teacher_segments",
    "teacher_to_student_transitions",
    "teacher_without_immediate_student_reply",
    "teacher_student_teacher_chains",

]:

    print(
        f"\n{key}: "
        f"{global_stats[key]}"
    )


assert (
    global_stats[
        "labeled_teacher_moves"
    ]
    == 18607
)


# ============================================================
# COVERAGE RATE
# ============================================================

transition_rate = (

    global_stats[
        "teacher_to_student_transitions"
    ]

    /

    global_stats[
        "labeled_teacher_moves"
    ]
)


chain_rate = (

    global_stats[
        "teacher_student_teacher_chains"
    ]

    /

    global_stats[
        "labeled_teacher_moves"
    ]
)


print(
    f"\nTeacher -> student coverage: "
    f"{transition_rate * 100:.2f}%"
)

print(
    f"Teacher -> student -> teacher coverage: "
    f"{chain_rate * 100:.2f}%"
)


# ============================================================
# SPLIT COVERAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CUSTOM SPLIT TRANSITION COVERAGE"
)

print(
    "=" * 70
)


expected_split_examples = {

    "train":
        14905,

    "validation":
        1850,

    "test":
        1852,
}


for split_name in [
    "train",
    "validation",
    "test",
]:

    stats = split_stats[
        split_name
    ]


    labeled = stats[
        "labeled_teacher_moves"
    ]

    replies = stats[
        "teacher_to_student_transitions"
    ]

    no_reply = stats[
        "teacher_without_student_reply"
    ]


    print(
        f"\n{split_name.upper()}"
    )

    print(
        "  Labeled teacher moves:",
        labeled
    )

    print(
        "  Student replies:",
        replies
    )

    print(
        "  No immediate student reply:",
        no_reply
    )


    if labeled:

        print(
            "  Reply coverage:",
            f"{(
                replies
                / labeled
                * 100
            ):.2f}%"
        )


    assert (
        labeled
        ==
        expected_split_examples[
            split_name
        ]
    )


# ============================================================
# MOVE-SPECIFIC COVERAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRANSITION COVERAGE BY MOVE"
)

print(
    "=" * 70
)


move_summary = {}


for move in LABELS:

    stats = move_stats[
        move
    ]


    total = stats[
        "total"
    ]

    replies = stats[
        "has_student_reply"
    ]

    next_teacher = stats[
        "has_next_teacher_move"
    ]


    reply_rate = (

        replies
        / total

        if total

        else 0.0
    )


    chain_move_rate = (

        next_teacher
        / total

        if total

        else 0.0
    )


    move_summary[
        move
    ] = {

        "total":
            total,

        "student_reply":
            replies,

        "student_reply_rate":
            reply_rate,

        "next_teacher_move":
            next_teacher,

        "next_teacher_move_rate":
            chain_move_rate,
    }


    print(
        f"\n{move}"
    )

    print(
        "  Total:",
        total
    )

    print(
        "  Student reply:",
        replies
    )

    print(
        "  Reply coverage:",
        f"{reply_rate * 100:.2f}%"
    )

    print(
        "  Student + next teacher:",
        next_teacher
    )

    print(
        "  Chain coverage:",
        f"{chain_move_rate * 100:.2f}%"
    )


# ============================================================
# STUDENT RESPONSE LENGTH
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STUDENT REPLY LENGTH"
)

print(
    "=" * 70
)


if student_reply_word_lengths:

    sorted_words = sorted(
        student_reply_word_lengths
    )


    def percentile(values, p):

        index = int(
            round(
                (
                    len(values)
                    - 1
                )
                * p
            )
        )

        return values[
            index
        ]


    reply_length_summary = {

        "count":
            len(
                sorted_words
            ),

        "minimum_words":
            min(
                sorted_words
            ),

        "median_words":
            percentile(
                sorted_words,
                0.50,
            ),

        "p90_words":
            percentile(
                sorted_words,
                0.90,
            ),

        "p95_words":
            percentile(
                sorted_words,
                0.95,
            ),

        "maximum_words":
            max(
                sorted_words
            ),
    }


    for key, value in (
        reply_length_summary.items()
    ):

        print(
            f"\n{key}: "
            f"{value}"
        )

else:

    reply_length_summary = {}


# ============================================================
# TRAIN-ONLY SELF-IMPROVEMENT POOL
#
# IMPORTANT:
# Validation remains for model selection.
# Test remains untouched.
#
# We count only transitions belonging to our frozen TRAIN split.
# ============================================================

train_transition_records = [

    record

    for record in transition_records

    if record[
        "custom_split"
    ] == "train"
]


train_transition_move_counts = Counter(

    record[
        "teacher_move"
    ]

    for record
    in train_transition_records
)


print(
    "\n"
    + "=" * 70
)

print(
    "TRAIN-ONLY SELF-IMPROVEMENT POOL"
)

print(
    "=" * 70
)


print(
    "\nUsable teacher -> student transitions:",
    len(
        train_transition_records
    )
)


for label in LABELS:

    print(
        f"{label:10s}: "
        f"{train_transition_move_counts[label]}"
    )


# ============================================================
# SAVE SUMMARY
#
# We deliberately do NOT write the transition text dataset yet.
# First we inspect these statistics and decide what constitutes
# a valid feedback signal.
# ============================================================

result = {

    "global": {

        key:
            int(value)

        for key, value
        in global_stats.items()
    },

    "teacher_to_student_coverage":
        float(
            transition_rate
        ),

    "teacher_student_teacher_coverage":
        float(
            chain_rate
        ),

    "split_stats": {

        split_name:
            {

                key:
                    int(value)

                for key, value
                in stats.items()
            }

        for split_name, stats
        in split_stats.items()
    },

    "move_summary":
        move_summary,

    "student_reply_length":
        reply_length_summary,

    "train_self_improvement_pool": {

        "usable_transitions":
            len(
                train_transition_records
            ),

        "move_counts":
            dict(
                train_transition_move_counts
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
    "SI0 TRANSITION AUDIT COMPLETE"
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
