import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

MATHDIAL_TRAIN_PATH = Path(
    "data/raw/mathdial/train.jsonl"
)

MATHDIAL_TEST_PATH = Path(
    "data/raw/mathdial/test.jsonl"
)

HARD_TRAIN_PATH = Path(
    "data/processed/train.jsonl"
)

HARD_VAL_PATH = Path(
    "data/processed/validation.jsonl"
)

HARD_TEST_PATH = Path(
    "data/processed/test.jsonl"
)


VALID_LABELS = {
    "generic",
    "probing",
    "focus",
    "telling",
}


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}, line {line_number}: {e}"
                )

    return rows


# ============================================================
# NORMALIZE PROBLEM TEXT
# ============================================================

def normalize_problem(text):

    return " ".join(
        text.lower().split()
    )


# ============================================================
# PARSE MATHDIAL CONVERSATION
# ============================================================

TEACHER_PATTERN = re.compile(
    r"^Teacher:\s*"
    r"\((generic|probing|focus|telling)\)"
    r"\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


UNLABELED_TEACHER_PATTERN = re.compile(
    r"^Teacher:\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def inspect_conversation(
    conversation
):

    segments = [
        segment.strip()
        for segment in conversation.split(
            "|EOM|"
        )
        if segment.strip()
    ]

    label_counts = Counter()

    labeled_teacher_turns = 0
    unlabeled_teacher_turns = 0
    non_teacher_turns = 0
    malformed_segments = 0

    for segment in segments:

        teacher_match = (
            TEACHER_PATTERN.match(
                segment
            )
        )

        if teacher_match:

            label = (
                teacher_match
                .group(1)
                .lower()
            )

            label_counts[label] += 1

            labeled_teacher_turns += 1

            continue

        unlabeled_teacher_match = (
            UNLABELED_TEACHER_PATTERN
            .match(segment)
        )

        if unlabeled_teacher_match:

            unlabeled_teacher_turns += 1

            continue

        # Student turns generally use the
        # student's name rather than "Student".
        if ":" in segment:

            non_teacher_turns += 1

        else:

            malformed_segments += 1

    return {
        "labels":
            label_counts,

        "labeled_teacher_turns":
            labeled_teacher_turns,

        "unlabeled_teacher_turns":
            unlabeled_teacher_turns,

        "non_teacher_turns":
            non_teacher_turns,

        "malformed_segments":
            malformed_segments,
    }


# ============================================================
# INSPECT ONE MATHDIAL SPLIT
# ============================================================

def inspect_mathdial_split(
    name,
    rows
):

    global_labels = Counter()

    total_labeled_teacher_turns = 0
    total_unlabeled_teacher_turns = 0
    total_non_teacher_turns = 0
    malformed_segments = 0

    rows_without_labeled_teacher_move = 0

    qids = set()
    questions = set()

    scenarios = Counter()

    for row in rows:

        qids.add(
            row["qid"]
        )

        questions.add(
            normalize_problem(
                row["question"]
            )
        )

        scenarios[
            row["scenario"]
        ] += 1

        result = inspect_conversation(
            row["conversation"]
        )

        global_labels.update(
            result["labels"]
        )

        total_labeled_teacher_turns += (
            result[
                "labeled_teacher_turns"
            ]
        )

        total_unlabeled_teacher_turns += (
            result[
                "unlabeled_teacher_turns"
            ]
        )

        total_non_teacher_turns += (
            result[
                "non_teacher_turns"
            ]
        )

        malformed_segments += (
            result[
                "malformed_segments"
            ]
        )

        if (
            result[
                "labeled_teacher_turns"
            ]
            == 0
        ):

            rows_without_labeled_teacher_move += 1

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MATHDIAL {name.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRows/dialogues: "
        f"{len(rows)}"
    )

    print(
        f"Unique qids: "
        f"{len(qids)}"
    )

    print(
        f"Unique normalized questions: "
        f"{len(questions)}"
    )

    print(
        f"\nLabeled teacher turns: "
        f"{total_labeled_teacher_turns}"
    )

    print(
        f"Unlabeled Teacher segments: "
        f"{total_unlabeled_teacher_turns}"
    )

    print(
        f"Non-teacher segments: "
        f"{total_non_teacher_turns}"
    )

    print(
        f"Malformed segments: "
        f"{malformed_segments}"
    )

    print(
        f"Rows with no labeled teacher move: "
        f"{rows_without_labeled_teacher_move}"
    )

    print(
        "\nPedagogical move distribution:"
    )

    for label in [
        "generic",
        "probing",
        "focus",
        "telling",
    ]:

        count = (
            global_labels[label]
        )

        if total_labeled_teacher_turns:

            percentage = (
                count
                / total_labeled_teacher_turns
                * 100
            )

        else:

            percentage = 0

        print(
            f"{label:10s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

    print(
        "\nScenario distribution:"
    )

    for scenario, count in (
        sorted(
            scenarios.items()
        )
    ):

        print(
            f"Scenario {scenario}: "
            f"{count}"
        )

    return {
        "qids":
            qids,

        "questions":
            questions,

        "labels":
            global_labels,

        "labeled_teacher_turns":
            total_labeled_teacher_turns,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MATHDIAL DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load MathDial
    # --------------------------------------------------------

    mathdial_train = load_jsonl(
        MATHDIAL_TRAIN_PATH
    )

    mathdial_test = load_jsonl(
        MATHDIAL_TEST_PATH
    )

    # --------------------------------------------------------
    # Load our existing hard splits
    # --------------------------------------------------------

    hard_train = load_jsonl(
        HARD_TRAIN_PATH
    )

    hard_val = load_jsonl(
        HARD_VAL_PATH
    )

    hard_test = load_jsonl(
        HARD_TEST_PATH
    )

    # --------------------------------------------------------
    # Show first row schema
    # --------------------------------------------------------

    print(
        "\nMathDial first-row keys:"
    )

    print(
        mathdial_train[0].keys()
    )

    # --------------------------------------------------------
    # Inspect MathDial train/test
    # --------------------------------------------------------

    train_info = (
        inspect_mathdial_split(
            "train",
            mathdial_train
        )
    )

    test_info = (
        inspect_mathdial_split(
            "test",
            mathdial_test
        )
    )

    # ========================================================
    # OUR CURRENT HARD SPLIT PROBLEMS
    # ========================================================

    hard_train_problems = {
        normalize_problem(
            row["problem"]
        )
        for row in hard_train
    }

    hard_val_problems = {
        normalize_problem(
            row["problem"]
        )
        for row in hard_val
    }

    hard_test_problems = {
        normalize_problem(
            row["problem"]
        )
        for row in hard_test
    }

    mathdial_train_problems = (
        train_info[
            "questions"
        ]
    )

    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MATHDIAL TRAIN / CURRENT HARD SPLIT OVERLAP"
    )

    print(
        "=" * 70
    )

    overlap_train = (
        mathdial_train_problems
        & hard_train_problems
    )

    overlap_val = (
        mathdial_train_problems
        & hard_val_problems
    )

    overlap_test = (
        mathdial_train_problems
        & hard_test_problems
    )

    print(
        f"\nMathDial train problems "
        f"overlapping hard TRAIN: "
        f"{len(overlap_train)}"
    )

    print(
        f"MathDial train problems "
        f"overlapping hard VALIDATION: "
        f"{len(overlap_val)}"
    )

    print(
        f"MathDial train problems "
        f"overlapping hard TEST: "
        f"{len(overlap_test)}"
    )

    # ========================================================
    # CLEAN NOVEL PROBLEMS
    # ========================================================

    all_hard_problems = (
        hard_train_problems
        | hard_val_problems
        | hard_test_problems
    )

    novel_mathdial_train_problems = (
        mathdial_train_problems
        - all_hard_problems
    )

    novel_rows = [

        row

        for row in mathdial_train

        if normalize_problem(
            row["question"]
        )
        in novel_mathdial_train_problems
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLEAN MATHDIAL TRAIN CANDIDATES"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMathDial train unique problems "
        f"not present anywhere in "
        f"MathDialBridge-hard: "
        f"{len(novel_mathdial_train_problems)}"
    )

    print(
        f"MathDial train dialogues "
        f"using those novel problems: "
        f"{len(novel_rows)}"
    )

    # --------------------------------------------------------
    # Count labels in clean candidate rows
    # --------------------------------------------------------

    novel_labels = Counter()

    novel_teacher_turns = 0

    for row in novel_rows:

        result = inspect_conversation(
            row["conversation"]
        )

        novel_labels.update(
            result["labels"]
        )

        novel_teacher_turns += (
            result[
                "labeled_teacher_turns"
            ]
        )

    print(
        f"\nLabeled teacher moves "
        f"available in clean candidates: "
        f"{novel_teacher_turns}"
    )

    print(
        "\nClean candidate label distribution:"
    )

    for label in [
        "generic",
        "probing",
        "focus",
        "telling",
    ]:

        count = (
            novel_labels[label]
        )

        percentage = (

            count
            / novel_teacher_turns
            * 100

            if novel_teacher_turns

            else 0
        )

        print(
            f"{label:10s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 70
    )

    assert (
        len(
            novel_mathdial_train_problems
            & hard_val_problems
        )
        == 0
    )

    assert (
        len(
            novel_mathdial_train_problems
            & hard_test_problems
        )
        == 0
    )

    assert (
        set(
            train_info[
                "labels"
            ].keys()
        )
        <= VALID_LABELS
    )

    print(
        "\nMathDial label taxonomy check: PASSED"
    )

    print(
        "Hard validation overlap "
        "filter check: PASSED"
    )

    print(
        "Hard test overlap "
        "filter check: PASSED"
    )

    print(
        "\nMATHDIAL INSPECTION COMPLETE"
    )


if __name__ == "__main__":
    main()
