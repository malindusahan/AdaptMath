import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIG
# ============================================================

SEED = 42

RAW_TRAIN_PATH = Path(
    "data/raw/mathdial/train.jsonl"
)

RAW_TEST_PATH = Path(
    "data/raw/mathdial/test.jsonl"
)

OUTPUT_DIR = Path(
    "data/processed/mathdial"
)

RESULT_DIR = Path(
    "results/preprocessing"
)


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


LABEL_SET = set(LABELS)


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
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

    return rows


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def normalize_question(text):

    return normalize_text(text)


# ============================================================
# PARSE MATHDIAL CONVERSATIONS
# ============================================================

LABELED_TEACHER_PATTERN = re.compile(
    r"^Teacher:\s*"
    r"\((generic|probing|focus|telling)\)"
    r"\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


UNLABELED_TEACHER_PATTERN = re.compile(
    r"^Teacher:\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_conversation(conversation):

    turns = []

    raw_segments = [
        segment.strip()
        for segment
        in conversation.split("|EOM|")
        if segment.strip()
    ]


    for segment in raw_segments:

        # ----------------------------------------------------
        # Labeled Teacher turn
        # ----------------------------------------------------

        match = LABELED_TEACHER_PATTERN.match(
            segment
        )

        if match:

            move = (
                match.group(1)
                .lower()
            )

            text = (
                match.group(2)
                .strip()
            )

            turns.append({
                "user": "Teacher",
                "text": text,
                "dialog_act": move,
            })

            continue


        # ----------------------------------------------------
        # Unlabeled Teacher turn
        # ----------------------------------------------------

        match = UNLABELED_TEACHER_PATTERN.match(
            segment
        )

        if match:

            text = (
                match.group(1)
                .strip()
            )

            turns.append({
                "user": "Teacher",
                "text": text,
                "dialog_act": None,
            })

            continue


        # ----------------------------------------------------
        # Student turn
        #
        # MathDial normally uses a student name:
        # "James: ..."
        # ----------------------------------------------------

        if ":" in segment:

            _, text = segment.split(
                ":",
                1
            )

            turns.append({
                "user": "Student",
                "text": text.strip(),
                "dialog_act": None,
            })

            continue


        raise ValueError(
            "Could not parse conversation segment:\n"
            + segment
        )


    return turns


# ============================================================
# UNION-FIND
#
# We group dialogues together whenever they share:
#   - qid
#   OR
#   - normalized question
#
# This guarantees BOTH qid-level and question-level
# separation between train / validation / test.
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

        self.rank = [
            0
        ] * n


    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x


    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return


        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            root_a, root_b = (
                root_b,
                root_a
            )


        self.parent[root_b] = (
            root_a
        )


        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):

            self.rank[root_a] += 1


def build_leakage_groups(rows):

    uf = UnionFind(
        len(rows)
    )

    first_qid = {}
    first_question = {}


    for index, row in enumerate(rows):

        qid = str(
            row["qid"]
        )

        question = normalize_question(
            row["question"]
        )


        # ----------------------------------------------------
        # Same qid
        # ----------------------------------------------------

        if qid in first_qid:

            uf.union(
                index,
                first_qid[qid]
            )

        else:

            first_qid[qid] = index


        # ----------------------------------------------------
        # Same normalized question
        # ----------------------------------------------------

        if question in first_question:

            uf.union(
                index,
                first_question[
                    question
                ]
            )

        else:

            first_question[
                question
            ] = index


    roots = [
        uf.find(index)
        for index in range(
            len(rows)
        )
    ]


    # Convert arbitrary DSU roots into
    # compact deterministic group IDs.

    root_to_group = {}

    group_ids = []


    for root in roots:

        if root not in root_to_group:

            root_to_group[root] = (
                len(root_to_group)
            )

        group_ids.append(
            root_to_group[root]
        )


    return group_ids


# ============================================================
# BUILD TURN-LEVEL SUPERVISED EXAMPLES
# ============================================================

def build_examples(
    rows,
    group_ids
):

    examples = []

    unlabeled_teacher_turns = 0

    total_teacher_turns = 0
    total_student_turns = 0


    for row_index, (
        row,
        group_id
    ) in enumerate(
        zip(
            rows,
            group_ids
        )
    ):

        turns = parse_conversation(
            row["conversation"]
        )

        history = []


        for turn_index, turn in enumerate(
            turns
        ):

            speaker = turn["user"]
            text = turn["text"]
            move = turn["dialog_act"]


            if speaker == "Teacher":

                total_teacher_turns += 1


                # =================================================
                # CREATE TARGET EXAMPLE BEFORE ADDING
                # CURRENT TEACHER RESPONSE TO HISTORY
                # =================================================

                if move is not None:

                    assert (
                        move in LABEL_SET
                    )


                    examples.append({

                        "example_id":
                            (
                                f"mathdial_"
                                f"{row_index:04d}_"
                                f"turn_"
                                f"{turn_index:03d}"
                            ),

                        "dialogue_id":
                            (
                                f"mathdial_"
                                f"{row_index:04d}"
                            ),

                        "source_row_index":
                            row_index,

                        "original_split":
                            row[
                                "_original_split"
                            ],

                        "qid":
                            str(
                                row["qid"]
                            ),

                        "group_id":
                            int(
                                group_id
                            ),

                        "scenario":
                            row.get(
                                "scenario"
                            ),

                        "problem":
                            row["question"],

                        "history":
                            [
                                {
                                    "user":
                                        old_turn[
                                            "user"
                                        ],

                                    "text":
                                        old_turn[
                                            "text"
                                        ],
                                }

                                for old_turn
                                in history
                            ],

                        "history_num_turns":
                            len(history),

                        "target_move":
                            move,
                    })


                else:

                    unlabeled_teacher_turns += 1


            elif speaker == "Student":

                total_student_turns += 1


            # =================================================
            # Store only speaker + clean text.
            #
            # Historical dialog_act labels are NEVER placed
            # in observable history.
            # =================================================

            history.append({
                "user": speaker,
                "text": text,
            })


    stats = {
        "teacher_turns":
            total_teacher_turns,

        "student_turns":
            total_student_turns,

        "unlabeled_teacher_turns":
            unlabeled_teacher_turns,
    }


    return (
        examples,
        stats
    )


# ============================================================
# SPLIT QUALITY
# ============================================================

def label_distribution(examples):

    counts = Counter(
        example["target_move"]
        for example in examples
    )

    total = len(examples)

    return {
        label:
            (
                counts[label]
                / total
                if total
                else 0.0
            )

        for label in LABELS
    }


def split_score(
    train_examples,
    val_examples,
    test_examples,
    global_distribution
):

    total = (
        len(train_examples)
        + len(val_examples)
        + len(test_examples)
    )


    target_sizes = {
        "train": 0.80,
        "validation": 0.10,
        "test": 0.10,
    }


    split_examples = {
        "train":
            train_examples,

        "validation":
            val_examples,

        "test":
            test_examples,
    }


    score = 0.0


    # --------------------------------------------------------
    # Example-count balance
    # --------------------------------------------------------

    for split_name, subset in (
        split_examples.items()
    ):

        actual_ratio = (
            len(subset)
            / total
        )

        score += (
            abs(
                actual_ratio
                - target_sizes[
                    split_name
                ]
            )
            * 2.0
        )


    # --------------------------------------------------------
    # Label-distribution balance
    # --------------------------------------------------------

    for subset in (
        train_examples,
        val_examples,
        test_examples
    ):

        distribution = (
            label_distribution(
                subset
            )
        )


        for label in LABELS:

            score += abs(
                distribution[label]
                - global_distribution[
                    label
                ]
            )


    return score


# ============================================================
# FIND GOOD GROUPED SPLIT
#
# We try multiple deterministic group splits and keep the one
# whose size + label distributions best match the full corpus.
# ============================================================

def create_grouped_split(
    examples,
    candidate_count=300
):

    groups = [
        example["group_id"]
        for example in examples
    ]


    global_distribution = (
        label_distribution(
            examples
        )
    )


    best = None


    dummy_x = list(
        range(
            len(examples)
        )
    )


    for candidate_offset in range(
        candidate_count
    ):

        random_state = (
            SEED
            + candidate_offset
        )


        # ----------------------------------------------------
        # 80% train / 20% temporary
        # ----------------------------------------------------

        first_splitter = (
            GroupShuffleSplit(

                n_splits=1,

                test_size=0.20,

                random_state=
                    random_state,
            )
        )


        train_indices, temp_indices = (
            next(
                first_splitter.split(
                    dummy_x,
                    groups=groups
                )
            )
        )


        temp_examples = [
            examples[index]
            for index
            in temp_indices
        ]


        temp_groups = [
            example["group_id"]
            for example
            in temp_examples
        ]


        temp_dummy = list(
            range(
                len(temp_examples)
            )
        )


        # ----------------------------------------------------
        # Split temporary data 50/50
        # -> approximately 10% validation / 10% test
        # ----------------------------------------------------

        second_splitter = (
            GroupShuffleSplit(

                n_splits=1,

                test_size=0.50,

                random_state=
                    random_state
                    + 10000,
            )
        )


        val_local_indices, test_local_indices = (
            next(
                second_splitter.split(
                    temp_dummy,
                    groups=temp_groups
                )
            )
        )


        train_examples = [
            examples[index]
            for index
            in train_indices
        ]


        val_examples = [
            temp_examples[index]
            for index
            in val_local_indices
        ]


        test_examples = [
            temp_examples[index]
            for index
            in test_local_indices
        ]


        score = split_score(
            train_examples,
            val_examples,
            test_examples,
            global_distribution,
        )


        if (
            best is None
            or score < best["score"]
        ):

            best = {
                "score":
                    score,

                "random_state":
                    random_state,

                "train":
                    train_examples,

                "validation":
                    val_examples,

                "test":
                    test_examples,
            }


    return best


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    path,
    rows
):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# SPLIT SUMMARY
# ============================================================

def print_split_summary(
    name,
    examples
):

    dialogues = {
        example["dialogue_id"]
        for example in examples
    }

    qids = {
        example["qid"]
        for example in examples
    }

    groups = {
        example["group_id"]
        for example in examples
    }

    questions = {
        normalize_question(
            example["problem"]
        )
        for example in examples
    }

    counts = Counter(
        example["target_move"]
        for example in examples
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        name.upper()
    )

    print(
        "-" * 70
    )


    print(
        f"Examples: "
        f"{len(examples)}"
    )

    print(
        f"Dialogues: "
        f"{len(dialogues)}"
    )

    print(
        f"Groups: "
        f"{len(groups)}"
    )

    print(
        f"Unique qids: "
        f"{len(qids)}"
    )

    print(
        f"Unique questions: "
        f"{len(questions)}"
    )


    print(
        "\nLabel distribution:"
    )


    for label in LABELS:

        count = counts[label]

        percentage = (
            count
            / len(examples)
            * 100
        )


        print(
            f"{label:10s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)


    print("=" * 70)
    print("MATHDIAL-ONLY PREPROCESSING")
    print("=" * 70)


    # ========================================================
    # LOAD AND COMBINE OFFICIAL SPLITS
    # ========================================================

    raw_train = load_jsonl(
        RAW_TRAIN_PATH
    )

    raw_test = load_jsonl(
        RAW_TEST_PATH
    )


    for row in raw_train:

        row["_original_split"] = (
            "official_train"
        )


    for row in raw_test:

        row["_original_split"] = (
            "official_test"
        )


    rows = (
        raw_train
        + raw_test
    )


    print(
        f"\nOfficial train dialogues: "
        f"{len(raw_train)}"
    )

    print(
        f"Official test dialogues: "
        f"{len(raw_test)}"
    )

    print(
        f"Combined dialogues: "
        f"{len(rows)}"
    )


    # ========================================================
    # BUILD LEAKAGE GROUPS
    # ========================================================

    group_ids = (
        build_leakage_groups(
            rows
        )
    )


    print(
        f"Leakage-safe groups: "
        f"{len(set(group_ids))}"
    )


    # ========================================================
    # BUILD TURN EXAMPLES
    # ========================================================

    examples, turn_stats = (
        build_examples(
            rows,
            group_ids
        )
    )


    print(
        f"\nTotal supervised examples: "
        f"{len(examples)}"
    )


    print(
        f"Teacher turns: "
        f"{turn_stats['teacher_turns']}"
    )

    print(
        f"Student turns: "
        f"{turn_stats['student_turns']}"
    )

    print(
        f"Unlabeled teacher turns: "
        f"{turn_stats['unlabeled_teacher_turns']}"
    )


    # ========================================================
    # GLOBAL LABEL DISTRIBUTION
    # ========================================================

    global_counts = Counter(
        example["target_move"]
        for example in examples
    )


    print(
        "\nGlobal move distribution:"
    )


    for label in LABELS:

        count = global_counts[label]

        percentage = (
            count
            / len(examples)
            * 100
        )

        print(
            f"{label:10s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # CREATE GROUPED SPLITS
    # ========================================================

    print(
        "\nSearching for balanced "
        "qid/question-safe split..."
    )


    split = create_grouped_split(
        examples,
        candidate_count=300
    )


    train_examples = split["train"]
    val_examples = split[
        "validation"
    ]
    test_examples = split["test"]


    print(
        f"\nSelected random state: "
        f"{split['random_state']}"
    )

    print(
        f"Split balance score: "
        f"{split['score']:.6f}"
    )


    print_split_summary(
        "train",
        train_examples
    )

    print_split_summary(
        "validation",
        val_examples
    )

    print_split_summary(
        "test",
        test_examples
    )


    # ========================================================
    # LEAKAGE CHECKS
    # ========================================================

    def qids(subset):

        return {
            example["qid"]
            for example in subset
        }


    def questions(subset):

        return {
            normalize_question(
                example["problem"]
            )
            for example in subset
        }


    def groups(subset):

        return {
            example["group_id"]
            for example in subset
        }


    train_qids = qids(
        train_examples
    )

    val_qids = qids(
        val_examples
    )

    test_qids = qids(
        test_examples
    )


    train_questions = questions(
        train_examples
    )

    val_questions = questions(
        val_examples
    )

    test_questions = questions(
        test_examples
    )


    train_groups = groups(
        train_examples
    )

    val_groups = groups(
        val_examples
    )

    test_groups = groups(
        test_examples
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "LEAKAGE VALIDATION"
    )

    print(
        "=" * 70
    )


    checks = {
        "Train / Validation qid overlap":
            len(
                train_qids
                & val_qids
            ),

        "Train / Test qid overlap":
            len(
                train_qids
                & test_qids
            ),

        "Validation / Test qid overlap":
            len(
                val_qids
                & test_qids
            ),

        "Train / Validation question overlap":
            len(
                train_questions
                & val_questions
            ),

        "Train / Test question overlap":
            len(
                train_questions
                & test_questions
            ),

        "Validation / Test question overlap":
            len(
                val_questions
                & test_questions
            ),

        "Train / Validation group overlap":
            len(
                train_groups
                & val_groups
            ),

        "Train / Test group overlap":
            len(
                train_groups
                & test_groups
            ),

        "Validation / Test group overlap":
            len(
                val_groups
                & test_groups
            ),
    }


    for name, value in checks.items():

        print(
            f"{name}: {value}"
        )


    assert all(
        value == 0
        for value in checks.values()
    )


    # ========================================================
    # INPUT / TARGET LEAKAGE CHECKS
    # ========================================================

    invalid_labels = 0
    history_act_leakage = 0
    history_length_errors = 0


    for example in examples:

        if (
            example["target_move"]
            not in LABEL_SET
        ):

            invalid_labels += 1


        if (
            len(example["history"])
            != example[
                "history_num_turns"
            ]
        ):

            history_length_errors += 1


        for history_turn in (
            example["history"]
        ):

            if (
                "dialog_act"
                in history_turn
            ):

                history_act_leakage += 1


    print(
        "\nInvalid target labels:",
        invalid_labels
    )

    print(
        "Historical dialog_act leakage:",
        history_act_leakage
    )

    print(
        "History length errors:",
        history_length_errors
    )


    assert invalid_labels == 0
    assert history_act_leakage == 0
    assert history_length_errors == 0


    print(
        "\nALL LEAKAGE CHECKS: PASSED"
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    save_jsonl(
        OUTPUT_DIR
        / "all_examples.jsonl",
        examples
    )

    save_jsonl(
        OUTPUT_DIR
        / "train.jsonl",
        train_examples
    )

    save_jsonl(
        OUTPUT_DIR
        / "validation.jsonl",
        val_examples
    )

    save_jsonl(
        OUTPUT_DIR
        / "test.jsonl",
        test_examples
    )


    summary = {
        "seed":
            SEED,

        "selected_random_state":
            split[
                "random_state"
            ],

        "combined_dialogues":
            len(rows),

        "groups":
            len(
                set(group_ids)
            ),

        "total_examples":
            len(examples),

        "teacher_turns":
            turn_stats[
                "teacher_turns"
            ],

        "student_turns":
            turn_stats[
                "student_turns"
            ],

        "unlabeled_teacher_turns":
            turn_stats[
                "unlabeled_teacher_turns"
            ],

        "global_label_distribution":
            dict(
                global_counts
            ),

        "train_examples":
            len(
                train_examples
            ),

        "validation_examples":
            len(
                val_examples
            ),

        "test_examples":
            len(
                test_examples
            ),

        "train_label_distribution":
            dict(
                Counter(
                    example[
                        "target_move"
                    ]
                    for example
                    in train_examples
                )
            ),

        "validation_label_distribution":
            dict(
                Counter(
                    example[
                        "target_move"
                    ]
                    for example
                    in val_examples
                )
            ),

        "test_label_distribution":
            dict(
                Counter(
                    example[
                        "target_move"
                    ]
                    for example
                    in test_examples
                )
            ),

        "leakage_checks":
            checks,
    }


    summary_path = (
        RESULT_DIR
        / "mathdial_preprocessing_summary.json"
    )


    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "OUTPUT"
    )

    print(
        "=" * 70
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "all_examples.jsonl"
    )

    print(
        OUTPUT_DIR
        / "train.jsonl"
    )

    print(
        OUTPUT_DIR
        / "validation.jsonl"
    )

    print(
        OUTPUT_DIR
        / "test.jsonl"
    )

    print(
        summary_path
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "MATHDIAL PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
