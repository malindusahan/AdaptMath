import json
import random
from pathlib import Path
from collections import defaultdict


TRAIN_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_train.jsonl"
)

VAL_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_validation.jsonl"
)

TRAIN_OUT = Path(
    "data/processed/self_improvement/"
    "response_contrastive_train.jsonl"
)

VAL_OUT = Path(
    "data/processed/self_improvement/"
    "response_contrastive_validation.jsonl"
)

RESULT_OUT = Path(
    "results/self_improvement/"
    "si3_contrastive_dataset_summary.json"
)


NEGATIVES_PER_POSITIVE = 3

STRICT_WINDOW = (
    0.70,
    1.30,
)

FALLBACK_WINDOW = (
    0.50,
    1.50,
)


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


def save_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def normalize_text(text):

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def word_count(text):

    return len(
        str(text).split()
    )


train_rows = load_jsonl(
    TRAIN_PATH
)

val_rows = load_jsonl(
    VAL_PATH
)


assert len(train_rows) == 9989
assert len(val_rows) == 2509


print("=" * 70)
print("SI3-B â€” CONTRASTIVE DATASET CONSTRUCTION")
print("=" * 70)


# ============================================================
# POOLS
# ============================================================

def build_pools(rows):

    pools = defaultdict(list)

    for index, row in enumerate(rows):

        pools[
            row["teacher_move"]
        ].append({

            "index":
                index,

            "transition_id":
                row["transition_id"],

            "qid":
                str(
                    row["qid"]
                ),

            "source_row_index":
                int(
                    row["source_row_index"]
                ),

            "response":
                row["student_reply"],

            "normalized_response":
                normalize_text(
                    row["student_reply"]
                ),

            "words":
                word_count(
                    row["student_reply"]
                ),
        })

    return pools


def candidates_for_window(
    row,
    row_index,
    pools,
    low_ratio,
    high_ratio,
):

    positive = normalize_text(
        row["student_reply"]
    )

    positive_words = word_count(
        row["student_reply"]
    )

    min_words = max(
        1,
        int(
            positive_words
            * low_ratio
        ),
    )

    max_words = max(
        min_words,
        int(
            positive_words
            * high_ratio
            + 0.999999
        ),
    )

    candidates = []

    for candidate in pools[
        row["teacher_move"]
    ]:

        if (
            candidate["index"]
            == row_index
        ):
            continue

        if (
            candidate["qid"]
            == str(row["qid"])
        ):
            continue

        if (
            candidate["source_row_index"]
            == int(
                row["source_row_index"]
            )
        ):
            continue

        if (
            candidate["normalized_response"]
            == positive
        ):
            continue

        if not (
            min_words
            <= candidate["words"]
            <= max_words
        ):
            continue

        candidates.append(
            candidate
        )

    # --------------------------------------------------------
    # Keep unique negative response texts.
    # --------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = candidate[
            "normalized_response"
        ]

        if key not in unique:

            unique[key] = candidate

    return list(
        unique.values()
    )


# ============================================================
# BUILD ONE SPLIT
# ============================================================

def build_split(
    name,
    rows,
    seed,
):

    rng = random.Random(
        seed
    )

    pools = build_pools(
        rows
    )

    output = []

    strict_count = 0
    fallback_count = 0
    failed = []

    for index, row in enumerate(rows):

        candidates = candidates_for_window(
            row,
            index,
            pools,
            STRICT_WINDOW[0],
            STRICT_WINDOW[1],
        )

        sampling_rule = (
            "strict_70_130"
        )

        if (
            len(candidates)
            <
            NEGATIVES_PER_POSITIVE
        ):

            candidates = candidates_for_window(
                row,
                index,
                pools,
                FALLBACK_WINDOW[0],
                FALLBACK_WINDOW[1],
            )

            sampling_rule = (
                "fallback_50_150"
            )

        if (
            len(candidates)
            <
            NEGATIVES_PER_POSITIVE
        ):

            failed.append({

                "index":
                    index,

                "transition_id":
                    row["transition_id"],

                "qid":
                    row["qid"],

                "teacher_move":
                    row["teacher_move"],

                "available":
                    len(candidates),
            })

            continue

        selected = rng.sample(
            candidates,
            NEGATIVES_PER_POSITIVE,
        )

        if (
            sampling_rule
            == "strict_70_130"
        ):
            strict_count += 1

        else:
            fallback_count += 1

        output.append({

            "transition_id":
                row["transition_id"],

            "group_id":
                row["group_id"],

            "qid":
                row["qid"],

            "source_row_index":
                row["source_row_index"],

            "problem":
                row["problem"],

            "pre_history":
                row["pre_history"],

            "teacher_move":
                row["teacher_move"],

            "teacher_text":
                row["teacher_text"],

            "positive_response":
                row["student_reply"],

            "negative_responses": [

                candidate["response"]

                for candidate
                in selected
            ],

            "negative_metadata": [

                {
                    "transition_id":
                        candidate[
                            "transition_id"
                        ],

                    "qid":
                        candidate[
                            "qid"
                        ],

                    "source_row_index":
                        candidate[
                            "source_row_index"
                        ],

                    "word_count":
                        candidate[
                            "words"
                        ],
                }

                for candidate
                in selected
            ],

            "sampling_rule":
                sampling_rule,
        })

    print(
        f"\n{name}"
    )

    print(
        "  Source transitions:",
        len(rows),
    )

    print(
        "  Constructed contexts:",
        len(output),
    )

    print(
        "  Strict-window contexts:",
        strict_count,
    )

    print(
        "  Fallback-window contexts:",
        fallback_count,
    )

    print(
        "  Failed contexts:",
        len(failed),
    )

    return (
        output,
        strict_count,
        fallback_count,
        failed,
    )


print(
    "\n"
    + "=" * 70
)

print(
    "DATASET CONSTRUCTION"
)

print(
    "=" * 70
)


(
    contrastive_train,
    train_strict,
    train_fallback,
    train_failed,
) = build_split(
    "SI TRAIN",
    train_rows,
    seed=42,
)


(
    contrastive_val,
    val_strict,
    val_fallback,
    val_failed,
) = build_split(
    "SI VALIDATION",
    val_rows,
    seed=43,
)


# ============================================================
# INTEGRITY CHECK
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "INTEGRITY VALIDATION"
)

print(
    "=" * 70
)


errors = []


def validate_records(
    split_name,
    records,
):

    for record in records:

        positive_norm = normalize_text(
            record[
                "positive_response"
            ]
        )

        negatives = record[
            "negative_responses"
        ]

        metadata = record[
            "negative_metadata"
        ]

        if len(negatives) != 3:

            errors.append(
                f"{split_name}: "
                "wrong negative count"
            )

        normalized_negatives = [

            normalize_text(text)

            for text in negatives
        ]

        if (
            len(
                set(
                    normalized_negatives
                )
            )
            != 3
        ):

            errors.append(
                f"{split_name}: "
                "duplicate negative text"
            )

        if (
            positive_norm
            in normalized_negatives
        ):

            errors.append(
                f"{split_name}: "
                "positive used as negative"
            )

        for candidate in metadata:

            if (
                str(
                    candidate["qid"]
                )
                ==
                str(
                    record["qid"]
                )
            ):

                errors.append(
                    f"{split_name}: "
                    "same-qid negative"
                )

            if (
                int(
                    candidate[
                        "source_row_index"
                    ]
                )
                ==
                int(
                    record[
                        "source_row_index"
                    ]
                )
            ):

                errors.append(
                    f"{split_name}: "
                    "same-source negative"
                )


validate_records(
    "train",
    contrastive_train,
)

validate_records(
    "validation",
    contrastive_val,
)


train_groups = {

    str(
        row["group_id"]
    )

    for row in contrastive_train
}


val_groups = {

    str(
        row["group_id"]
    )

    for row in contrastive_val
}


train_qids = {

    str(
        row["qid"]
    )

    for row in contrastive_train
}


val_qids = {

    str(
        row["qid"]
    )

    for row in contrastive_val
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


print(
    "\nIntegrity errors:",
    len(errors),
)

print(
    "Train/validation group overlap:",
    group_overlap,
)

print(
    "Train/validation qid overlap:",
    qid_overlap,
)


passed = (
    len(errors) == 0
    and
    len(train_failed) == 0
    and
    len(val_failed) == 0
    and
    len(contrastive_train)
    == len(train_rows)
    and
    len(contrastive_val)
    == len(val_rows)
    and
    group_overlap == 0
    and
    qid_overlap == 0
)


# ============================================================
# SAMPLE OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SAMPLE CONTRASTIVE CONTEXTS"
)

print(
    "=" * 70
)


for record in contrastive_train[:3]:

    print(
        "\n"
        + "-" * 70
    )

    print(
        "QID:",
        record["qid"],
    )

    print(
        "Move:",
        record["teacher_move"],
    )

    print(
        "Sampling:",
        record["sampling_rule"],
    )

    print(
        "\nTeacher:"
    )

    print(
        record["teacher_text"]
    )

    print(
        "\nPOSITIVE:"
    )

    print(
        record["positive_response"]
    )

    for i, negative in enumerate(
        record["negative_responses"],
        start=1,
    ):

        print(
            f"\nNEGATIVE {i}:"
        )

        print(
            negative
        )


# ============================================================
# SAVE ONLY IF VALID
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CONSTRUCTION VALIDATION"
)

print(
    "=" * 70
)


if passed:

    print(
        "\nVALIDATION: PASSED"
    )

    save_jsonl(
        TRAIN_OUT,
        contrastive_train,
    )

    save_jsonl(
        VAL_OUT,
        contrastive_val,
    )

    print(
        "\nSaved:"
    )

    print(
        TRAIN_OUT
    )

    print(
        VAL_OUT
    )

else:

    print(
        "\nVALIDATION: FAILED"
    )

    print(
        "Dataset files were NOT written."
    )


summary = {

    "experiment":
        "SI3-B_contrastive_dataset_construction",

    "train": {
        "contexts":
            len(contrastive_train),

        "strict":
            train_strict,

        "fallback":
            train_fallback,

        "failed":
            len(train_failed),
    },

    "validation": {
        "contexts":
            len(contrastive_val),

        "strict":
            val_strict,

        "fallback":
            val_fallback,

        "failed":
            len(val_failed),
    },

    "integrity_errors":
        len(errors),

    "group_overlap":
        group_overlap,

    "qid_overlap":
        qid_overlap,

    "validation_passed":
        passed,
}


RESULT_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with RESULT_OUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\nSummary:"
)

print(
    RESULT_OUT
)
