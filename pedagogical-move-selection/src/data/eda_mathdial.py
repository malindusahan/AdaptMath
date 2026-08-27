import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    "data/processed/mathdial"
)

RESULT_DIR = Path(
    "results/preprocessing"
)

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "validation.jsonl"

SUMMARY_PATH = (
    RESULT_DIR
    / "mathdial_eda_summary.json"
)


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


# ============================================================
# LOAD
# ============================================================

def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


train = load_jsonl(
    TRAIN_PATH
)

validation = load_jsonl(
    VAL_PATH
)


# ============================================================
# SUMMARIZE A SPLIT
# ============================================================

def summarize_split(
    name,
    rows
):

    history_lengths = [
        row["history_num_turns"]
        for row in rows
    ]

    label_counts = Counter(
        row["target_move"]
        for row in rows
    )

    dialogue_counts = Counter(
        row["dialogue_id"]
        for row in rows
    )

    qids = {
        row["qid"]
        for row in rows
    }


    examples_per_dialogue = list(
        dialogue_counts.values()
    )


    summary = {

        "examples":
            len(rows),

        "dialogues":
            len(dialogue_counts),

        "unique_qids":
            len(qids),

        "history_length": {

            "min":
                min(history_lengths),

            "max":
                max(history_lengths),

            "mean":
                mean(history_lengths),

            "median":
                median(history_lengths),
        },

        "examples_per_dialogue": {

            "min":
                min(
                    examples_per_dialogue
                ),

            "max":
                max(
                    examples_per_dialogue
                ),

            "mean":
                mean(
                    examples_per_dialogue
                ),

            "median":
                median(
                    examples_per_dialogue
                ),
        },

        "label_counts":
            dict(label_counts),
    }


    print(
        "\n"
        + "=" * 70
    )

    print(
        name.upper()
    )

    print(
        "=" * 70
    )


    print(
        f"\nExamples: "
        f"{summary['examples']}"
    )

    print(
        f"Dialogues: "
        f"{summary['dialogues']}"
    )

    print(
        f"Unique qids: "
        f"{summary['unique_qids']}"
    )


    print(
        "\nHistory length:"
    )

    print(
        f"  min: "
        f"{summary['history_length']['min']}"
    )

    print(
        f"  max: "
        f"{summary['history_length']['max']}"
    )

    print(
        f"  mean: "
        f"{summary['history_length']['mean']:.2f}"
    )

    print(
        f"  median: "
        f"{summary['history_length']['median']}"
    )


    print(
        "\nExamples per dialogue:"
    )

    print(
        f"  min: "
        f"{summary['examples_per_dialogue']['min']}"
    )

    print(
        f"  max: "
        f"{summary['examples_per_dialogue']['max']}"
    )

    print(
        f"  mean: "
        f"{summary['examples_per_dialogue']['mean']:.2f}"
    )

    print(
        f"  median: "
        f"{summary['examples_per_dialogue']['median']}"
    )


    print(
        "\nMove distribution:"
    )


    for label in LABELS:

        count = label_counts[label]

        percentage = (
            count
            / len(rows)
            * 100
        )

        print(
            f"{label:10s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )


    return summary


# ============================================================
# RUN SUMMARIES
# ============================================================

print("=" * 70)
print("MATHDIAL TRAIN / VALIDATION EDA")
print("=" * 70)


train_summary = summarize_split(
    "train",
    train
)

validation_summary = summarize_split(
    "validation",
    validation
)


# ============================================================
# IMBALANCE RATIO
# ============================================================

train_counts = Counter(
    row["target_move"]
    for row in train
)


largest_class = max(
    train_counts.values()
)

smallest_class = min(
    train_counts.values()
)


imbalance_ratio = (
    largest_class
    / smallest_class
)


print(
    "\n"
    + "=" * 70
)

print(
    "CLASS BALANCE"
)

print(
    "=" * 70
)


print(
    f"\nLargest class: "
    f"{largest_class}"
)

print(
    f"Smallest class: "
    f"{smallest_class}"
)

print(
    f"Imbalance ratio: "
    f"{imbalance_ratio:.2f}:1"
)


# ============================================================
# HISTORY LENGTH BY TARGET MOVE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAIN HISTORY LENGTH BY MOVE"
)

print(
    "=" * 70
)


history_by_label = {}


for label in LABELS:

    values = [
        row["history_num_turns"]

        for row in train

        if row["target_move"] == label
    ]


    history_by_label[label] = {

        "count":
            len(values),

        "mean":
            mean(values),

        "median":
            median(values),

        "min":
            min(values),

        "max":
            max(values),
    }


    print(
        f"\n{label}:"
    )

    print(
        f"  count: "
        f"{len(values)}"
    )

    print(
        f"  mean history turns: "
        f"{mean(values):.2f}"
    )

    print(
        f"  median: "
        f"{median(values)}"
    )

    print(
        f"  range: "
        f"{min(values)} - "
        f"{max(values)}"
    )


# ============================================================
# PLOT 1 â€” LABEL DISTRIBUTION
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


label_values = [
    train_counts[label]
    for label in LABELS
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    LABELS,
    label_values
)

plt.xlabel(
    "Pedagogical move"
)

plt.ylabel(
    "Training examples"
)

plt.title(
    "MathDial Training Move Distribution"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "mathdial_label_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2 â€” HISTORY LENGTH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    [
        row["history_num_turns"]
        for row in train
    ],
    bins=range(
        0,
        max(
            row["history_num_turns"]
            for row in train
        )
        + 2
    ),
)

plt.xlabel(
    "Conversation history turns"
)

plt.ylabel(
    "Number of training examples"
)

plt.title(
    "MathDial Training History Length"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "mathdial_history_length.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "train":
        train_summary,

    "validation":
        validation_summary,

    "train_imbalance_ratio":
        imbalance_ratio,

    "history_by_target_move":
        history_by_label,
}


with SUMMARY_PATH.open(
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
    SUMMARY_PATH
)

print(
    RESULT_DIR
    / "mathdial_label_distribution.png"
)

print(
    RESULT_DIR
    / "mathdial_history_length.png"
)


print(
    "\nMATHDIAL EDA COMPLETE"
)
