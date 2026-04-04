import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("results/baselines")

MD2_PATH = (
    BASE
    / "md2_validation_predictions.jsonl"
)

MD3_PATH = (
    BASE
    / "md3_modernbert_predictions.jsonl"
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
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


md2_rows = load_jsonl(
    MD2_PATH
)

md3_rows = load_jsonl(
    MD3_PATH
)


print("=" * 70)
print("MD2 / MD3 PREDICTION VALIDATION")
print("=" * 70)


print(
    "\nMD2 rows:",
    len(md2_rows),
)

print(
    "MD3 rows:",
    len(md3_rows),
)


assert len(md2_rows) == 1850
assert len(md3_rows) == 1850


# ============================================================
# MATCH BY EXAMPLE ID
# ============================================================

md2_by_id = {

    row["example_id"]:
        row

    for row in md2_rows
}


md3_by_id = {

    row["example_id"]:
        row

    for row in md3_rows
}


md2_ids = set(
    md2_by_id
)

md3_ids = set(
    md3_by_id
)


print(
    "\nExample-ID overlap:",
    len(
        md2_ids
        & md3_ids
    ),
)

print(
    "Only in MD2:",
    len(
        md2_ids
        - md3_ids
    ),
)

print(
    "Only in MD3:",
    len(
        md3_ids
        - md2_ids
    ),
)


assert md2_ids == md3_ids


# ============================================================
# SORT IDENTICALLY
# ============================================================

example_ids = sorted(
    md2_ids
)


md2 = [
    md2_by_id[x]
    for x in example_ids
]

md3 = [
    md3_by_id[x]
    for x in example_ids
]


# ============================================================
# TRUE LABEL CONSISTENCY
# ============================================================

true_md2 = [
    row["true_move"]
    for row in md2
]

true_md3 = [
    row["true_move"]
    for row in md3
]


assert true_md2 == true_md3


# ============================================================
# PROBABILITY VALIDATION
# ============================================================

def probability_matrix(rows):

    return np.array([

        [
            row["probabilities"][label]
            for label in LABELS
        ]

        for row in rows
    ])


p2 = probability_matrix(
    md2
)

p3 = probability_matrix(
    md3
)


print(
    "\nMD2 probability shape:",
    p2.shape,
)

print(
    "MD3 probability shape:",
    p3.shape,
)


assert p2.shape == (
    1850,
    4,
)

assert p3.shape == (
    1850,
    4,
)


assert np.allclose(
    p2.sum(axis=1),
    1.0,
    atol=1e-4,
)

assert np.allclose(
    p3.sum(axis=1),
    1.0,
    atol=1e-4,
)


# ============================================================
# TRUE IDS
# ============================================================

label2id = {

    label: index

    for index, label
    in enumerate(LABELS)
}


y_true = np.array([

    label2id[label]

    for label in true_md2
])


# ============================================================
# BASIC METRIC
# ============================================================

def evaluate(
    name,
    probabilities,
    mask=None,
):

    if mask is None:

        mask = np.ones(
            len(y_true),
            dtype=bool,
        )


    true_subset = (
        y_true[mask]
    )

    probabilities_subset = (
        probabilities[mask]
    )


    pred = np.argmax(
        probabilities_subset,
        axis=1,
    )


    accuracy = accuracy_score(
        true_subset,
        pred,
    )


    macro_f1 = f1_score(

        true_subset,
        pred,

        labels=[
            0,
            1,
            2,
            3,
        ],

        average="macro",

        zero_division=0,
    )


    return (
        accuracy,
        macro_f1,
        len(true_subset),
    )


# ============================================================
# REPRODUCE OVERALL RESULTS
# ============================================================

md2_accuracy, md2_f1, _ = (
    evaluate(
        "MD2",
        p2,
    )
)


md3_accuracy, md3_f1, _ = (
    evaluate(
        "MD3",
        p3,
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "OVERALL REPRODUCTION"
)

print(
    "=" * 70
)


print(
    f"\nMD2 accuracy: "
    f"{md2_accuracy:.6f}"
)

print(
    f"MD2 Macro-F1: "
    f"{md2_f1:.6f}"
)


print(
    f"\nMD3 accuracy: "
    f"{md3_accuracy:.6f}"
)

print(
    f"MD3 Macro-F1: "
    f"{md3_f1:.6f}"
)


# ============================================================
# EXACT MATCHED LONG-DIALOGUE ANALYSIS
#
# Define dialogue length structurally rather than using
# tokenizer-specific token counts.
#
# This avoids comparing different >512 subsets.
# ============================================================

history_turns = np.array([

    row["history_num_turns"]

    for row in md2
])


print(
    "\n"
    + "=" * 70
)

print(
    "MATCHED HISTORY-LENGTH ANALYSIS"
)

print(
    "=" * 70
)


for threshold in [
    4,
    6,
    8,
    10,
    12,
    16,
    20,
]:

    mask = (
        history_turns
        >= threshold
    )


    if mask.sum() < 20:
        continue


    _, f1_md2, n = evaluate(
        "MD2",
        p2,
        mask,
    )

    _, f1_md3, _ = evaluate(
        "MD3",
        p3,
        mask,
    )


    print(
        f"\nHistory >= {threshold:2d} turns"
    )

    print(
        f"  Examples: "
        f"{n}"
    )

    print(
        f"  MD2 Macro-F1: "
        f"{f1_md2:.6f}"
    )

    print(
        f"  MD3 Macro-F1: "
        f"{f1_md3:.6f}"
    )

    print(
        f"  MD3 - MD2: "
        f"{(
            f1_md3
            - f1_md2
        ):+.6f}"
    )


# ============================================================
# SIMPLE PROBABILITY ENSEMBLES
#
# alpha = weight on MD2
#
# combined =
#   alpha * MD2
#   +
#   (1-alpha) * MD3
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PROBABILITY ENSEMBLE"
)

print(
    "=" * 70
)


best_ensemble = None


for alpha in np.arange(
    0.0,
    1.01,
    0.10,
):

    combined = (

        alpha * p2

        +

        (1.0 - alpha) * p3
    )


    accuracy, macro_f1, _ = (
        evaluate(
            "ensemble",
            combined,
        )
    )


    print(
        f"\nMD2 weight "
        f"{alpha:.1f} | "
        f"MD3 weight "
        f"{1-alpha:.1f}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"  Macro-F1: "
        f"{macro_f1:.6f}"
    )


    candidate = {

        "alpha":
            float(alpha),

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),
    }


    if (
        best_ensemble is None

        or

        candidate[
            "macro_f1"
        ]
        >
        best_ensemble[
            "macro_f1"
        ]
    ):

        best_ensemble = (
            candidate
        )


# ============================================================
# CONTEXT-AWARE GATING
#
# MD2 for shorter histories.
# MD3 for longer histories.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CONTEXT-AWARE MODEL GATING"
)

print(
    "=" * 70
)


best_gate = None


for threshold in range(
    2,
    25,
):

    combined = p2.copy()


    use_md3 = (
        history_turns
        >= threshold
    )


    combined[
        use_md3
    ] = p3[
        use_md3
    ]


    accuracy, macro_f1, _ = (
        evaluate(
            "gate",
            combined,
        )
    )


    candidate = {

        "threshold":
            threshold,

        "md3_examples":
            int(
                use_md3.sum()
            ),

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),
    }


    if (
        best_gate is None

        or

        candidate[
            "macro_f1"
        ]
        >
        best_gate[
            "macro_f1"
        ]
    ):

        best_gate = (
            candidate
        )


print(
    "\nBest gating threshold:"
)

print(
    best_gate
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALIDATION SUMMARY"
)

print(
    "=" * 70
)


print(
    f"\nMD2 Macro-F1: "
    f"{md2_f1:.6f}"
)

print(
    f"MD3 Macro-F1: "
    f"{md3_f1:.6f}"
)


print(
    "\nBest probability ensemble:"
)

print(
    best_ensemble
)


print(
    "\nBest history-aware gate:"
)

print(
    best_gate
)


print(
    "\nMD2 / MD3 ANALYSIS: PASSED"
)
