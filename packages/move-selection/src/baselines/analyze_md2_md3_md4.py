import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    "results/baselines"
)


MD2_PATH = (
    BASE
    / "md2_validation_predictions.jsonl"
)

MD3_PATH = (
    BASE
    / "md3_modernbert_predictions.jsonl"
)

MD4_PATH = (
    BASE
    / "md4_weighted_distilroberta_predictions.jsonl"
)


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


LABEL2ID = {
    label: index
    for index, label
    in enumerate(LABELS)
}


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

md4_rows = load_jsonl(
    MD4_PATH
)


print("=" * 70)
print("MD2 / MD3 / MD4 ANALYSIS")
print("=" * 70)


print(
    "\nMD2 rows:",
    len(md2_rows),
)

print(
    "MD3 rows:",
    len(md3_rows),
)

print(
    "MD4 rows:",
    len(md4_rows),
)


assert len(md2_rows) == 1850
assert len(md3_rows) == 1850
assert len(md4_rows) == 1850


# ============================================================
# MATCH BY EXAMPLE ID
# ============================================================

def index_rows(rows):

    return {

        row["example_id"]:
            row

        for row in rows
    }


md2_by_id = index_rows(
    md2_rows
)

md3_by_id = index_rows(
    md3_rows
)

md4_by_id = index_rows(
    md4_rows
)


ids2 = set(md2_by_id)
ids3 = set(md3_by_id)
ids4 = set(md4_by_id)


print(
    "\nMD2/MD3 overlap:",
    len(ids2 & ids3),
)

print(
    "MD2/MD4 overlap:",
    len(ids2 & ids4),
)

print(
    "MD3/MD4 overlap:",
    len(ids3 & ids4),
)


assert ids2 == ids3 == ids4


example_ids = sorted(
    ids2
)


md2 = [
    md2_by_id[x]
    for x in example_ids
]

md3 = [
    md3_by_id[x]
    for x in example_ids
]

md4 = [
    md4_by_id[x]
    for x in example_ids
]


# ============================================================
# TRUE LABEL VALIDATION
# ============================================================

true2 = [
    row["true_move"]
    for row in md2
]

true3 = [
    row["true_move"]
    for row in md3
]

true4 = [
    row["true_move"]
    for row in md4
]


assert (
    true2
    == true3
    == true4
)


y_true = np.array([

    LABEL2ID[label]

    for label in true2
])


# ============================================================
# PROBABILITIES
# ============================================================

def probability_matrix(rows):

    return np.array([

        [
            row[
                "probabilities"
            ][label]

            for label in LABELS
        ]

        for row in rows

    ], dtype=np.float64)


p2 = probability_matrix(
    md2
)

p3 = probability_matrix(
    md3
)

p4 = probability_matrix(
    md4
)


for name, probabilities in [

    ("MD2", p2),
    ("MD3", p3),
    ("MD4", p4),

]:

    assert probabilities.shape == (
        1850,
        4,
    )

    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        atol=1e-4,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_probabilities(
    probabilities
):

    pred = np.argmax(
        probabilities,
        axis=1,
    )


    accuracy = accuracy_score(
        y_true,
        pred,
    )


    macro_f1 = f1_score(

        y_true,
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


    per_class = f1_score(

        y_true,
        pred,

        labels=[
            0,
            1,
            2,
            3,
        ],

        average=None,

        zero_division=0,
    )


    matrix = confusion_matrix(

        y_true,
        pred,

        labels=[
            0,
            1,
            2,
            3,
        ],
    )


    return {

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "per_class_f1": {

            label:
                float(score)

            for label, score
            in zip(
                LABELS,
                per_class
            )
        },

        "confusion_matrix":
            matrix.tolist(),
    }


# ============================================================
# REPRODUCE INDIVIDUAL MODELS
# ============================================================

r2 = evaluate_probabilities(
    p2
)

r3 = evaluate_probabilities(
    p3
)

r4 = evaluate_probabilities(
    p4
)


print(
    "\n"
    + "=" * 70
)

print(
    "INDIVIDUAL MODEL REPRODUCTION"
)

print(
    "=" * 70
)


for name, result in [

    ("MD2", r2),
    ("MD3", r3),
    ("MD4", r4),

]:

    print(
        f"\n{name}"
    )

    print(
        f"  Accuracy: "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1: "
        f"{result['macro_f1']:.6f}"
    )


# ============================================================
# PAIRWISE ENSEMBLES
#
# Coarse 0.1 grid only.
#
# We intentionally avoid a very fine grid because that would
# over-tune the validation set for a tiny numerical gain.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PAIRWISE PROBABILITY ENSEMBLES"
)

print(
    "=" * 70
)


def search_pair(
    name_a,
    p_a,
    name_b,
    p_b,
):

    best = None


    print(
        f"\n{name_a} + {name_b}"
    )


    for alpha in np.arange(
        0.0,
        1.01,
        0.10,
    ):

        combined = (

            alpha
            * p_a

            +

            (
                1.0
                - alpha
            )
            * p_b
        )


        result = (
            evaluate_probabilities(
                combined
            )
        )


        candidate = {

            f"{name_a}_weight":
                float(alpha),

            f"{name_b}_weight":
                float(
                    1.0
                    - alpha
                ),

            "accuracy":
                result[
                    "accuracy"
                ],

            "macro_f1":
                result[
                    "macro_f1"
                ],
        }


        if (
            best is None

            or

            candidate[
                "macro_f1"
            ]
            >
            best[
                "macro_f1"
            ]
        ):

            best = candidate


    print(
        "  Best:",
        best
    )


    return best


best_23 = search_pair(
    "MD2",
    p2,
    "MD3",
    p3,
)


best_24 = search_pair(
    "MD2",
    p2,
    "MD4",
    p4,
)


best_34 = search_pair(
    "MD3",
    p3,
    "MD4",
    p4,
)


# ============================================================
# THREE-MODEL ENSEMBLE
#
# Coarse weights in steps of 0.1.
#
# Require:
#
# w2 + w3 + w4 = 1
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "THREE-MODEL PROBABILITY ENSEMBLE"
)

print(
    "=" * 70
)


best_three = None


grid = np.arange(
    0.0,
    1.01,
    0.10,
)


for w2 in grid:

    for w3 in grid:

        w4 = (
            1.0
            - w2
            - w3
        )


        if w4 < -1e-9:
            continue


        if w4 > 1.0 + 1e-9:
            continue


        w4 = max(
            0.0,
            w4
        )


        combined = (

            w2 * p2
            +
            w3 * p3
            +
            w4 * p4
        )


        result = (
            evaluate_probabilities(
                combined
            )
        )


        candidate = {

            "md2_weight":
                float(w2),

            "md3_weight":
                float(w3),

            "md4_weight":
                float(w4),

            "accuracy":
                result[
                    "accuracy"
                ],

            "macro_f1":
                result[
                    "macro_f1"
                ],
        }


        if (
            best_three is None

            or

            candidate[
                "macro_f1"
            ]
            >
            best_three[
                "macro_f1"
            ]
        ):

            best_three = candidate


print(
    "\nBest 3-model ensemble:"
)

print(
    best_three
)


# ============================================================
# CONTEXT-LIMIT HYBRID
#
# This is the main experiment.
#
# MD4:
#   short conversations where DistilRoBERTa sees everything
#
# MD3:
#   conversations that DistilRoBERTa would truncate
#
# IMPORTANT:
# MD2 and MD4 use the same tokenizer/input construction,
# so MD4's stored original_token_length gives an exact
# DistilRoBERTa <=512 / >512 split.
# ============================================================

distil_original_lengths = np.array([

    row[
        "original_token_length"
    ]

    for row in md4
])


short_mask = (
    distil_original_lengths
    <= 512
)

long_mask = (
    distil_original_lengths
    > 512
)


print(
    "\n"
    + "=" * 70
)

print(
    "CONTEXT-LIMIT HYBRID"
)

print(
    "=" * 70
)


print(
    "\nShort examples:",
    int(
        short_mask.sum()
    )
)

print(
    "Long examples:",
    int(
        long_mask.sum()
    )
)


assert short_mask.sum() == 1507
assert long_mask.sum() == 343


# MD4 for <=512.
# MD3 for >512.

context_hybrid = (
    p4.copy()
)


context_hybrid[
    long_mask
] = p3[
    long_mask
]


context_result = (
    evaluate_probabilities(
        context_hybrid
    )
)


print(
    "\nMD4 <=512 + MD3 >512:"
)

print(
    f"  Accuracy: "
    f"{context_result['accuracy']:.6f}"
)

print(
    f"  Macro-F1: "
    f"{context_result['macro_f1']:.6f}"
)


print(
    "\nPer-class F1:"
)


for label in LABELS:

    print(
        f"  {label:10s}: "
        f"{context_result['per_class_f1'][label]:.6f}"
    )


print(
    "\nConfusion Matrix:"
)

print(
    np.array(
        context_result[
            "confusion_matrix"
        ]
    )
)


# ============================================================
# OTHER NATURAL CONTEXT HYBRIDS
#
# Compare:
#   MD2 short + MD3 long
# against
#   MD4 short + MD3 long
#
# No threshold tuning.
# ============================================================

hybrid_md2_md3 = (
    p2.copy()
)


hybrid_md2_md3[
    long_mask
] = p3[
    long_mask
]


hybrid_23_result = (
    evaluate_probabilities(
        hybrid_md2_md3
    )
)


print(
    "\nMD2 <=512 + MD3 >512:"
)

print(
    f"  Accuracy: "
    f"{hybrid_23_result['accuracy']:.6f}"
)

print(
    f"  Macro-F1: "
    f"{hybrid_23_result['macro_f1']:.6f}"
)


# ============================================================
# SHORT/LONG MODEL ANALYSIS ON EXACT SAME SUBSETS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "EXACT SHORT/LONG COMPARISON"
)

print(
    "=" * 70
)


def subset_f1(
    probabilities,
    mask,
):

    predictions = np.argmax(
        probabilities[
            mask
        ],
        axis=1,
    )


    return f1_score(

        y_true[
            mask
        ],

        predictions,

        labels=[
            0,
            1,
            2,
            3,
        ],

        average="macro",

        zero_division=0,
    )


for name, probabilities in [

    ("MD2", p2),
    ("MD3", p3),
    ("MD4", p4),

]:

    short_f1 = subset_f1(
        probabilities,
        short_mask,
    )

    long_f1 = subset_f1(
        probabilities,
        long_mask,
    )


    print(
        f"\n{name}"
    )

    print(
        f"  <=512 Macro-F1: "
        f"{short_f1:.6f}"
    )

    print(
        f"  >512 Macro-F1: "
        f"{long_f1:.6f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

candidates = {

    "MD2":
        r2["macro_f1"],

    "MD3":
        r3["macro_f1"],

    "MD4":
        r4["macro_f1"],

    "MD2_MD3_pair":
        best_23[
            "macro_f1"
        ],

    "MD2_MD4_pair":
        best_24[
            "macro_f1"
        ],

    "MD3_MD4_pair":
        best_34[
            "macro_f1"
        ],

    "three_model":
        best_three[
            "macro_f1"
        ],

    "MD2_short_MD3_long":
        hybrid_23_result[
            "macro_f1"
        ],

    "MD4_short_MD3_long":
        context_result[
            "macro_f1"
        ],
}


best_name = max(
    candidates,
    key=candidates.get,
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL VALIDATION SUMMARY"
)

print(
    "=" * 70
)


for name, score in candidates.items():

    print(
        f"{name:24s}: "
        f"{score:.6f}"
    )


print(
    "\nBest candidate:"
)

print(
    best_name,
    f"{candidates[best_name]:.6f}"
)


print(
    "\nMD2 / MD3 / MD4 ANALYSIS: PASSED"
)
