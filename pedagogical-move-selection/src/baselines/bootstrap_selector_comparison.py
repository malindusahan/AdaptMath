import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BOOTSTRAP_ITERATIONS = 5000


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


OUTPUT_PATH = (
    BASE
    / "selector_bootstrap_comparison.json"
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


assert len(md2_rows) == 1850
assert len(md3_rows) == 1850
assert len(md4_rows) == 1850


# ============================================================
# ALIGN BY EXAMPLE ID
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


assert (
    set(md2_by_id)
    ==
    set(md3_by_id)
    ==
    set(md4_by_id)
)


example_ids = sorted(
    md2_by_id
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
# TRUE LABELS
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


assert true2 == true3 == true4


y_true = np.array([

    LABEL2ID[label]

    for label in true2

], dtype=np.int64)


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


for probabilities in [
    p2,
    p3,
    p4,
]:

    assert probabilities.shape == (
        1850,
        4,
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4,
    )


# ============================================================
# BUILD FINAL CANDIDATES
# ============================================================

pred2 = np.argmax(
    p2,
    axis=1,
)

pred3 = np.argmax(
    p3,
    axis=1,
)

pred4 = np.argmax(
    p4,
    axis=1,
)


# ------------------------------------------------------------
# Best MD2 + MD4 probability ensemble
#
# From previous validation:
#   20% MD2
#   80% MD4
# ------------------------------------------------------------

p24 = (
    0.2 * p2
    +
    0.8 * p4
)


pred24 = np.argmax(
    p24,
    axis=1,
)


# ------------------------------------------------------------
# Best 3-model probability ensemble
#
# From previous validation:
#   50% MD2
#   20% MD3
#   30% MD4
# ------------------------------------------------------------

p_three = (
    0.5 * p2
    +
    0.2 * p3
    +
    0.3 * p4
)


pred_three = np.argmax(
    p_three,
    axis=1,
)


# ------------------------------------------------------------
# Context-limit hybrid
#
# Exact DistilRoBERTa token length:
#
#   <=512 â†’ MD4
#   >512  â†’ MD3
#
# ------------------------------------------------------------

distil_lengths = np.array([

    row[
        "original_token_length"
    ]

    for row in md4

], dtype=np.int64)


short_mask = (
    distil_lengths
    <= 512
)

long_mask = (
    distil_lengths
    > 512
)


assert int(
    short_mask.sum()
) == 1507


assert int(
    long_mask.sum()
) == 343


pred_hybrid = (
    pred4.copy()
)


pred_hybrid[
    long_mask
] = pred3[
    long_mask
]


# ============================================================
# CANDIDATES
# ============================================================

predictions = {

    "MD2":
        pred2,

    "MD3":
        pred3,

    "MD4":
        pred4,

    "MD2_MD4_ensemble":
        pred24,

    "three_model_ensemble":
        pred_three,

    "context_hybrid":
        pred_hybrid,
}


# ============================================================
# METRIC HELPERS
# ============================================================

def macro_f1(
    true,
    pred,
):

    return f1_score(

        true,
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


def per_class_f1(
    true,
    pred,
):

    scores = f1_score(

        true,
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


    return {

        label:
            float(score)

        for label, score
        in zip(
            LABELS,
            scores
        )
    }


# ============================================================
# POINT ESTIMATES
# ============================================================

print("=" * 70)

print(
    "POINT ESTIMATES"
)

print("=" * 70)


point_results = {}


for name, pred in predictions.items():

    result = {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    pred
                )
            ),

        "macro_f1":
            float(
                macro_f1(
                    y_true,
                    pred
                )
            ),

        "per_class_f1":
            per_class_f1(
                y_true,
                pred
            ),
    }


    point_results[
        name
    ] = result


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
# HYBRID PER-CLASS RESULT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CONTEXT HYBRID PER-CLASS F1"
)

print(
    "=" * 70
)


for label in LABELS:

    print(
        f"\n{label:10s}: "
        f"{point_results['context_hybrid']['per_class_f1'][label]:.6f}"
    )


# ============================================================
# STRATIFIED PAIRED BOOTSTRAP
#
# Resample WITHIN each true class.
#
# Therefore every bootstrap sample has the same class
# distribution as the actual validation set.
#
# All models receive exactly the same sampled indices.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STRATIFIED PAIRED BOOTSTRAP"
)

print(
    "=" * 70
)


rng = np.random.default_rng(
    SEED
)


class_indices = {

    class_id:
        np.where(
            y_true == class_id
        )[0]

    for class_id in range(
        len(LABELS)
    )
}


for class_id, label in enumerate(
    LABELS
):

    print(
        f"\n{label:10s}: "
        f"{len(class_indices[class_id])} examples"
    )


bootstrap_scores = {

    name:
        np.empty(
            BOOTSTRAP_ITERATIONS,
            dtype=np.float64,
        )

    for name in predictions
}


for iteration in range(
    BOOTSTRAP_ITERATIONS
):

    sampled_parts = []


    for class_id in range(
        len(LABELS)
    ):

        indices = class_indices[
            class_id
        ]


        sampled = rng.choice(

            indices,

            size=len(
                indices
            ),

            replace=True,
        )


        sampled_parts.append(
            sampled
        )


    sampled_indices = (
        np.concatenate(
            sampled_parts
        )
    )


    sampled_true = y_true[
        sampled_indices
    ]


    for name, pred in predictions.items():

        bootstrap_scores[
            name
        ][iteration] = macro_f1(

            sampled_true,

            pred[
                sampled_indices
            ],
        )


# ============================================================
# CONFIDENCE INTERVAL HELPER
# ============================================================

def percentile_ci(values):

    lower = np.percentile(
        values,
        2.5,
    )

    median = np.percentile(
        values,
        50.0,
    )

    upper = np.percentile(
        values,
        97.5,
    )


    return (
        float(lower),
        float(median),
        float(upper),
    )


# ============================================================
# MODEL SCORE INTERVALS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BOOTSTRAP MACRO-F1 INTERVALS"
)

print(
    "=" * 70
)


score_intervals = {}


for name in predictions:

    lower, median, upper = (
        percentile_ci(
            bootstrap_scores[
                name
            ]
        )
    )


    score_intervals[
        name
    ] = {

        "lower_95":
            lower,

        "median":
            median,

        "upper_95":
            upper,
    }


    print(
        f"\n{name}"
    )

    print(
        f"  Point estimate: "
        f"{point_results[name]['macro_f1']:.6f}"
    )

    print(
        f"  Bootstrap median: "
        f"{median:.6f}"
    )

    print(
        f"  95% interval: "
        f"[{lower:.6f}, "
        f"{upper:.6f}]"
    )


# ============================================================
# PAIRED DIFFERENCES
#
# Primary comparisons:
#
# Hybrid - MD2
# Hybrid - MD4
# Hybrid - best MD2/MD4 ensemble
# Hybrid - 3-model ensemble
# ============================================================

comparisons = [

    (
        "context_hybrid",
        "MD2",
    ),

    (
        "context_hybrid",
        "MD4",
    ),

    (
        "context_hybrid",
        "MD2_MD4_ensemble",
    ),

    (
        "context_hybrid",
        "three_model_ensemble",
    ),
]


print(
    "\n"
    + "=" * 70
)

print(
    "PAIRED BOOTSTRAP DIFFERENCES"
)

print(
    "=" * 70
)


difference_results = {}


for model_a, model_b in comparisons:

    differences = (

        bootstrap_scores[
            model_a
        ]

        -

        bootstrap_scores[
            model_b
        ]
    )


    lower, median, upper = (
        percentile_ci(
            differences
        )
    )


    point_difference = (

        point_results[
            model_a
        ][
            "macro_f1"
        ]

        -

        point_results[
            model_b
        ][
            "macro_f1"
        ]
    )


    proportion_positive = float(

        np.mean(
            differences
            > 0
        )
    )


    proportion_nonpositive = float(

        np.mean(
            differences
            <= 0
        )
    )


    key = (
        f"{model_a}_minus_"
        f"{model_b}"
    )


    difference_results[
        key
    ] = {

        "point_difference":
            float(
                point_difference
            ),

        "lower_95":
            lower,

        "median":
            median,

        "upper_95":
            upper,

        "bootstrap_fraction_positive":
            proportion_positive,

        "bootstrap_fraction_nonpositive":
            proportion_nonpositive,
    }


    print(
        f"\n{model_a} - {model_b}"
    )

    print(
        f"  Point difference: "
        f"{point_difference:+.6f}"
    )

    print(
        f"  Bootstrap median: "
        f"{median:+.6f}"
    )

    print(
        f"  95% interval: "
        f"[{lower:+.6f}, "
        f"{upper:+.6f}]"
    )

    print(
        f"  Fraction > 0: "
        f"{proportion_positive:.4f}"
    )


# ============================================================
# SHORT / LONG VALIDATION SIZE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ROUTING SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nMD4 routed examples:",
    int(
        short_mask.sum()
    ),
)

print(
    "MD3 routed examples:",
    int(
        long_mask.sum()
    ),
)


print(
    f"\nMD4 percentage: "
    f"{short_mask.mean() * 100:.2f}%"
)

print(
    f"MD3 percentage: "
    f"{long_mask.mean() * 100:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

result = {

    "bootstrap_iterations":
        BOOTSTRAP_ITERATIONS,

    "seed":
        SEED,

    "routing_rule":
        (
            "MD4 if DistilRoBERTa "
            "original token length <=512; "
            "MD3 otherwise"
        ),

    "short_examples":
        int(
            short_mask.sum()
        ),

    "long_examples":
        int(
            long_mask.sum()
        ),

    "point_results":
        point_results,

    "score_intervals":
        score_intervals,

    "paired_differences":
        difference_results,
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
    "BOOTSTRAP ANALYSIS COMPLETE"
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
