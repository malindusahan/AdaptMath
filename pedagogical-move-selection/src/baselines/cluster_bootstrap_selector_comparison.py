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
    / "selector_cluster_bootstrap_comparison.json"
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
# CONSISTENCY
# ============================================================

for i in range(
    len(example_ids)
):

    assert (
        md2[i]["true_move"]
        ==
        md3[i]["true_move"]
        ==
        md4[i]["true_move"]
    )

    assert (
        str(md2[i]["qid"])
        ==
        str(md3[i]["qid"])
        ==
        str(md4[i]["qid"])
    )


# ============================================================
# TRUE LABELS
# ============================================================

y_true = np.array([

    LABEL2ID[
        row["true_move"]
    ]

    for row in md2

], dtype=np.int64)


# ============================================================
# QID CLUSTERS
# ============================================================

qids = np.array([

    str(
        row["qid"]
    )

    for row in md2

])


unique_qids = np.unique(
    qids
)


print("=" * 70)

print(
    "QID-CLUSTERED PAIRED BOOTSTRAP"
)

print("=" * 70)


print(
    "\nValidation examples:",
    len(y_true)
)

print(
    "Unique validation qids:",
    len(unique_qids)
)


# Our frozen validation split
# should contain 111 qids.

assert len(
    unique_qids
) == 111


qid_to_indices = {}


for qid in unique_qids:

    qid_to_indices[qid] = np.where(
        qids == qid
    )[0]


cluster_sizes = np.array([

    len(
        qid_to_indices[qid]
    )

    for qid in unique_qids

])


print(
    "\nMinimum examples/qid:",
    int(
        cluster_sizes.min()
    )
)

print(
    "Maximum examples/qid:",
    int(
        cluster_sizes.max()
    )
)

print(
    "Mean examples/qid:",
    float(
        cluster_sizes.mean()
    )
)

print(
    "Median examples/qid:",
    float(
        np.median(
            cluster_sizes
        )
    )
)


# ============================================================
# PROBABILITY MATRICES
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

        probabilities.sum(
            axis=1
        ),

        1.0,

        atol=1e-4,
    )


# ============================================================
# INDIVIDUAL MODEL PREDICTIONS
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


# ============================================================
# MD2 + MD4 ENSEMBLE
#
# Previously selected validation weights:
#
# 20% MD2
# 80% MD4
# ============================================================

p24 = (

    0.2 * p2

    +

    0.8 * p4
)


pred24 = np.argmax(
    p24,
    axis=1,
)


# ============================================================
# THREE-MODEL ENSEMBLE
#
# Previously selected:
#
# 50% MD2
# 20% MD3
# 30% MD4
# ============================================================

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


# ============================================================
# CONTEXT HYBRID
#
# DistilRoBERTa original length:
#
# <=512 -> MD4
# >512  -> MD3
#
# No validation-tuned threshold.
# ============================================================

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
# METRICS
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


def evaluate(
    true,
    pred,
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    true,
                    pred
                )
            ),

        "macro_f1":
            float(
                macro_f1(
                    true,
                    pred
                )
            ),
    }


# ============================================================
# POINT ESTIMATES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "POINT ESTIMATES"
)

print(
    "=" * 70
)


point_results = {}


for name, prediction in predictions.items():

    result = evaluate(
        y_true,
        prediction,
    )


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
# CLUSTER BOOTSTRAP
#
# IMPORTANT:
#
# We resample QIDs, NOT individual teacher turns.
#
# For each sampled QID, ALL examples belonging to that
# problem cluster are added to the bootstrap sample.
#
# If a QID is sampled twice, its complete collection of
# turns appears twice.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RUNNING CLUSTER BOOTSTRAP"
)

print(
    "=" * 70
)


print(
    f"\nIterations: "
    f"{BOOTSTRAP_ITERATIONS}"
)


rng = np.random.default_rng(
    SEED
)


bootstrap_scores = {

    name:
        np.empty(
            BOOTSTRAP_ITERATIONS,
            dtype=np.float64,
        )

    for name in predictions
}


bootstrap_sample_sizes = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=np.int64,
)


for iteration in range(
    BOOTSTRAP_ITERATIONS
):

    sampled_qids = rng.choice(

        unique_qids,

        size=len(
            unique_qids
        ),

        replace=True,
    )


    sampled_indices = np.concatenate([

        qid_to_indices[qid]

        for qid
        in sampled_qids

    ])


    bootstrap_sample_sizes[
        iteration
    ] = len(
        sampled_indices
    )


    sampled_true = y_true[
        sampled_indices
    ]


    for name, prediction in predictions.items():

        bootstrap_scores[
            name
        ][iteration] = macro_f1(

            sampled_true,

            prediction[
                sampled_indices
            ],
        )


# ============================================================
# BOOTSTRAP SAMPLE-SIZE DIAGNOSTIC
#
# Because cluster sizes differ, bootstrap samples do not
# necessarily contain exactly 1850 turns.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BOOTSTRAP SAMPLE-SIZE DIAGNOSTIC"
)

print(
    "=" * 70
)


print(
    "\nMinimum sampled turns:",
    int(
        bootstrap_sample_sizes.min()
    )
)

print(
    "Median sampled turns:",
    float(
        np.median(
            bootstrap_sample_sizes
        )
    )
)

print(
    "Maximum sampled turns:",
    int(
        bootstrap_sample_sizes.max()
    )
)


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def percentile_ci(values):

    return (

        float(
            np.percentile(
                values,
                2.5
            )
        ),

        float(
            np.percentile(
                values,
                50.0
            )
        ),

        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


# ============================================================
# MODEL INTERVALS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "QID-CLUSTERED MACRO-F1 INTERVALS"
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
        f"  Cluster-bootstrap median: "
        f"{median:.6f}"
    )

    print(
        f"  95% interval: "
        f"[{lower:.6f}, "
        f"{upper:.6f}]"
    )


# ============================================================
# PAIRED DIFFERENCES
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
    "QID-CLUSTERED PAIRED DIFFERENCES"
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


    fraction_positive = float(

        np.mean(
            differences
            > 0
        )
    )


    key = (
        f"{model_a}"
        f"_minus_"
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

        "fraction_positive":
            fraction_positive,
    }


    print(
        f"\n{model_a} - {model_b}"
    )

    print(
        f"  Point difference: "
        f"{point_difference:+.6f}"
    )

    print(
        f"  Cluster-bootstrap median: "
        f"{median:+.6f}"
    )

    print(
        f"  95% interval: "
        f"[{lower:+.6f}, "
        f"{upper:+.6f}]"
    )

    print(
        f"  Fraction > 0: "
        f"{fraction_positive:.4f}"
    )


# ============================================================
# ROUTING STRUCTURE BY QID
#
# Useful to see whether long examples are concentrated
# in only a tiny number of problems.
# ============================================================

long_qids = set(
    qids[
        long_mask
    ]
)


short_qids = set(
    qids[
        short_mask
    ]
)


only_long_qids = (
    long_qids
    - short_qids
)


mixed_qids = (
    long_qids
    & short_qids
)


print(
    "\n"
    + "=" * 70
)

print(
    "ROUTING CLUSTER SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nTotal qids:",
    len(unique_qids)
)

print(
    "QIDs containing >512 examples:",
    len(long_qids)
)

print(
    "QIDs containing <=512 examples:",
    len(short_qids)
)

print(
    "Mixed short+long QIDs:",
    len(mixed_qids)
)

print(
    "Only-long QIDs:",
    len(only_long_qids)
)


print(
    "\nShort routed turns:",
    int(
        short_mask.sum()
    )
)

print(
    "Long routed turns:",
    int(
        long_mask.sum()
    )
)


# ============================================================
# SAVE
# ============================================================

result = {

    "method":
        "paired_qid_cluster_bootstrap",

    "bootstrap_iterations":
        BOOTSTRAP_ITERATIONS,

    "seed":
        SEED,

    "validation_examples":
        int(
            len(y_true)
        ),

    "validation_qids":
        int(
            len(unique_qids)
        ),

    "routing_rule":
        (
            "MD4 if DistilRoBERTa "
            "original token length <=512; "
            "MD3 otherwise"
        ),

    "point_results":
        point_results,

    "score_intervals":
        score_intervals,

    "paired_differences":
        difference_results,

    "sample_size_diagnostic": {

        "minimum":
            int(
                bootstrap_sample_sizes.min()
            ),

        "median":
            float(
                np.median(
                    bootstrap_sample_sizes
                )
            ),

        "maximum":
            int(
                bootstrap_sample_sizes.max()
            ),
    },

    "routing_clusters": {

        "total_qids":
            int(
                len(unique_qids)
            ),

        "long_qids":
            int(
                len(long_qids)
            ),

        "short_qids":
            int(
                len(short_qids)
            ),

        "mixed_qids":
            int(
                len(mixed_qids)
            ),

        "only_long_qids":
            int(
                len(only_long_qids)
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
    "QID CLUSTER BOOTSTRAP COMPLETE"
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
