import json
from pathlib import Path

import numpy as np

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_train.jsonl"
)

VAL_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_validation.jsonl"
)


OUTPUT_PATH = Path(
    "results/self_improvement/"
    "si1_depth_confound_audit.json"
)


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


train_transitions = load_jsonl(
    TRAIN_PATH
)

val_transitions = load_jsonl(
    VAL_PATH
)


assert len(train_transitions) == 9989
assert len(val_transitions) == 2509


print("=" * 70)
print("SI1-C â€” VALUE DEPTH-CONFOUND AUDIT")
print("=" * 70)


# ============================================================
# FORMATTING
# ============================================================

def format_history(history):

    if not history:

        return (
            "[No previous conversation]"
        )


    return "\n".join(

        f"{turn['user']}: "
        f"{turn['text']}"

        for turn in history
    )


def format_state(
    problem,
    history,
):

    return (
        "Problem:\n"
        f"{problem}"
        "\n\n"
        "Conversation:\n"
        f"{format_history(history)}"
    )


# ============================================================
# DEDUPLICATE STATES
# ============================================================

def build_unique_states(
    transitions
):

    states = {}


    for transition in transitions:

        outcome = (
            transition[
                "terminal_outcome"
            ]
        )


        source_index = int(
            transition[
                "source_row_index"
            ]
        )


        for history in [

            transition[
                "pre_history"
            ],

            transition[
                "post_history"
            ],

        ]:

            signature = json.dumps(
                history,
                ensure_ascii=False,
                sort_keys=True,
            )


            key = (
                source_index,
                signature,
            )


            if key in states:

                assert (
                    states[key][
                        "terminal_outcome"
                    ]
                    == outcome
                )

                continue


            states[key] = {

                "source_row_index":
                    source_index,

                "group_id":
                    str(
                        transition[
                            "group_id"
                        ]
                    ),

                "problem":
                    transition[
                        "problem"
                    ],

                "history":
                    history,

                "history_num_turns":
                    len(history),

                "terminal_outcome":
                    outcome,
            }


    return list(
        states.values()
    )


train_states = build_unique_states(
    train_transitions
)

val_states = build_unique_states(
    val_transitions
)


print(
    "\nTrain unique states:",
    len(train_states),
)

print(
    "Validation unique states:",
    len(val_states),
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_groups = {

    row[
        "group_id"
    ]

    for row
    in train_states
}


val_groups = {

    row[
        "group_id"
    ]

    for row
    in val_states
}


print(
    "\nGroup overlap:",
    len(
        train_groups
        & val_groups
    ),
)


assert len(
    train_groups
    & val_groups
) == 0


# ============================================================
# BINARY TARGET
#
# clean success
# vs
# revealed/failure
#
# This is ONLY for diagnosing V(s)=P(clean success).
# ============================================================

def clean_target(row):

    return int(
        row[
            "terminal_outcome"
        ]
        == "clean_success"
    )


y_train = np.array([

    clean_target(row)

    for row in train_states

], dtype=np.int64)


y_val = np.array([

    clean_target(row)

    for row in val_states

], dtype=np.int64)


train_depth = np.array([

    row[
        "history_num_turns"
    ]

    for row in train_states

], dtype=np.float64)


val_depth = np.array([

    row[
        "history_num_turns"
    ]

    for row in val_states

], dtype=np.float64)


# ============================================================
# CLEAN-SUCCESS RATE BY OBSERVED DEPTH
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CLEAN-SUCCESS RATE BY HISTORY DEPTH"
)

print(
    "=" * 70
)


# Inclusive lower bound,
# exclusive upper bound.
#
# Last bin has no upper limit.

depth_bins = [

    (
        "0-2",
        0,
        3,
    ),

    (
        "3-4",
        3,
        5,
    ),

    (
        "5-6",
        5,
        7,
    ),

    (
        "7-8",
        7,
        9,
    ),

    (
        "9-12",
        9,
        13,
    ),

    (
        "13-16",
        13,
        17,
    ),

    (
        "17+",
        17,
        None,
    ),
]


depth_rate_results = {}


for name, lower, upper in depth_bins:

    if upper is None:

        mask = (
            val_depth
            >= lower
        )

    else:

        mask = (
            (
                val_depth
                >= lower
            )

            &

            (
                val_depth
                < upper
            )
        )


    n = int(
        mask.sum()
    )


    if n == 0:

        continue


    clean_rate = float(
        y_val[
            mask
        ].mean()
    )


    depth_rate_results[
        name
    ] = {

        "n":
            n,

        "clean_rate":
            clean_rate,
    }


    print(
        f"\nHistory turns {name}"
    )

    print(
        f"  n: "
        f"{n}"
    )

    print(
        f"  Clean-success rate: "
        f"{clean_rate:.4f}"
    )


# ============================================================
# DEPTH-ONLY MODEL
# ============================================================

depth_scaler = (
    StandardScaler()
)


X_depth_train = (
    depth_scaler.fit_transform(
        train_depth.reshape(
            -1,
            1,
        )
    )
)


X_depth_val = (
    depth_scaler.transform(
        val_depth.reshape(
            -1,
            1,
        )
    )
)


depth_model = (
    LogisticRegression(

        random_state=42,

        max_iter=1000,
    )
)


depth_model.fit(
    X_depth_train,
    y_train,
)


p_depth = (
    depth_model.predict_proba(
        X_depth_val
    )[:, 1]
)


depth_auc = roc_auc_score(
    y_val,
    p_depth,
)


depth_logloss = log_loss(
    y_val,
    p_depth,
)


depth_brier = brier_score_loss(
    y_val,
    p_depth,
)


print(
    "\n"
    + "=" * 70
)

print(
    "DEPTH-ONLY CLEAN-SUCCESS MODEL"
)

print(
    "=" * 70
)


print(
    f"\nROC-AUC: "
    f"{depth_auc:.6f}"
)

print(
    f"Log loss: "
    f"{depth_logloss:.6f}"
)

print(
    f"Brier: "
    f"{depth_brier:.6f}"
)


print(
    "\nDepth logistic coefficient:",
    float(
        depth_model.coef_[
            0,
            0
        ]
    ),
)


# ============================================================
# TEXT-ONLY BINARY MODEL
#
# Re-estimate directly as clean vs non-clean so comparison
# with the depth-only model is exact.
# ============================================================

X_train_text = [

    format_state(
        row[
            "problem"
        ],
        row[
            "history"
        ],
    )

    for row in train_states
]


X_val_text = [

    format_state(
        row[
            "problem"
        ],
        row[
            "history"
        ],
    )

    for row in val_states
]


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(
        1,
        2,
    ),

    min_df=2,

    max_features=100000,

    sublinear_tf=True,
)


X_text_train = (
    vectorizer.fit_transform(
        X_train_text
    )
)


X_text_val = (
    vectorizer.transform(
        X_val_text
    )
)


text_model = (
    LogisticRegression(

        random_state=42,

        max_iter=2000,
    )
)


text_model.fit(
    X_text_train,
    y_train,
)


p_text = (
    text_model.predict_proba(
        X_text_val
    )[:, 1]
)


text_auc = roc_auc_score(
    y_val,
    p_text,
)


text_logloss = log_loss(
    y_val,
    p_text,
)


text_brier = brier_score_loss(
    y_val,
    p_text,
)


print(
    "\n"
    + "=" * 70
)

print(
    "TEXT-ONLY CLEAN-SUCCESS MODEL"
)

print(
    "=" * 70
)


print(
    f"\nROC-AUC: "
    f"{text_auc:.6f}"
)

print(
    f"Log loss: "
    f"{text_logloss:.6f}"
)

print(
    f"Brier: "
    f"{text_brier:.6f}"
)


print(
    f"\nText AUC - depth AUC: "
    f"{(
        text_auc
        - depth_auc
    ):+.6f}"
)


# ============================================================
# TEXT + EXPLICIT DEPTH MODEL
#
# This tests whether text has additional information once
# observable history depth is included explicitly.
# ============================================================

X_combined_train = hstack([

    X_text_train,

    csr_matrix(
        X_depth_train
    ),

]).tocsr()


X_combined_val = hstack([

    X_text_val,

    csr_matrix(
        X_depth_val
    ),

]).tocsr()


combined_model = (
    LogisticRegression(

        random_state=42,

        max_iter=2000,
    )
)


combined_model.fit(
    X_combined_train,
    y_train,
)


p_combined = (
    combined_model.predict_proba(
        X_combined_val
    )[:, 1]
)


combined_auc = roc_auc_score(
    y_val,
    p_combined,
)


combined_logloss = log_loss(
    y_val,
    p_combined,
)


combined_brier = (
    brier_score_loss(
        y_val,
        p_combined,
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "TEXT + DEPTH CLEAN-SUCCESS MODEL"
)

print(
    "=" * 70
)


print(
    f"\nROC-AUC: "
    f"{combined_auc:.6f}"
)

print(
    f"Log loss: "
    f"{combined_logloss:.6f}"
)

print(
    f"Brier: "
    f"{combined_brier:.6f}"
)


print(
    f"\nCombined AUC - depth AUC: "
    f"{(
        combined_auc
        - depth_auc
    ):+.6f}"
)


# ============================================================
# FUNCTIONS FOR TRANSITION SCORING
# ============================================================

EPS = 1e-6


def logit(
    probability,
):

    probability = np.clip(
        probability,
        EPS,
        1.0 - EPS,
    )


    return np.log(
        probability
        /
        (
            1.0
            - probability
        )
    )


def text_probability(
    problem,
    history,
):

    text = format_state(
        problem,
        history,
    )


    features = (
        vectorizer.transform(
            [
                text
            ]
        )
    )


    return float(
        text_model
        .predict_proba(
            features
        )[0, 1]
    )


def depth_probability(
    history,
):

    depth = np.array(
        [
            [
                len(history)
            ]
        ],
        dtype=np.float64,
    )


    depth_scaled = (
        depth_scaler.transform(
            depth
        )
    )


    return float(
        depth_model
        .predict_proba(
            depth_scaled
        )[0, 1]
    )


# ============================================================
# RAW VS DEPTH-ADJUSTED TRANSITION DELTA
#
# Raw:
#
#     Î”V_text
#
# Adjusted:
#
#     residual(s)
#       =
#     logit(P_text(clean|s))
#       -
#     logit(P_depth(clean|depth(s)))
#
# Then:
#
#     Î”V_adjusted
#       =
#     residual(post)
#       -
#     residual(pre)
#
# If the negative raw delta mostly comes from dialogue
# progression/depth, this adjustment should remove much
# of the systematic decline.
# ============================================================

transition_results = []


for transition in val_transitions:

    problem = (
        transition[
            "problem"
        ]
    )


    pre_history = (
        transition[
            "pre_history"
        ]
    )


    post_history = (
        transition[
            "post_history"
        ]
    )


    pre_text = text_probability(
        problem,
        pre_history,
    )


    post_text = text_probability(
        problem,
        post_history,
    )


    pre_depth = depth_probability(
        pre_history
    )


    post_depth = depth_probability(
        post_history
    )


    raw_delta = (
        post_text
        - pre_text
    )


    pre_residual = (

        logit(
            pre_text
        )

        -

        logit(
            pre_depth
        )
    )


    post_residual = (

        logit(
            post_text
        )

        -

        logit(
            post_depth
        )
    )


    adjusted_delta = (

        post_residual
        - pre_residual
    )


    transition_results.append({

        "terminal_outcome":
            transition[
                "terminal_outcome"
            ],

        "teacher_move":
            transition[
                "teacher_move"
            ],

        "pre_depth":
            len(
                pre_history
            ),

        "post_depth":
            len(
                post_history
            ),

        "raw_delta":
            float(
                raw_delta
            ),

        "adjusted_delta":
            float(
                adjusted_delta
            ),
    })


# ============================================================
# DELTA SUMMARY
# ============================================================

def summarize_delta(
    values,
):

    values = np.array(
        values,
        dtype=np.float64,
    )


    return {

        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "positive_fraction":
            float(
                np.mean(
                    values
                    > 0
                )
            ),
    }


print(
    "\n"
    + "=" * 70
)

print(
    "RAW VS DEPTH-ADJUSTED DELTA BY OUTCOME"
)

print(
    "=" * 70
)


delta_by_outcome = {}


for outcome in [

    "clean_success",
    "revealed_success",
    "failure",

]:

    subset = [

        row

        for row
        in transition_results

        if row[
            "terminal_outcome"
        ]
        == outcome
    ]


    raw = summarize_delta([

        row[
            "raw_delta"
        ]

        for row in subset
    ])


    adjusted = summarize_delta([

        row[
            "adjusted_delta"
        ]

        for row in subset
    ])


    delta_by_outcome[
        outcome
    ] = {

        "raw":
            raw,

        "depth_adjusted":
            adjusted,
    }


    print(
        f"\n{outcome}"
    )

    print(
        f"  n: "
        f"{raw['n']}"
    )


    print(
        "\n  RAW"
    )

    print(
        f"    mean: "
        f"{raw['mean']:+.6f}"
    )

    print(
        f"    median: "
        f"{raw['median']:+.6f}"
    )

    print(
        f"    fraction > 0: "
        f"{raw['positive_fraction']:.4f}"
    )


    print(
        "\n  DEPTH-ADJUSTED"
    )

    print(
        f"    mean: "
        f"{adjusted['mean']:+.6f}"
    )

    print(
        f"    median: "
        f"{adjusted['median']:+.6f}"
    )

    print(
        f"    fraction > 0: "
        f"{adjusted['positive_fraction']:.4f}"
    )


# ============================================================
# DEPTH-ADJUSTED DELTA BY MOVE
#
# Descriptive only.
# Still NOT causal move effectiveness.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "DEPTH-ADJUSTED DELTA BY MOVE"
)

print(
    "=" * 70
)


delta_by_move = {}


for move in [

    "generic",
    "probing",
    "focus",
    "telling",

]:

    values = [

        row[
            "adjusted_delta"
        ]

        for row
        in transition_results

        if row[
            "teacher_move"
        ]
        == move
    ]


    summary = summarize_delta(
        values
    )


    delta_by_move[
        move
    ] = summary


    print(
        f"\n{move}"
    )

    print(
        f"  n: "
        f"{summary['n']}"
    )

    print(
        f"  mean adjusted Î”: "
        f"{summary['mean']:+.6f}"
    )

    print(
        f"  median adjusted Î”: "
        f"{summary['median']:+.6f}"
    )

    print(
        f"  fraction > 0: "
        f"{summary['positive_fraction']:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

result = {

    "experiment":
        "SI1-C_depth_confound_audit",

    "train_unique_states":
        len(
            train_states
        ),

    "validation_unique_states":
        len(
            val_states
        ),

    "depth_clean_rates":
        depth_rate_results,

    "depth_only": {

        "roc_auc":
            float(
                depth_auc
            ),

        "log_loss":
            float(
                depth_logloss
            ),

        "brier":
            float(
                depth_brier
            ),

        "coefficient":
            float(
                depth_model.coef_[
                    0,
                    0
                ]
            ),
    },

    "text_only": {

        "roc_auc":
            float(
                text_auc
            ),

        "log_loss":
            float(
                text_logloss
            ),

        "brier":
            float(
                text_brier
            ),
    },

    "text_plus_depth": {

        "roc_auc":
            float(
                combined_auc
            ),

        "log_loss":
            float(
                combined_logloss
            ),

        "brier":
            float(
                combined_brier
            ),
    },

    "delta_by_outcome":
        delta_by_outcome,

    "depth_adjusted_delta_by_move":
        delta_by_move,
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
    "SI1-C DEPTH-CONFOUND AUDIT COMPLETE"
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
