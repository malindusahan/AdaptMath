import json
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
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


RESULT_PATH = Path(
    "results/self_improvement/"
    "si1_value_baseline.json"
)


LABELS = [
    "clean_success",
    "revealed_success",
    "failure",
]


LABEL2ID = {
    label: i
    for i, label
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


train_transitions = load_jsonl(
    TRAIN_PATH
)

val_transitions = load_jsonl(
    VAL_PATH
)


assert len(train_transitions) == 9989
assert len(val_transitions) == 2509


print("=" * 70)
print("SI1-B â€” VALUE MODEL CPU BASELINE")
print("=" * 70)


print(
    "\nTrain transitions:",
    len(train_transitions),
)

print(
    "Validation transitions:",
    len(val_transitions),
)


# ============================================================
# STATE FORMAT
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
# BUILD UNIQUE STATES
#
# Important:
#
# post-state of transition t is frequently the same
# observable state as pre-state of transition t+1.
#
# We therefore deduplicate states.
# ============================================================

def build_unique_states(
    transitions
):

    state_map = {}


    for transition in transitions:

        outcome = (
            transition[
                "terminal_outcome"
            ]
        )


        source_index = (
            transition[
                "source_row_index"
            ]
        )


        group_id = (
            transition[
                "group_id"
            ]
        )


        problem = (
            transition[
                "problem"
            ]
        )


        for state_type, history in [

            (
                "pre",
                transition[
                    "pre_history"
                ],
            ),

            (
                "post",
                transition[
                    "post_history"
                ],
            ),

        ]:

            # Observable-state identity.
            #
            # We include source dialogue because two
            # dialogues can happen to contain identical
            # text while representing separate trajectories.

            history_signature = json.dumps(
                history,
                ensure_ascii=False,
                sort_keys=True,
            )


            key = (
                source_index,
                history_signature,
            )


            if key in state_map:

                existing = state_map[
                    key
                ]


                assert (
                    existing[
                        "terminal_outcome"
                    ]
                    == outcome
                )

                continue


            state_map[
                key
            ] = {

                "source_row_index":
                    source_index,

                "group_id":
                    group_id,

                "problem":
                    problem,

                "history":
                    history,

                "history_num_turns":
                    len(history),

                "terminal_outcome":
                    outcome,

                "origin":
                    state_type,
            }


    return list(
        state_map.values()
    )


train_states = build_unique_states(
    train_transitions
)

val_states = build_unique_states(
    val_transitions
)


print(
    "\n"
    + "=" * 70
)

print(
    "UNIQUE STATE DATASET"
)

print(
    "=" * 70
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

    str(
        row["group_id"]
    )

    for row
    in train_states
}


val_groups = {

    str(
        row["group_id"]
    )

    for row
    in val_states
}


train_sources = {

    int(
        row[
            "source_row_index"
        ]
    )

    for row
    in train_states
}


val_sources = {

    int(
        row[
            "source_row_index"
        ]
    )

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

print(
    "Source-dialogue overlap:",
    len(
        train_sources
        & val_sources
    ),
)


assert len(
    train_groups
    & val_groups
) == 0


assert len(
    train_sources
    & val_sources
) == 0


# ============================================================
# INPUT / TARGET
# ============================================================

X_train = [

    format_state(
        row[
            "problem"
        ],
        row[
            "history"
        ],
    )

    for row
    in train_states
]


X_val = [

    format_state(
        row[
            "problem"
        ],
        row[
            "history"
        ],
    )

    for row
    in val_states
]


y_train = np.array([

    LABEL2ID[
        row[
            "terminal_outcome"
        ]
    ]

    for row
    in train_states

])


y_val = np.array([

    LABEL2ID[
        row[
            "terminal_outcome"
        ]
    ]

    for row
    in val_states

])


# ============================================================
# LABEL DISTRIBUTIONS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STATE OUTCOME DISTRIBUTION"
)

print(
    "=" * 70
)


for split_name, labels in [

    ("TRAIN", y_train),
    ("VALIDATION", y_val),

]:

    print(
        f"\n{split_name}"
    )


    for label_id, label in enumerate(
        LABELS
    ):

        count = int(
            (
                labels
                == label_id
            ).sum()
        )


        print(
            f"  {label:18s}: "
            f"{count:5d} "
            f"({(
                count
                / len(labels)
                * 100
            ):.2f}%)"
        )


# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_class = int(
    np.bincount(
        y_train
    ).argmax()
)


majority_pred = np.full(
    len(y_val),
    majority_class,
)


majority_accuracy = (
    accuracy_score(
        y_val,
        majority_pred,
    )
)


majority_macro_f1 = (
    f1_score(
        y_val,
        majority_pred,
        labels=[
            0,
            1,
            2,
        ],
        average="macro",
        zero_division=0,
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "SI1-V0 â€” MAJORITY BASELINE"
)

print(
    "=" * 70
)


print(
    "\nPredicted class:",
    LABELS[
        majority_class
    ],
)

print(
    f"Accuracy: "
    f"{majority_accuracy:.6f}"
)

print(
    f"Macro-F1: "
    f"{majority_macro_f1:.6f}"
)


# ============================================================
# TF-IDF LOGISTIC REGRESSION
#
# No class weighting initially.
#
# We need probability estimates for V(s), so we first test
# the natural empirical distribution instead of deliberately
# changing the class priors.
# ============================================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            ngram_range=(
                1,
                2,
            ),

            min_df=2,

            max_features=
                100000,

            sublinear_tf=True,
        ),
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=
                2000,

            solver=
                "lbfgs",

            random_state=
                42,
        ),
    ),
])


print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING SI1-V1 TF-IDF VALUE MODEL"
)

print(
    "=" * 70
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = (
    model.predict_proba(
        X_val
    )
)


predictions = np.argmax(
    probabilities,
    axis=1,
)


accuracy = accuracy_score(
    y_val,
    predictions,
)


macro_f1 = f1_score(

    y_val,
    predictions,

    labels=[
        0,
        1,
        2,
    ],

    average="macro",

    zero_division=0,
)


per_class_f1 = f1_score(

    y_val,
    predictions,

    labels=[
        0,
        1,
        2,
    ],

    average=None,

    zero_division=0,
)


cross_entropy = log_loss(

    y_val,
    probabilities,

    labels=[
        0,
        1,
        2,
    ],
)


# ============================================================
# MULTICLASS BRIER SCORE
#
# Mean squared error between predicted probability vector
# and one-hot terminal outcome.
# ============================================================

one_hot = np.eye(
    len(LABELS)
)[
    y_val
]


brier = float(
    np.mean(
        np.sum(
            (
                probabilities
                - one_hot
            )
            ** 2,
            axis=1,
        )
    )
)


# ============================================================
# CLEAN-SUCCESS DISCRIMINATION
#
# This matters because our first proposed value is:
#
# V(s) = P(clean_success | state)
# ============================================================

clean_binary = (
    y_val
    ==
    LABEL2ID[
        "clean_success"
    ]
).astype(
    np.int64
)


clean_probability = (
    probabilities[
        :,
        LABEL2ID[
            "clean_success"
        ]
    ]
)


clean_auc = roc_auc_score(
    clean_binary,
    clean_probability,
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SI1-V1 VALIDATION RESULTS"
)

print(
    "=" * 70
)


print(
    f"\nAccuracy: "
    f"{accuracy:.6f}"
)

print(
    f"Macro-F1: "
    f"{macro_f1:.6f}"
)

print(
    f"Log loss: "
    f"{cross_entropy:.6f}"
)

print(
    f"Multiclass Brier: "
    f"{brier:.6f}"
)

print(
    f"Clean-success ROC-AUC: "
    f"{clean_auc:.6f}"
)


print(
    "\nPer-class F1:"
)


for label, score in zip(
    LABELS,
    per_class_f1,
):

    print(
        f"  {label:18s}: "
        f"{score:.6f}"
    )


print(
    "\nConfusion Matrix:"
)

print(
    np.array(
        confusion_matrix(
            y_val,
            predictions,
            labels=[
                0,
                1,
                2,
            ],
        )
    )
)


print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_val,
        predictions,

        labels=[
            0,
            1,
            2,
        ],

        target_names=
            LABELS,

        digits=4,

        zero_division=0,
    )
)


# ============================================================
# TRANSITION DELTA ANALYSIS
#
# Score PRE and POST states from the ORIGINAL SI validation
# transition set.
#
# This is the first test of whether:
#
# V(post) - V(pre)
#
# behaves like a useful progress signal.
# ============================================================

def model_clean_probability(
    problem,
    history,
):

    text = format_state(
        problem,
        history,
    )


    probs = model.predict_proba(
        [
            text
        ]
    )[0]


    return float(
        probs[
            LABEL2ID[
                "clean_success"
            ]
        ]
    )


transition_deltas = []


for transition in val_transitions:

    before = (
        model_clean_probability(

            transition[
                "problem"
            ],

            transition[
                "pre_history"
            ],
        )
    )


    after = (
        model_clean_probability(

            transition[
                "problem"
            ],

            transition[
                "post_history"
            ],
        )
    )


    delta = (
        after
        - before
    )


    transition_deltas.append({

        "terminal_outcome":
            transition[
                "terminal_outcome"
            ],

        "teacher_move":
            transition[
                "teacher_move"
            ],

        "remaining_teacher_moves":
            transition[
                "remaining_teacher_moves"
            ],

        "before":
            before,

        "after":
            after,

        "delta":
            delta,
    })


# ============================================================
# DELTA BY TERMINAL OUTCOME
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALUE DELTA BY TERMINAL OUTCOME"
)

print(
    "=" * 70
)


delta_outcome_summary = {}


for outcome in LABELS:

    values = np.array([

        row[
            "delta"
        ]

        for row
        in transition_deltas

        if row[
            "terminal_outcome"
        ]
        == outcome

    ], dtype=np.float64)


    summary = {

        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "positive_fraction":
            float(
                np.mean(
                    values
                    > 0
                )
            ),
    }


    delta_outcome_summary[
        outcome
    ] = summary


    print(
        f"\n{outcome}"
    )

    print(
        f"  n: "
        f"{summary['n']}"
    )

    print(
        f"  mean Î”V: "
        f"{summary['mean']:+.6f}"
    )

    print(
        f"  median Î”V: "
        f"{summary['median']:+.6f}"
    )

    print(
        f"  fraction Î”V > 0: "
        f"{summary['positive_fraction']:.4f}"
    )


# ============================================================
# DELTA BY MOVE
#
# DESCRIPTIVE ONLY.
#
# This is NOT causal move effectiveness.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALUE DELTA BY OBSERVED MOVE"
)

print(
    "=" * 70
)


delta_move_summary = {}


for move in [
    "generic",
    "probing",
    "focus",
    "telling",
]:

    values = np.array([

        row[
            "delta"
        ]

        for row
        in transition_deltas

        if row[
            "teacher_move"
        ]
        == move

    ], dtype=np.float64)


    summary = {

        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "positive_fraction":
            float(
                np.mean(
                    values
                    > 0
                )
            ),
    }


    delta_move_summary[
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
        f"  mean Î”V: "
        f"{summary['mean']:+.6f}"
    )

    print(
        f"  median Î”V: "
        f"{summary['median']:+.6f}"
    )

    print(
        f"  fraction Î”V > 0: "
        f"{summary['positive_fraction']:.4f}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

result = {

    "experiment":
        "SI1-V1_tfidf_outcome_value",

    "value_definition":
        (
            "P(clean_success | "
            "problem, observable_history)"
        ),

    "train_unique_states":
        len(
            train_states
        ),

    "validation_unique_states":
        len(
            val_states
        ),

    "majority_baseline": {

        "accuracy":
            float(
                majority_accuracy
            ),

        "macro_f1":
            float(
                majority_macro_f1
            ),
    },

    "value_model": {

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "log_loss":
            float(
                cross_entropy
            ),

        "multiclass_brier":
            float(
                brier
            ),

        "clean_success_auc":
            float(
                clean_auc
            ),

        "per_class_f1": {

            label:
                float(score)

            for label, score
            in zip(
                LABELS,
                per_class_f1,
            )
        },
    },

    "delta_by_outcome":
        delta_outcome_summary,

    "delta_by_move":
        delta_move_summary,
}


with RESULT_PATH.open(
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
    "SI1-V1 VALUE BASELINE COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nSaved:"
)

print(
    RESULT_PATH
)
