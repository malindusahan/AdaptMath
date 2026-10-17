import json
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    "data/processed/mathdial"
)

TRAIN_PATH = (
    DATA_DIR / "train.jsonl"
)

VAL_PATH = (
    DATA_DIR / "validation.jsonl"
)

RESULT_DIR = Path(
    "results/baselines"
)

MAJORITY_RESULT_PATH = (
    RESULT_DIR
    / "mathdial_majority_validation.json"
)

TFIDF_RESULT_PATH = (
    RESULT_DIR
    / "mathdial_tfidf_logreg_validation.json"
)


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


# ============================================================
# LOAD DATA
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


print("=" * 70)
print("MATHDIAL-ONLY CPU BASELINES")
print("=" * 70)


print(
    f"\nTrain examples: "
    f"{len(train)}"
)

print(
    f"Validation examples: "
    f"{len(validation)}"
)


# ============================================================
# FORMAT INPUT
#
# Exactly the information available to the future model:
#
# Problem + previous conversation
#
# No target teacher utterance.
# No dialog-act labels.
# No hidden MathDial metadata.
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


def build_input(example):

    return (
        "Problem:\n"
        f"{example['problem']}"
        "\n\n"
        "Conversation:\n"
        f"{format_history(example['history'])}"
        "\n\n"
        "Next teacher pedagogical move:"
    )


X_train_text = [
    build_input(example)
    for example in train
]

X_val_text = [
    build_input(example)
    for example in validation
]


y_train = [
    example["target_move"]
    for example in train
]

y_val = [
    example["target_move"]
    for example in validation
]


# ============================================================
# COMMON EVALUATION
# ============================================================

def evaluate(
    name,
    true_labels,
    predicted_labels
):

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )


    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


    per_class = f1_score(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average=None,
        zero_division=0,
    )


    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )


    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS,
    )


    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print("=" * 70)


    print(
        f"\nValidation Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"Validation Macro-F1: "
        f"{macro_f1:.6f}"
    )


    print(
        "\nPer-class F1:"
    )


    for label, score in zip(
        LABELS,
        per_class
    ):

        print(
            f"{label:10s}: "
            f"{score:.6f}"
        )


    print(
        "\nClassification Report:"
    )

    print(report)


    print(
        "Confusion Matrix:"
    )

    print(
        "Label order:",
        LABELS
    )

    print(matrix)


    return {

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

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
# BASELINE 1 â€” MAJORITY CLASS
# ============================================================

train_distribution = Counter(
    y_train
)


print(
    "\nTraining label distribution:"
)


for label in LABELS:

    print(
        f"{label:10s}: "
        f"{train_distribution[label]}"
    )


majority_class = max(
    LABELS,
    key=lambda label:
        train_distribution[label]
)


print(
    f"\nMajority class: "
    f"{majority_class}"
)


majority_predictions = [
    majority_class
] * len(y_val)


majority_result = evaluate(
    "MD0 â€” MAJORITY BASELINE",
    y_val,
    majority_predictions,
)


majority_result[
    "experiment"
] = "MD0_mathdial_majority"

majority_result[
    "majority_class"
] = majority_class


# ============================================================
# BASELINE 2 â€” TF-IDF + LOGISTIC REGRESSION
#
# Ordinary logistic regression:
# no class weights.
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING MD1 â€” TF-IDF + LOGISTIC REGRESSION"
)

print(
    "=" * 70
)


vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    min_df=2,

    max_features=50000,

    sublinear_tf=True,

    lowercase=True,
)


X_train = vectorizer.fit_transform(
    X_train_text
)

X_val = vectorizer.transform(
    X_val_text
)


print(
    f"\nTF-IDF vocabulary size: "
    f"{len(vectorizer.vocabulary_)}"
)

print(
    f"Train feature matrix: "
    f"{X_train.shape}"
)

print(
    f"Validation feature matrix: "
    f"{X_val.shape}"
)


classifier = LogisticRegression(

    max_iter=3000,

    random_state=42,

    solver="lbfgs",
)


classifier.fit(
    X_train,
    y_train
)


tfidf_predictions = (
    classifier.predict(
        X_val
    )
)


tfidf_result = evaluate(
    "MD1 â€” TF-IDF + LOGISTIC REGRESSION",
    y_val,
    tfidf_predictions,
)


tfidf_result[
    "experiment"
] = (
    "MD1_mathdial_tfidf_logreg"
)


tfidf_result[
    "configuration"
] = {

    "ngram_range":
        [1, 2],

    "min_df":
        2,

    "max_features":
        50000,

    "sublinear_tf":
        True,

    "class_weight":
        None,
}


# ============================================================
# COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BASELINE COMPARISON"
)

print(
    "=" * 70
)


print(
    f"\nMD0 Majority Macro-F1: "
    f"{majority_result['macro_f1']:.6f}"
)

print(
    f"MD1 TF-IDF Macro-F1: "
    f"{tfidf_result['macro_f1']:.6f}"
)


difference = (

    tfidf_result["macro_f1"]
    - majority_result["macro_f1"]
)


print(
    f"Improvement: "
    f"{difference:+.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with MAJORITY_RESULT_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        majority_result,
        f,
        indent=2
    )


with TFIDF_RESULT_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        tfidf_result,
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
    MAJORITY_RESULT_PATH
)

print(
    TFIDF_RESULT_PATH
)


print(
    "\nMATHDIAL BASELINES COMPLETE"
)
