import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

MD3_PREDICTIONS = Path(
    "artifacts/frozen_selector/md3/"
    "md3_modernbert_predictions.jsonl"
)

MD3_VALIDATION = Path(
    "artifacts/frozen_selector/md3/"
    "md3_modernbert_validation.json"
)

MD4_PREDICTIONS = Path(
    "artifacts/frozen_selector/md4/"
    "md4_weighted_distilroberta_predictions.jsonl"
)

MD4_VALIDATION = Path(
    "artifacts/frozen_selector/md4/"
    "md4_weighted_distilroberta_validation.json"
)


RECOVERED_PREDICTIONS = Path(
    "artifacts/frozen_selector/"
    "recovered_context_routed_hybrid_validation_predictions.jsonl"
)


RESULT_PATH = Path(
    "results/self_improvement/"
    "si7a_frozen_selector_recovery_validation.json"
)


# ============================================================
# FROZEN ARCHITECTURE
# ============================================================

LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


ROUTING_THRESHOLD = 512


# These are the exact metrics already frozen earlier
# in the project.
EXPECTED_FROZEN_ACCURACY = (
    0.5005405405405405
)

EXPECTED_FROZEN_MACRO_F1 = (
    0.48676777751987244
)


EXPECTED_VALIDATION_EXAMPLES = 1850

EXPECTED_MD4_SHORT = 1507

EXPECTED_MD3_LONG = 343


TOLERANCE = 1e-10

PROBABILITY_TOLERANCE = 1e-5


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            records.append(
                record
            )

    return records


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_required_fields(
    records,
    model_name,
):

    required_fields = {
        "example_id",
        "qid",
        "history_num_turns",
        "original_token_length",
        "true_move",
        "predicted_move",
        "confidence",
        "probabilities",
    }

    errors = []

    for index, record in enumerate(
        records
    ):

        missing = (
            required_fields
            -
            set(
                record.keys()
            )
        )

        if missing:

            errors.append(
                f"{model_name} row {index}: "
                f"missing fields {sorted(missing)}"
            )

    return errors


def validate_unique_example_ids(
    records,
    model_name,
):

    example_ids = [
        record["example_id"]
        for record in records
    ]

    unique_count = len(
        set(
            example_ids
        )
    )

    if unique_count != len(
        example_ids
    ):

        raise AssertionError(
            f"{model_name} contains duplicate "
            f"example IDs."
        )


def validate_probabilities(
    records,
    model_name,
):

    errors = []

    for index, record in enumerate(
        records
    ):

        probabilities = (
            record[
                "probabilities"
            ]
        )

        if set(
            probabilities.keys()
        ) != set(
            LABELS
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"unexpected probability labels."
            )

            continue

        values = np.array(
            [
                probabilities[label]
                for label in LABELS
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"non-finite probability."
            )

            continue

        if np.any(
            values < 0.0
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"negative probability."
            )

        probability_sum = float(
            np.sum(
                values
            )
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=PROBABILITY_TOLERANCE,
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"probabilities sum to "
                f"{probability_sum}."
            )

        predicted_from_probs = max(
            LABELS,
            key=lambda label:
                probabilities[
                    label
                ],
        )

        if (
            predicted_from_probs
            !=
            record[
                "predicted_move"
            ]
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"predicted_move does not "
                f"match probability argmax."
            )

        max_probability = max(
            probabilities.values()
        )

        if not np.isclose(
            float(
                record[
                    "confidence"
                ]
            ),
            float(
                max_probability
            ),
            atol=PROBABILITY_TOLERANCE,
        ):

            errors.append(
                f"{model_name} row {index}: "
                f"confidence does not match "
                f"maximum probability."
            )

    return errors


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    records,
):

    y_true = [
        record[
            "true_move"
        ]
        for record in records
    ]

    y_pred = [
        record[
            "predicted_move"
        ]
        for record in records
    ]

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
        )
    )

    per_class_f1_values = (
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average=None,
        )
    )

    per_class_f1 = {
        label:
            float(
                score
            )

        for label, score
        in zip(
            LABELS,
            per_class_f1_values,
        )
    }

    matrix = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )
        .tolist()
    )

    return {
        "examples":
            len(
                records
            ),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "per_class_f1":
            per_class_f1,

        "confusion_matrix":
            matrix,
    }


def assert_metric_close(
    observed,
    expected,
    metric_name,
):

    if not np.isclose(
        observed,
        expected,
        atol=TOLERANCE,
        rtol=0.0,
    ):

        raise AssertionError(
            f"{metric_name} mismatch: "
            f"observed={observed}, "
            f"expected={expected}"
        )


# ============================================================
# MODEL-SPECIFIC VALIDATION
# ============================================================

def validate_model_summary(
    model_name,
    predictions,
    validation_summary,
):

    metrics = compute_metrics(
        predictions
    )

    expected_overall = (
        validation_summary[
            "overall"
        ]
    )

    assert (
        metrics[
            "examples"
        ]
        ==
        expected_overall[
            "examples"
        ]
    )

    assert_metric_close(
        metrics[
            "accuracy"
        ],
        expected_overall[
            "accuracy"
        ],
        f"{model_name} accuracy",
    )

    assert_metric_close(
        metrics[
            "macro_f1"
        ],
        expected_overall[
            "macro_f1"
        ],
        f"{model_name} macro-F1",
    )

    return metrics


# ============================================================
# HYBRID RECOVERY
# ============================================================

def recover_hybrid(
    md3_predictions,
    md4_predictions,
):

    md3_by_id = {
        record[
            "example_id"
        ]:
            record

        for record
        in md3_predictions
    }

    recovered = []

    short_count = 0

    long_count = 0

    for md4_record in md4_predictions:

        example_id = (
            md4_record[
                "example_id"
            ]
        )

        md3_record = (
            md3_by_id[
                example_id
            ]
        )

        # --------------------------------------------
        # IMPORTANT:
        #
        # Routing uses the ORIGINAL MD4 /
        # DistilRoBERTa token length.
        #
        # It does NOT use ModernBERT's token length.
        # --------------------------------------------

        md4_token_length = int(
            md4_record[
                "original_token_length"
            ]
        )

        if (
            md4_token_length
            <=
            ROUTING_THRESHOLD
        ):

            source_model = "MD4"

            selected_record = (
                md4_record
            )

            short_count += 1

        else:

            source_model = "MD3"

            selected_record = (
                md3_record
            )

            long_count += 1

        recovered.append(
            {
                "example_id":
                    example_id,

                "qid":
                    md4_record[
                        "qid"
                    ],

                "history_num_turns":
                    md4_record[
                        "history_num_turns"
                    ],

                "md4_original_token_length":
                    md4_token_length,

                "md3_original_token_length":
                    int(
                        md3_record[
                            "original_token_length"
                        ]
                    ),

                "route":
                    (
                        "short_md4"
                        if source_model == "MD4"
                        else "long_md3"
                    ),

                "source_model":
                    source_model,

                "true_move":
                    md4_record[
                        "true_move"
                    ],

                "predicted_move":
                    selected_record[
                        "predicted_move"
                    ],

                "confidence":
                    float(
                        selected_record[
                            "confidence"
                        ]
                    ),

                "probabilities": {
                    label:
                        float(
                            selected_record[
                                "probabilities"
                            ][
                                label
                            ]
                        )

                    for label
                    in LABELS
                },
            }
        )

    return (
        recovered,
        short_count,
        long_count,
    )


# ============================================================
# CROSS-MODEL INTEGRITY
# ============================================================

def validate_cross_model_alignment(
    md3_predictions,
    md4_predictions,
):

    md3_ids = [
        record[
            "example_id"
        ]
        for record in md3_predictions
    ]

    md4_ids = [
        record[
            "example_id"
        ]
        for record in md4_predictions
    ]

    assert (
        set(
            md3_ids
        )
        ==
        set(
            md4_ids
        )
    ), (
        "MD3 and MD4 example-ID sets "
        "do not match."
    )

    md3_by_id = {
        record[
            "example_id"
        ]:
            record

        for record
        in md3_predictions
    }

    alignment_errors = []

    for md4_record in md4_predictions:

        example_id = (
            md4_record[
                "example_id"
            ]
        )

        md3_record = (
            md3_by_id[
                example_id
            ]
        )

        if (
            md3_record[
                "qid"
            ]
            !=
            md4_record[
                "qid"
            ]
        ):

            alignment_errors.append(
                (
                    example_id,
                    "qid_mismatch",
                )
            )

        if (
            md3_record[
                "true_move"
            ]
            !=
            md4_record[
                "true_move"
            ]
        ):

            alignment_errors.append(
                (
                    example_id,
                    "true_move_mismatch",
                )
            )

        if (
            md3_record[
                "history_num_turns"
            ]
            !=
            md4_record[
                "history_num_turns"
            ]
        ):

            alignment_errors.append(
                (
                    example_id,
                    "history_turn_mismatch",
                )
            )

    if alignment_errors:

        raise AssertionError(
            "MD3/MD4 alignment errors: "
            f"{alignment_errors[:10]}"
        )

    return 0


# ============================================================
# SAVE
# ============================================================

def save_jsonl(
    path,
    records,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "SI7-A FROZEN SELECTOR RECOVERY VALIDATION"
    )
    print("=" * 76)
    print()

    # --------------------------------------------------------
    # File presence
    # --------------------------------------------------------

    required_paths = [
        MD3_PREDICTIONS,
        MD3_VALIDATION,
        MD4_PREDICTIONS,
        MD4_VALIDATION,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    print(
        "Artifact files found: PASSED"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    md3_predictions = load_jsonl(
        MD3_PREDICTIONS
    )

    md4_predictions = load_jsonl(
        MD4_PREDICTIONS
    )

    md3_validation = load_json(
        MD3_VALIDATION
    )

    md4_validation = load_json(
        MD4_VALIDATION
    )

    print(
        "MD3 prediction rows:",
        len(
            md3_predictions
        ),
    )

    print(
        "MD4 prediction rows:",
        len(
            md4_predictions
        ),
    )

    assert (
        len(
            md3_predictions
        )
        ==
        EXPECTED_VALIDATION_EXAMPLES
    )

    assert (
        len(
            md4_predictions
        )
        ==
        EXPECTED_VALIDATION_EXAMPLES
    )

    print(
        "Prediction counts: PASSED"
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    field_errors = []

    field_errors.extend(
        validate_required_fields(
            md3_predictions,
            "MD3",
        )
    )

    field_errors.extend(
        validate_required_fields(
            md4_predictions,
            "MD4",
        )
    )

    if field_errors:

        raise AssertionError(
            "\n".join(
                field_errors[:20]
            )
        )

    print(
        "Prediction schemas: PASSED"
    )

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    validate_unique_example_ids(
        md3_predictions,
        "MD3",
    )

    validate_unique_example_ids(
        md4_predictions,
        "MD4",
    )

    print(
        "Unique example IDs: PASSED"
    )

    # --------------------------------------------------------
    # Probability integrity
    # --------------------------------------------------------

    probability_errors = []

    probability_errors.extend(
        validate_probabilities(
            md3_predictions,
            "MD3",
        )
    )

    probability_errors.extend(
        validate_probabilities(
            md4_predictions,
            "MD4",
        )
    )

    if probability_errors:

        raise AssertionError(
            "\n".join(
                probability_errors[:20]
            )
        )

    print(
        "Probability integrity: PASSED"
    )

    # --------------------------------------------------------
    # Cross-model alignment
    # --------------------------------------------------------

    validate_cross_model_alignment(
        md3_predictions,
        md4_predictions,
    )

    print(
        "MD3 / MD4 alignment: PASSED"
    )

    # --------------------------------------------------------
    # Reproduce standalone model metrics
    # --------------------------------------------------------

    md3_metrics = (
        validate_model_summary(
            "MD3",
            md3_predictions,
            md3_validation,
        )
    )

    md4_metrics = (
        validate_model_summary(
            "MD4",
            md4_predictions,
            md4_validation,
        )
    )

    print()
    print(
        "MD3 reproduced:"
    )

    print(
        "  Accuracy:",
        md3_metrics[
            "accuracy"
        ],
    )

    print(
        "  Macro-F1:",
        md3_metrics[
            "macro_f1"
        ],
    )

    print()

    print(
        "MD4 reproduced:"
    )

    print(
        "  Accuracy:",
        md4_metrics[
            "accuracy"
        ],
    )

    print(
        "  Macro-F1:",
        md4_metrics[
            "macro_f1"
        ],
    )

    print()

    print(
        "Standalone model reproduction: PASSED"
    )

    # --------------------------------------------------------
    # Recover exact frozen hybrid
    # --------------------------------------------------------

    (
        recovered,
        short_count,
        long_count,
    ) = recover_hybrid(
        md3_predictions=md3_predictions,
        md4_predictions=md4_predictions,
    )

    print()
    print(
        "Routing counts:"
    )

    print(
        "  MD4 <=512:",
        short_count,
    )

    print(
        "  MD3 >512 :",
        long_count,
    )

    assert (
        short_count
        ==
        EXPECTED_MD4_SHORT
    )

    assert (
        long_count
        ==
        EXPECTED_MD3_LONG
    )

    assert (
        short_count
        +
        long_count
        ==
        EXPECTED_VALIDATION_EXAMPLES
    )

    print(
        "Routing recovery: PASSED"
    )

    # --------------------------------------------------------
    # Hybrid metrics
    # --------------------------------------------------------

    hybrid_metrics = (
        compute_metrics(
            recovered
        )
    )

    print()
    print(
        "Recovered frozen hybrid:"
    )

    print(
        "  Examples:",
        hybrid_metrics[
            "examples"
        ],
    )

    print(
        "  Accuracy:",
        hybrid_metrics[
            "accuracy"
        ],
    )

    print(
        "  Macro-F1:",
        hybrid_metrics[
            "macro_f1"
        ],
    )

    print()

    assert_metric_close(
        hybrid_metrics[
            "accuracy"
        ],
        EXPECTED_FROZEN_ACCURACY,
        "Frozen hybrid accuracy",
    )

    assert_metric_close(
        hybrid_metrics[
            "macro_f1"
        ],
        EXPECTED_FROZEN_MACRO_F1,
        "Frozen hybrid Macro-F1",
    )

    print(
        "Frozen metric reproduction: PASSED"
    )

    # --------------------------------------------------------
    # Only materialize recovered predictions after all
    # validation checks have passed.
    # --------------------------------------------------------

    save_jsonl(
        RECOVERED_PREDICTIONS,
        recovered,
    )

    print()

    print(
        "Recovered predictions saved:"
    )

    print(
        " ",
        RECOVERED_PREDICTIONS,
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    result_payload = {
        "experiment":
            "si7a_frozen_selector_recovery",

        "status":
            "passed",

        "architecture":
            "context_routed_hybrid",

        "routing_rule": {
            "tokenizer_source":
                "MD4_DistilRoBERTa",

            "threshold":
                ROUTING_THRESHOLD,

            "le_threshold_model":
                "MD4_weighted_DistilRoBERTa",

            "gt_threshold_model":
                "MD3_ModernBERT",
        },

        "artifact_counts": {
            "md3_predictions":
                len(
                    md3_predictions
                ),

            "md4_predictions":
                len(
                    md4_predictions
                ),

            "short_md4":
                short_count,

            "long_md3":
                long_count,
        },

        "md3_metrics":
            md3_metrics,

        "md4_metrics":
            md4_metrics,

        "recovered_hybrid_metrics":
            hybrid_metrics,

        "expected_frozen_metrics": {
            "accuracy":
                EXPECTED_FROZEN_ACCURACY,

            "macro_f1":
                EXPECTED_FROZEN_MACRO_F1,
        },

        "recovered_predictions_file":
            str(
                RECOVERED_PREDICTIONS
            ),
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_payload,
            f,
            indent=2,
        )

    print()

    print(
        "Validation report saved:"
    )

    print(
        " ",
        RESULT_PATH,
    )

    print()
    print("=" * 76)
    print(
        "SI7-A FROZEN SELECTOR RECOVERY VALIDATION PASSED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()