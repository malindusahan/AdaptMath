import argparse
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DEFAULT_INPUT = Path(
    "data/synthetic/self_improvement/"
    "trajectory_schema_example.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/synthetic/self_improvement/"
    "trajectory_prefix_examples.jsonl"
)

RESULT_PATH = Path(
    "results/self_improvement/"
    "si4_trajectory_prefix_validation.json"
)


MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


# ============================================================
# HELPERS
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


def save_jsonl(
    path,
    rows,
):

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


# ============================================================
# BUILD PREFIX / DECISION EXAMPLES
# ============================================================

def build_prefix_examples(
    record
):

    episode_id = record[
        "episode_id"
    ]

    attempt_index = int(
        record[
            "attempt_index"
        ]
    )

    problem = record[
        "problem"
    ]

    mastery_before = record[
        "mastery_before"
    ]

    skill_id = mastery_before[
        "skill_id"
    ]

    mastery_before_value = float(
        mastery_before[
            "mastery"
        ]
    )


    feedback = record[
        "external_feedback"
    ]

    evaluator_score = int(
        feedback[
            "evaluator_score"
        ]
    )

    mastery_after = feedback[
        "mastery_after"
    ]

    mastery_after_value = float(
        mastery_after[
            "mastery"
        ]
    )


    # --------------------------------------------------------
    # Targets supplied indirectly by external components.
    # --------------------------------------------------------

    evaluator_rate = (
        evaluator_score
        / 3.0
    )

    session_success = (
        evaluator_score
        == 3
    )

    mastery_delta = (
        mastery_after_value
        -
        mastery_before_value
    )


    examples = []


    for turn in record[
        "trajectory"
    ]:

        selected_move = turn[
            "selected_move"
        ]


        assert (
            selected_move
            in MOVES
        )


        # ====================================================
        # IMPORTANT:
        #
        # This contains ONLY information available at the
        # exact time the pedagogical move was chosen.
        #
        # teacher_response is NOT input.
        # student_response is NOT input.
        # future turns are NOT input.
        # ====================================================

        outcome_model_input = {

            "problem":
                problem,

            "skill_id":
                skill_id,

            "mastery_before":
                mastery_before_value,

            "history":
                turn[
                    "history_before"
                ],

            "candidate_move":
                selected_move,
        }


        # ====================================================
        # Final attempt-level outcomes.
        #
        # These remain separate targets.
        # ====================================================

        targets = {

            "evaluator_score":
                evaluator_score,

            "evaluator_rate":
                evaluator_rate,

            "session_success":
                session_success,

            "mastery_after":
                mastery_after_value,

            "mastery_delta":
                mastery_delta,
        }


        # ====================================================
        # Metadata is retained for auditing.
        #
        # It is NOT automatically part of outcome-model input.
        # ====================================================

        metadata = {

            "episode_id":
                episode_id,

            "attempt_index":
                attempt_index,

            "turn_index":
                int(
                    turn[
                        "turn_index"
                    ]
                ),

            "policy_version":
                record[
                    "policy_version"
                ],

            "base_move_probs":
                turn[
                    "base_move_probs"
                ],

            # These occurred AFTER move selection.
            # Audit/reference only.
            "teacher_response":
                turn[
                    "teacher_response"
                ],

            "student_response":
                turn[
                    "student_response"
                ],
        }


        examples.append({

            "example_id":
                (
                    f"{episode_id}"
                    f"_attempt{attempt_index}"
                    f"_turn{turn['turn_index']}"
                ),

            "outcome_model_input":
                outcome_model_input,

            "targets":
                targets,

            "metadata":
                metadata,
        })


    return examples


# ============================================================
# VALIDATION
# ============================================================

def validate_examples(
    source_records,
    examples,
):

    errors = []


    expected_examples = sum(

        len(
            record[
                "trajectory"
            ]
        )

        for record
        in source_records
    )


    if (
        len(examples)
        != expected_examples
    ):

        errors.append(
            "Number of prefix examples does not "
            "equal total tutoring turns."
        )


    seen_ids = set()


    for index, example in enumerate(
        examples
    ):

        example_id = example[
            "example_id"
        ]


        if example_id in seen_ids:

            errors.append(
                f"Duplicate example_id: "
                f"{example_id}"
            )


        seen_ids.add(
            example_id
        )


        model_input = example[
            "outcome_model_input"
        ]


        # ----------------------------------------------------
        # Exact allowed input fields.
        # ----------------------------------------------------

        expected_input_fields = {

            "problem",
            "skill_id",
            "mastery_before",
            "history",
            "candidate_move",
        }


        if (
            set(
                model_input.keys()
            )
            != expected_input_fields
        ):

            errors.append(
                f"{example_id}: incorrect "
                "outcome-model input fields."
            )


        # ----------------------------------------------------
        # Explicit future-information exclusion.
        # ----------------------------------------------------

        forbidden_input_fields = {

            "teacher_response",
            "student_response",
            "evaluator_score",
            "mastery_after",
            "mastery_delta",
            "session_success",
            "ground_truth",
            "student_profile",
            "student_incorrect_solution",
            "self-correctness",
        }


        overlap = (

            set(
                model_input.keys()
            )

            &
            forbidden_input_fields
        )


        if overlap:

            errors.append(
                f"{example_id}: future/privileged "
                f"fields in model input: "
                f"{sorted(overlap)}"
            )


        if (
            model_input[
                "candidate_move"
            ]
            not in MOVES
        ):

            errors.append(
                f"{example_id}: invalid move."
            )


        mastery_before = model_input[
            "mastery_before"
        ]


        if not (
            0.0
            <= mastery_before
            <= 1.0
        ):

            errors.append(
                f"{example_id}: mastery_before "
                "outside [0,1]."
            )


        targets = example[
            "targets"
        ]


        score = targets[
            "evaluator_score"
        ]


        if score not in {
            0,
            1,
            2,
            3,
        }:

            errors.append(
                f"{example_id}: invalid "
                "evaluator score."
            )


        expected_rate = (
            score
            / 3.0
        )


        if abs(
            targets[
                "evaluator_rate"
            ]
            -
            expected_rate
        ) > 1e-8:

            errors.append(
                f"{example_id}: incorrect "
                "evaluator_rate."
            )


        expected_success = (
            score
            == 3
        )


        if (
            targets[
                "session_success"
            ]
            != expected_success
        ):

            errors.append(
                f"{example_id}: incorrect "
                "session_success."
            )


        expected_delta = (

            targets[
                "mastery_after"
            ]

            -

            mastery_before
        )


        if abs(
            targets[
                "mastery_delta"
            ]
            -
            expected_delta
        ) > 1e-8:

            errors.append(
                f"{example_id}: incorrect "
                "mastery_delta."
            )


    return errors


# ============================================================
# DISPLAY
# ============================================================

def display_example(
    example
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Example ID:",
        example[
            "example_id"
        ],
    )


    print(
        "\nOUTCOME-MODEL INPUT"
    )

    print(
        json.dumps(
            example[
                "outcome_model_input"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


    print(
        "\nTARGETS"
    )

    print(
        json.dumps(
            example[
                "targets"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


    print(
        "\nAUDIT METADATA"
    )

    print(
        json.dumps(
            example[
                "metadata"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )


    args = parser.parse_args()


    print("=" * 70)

    print(
        "SI4-B1 â€” TRAJECTORY PREFIX / "
        "DECISION EXAMPLE VALIDATION"
    )

    print("=" * 70)


    records = load_jsonl(
        args.input
    )


    print(
        "\nTrajectory records:",
        len(records),
    )


    all_examples = []


    for record in records:

        all_examples.extend(
            build_prefix_examples(
                record
            )
        )


    print(
        "Tutoring decision examples:",
        len(
            all_examples
        ),
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    errors = validate_examples(
        records,
        all_examples,
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREFIX DATASET VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        "\nValidation errors:",
        len(errors),
    )


    if errors:

        for error in errors[:30]:

            print(
                " -",
                error,
            )


        print(
            "\nVALIDATION: FAILED"
        )


    else:

        print(
            "\nVALIDATION: PASSED"
        )


    # ========================================================
    # EXAMPLES
    # ========================================================

    if not errors:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PREFIX EXAMPLES"
        )

        print(
            "=" * 70
        )


        for example in all_examples[:3]:

            display_example(
                example
            )


        save_jsonl(
            args.output,
            all_examples,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    result = {

        "experiment":
            "SI4-B1_trajectory_prefix_validation",

        "source":
            str(
                args.input
            ),

        "trajectory_records":
            len(records),

        "decision_examples":
            len(
                all_examples
            ),

        "validation_errors":
            len(errors),

        "validation_passed":
            len(errors) == 0,

        "outcome_model_input_fields": [

            "problem",
            "skill_id",
            "mastery_before",
            "history",
            "candidate_move",
        ],

        "targets": [

            "evaluator_score",
            "evaluator_rate",
            "session_success",
            "mastery_after",
            "mastery_delta",
        ],
    }


    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


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
        "CREDIT-ASSIGNMENT CONTRACT"
    )

    print(
        "=" * 70
    )


    print(
        "\nOne completed attempt produces "
        "one example per move decision."
    )

    print(
        "Each example receives the eventual "
        "attempt-level external outcomes."
    )

    print(
        "This is outcome association, "
        "NOT a causal label for the move."
    )

    print(
        "Teacher/student responses after the "
        "decision are excluded from that "
        "decision's model input."
    )

    print(
        "Evaluator and mastery remain "
        "separate prediction targets."
    )


    print(
        "\nSaved summary:"
    )

    print(
        RESULT_PATH
    )


    if not errors:

        print(
            "\nSaved prefix examples:"
        )

        print(
            args.output
        )


if __name__ == "__main__":

    main()
