import argparse
import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


SYNTHETIC_PATH = Path(
    "data/synthetic/self_improvement/"
    "trajectory_schema_example.jsonl"
)

RESULT_PATH = Path(
    "results/self_improvement/"
    "si4_trajectory_schema_validation.json"
)


# ============================================================
# PRIVILEGED / DATASET-ONLY FIELDS
#
# These must never become runtime inputs to this component.
# ============================================================

FORBIDDEN_FIELDS = {
    "ground_truth",
    "student_incorrect_solution",
    "student_profile",
    "teacher_described_confusion",
    "self-correctness",
    "self-typical-confusion",
    "self-typical-interactions",
}


# ============================================================
# SYNTHETIC EXAMPLE
#
# This is ONLY an interface test.
# It must never be used for training.
# ============================================================

EXAMPLE_RECORD = {

    "episode_id":
        "synthetic_episode_001",

    "attempt_index":
        0,

    "problem":
        (
            "A recipe uses 3/4 cup of flour per batch. "
            "How much flour is needed for 4 batches?"
        ),

    "mastery_before": {

        "skill_id":
            "fraction_multiplication",

        "mastery":
            0.42,
    },

    "policy_version":
        "frozen_supervised_selector_v1",

    "trajectory": [

        {
            "turn_index":
                0,

            "history_before": [

                {
                    "user":
                        "Student",

                    "text":
                        (
                            "I am not sure whether I should "
                            "add or multiply."
                        ),
                }
            ],

            "base_move_probs": {

                "generic":
                    0.08,

                "probing":
                    0.34,

                "focus":
                    0.48,

                "telling":
                    0.10,
            },

            "selected_move":
                "focus",

            "teacher_response":
                (
                    "You know how much flour one batch uses. "
                    "What changes when you make four batches?"
                ),

            "student_response":
                (
                    "The amount is repeated four times, "
                    "so I think I should multiply."
                ),
        },

        {
            "turn_index":
                1,

            "history_before": [

                {
                    "user":
                        "Student",

                    "text":
                        (
                            "I am not sure whether I should "
                            "add or multiply."
                        ),
                },

                {
                    "user":
                        "Teacher",

                    "text":
                        (
                            "You know how much flour one batch uses. "
                            "What changes when you make four batches?"
                        ),
                },

                {
                    "user":
                        "Student",

                    "text":
                        (
                            "The amount is repeated four times, "
                            "so I think I should multiply."
                        ),
                },
            ],

            "base_move_probs": {

                "generic":
                    0.06,

                "probing":
                    0.51,

                "focus":
                    0.32,

                "telling":
                    0.11,
            },

            "selected_move":
                "probing",

            "teacher_response":
                (
                    "How would you write that multiplication "
                    "using 3/4 and 4?"
                ),

            "student_response":
                (
                    "I would write 3/4 times 4."
                ),
        },
    ],

    "external_feedback": {

        "evaluator_score":
            3,

        "mastery_after": {

            "skill_id":
                "fraction_multiplication",

            "mastery":
                0.67,
        },
    },
}


# ============================================================
# FILE HELPERS
# ============================================================

def write_jsonl(
    path,
    records,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path):

    records = []

    with path.open(
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

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                )

    return records


# ============================================================
# RECURSIVE FORBIDDEN-FIELD CHECK
# ============================================================

def find_forbidden_fields(
    value,
    path="record",
):

    violations = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            child_path = (
                f"{path}.{key}"
            )

            if key in FORBIDDEN_FIELDS:

                violations.append(
                    child_path
                )

            violations.extend(
                find_forbidden_fields(
                    child,
                    child_path,
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):

            violations.extend(
                find_forbidden_fields(
                    child,
                    f"{path}[{index}]",
                )
            )

    return violations


# ============================================================
# BASIC VALIDATORS
# ============================================================

def validate_mastery(
    value,
    location,
    errors,
):

    if not isinstance(
        value,
        dict,
    ):

        errors.append(
            f"{location} must be an object."
        )

        return


    if set(
        value.keys()
    ) != {
        "skill_id",
        "mastery",
    }:

        errors.append(
            f"{location} must contain exactly "
            "'skill_id' and 'mastery'."
        )

        return


    skill_id = value[
        "skill_id"
    ]

    mastery = value[
        "mastery"
    ]


    if (
        not isinstance(
            skill_id,
            str,
        )
        or
        not skill_id.strip()
    ):

        errors.append(
            f"{location}.skill_id must be "
            "a non-empty string."
        )


    if (
        isinstance(
            mastery,
            bool,
        )
        or
        not isinstance(
            mastery,
            (int, float),
        )
    ):

        errors.append(
            f"{location}.mastery must be numeric."
        )

    elif not (
        0.0
        <= float(mastery)
        <= 1.0
    ):

        errors.append(
            f"{location}.mastery must be "
            "between 0 and 1."
        )


def validate_history(
    history,
    location,
    errors,
):

    if not isinstance(
        history,
        list,
    ):

        errors.append(
            f"{location} must be a list."
        )

        return


    for index, turn in enumerate(
        history
    ):

        turn_location = (
            f"{location}[{index}]"
        )


        if not isinstance(
            turn,
            dict,
        ):

            errors.append(
                f"{turn_location} must be an object."
            )

            continue


        if set(
            turn.keys()
        ) != {
            "user",
            "text",
        }:

            errors.append(
                f"{turn_location} must contain exactly "
                "'user' and 'text'."
            )

            continue


        if turn[
            "user"
        ] not in {
            "Student",
            "Teacher",
        }:

            errors.append(
                f"{turn_location}.user must be "
                "'Student' or 'Teacher'."
            )


        if (
            not isinstance(
                turn[
                    "text"
                ],
                str,
            )
            or
            not turn[
                "text"
            ].strip()
        ):

            errors.append(
                f"{turn_location}.text must be "
                "a non-empty string."
            )


def validate_probabilities(
    probabilities,
    location,
    errors,
):

    if not isinstance(
        probabilities,
        dict,
    ):

        errors.append(
            f"{location} must be an object."
        )

        return


    if set(
        probabilities.keys()
    ) != set(
        MOVES
    ):

        errors.append(
            f"{location} must contain exactly: "
            f"{MOVES}"
        )

        return


    total = 0.0


    for move in MOVES:

        value = probabilities[
            move
        ]


        if (
            isinstance(
                value,
                bool,
            )
            or
            not isinstance(
                value,
                (int, float),
            )
        ):

            errors.append(
                f"{location}.{move} "
                "must be numeric."
            )

            continue


        value = float(
            value
        )


        if not (
            0.0
            <= value
            <= 1.0
        ):

            errors.append(
                f"{location}.{move} must be "
                "between 0 and 1."
            )


        total += value


    if abs(
        total - 1.0
    ) > 1e-4:

        errors.append(
            f"{location} probabilities must sum "
            f"to 1.0; found {total:.6f}."
        )


# ============================================================
# RECORD VALIDATOR
# ============================================================

def validate_record(
    record,
    record_index,
):

    errors = []

    prefix = (
        f"record[{record_index}]"
    )


    required_keys = {

        "episode_id",
        "attempt_index",
        "problem",
        "mastery_before",
        "policy_version",
        "trajectory",
        "external_feedback",
    }


    if not isinstance(
        record,
        dict,
    ):

        return [
            f"{prefix} must be an object."
        ]


    missing = (
        required_keys
        -
        set(record.keys())
    )


    if missing:

        errors.append(
            f"{prefix} missing keys: "
            f"{sorted(missing)}"
        )


        return errors


    # --------------------------------------------------------
    # No privileged MathDial-only fields anywhere.
    # --------------------------------------------------------

    forbidden = (
        find_forbidden_fields(
            record,
            prefix,
        )
    )


    for location in forbidden:

        errors.append(
            f"Forbidden runtime field: "
            f"{location}"
        )


    # --------------------------------------------------------
    # Episode metadata
    # --------------------------------------------------------

    if (
        not isinstance(
            record[
                "episode_id"
            ],
            str,
        )
        or
        not record[
            "episode_id"
        ].strip()
    ):

        errors.append(
            f"{prefix}.episode_id must be "
            "a non-empty string."
        )


    attempt_index = record[
        "attempt_index"
    ]


    if (
        isinstance(
            attempt_index,
            bool,
        )
        or
        not isinstance(
            attempt_index,
            int,
        )
        or
        attempt_index < 0
    ):

        errors.append(
            f"{prefix}.attempt_index must be "
            "an integer >= 0."
        )


    if (
        not isinstance(
            record[
                "problem"
            ],
            str,
        )
        or
        not record[
            "problem"
        ].strip()
    ):

        errors.append(
            f"{prefix}.problem must be "
            "a non-empty string."
        )


    if (
        not isinstance(
            record[
                "policy_version"
            ],
            str,
        )
        or
        not record[
            "policy_version"
        ].strip()
    ):

        errors.append(
            f"{prefix}.policy_version must be "
            "a non-empty string."
        )


    # --------------------------------------------------------
    # Mastery before
    # --------------------------------------------------------

    validate_mastery(
        record[
            "mastery_before"
        ],
        f"{prefix}.mastery_before",
        errors,
    )


    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    trajectory = record[
        "trajectory"
    ]


    if (
        not isinstance(
            trajectory,
            list,
        )
        or
        len(
            trajectory
        ) == 0
    ):

        errors.append(
            f"{prefix}.trajectory must contain "
            "at least one tutoring turn."
        )

    else:

        previous_turn = None


        for turn_index, turn in enumerate(
            trajectory
        ):

            location = (
                f"{prefix}.trajectory"
                f"[{turn_index}]"
            )


            required_turn_keys = {

                "turn_index",
                "history_before",
                "base_move_probs",
                "selected_move",
                "teacher_response",
                "student_response",
            }


            if not isinstance(
                turn,
                dict,
            ):

                errors.append(
                    f"{location} must be an object."
                )

                continue


            missing_turn_keys = (

                required_turn_keys
                -
                set(
                    turn.keys()
                )
            )


            if missing_turn_keys:

                errors.append(
                    f"{location} missing keys: "
                    f"{sorted(missing_turn_keys)}"
                )

                continue


            if (
                turn[
                    "turn_index"
                ]
                != turn_index
            ):

                errors.append(
                    f"{location}.turn_index should "
                    f"be {turn_index}."
                )


            validate_history(
                turn[
                    "history_before"
                ],
                f"{location}.history_before",
                errors,
            )


            validate_probabilities(
                turn[
                    "base_move_probs"
                ],
                f"{location}.base_move_probs",
                errors,
            )


            if (
                turn[
                    "selected_move"
                ]
                not in MOVES
            ):

                errors.append(
                    f"{location}.selected_move "
                    f"must be one of {MOVES}."
                )


            for field in [
                "teacher_response",
                "student_response",
            ]:

                if (
                    not isinstance(
                        turn[
                            field
                        ],
                        str,
                    )
                    or
                    not turn[
                        field
                    ].strip()
                ):

                    errors.append(
                        f"{location}.{field} must "
                        "be a non-empty string."
                    )


            # =================================================
            # HISTORY CONTINUITY
            #
            # At turn t+1, history must equal:
            #
            # history_before(t)
            # + teacher_response(t)
            # + student_response(t)
            #
            # This ensures future turns were not accidentally
            # inserted into earlier states.
            # =================================================

            if (
                previous_turn
                is not None
            ):

                expected_history = (

                    previous_turn[
                        "history_before"
                    ]

                    + [

                        {
                            "user":
                                "Teacher",

                            "text":
                                previous_turn[
                                    "teacher_response"
                                ],
                        },

                        {
                            "user":
                                "Student",

                            "text":
                                previous_turn[
                                    "student_response"
                                ],
                        },
                    ]
                )


                if (
                    turn[
                        "history_before"
                    ]
                    != expected_history
                ):

                    errors.append(
                        f"{location}.history_before "
                        "does not continue exactly "
                        "from the previous tutoring turn."
                    )


            previous_turn = turn


    # --------------------------------------------------------
    # External feedback
    # --------------------------------------------------------

    feedback = record[
        "external_feedback"
    ]


    if not isinstance(
        feedback,
        dict,
    ):

        errors.append(
            f"{prefix}.external_feedback "
            "must be an object."
        )

    else:

        if set(
            feedback.keys()
        ) != {
            "evaluator_score",
            "mastery_after",
        }:

            errors.append(
                f"{prefix}.external_feedback must "
                "contain exactly evaluator_score "
                "and mastery_after."
            )

        else:

            score = feedback[
                "evaluator_score"
            ]


            if (
                isinstance(
                    score,
                    bool,
                )
                or
                not isinstance(
                    score,
                    int,
                )
                or
                score not in {
                    0,
                    1,
                    2,
                    3,
                }
            ):

                errors.append(
                    f"{prefix}.external_feedback."
                    "evaluator_score must be "
                    "0, 1, 2, or 3."
                )


            validate_mastery(
                feedback[
                    "mastery_after"
                ],
                (
                    f"{prefix}.external_feedback."
                    "mastery_after"
                ),
                errors,
            )


            # --------------------------------------------
            # Same skill must be measured before/after.
            # --------------------------------------------

            before = record[
                "mastery_before"
            ]

            after = feedback[
                "mastery_after"
            ]


            if (
                isinstance(
                    before,
                    dict,
                )
                and
                isinstance(
                    after,
                    dict,
                )
                and
                "skill_id" in before
                and
                "skill_id" in after
                and
                before[
                    "skill_id"
                ]
                !=
                after[
                    "skill_id"
                ]
            ):

                errors.append(
                    f"{prefix}: mastery_before and "
                    "mastery_after refer to different "
                    "skill IDs."
                )


    return errors


# ============================================================
# DERIVED OUTCOME SUMMARY
#
# These are calculated for analysis only.
# They are NOT extra external component requirements.
# ============================================================

def derive_outcomes(
    record,
):

    score = record[
        "external_feedback"
    ][
        "evaluator_score"
    ]


    mastery_before = float(
        record[
            "mastery_before"
        ][
            "mastery"
        ]
    )


    mastery_after = float(
        record[
            "external_feedback"
        ][
            "mastery_after"
        ][
            "mastery"
        ]
    )


    return {

        "episode_id":
            record[
                "episode_id"
            ],

        "attempt_index":
            record[
                "attempt_index"
            ],

        "skill_id":
            record[
                "mastery_before"
            ][
                "skill_id"
            ],

        "num_tutoring_turns":
            len(
                record[
                    "trajectory"
                ]
            ),

        "evaluator_score":
            score,

        "evaluation_rate":
            score / 3.0,

        "requires_reteach":
            score < 3,

        "mastery_before":
            mastery_before,

        "mastery_after":
            mastery_after,

        "mastery_delta":
            mastery_after
            -
            mastery_before,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file containing completed "
            "trajectory records. If omitted, a synthetic "
            "interface example is generated and validated."
        ),
    )


    args = parser.parse_args()


    print("=" * 70)

    print(
        "SI4-A â€” TRAJECTORY RECORD SCHEMA VALIDATION"
    )

    print("=" * 70)


    if args.path is None:

        print(
            "\nNo integrated system file supplied."
        )

        print(
            "Creating SYNTHETIC interface example only."
        )


        write_jsonl(
            SYNTHETIC_PATH,
            [
                EXAMPLE_RECORD
            ],
        )


        source_path = (
            SYNTHETIC_PATH
        )


        print(
            "\nSynthetic file:"
        )

        print(
            source_path
        )

    else:

        source_path = (
            args.path
        )


    records = load_jsonl(
        source_path
    )


    print(
        "\nRecords loaded:",
        len(records),
    )


    all_errors = []


    for index, record in enumerate(
        records
    ):

        errors = validate_record(
            record,
            index,
        )


        all_errors.extend(
            errors
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "SCHEMA VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        "\nValidation errors:",
        len(
            all_errors
        ),
    )


    if all_errors:

        for error in all_errors[
            :30
        ]:

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


    # --------------------------------------------------------
    # Derived outcomes only if schema passed
    # --------------------------------------------------------

    derived = []


    if not all_errors:

        derived = [

            derive_outcomes(
                record
            )

            for record
            in records
        ]


        print(
            "\n"
            + "=" * 70
        )

        print(
            "DERIVED FEEDBACK EXAMPLE"
        )

        print(
            "=" * 70
        )


        for row in derived[
            :3
        ]:

            print(
                "\n"
                + json.dumps(
                    row,
                    indent=2,
                )
            )


    # --------------------------------------------------------
    # Interface summary
    # --------------------------------------------------------

    summary = {

        "experiment":
            "SI4-A_trajectory_schema_validation",

        "source_path":
            str(
                source_path
            ),

        "synthetic_only":
            args.path is None,

        "records":
            len(records),

        "validation_errors":
            len(
                all_errors
            ),

        "validation_passed":
            len(
                all_errors
            ) == 0,

        "forbidden_runtime_fields":
            sorted(
                FORBIDDEN_FIELDS
            ),

        "derived_examples":
            derived[
                :3
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
            summary,
            f,
            indent=2,
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "INTERFACE CONTRACT"
    )

    print(
        "=" * 70
    )


    print(
        "\nOne record = one tutoring attempt "
        "between evaluator checkpoints."
    )

    print(
        "Evaluator and mastery values are "
        "EXTERNAL inputs."
    )

    print(
        "Evaluator question generation is "
        "NOT part of this component."
    )

    print(
        "Mastery estimation is "
        "NOT part of this component."
    )

    print(
        "Failed attempts may share episode_id "
        "with later reteaching attempts."
    )


    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":

    main()
