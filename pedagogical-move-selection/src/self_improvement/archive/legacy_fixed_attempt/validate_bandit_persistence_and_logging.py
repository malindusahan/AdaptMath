import json
from pathlib import Path

from src.self_improvement.attempt_level_bandit import (
    AttemptLevelBanditPolicy,
)

from src.self_improvement.attempt_experience_logger import (
    AttemptExperienceLogger,
    SCHEMA_VERSION,
)


STATE_DIR = Path(
    "data/synthetic/self_improvement/"
    "bandit_state_validation"
)


LOG_DIR = Path(
    "data/synthetic/self_improvement/"
    "bandit_log_validation"
)


RESULT_PATH = Path(
    "results/self_improvement/"
    "si5c2_persistence_logging_validation.json"
)


def make_policy(
    algorithm,
    data_origin,
):

    return AttemptLevelBanditPolicy(
        algorithm=
            algorithm,

        alpha=
            0.75,

        lints_exploration_scale=
            0.50,

        max_probability_gap=
            0.15,

        seed=
            42,

        data_origin=
            data_origin,
    )


def run_attempt(
    policy,
):

    start = policy.start_attempt(
        skill_id=
            "fraction_multiplication",

        mastery_before=
            0.40,

        attempt_index=
            0,
    )


    selected_arm = (
        start[
            "selected_arm"
        ]
    )


    turns = []


    synthetic_turns = [
        {
            "history_before": [
                {
                    "user": "Student",
                    "text": "I am not sure."
                }
            ],

            "base_move_probs": {
                "generic": 0.08,
                "probing": 0.32,
                "focus": 0.37,
                "telling": 0.23,
            },

            "teacher_response":
                "What operation do you think "
                "the problem needs?",

            "student_response":
                "Maybe multiplication.",
        },

        {
            "history_before": [
                {
                    "user": "Student",
                    "text": "I am not sure."
                },

                {
                    "user": "Teacher",
                    "text": "What operation do you "
                            "think the problem needs?"
                },

                {
                    "user": "Student",
                    "text": "Maybe multiplication."
                },
            ],

            "base_move_probs": {
                "generic": 0.09,
                "probing": 0.36,
                "focus": 0.35,
                "telling": 0.20,
            },

            "teacher_response":
                "Why does multiplication make sense?",

            "student_response":
                "Because the fraction is used "
                "for each group.",
        },
    ]


    for (
        turn_index,
        item,
    ) in enumerate(
        synthetic_turns
    ):

        decision = (
            policy.apply_overlay(
                item[
                    "base_move_probs"
                ]
            )
        )


        turns.append({
            "turn_index":
                turn_index,

            "history_before":
                item[
                    "history_before"
                ],

            "base_move_probs":
                decision[
                    "base_move_probs"
                ],

            "base_move":
                decision[
                    "base_move"
                ],

            "final_move":
                decision[
                    "final_move"
                ],

            "overridden":
                decision[
                    "overridden"
                ],

            "teacher_response":
                item[
                    "teacher_response"
                ],

            "student_response":
                item[
                    "student_response"
                ],
        })


    finish = policy.finish_attempt(
        evaluator_score=
            2,

        mastery_after=
            0.55,
    )


    return (
        selected_arm,
        turns,
        finish,
    )


def validate_algorithm(
    algorithm,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"VALIDATING {algorithm.upper()}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # SYNTHETIC POLICY
    # ========================================================

    policy = make_policy(
        algorithm=
            algorithm,

        data_origin=
            "synthetic",
    )


    (
        selected_arm,
        turns,
        finish,
    ) = run_attempt(
        policy
    )


    assert (
        policy.get_total_updates()
        ==
        1
    )


    # ========================================================
    # SAVE SYNTHETIC STATE
    # ========================================================

    state_path = (
        STATE_DIR
        /
        f"{algorithm}_synthetic_state.json"
    )


    policy.save_state(
        state_path
    )


    print(
        "\nSaved synthetic state:",
        state_path
    )


    # ========================================================
    # LOAD INTO ANOTHER SYNTHETIC POLICY
    # ========================================================

    restored = make_policy(
        algorithm=
            algorithm,

        data_origin=
            "synthetic",
    )


    load_result = (
        restored.load_state(
            state_path
        )
    )


    persistence_passed = (
        restored.get_total_updates()
        ==
        policy.get_total_updates()

        and

        restored.get_skill_updates(
            "fraction_multiplication"
        )
        ==
        1
    )


    print(
        "Synthetic state reload:",
        (
            "PASSED"
            if persistence_passed
            else "FAILED"
        )
    )


    assert persistence_passed


    # ========================================================
    # CRITICAL TEST:
    #
    # REAL POLICY MUST REFUSE SYNTHETIC STATE.
    # ========================================================

    real_policy = make_policy(
        algorithm=
            algorithm,

        data_origin=
            "real",
    )


    synthetic_to_real_blocked = False


    try:

        real_policy.load_state(
            state_path
        )

    except ValueError as error:

        if (
            "data-origin mismatch"
            in str(
                error
            ).lower()
        ):

            synthetic_to_real_blocked = True


    print(
        "Synthetic -> real load blocked:",
        (
            "PASSED"
            if synthetic_to_real_blocked
            else "FAILED"
        )
    )


    assert (
        synthetic_to_real_blocked
    )


    # ========================================================
    # ATTEMPT LOGGING
    # ========================================================

    log_path = (
        LOG_DIR
        /
        f"{algorithm}_synthetic_attempts.jsonl"
    )


    # Clean validation artifact so repeated
    # validation runs stay deterministic.
    if log_path.exists():

        log_path.unlink()


    logger = AttemptExperienceLogger(
        path=
            log_path,

        data_origin=
            "synthetic",
    )


    record = {
        "schema_version":
            SCHEMA_VERSION,

        "data_origin":
            "synthetic",

        "episode_id":
            "synthetic_episode_001",

        "attempt_index":
            0,

        "problem":
            (
                "A recipe uses 3/4 cup per batch. "
                "How much is needed for multiple batches?"
            ),

        "mastery_before": {
            "skill_id":
                "fraction_multiplication",

            "mastery":
                0.40,
        },

        "policy": {
            "base_selector_version":
                "frozen_supervised_selector_v1",

            "bandit_algorithm":
                algorithm,

            "selected_arm":
                selected_arm,

            "max_probability_gap":
                0.15,
        },

        "trajectory":
            turns,

        "external_feedback": {
            "evaluator_score":
                finish[
                    "evaluator_score"
                ],

            "evaluator_rate":
                finish[
                    "reward"
                ],

            "mastery_after": {
                "skill_id":
                    "fraction_multiplication",

                "mastery":
                    finish[
                        "mastery_after"
                    ],
            },

            "mastery_delta":
                finish[
                    "mastery_delta"
                ],
        },

        "bandit_update": {
            "reward":
                finish[
                    "reward"
                ],

            "updates_after":
                finish[
                    "bandit_updates_after"
                ],
        },
    }


    logger.append(
        record
    )


    lines = (
        log_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .splitlines()
    )


    logging_passed = (
        len(
            lines
        )
        ==
        1
    )


    loaded_record = json.loads(
        lines[
            0
        ]
    )


    logging_passed = (
        logging_passed

        and

        loaded_record[
            "data_origin"
        ]
        ==
        "synthetic"

        and

        loaded_record[
            "policy"
        ][
            "bandit_algorithm"
        ]
        ==
        algorithm

        and

        loaded_record[
            "external_feedback"
        ][
            "evaluator_score"
        ]
        ==
        2

        and

        len(
            loaded_record[
                "trajectory"
            ]
        )
        ==
        2
    )


    print(
        "Attempt logging:",
        (
            "PASSED"
            if logging_passed
            else "FAILED"
        )
    )


    assert logging_passed


    # ========================================================
    # LOGGER ORIGIN BARRIER
    # ========================================================

    real_logger = (
        AttemptExperienceLogger(
            path=
                LOG_DIR
                /
                f"{algorithm}_real_validation.jsonl",

            data_origin=
                "real",
        )
    )


    logger_origin_blocked = False


    try:

        real_logger.append(
            record
        )

    except ValueError:

        logger_origin_blocked = True


    print(
        "Synthetic record -> real logger blocked:",
        (
            "PASSED"
            if logger_origin_blocked
            else "FAILED"
        )
    )


    assert (
        logger_origin_blocked
    )


    return {
        "algorithm":
            algorithm,

        "persistence_passed":
            bool(
                persistence_passed
            ),

        "synthetic_to_real_state_blocked":
            bool(
                synthetic_to_real_blocked
            ),

        "attempt_logging_passed":
            bool(
                logging_passed
            ),

        "synthetic_to_real_log_blocked":
            bool(
                logger_origin_blocked
            ),
    }


def main():

    print(
        "=" * 70
    )

    print(
        "SI5-C2 - BANDIT PERSISTENCE + LOGGING VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        "\nSynthetic state may load into synthetic policy: YES"
    )

    print(
        "Synthetic state may load into real policy: NO"
    )

    print(
        "Synthetic records may enter real log: NO"
    )


    results = []


    for algorithm in [
        "linucb",
        "lints",
    ]:

        results.append(
            validate_algorithm(
                algorithm
            )
        )


    all_passed = all(
        result[
            "persistence_passed"
        ]

        and

        result[
            "synthetic_to_real_state_blocked"
        ]

        and

        result[
            "attempt_logging_passed"
        ]

        and

        result[
            "synthetic_to_real_log_blocked"
        ]

        for result
        in results
    )


    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "experiment":
                    "SI5-C2_persistence_logging_validation",

                "simulation_only":
                    True,

                "algorithms":
                    results,

                "synthetic_real_state_separation":
                    True,

                "synthetic_real_log_separation":
                    True,

                "overall_validation_passed":
                    bool(
                        all_passed
                    ),
            },
            f,
            indent=2,
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "SI5-C2 STATUS"
    )

    print(
        "=" * 70
    )


    print(
        "\nOverall:",
        (
            "PASSED"
            if all_passed
            else "FAILED"
        )
    )


    print(
        "\nResearch data rule:"
    )

    print(
        "Synthetic state and synthetic attempt logs "
        "are isolated from real research state."
    )


    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":

    main()
