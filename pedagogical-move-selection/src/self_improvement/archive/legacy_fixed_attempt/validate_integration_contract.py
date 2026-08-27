import json
from pathlib import Path

from src.self_improvement.attempt_level_bandit import (
    AttemptLevelBanditPolicy,
)

from src.self_improvement.attempt_experience_logger import (
    AttemptExperienceLogger,
)

from src.self_improvement.integrated_attempt_controller import (
    IntegratedTutoringAttemptController,
)


LOG_PATH = Path(
    "data/synthetic/self_improvement/"
    "integration_contract/"
    "synthetic_integrated_attempts.jsonl"
)


RESULT_PATH = Path(
    "results/self_improvement/"
    "si5c3_integration_contract_validation.json"
)


# ============================================================
# EXTERNAL COMPONENT STUBS
#
# These are NOT models.
#
# They only simulate APIs that other project components
# will eventually provide.
# ============================================================


class FrozenSelectorStub:

    version = (
        "frozen_supervised_selector_v1"
    )


    def __init__(
        self,
    ):

        self.call_count = 0


    def predict(
        self,
        problem,
        history,
    ):

        outputs = [
            {
                "generic": 0.08,
                "probing": 0.32,
                "focus": 0.37,
                "telling": 0.23,
            },

            {
                "generic": 0.09,
                "probing": 0.36,
                "focus": 0.35,
                "telling": 0.20,
            },

            {
                "generic": 0.07,
                "probing": 0.31,
                "focus": 0.38,
                "telling": 0.24,
            },
        ]


        result = outputs[
            self.call_count
            %
            len(
                outputs
            )
        ]


        self.call_count += 1


        return result


class TutorAgentStub:

    def __init__(
        self,
    ):

        self.received_moves = []


    def generate(
        self,
        problem,
        history,
        pedagogical_move,
    ):

        self.received_moves.append(
            pedagogical_move
        )


        return (
            f"Synthetic tutor response using "
            f"{pedagogical_move}."
        )


# ============================================================
# ONE COMPLETE ATTEMPT
# ============================================================


def run_attempt(
    controller,
    selector,
    tutor,
    episode_id,
    attempt_index,
    mastery_before,
    previous_evaluator_score,
    previous_mastery_delta,
    evaluator_score,
    mastery_after,
):

    problem = (
        "A recipe uses 3/4 cup for each batch. "
        "How much is needed for several batches?"
    )


    history = [
        {
            "user":
                "Student",

            "text":
                "I am not sure whether to "
                "add or multiply."
        }
    ]


    start = controller.start_attempt(
        episode_id=
            episode_id,

        attempt_index=
            attempt_index,

        problem=
            problem,

        mastery_before={
            "skill_id":
                "fraction_multiplication",

            "mastery":
                mastery_before,
        },

        previous_evaluator_score=
            previous_evaluator_score,

        previous_mastery_delta=
            previous_mastery_delta,
    )


    selected_arm = (
        start[
            "selected_arm"
        ]
    )


    updates_at_start = (
        controller.bandit_policy
        .get_total_updates()
    )


    student_replies = [
        "Maybe multiplication.",

        "Because the fraction is needed "
        "for each batch.",

        "I think I understand now.",
    ]


    final_moves = []


    for (
        turn_index,
        student_reply,
    ) in enumerate(
        student_replies
    ):

        # ====================================================
        # FROZEN SELECTOR
        # ====================================================

        base_probs = selector.predict(
            problem=
                problem,

            history=
                history,
        )


        # ====================================================
        # FIXED BANDIT OVERLAY
        # ====================================================

        decision = controller.select_move(
            history_before=
                history,

            base_move_probs=
                base_probs,
        )


        assert (
            decision[
                "selected_arm"
            ]
            ==
            selected_arm
        )


        final_move = (
            decision[
                "final_move"
            ]
        )


        final_moves.append(
            final_move
        )


        # ====================================================
        # EXTERNAL TUTOR AGENT
        # ====================================================

        teacher_response = tutor.generate(
            problem=
                problem,

            history=
                history,

            pedagogical_move=
                final_move,
        )


        # ====================================================
        # EXTERNAL STUDENT RESPONSE
        #
        # Synthetic stub only.
        # ====================================================

        controller.record_turn(
            teacher_response=
                teacher_response,

            student_response=
                student_reply,
        )


        history.append({
            "user":
                "Teacher",

            "text":
                teacher_response,
        })


        history.append({
            "user":
                "Student",

            "text":
                student_reply,
        })


        # ====================================================
        # ABSOLUTELY NO LEARNING MID-ATTEMPT
        # ====================================================

        assert (
            controller.bandit_policy
            .get_total_updates()
            ==
            updates_at_start
        )


    # ========================================================
    # ATTEMPT HAS NOW FINISHED
    #
    # External evaluator + mastery component provide outputs.
    # ========================================================

    result = controller.finish_attempt(
        evaluator_score=
            evaluator_score,

        mastery_after={
            "skill_id":
                "fraction_multiplication",

            "mastery":
                mastery_after,
        },
    )


    assert (
        controller.bandit_policy
        .get_total_updates()
        ==
        updates_at_start
        + 1
    )


    return {
        "selected_arm":
            selected_arm,

        "final_moves":
            final_moves,

        "result":
            result,
    }


def main():

    print(
        "=" * 70
    )

    print(
        "SI5-C3 - INTEGRATION CONTRACT VALIDATION"
    )

    print(
        "=" * 70
    )


    if LOG_PATH.exists():

        LOG_PATH.unlink()


    policy = AttemptLevelBanditPolicy(
        algorithm=
            "linucb",

        alpha=
            0.75,

        max_probability_gap=
            0.15,

        seed=
            42,

        data_origin=
            "synthetic",
    )


    logger = AttemptExperienceLogger(
        path=
            LOG_PATH,

        data_origin=
            "synthetic",
    )


    selector = (
        FrozenSelectorStub()
    )


    tutor = (
        TutorAgentStub()
    )


    controller = (
        IntegratedTutoringAttemptController(
            bandit_policy=
                policy,

            experience_logger=
                logger,

            base_selector_version=
                selector.version,
        )
    )


    # ========================================================
    # CONTRACT DESCRIPTION
    # ========================================================

    print(
        "\nIntegration flow:"
    )

    print(
        "Student/history"
    )

    print(
        "  -> Frozen selector"
    )

    print(
        "  -> Fixed bandit overlay"
    )

    print(
        "  -> Tutor agent"
    )

    print(
        "  -> Student"
    )

    print(
        "  -> repeat"
    )

    print(
        "  -> ATTEMPT FINISHES"
    )

    print(
        "  -> External evaluator + mastery"
    )

    print(
        "  -> ONE bandit update"
    )


    # ========================================================
    # ATTEMPT 0
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ATTEMPT 0"
    )

    print(
        "=" * 70
    )


    attempt_0 = run_attempt(
        controller=
            controller,

        selector=
            selector,

        tutor=
            tutor,

        episode_id=
            "synthetic_episode_001",

        attempt_index=
            0,

        mastery_before=
            0.40,

        previous_evaluator_score=
            None,

        previous_mastery_delta=
            None,

        evaluator_score=
            2,

        mastery_after=
            0.55,
    )


    print(
        "\nSelected arm:",
        attempt_0[
            "selected_arm"
        ]
    )

    print(
        "Final moves:",
        attempt_0[
            "final_moves"
        ]
    )

    print(
        "Bandit updates after attempt:",
        policy.get_total_updates()
    )

    print(
        "Evaluator:",
        attempt_0[
            "result"
        ][
            "bandit_result"
        ][
            "evaluator_score"
        ]
    )

    print(
        "Mastery delta:",
        f"{attempt_0['result']['bandit_result']['mastery_delta']:+.4f}"
    )


    assert (
        policy.get_total_updates()
        ==
        1
    )


    # ========================================================
    # ATTEMPT 1 - RETEACH
    #
    # Previous attempt feedback is now observable context.
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETEACH ATTEMPT 1"
    )

    print(
        "=" * 70
    )


    attempt_1 = run_attempt(
        controller=
            controller,

        selector=
            selector,

        tutor=
            tutor,

        episode_id=
            "synthetic_episode_001",

        attempt_index=
            1,

        mastery_before=
            0.55,

        previous_evaluator_score=
            2,

        previous_mastery_delta=
            0.15,

        evaluator_score=
            3,

        mastery_after=
            0.70,
    )


    print(
        "\nSelected arm:",
        attempt_1[
            "selected_arm"
        ]
    )

    print(
        "Final moves:",
        attempt_1[
            "final_moves"
        ]
    )

    print(
        "Bandit updates after attempt:",
        policy.get_total_updates()
    )

    print(
        "Evaluator:",
        attempt_1[
            "result"
        ][
            "bandit_result"
        ][
            "evaluator_score"
        ]
    )

    print(
        "Mastery delta:",
        f"{attempt_1['result']['bandit_result']['mastery_delta']:+.4f}"
    )


    assert (
        policy.get_total_updates()
        ==
        2
    )


    # ========================================================
    # VERIFY TUTOR RECEIVED FINAL MOVES
    # ========================================================

    expected_tutor_calls = (
        attempt_0[
            "final_moves"
        ]
        +
        attempt_1[
            "final_moves"
        ]
    )


    tutor_move_contract_passed = (
        tutor.received_moves
        ==
        expected_tutor_calls
    )


    # ========================================================
    # LOG VALIDATION
    # ========================================================

    lines = (
        LOG_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .splitlines()
    )


    records = [
        json.loads(
            line
        )

        for line
        in lines
    ]


    logging_passed = (
        len(
            records
        )
        ==
        2
    )


    same_episode_passed = (
        records[
            0
        ][
            "episode_id"
        ]
        ==
        records[
            1
        ][
            "episode_id"
        ]
        ==
        "synthetic_episode_001"
    )


    attempt_indices_passed = (
        records[
            0
        ][
            "attempt_index"
        ]
        ==
        0

        and

        records[
            1
        ][
            "attempt_index"
        ]
        ==
        1
    )


    feedback_context_passed = (
        records[
            0
        ][
            "attempt_context"
        ][
            "previous_evaluator_score"
        ]
        is None

        and

        records[
            1
        ][
            "attempt_context"
        ][
            "previous_evaluator_score"
        ]
        ==
        2
    )


    update_timing_passed = (
        records[
            0
        ][
            "bandit_update"
        ][
            "updates_before"
        ]
        ==
        0

        and

        records[
            0
        ][
            "bandit_update"
        ][
            "updates_after"
        ]
        ==
        1

        and

        records[
            1
        ][
            "bandit_update"
        ][
            "updates_before"
        ]
        ==
        1

        and

        records[
            1
        ][
            "bandit_update"
        ][
            "updates_after"
        ]
        ==
        2
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOG + INTERFACE VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        "\nLogged attempts:",
        len(
            records
        )
    )


    print(
        "Same episode / reteach indexing:",
        (
            "PASSED"
            if (
                same_episode_passed
                and
                attempt_indices_passed
            )
            else "FAILED"
        )
    )


    print(
        "Previous feedback enters next attempt only:",
        (
            "PASSED"
            if feedback_context_passed
            else "FAILED"
        )
    )


    print(
        "Tutor received final reranked moves:",
        (
            "PASSED"
            if tutor_move_contract_passed
            else "FAILED"
        )
    )


    print(
        "Bandit updates only after attempts:",
        (
            "PASSED"
            if update_timing_passed
            else "FAILED"
        )
    )


    all_passed = all(
        [
            logging_passed,
            same_episode_passed,
            attempt_indices_passed,
            feedback_context_passed,
            tutor_move_contract_passed,
            update_timing_passed,
        ]
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
                    "SI5-C3_integration_contract_validation",

                "simulation_only":
                    True,

                "real_external_components_used":
                    False,

                "algorithm":
                    "linucb",

                "attempts":
                    2,

                "bandit_updates":
                    policy.get_total_updates(),

                "tutor_move_contract_passed":
                    tutor_move_contract_passed,

                "feedback_context_passed":
                    feedback_context_passed,

                "update_timing_passed":
                    update_timing_passed,

                "logging_passed":
                    logging_passed,

                "overall_validation_passed":
                    all_passed,
            },
            f,
            indent=2,
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "SI5-C3 STATUS"
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
        "\nIMPORTANT:"
    )

    print(
        "External components are synthetic stubs only."
    )

    print(
        "This validates interfaces and timing, "
        "not educational performance."
    )


    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":

    main()
