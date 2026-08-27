from src.self_improvement.attempt_level_bandit import (
    AttemptLevelBanditPolicy,
    CONTEXT_FEATURE_NAMES,
)


def run_validation(
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


    policy = AttemptLevelBanditPolicy(
        algorithm=algorithm,

        # Development values only.
        alpha=0.75,
        lints_exploration_scale=0.50,
        max_probability_gap=0.15,

        seed=42,
    )


    # ========================================================
    # 1. NO UPDATE BEFORE AN ATTEMPT
    # ========================================================

    assert (
        policy.get_total_updates()
        ==
        0
    )


    # ========================================================
    # 2. START ATTEMPT
    # ========================================================

    start = policy.start_attempt(
        skill_id=
            "fraction_multiplication",

        mastery_before=
            0.40,

        attempt_index=
            0,

        previous_evaluator_score=
            None,

        previous_mastery_delta=
            None,
    )


    selected_arm = start[
        "selected_arm"
    ]


    print(
        "\nSelected arm:",
        selected_arm
    )


    assert (
        policy.get_total_updates()
        ==
        0
    )


    # ========================================================
    # 3. APPLY SAME LOCKED POLICY ACROSS MULTIPLE TURNS
    # ========================================================

    turn_probabilities = [
        {
            "generic": 0.08,
            "probing": 0.32,
            "focus": 0.37,
            "telling": 0.23,
        },

        {
            "generic": 0.10,
            "probing": 0.35,
            "focus": 0.33,
            "telling": 0.22,
        },

        {
            "generic": 0.07,
            "probing": 0.29,
            "focus": 0.38,
            "telling": 0.26,
        },
    ]


    returned_arms = []


    for turn_index, probs in enumerate(
        turn_probabilities
    ):

        result = policy.apply_overlay(
            probs
        )


        returned_arms.append(
            result[
                "selected_arm"
            ]
        )


        print(
            f"\nTurn {turn_index}"
        )

        print(
            "  Base move:",
            result[
                "base_move"
            ]
        )

        print(
            "  Final move:",
            result[
                "final_move"
            ]
        )

        print(
            "  Override:",
            result[
                "overridden"
            ]
        )


        # Absolutely no learning here.
        assert (
            policy.get_total_updates()
            ==
            0
        )


    assert all(
        arm
        ==
        selected_arm

        for arm
        in returned_arms
    )


    # ========================================================
    # 4. FINISH ATTEMPT
    #
    # ONLY NOW may update occur.
    # ========================================================

    finish = policy.finish_attempt(
        evaluator_score=2,
        mastery_after=0.55,
    )


    assert (
        policy.get_total_updates()
        ==
        1
    )


    assert (
        policy.get_skill_updates(
            "fraction_multiplication"
        )
        ==
        1
    )


    print(
        "\nAttempt finished"
    )

    print(
        "  Evaluator:",
        finish[
            "evaluator_score"
        ]
    )

    print(
        "  Reward:",
        finish[
            "reward"
        ]
    )

    print(
        "  Mastery delta:",
        f"{finish['mastery_delta']:+.4f}"
    )

    print(
        "  Updates:",
        policy.get_total_updates()
    )


    # ========================================================
    # 5. SECOND ATTEMPT
    #
    # Previous feedback is now observable context.
    # ========================================================

    policy.start_attempt(
        skill_id=
            "fraction_multiplication",

        mastery_before=
            0.55,

        attempt_index=
            1,

        previous_evaluator_score=
            2,

        previous_mastery_delta=
            0.15,
    )


    policy.apply_overlay(
        {
            "generic": 0.10,
            "probing": 0.34,
            "focus": 0.36,
            "telling": 0.20,
        }
    )


    # Still exactly one previous update.
    assert (
        policy.get_total_updates()
        ==
        1
    )


    policy.finish_attempt(
        evaluator_score=3,
        mastery_after=0.70,
    )


    assert (
        policy.get_total_updates()
        ==
        2
    )


    # ========================================================
    # 6. LIFECYCLE ERROR TESTS
    # ========================================================

    lifecycle_errors_passed = True


    try:

        policy.apply_overlay(
            {
                "generic": 0.25,
                "probing": 0.25,
                "focus": 0.25,
                "telling": 0.25,
            }
        )

        lifecycle_errors_passed = False

    except RuntimeError:

        pass


    try:

        policy.finish_attempt(
            evaluator_score=3,
            mastery_after=0.80,
        )

        lifecycle_errors_passed = False

    except RuntimeError:

        pass


    policy.start_attempt(
        skill_id=
            "basic_division",

        mastery_before=
            0.30,
    )


    try:

        policy.start_attempt(
            skill_id=
                "basic_division",

            mastery_before=
                0.30,
        )

        lifecycle_errors_passed = False

    except RuntimeError:

        pass


    # Close active attempt.
    policy.finish_attempt(
        evaluator_score=1,
        mastery_after=0.35,
    )


    print(
        "\nLifecycle guards:",
        (
            "PASSED"
            if lifecycle_errors_passed
            else "FAILED"
        )
    )


    assert lifecycle_errors_passed


    # ========================================================
    # 7. RESET TEST
    # ========================================================

    assert (
        policy.get_total_updates()
        ==
        3
    )


    policy.reset_state()


    reset_passed = (
        policy.get_total_updates()
        ==
        0

        and

        len(
            policy.bandits
        )
        ==
        0

        and

        policy.active_attempt
        is None
    )


    print(
        "Reset state:",
        (
            "PASSED"
            if reset_passed
            else "FAILED"
        )
    )


    assert reset_passed


    return {
        "algorithm":
            algorithm,

        "passed":
            True,
    }


def main():

    print(
        "=" * 70
    )

    print(
        "SI5-C1 - ATTEMPT-LEVEL BANDIT COMPONENT VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        "\nContext features available BEFORE attempt:"
    )


    for feature in CONTEXT_FEATURE_NAMES:

        print(
            "  -",
            feature
        )


    print(
        "\nCurrent-attempt evaluator_score in context: NO"
    )

    print(
        "Current-attempt mastery_after in context: NO"
    )

    print(
        "Mid-attempt learning allowed: NO"
    )


    results = []


    for algorithm in [
        "linucb",
        "lints",
    ]:

        results.append(
            run_validation(
                algorithm
            )
        )


    all_passed = all(
        result[
            "passed"
        ]

        for result
        in results
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "SI5-C1 STATUS"
    )

    print(
        "=" * 70
    )


    print(
        "\nLinUCB lifecycle:",
        "PASSED"
    )

    print(
        "LinTS lifecycle:",
        "PASSED"
    )

    print(
        "Overall:",
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
        "This validates software lifecycle only."
    )

    print(
        "No synthetic learned state should be carried "
        "into the real integrated experiment."
    )


if __name__ == "__main__":

    main()