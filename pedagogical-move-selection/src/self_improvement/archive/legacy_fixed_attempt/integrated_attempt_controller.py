from __future__ import annotations

from copy import deepcopy


class IntegratedTutoringAttemptController:

    def __init__(
        self,
        bandit_policy,
        experience_logger,
        base_selector_version,
    ):

        self.bandit_policy = bandit_policy

        self.experience_logger = (
            experience_logger
        )

        self.base_selector_version = str(
            base_selector_version
        )

        if (
            self.bandit_policy.data_origin
            !=
            self.experience_logger.data_origin
        ):
            raise ValueError(
                "Bandit policy and experience logger "
                "must use the same data_origin."
            )

        self.active_record = None

        self.pending_turn = None

        self.updates_at_attempt_start = None


    # ========================================================
    # START ATTEMPT
    #
    # Called BEFORE tutoring begins.
    #
    # No current-attempt evaluator result exists yet.
    # No mastery_after exists yet.
    # ========================================================

    def start_attempt(
        self,
        episode_id,
        attempt_index,
        problem,
        mastery_before,
        previous_evaluator_score=None,
        previous_mastery_delta=None,
    ):

        if self.active_record is not None:

            raise RuntimeError(
                "An attempt is already active."
            )


        if self.pending_turn is not None:

            raise RuntimeError(
                "Unexpected pending turn."
            )


        episode_id = str(
            episode_id
        ).strip()

        if not episode_id:

            raise ValueError(
                "episode_id cannot be empty."
            )


        problem = str(
            problem
        ).strip()

        if not problem:

            raise ValueError(
                "problem cannot be empty."
            )


        if not isinstance(
            mastery_before,
            dict,
        ):

            raise ValueError(
                "mastery_before must be a dictionary."
            )


        skill_id = str(
            mastery_before.get(
                "skill_id",
                "",
            )
        ).strip()


        if not skill_id:

            raise ValueError(
                "mastery_before.skill_id is required."
            )


        if "mastery" not in mastery_before:

            raise ValueError(
                "mastery_before.mastery is required."
            )


        mastery_value = float(
            mastery_before[
                "mastery"
            ]
        )


        bandit_start = (
            self.bandit_policy.start_attempt(
                skill_id=
                    skill_id,

                mastery_before=
                    mastery_value,

                attempt_index=
                    attempt_index,

                previous_evaluator_score=
                    previous_evaluator_score,

                previous_mastery_delta=
                    previous_mastery_delta,
            )
        )


        self.updates_at_attempt_start = (
            self.bandit_policy.get_total_updates()
        )


        self.active_record = {
            "schema_version":
                "self_improvement_attempt_v1",

            "data_origin":
                self.bandit_policy.data_origin,

            "episode_id":
                episode_id,

            "attempt_index":
                int(
                    attempt_index
                ),

            "problem":
                problem,

            "mastery_before": {
                "skill_id":
                    skill_id,

                "mastery":
                    mastery_value,
            },

            # Previous attempt information is observable
            # BEFORE the current attempt starts.
            "attempt_context": {
                "previous_evaluator_score":
                    previous_evaluator_score,

                "previous_mastery_delta":
                    previous_mastery_delta,
            },

            "policy": {
                "base_selector_version":
                    self.base_selector_version,

                "bandit_algorithm":
                    self.bandit_policy.algorithm,

                "selected_arm":
                    bandit_start[
                        "selected_arm"
                    ],

                "max_probability_gap":
                    self.bandit_policy.max_probability_gap,
            },

            "trajectory": [],

            "external_feedback": None,

            "bandit_update": None,
        }


        return {
            "episode_id":
                episode_id,

            "attempt_index":
                int(
                    attempt_index
                ),

            "skill_id":
                skill_id,

            "selected_arm":
                bandit_start[
                    "selected_arm"
                ],

            "bandit_algorithm":
                self.bandit_policy.algorithm,
        }


    # ========================================================
    # SELECT MOVE FOR ONE TUTORING TURN
    #
    # INPUT:
    #   observable history_before
    #   frozen selector probabilities
    #
    # OUTPUT:
    #   final move to send to tutor agent
    #
    # NO LEARNING OCCURS HERE.
    # ========================================================

    def select_move(
        self,
        history_before,
        base_move_probs,
    ):

        if self.active_record is None:

            raise RuntimeError(
                "No active attempt."
            )


        if self.pending_turn is not None:

            raise RuntimeError(
                "The previous tutoring decision "
                "has not been completed yet."
            )


        updates_before = (
            self.bandit_policy.get_total_updates()
        )


        decision = (
            self.bandit_policy.apply_overlay(
                base_move_probs
            )
        )


        updates_after = (
            self.bandit_policy.get_total_updates()
        )


        if updates_after != updates_before:

            raise RuntimeError(
                "Bandit updated during move selection."
            )


        if (
            updates_after
            !=
            self.updates_at_attempt_start
        ):

            raise RuntimeError(
                "Bandit state changed during "
                "an active attempt."
            )


        turn_index = len(
            self.active_record[
                "trajectory"
            ]
        )


        # Deep copy is important.
        #
        # The main tutoring system will continue
        # modifying its conversation history after
        # this decision.
        self.pending_turn = {
            "turn_index":
                turn_index,

            "history_before":
                deepcopy(
                    history_before
                ),

            "base_move_probs":
                deepcopy(
                    decision[
                        "base_move_probs"
                    ]
                ),

            "base_move":
                decision[
                    "base_move"
                ],

            "final_move":
                decision[
                    "final_move"
                ],

            "selected_arm":
                decision[
                    "selected_arm"
                ],

            "preferred_move":
                decision[
                    "preferred_move"
                ],

            "overridden":
                decision[
                    "overridden"
                ],

            "probability_gap":
                decision[
                    "probability_gap"
                ],
        }


        return {
            "turn_index":
                turn_index,

            "base_move":
                decision[
                    "base_move"
                ],

            "final_move":
                decision[
                    "final_move"
                ],

            "selected_arm":
                decision[
                    "selected_arm"
                ],

            "overridden":
                decision[
                    "overridden"
                ],

            "probability_gap":
                decision[
                    "probability_gap"
                ],
        }


    # ========================================================
    # RECORD TUTOR + STUDENT RESPONSES
    #
    # Called AFTER:
    #
    # final_move
    #     â†“
    # tutor agent
    #     â†“
    # teacher response
    #     â†“
    # student response
    #
    # Still NO bandit learning.
    # ========================================================

    def record_turn(
        self,
        teacher_response,
        student_response,
    ):

        if self.active_record is None:

            raise RuntimeError(
                "No active attempt."
            )


        if self.pending_turn is None:

            raise RuntimeError(
                "No pending tutoring decision."
            )


        updates_before = (
            self.bandit_policy.get_total_updates()
        )


        completed_turn = deepcopy(
            self.pending_turn
        )


        completed_turn[
            "teacher_response"
        ] = str(
            teacher_response
        )


        completed_turn[
            "student_response"
        ] = str(
            student_response
        )


        self.active_record[
            "trajectory"
        ].append(
            completed_turn
        )


        self.pending_turn = None


        updates_after = (
            self.bandit_policy.get_total_updates()
        )


        if updates_after != updates_before:

            raise RuntimeError(
                "Bandit updated while recording "
                "an interaction."
            )


        if (
            updates_after
            !=
            self.updates_at_attempt_start
        ):

            raise RuntimeError(
                "Bandit state changed during "
                "an active attempt."
            )


        return deepcopy(
            completed_turn
        )


    # ========================================================
    # FINISH ATTEMPT
    #
    # Only now:
    #
    # external evaluator result exists
    # mastery_after exists
    #
    # Exactly ONE bandit update occurs here.
    # ========================================================

    def finish_attempt(
        self,
        evaluator_score,
        mastery_after,
    ):

        if self.active_record is None:

            raise RuntimeError(
                "No active attempt."
            )


        if self.pending_turn is not None:

            raise RuntimeError(
                "Cannot finish attempt while a "
                "tutoring decision is incomplete."
            )


        if not isinstance(
            mastery_after,
            dict,
        ):

            raise ValueError(
                "mastery_after must be a dictionary."
            )


        expected_skill = (
            self.active_record[
                "mastery_before"
            ][
                "skill_id"
            ]
        )


        actual_skill = str(
            mastery_after.get(
                "skill_id",
                "",
            )
        ).strip()


        if actual_skill != expected_skill:

            raise ValueError(
                "mastery_after.skill_id must match "
                "mastery_before.skill_id."
            )


        if "mastery" not in mastery_after:

            raise ValueError(
                "mastery_after.mastery is required."
            )


        mastery_after_value = float(
            mastery_after[
                "mastery"
            ]
        )


        updates_before = (
            self.bandit_policy.get_total_updates()
        )


        if (
            updates_before
            !=
            self.updates_at_attempt_start
        ):

            raise RuntimeError(
                "Unexpected bandit update occurred "
                "before attempt completion."
            )


        finish_result = (
            self.bandit_policy.finish_attempt(
                evaluator_score=
                    evaluator_score,

                mastery_after=
                    mastery_after_value,
            )
        )


        updates_after = (
            self.bandit_policy.get_total_updates()
        )


        if (
            updates_after
            !=
            updates_before
            + 1
        ):

            raise RuntimeError(
                "Expected exactly one bandit update "
                "after attempt completion."
            )


        self.active_record[
            "external_feedback"
        ] = {
            "evaluator_score":
                int(
                    evaluator_score
                ),

            "evaluator_rate":
                finish_result[
                    "reward"
                ],

            "mastery_after": {
                "skill_id":
                    actual_skill,

                "mastery":
                    mastery_after_value,
            },

            "mastery_delta":
                finish_result[
                    "mastery_delta"
                ],
        }


        self.active_record[
            "bandit_update"
        ] = {
            "reward":
                finish_result[
                    "reward"
                ],

            "updates_before":
                int(
                    updates_before
                ),

            "updates_after":
                int(
                    updates_after
                ),
        }


        completed_record = deepcopy(
            self.active_record
        )


        self.experience_logger.append(
            completed_record
        )


        self.active_record = None

        self.pending_turn = None

        self.updates_at_attempt_start = None


        return {
            "record":
                completed_record,

            "bandit_result":
                finish_result,
        }
