from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = (
    "self_improvement_attempt_v1"
)


class AttemptExperienceLogger:

    def __init__(
        self,
        path,
        data_origin,
    ):

        self.path = Path(
            path
        )


        self.data_origin = (
            str(
                data_origin
            )
            .strip()
            .lower()
        )


        if self.data_origin not in {
            "synthetic",
            "real",
        }:

            raise ValueError(
                "data_origin must be "
                "'synthetic' or 'real'."
            )


        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


    def validate_record(
        self,
        record,
    ):

        errors = []


        if (
            record.get(
                "schema_version"
            )
            !=
            SCHEMA_VERSION
        ):

            errors.append(
                "Invalid schema_version."
            )


        if (
            record.get(
                "data_origin"
            )
            !=
            self.data_origin
        ):

            errors.append(
                "Record data_origin does not "
                "match logger data_origin."
            )


        required = [
            "episode_id",
            "attempt_index",
            "problem",
            "mastery_before",
            "policy",
            "trajectory",
            "external_feedback",
        ]


        for key in required:

            if key not in record:

                errors.append(
                    f"Missing required field: {key}"
                )


        if errors:

            return errors


        mastery_before = (
            record[
                "mastery_before"
            ]
        )


        if (
            "skill_id"
            not in mastery_before
        ):

            errors.append(
                "mastery_before.skill_id missing."
            )


        if (
            "mastery"
            not in mastery_before
        ):

            errors.append(
                "mastery_before.mastery missing."
            )


        else:

            mastery_value = float(
                mastery_before[
                    "mastery"
                ]
            )


            if not (
                0.0
                <=
                mastery_value
                <=
                1.0
            ):

                errors.append(
                    "mastery_before.mastery "
                    "must be in [0, 1]."
                )


        policy = (
            record[
                "policy"
            ]
        )


        for key in [
            "base_selector_version",
            "bandit_algorithm",
            "selected_arm",
            "max_probability_gap",
        ]:

            if key not in policy:

                errors.append(
                    f"policy.{key} missing."
                )


        trajectory = (
            record[
                "trajectory"
            ]
        )


        if not isinstance(
            trajectory,
            list,
        ):

            errors.append(
                "trajectory must be a list."
            )


        else:

            for (
                expected_turn_index,
                turn,
            ) in enumerate(
                trajectory
            ):

                if (
                    turn.get(
                        "turn_index"
                    )
                    !=
                    expected_turn_index
                ):

                    errors.append(
                        "trajectory turn_index "
                        "sequence invalid."
                    )


                for key in [
                    "history_before",
                    "base_move_probs",
                    "base_move",
                    "final_move",
                    "overridden",
                    "teacher_response",
                    "student_response",
                ]:

                    if key not in turn:

                        errors.append(
                            "trajectory turn missing "
                            f"field: {key}"
                        )


        external = (
            record[
                "external_feedback"
            ]
        )


        if (
            "evaluator_score"
            not in external
        ):

            errors.append(
                "external_feedback."
                "evaluator_score missing."
            )


        else:

            evaluator_score = int(
                external[
                    "evaluator_score"
                ]
            )


            if evaluator_score not in {
                0,
                1,
                2,
                3,
            }:

                errors.append(
                    "evaluator_score must be "
                    "0, 1, 2, or 3."
                )


        mastery_after = (
            external.get(
                "mastery_after"
            )
        )


        if not isinstance(
            mastery_after,
            dict,
        ):

            errors.append(
                "external_feedback.mastery_after "
                "must be an object."
            )


        else:

            before_skill = (
                mastery_before.get(
                    "skill_id"
                )
            )


            after_skill = (
                mastery_after.get(
                    "skill_id"
                )
            )


            if (
                before_skill
                !=
                after_skill
            ):

                errors.append(
                    "skill_id must be identical "
                    "before and after attempt."
                )


            if (
                "mastery"
                not in mastery_after
            ):

                errors.append(
                    "mastery_after.mastery missing."
                )


            else:

                value = float(
                    mastery_after[
                        "mastery"
                    ]
                )


                if not (
                    0.0
                    <=
                    value
                    <=
                    1.0
                ):

                    errors.append(
                        "mastery_after.mastery "
                        "must be in [0, 1]."
                    )


        return errors


    def append(
        self,
        record,
    ):

        errors = self.validate_record(
            record
        )


        if errors:

            raise ValueError(
                "Attempt record failed validation:\n"
                +
                "\n".join(
                    f"- {error}"
                    for error
                    in errors
                )
            )


        from .postgres_repository import append_experience, postgres_enabled
        if postgres_enabled():
            append_experience(record)
            return self.path

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            f.write(
                "\n"
            )


        return self.path
