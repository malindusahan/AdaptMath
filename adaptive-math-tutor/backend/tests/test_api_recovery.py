import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api import tutor as tutor_api
from app.clients.memory.memory_client import MemoryAuthenticatedUser
from app.integrations.adaptive_component_coordinator import AdaptiveRestartRequiredError


class TutorApiRecoveryTests(unittest.TestCase):
    def _snapshot(self, next_node: str, interrupt_type: str, values=None):
        return SimpleNamespace(
            values=values or {
                "tutor_response": "What would you try next?",
                "route": "planned_tutor",
                "route_reason": "Planner was useful.",
                "complexity_score": 0.25,
                "reteach_round": 0,
                "turn_count": 2,
            },
            next=(next_node,),
            tasks=(
                SimpleNamespace(
                    interrupts=(
                        SimpleNamespace(value={"type": interrupt_type}),
                    )
                ),
            ),
        )

    def test_student_turn_interrupt_is_recoverable(self):
        response = tutor_api._build_snapshot_response(
            "thread-student",
            self._snapshot("await_student_response", "student_response_required"),
        )
        self.assertEqual(response.status, "student_response_required")
        self.assertEqual(response.turn_count, 2)

    def test_move_interrupt_is_recoverable(self):
        response = tutor_api._build_snapshot_response(
            "thread-move",
            self._snapshot("await_pedagogical_move", "pedagogical_move_required"),
        )
        self.assertEqual(response.status, "pedagogical_move_required")
        self.assertEqual(
            set(response.allowed_pedagogical_moves),
            {"telling", "focus", "generic", "probing"},
        )

    def test_assessment_interrupt_recovers_three_questions_without_answers(self):
        values = {
            "tutor_response": "Good reasoning.",
            "route": "planned_tutor",
            "complexity_score": 0.25,
            "assessment_questions": [
                {"question_id": "q1", "question": "Q1", "expected_answer": "A1"},
                {"question_id": "q2", "question": "Q2", "expected_answer": "A2"},
                {"question_id": "q3", "question": "Q3", "expected_answer": "A3"},
            ],
        }
        response = tutor_api._build_snapshot_response(
            "thread-assess",
            self._snapshot("await_answers", "assessment_required", values=values),
        )
        self.assertEqual(response.status, "assessment_required")
        self.assertEqual(len(response.questions), 3)
        for question in response.model_dump()["questions"]:
            self.assertNotIn("expected_answer", question)

    def test_active_student_lookup_returns_newest_durable_active_lesson(self):
        active_snapshot = self._snapshot(
            "await_student_response",
            "student_response_required",
            values={
                "student_id": "student-a",
                "adaptive_lifecycle_status": "active",
                "tutor_response": "Continue this lesson.",
                "turn_count": 1,
            },
        )

        with patch.object(
            tutor_api,
            "list_checkpoint_thread_ids",
            return_value=["thread-active", "thread-other"],
        ):
            with patch.object(
                tutor_api.tutor_graph,
                "get_state",
                return_value=active_snapshot,
            ) as get_state_mock:
                response = tutor_api._find_active_student_session("student-a")

        self.assertIsNotNone(response)
        self.assertEqual(response.thread_id, "thread-active")
        self.assertEqual(response.status, "student_response_required")
        get_state_mock.assert_called_once()

    def test_active_student_endpoint_returns_none_when_no_lesson_exists(self):
        with patch.object(tutor_api, "_find_active_student_session", return_value=None):
            response = tutor_api.get_active_tutor_session(
                MemoryAuthenticatedUser(
                    user_id="user-a",
                    username="student-a",
                    role="STUDENT",
                    student_id="student-a",
                    age=15,
                )
            )

        self.assertIsNone(response)

    def test_abandon_active_lesson_aborts_attempt_and_persists_lifecycle(self):
        snapshot = self._snapshot(
            "await_student_response",
            "student_response_required",
            values={
                "student_id": "student-a",
                "thread_id": "thread-active",
                "attempt_id": "thread-active:1",
                "adaptive_lifecycle_status": "active",
                "tutor_response": "What would you try next?",
                "turn_count": 1,
            },
        )

        with patch.object(tutor_api, "_get_persisted_snapshot", return_value=snapshot):
            with patch.object(
                tutor_api.adaptive_coordinator,
                "abort_attempt",
                return_value={"adaptive_lifecycle_status": "aborted"},
            ) as abort_mock:
                with patch.object(tutor_api.tutor_graph, "update_state") as update_mock:
                    response = tutor_api.abandon_tutor_session("thread-active")

        self.assertEqual(response.status, "complete")
        abort_mock.assert_called_once()
        update_mock.assert_called_once_with(
            tutor_api._thread_config("thread-active"),
            {"adaptive_lifecycle_status": "aborted"},
        )

    def test_abandon_is_idempotent_for_already_aborted_lesson(self):
        snapshot = self._snapshot(
            "await_student_response",
            "student_response_required",
            values={
                "student_id": "student-a",
                "adaptive_lifecycle_status": "aborted",
            },
        )

        with patch.object(tutor_api, "_get_persisted_snapshot", return_value=snapshot):
            with patch.object(
                tutor_api.adaptive_coordinator,
                "abort_attempt",
            ) as abort_mock:
                with patch.object(tutor_api.tutor_graph, "update_state") as update_mock:
                    response = tutor_api.abandon_tutor_session("thread-active")

        self.assertEqual(response.status, "complete")
        abort_mock.assert_not_called()
        update_mock.assert_not_called()

    def test_abandon_closes_unrestorable_post_restart_lesson(self):
        snapshot = self._snapshot(
            "await_student_response",
            "student_response_required",
            values={
                "student_id": "student-a",
                "adaptive_lifecycle_status": "active",
            },
        )

        with patch.object(tutor_api, "_get_persisted_snapshot", return_value=snapshot):
            with patch.object(
                tutor_api.adaptive_coordinator,
                "abort_attempt",
                side_effect=AdaptiveRestartRequiredError("restart"),
            ):
                with patch.object(tutor_api.tutor_graph, "update_state") as update_mock:
                    response = tutor_api.abandon_tutor_session("thread-active")

        self.assertEqual(response.status, "complete")
        update_mock.assert_called_once_with(
            tutor_api._thread_config("thread-active"),
            {"adaptive_lifecycle_status": "aborted"},
        )

    def test_submit_student_response_resumes_only_student_interrupt(self):
        graph_result = {
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "type": "pedagogical_move_required",
                        "message": "Need next move",
                        "allowed_moves": ["telling", "focus", "generic", "probing"],
                    }
                )
            ],
            "tutor_response": "What would you try next?",
            "turn_count": 1,
        }

        with patch.object(
            tutor_api,
            "_get_persisted_snapshot",
            return_value=self._snapshot(
                "await_student_response",
                "student_response_required",
            ),
        ):
            with patch.object(
                tutor_api.tutor_graph,
                "invoke",
                return_value=graph_result,
            ) as invoke_mock:
                response = tutor_api.submit_student_response(
                    "thread-student",
                    tutor_api.TutorStudentResponseRequest(
                        response="I would divide 192 by 48."
                    ),
                )

        command = invoke_mock.call_args.args[0]
        self.assertEqual(
            command.resume,
            {"response": "I would divide 192 by 48."},
        )
        self.assertEqual(response.status, "pedagogical_move_required")

    def test_submit_move_produces_next_teacher_turn_interrupt(self):
        graph_result = {
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "type": "student_response_required",
                        "message": "Reply to tutor",
                        "teacher_message": "Why does division help here?",
                    }
                )
            ],
            "tutor_response": "Why does division help here?",
            "turn_count": 2,
        }

        with patch.object(
            tutor_api,
            "_get_persisted_snapshot",
            return_value=self._snapshot(
                "await_pedagogical_move",
                "pedagogical_move_required",
            ),
        ):
            with patch.object(
                tutor_api.tutor_graph,
                "invoke",
                return_value=graph_result,
            ):
                response = tutor_api.submit_pedagogical_move(
                    "thread-move",
                    tutor_api.TutorPedagogicalMoveRequest(
                        pedagogical_move="probing"
                    ),
                )

        self.assertEqual(response.status, "student_response_required")
        self.assertEqual(response.tutor_response, "Why does division help here?")


if __name__ == "__main__":
    unittest.main()
