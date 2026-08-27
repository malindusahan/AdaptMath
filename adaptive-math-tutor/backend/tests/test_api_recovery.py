import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api import tutor as tutor_api


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
