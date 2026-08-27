import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from app.api.tutor import (
    TutorAnswerRequest,
    _build_public_response,
    _safe_answer_evidence,
    _safe_evaluation,
)
from app.graph import workflow as workflow_module
from app.schemas.evaluator import StudentAnswer


class ApiPrivacyTests(unittest.TestCase):
    def test_safe_answer_evidence_removes_expected_answer(self) -> None:
        internal_answer = {
            "question_id": "q1",
            "question": "What is 2 + 2?",
            "student_answer": "4",
            "expected_answer": "4",
            "is_correct": True,
            "feedback": "Correct.",
            "identified_error": None,
            "internal_note": "private",
        }

        result = _safe_answer_evidence(internal_answer)

        self.assertEqual(result["question_id"], "q1")
        self.assertEqual(result["student_answer"], "4")
        self.assertNotIn("expected_answer", result)
        self.assertNotIn("internal_note", result)

    def test_safe_evaluation_removes_expected_answers(self) -> None:
        graph_result = {
            "evaluation_result": {
                "correct_answers": [
                    {
                        "question_id": "q1",
                        "question": "What is 2 + 2?",
                        "student_answer": "4",
                        "expected_answer": "4",
                        "is_correct": True,
                        "feedback": "Correct.",
                        "identified_error": None,
                    }
                ],
                "wrong_answers": [
                    {
                        "question_id": "q2",
                        "question": "What is 3 + 3?",
                        "student_answer": "5",
                        "expected_answer": "6",
                        "is_correct": False,
                        "feedback": "Incorrect.",
                        "identified_error": "addition error",
                    }
                ],
                "identified_errors": ["addition error"],
                "needs_reteaching": True,
                "overall_feedback": "Review addition.",
            }
        }

        result = _safe_evaluation(graph_result)

        self.assertIsNotNone(result)
        assert result is not None

        for answer in (
            result["correct_answers"]
            + result["wrong_answers"]
        ):
            self.assertNotIn("expected_answer", answer)

    def test_public_interrupt_response_drops_expected_answer(self) -> None:
        graph_result = {
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "type": "assessment_required",
                        "message": "Please answer all assessment questions.",
                        "questions": [
                            {
                                "question_id": "q1",
                                "question": "What is 2 + 2?",
                                "expected_answer": "4",
                            },
                            {
                                "question_id": "q2",
                                "question": "What is 3 + 3?",
                                "expected_answer": "6",
                            },
                            {
                                "question_id": "q3",
                                "question": "What is 4 + 4?",
                                "expected_answer": "8",
                            },
                        ],
                    }
                )
            ],
            "route": "planned_tutor",
            "tutor_response": "Teaching response",
            "complexity_score": 0.4,
        }

        response = _build_public_response(
            "thread-test-1",
            graph_result,
        )

        dumped = response.model_dump()

        self.assertEqual(
            dumped["status"],
            "assessment_required",
        )
        self.assertEqual(len(dumped["questions"]), 3)

        for question in dumped["questions"]:
            self.assertNotIn(
                "expected_answer",
                question,
            )


class AnswerSubmissionSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "assessment_questions": [
                {
                    "question_id": "q1",
                    "question": "What is 2 + 2?",
                    "expected_answer": "4",
                },
                {
                    "question_id": "q2",
                    "question": "What is 3 + 3?",
                    "expected_answer": "6",
                },
                {
                    "question_id": "q3",
                    "question": "What is 4 + 4?",
                    "expected_answer": "8",
                },
            ]
        }

    def test_answer_request_rejects_empty_answer_list(self) -> None:
        with self.assertRaises(ValidationError):
            TutorAnswerRequest(answers=[])

    def test_student_answer_rejects_blank_fields(self) -> None:
        with self.assertRaises(ValidationError):
            StudentAnswer(
                question_id="",
                answer="4",
            )

        with self.assertRaises(ValidationError):
            StudentAnswer(
                question_id="q1",
                answer="",
            )

    def test_await_answers_rejects_duplicate_question_ids(self) -> None:
        submitted = [
            {
                "question_id": "q1",
                "answer": "4",
            },
            {
                "question_id": "q1",
                "answer": "4 again",
            },
        ]

        with patch.object(
            workflow_module,
            "interrupt",
            return_value=submitted,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "must be answered once",
            ):
                workflow_module.await_answers_node(
                    self.state
                )

    def test_await_answers_rejects_missing_or_unknown_ids(self) -> None:
        submitted = [
            {
                "question_id": "q1",
                "answer": "4",
            },
            {
                "question_id": "q3",
                "answer": "7",
            },
        ]

        with patch.object(
            workflow_module,
            "interrupt",
            return_value=submitted,
        ):
            with self.assertRaisesRegex(
    ValueError,
    "all three assessment questions",
):
                workflow_module.await_answers_node(
                    self.state
                )

    def test_await_answers_accepts_exactly_one_answer_per_question(self) -> None:
        submitted = [
            {
                "question_id": "q1",
                "answer": "4",
            },
            {
                "question_id": "q2",
                "answer": "6",
            },
            {
                "question_id": "q3",
                "answer": "8",
            },
        ]

        with patch.object(
            workflow_module,
            "interrupt",
            return_value=submitted,
        ):
            result = workflow_module.await_answers_node(
                self.state
            )

        self.assertEqual(
            result["student_answers"],
            [
                {"question_id": "q1", "answer": "4"},
                {"question_id": "q2", "answer": "6"},
                {"question_id": "q3", "answer": "8"},
            ],
        )
        self.assertEqual(len(result["conversation_history"]), 6)
        self.assertEqual(result["conversation_history"][0]["role"], "teacher")
        self.assertEqual(result["conversation_history"][1]["role"], "student")


if __name__ == "__main__":
    unittest.main()