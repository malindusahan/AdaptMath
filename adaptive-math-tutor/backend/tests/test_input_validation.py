import unittest

from pydantic import ValidationError

from app.api.complexity import ComplexityRequest
from app.api.tutor import (
    TutorAnswerRequest,
    TutorStartRequest,
    TutorStudentResponseRequest,
)
from app.schemas.evaluator import StudentAnswer
from app.schemas.memory import LearnerHistoryItem
from app.schemas.planner import PlannerInput


class InputValidationTests(unittest.TestCase):
    def _valid_start_payload(self):
        return {
            "student_id": "student-001",
            "age": 15,
            "question": "Solve 3x + 5 = 20.",
            "topic": "algebra",
            "subtopic": "linear equations",
            "target_skill": "Equation Solving Two or Fewer Steps",
            "relevant_history": [],
            "previous_errors": [],
            "previous_strategies": [],
        }

    def test_tutor_start_trims_user_text(self):
        payload = self._valid_start_payload()
        payload.update(
            student_id="  student-001  ",
            question="  Solve 3x + 5 = 20.  ",
            topic="  algebra  ",
            subtopic="  linear equations  ",
        )

        request = TutorStartRequest(**payload)

        self.assertEqual(request.student_id, "student-001")
        self.assertEqual(request.question, "Solve 3x + 5 = 20.")
        self.assertEqual(request.topic, "algebra")
        self.assertEqual(request.subtopic, "linear equations")

    def test_tutor_start_requires_explicit_target_skill(self):
        payload = self._valid_start_payload()
        del payload["target_skill"]

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_tutor_start_rejects_whitespace_only_required_text(self):
        payload = self._valid_start_payload()
        payload["question"] = "   \t  "

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_tutor_start_normalizes_blank_optional_subtopic_to_none(self):
        payload = self._valid_start_payload()
        payload["subtopic"] = "   "

        request = TutorStartRequest(**payload)
        self.assertIsNone(request.subtopic)

    def test_tutor_start_rejects_out_of_scope_age(self):
        payload = self._valid_start_payload()
        payload["age"] = 19

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_tutor_start_rejects_oversized_question(self):
        payload = self._valid_start_payload()
        payload["question"] = "x" * 10_001

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_tutor_start_rejects_unknown_fields(self):
        payload = self._valid_start_payload()
        payload["unexpected_field"] = "should not be silently ignored"

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_tutor_start_rejects_frontend_pedagogical_move(self):
        payload = self._valid_start_payload()
        payload["pedagogical_move"] = "focus"

        with self.assertRaises(ValidationError):
            TutorStartRequest(**payload)

    def test_student_answer_rejects_whitespace_only_answer(self):
        with self.assertRaises(ValidationError):
            StudentAnswer(question_id="q1", answer="   ")

    def test_student_answer_trims_fields(self):
        answer = StudentAnswer(question_id="  q1  ", answer="  x = 5  ")
        self.assertEqual(answer.question_id, "q1")
        self.assertEqual(answer.answer, "x = 5")

    def test_answer_request_rejects_duplicate_question_ids_early(self):
        with self.assertRaises(ValidationError):
            TutorAnswerRequest(
                answers=[
                    StudentAnswer(question_id="q1", answer="4"),
                    StudentAnswer(question_id="q1", answer="5"),
                ]
            )

    def test_history_item_rejects_blank_event_type(self):
        with self.assertRaises(ValidationError):
            LearnerHistoryItem(event_type="   ")

    def test_complexity_request_uses_same_text_normalization(self):
        request = ComplexityRequest(question="  What is 7 + 5?  ")
        self.assertEqual(request.question, "What is 7 + 5?")

    def test_planner_input_rejects_blank_context_items(self):
        with self.assertRaises(ValidationError):
            PlannerInput(
                question="Solve 3x + 5 = 20.",
                topic="algebra",
                student_age=15,
                complexity_score=0.25,
                previous_errors=["   "],
            )

    def test_student_dialogue_response_rejects_blank_text(self):
        with self.assertRaises(ValidationError):
            TutorStudentResponseRequest(response="   ")


if __name__ == "__main__":
    unittest.main()
