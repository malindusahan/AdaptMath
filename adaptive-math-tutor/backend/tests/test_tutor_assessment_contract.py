import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from app.agents.evaluator.evaluator_agent import EvaluatorAgent
from app.schemas.assessment import (
    AssessmentGenerationInput,
    AssessmentGenerationOutput,
    AssessmentQuestion,
)
from app.schemas.dialogue import DialogueTurn
from app.schemas.tutor import TutorOutput


class AssessmentContractTests(unittest.TestCase):
    """Assessment belongs to Evaluator and always contains exactly 3 items."""

    @staticmethod
    def questions() -> list[AssessmentQuestion]:
        return [
            AssessmentQuestion(
                question_id="q1",
                question="What operation compares 192 and 48?",
                expected_answer="division",
            ),
            AssessmentQuestion(
                question_id="q2",
                question="Calculate 192 / 48.",
                expected_answer="4",
            ),
            AssessmentQuestion(
                question_id="q3",
                question="Why does that ratio answer the problem?",
                expected_answer="It shows how many times 48 fits into 192.",
            ),
        ]

    @staticmethod
    def generation_input() -> AssessmentGenerationInput:
        return AssessmentGenerationInput(
            original_question="Compare the total area 192 with folded area 48.",
            topic="ratio",
            student_age=13,
            conversation_history=[
                DialogueTurn(
                    role="teacher",
                    content="What should we compare?",
                    pedagogical_move="focus",
                ),
                DialogueTurn(role="student", content="192 and 48."),
            ],
        )

    def test_tutor_output_contains_only_one_teacher_turn(self) -> None:
        output = TutorOutput(teaching_response="What would you try first?")
        self.assertEqual(output.teaching_response, "What would you try first?")

        with self.assertRaises(ValidationError):
            TutorOutput(
                teaching_response="Teacher turn",
                assessment_questions=self.questions(),  # type: ignore[call-arg]
            )

    def test_assessment_requires_exactly_three_questions(self) -> None:
        output = AssessmentGenerationOutput(
            assessment_questions=self.questions()
        )
        self.assertEqual(len(output.assessment_questions), 3)

        with self.assertRaises(ValidationError):
            AssessmentGenerationOutput(
                assessment_questions=self.questions()[:2]
            )

    def test_assessment_rejects_duplicate_question_ids(self) -> None:
        duplicate = self.questions()
        duplicate[2] = duplicate[2].model_copy(update={"question_id": "q1"})
        with self.assertRaises(ValidationError):
            AssessmentGenerationOutput(assessment_questions=duplicate)

    def test_evaluator_assessment_generator_retries_once(self) -> None:
        agent = EvaluatorAgent.__new__(EvaluatorAgent)
        valid = AssessmentGenerationOutput(assessment_questions=self.questions())
        agent._generate_assessment_once = Mock(
            side_effect=[RuntimeError("temporary invalid output"), valid]
        )

        result = EvaluatorAgent.generate_assessment(
            agent,
            self.generation_input(),
        )

        self.assertEqual(len(result.assessment_questions), 3)
        self.assertEqual(agent._generate_assessment_once.call_count, 2)


if __name__ == "__main__":
    unittest.main()
