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
    def transfer_questions() -> list[AssessmentQuestion]:
        return [
            AssessmentQuestion(
                question_id="transfer_application",
                question=(
                    "A coach places 63 cones into 9 equal groups. "
                    "How many cones are in each group?"
                ),
                expected_answer="7 cones",
            ),
            AssessmentQuestion(
                question_id="reasoning_explanation",
                question=(
                    "Explain why division is appropriate when a quantity is "
                    "separated into equal groups."
                ),
                expected_answer=(
                    "Division finds the amount in each equal group."
                ),
            ),
            AssessmentQuestion(
                question_id="error_analysis",
                question=(
                    "A learner says 56 objects split among 8 teams gives 8 "
                    "objects per team. Identify and correct the error."
                ),
                expected_answer="The quotient is 7 objects per team, not 8.",
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
        valid = AssessmentGenerationOutput(
            assessment_questions=self.transfer_questions()
        )
        agent._generate_assessment_once = Mock(
            side_effect=[RuntimeError("temporary invalid output"), valid]
        )

        result = EvaluatorAgent.generate_assessment(
            agent,
            self.generation_input(),
        )

        self.assertEqual(len(result.assessment_questions), 3)
        self.assertEqual(agent._generate_assessment_once.call_count, 2)
        second_feedback = agent._generate_assessment_once.call_args_list[1].kwargs[
            "rejection_feedback"
        ]
        self.assertIn("previous draft was rejected", second_feedback)

    def test_variability_rejects_original_numbers_and_story_copy(self) -> None:
        agent = EvaluatorAgent.__new__(EvaluatorAgent)
        copied = AssessmentGenerationOutput(
            assessment_questions=self.questions()
        )
        with self.assertRaisesRegex(ValueError, "original numerical values"):
            agent._validate_assessment_variability(
                self.generation_input(),
                copied,
            )

    def test_variability_accepts_transfer_and_varied_evidence(self) -> None:
        agent = EvaluatorAgent.__new__(EvaluatorAgent)
        transfer = AssessmentGenerationOutput(
            assessment_questions=self.transfer_questions()
        )
        agent._validate_assessment_variability(
            self.generation_input(),
            transfer,
        )

    def test_variability_rejection_never_strands_completed_lesson(self) -> None:
        agent = EvaluatorAgent.__new__(EvaluatorAgent)
        first = AssessmentGenerationOutput(
            assessment_questions=self.questions()
        )
        second_questions = self.questions()
        second_questions[0] = second_questions[0].model_copy(
            update={"question": "What operation compares 48 and 192?"}
        )
        second = AssessmentGenerationOutput(
            assessment_questions=second_questions
        )
        agent._generate_assessment_once = Mock(side_effect=[first, second])

        result = EvaluatorAgent.generate_assessment(
            agent,
            self.generation_input(),
        )

        self.assertEqual(result, second)
        self.assertEqual(agent._generate_assessment_once.call_count, 2)
        feedback = agent._generate_assessment_once.call_args_list[1].kwargs[
            "rejection_feedback"
        ]
        self.assertIn("original numerical values", feedback)


if __name__ == "__main__":
    unittest.main()
