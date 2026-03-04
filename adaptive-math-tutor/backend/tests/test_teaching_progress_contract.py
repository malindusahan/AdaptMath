import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas.progress import TeachingProgressOutput


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class TeachingProgressContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            BACKEND_ROOT
            / "app"
            / "agents"
            / "progress"
            / "teaching_progress_agent.py"
        ).read_text(encoding="utf-8")

    def test_progress_judge_has_only_continue_or_assess_contract(self) -> None:
        schema = (
            BACKEND_ROOT / "app" / "schemas" / "progress.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"continue_teaching"', schema)
        self.assertIn('"ready_for_assessment"', schema)

    def test_progress_judge_does_not_select_mathdial_move(self) -> None:
        self.assertIn("Never choose, recommend, replace, or", self.source)
        self.assertNotIn("PedagogicalMoveSelection", self.source)

    def test_no_fixed_turn_count_decides_completion(self) -> None:
        self.assertIn("Do not use fixed turn counts", self.source)
        self.assertNotIn("turn_count >=", self.source)
        self.assertNotIn("turn_count ==", self.source)

    def test_progress_judge_returns_dialogue_correctness_evidence(self) -> None:
        schema = (
            BACKEND_ROOT / "app" / "schemas" / "progress.py"
        ).read_text(encoding="utf-8")
        self.assertIn("latest_response_correctness", schema)
        self.assertIn("correctness_confidence", schema)
        self.assertIn("verified_math_evidence", schema)
        self.assertIn("classify the learner's LATEST response", self.source)

    def test_assessment_readiness_rejects_unfinished_original_problem(self) -> None:
        with self.assertRaises(ValidationError):
            TeachingProgressOutput(
                status="ready_for_assessment",
                reason="The learner named division as the next step.",
                latest_response_correctness="correct",
                correctness_confidence=0.95,
                correctness_reason="Division is the correct next operation.",
                latest_response_evidence_category="mathematical_evidence",
                original_problem_completion="not_yet",
                reasoning_sufficient_for_assessment=True,
                readiness_evidence="The final quotient was not computed.",
            )

    def test_assessment_readiness_requires_reasoning(self) -> None:
        with self.assertRaises(ValidationError):
            TeachingProgressOutput(
                status="ready_for_assessment",
                reason="The learner repeated the answer.",
                latest_response_correctness="correct",
                correctness_confidence=0.8,
                correctness_reason="The repeated result is numerically correct.",
                latest_response_evidence_category="mathematical_evidence",
                original_problem_completion="complete",
                reasoning_sufficient_for_assessment=False,
                readiness_evidence="No supporting reasoning was demonstrated.",
            )

    def test_completed_problem_with_reasoning_can_be_ready(self) -> None:
        result = TeachingProgressOutput(
            status="ready_for_assessment",
            reason="The learner completed the calculation and explained it.",
            latest_response_correctness="correct",
            correctness_confidence=0.98,
            correctness_reason="The final result and reasoning are correct.",
            latest_response_evidence_category="mathematical_evidence",
            original_problem_completion="complete",
            reasoning_sufficient_for_assessment=True,
            readiness_evidence="The learner stated the contextual final answer.",
        )
        self.assertEqual(result.status, "ready_for_assessment")


if __name__ == "__main__":
    unittest.main()
