import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
