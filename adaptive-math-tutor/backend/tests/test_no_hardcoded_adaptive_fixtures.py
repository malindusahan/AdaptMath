import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class HardcodedAdaptiveFixtureRegressionTests(unittest.TestCase):
    def test_workflow_has_no_development_strategy_fixture(self) -> None:
        source = (BACKEND_ROOT / "app" / "graph" / "workflow.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("Development Worked Example", source)
        self.assertNotIn("dev-reteach-1", source)
        self.assertNotIn("development_fixture", source)

    def test_active_tutor_has_no_exact_commuting_cubic_case_constants(self) -> None:
        source = (
            BACKEND_ROOT / "app" / "agents" / "tutor" / "tutor_agent.py"
        ).read_text(encoding="utf-8")

        forbidden_fragments = [
            "p(0)=-24",
            "q(0)=30",
            "a*x**3+b*x**2+c*x-24",
            "d*x**3+e*x**2+f*x+30",
        ]

        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)

    def test_legacy_tutor_contains_no_executable_problem_specific_logic(self) -> None:
        source = (
            BACKEND_ROOT
            / "app"
            / "agents"
            / "tutor"
            / "tutor_agent_before_compaction.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("p(0)=-24", source)
        self.assertNotIn("q(0)=30", source)
        self.assertNotIn("class TutorAgent", source)

    def test_active_workflow_does_not_select_pedagogical_move(self) -> None:
        source = (
            BACKEND_ROOT
            / "app"
            / "graph"
            / "workflow.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "strategy_agent",
            source,
        )
        self.assertNotIn(
            "select_strategy(",
            source,
        )

        self.assertIn(
            "PedagogicalMoveSelection.model_validate",
            source,
        )


if __name__ == "__main__":
    unittest.main()
