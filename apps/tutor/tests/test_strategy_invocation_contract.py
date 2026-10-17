import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class PedagogicalMoveIntegrationContractTests(unittest.TestCase):
    """Every Tutor turn consumes an Omash-selected move; this component never selects it."""

    def setUp(self) -> None:
        self.workflow = (
            BACKEND_ROOT / "app" / "graph" / "workflow.py"
        ).read_text(encoding="utf-8")
        self.tutor = (
            BACKEND_ROOT / "app" / "agents" / "tutor" / "tutor_agent.py"
        ).read_text(encoding="utf-8")

    def test_active_workflow_does_not_import_strategy_agent(self) -> None:
        self.assertNotIn("StrategyAgent", self.workflow)
        self.assertNotIn("select_strategy(", self.workflow)

    def test_active_graph_delegates_move_selection_to_frozen_v3(self) -> None:
        self.assertIn("adaptive_coordinator.run_tutor_turn", self.workflow)
        self.assertIn("AdaptiveTutorPipeline", (
            BACKEND_ROOT / "app" / "integrations" /
            "adaptive_component_coordinator.py"
        ).read_text(encoding="utf-8"))
        self.assertNotIn(
            'builder.add_node("await_pedagogical_move"',
            self.workflow,
        )

    def test_tutor_prompt_forbids_replacing_omash_move(self) -> None:
        self.assertIn("You MUST execute exactly that move", self.tutor)
        self.assertIn("Do not choose", self.tutor)

    def test_selected_move_is_diagnostic_not_client_input(self) -> None:
        self.assertIn('"pedagogical_move": result["pedagogical_move"]', self.workflow)


if __name__ == "__main__":
    unittest.main()
