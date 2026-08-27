import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas.router import RouterOutput


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class RouterV5PromptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            BACKEND_ROOT
            / "app"
            / "agents"
            / "router"
            / "router_agent.py"
        ).read_text(encoding="utf-8")

    def test_router_has_only_direct_and_planned_routes(self) -> None:
        RouterOutput(
            route="direct_tutor",
            reason="No separate planning stage is needed.",
        )
        RouterOutput(
            route="planned_tutor",
            reason="A separate plan would add useful structure.",
        )

        with self.assertRaises(ValidationError):
            RouterOutput(
                route="planned_tutor_evaluate",
                reason="Legacy route.",
            )

    def test_assessment_is_explicitly_outside_router_decision(self) -> None:
        self.assertIn(
            "Assessment is universal",
            self.source,
        )
        self.assertIn(
            "NOT part of your routing decision",
            self.source,
        )

    def test_router_does_not_select_pedagogical_move(self) -> None:
        self.assertIn(
            "Do not select the pedagogical move.",
            self.source,
        )
        self.assertIn(
            "selected by an external move-selector component",
            self.source,
        )

    def test_continuous_complexity_is_not_categorized(self) -> None:
        self.assertIn(
            "Do not create numerical thresholds.",
            self.source,
        )
        self.assertIn(
            "Do not convert it to Easy, Medium, or Hard categories.",
            self.source,
        )

    def test_missing_history_is_unknown(self) -> None:
        self.assertIn(
            "Treat missing learner history as UNKNOWN.",
            self.source,
        )

    def test_no_fixed_problem_to_route_mapping(self) -> None:
        forbidden = [
            "fraction ->",
            "algebra ->",
            "geometry ->",
            "score < 0.3",
            "score > 0.7",
        ]

        lower_source = self.source.lower()

        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(
                    fragment.lower(),
                    lower_source,
                )


if __name__ == "__main__":
    unittest.main()
