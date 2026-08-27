import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class MultiTurnTutorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = (
            BACKEND_ROOT / "app" / "graph" / "workflow.py"
        ).read_text(encoding="utf-8")
        self.tutor = (
            BACKEND_ROOT / "app" / "agents" / "tutor" / "tutor_agent.py"
        ).read_text(encoding="utf-8")
        self.evaluator = (
            BACKEND_ROOT / "app" / "agents" / "evaluator" / "evaluator_agent.py"
        ).read_text(encoding="utf-8")

    def test_tutor_generates_one_teacher_turn_not_full_assessment(self) -> None:
        self.assertIn("Your output is ONE teacher move", self.tutor)
        self.assertNotIn("self.assessment_prompt", self.tutor)


    def test_one_turn_contract_is_atomic_not_a_solution_plan(self) -> None:
        self.assertIn("ATOMIC DIALOGUE CONTRACT", self.tutor)
        self.assertIn("Execute exactly ONE pedagogical action", self.tutor)
        self.assertIn("Never reveal an ordered chain", self.tutor)
        self.assertIn("Never ask the learner for the complete solution", self.tutor)

    def test_move_execution_is_narrowed_per_turn(self) -> None:
        self.assertIn("generic: make ONE broad conversational invitation", self.tutor)
        self.assertIn("focus: direct attention to ONE relevant", self.tutor)
        self.assertIn("probing: investigate ONE current claim", self.tutor)
        self.assertIn("telling: explicitly provide ONE needed fact", self.tutor)

    def test_generic_opening_turn_contract_rejects_full_solution_planning(self) -> None:
        self.assertIn("PASS, generic", self.tutor)
        self.assertIn("How would you start solving this problem?", self.tutor)
        self.assertIn("What numbers would you write down", self.tutor)
        self.assertIn("requests the whole solution plan", self.tutor)

    def test_teacher_turn_uses_structured_atomic_self_check(self) -> None:
        self.assertIn("build_teacher_turn_schema", self.tutor)
        self.assertIn("reveals_future_steps", self.tutor)
        self.assertIn("multiple_independent_requests", self.tutor)
        self.assertIn("blends_multiple_pedagogical_actions", self.tutor)

    def test_graph_waits_for_student_after_every_teacher_turn(self) -> None:
        self.assertIn(
            'builder.add_edge("adaptive_tutor_turn", "await_student_response")',
            self.workflow,
        )
        self.assertIn("await_student_response_node", self.workflow)


    def test_first_teacher_turn_starts_the_adaptive_attempt(self) -> None:
        self.assertIn(
            'builder.add_edge("prepare_math", "start_adaptive_attempt")',
            self.workflow,
        )
        self.assertIn(
            'builder.add_edge("start_adaptive_attempt", "adaptive_tutor_turn")',
            self.workflow,
        )

    def test_each_continuing_turn_runs_the_adaptive_pipeline(self) -> None:
        self.assertIn('"adaptive_turn": "adaptive_tutor_turn"', self.workflow)
        self.assertNotIn(
            'builder.add_edge("await_pedagogical_move", "tutor")',
            self.workflow,
        )

    def test_evaluator_owns_three_question_generation(self) -> None:
        self.assertIn("def generate_assessment", self.evaluator)
        self.assertIn("EXACTLY THREE", self.evaluator)
        self.assertIn('builder.add_node("generate_assessment"', self.workflow)


if __name__ == "__main__":
    unittest.main()
