import unittest
from unittest.mock import patch

from app.graph import workflow as workflow_module
from app.graph.workflow import (
    adaptive_tutor_turn_node,
    after_memory,
    after_teaching_progress,
    await_pedagogical_move_node,
    await_student_response_node,
    begin_reteaching_node,
    choose_initial_workflow,
    complete_adaptive_attempt_node,
    start_adaptive_attempt_node,
)


class GraphOrchestrationTests(unittest.TestCase):
    """Offline tests for deterministic graph mechanics only."""

    def test_choose_initial_workflow_direct_tutor(self) -> None:
        self.assertEqual(
            choose_initial_workflow({"route": "direct_tutor"}),
            "direct_tutor",
        )

    def test_choose_initial_workflow_planned_tutor(self) -> None:
        self.assertEqual(
            choose_initial_workflow({"route": "planned_tutor"}),
            "planned_tutor",
        )

    def test_progress_continue_runs_fresh_adaptive_turn(self) -> None:
        self.assertEqual(
            after_teaching_progress({"teaching_status": "continue_teaching"}),
            "adaptive_turn",
        )

    def test_progress_ready_generates_assessment(self) -> None:
        self.assertEqual(
            after_teaching_progress({"teaching_status": "ready_for_assessment"}),
            "generate_assessment",
        )

    def test_student_response_is_appended_to_dialogue(self) -> None:
        state = {
            "conversation_history": [
                {
                    "role": "teacher",
                    "content": "How would you start?",
                    "pedagogical_move": "generic",
                }
            ]
        }
        with patch.object(
            workflow_module,
            "interrupt",
            return_value={"response": "I would find the total area first."},
        ):
            result = await_student_response_node(state)

        self.assertEqual(
            result["latest_student_response"],
            "I would find the total area first.",
        )
        self.assertEqual(len(result["conversation_history"]), 2)
        self.assertEqual(result["conversation_history"][-1]["role"], "student")

    def test_external_move_is_supplied_for_next_turn(self) -> None:
        with patch.object(
            workflow_module,
            "interrupt",
            return_value={"move": "probing"},
        ):
            result = await_pedagogical_move_node({})

        self.assertEqual(result, {"pedagogical_move": "probing"})

    def test_invalid_external_move_is_rejected(self) -> None:
        with patch.object(
            workflow_module,
            "interrupt",
            return_value={"move": "made_up_move"},
        ):
            with self.assertRaises(Exception):
                await_pedagogical_move_node({})

    def test_after_memory_starts_reteaching_when_needed(self) -> None:
        self.assertEqual(
            after_memory({"needs_reteaching": True}),
            "begin_reteaching",
        )

    def test_after_memory_finishes_when_reteaching_not_needed(self) -> None:
        self.assertEqual(after_memory({"needs_reteaching": False}), "finish")

    def test_begin_reteaching_preserves_already_started_next_attempt(self) -> None:
        result = begin_reteaching_node(
            {
                "reteach_round": 2,
                "attempt_id": "thread:2",
                "adaptive_attempt_index": 2,
            }
        )
        self.assertEqual(result["reteach_round"], 3)
        self.assertEqual(result["teaching_phase"], "reteaching")
        self.assertNotIn("attempt_id", result)
        self.assertEqual(result["assessment_questions"], [])

    def test_adaptive_nodes_delegate_to_coordinator(self) -> None:
        class FakeCoordinator:
            def start_attempt(self, state):
                return {"adaptive_lifecycle_status": "active", "attempt_id": state["attempt_id"]}

            def run_tutor_turn(self, state, *, tutor_agent):
                del tutor_agent
                state["conversation_history"].append(
                    {
                        "role": "teacher",
                        "content": "Try isolating the variable.",
                        "pedagogical_move": "focus",
                    }
                )
                return {
                    "attempt_id": state["attempt_id"],
                    "turn_index": 1,
                    "tutor_response": "Try isolating the variable.",
                    "pedagogical_move": "focus",
                }

            def finish_attempt(self, state):
                return {
                    "completed_attempt_id": state["attempt_id"],
                    "adaptive_lifecycle_status": "completed",
                }

        state = {
            "attempt_id": "thread:1",
            "conversation_history": [],
            "turn_count": 0,
        }
        with patch.object(workflow_module, "adaptive_coordinator", FakeCoordinator()):
            self.assertEqual(
                start_adaptive_attempt_node(state)["adaptive_lifecycle_status"],
                "active",
            )
            turn = adaptive_tutor_turn_node(state)
            completed = complete_adaptive_attempt_node(state)

        self.assertEqual(turn["pedagogical_move"], "focus")
        self.assertEqual(len(turn["conversation_history"]), 1)
        self.assertEqual(completed["completed_attempt_id"], "thread:1")


if __name__ == "__main__":
    unittest.main()
