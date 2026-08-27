import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.response_verifier import TeachingResponseVerifier
from app.agents.tutor.tutor_agent import TutorAgent
from app.schemas.tutor import TutorInput


class TutorResponseVerificationTests(unittest.TestCase):
    @staticmethod
    def make_tutor_input() -> TutorInput:
        return TutorInput(
            question=(
                "Three positive integers a, b, and c satisfy "
                "a+b+c=30, ab+bc+ca=269, abc=660. Find them."
            ),
            topic="algebra",
            subtopic="symmetric polynomials",
            student_age=18,
            complexity_score=0.65,
            pedagogical_move="probing",
            conversation_history=[],
            previous_errors=[],
            reteaching=False,
        )

    @staticmethod
    def make_response(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )

    @classmethod
    def make_teacher_turn_response(
        cls,
        message: str,
        *,
        move: str = "probing",
        reveals_future_steps: bool = False,
        multiple_independent_requests: bool = False,
        blends_multiple_pedagogical_actions: bool = False,
    ) -> SimpleNamespace:
        return cls.make_response(
            json.dumps(
                {
                    "selected_move": move,
                    "immediate_target": "one immediate reasoning target",
                    "learner_action": "respond to one prompt",
                    "teacher_message": message,
                    "reveals_future_steps": reveals_future_steps,
                    "multiple_independent_requests": multiple_independent_requests,
                    "blends_multiple_pedagogical_actions": (
                        blends_multiple_pedagogical_actions
                    ),
                }
            )
        )

    def make_verifier(self) -> TeachingResponseVerifier:
        verifier = TeachingResponseVerifier.__new__(
            TeachingResponseVerifier
        )
        verifier.math_verifier = SymPyMathVerifier()
        return verifier

    def test_deterministic_audit_rejects_false_arithmetic(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [
                    {
                        "source_text": (
                            "1000 - 3000 + 2690 - 660 = -970"
                        ),
                        "left_expression": "1000-3000+2690-660",
                        "right_expression": "-970",
                    }
                ],
                "logical_issues": [],
            }
        )

        self.assertFalse(result["approved"])
        self.assertEqual(len(result["deterministic_failures"]), 1)

    def test_deterministic_audit_accepts_correct_factorization(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [
                    {
                        "source_text": (
                            "x^3 - 30x^2 + 269x - 660 "
                            "= (x-4)(x-11)(x-15)"
                        ),
                        "left_expression": "x**3-30*x**2+269*x-660",
                        "right_expression": "(x-4)*(x-11)*(x-15)",
                    }
                ],
                "logical_issues": [],
            }
        )

        self.assertTrue(result["approved"])
        self.assertEqual(result["deterministic_failures"], [])

    def test_audit_rejects_clear_logical_contradiction(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [],
                "logical_issues": [
                    {
                        "source_text": (
                            "11 is the only integer divisor that is a root."
                        ),
                        "reason": (
                            "The same response later identifies 4 and 15 "
                            "as additional integer roots."
                        ),
                    }
                ],
            }
        )

        self.assertFalse(result["approved"])
        self.assertEqual(len(result["logical_issues"]), 1)


    def test_audit_rejects_over_scoped_multi_step_teacher_turn(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [],
                "logical_issues": [],
                "turn_scope_issues": [
                    {
                        "source_text": (
                            "First find the total books, then subtract 18, "
                            "then divide by 9."
                        ),
                        "reason": (
                            "The draft reveals several future solution steps "
                            "instead of executing one atomic teacher move."
                        ),
                    }
                ],
            }
        )

        self.assertFalse(result["approved"])
        self.assertEqual(len(result["turn_scope_issues"]), 1)

    def test_audit_accepts_atomic_turn_when_no_scope_issue_is_reported(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [],
                "logical_issues": [],
                "turn_scope_issues": [],
            }
        )

        self.assertTrue(result["approved"])
        self.assertEqual(result["turn_scope_issues"], [])

    def test_scope_failure_feedback_requests_single_turn_rewrite(self) -> None:
        feedback = TeachingResponseVerifier.correction_feedback(
            {
                "deterministic_failures": [],
                "logical_issues": [],
                "turn_scope_issues": [
                    {
                        "source_text": "Do step 1, step 2, and step 3.",
                        "reason": "Several future reasoning steps were bundled.",
                    }
                ],
            }
        )

        self.assertIn("Single-turn pedagogy/scope issues", feedback)
        self.assertIn("current teacher turn", feedback)

    def test_explicit_scope_verdict_rejects_full_plan_even_without_issue_array(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [],
                "logical_issues": [],
                "turn_scope_issues": [],
                "turn_scope_verdict": "fail",
                "move_alignment_verdict": "pass",
                "scope_reason": "The response requests the complete multi-step plan.",
                "move_alignment_reason": "The supplied move was generic.",
            }
        )

        self.assertFalse(result["approved"])
        self.assertEqual(result["turn_scope_verdict"], "fail")

    def test_move_alignment_verdict_is_required_for_approval(self) -> None:
        verifier = self.make_verifier()
        result = verifier.evaluate_payload(
            {
                "checkable_claims": [],
                "logical_issues": [],
                "turn_scope_issues": [],
                "turn_scope_verdict": "pass",
                "move_alignment_verdict": "fail",
                "scope_reason": "Atomic scope is acceptable.",
                "move_alignment_reason": "The response blended focus and telling.",
            }
        )

        self.assertFalse(result["approved"])
        self.assertEqual(result["move_alignment_verdict"], "fail")

    def test_verifier_prompt_explicitly_rejects_observed_generic_failure(self) -> None:
        prompt = TeachingResponseVerifier.audit_prompt
        self.assertIn("What numbers would you write down", prompt)
        self.assertIn("multiplication", prompt)
        self.assertIn("requests multiple learner tasks", prompt)
        self.assertIn("How would you start solving this problem?", prompt)

    def test_teaching_finalizer_retries_before_audit_when_self_check_reports_future_steps(self) -> None:
        agent = TutorAgent.__new__(TutorAgent)
        agent.model_name = "test-model"
        agent.final_teaching_prompt = "test finalizer prompt"
        agent.max_response_verification_attempts = 3

        create = Mock(
            side_effect=[
                self.make_teacher_turn_response(
                    "First multiply, then subtract, then divide.",
                    reveals_future_steps=True,
                    multiple_independent_requests=True,
                ),
                self.make_teacher_turn_response(
                    "How would you start solving this?"
                ),
            ]
        )
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        passed_audit = {
            "approved": True,
            "deterministic_failures": [],
            "logical_issues": [],
            "turn_scope_issues": [],
            "turn_scope_verdict": "pass",
            "move_alignment_verdict": "pass",
            "scope_reason": "Atomic.",
            "move_alignment_reason": "Aligned.",
            "inconclusive_checks": [],
        }
        agent.response_verifier = SimpleNamespace(
            audit=Mock(return_value=passed_audit),
            correction_feedback=Mock(return_value=""),
        )

        result = TutorAgent._generate_teaching_response(
            agent,
            self.make_tutor_input(),
            evidence=[],
        )

        self.assertEqual(result, "How would you start solving this?")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(agent.response_verifier.audit.call_count, 1)
        second_user_message = create.call_args_list[1].kwargs["messages"][1]["content"]
        self.assertIn("structured teacher-turn self-check failed", second_user_message)

    def test_teaching_finalizer_retries_after_failed_audit(self) -> None:
        agent = TutorAgent.__new__(TutorAgent)
        agent.model_name = "test-model"
        agent.final_teaching_prompt = "test finalizer prompt"
        agent.max_response_verification_attempts = 3

        create = Mock(
            side_effect=[
                self.make_teacher_turn_response(
                    "Incorrect draft: 1000 - 3000 + 2690 - 660 = -970."
                ),
                self.make_teacher_turn_response(
                    "Corrected draft: 1000 - 3000 + 2690 - 660 = 30."
                ),
            ]
        )
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        failed_audit = {
            "approved": False,
            "deterministic_failures": [
                {
                    "source_text": (
                        "1000 - 3000 + 2690 - 660 = -970"
                    ),
                    "left_expression": "1000-3000+2690-660",
                    "right_expression": "-970",
                    "verified": False,
                }
            ],
            "logical_issues": [],
            "turn_scope_issues": [],
            "turn_scope_verdict": "pass",
            "move_alignment_verdict": "pass",
            "scope_reason": "Atomic.",
            "move_alignment_reason": "Aligned.",
            "inconclusive_checks": [],
        }
        passed_audit = {
            "approved": True,
            "deterministic_failures": [],
            "logical_issues": [],
            "turn_scope_issues": [],
            "turn_scope_verdict": "pass",
            "move_alignment_verdict": "pass",
            "scope_reason": "Atomic.",
            "move_alignment_reason": "Aligned.",
            "inconclusive_checks": [],
        }

        agent.response_verifier = SimpleNamespace(
            audit=Mock(side_effect=[failed_audit, passed_audit]),
            correction_feedback=Mock(
                return_value="Correct the false arithmetic claim."
            ),
        )

        result = TutorAgent._generate_teaching_response(
            agent,
            self.make_tutor_input(),
            evidence=[],
        )

        self.assertIn("= 30", result)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(agent.response_verifier.audit.call_count, 2)

        second_user_message = create.call_args_list[1].kwargs[
            "messages"
        ][1]["content"]
        self.assertIn(
            "PREVIOUS DRAFT FEEDBACK",
            second_user_message,
        )
        self.assertIn(
            "Correct the false arithmetic claim.",
            second_user_message,
        )


if __name__ == "__main__":
    unittest.main()
