import json
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.response_verifier import TeachingResponseVerifier
from app.agents.tutor.tutor_agent import TutorAgent, build_teacher_turn_schema
from app.schemas.tutor import TutorInput, TutorMathInput


def _passing_inline_turn() -> SimpleNamespace:
    payload = {
        "selected_move": "focus",
        "immediate_target": "identify the two circular regions",
        "learner_action": "compare their areas",
        "teacher_message": (
            "Focus on the area that remains after the flower bed is removed. "
            "Which two circular areas need to be compared?"
        ),
        "reveals_future_steps": False,
        "multiple_independent_requests": False,
        "blends_multiple_pedagogical_actions": False,
        "audit": {
            "checkable_claims": [],
            "logical_issues": [],
            "turn_scope_issues": [],
            "turn_scope_verdict": "pass",
            "move_alignment_verdict": "pass",
            "scope_reason": "One immediate comparison target.",
            "move_alignment_reason": "The message realizes focus.",
        },
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def test_teacher_turn_schema_requires_inline_audit() -> None:
    schema = build_teacher_turn_schema("focus")
    assert "audit" in schema["required"]
    assert schema["properties"]["audit"]["additionalProperties"] is False


def test_passing_teacher_turn_uses_one_provider_call_and_no_auditor_call() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent.model_name = "test-model"
    agent.final_teaching_prompt = "test finalizer prompt"
    agent.max_response_verification_attempts = 2

    create = Mock(return_value=_passing_inline_turn())
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    verifier = TeachingResponseVerifier.__new__(TeachingResponseVerifier)
    verifier.math_verifier = SymPyMathVerifier()
    verifier.audit = Mock(side_effect=AssertionError("unexpected auditor API call"))
    agent.response_verifier = verifier

    result = TutorAgent._generate_teaching_response(
        agent,
        TutorInput(
            question=(
                "A park has radius 12 and a central flower bed has radius 5. "
                "Find the remaining area."
            ),
            topic="Geometry",
            subtopic="Area of circles",
            student_age=15,
            complexity_score=0.3,
            pedagogical_move="focus",
            conversation_history=[],
            previous_errors=[],
            reteaching=False,
        ),
        evidence=[],
    )

    assert "two circular areas" in result
    assert create.call_count == 1
    verifier.audit.assert_not_called()


def test_mathematical_claim_gets_one_independent_auditor_call() -> None:
    turn = json.loads(_passing_inline_turn().choices[0].message.content)
    turn["teacher_message"] = "The circle-area formula is A = pi*r**2."
    turn["audit"]["checkable_claims"] = [
        {
            "source_text": "A = pi*r**2",
            "left_expression": "pi*r**2",
            "right_expression": "pi*r**2",
        }
    ]

    agent = TutorAgent.__new__(TutorAgent)
    agent.model_name = "test-model"
    agent.final_teaching_prompt = "test finalizer prompt"
    agent.max_response_verification_attempts = 2
    create = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(turn))
                )
            ]
        )
    )
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    verifier = TeachingResponseVerifier.__new__(TeachingResponseVerifier)
    verifier.math_verifier = SymPyMathVerifier()
    verifier.audit = Mock(
        return_value={
            "approved": True,
            "deterministic_failures": [],
            "logical_issues": [],
            "turn_scope_issues": [],
            "turn_scope_verdict": "pass",
            "move_alignment_verdict": "pass",
            "inconclusive_checks": [],
        }
    )
    agent.response_verifier = verifier

    TutorAgent._generate_teaching_response(
        agent,
        TutorInput(
            question="What formula gives the area of a circle?",
            topic="Geometry",
            subtopic="Area of circles",
            student_age=15,
            complexity_score=0.3,
            pedagogical_move="focus",
            conversation_history=[],
        ),
        evidence=[],
    )

    assert create.call_count == 1
    assert verifier.audit.call_count == 1


def _preparation_agent(*payloads: dict) -> TutorAgent:
    agent = TutorAgent.__new__(TutorAgent)
    agent.question_preparation_prompt = "prepare"
    agent.max_question_preparation_attempts = 2
    agent.tool_map = {}
    agent.preparation_model = SimpleNamespace(
        invoke=Mock(side_effect=list(payloads))
    )
    return agent


def _math_input() -> TutorMathInput:
    return TutorMathInput(
        question="What is 3 + 4?",
        topic="Arithmetic",
        subtopic="Addition",
        student_age=12,
        complexity_score=0.2,
    )


def test_question_preparation_combines_plan_and_math_in_one_call() -> None:
    payload = {
        "planner_output": {
            "learning_goal": "Understand addition as combining quantities.",
            "required_concepts": ["addition"],
            "teaching_sequence": ["Elicit a representation."],
            "anticipated_difficulties": [],
            "tutor_guidance": ["Keep turns atomic."],
        },
        "tool_actions": [],
        "verification_checks": [
            {"left_expression": "3+4", "right_expression": "7"}
        ],
    }
    agent = _preparation_agent(payload)

    plan, evidence = TutorAgent.prepare_question(
        agent,
        _math_input(),
        include_planner=True,
    )

    assert plan is not None
    assert plan.learning_goal.startswith("Understand addition")
    assert evidence[-1]["action"] == "final_claim_verification"
    assert agent.preparation_model.invoke.call_count == 1


def test_question_preparation_allows_at_most_one_repair_call() -> None:
    invalid = {
        "tool_actions": [],
        "verification_checks": [
            {"left_expression": "3+4", "right_expression": "8"}
        ],
    }
    repaired = {
        "tool_actions": [],
        "verification_checks": [
            {"left_expression": "3+4", "right_expression": "7"}
        ],
    }
    agent = _preparation_agent(invalid, repaired)

    plan, evidence = TutorAgent.prepare_question(
        agent,
        _math_input(),
        include_planner=False,
    )

    assert plan is None
    assert evidence[-1]["action"] == "final_claim_verification"
    assert agent.preparation_model.invoke.call_count == 2


def test_question_preparation_falls_back_after_one_failed_repair() -> None:
    invalid = {
        "tool_actions": [],
        "verification_checks": [
            {"left_expression": "3+4", "right_expression": "8"}
        ],
    }
    agent = _preparation_agent(invalid, invalid)

    plan, evidence = TutorAgent.prepare_question(
        agent,
        _math_input(),
        include_planner=True,
    )

    assert plan is not None
    assert plan.required_concepts == ["Addition"]
    assert evidence == [
        {
            "action": "verification_unavailable_fallback",
            "result": (
                "Question preparation could not establish reusable final "
                "claim checks. Avoid unsupported numerical or symbolic "
                "claims; use a problem-grounded atomic learner step. "
                "Validation categories: 2."
            ),
        }
    ]
    assert agent.preparation_model.invoke.call_count == 2


def test_question_preparation_falls_back_after_provider_failures() -> None:
    agent = _preparation_agent()
    agent.preparation_model.invoke = Mock(side_effect=TimeoutError("provider timeout"))

    plan, evidence = TutorAgent.prepare_question(
        agent,
        _math_input(),
        include_planner=True,
    )

    assert plan is not None
    assert evidence[0]["action"] == "verification_unavailable_fallback"
    assert agent.preparation_model.invoke.call_count == 2


def test_tutor_generation_failure_returns_selected_move_fallback() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent._finalize_tutor_output = Mock(side_effect=RuntimeError("invalid draft"))
    tutor_input = TutorInput(
        question="Find the remaining area.",
        topic="Geometry",
        subtopic="Area of circles",
        student_age=15,
        complexity_score=0.3,
        pedagogical_move="focus",
        conversation_history=[],
    )

    result = TutorAgent.teach_turn(agent, tutor_input, evidence=[])

    assert result.fallback_used is True
    assert "Find the remaining area" not in result.teaching_response
    assert "Area of circles" in result.teaching_response
    assert "whole quantity" not in result.teaching_response
    assert agent._finalize_tutor_output.call_count == 1


def test_elevation_focus_fallback_uses_the_actual_sign_change() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent._finalize_tutor_output = Mock(side_effect=RuntimeError("invalid draft"))
    tutor_input = TutorInput(
        question=(
            "A diver starts at -18 meters relative to sea level, descends 12 "
            "meters, then rises 7 meters. Find the final elevation."
        ),
        topic="Integers",
        subtopic="Absolute Value",
        student_age=14,
        complexity_score=0.3,
        pedagogical_move="focus",
        conversation_history=[
            {"role": "student", "content": "I don't know."},
        ],
    )

    result = TutorAgent.teach_turn(agent, tutor_input, evidence=[])

    assert result.fallback_used is True
    assert "descent" in result.teaching_response
    assert "negative" in result.teaching_response
    assert "whole quantity" not in result.teaching_response


def test_elevation_telling_fallback_works_the_learners_integer_step() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent._finalize_tutor_output = Mock(side_effect=RuntimeError("invalid draft"))
    tutor_input = TutorInput(
        question=(
            "A diver starts at -18 meters relative to sea level, descends 12 "
            "meters, then rises 7 meters. Find the final elevation."
        ),
        topic="Integers",
        subtopic="Absolute Value",
        student_age=14,
        complexity_score=0.3,
        pedagogical_move="telling",
        conversation_history=[
            {"role": "student", "content": "add -18 to -12"},
        ],
    )

    result = TutorAgent.teach_turn(agent, tutor_input, evidence=[])

    assert result.fallback_used is True
    assert "-12 + (-18) = -30" in result.teaching_response
    assert "calculation you proposed" in result.teaching_response


def test_elevation_fallback_advances_instead_of_repeating_the_sign_cue() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent._finalize_tutor_output = Mock(side_effect=RuntimeError("provider unavailable"))
    question = (
        "A diver starts at **−18 meters** relative to sea level. They descend "
        "another **12 meters**, then rise **7 meters**. What is the final "
        "elevation and its absolute value?"
    )

    identified_values = TutorAgent.teach_turn(
        agent,
        TutorInput(
            question=question,
            topic="Integers",
            subtopic="Absolute Value",
            student_age=14,
            complexity_score=0.3,
            pedagogical_move="probing",
            conversation_history=[
                {"role": "teacher", "content": "Represent the descent as negative."},
                {"role": "student", "content": "so -18 and -12"},
            ],
        ),
        evidence=[],
    )
    assert "-18 + (-12)" in identified_values.teaching_response
    assert "What signed number represents" not in identified_values.teaching_response

    confused_roles = TutorAgent.teach_turn(
        agent,
        TutorInput(
            question=question,
            topic="Integers",
            subtopic="Absolute Value",
            student_age=14,
            complexity_score=0.3,
            pedagogical_move="focus",
            conversation_history=[
                {"role": "student", "content": "so -18 and -12"},
                {"role": "teacher", "content": "What is the downward change?"},
                {"role": "student", "content": "-18"},
            ],
        ),
        evidence=[],
    )
    assert "starting elevation, not the new downward change" in (
        confused_roles.teaching_response
    )
    assert "-12" in confused_roles.teaching_response

    acknowledged = TutorAgent.teach_turn(
        agent,
        TutorInput(
            question=question,
            topic="Integers",
            subtopic="Absolute Value",
            student_age=14,
            complexity_score=0.3,
            pedagogical_move="focus",
            conversation_history=[
                {"role": "student", "content": "so -18 and -12"},
                {"role": "teacher", "content": "-18 is the start; -12 is the change."},
                {"role": "student", "content": "ok"},
            ],
        ),
        evidence=[],
    )
    assert "Evaluate -18 + (-12)" in acknowledged.teaching_response
    assert "Focus on the sign of the change" not in acknowledged.teaching_response


def test_elevation_fallback_tracks_intermediate_final_and_absolute_values() -> None:
    agent = TutorAgent.__new__(TutorAgent)
    agent._finalize_tutor_output = Mock(side_effect=RuntimeError("provider unavailable"))
    question = (
        "A diver starts at -18 meters relative to sea level, descends 12 meters, "
        "then rises 7 meters. Find the final elevation and absolute value."
    )

    cases = (
        ("-30", "focus", "-30 + 7"),
        ("-23", "probing", "|-23|"),
        ("23", "telling", "23 meters from sea level"),
    )
    for response, move, expected in cases:
        result = TutorAgent.teach_turn(
            agent,
            TutorInput(
                question=question,
                topic="Integers",
                subtopic="Absolute Value",
                student_age=14,
                complexity_score=0.3,
                pedagogical_move=move,
                conversation_history=[
                    {"role": "student", "content": response},
                ],
            ),
            evidence=[],
        )
        assert expected in result.teaching_response
