from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.api import tutor as tutor_api
from app.clients.memory.memory_client import MemoryAuthenticatedUser
from app.graph import workflow
from app.main import app
from app.integrations.adaptive_component_coordinator import _safe_bkt_demo_activity
import app.core.user_auth as user_auth


STUDENT = MemoryAuthenticatedUser(
    user_id="user-a",
    username="student-a",
    role="STUDENT",
    student_id="student-a",
    age=15,
)


class AuthClient:
    def authenticate_user(self, token: str):
        return STUDENT if token == "valid-token" else None


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: AuthClient())
    return TestClient(app, raise_server_exceptions=False)


def _snapshot(*, student_id: str = "student-a") -> SimpleNamespace:
    return SimpleNamespace(
        values={
            "student_id": student_id,
            "question": "Solve 2x + 3 = 11.",
            "age": 15,
            "topic": "Algebra",
            "subtopic": "Linear equations",
            "target_skill": "Equation Solving Two or Fewer Steps",
            "complexity_score": 0.42,
            "route": "planned_tutor",
            "route_reason": "Learner context benefits from a plan.",
            "planner_output": {"private_plan": "must not be returned"},
            "verified_math_evidence": [
                {"expected_answer": "x = 4", "private_work": "hidden"}
            ],
            "relevant_history": [{"private": "history"}],
            "previous_errors": ["sign error"],
            "previous_strategies": ["inverse operations"],
            "adaptive_turn_diagnostics": {
                "action_event_id": "thread-a:1:action:2",
                "action_turn_index": 2,
                "pedagogical_move": "probing",
                "selected_arm": "probing",
                "base_move": "focus",
                # Historical meaning was selected != base. The public trace
                # must derive a real post-selection mismatch instead.
                "overridden": True,
                "selector": {
                    "raw": {
                        "argmax": "focus",
                        "probabilities": {
                            "generic": 0.05,
                            "probing": 0.10,
                            "focus": 0.80,
                            "telling": 0.05,
                        },
                    },
                    "effective": {
                        "argmax": "focus",
                        "probabilities": {
                            "generic": 0.05,
                            "probing": 0.10,
                            "focus": 0.80,
                            "telling": 0.05,
                        },
                    },
                },
                "mrb1_scores": {
                    "Mistake_Identification": 0.7,
                    "Mistake_Location": 0.8,
                    "Providing_Guidance": 0.9,
                    "Actionability": 0.6,
                },
                "adaptive_decision": {
                    "policy_mode": "LIVE",
                    "behavior_policy": "turn_lints_live_v1",
                    "treatment_assignment_source": "turn_lints_posterior_sample",
                    "available_arms": ["generic", "probing", "focus", "telling"],
                    "sampled_scores": {
                        "generic": -0.2,
                        "probing": 0.9,
                        "focus": 0.4,
                        "telling": 0.1,
                    },
                    "policy_scores": {
                        "generic": -0.2,
                        "probing": 0.9,
                        "focus": 0.4,
                        "telling": 0.1,
                    },
                    "selected_arm": "probing",
                    "base_move": "focus",
                    "reward_mode": "headroom_normalized",
                    "policy_context": {
                        "schema_id": "turn_lints_context_v1:S+K+L:test",
                        "enabled_blocks": ["S", "K", "L"],
                        "feature_names": [
                            "selector_p_generic",
                            "selector_p_probing",
                            "selector_p_focus",
                            "selector_p_telling",
                            "mastery_before",
                            "previous_mastery_delta",
                            "previous_mastery_delta_missing",
                            "previous_reasoning_probability",
                            "previous_uncertainty_probability",
                            "previous_clarification_probability",
                            "previous_learner_signals_missing",
                        ],
                        "vector": [
                            0.05, 0.10, 0.80, 0.05,
                            0.40, 0.10, 0.00,
                            0.70, 0.20, 0.10, 0.00,
                        ],
                    },
                },
                "tutor_generation_fallback_used": False,
            },
            "move_selector_last_outcome": {
                "action_event_id": "thread-a:1:action:1",
                "resolution_status": "observed_bkt_update",
                "outcome_observed": True,
                "mastery_before": 0.30,
                "mastery_after": 0.40,
                "raw_delta": 0.10,
                "configured_reward_mode": "headroom_normalized",
                "configured_reward_value": 0.142857,
                "reward_attributed_to_move": "focus",
                "posterior_updated": True,
                "posterior_update_weight": 1.0,
                "posterior_total_updates_before": 615,
                "posterior_total_updates_after": 616,
                "update_id": "thread-a:1:action:1",
                "update_timestamp": "2026-09-01T00:00:00Z",
            },
            "bkt_last_update": {
                "schema_version": "adaptmath_bkt_activity_v2",
                "event": "dialogue_turn_bkt_update",
                "recorded_at_utc": "2026-09-01T00:00:01Z",
                "attempt_id": "thread-a:1",
                "session_id": "thread-a:1:dialogue:1",
                "target_skill": "Equation Solving Two or Fewer Steps",
                "turn_index": 1,
                "update_timing": "after_student_dialogue_turn",
                "private_answer": "must never be returned",
                "bkt_parameters": {
                    "prior": 0.2,
                    "learns": 0.1,
                    "guesses": 0.2,
                    "slips": 0.1,
                    "forgets": 0.0,
                },
                "effective_initial_prior": {
                    "probability": 0.2,
                    "source": "population",
                },
                "history": {
                    "observation_count_before_dialogue_turn": 4,
                    "observations_applied_in_dialogue_turn": 1,
                    "observation_count_after_dialogue_turn": 5,
                    "persisted_suffix_matches_resolved_updates": True,
                },
                "mastery": {
                    "before": 0.4,
                    "after": 0.5003125,
                    "delta": 0.1003125,
                    "recomputed_before": 0.4,
                    "recomputed_after": 0.5003125,
                    "before_consistent": True,
                    "after_consistent": True,
                },
                "observations": [
                    {
                        "event_id": "thread-a:1:dialogue:1:turn_1:target",
                        "source_action_event_id": "thread-a:1:action:1",
                        "skill_id": "Equation Solving Two or Fewer Steps",
                        "primary_signal": "correct_answer",
                        "resolver_version": "2.0",
                        "bkt_update": {
                            "should_update": True,
                            "outcome": 1,
                            "evidence_weight": 1.0,
                            "evaluator_confidence": 0.9,
                            "behavioural_confidence": 0.0,
                            "behaviour_factor": 1.0,
                            "update_confidence": 0.25,
                            "observation_source": "evaluator",
                            "contributors": [],
                        },
                        "behaviour": {
                            "reasoning_probability": 0.8,
                            "reasoning_present": True,
                            "uncertainty_probability": 0.1,
                            "uncertainty_present": False,
                            "clarification_probability": 0.05,
                            "clarification_present": False,
                        },
                        "repeated_misunderstanding": False,
                        "evidence": {
                            "correctness": "correct",
                            "reported_evaluator_confidence": 0.9,
                            "applied_evaluator_confidence": 0.25,
                            "evaluator_confidence_cap": 0.25,
                            "evidence_category": "mathematical_evidence",
                            "evaluator_source": "adaptmath_teaching_progress_judge",
                            "evaluator_reason": "private evaluator reason",
                            "student_text": "private learner response",
                        },
                        "mastery_transition": {
                            "mastery_before_observation": 0.4,
                            "mastery_after_observation": 0.5003125,
                            "delta_mastery": 0.1003125,
                        },
                        "persistence_status": "persisted",
                    }
                ],
                "persisted_skill_update": {
                    "skill": "Equation Solving Two or Fewer Steps",
                    "probability": 0.5003125,
                    "label": "partial",
                    "effective_initial_prior": 0.2,
                    "effective_initial_prior_source": "population",
                    "compatibility_rebased": False,
                },
            },
            "assessment_questions": [
                {
                    "question_id": "q1",
                    "question": "Private assessment question",
                    "expected_answer": "private answer",
                }
            ],
            "evaluation_result": {
                "correct_answers": [{"expected_answer": "private answer"}],
                "wrong_answers": [],
                "needs_reteaching": False,
            },
            "teaching_status": "ready_for_assessment",
            "teaching_progress_reason": "The learner demonstrated the skill.",
            "turn_count": 2,
            "reteach_round": 0,
            "attempt_id": "thread-a:1",
            "adaptive_attempt_index": 1,
        },
        next=(),
        tasks=(),
    )


def test_demo_trace_returns_counts_and_safe_diagnostics(monkeypatch):
    monkeypatch.setattr(
        tutor_api,
        "_get_persisted_snapshot",
        lambda _thread_id: _snapshot(),
    )

    response = _client(monkeypatch).get(
        "/tutor/thread-a/demo-trace",
        headers={"Authorization": "Bearer valid-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["phase"] == "complete"
    assert payload["planner_used"] is True
    assert payload["planner_output_available"] is True
    assert payload["verified_math_evidence_count"] == 1
    assert payload["assessment_question_count"] == 1
    assert payload["correct_answer_count"] == 1
    assert payload["selector"] == "focus"
    assert payload["selected_arm"] == payload["pedagogical_move"] == "probing"
    assert payload["move_overridden"] is False
    selector = payload["move_selector"]
    assert selector["context_dimension"] == 11
    assert selector["enabled_blocks"] == ["S", "K", "L"]
    assert len(selector["context_features"]) == 11
    assert selector["md7_probabilities"]["focus"] == 0.8
    assert selector["selected_move"] == selector["tutor_move"] == "probing"
    assert selector["post_selection_override"] is False
    assert selector["sampled_scores"]["probing"] == 0.9
    assert selector["latest_reward_update"]["reward_value"] == 0.142857
    assert selector["latest_reward_update"]["posterior_total_updates_after"] == 616
    bkt = payload["bkt_update"]
    assert bkt["target_skill"] == "Equation Solving Two or Fewer Steps"
    assert bkt["parameters"] == {
        "population_prior": 0.2,
        "learn": 0.1,
        "guess": 0.2,
        "slip": 0.1,
        "forget": 0.0,
    }
    assert bkt["history"]["observations_before_update"] == 4
    assert bkt["history"]["observations_after_update"] == 5
    assert bkt["observations"][0]["should_update"] is True
    assert bkt["observations"][0]["update_confidence"] == 0.25
    assert bkt["observations"][0]["persistence_status"] == "persisted"
    calculation = bkt["observations"][0]["calculation"]
    assert calculation["full_observation_posterior"] == 0.75
    assert calculation["confidence_weighted_posterior"] == pytest.approx(0.4875)
    assert calculation["matches_recorded_mastery"] is True
    assert "expected_answer" not in response.text
    assert "private answer" not in response.text
    assert "private_plan" not in response.text
    assert "private learner response" not in response.text
    assert "private evaluator reason" not in response.text
    assert "must never be returned" not in response.text


def test_demo_trace_rejects_access_to_another_students_thread(monkeypatch):
    monkeypatch.setattr(
        tutor_api,
        "_get_persisted_snapshot",
        lambda _thread_id: _snapshot(student_id="student-b"),
    )

    response = _client(monkeypatch).get(
        "/tutor/thread-b/demo-trace",
        headers={"Authorization": "Bearer valid-token"},
    )

    assert response.status_code == 403


def test_teaching_progress_checkpoints_latest_selector_reward(monkeypatch):
    outcome = {
        "action_event_id": "thread-a:1:action:1",
        "configured_reward_value": 0.25,
        "posterior_updated": True,
    }
    monkeypatch.setattr(
        workflow.progress_agent,
        "judge",
        lambda _input: SimpleNamespace(
            status="continue_teaching",
            reason="Continue with one more step.",
            latest_response_correctness="correct",
            correctness_confidence=0.9,
            correctness_reason="The step is correct.",
            latest_response_evidence_category="mathematical_evidence",
        ),
    )
    bkt_update = {
        "event": "dialogue_turn_bkt_update",
        "target_skill": "Equation Solving Two or Fewer Steps",
    }
    monkeypatch.setattr(
        workflow.adaptive_coordinator,
        "process_dialogue_turn",
        lambda *_args, **_kwargs: {
            "turn_lints_outcome": outcome,
            "bkt_last_update": bkt_update,
        },
    )

    result = workflow.teaching_progress_node(
        {
            "question": "Solve x + 2 = 5.",
            "topic": "Algebra",
            "subtopic": "Equations",
            "age": 15,
            "planner_output": None,
            "conversation_history": [
                {"role": "teacher", "content": "What should you subtract?"},
                {"role": "student", "content": "Subtract 2."},
            ],
            "previous_errors": [],
            "teaching_phase": "initial",
            "verified_math_evidence": [],
        }
    )

    assert result["move_selector_last_outcome"] == outcome
    assert result["bkt_last_update"] == bkt_update


def test_bkt_checkpoint_projection_removes_raw_learning_content():
    event_id = "thread-a:1:dialogue:1:turn_1:target"
    activity = {
        "event": "dialogue_turn_bkt_update",
        "target_skill": "Target",
        "dialogue": {
            "teacher_text": "private teacher prompt",
            "student_text": "private learner response",
            "evaluator_reason": "private evaluator explanation",
        },
        "bkt_parameters": {
            "prior": 0.2,
            "learns": 0.1,
            "guesses": 0.2,
            "slips": 0.1,
            "forgets": 0.0,
        },
        "observations": [
            {
                "dialogue_evidence": {
                    "student_text": "private learner response",
                    "teacher_text": "private teacher prompt",
                    "evaluator_reason": "private evaluator explanation",
                    "correctness": "correct",
                    "reported_evaluator_confidence": 0.9,
                },
                "resolved_signal": {
                    "event_id": event_id,
                    "skill_id": "Target",
                    "primary_signal": "correct_answer",
                    "bkt_update": {
                        "should_update": True,
                        "outcome": 1,
                        "update_confidence": 0.25,
                    },
                },
                "mastery_transition": {
                    "mastery_before_observation": 0.2,
                    "mastery_after_observation": 0.3,
                },
            }
        ],
    }
    result = {
        "knowledge_graph_result": {
            "observation_results": [
                {"event_id": event_id, "persistence_status": "persisted"}
            ]
        }
    }

    projected = _safe_bkt_demo_activity(activity, result)

    assert projected is not None
    serialized = repr(projected)
    assert "private learner response" not in serialized
    assert "private teacher prompt" not in serialized
    assert "private evaluator explanation" not in serialized
    assert projected["observations"][0]["persistence_status"] == "persisted"


def test_router_comparison_reuses_one_complexity_score_without_starting_session(
    monkeypatch,
):
    complexity_calls: list[dict] = []
    routed_states: list[dict] = []

    def fake_complexity(state):
        complexity_calls.append(dict(state))
        return {"complexity_score": 0.375}

    def fake_router(state):
        routed_states.append(dict(state))
        return {
            "route": "planned_tutor" if state["previous_errors"] else "direct_tutor",
            "route_reason": "Controlled test route.",
        }

    monkeypatch.setattr(tutor_api, "complexity_node", fake_complexity)
    monkeypatch.setattr(tutor_api, "router_node", fake_router)

    response = _client(monkeypatch).post(
        "/tutor/demo/compare-router",
        headers={"Authorization": "Bearer valid-token"},
        json={
            "question": "Solve 2x + 3 = 11.",
            "topic": "Algebra",
            "scenarios": [
                {
                    "label": "New learner",
                    "age": 8,
                    "previous_errors": [],
                    "previous_strategies": [],
                },
                {
                    "label": "Prior misconception",
                    "age": 18,
                    "previous_errors": ["sign error"],
                    "previous_strategies": [],
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert complexity_calls == [{"question": "Solve 2x + 3 = 11."}]
    assert len(routed_states) == 2
    assert {state["complexity_score"] for state in routed_states} == {0.375}
    assert [item["route"] for item in response.json()["scenarios"]] == [
        "direct_tutor",
        "planned_tutor",
    ]


def test_router_comparison_requires_authentication():
    response = TestClient(app, raise_server_exceptions=False).post(
        "/tutor/demo/compare-router",
        json={
            "question": "Solve 2x = 8.",
            "topic": "Algebra",
            "scenarios": [
                {"label": "A", "age": 8},
                {"label": "B", "age": 18},
            ],
        },
    )

    assert response.status_code == 401
