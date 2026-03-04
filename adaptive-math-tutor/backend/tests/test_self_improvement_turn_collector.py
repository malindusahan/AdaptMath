from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

MOVE_ROOT = Path(__file__).resolve().parents[3] / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from app.integrations.self_improvement_turn_collector import (
    PassiveTurnCollector,
    SCHEMA_VERSION,
    TurnCollectionIntegrityError,
)
from src.self_improvement.lints_policy import TrueDisjointLinTS


RAW = {
    "generic": 0.10,
    "probing": 0.60,
    "focus": 0.20,
    "telling": 0.10,
}
EFFECTIVE = {
    "generic": 0.0,
    "probing": 0.0,
    "focus": 0.0,
    "telling": 1.0,
}
MRB1 = {
    "Mistake_Identification": 0.8,
    "Mistake_Location": 0.7,
    "Providing_Guidance": 0.9,
    "Actionability": 0.6,
}
MOVE_ORDER = ["generic", "probing", "focus", "telling"]


def _version_provenance():
    return {
        "runtime": {
            "selector_mode": "ordinary-md7r1-v1",
            "selector_deployment_status": "active_runtime",
            "learner_agency_version": "learner_agency_telling_escalation_v1",
            "lints_policy_lineage": "MD7-R1 fresh LinTS v1",
            "lints_policy_version": "true_disjoint_lints_v3",
            "c3_version": "turn_lints_v3_c3_9d",
            "tau": 0.10,
            "move_order": MOVE_ORDER,
            "resolver_version": "2.0",
            "bkt_config_version": "confidence_weighted_bkt_v1",
            "bkt_config_sha256": "bkt-sha256",
        },
        "selector": {
            "checkpoint": "md7r1_epoch3",
            "model_sha256": "md7-sha256",
            "selector_mode": "ordinary-md7r1-v1",
            "deployment_status": "active_runtime",
            "learner_agency_version": "learner_agency_telling_escalation_v1",
            "lints_policy_lineage": "MD7-R1 fresh LinTS v1",
            "lints_policy_version": "true_disjoint_lints_v3",
            "c3_version": "turn_lints_v3_c3_9d",
            "tau": 0.10,
            "move_order": MOVE_ORDER,
        },
        "tutor_quality": {
            "model_version": "frozen_mrb1",
            "model_sha256": "mrb1-sha256",
        },
        "policy": {
            "policy_lineage": "MD7-R1 fresh LinTS v1",
            "policy_state_identifier": (
                "ordinary-md7r1-v1:MD7-R1 fresh LinTS v1"
            ),
            "policy_state_relative_path": (
                "adaptive-math-tutor/backend/runtime/"
                "adaptive_demo_md7r1_real_v1/policy_state.json"
            ),
        },
    }


def _state():
    return {
        "student_id": "account-not-written",
        "thread_id": "thread-1",
        "question": "Solve x + 2 = 5.",
        "target_skill": "algebra",
        "teaching_phase": "initial",
    }


def _turn(action_id="thread-1:1:action:1"):
    return {
        "attempt_id": "thread-1:1",
        "adaptive_attempt_index": 1,
        "action_event_id": action_id,
        "action_turn_index": 1,
        "thread_turn_count": 4,
        "attempt_start_mastery": 0.2,
        "mastery_at_action": 0.41,
        "tutor_response": "Subtract two from both sides.",
        "mrb1_scores": MRB1,
        "selector": {
            "raw": {"probabilities": RAW, "argmax": "probing"},
            "learner_agency": {
                "triggered": True,
                "reason": "explicit_answer_request",
            },
            "effective": {
                "probabilities": EFFECTIVE,
                "argmax": "telling",
            },
        },
        "adaptive_decision": {
            "context_name": "C3",
            "context_feature_names": [f"f{i}" for i in range(9)],
            "context": [0.0] * 9,
            "eligible_arms": ["baseline", "telling_bias"],
            "sampled_scores": {"baseline": 0.1, "telling_bias": 0.2},
            "selected_arm": "telling_bias",
            "base_move": "telling",
            "final_move": "focus",
            "overridden": True,
            "gap": 0.02,
            "gap_threshold": 0.10,
        },
    }


def _result(action_id, *, should_update=True):
    event_id = f"{action_id}:resolver:algebra"
    return {
        "resolved_events": [
            {
                "event_id": event_id,
                "source_action_event_id": action_id,
                "skill_id": "algebra",
                "primary_signal": (
                    "correct_answer" if should_update else "no_update"
                ),
                "resolver_version": "2.0",
                "bkt_update": {
                    "should_update": should_update,
                    "outcome": 1 if should_update else None,
                    "evidence_weight": 1.0 if should_update else 0.0,
                    "behaviour_factor": 1.0,
                    "update_confidence": 0.5 if should_update else 0.0,
                    "observation_source": (
                        "evaluator" if should_update else "none"
                    ),
                    "contributors": [],
                },
            }
        ],
        "knowledge_graph_result": {
            "observation_results": [
                {
                    "event_id": event_id,
                    "source_action_event_id": action_id,
                    "skill": "algebra",
                    "should_update": should_update,
                    "observation_source": (
                        "evaluator" if should_update else "none"
                    ),
                    "outcome": 1 if should_update else None,
                    "update_confidence": 0.5 if should_update else 0.0,
                    "mastery_before": 0.41,
                    "mastery_after": 0.52 if should_update else 0.41,
                    "delta_mastery": 0.11 if should_update else 0.0,
                }
            ]
        },
    }


def _evidence(action_id):
    return {
        "action_event_id": action_id,
        "student_text": "x is 3",
        "correctness": "correct",
        "reported_evaluator_confidence": 0.9,
        "applied_evaluator_confidence": 0.5,
        "evaluator_reason": "The equation is solved correctly.",
        "evaluator_source": "adaptmath_teaching_progress_judge",
    }


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_observed_turn_preserves_raw_effective_final_mastery_and_mrb1(tmp_path):
    collector = PassiveTurnCollector(
        tmp_path,
        data_mode="real",
        version_provenance=_version_provenance(),
    )
    state = _state()
    turn = _turn()
    before_state = copy.deepcopy(state)
    before_turn = copy.deepcopy(turn)
    collector.stage_runtime_action(
        state=state,
        history_before_action=[{"user": "student", "text": "Please answer."}],
        turn_result=turn,
    )
    collector.complete_dialogue_action(
        action_event_id=turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        student_model_result=_result(turn["action_event_id"]),
    )
    assert state == before_state
    assert turn == before_turn

    record = _rows(collector.turn_path)[0]
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["provenance"]["data_mode"] == "real"
    assert record["selector"]["raw"]["probabilities"] == RAW
    assert record["selector"]["effective"]["probabilities"] == EFFECTIVE
    assert record["selector"]["move_order"] == MOVE_ORDER
    assert record["adaptive_decision"]["final_move"] == "focus"
    assert record["state_before_action"]["attempt_start_mastery"] == 0.2
    assert record["state_before_action"]["mastery_at_action"] == 0.41
    assert record["tutor_quality"]["scores"] == MRB1
    assert record["next_learner_observation"]["status"] == "observed_update"
    assert record["learning_state_update"]["delta_mastery"] == 0.11
    assert "account-not-written" not in collector.turn_path.read_text(
        encoding="utf-8"
    )


def test_resolved_no_update_is_retained_with_zero_delta(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="synthetic")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.complete_dialogue_action(
        action_event_id=turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        student_model_result=_result(
            turn["action_event_id"],
            should_update=False,
        ),
    )
    record = _rows(collector.turn_path)[0]
    assert record["next_learner_observation"]["status"] == "resolved_no_update"
    assert record["learning_state_update"]["should_update"] is False
    assert record["learning_state_update"]["mastery_before"] == 0.41
    assert record["learning_state_update"]["mastery_after"] == 0.41
    assert record["learning_state_update"]["delta_mastery"] == 0.0


@pytest.mark.parametrize("status", ["censored_no_response", "processing_failed"])
def test_terminal_no_evidence_states_never_fabricate_bkt(tmp_path, status):
    collector = PassiveTurnCollector(tmp_path, data_mode="synthetic")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.finalize_without_response(
        turn["action_event_id"],
        status=status,
        failure_reason="fixture" if status == "processing_failed" else None,
    )
    record = _rows(collector.turn_path)[0]
    assert record["next_learner_observation"]["status"] == status
    assert record["learning_state_update"]["resolver_event_id"] is None
    assert record["learning_state_update"]["outcome"] is None
    assert record["learning_state_update"]["delta_mastery"] is None


def test_duplicate_action_is_idempotent_and_jsonl_stays_parseable(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    first = collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    assert collector.stage_action(first) is False
    collector.finalize_without_response(turn["action_event_id"])
    assert len(_rows(collector.turn_path)) == 1


def test_duplicate_action_identity_rejects_conflicting_action_content(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    staged = collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.finalize_without_response(turn["action_event_id"])
    conflicting = copy.deepcopy(staged)
    conflicting["tutor_response"] = "Conflicting response."
    with pytest.raises(TurnCollectionIntegrityError, match="different action"):
        collector.stage_action(conflicting)


def test_processing_failure_is_durable_and_distinct_from_censoring(tmp_path):
    first = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    first.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    first.note_processing_failure(
        turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        failure_reason="student_model_processing_failed:RuntimeError",
    )

    recovered = PassiveTurnCollector(tmp_path, data_mode="real")
    row = _rows(recovered.turn_path)[0]
    assert row["next_learner_observation"]["status"] == "processing_failed"
    assert row["next_learner_observation"]["student_text"] == "x is 3"
    assert row["learning_state_update"]["resolver_event_id"] is None


def test_controlled_restart_preserves_and_reads_pending_action(tmp_path):
    first = PassiveTurnCollector(
        tmp_path,
        data_mode="real",
        preserve_pending_for_restart=True,
    )
    turn = _turn()
    first.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )

    recovered = PassiveTurnCollector(
        tmp_path,
        data_mode="real",
        preserve_pending_for_restart=True,
    )

    assert not recovered.turn_path.exists()
    assert recovered._pending_path(turn["action_event_id"]).exists()
    actions = recovered.finalized_actions_for_attempt(turn["attempt_id"])
    assert [row["identity"]["action_event_id"] for row in actions] == [
        turn["action_event_id"]
    ]


def test_successful_retry_replaces_provisional_processing_failure(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.note_processing_failure(
        turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        failure_reason="student_model_processing_failed:RuntimeError",
    )
    collector.complete_dialogue_action(
        action_event_id=turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        student_model_result=_result(turn["action_event_id"]),
    )

    row = _rows(collector.turn_path)[0]
    assert row["next_learner_observation"]["status"] == "observed_update"
    assert not list(collector.failure_dir.glob("*.json"))


def test_completed_outcome_replay_after_cleanup_is_idempotent(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    kwargs = {
        "action_event_id": turn["action_event_id"],
        "dialogue_evidence": _evidence(turn["action_event_id"]),
        "student_model_result": _result(turn["action_event_id"]),
    }
    assert collector.complete_dialogue_action(**kwargs) is True
    assert collector.complete_dialogue_action(**kwargs) is False
    assert len(_rows(collector.turn_path)) == 1


def test_restart_after_final_append_before_cleanup_keeps_final_result(
    tmp_path,
    monkeypatch,
):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    pending_path = collector._pending_path(turn["action_event_id"])
    original_unlink = Path.unlink
    failed = False

    def fail_pending_cleanup(path, *args, **kwargs):
        nonlocal failed
        if path == pending_path and not failed:
            failed = True
            raise OSError("simulated crash before pending cleanup")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending_cleanup)
    with pytest.raises(OSError, match="simulated crash"):
        collector.complete_dialogue_action(
            action_event_id=turn["action_event_id"],
            dialogue_evidence=_evidence(turn["action_event_id"]),
            student_model_result=_result(turn["action_event_id"]),
        )
    assert pending_path.exists()
    assert len(_rows(collector.turn_path)) == 1

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = PassiveTurnCollector(tmp_path, data_mode="real")
    row = _rows(recovered.turn_path)[0]
    assert row["next_learner_observation"]["status"] == "observed_update"
    assert not pending_path.exists()

    assert collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    assert len(_rows(collector.turn_path)) == 1


def test_optional_diagnostics_may_be_absent(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="synthetic")
    turn = _turn()
    turn.pop("selector")
    turn.pop("adaptive_decision")
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.finalize_without_response(turn["action_event_id"])
    record = _rows(collector.turn_path)[0]
    assert record["selector"] == {}
    assert record["adaptive_decision"] == {}


def test_formal_assessment_is_a_separate_attempt_level_artifact(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.finalize_without_response(turn["action_event_id"])
    collector.record_assessment_outcome(
        {
            "identity": {"attempt_id": "thread-1:1"},
            "evidence_type": "formal_three_question_assessment",
            "observation_results": [{"event_id": "assessment-1"}],
        }
    )
    assert len(_rows(collector.turn_path)) == 1
    assert len(_rows(collector.assessment_path)) == 1
    assert "assessment-1" not in collector.turn_path.read_text(encoding="utf-8")


def test_future_assessment_and_summary_have_portable_standalone_provenance(
    tmp_path,
):
    collector = PassiveTurnCollector(
        tmp_path,
        data_mode="real",
        version_provenance=_version_provenance(),
    )
    started_at = "2026-08-28T02:00:00.123456Z"
    collector.record_assessment_outcome(
        {
            "provenance": {"timestamp": "2026-08-28T02:10:00Z"},
            "identity": {"attempt_id": "thread-1:1"},
            "attempt_started_at": started_at,
            "evidence_type": "formal_three_question_assessment",
            "observation_results": [],
        }
    )
    source_summary = {
        "provenance": {"timestamp": "2026-08-28T02:11:00Z"},
        "identity": {"attempt_id": "thread-1:1"},
        "attempt_started_at": started_at,
        "completion_status": "completed",
        "adaptive_completion": {
            "reward": 0.1,
            "policy_state_path": (
                r"C:\Users\researcher\private\runtime\policy_state.json"
            ),
        },
    }
    before = copy.deepcopy(source_summary)
    collector.record_attempt_summary(source_summary)

    assessment = _rows(collector.assessment_path)[0]
    summary = _rows(collector.attempt_path)[0]
    assert source_summary == before
    assert summary["attempt_started_at"] == started_at
    assert summary["completion_status"] == "completed"
    assert "policy_state_path" not in summary["adaptive_completion"]
    raw_summary = collector.attempt_path.read_text(encoding="utf-8")
    assert r"C:\Users" not in raw_summary
    assert "researcher" not in raw_summary

    for artifact in (assessment, summary):
        provenance = artifact["provenance"]
        assert provenance["data_mode"] == "real"
        assert provenance["selector_mode"] == "ordinary-md7r1-v1"
        assert provenance["active_selector"]["checkpoint"] == "md7r1_epoch3"
        assert provenance["active_selector"]["model_sha256"] == "md7-sha256"
        assert provenance["active_selector"]["move_order"] == MOVE_ORDER
        assert provenance["learner_agency_version"] == (
            "learner_agency_telling_escalation_v1"
        )
        assert provenance["lints_policy_version"] == "true_disjoint_lints_v3"
        assert provenance["lints_policy_lineage"] == "MD7-R1 fresh LinTS v1"
        assert provenance["c3_version"] == "turn_lints_v3_c3_9d"
        assert provenance["tau"] == 0.10
        assert provenance["tutor_quality"] == {
            "model_version": "frozen_mrb1",
            "model_sha256": "mrb1-sha256",
        }
        assert provenance["resolver_version"] == "2.0"
        assert provenance["bkt_config_version"] == "confidence_weighted_bkt_v1"
        assert provenance["bkt_config_sha256"] == "bkt-sha256"
        assert provenance["policy"]["policy_state_relative_path"].endswith(
            "adaptive_demo_md7r1_real_v1/policy_state.json"
        )


def test_misconfigured_policy_provenance_cannot_serialize_absolute_path(tmp_path):
    provenance = _version_provenance()
    provenance["policy"]["policy_state_relative_path"] = (
        r"C:\Users\researcher\private\policy_state.json"
    )
    collector = PassiveTurnCollector(
        tmp_path,
        data_mode="real",
        version_provenance=provenance,
    )
    collector.record_attempt_summary(
        {
            "identity": {"attempt_id": "thread-1:1"},
            "attempt_started_at": "2026-08-28T02:00:00Z",
            "completion_status": "completed",
        }
    )
    summary = _rows(collector.attempt_path)[0]
    assert summary["provenance"]["policy"]["policy_state_relative_path"] is None
    assert "researcher" not in collector.attempt_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("started_at", "status"),
    [
        ("2026-08-28T02:00:00+05:30", "completed"),
        ("2026-08-28T02:00:00Z", "active"),
    ],
)
def test_future_summary_rejects_non_utc_start_or_nonterminal_status(
    tmp_path,
    started_at,
    status,
):
    collector = PassiveTurnCollector(tmp_path, data_mode="real")
    with pytest.raises(ValueError):
        collector.record_attempt_summary(
            {
                "identity": {"attempt_id": "thread-1:1"},
                "attempt_started_at": started_at,
                "completion_status": status,
            }
        )


def test_restart_recovers_staged_action_as_censored(tmp_path):
    first = PassiveTurnCollector(tmp_path, data_mode="real")
    turn = _turn()
    first.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    second = PassiveTurnCollector(tmp_path, data_mode="real")
    record = _rows(second.turn_path)[0]
    assert record["next_learner_observation"]["status"] == (
        "censored_no_response"
    )
    assert not list(second.pending_dir.glob("*.json"))


def test_collector_does_not_change_move_lints_or_bkt_result(tmp_path):
    collector = PassiveTurnCollector(tmp_path, data_mode="synthetic")
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    policy_before = copy.deepcopy(policy.state_dict())
    turn = _turn()
    bkt_result = _result(turn["action_event_id"])
    bkt_before = copy.deepcopy(bkt_result)
    final_move = turn["adaptive_decision"]["final_move"]

    collector.stage_runtime_action(
        state=_state(),
        history_before_action=[],
        turn_result=turn,
    )
    collector.complete_dialogue_action(
        action_event_id=turn["action_event_id"],
        dialogue_evidence=_evidence(turn["action_event_id"]),
        student_model_result=bkt_result,
    )

    assert turn["adaptive_decision"]["final_move"] == final_move
    assert policy.state_dict() == policy_before
    assert bkt_result == bkt_before
