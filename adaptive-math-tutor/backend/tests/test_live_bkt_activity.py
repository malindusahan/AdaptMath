from __future__ import annotations

import json

import pytest

from app.integrations.live_bkt_activity import LiveBKTActivityLog


class _Predictor:
    params = {
        "Target": {
            "prior": 0.2,
            "learns": 0.1,
            "guesses": 0.2,
            "slips": 0.1,
            "forgets": 0.0,
        }
    }

    @staticmethod
    def predict(skill, observations, *, initial_prior):
        assert skill == "Target"
        return float(
            initial_prior
            + sum((0.05 if outcome else -0.01) * confidence
                  for outcome, confidence in observations)
        )


class _Graph:
    predictor = _Predictor()
    observations = [(0, 0.2)]

    @classmethod
    def get_attempts(cls, student_id, skill):
        assert (student_id, skill) == ("student-1", "Target")
        return list(cls.observations)

    @staticmethod
    def get_effective_initial_prior(student_id, skill):
        assert (student_id, skill) == ("student-1", "Target")
        return {
            "effective_initial_prior": 0.2,
            "prior_source": "population",
        }


def _dialogue_activity():
    mastery_after = _Predictor.predict(
        "Target",
        _Graph.observations,
        initial_prior=0.2,
    )
    return {
        "student_id": "student-1",
        "attempt_id": "attempt-1",
        "target_skill": "Target",
        "dialogue_evidence": {
            "turn_index": 1,
            "teacher_text": "What would you try first?",
            "student_text": "I do not know.",
            "correctness": "unknown",
        },
        "student_model_result": {
            "incremental_learning_outcome": {
                "mastery_before": 0.2,
                "mastery_after": mastery_after,
                "delta_mastery": mastery_after - 0.2,
            },
            "resolved_events": [
                {
                    "event_id": "attempt-1:dialogue:1:turn_1:target",
                    "student_id": "student-1",
                    "skill_id": "Target",
                    "primary_signal": "behavioural_difficulty",
                    "resolver_version": "2.0",
                    "behaviour": {"uncertainty_present": True},
                    "history": {"repeated_misunderstanding": False},
                    "bkt_update": {
                        "should_update": True,
                        "outcome": 0,
                        "update_confidence": 0.2,
                        "observation_source": "behavioural_proxy",
                    },
                }
            ],
            "knowledge_graph_result": {
                "session_id": "attempt-1:dialogue:1",
                "skills_updated": [
                    {
                        "skill": "Target",
                        "probability": mastery_after,
                        "label": "weak",
                    }
                ],
            },
        },
        "knowledge_graph": _Graph(),
    }


def test_live_bkt_activity_is_append_only_and_restart_idempotent(tmp_path):
    activity = _dialogue_activity()
    log = LiveBKTActivityLog(tmp_path)

    first = log.record_dialogue_bkt_activity(**activity)
    assert first["schema_version"] == "adaptmath_bkt_activity_v2"
    assert first["event"] == "dialogue_turn_bkt_update"
    assert first["mastery"]["after_consistent"] is True

    restarted = LiveBKTActivityLog(tmp_path)
    duplicate = restarted.record_dialogue_bkt_activity(**activity)
    assert duplicate == first
    lines = (tmp_path / "bkt_activity.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == first


def test_live_bkt_activity_rejects_incompatible_existing_schema(tmp_path):
    path = tmp_path / "bkt_activity.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "unknown",
                "event": "dialogue_turn_bkt_update",
                "attempt_id": "attempt-1",
                "turn_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible schema"):
        LiveBKTActivityLog(tmp_path)
