"""Focused PostgreSQL idempotency checks for research runtime persistence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import uuid


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pedagogical-move-selection"))
sys.path.insert(0, str(WORKSPACE / "adaptive-math-tutor" / "backend"))

from src.self_improvement.postgres_repository import (  # noqa: E402
    C3_FEATURES,
    add_response_quality,
    complete_action,
    load_policy_snapshot,
    stage_action,
)
from app.integrations.self_improvement_turn_collector import (  # noqa: E402
    PassiveTurnCollector,
    SCHEMA_VERSION as PASSIVE_SCHEMA_VERSION,
)
from postgres_unification_common import connect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    args = parser.parse_args()
    os.environ["RESEARCH_DATABASE_URL"] = args.database_url
    os.environ["RESEARCH_PERSISTENCE"] = "postgres"

    suffix = uuid.uuid4().hex
    action_id = f"research-verify:{suffix}:1:action:1"
    attempt_id = f"research-verify:{suffix}:1"
    update_id = f"research-verify:update:{suffix}"
    policy_key = f"research-verify:policy:{suffix}"
    context = {
        "schema_version": "C3",
        "schema_id": "C3",
        "context_dimension": 9,
        "feature_names": list(C3_FEATURES),
        "vector": [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    action = {
        "action_event_id": action_id,
        "attempt_id": attempt_id,
        "thread_id": f"research-verify:{suffix}",
        "student_id": "research-verify-student",
        "target_skill": "Percent Of",
        "action_turn_index": 1,
        "selector_version": "MD6 verification",
        "md6_probabilities": {
            "generic": 0.25,
            "probing": 0.25,
            "focus": 0.25,
            "telling": 0.25,
        },
        "selected_arm": "generic",
        "actual_move": "generic",
        "context": context,
    }
    scores = {
        "Mistake_Identification": 0.1,
        "Mistake_Location": 0.2,
        "Providing_Guidance": 0.3,
        "Actionability": 0.4,
    }
    outcome = {
        "posterior_update_occurred": True,
        "update_id": update_id,
        "selected_arm": "generic",
        "configured_reward_value": 0.05,
        "posterior_update_weight": 1.0,
        "mastery_before": 0.4,
        "mastery_after": 0.45,
        "raw_delta": 0.05,
    }
    policy = {
        "schema_version": "research_verification_v1",
        "context_dimension": 9,
        "update_count": 1,
        "arms": ["generic", "probing", "focus", "telling"],
    }
    stage_action(action)
    add_response_quality(action_id, scores, "verification response")
    complete_action(
        action_id=action_id,
        outcome=outcome,
        policy_key=policy_key,
        policy_payload=policy,
    )
    # Exact retries must be no-ops.
    stage_action(action)
    add_response_quality(action_id, scores, "verification response")
    complete_action(
        action_id=action_id,
        outcome=outcome,
        policy_key=policy_key,
        policy_payload=policy,
    )
    assert load_policy_snapshot(policy_key) == policy

    passive_action_id = f"passive-verify:{suffix}:action:1"
    passive_attempt_id = f"passive-verify:{suffix}:1"
    with tempfile.TemporaryDirectory(prefix="adaptmath-passive-pg-") as directory:
        collector = PassiveTurnCollector(directory, data_mode="real")
        passive = {
            "schema_version": PASSIVE_SCHEMA_VERSION,
            "identity": {
                "action_event_id": passive_action_id,
                "action_turn_index": 1,
                "attempt_id": passive_attempt_id,
                "thread_id": f"passive-verify:{suffix}",
                "student_pseudonymous_id": "psn_verification",
                "target_skill": "Percent Of",
            },
        }
        assert collector.stage_action(passive) is True
        assert collector.complete_action(
            passive_action_id,
            next_learner_observation={"status": "censored_no_response"},
            learning_state_update=collector._empty_learning_state_update(),
        ) is True
        assert collector.complete_action(
            passive_action_id,
            next_learner_observation={"status": "censored_no_response"},
            learning_state_update=collector._empty_learning_state_update(),
        ) is False

    connection = connect(args.database_url)
    try:
        counts = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM research.policy_updates WHERE update_id = %s),
              (SELECT COUNT(*) FROM research.c3_contexts WHERE action_event_id = %s),
              (SELECT COUNT(*) FROM research.mrb1_scores WHERE action_event_id = %s),
              (SELECT COUNT(*) FROM research.experience_records WHERE event_id = %s)
            """,
            (update_id, action_id, action_id, f"passive:turn_outcomes:{passive_action_id}"),
        ).fetchone()
    finally:
        connection.close()
    assert tuple(int(value) for value in counts) == (1, 1, 1, 1)
    print("research_postgres=true")
    print("c3_dimension=9")
    print("mrb1_completed_response_persisted=true")
    print("policy_update_exactly_once=true")
    print("passive_action_atomic_completion=true")


if __name__ == "__main__":
    main()
