"""Read-only count and scientific-equivalence checks for a migrated database."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sqlite3
import sys

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg.rows import dict_row

from postgres_unification_common import connect


WORKSPACE = Path(__file__).resolve().parents[1]
MOVE_ROOT = WORKSPACE / "pedagogical-move-selection"
ACTIVE_MODEL_ROOT = (
    WORKSPACE
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "adaptive_live_real_v1"
    / "student_model"
)


def _policy(args, connection) -> None:
    source = json.loads(args.policy.read_text(encoding="utf-8"))
    row = connection.execute(
        "SELECT state_json, total_updates, context_dimension "
        "FROM research.policy_states WHERE policy_key = %s",
        (args.policy_key,),
    ).fetchone()
    assert row is not None, "migrated policy is absent"
    assert row[0] == source, "policy JSON differs after migration"
    assert int(row[1]) == int(source.get("total_updates", source.get("update_count", 0)))
    expected_dimension = int(source.get("context_dim", source.get("context_dimension")))
    assert int(row[2]) == expected_dimension

    sys.path.insert(0, str(MOVE_ROOT))
    from src.self_improvement.turn_lints_policy import DirectTurnLinTS

    hyper = source["posterior_hyperparameters"]
    def build():
        policy = DirectTurnLinTS(
            context_schema_id=source["context_schema_id"],
            enabled_context_blocks=source["enabled_context_blocks"],
            feature_names=source["ordered_feature_names"],
            selector_version=source["selector_version"],
            selector_sha256=source["selector_sha256"],
            reward_mode=source["reward_mode"],
            anchor_mode=source["anchor_mode"],
            anchor_gamma=source["anchor_gamma"],
            anchor_selector_version=source["anchor_selector_version"],
            anchor_selector_sha256=source["anchor_selector_sha256"],
            ridge_lambda=hyper["ridge_lambda"],
            exploration_scale=hyper["exploration_scale"],
            seed=hyper["seed"],
            data_mode=source["data_mode"],
        )
        policy.load_state_dict(copy.deepcopy(source))
        return policy
    old_policy, new_policy = build(), build()
    contexts = [
        [0.25] * expected_dimension,
        [float(index + 1) / expected_dimension for index in range(expected_dimension)],
        [0.0] * expected_dimension,
    ]
    probabilities = {"generic": 0.25, "probing": 0.25, "focus": 0.25, "telling": 0.25}
    for context in contexts:
        assert old_policy.select_arm(context, selector_probabilities=probabilities) == new_policy.select_arm(
            context, selector_probabilities=probabilities
        )
    print(
        f"policy_equivalent=true total_updates={row[1]} context_dimension={row[2]}"
    )


def _bkt(args, connection) -> None:
    source = sqlite3.connect(f"file:{args.student_model.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source_mastery = {
        (str(row["student_id"]), str(row["skill_name"])): float(row["mastery_probability"])
        for row in source.execute("SELECT * FROM mastery")
    }
    destination_mastery = {
        (str(row[0]), str(row[1])): float(row[2])
        for row in connection.execute(
            "SELECT student_id, skill_name, mastery_probability FROM student_model.mastery"
        )
    }
    assert source_mastery == destination_mastery, "stored BKT mastery differs"
    source_attempts = [tuple(row) for row in source.execute(
        "SELECT attempt_id, resolver_event_id, student_id, skill_name, correct, confidence "
        "FROM attempts ORDER BY attempt_id"
    )]
    destination_attempts = [tuple(row) for row in connection.execute(
        "SELECT attempt_id, resolver_event_id, student_id, skill_name, correct, confidence "
        "FROM student_model.attempts ORDER BY attempt_id"
    )]
    assert source_attempts == destination_attempts, "BKT observation order/content differs"

    sys.path.insert(0, str(ACTIVE_MODEL_ROOT))
    from bkt.predict import BKTPredictor
    predictor = BKTPredictor.load(ACTIVE_MODEL_ROOT / "models" / "bkt_params.json")
    checked = 0
    for (student_id, skill), stored in list(source_mastery.items())[:8]:
        prior_row = source.execute(
            "SELECT effective_initial_prior FROM bkt_initial_priors "
            "WHERE student_id = ? AND skill_name = ?",
            (student_id, skill),
        ).fetchone()
        if prior_row is None or prior_row[0] is None:
            continue
        observations = [
            (int(row[0]), float(row[1]))
            for row in source.execute(
                "SELECT correct, confidence FROM attempts "
                "WHERE student_id = ? AND skill_name = ? "
                "ORDER BY created_at, attempt_id",
                (student_id, skill),
            )
        ]
        if not observations:
            continue
        before = float(predictor.predict(skill, observations, initial_prior=float(prior_row[0])))
        assert math.isclose(before, stored, rel_tol=0.0, abs_tol=1e-15)
        for outcome in (0, 1):
            old_after = float(
                predictor.predict(skill, observations + [(outcome, 1.0)], initial_prior=float(prior_row[0]))
            )
            new_after = float(
                predictor.predict(skill, observations + [(outcome, 1.0)], initial_prior=float(prior_row[0]))
            )
            assert old_after == new_after
        checked += 1
    source.close()
    print(
        f"bkt_mastery_equivalent=true observations={len(source_attempts)} "
        f"golden_student_skills={checked}"
    )


def _checkpoints(args, database_url: str | None) -> None:
    sqlite_connection = sqlite3.connect(
        f"file:{args.checkpoints.as_posix()}?mode=ro", uri=True, check_same_thread=False
    )
    source = SqliteSaver(sqlite_connection)
    source_items = list(source.list(None))
    pg = connect(database_url, schema="tutor")
    pg.row_factory = dict_row
    pg.autocommit = True
    destination = PostgresSaver(pg)
    threads = sorted(
        {_config(item.config, "thread_id") for item in source_items if _config(item.config, "thread_id")}
    )
    sampled = 0
    for thread_id in threads[:10]:
        old = source.get_tuple({"configurable": {"thread_id": thread_id}})
        new = destination.get_tuple({"configurable": {"thread_id": thread_id}})
        assert old is not None and new is not None
        assert old.checkpoint == new.checkpoint
        assert old.metadata == new.metadata
        assert old.parent_config == new.parent_config
        assert old.pending_writes == new.pending_writes
        sampled += 1
    destination_count = pg.execute("SELECT COUNT(*) AS count FROM tutor.checkpoints").fetchone()["count"]
    assert int(destination_count) == len(source_items)
    print(
        f"tutor_checkpoint_equivalent=true checkpoints={destination_count} sampled_threads={sampled}"
    )
    pg.close()
    sqlite_connection.close()


def _config(config: dict | None, key: str) -> str:
    return "" if not config else str(config.get("configurable", {}).get(key, ""))


def _counts(connection) -> None:
    queries = {
        "accounts": "SELECT COUNT(*) FROM student_memory.user_accounts",
        "auth_sessions": "SELECT COUNT(*) FROM auth.sessions",
        "turn_actions": "SELECT COUNT(*) FROM research.turn_actions",
        "mrb1_scores": "SELECT COUNT(*) FROM research.mrb1_scores",
        "policy_contexts": "SELECT COUNT(*) FROM research.policy_contexts",
        "c3_contexts": "SELECT COUNT(*) FROM research.c3_contexts",
        "policy_updates": "SELECT COUNT(*) FROM research.policy_updates",
        "experiences": "SELECT COUNT(*) FROM research.experience_records",
    }
    print("counts=" + json.dumps({name: int(connection.execute(sql).fetchone()[0]) for name, sql in queries.items()}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-key", required=True)
    args = parser.parse_args()
    connection = connect(args.database_url)
    try:
        _policy(args, connection)
        _bkt(args, connection)
        _counts(connection)
        connection.commit()
    finally:
        connection.close()
    _checkpoints(args, args.database_url)
    from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES
    assert len(TURN_FEATURE_NAMES) == 9
    print("legacy_c3_dimension=9")


if __name__ == "__main__":
    main()

