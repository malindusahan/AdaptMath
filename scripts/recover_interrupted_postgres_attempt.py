"""Close an unrecoverable Tutor attempt without fabricating a learning reward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from verify_live_postgres_e2e import database_url, require


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    recovered_at = datetime.now(timezone.utc).isoformat()

    with psycopg.connect(database_url()) as connection:
        thread = connection.execute(
            "SELECT status FROM tutor.threads WHERE thread_id = %s FOR UPDATE",
            (args.thread_id,),
        ).fetchone()
        require(thread is not None, "Tutor thread does not exist")
        require(thread[0] == "ACTIVE", "Tutor thread is not active")
        rows = connection.execute(
            "SELECT action_event_id, payload FROM research.turn_actions "
            "WHERE thread_id = %s AND status = 'PENDING' FOR UPDATE",
            (args.thread_id,),
        ).fetchall()
        require(bool(rows), "No pending research action exists for this thread")
        for action_id, stored_payload in rows:
            payload = {
                **dict(stored_payload),
                "resolver_event_id": None,
                "mastery_before": None,
                "mastery_after": None,
                "raw_delta": None,
                "headroom_normalized_delta": None,
                "configured_reward_value": None,
                "outcome_observed": False,
                "posterior_update_occurred": False,
                "posterior_update_weight": 0.0,
                "update_id": None,
                "recovery": {
                    "kind": "interrupted_attempt_no_reward",
                    "reason": args.reason,
                    "recovered_at": recovered_at,
                },
            }
            connection.execute(
                "UPDATE research.turn_actions SET payload=%s, status='COMPLETED', "
                "completed_at=CURRENT_TIMESTAMP WHERE action_event_id=%s",
                (Jsonb(payload), action_id),
            )
        connection.execute(
            "UPDATE tutor.threads SET status='ABORTED', updated_at=CURRENT_TIMESTAMP, "
            "completed_at=CURRENT_TIMESTAMP WHERE thread_id=%s",
            (args.thread_id,),
        )
        connection.commit()
    print(f"recovered_thread={args.thread_id}")
    print(f"completed_pending_actions={len(rows)}")
    print("posterior_updates_added=0")


if __name__ == "__main__":
    main()
