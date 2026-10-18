"""Read-only checkpoint summary for the post-fix same-attempt retest."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


THREAD_ID = "4e578ab1-225b-4251-9f67-b837f0b3c7a8"
CHECKPOINT_DB = (
    Path(__file__).resolve().parents[2]
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "adaptive_demo"
    / "checkpoints.sqlite3"
)


def main() -> int:
    connection = sqlite3.connect(
        f"file:{CHECKPOINT_DB.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        checkpoints = list(
            SqliteSaver(connection).list(
                {"configurable": {"thread_id": THREAD_ID}},
                limit=1,
            )
        )
        if not checkpoints:
            raise RuntimeError("Post-fix retest checkpoint was not found.")

        checkpoint_tuple = checkpoints[0]
        values = checkpoint_tuple.checkpoint.get("channel_values", {})
        history = values.get("conversation_history", [])
        output = {
            "thread_id": values.get("thread_id"),
            "attempt_id": values.get("attempt_id"),
            "adaptive_attempt_index": values.get("adaptive_attempt_index"),
            "adaptive_lifecycle_status": values.get(
                "adaptive_lifecycle_status"
            ),
            "teaching_status": values.get("teaching_status"),
            "turn_count": values.get("turn_count"),
            "conversation_roles": [turn.get("role") for turn in history],
            "conversation_count": len(history),
            "pending_error_count": sum(
                channel == "__error__"
                for _, channel, _ in checkpoint_tuple.pending_writes or []
            ),
            "scheduled_nodes": sorted(
                key.removeprefix("branch:to:")
                for key, value in values.items()
                if key.startswith("branch:to:") and value is not None
            ),
        }
        print(json.dumps(output, indent=2))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
