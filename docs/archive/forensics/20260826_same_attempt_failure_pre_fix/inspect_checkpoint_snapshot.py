"""Read-only, content-safe summary of the preserved failing Tutor checkpoints."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


THREAD_ID = "9c909170-00a6-4a33-8ad8-9ef003c2102e"
SNAPSHOT = Path(__file__).with_name("checkpoints.snapshot.sqlite3").resolve()
SAFE_STATE_FIELDS = (
    "thread_id",
    "attempt_id",
    "adaptive_attempt_index",
    "student_id",
    "target_skill",
    "teaching_phase",
    "teaching_status",
    "turn_count",
    "adaptive_lifecycle_status",
    "completed_attempt_id",
)


def _safe_error(value: object) -> object:
    if isinstance(value, BaseException):
        return {"class": type(value).__name__, "message": str(value)}
    if isinstance(value, str):
        return value[:1000]
    return type(value).__name__


def main() -> int:
    connection = sqlite3.connect(
        f"file:{SNAPSHOT.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        saver = SqliteSaver(connection)
        tuples = list(
            saver.list({"configurable": {"thread_id": THREAD_ID}})
        )
        output: list[dict[str, object]] = []
        for checkpoint_tuple in reversed(tuples):
            checkpoint = checkpoint_tuple.checkpoint
            values = checkpoint.get("channel_values", {})
            metadata = checkpoint_tuple.metadata or {}
            pending: list[dict[str, object]] = []
            for task_id, channel, value in checkpoint_tuple.pending_writes or []:
                entry: dict[str, object] = {
                    "task_id": task_id,
                    "channel": channel,
                }
                if channel == "__error__":
                    entry["error"] = _safe_error(value)
                pending.append(entry)

            conversation = values.get("conversation_history", [])
            output.append(
                {
                    "checkpoint_id": checkpoint.get("id"),
                    "timestamp": checkpoint.get("ts"),
                    "metadata_source": metadata.get("source"),
                    "metadata_step": metadata.get("step"),
                    "metadata_write_nodes": sorted(
                        str(key) for key in (metadata.get("writes") or {})
                    ),
                    "state": {
                        field: values.get(field) for field in SAFE_STATE_FIELDS
                    },
                    "conversation_roles": [
                        turn.get("role")
                        for turn in conversation
                        if isinstance(turn, dict)
                    ],
                    "pending_write_channels": pending,
                    "scheduled_nodes": sorted(
                        key.removeprefix("branch:to:")
                        for key, value in values.items()
                        if key.startswith("branch:to:") and value is not None
                    ),
                }
            )
        print(json.dumps(output, indent=2, default=str))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
