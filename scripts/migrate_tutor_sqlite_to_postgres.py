"""Migrate official LangGraph SQLite checkpoints to the official PG saver."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg.rows import dict_row

from postgres_unification_common import connect


def _config_value(config: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not config:
        return default
    return str(config.get("configurable", {}).get(key, default))


def _identity_hash(items: list[Any]) -> str:
    identities = sorted(
        (
            _config_value(item.config, "thread_id"),
            _config_value(item.config, "checkpoint_ns"),
            _config_value(item.config, "checkpoint_id"),
            _config_value(item.parent_config, "checkpoint_id"),
        )
        for item in items
    )
    return hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _task_paths(connection: sqlite3.Connection) -> dict[tuple[str, str, str, str], str]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "writes" not in tables:
        return {}
    columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(writes)")]
    if "task_path" not in columns:
        return {}
    return {
        (str(thread), str(namespace), str(checkpoint), str(task)): str(path or "")
        for thread, namespace, checkpoint, task, path in connection.execute(
            "SELECT DISTINCT thread_id, checkpoint_ns, checkpoint_id, task_id, task_path FROM writes"
        )
    }


def migrate(source: Path, database_url: str | None, *, dry_run: bool) -> None:
    sqlite_connection = sqlite3.connect(
        f"file:{source.as_posix()}?mode=ro", uri=True, check_same_thread=False
    )
    source_saver = SqliteSaver(sqlite_connection)
    source_items = list(source_saver.list(None))
    source_writes = sum(len(item.pending_writes or ()) for item in source_items)
    source_hash = _identity_hash(source_items)
    print(
        f"source_checkpoints={len(source_items)} source_pending_writes={source_writes} "
        f"source_identity_sha256={source_hash} dry_run={str(dry_run).lower()}"
    )
    if dry_run:
        sqlite_connection.close()
        return

    task_paths = _task_paths(sqlite_connection)
    postgres_connection = connect(database_url, schema="tutor")
    postgres_connection.row_factory = dict_row
    postgres_connection.autocommit = True
    destination_saver = PostgresSaver(postgres_connection)
    migrated_keys: set[tuple[str, str, str]] = set()
    try:
        # Saver.list() is newest-first. Parents must exist before children so
        # pending-send/resume relationships remain identical.
        for item in reversed(source_items):
            thread_id = _config_value(item.config, "thread_id")
            namespace = _config_value(item.config, "checkpoint_ns")
            checkpoint_id = _config_value(item.config, "checkpoint_id")
            base_config = item.parent_config or {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": namespace,
                }
            }
            new_versions = dict(item.checkpoint.get("channel_versions", {}))
            destination_config = destination_saver.put(
                base_config,
                item.checkpoint,
                item.metadata,
                new_versions,
            )
            grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
            for task_id, channel, value in item.pending_writes or ():
                grouped[str(task_id)].append((str(channel), value))
            for task_id, writes in grouped.items():
                destination_saver.put_writes(
                    destination_config,
                    writes,
                    task_id,
                    task_paths.get((thread_id, namespace, checkpoint_id, task_id), ""),
                )
            migrated_keys.add((thread_id, namespace, checkpoint_id))

            channel_values = item.checkpoint.get("channel_values", {})
            if isinstance(channel_values, dict):
                student_id = channel_values.get("student_id")
                target_skill = channel_values.get("target_skill")
                if student_id:
                    postgres_connection.execute(
                        """
                        INSERT INTO tutor.threads
                            (thread_id, student_id, target_skill, status)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (thread_id) DO UPDATE SET
                            student_id = EXCLUDED.student_id,
                            target_skill = COALESCE(EXCLUDED.target_skill, tutor.threads.target_skill),
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            thread_id,
                            str(student_id),
                            None if target_skill is None else str(target_skill),
                            (
                                "COMPLETED"
                                if channel_values.get("adaptive_lifecycle_status") == "completed"
                                else "ABORTED"
                                if channel_values.get("adaptive_lifecycle_status") == "aborted"
                                else "ACTIVE"
                            ),
                        ),
                    )

        matched = 0
        for thread_id, namespace, checkpoint_id in migrated_keys:
            row = postgres_connection.execute(
                """
                SELECT 1 FROM tutor.checkpoints
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                """,
                (thread_id, namespace, checkpoint_id),
            ).fetchone()
            matched += int(row is not None)
        destination_writes = postgres_connection.execute(
            "SELECT COUNT(*) AS count FROM tutor.checkpoint_writes"
        ).fetchone()["count"]
        print(
            f"destination_matching_checkpoints={matched} match={str(matched == len(source_items)).lower()} "
            f"destination_total_pending_writes={destination_writes}"
        )
    finally:
        postgres_connection.close()
        sqlite_connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.source, args.database_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

