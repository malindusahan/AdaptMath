"""Idempotently import the legacy BKT SQLite state into PostgreSQL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from postgres_unification_common import connect


TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("students", ("student_id",)),
    ("mastery", ("student_id", "skill_name")),
    ("bkt_initial_priors", ("student_id", "skill_name")),
    ("sessions", ("session_id",)),
    ("attempts", ("attempt_id",)),
    ("resolved_events", ("event_id",)),
)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    payload = [
        {key: _json_value(value) for key, value in sorted(row.items())}
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def migrate(source: Path, database_url: str | None, *, dry_run: bool) -> None:
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    source_connection.row_factory = sqlite3.Row
    available = _source_tables(source_connection)
    version = int(source_connection.execute("PRAGMA user_version").fetchone()[0])
    print(f"source={source.resolve()}")
    print(f"source_schema_version={version}")

    destination = None if dry_run else connect(database_url, schema="student_model")
    try:
        for table, primary_key in TABLES:
            if table not in available:
                print(f"{table}:source=0 destination=0 match=true (source table absent)")
                continue
            source_rows = [dict(row) for row in source_connection.execute(f"SELECT * FROM {table}")]
            source_rows.sort(key=lambda row: tuple(str(row[key]) for key in primary_key))
            if dry_run:
                print(f"{table}:source={len(source_rows)} dry_run=true sha256={_hash_rows(source_rows)}")
                continue
            assert destination is not None
            for row in source_rows:
                where = " AND ".join(f"{key} = %s" for key in primary_key)
                key_values = tuple(row[key] for key in primary_key)
                existing = destination.execute(
                    f"SELECT * FROM student_model.{table} WHERE {where}", key_values
                ).fetchone()
                if existing is not None:
                    columns = [item.name for item in destination.execute(
                        f"SELECT * FROM student_model.{table} WHERE {where} LIMIT 0",
                        key_values,
                    ).description]
                    existing_map = dict(zip(columns, existing, strict=True))
                    for column, value in row.items():
                        observed = existing_map.get(column)
                        if isinstance(observed, datetime) and isinstance(value, str):
                            continue
                        if observed != value:
                            raise RuntimeError(
                                f"{table} primary key {key_values!r} already exists with different {column}."
                            )
                    continue
                columns = list(row)
                placeholders = ", ".join(["%s"] * len(columns))
                destination.execute(
                    f"INSERT INTO student_model.{table} ({', '.join(columns)}) VALUES ({placeholders})",
                    tuple(row[column] for column in columns),
                )
            if table == "attempts":
                destination.execute(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('student_model.attempts', 'attempt_id'),
                        GREATEST(COALESCE(MAX(attempt_id), 1), 1),
                        COUNT(*) > 0
                    ) FROM student_model.attempts
                    """
                )
            destination_count = destination.execute(
                f"SELECT COUNT(*) FROM student_model.{table}"
            ).fetchone()[0]
            print(
                f"{table}:source={len(source_rows)} destination={destination_count} "
                f"imported_keys={len(source_rows)} source_sha256={_hash_rows(source_rows)}"
            )
        if destination is not None:
            destination.commit()
    except Exception:
        if destination is not None:
            destination.rollback()
        raise
    finally:
        source_connection.close()
        if destination is not None:
            destination.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.source, args.database_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
