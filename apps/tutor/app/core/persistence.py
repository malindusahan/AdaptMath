"""Feature-gated LangGraph checkpoint persistence.

SQLite remains available as an explicit rollback mode. PostgreSQL uses the
official ``langgraph-checkpoint-postgres`` saver and the schema provisioned by
Alembic; normal Tutor startup never calls ``setup()`` or creates tables.
"""

from __future__ import annotations

import atexit
import os
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import get_settings


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_CLOSE_LOCK = Lock()
_CLOSED = False
_SQLITE_CONNECTION: sqlite3.Connection | None = None
_POSTGRES_POOL: Any | None = None


def _resolve_checkpoint_path(*, create_parent: bool) -> Path:
    configured = Path(get_settings().checkpoint_db_path).expanduser()
    resolved = (
        configured
        if configured.is_absolute()
        else (_BACKEND_ROOT / configured).resolve()
    )
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _postgres_conninfo() -> str:
    settings = get_settings()
    if settings.adaptmath_database_url is None:
        raise RuntimeError(
            "ADAPTMATH_DATABASE_URL is required when TUTOR_PERSISTENCE=postgres."
        )
    value = settings.adaptmath_database_url.get_secret_value().strip()
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value.removeprefix("postgresql+psycopg://")
    if not value.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("ADAPTMATH_DATABASE_URL must be a PostgreSQL URL.")
    separator = "&" if "?" in value else "?"
    return value + separator + "options=-c%20search_path%3Dtutor%2Cpublic"


PERSISTENCE_MODE = get_settings().tutor_persistence
CHECKPOINT_DB_PATH = _resolve_checkpoint_path(
    create_parent=PERSISTENCE_MODE == "sqlite"
)

if PERSISTENCE_MODE == "postgres":
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    _POSTGRES_POOL = ConnectionPool(
        conninfo=_postgres_conninfo(),
        open=True,
        min_size=1,
        max_size=int(os.getenv("TUTOR_DB_POOL_SIZE", "10")),
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        name="adaptmath-tutor",
    )
    tutor_checkpointer = PostgresSaver(_POSTGRES_POOL)
else:
    _SQLITE_CONNECTION = sqlite3.connect(
        str(CHECKPOINT_DB_PATH),
        check_same_thread=False,
        timeout=30.0,
    )
    _SQLITE_CONNECTION.execute("PRAGMA journal_mode=WAL")
    _SQLITE_CONNECTION.execute("PRAGMA synchronous=NORMAL")
    _SQLITE_CONNECTION.execute("PRAGMA busy_timeout=30000")
    _SQLITE_CONNECTION.commit()
    tutor_checkpointer = SqliteSaver(_SQLITE_CONNECTION)
    tutor_checkpointer.setup()


def get_checkpoint_db_path() -> Path:
    """Return the legacy SQLite path used by rollback and migration tools."""

    return CHECKPOINT_DB_PATH


def get_persistence_mode() -> str:
    return PERSISTENCE_MODE


def list_checkpoint_thread_ids() -> list[str]:
    """List durable Tutor thread IDs from newest to oldest."""

    if PERSISTENCE_MODE == "postgres":
        assert _POSTGRES_POOL is not None
        with _POSTGRES_POOL.connection() as connection:
            rows = connection.execute(
                """
                SELECT thread_id, MAX(checkpoint_id) AS latest_checkpoint
                FROM tutor.checkpoints
                GROUP BY thread_id
                ORDER BY latest_checkpoint DESC
                """
            ).fetchall()
            return [str(row["thread_id"]) for row in rows]

    connection = sqlite3.connect(
        f"file:{CHECKPOINT_DB_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=2.0,
    )
    try:
        rows = connection.execute(
            "SELECT thread_id, MAX(rowid) AS latest_row "
            "FROM checkpoints GROUP BY thread_id ORDER BY latest_row DESC"
        ).fetchall()
        return [str(row[0]) for row in rows if isinstance(row[0], str) and row[0]]
    finally:
        connection.close()


def check_persistence_ready() -> bool:
    """Verify the selected store and exact official saver migration level."""

    if PERSISTENCE_MODE == "postgres":
        assert _POSTGRES_POOL is not None
        with _POSTGRES_POOL.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    to_regclass('tutor.checkpoints') IS NOT NULL AS checkpoints,
                    to_regclass('tutor.checkpoint_blobs') IS NOT NULL AS blobs,
                    to_regclass('tutor.checkpoint_writes') IS NOT NULL AS writes,
                    COALESCE((SELECT MAX(v) FROM tutor.checkpoint_migrations), -1)
                        AS migration_version
                """
            ).fetchone()
            return bool(
                row
                and row["checkpoints"]
                and row["blobs"]
                and row["writes"]
                and int(row["migration_version"]) == 9
            )

    if not CHECKPOINT_DB_PATH.exists():
        return False
    connection = sqlite3.connect(
        f"file:{CHECKPOINT_DB_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=2.0,
    )
    try:
        row = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'checkpoints'"
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def upsert_thread_metadata(
    *, thread_id: str, student_id: str, target_skill: str | None
) -> None:
    if PERSISTENCE_MODE != "postgres":
        return
    assert _POSTGRES_POOL is not None
    with _POSTGRES_POOL.connection() as connection:
        connection.execute(
            """
            INSERT INTO tutor.threads (thread_id, student_id, target_skill)
            VALUES (%s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET
                student_id = EXCLUDED.student_id,
                target_skill = EXCLUDED.target_skill,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, student_id, target_skill),
        )


def set_thread_status(thread_id: str, status: str) -> None:
    if PERSISTENCE_MODE != "postgres":
        return
    if status not in {"ACTIVE", "COMPLETED", "ABORTED"}:
        raise ValueError("Unsupported Tutor thread status.")
    assert _POSTGRES_POOL is not None
    with _POSTGRES_POOL.connection() as connection:
        connection.execute(
            """
            UPDATE tutor.threads
            SET status = %s,
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CASE
                    WHEN %s THEN CURRENT_TIMESTAMP ELSE completed_at
                END
            WHERE thread_id = %s
            """,
            (status, status == "COMPLETED", thread_id),
        )


def close_persistence() -> None:
    """Close the selected process-level connection once."""

    global _CLOSED
    with _CLOSE_LOCK:
        if _CLOSED:
            return
        if _SQLITE_CONNECTION is not None:
            _SQLITE_CONNECTION.close()
        if _POSTGRES_POOL is not None:
            _POSTGRES_POOL.close()
        _CLOSED = True


atexit.register(close_persistence)
