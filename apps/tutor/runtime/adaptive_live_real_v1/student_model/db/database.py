"""
SQLite database connection and initialisation for the Meta-Agent.

Provides a single connection helper used by all modules in core/.
Schema is defined in db/schema.sql and applied on first run.
"""

import logging
import os
import re
import sqlite3
import threading
import atexit
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "meta_agent.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"
SCHEMA_VERSION = 2
_POSTGRES_POOLS: dict[str, Any] = {}
_POSTGRES_POOL_LOCK = threading.Lock()


def persistence_mode() -> str:
    mode = os.getenv("STUDENT_MODEL_PERSISTENCE", "sqlite").strip().lower()
    if mode not in {"sqlite", "postgres"}:
        raise ValueError(
            "STUDENT_MODEL_PERSISTENCE must be 'sqlite' or 'postgres'."
        )
    return mode


def _postgres_url() -> str:
    value = os.getenv(
        "STUDENT_MODEL_DATABASE_URL",
        os.getenv("ADAPTMATH_DATABASE_URL", ""),
    ).strip()
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value.removeprefix("postgresql+psycopg://")
    if not value.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            "STUDENT_MODEL_DATABASE_URL (or ADAPTMATH_DATABASE_URL) is required "
            "for PostgreSQL student-model persistence."
        )
    return value


def _postgres_schema() -> str:
    schema = os.getenv("STUDENT_MODEL_DATABASE_SCHEMA", "student_model").strip()
    if schema != "student_model":
        raise ValueError(
            "STUDENT_MODEL_DATABASE_SCHEMA must be exactly 'student_model'."
        )
    return schema


def _postgres_pool():
    url = _postgres_url()
    with _POSTGRES_POOL_LOCK:
        pool = _POSTGRES_POOLS.get(url)
        if pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg-pool is required for PostgreSQL student-model persistence."
                ) from exc

            def configure(connection: Any) -> None:
                connection.execute(
                    f"SET search_path TO {_postgres_schema()}, public"
                )
                connection.commit()

            pool = ConnectionPool(
                conninfo=url,
                open=True,
                min_size=1,
                max_size=int(os.getenv("STUDENT_MODEL_DB_POOL_SIZE", "8")),
                kwargs={"autocommit": False},
                configure=configure,
                name="adaptmath-student-model",
            )
            _POSTGRES_POOLS[url] = pool
        return pool


def _close_postgres_pools() -> None:
    for pool in tuple(_POSTGRES_POOLS.values()):
        pool.close()


atexit.register(_close_postgres_pools)


class PostgresCompatRow:
    """SQLite-row-compatible view over one Psycopg result tuple."""

    def __init__(self, columns: list[str], values: tuple[Any, ...]) -> None:
        self._columns = columns
        self._values = values
        self._mapping = dict(zip(columns, values, strict=True))

    def __getitem__(self, key: int | str) -> Any:
        return self._values[key] if isinstance(key, int) else self._mapping[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class PostgresCompatCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    def _columns(self) -> list[str]:
        return [str(item.name) for item in (self._cursor.description or ())]

    def fetchone(self) -> PostgresCompatRow | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return PostgresCompatRow(self._columns(), tuple(row))

    def fetchall(self) -> list[PostgresCompatRow]:
        columns = self._columns()
        return [PostgresCompatRow(columns, tuple(row)) for row in self._cursor.fetchall()]

    def __iter__(self):
        columns = self._columns()
        for row in self._cursor:
            yield PostgresCompatRow(columns, tuple(row))


def _postgres_sql(sql: str) -> str:
    statement = sql.strip().rstrip(";")
    if re.match(r"(?is)^INSERT\s+OR\s+IGNORE\s+INTO\s+", statement):
        statement = re.sub(
            r"(?is)^INSERT\s+OR\s+IGNORE\s+INTO\s+",
            "INSERT INTO ",
            statement,
            count=1,
        )
        statement += " ON CONFLICT DO NOTHING"
    elif re.match(r"(?is)^INSERT\s+OR\s+REPLACE\s+INTO\s+sessions\s+", statement):
        statement = re.sub(
            r"(?is)^INSERT\s+OR\s+REPLACE\s+INTO\s+",
            "INSERT INTO ",
            statement,
            count=1,
        )
        statement += (
            " ON CONFLICT (session_id) DO UPDATE SET "
            "student_id = EXCLUDED.student_id, "
            "concept_count = EXCLUDED.concept_count, "
            "processed_at = CURRENT_TIMESTAMP"
        )
    return statement.replace("?", "%s")


class PostgresCompatConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(
        self, sql: str, parameters: tuple[Any, ...] | list[Any] = ()
    ) -> PostgresCompatCursor:
        cursor = self._connection.cursor()
        cursor.execute(_postgres_sql(sql), parameters)
        return PostgresCompatCursor(cursor)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def acquire_student_skill_lock(
    connection: sqlite3.Connection | PostgresCompatConnection,
    student_id: str,
    skill_name: str,
) -> None:
    """Serialize one BKT history stream without changing its calculations."""

    if isinstance(connection, PostgresCompatConnection):
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"{student_id}\x1f{skill_name}",),
        )


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive migrations without inventing historical BKT priors."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        attempt_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(attempts)")
        }
        if "resolver_event_id" not in attempt_columns:
            conn.execute("ALTER TABLE attempts ADD COLUMN resolver_event_id TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_attempts_resolver_event_id
            ON attempts(resolver_event_id)
            WHERE resolver_event_id IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resolved_events (
                event_id TEXT PRIMARY KEY,
                source_action_event_id TEXT,
                student_id TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                should_update INTEGER NOT NULL CHECK (should_update IN (0, 1)),
                outcome INTEGER CHECK (outcome IN (0, 1) OR outcome IS NULL),
                update_confidence REAL NOT NULL CHECK (
                    update_confidence >= 0.0 AND update_confidence <= 1.0
                ),
                observation_source TEXT NOT NULL,
                primary_signal TEXT NOT NULL,
                session_id TEXT,
                mastery_before REAL,
                mastery_after REAL,
                delta_mastery REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
            """
        )
        resolved_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(resolved_events)")
        }
        for column in ("mastery_before", "mastery_after", "delta_mastery"):
            if column not in resolved_columns:
                conn.execute(
                    f"ALTER TABLE resolved_events ADD COLUMN {column} REAL"
                )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_resolved_events_source_action
            ON resolved_events(source_action_event_id)
            """
        )
        # A historical mastery/attempt pair may originally have used a
        # transferred prior that cannot be reconstructed from current state.
        # Record that uncertainty explicitly; never copy current/previous
        # mastery or silently claim the population prior was the original.
        conn.execute(
            """
            INSERT OR IGNORE INTO bkt_initial_priors
                (student_id, skill_name, effective_initial_prior, prior_source)
            SELECT pairs.student_id, pairs.skill_name, NULL, 'legacy_unknown'
            FROM (
                SELECT student_id, skill_name FROM mastery
                UNION
                SELECT student_id, skill_name FROM attempts
            ) AS pairs
            INNER JOIN students
                ON students.student_id = pairs.student_id
            """
        )
        current_version = conn.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        if current_version < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def initialise_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Create the database file and apply the schema if it doesn't exist.

    Idempotent — safe to call multiple times. The schema uses
    IF NOT EXISTS clauses so re-running it does not destroy data.

    Args:
        db_path: Where to create the SQLite file.
    """
    if persistence_mode() == "postgres":
        with get_connection(db_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS table_count
                FROM information_schema.tables
                WHERE table_schema = 'student_model'
                  AND table_name IN (
                      'students', 'mastery', 'bkt_initial_priors', 'attempts',
                      'resolved_events', 'sessions'
                  )
                """
            ).fetchone()
            if row is None or int(row["table_count"]) != 6:
                raise RuntimeError(
                    "student_model schema is not migrated; run Alembic first."
                )
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

    schema = SCHEMA_PATH.read_text()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema)
        _apply_migrations(conn)

    logger.info(f"Database initialised at {db_path}")


@contextmanager
def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
) -> Iterator[sqlite3.Connection | PostgresCompatConnection]:
    """
    Yield a SQLite connection with sensible defaults.

    Use as a context manager:

        with get_connection() as conn:
            cursor = conn.execute("SELECT * FROM students")
            ...

    The connection is automatically committed on success and rolled back
    on exception. Foreign keys are enforced.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        An open sqlite3.Connection with row factory enabled.
    """
    if persistence_mode() == "postgres":
        with _postgres_pool().connection() as raw_connection:
            conn = PostgresCompatConnection(raw_connection)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
