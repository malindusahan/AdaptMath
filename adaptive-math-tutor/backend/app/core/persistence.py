from __future__ import annotations

import atexit
import os
import sqlite3
from pathlib import Path
from threading import Lock

# Restrict checkpoint deserialization to LangGraph's strict
# MessagePack policy when supported by the installed checkpoint
# package. Set this before importing the SQLite saver.
os.environ.setdefault(
    "LANGGRAPH_STRICT_MSGPACK",
    "true",
)

from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import get_settings


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_CLOSE_LOCK = Lock()
_CLOSED = False


def _resolve_checkpoint_path() -> Path:
    settings = get_settings()
    configured = Path(
        settings.checkpoint_db_path
    ).expanduser()

    if configured.is_absolute():
        resolved = configured
    else:
        resolved = (
            _BACKEND_ROOT / configured
        ).resolve()

    resolved.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return resolved


CHECKPOINT_DB_PATH = _resolve_checkpoint_path()

# SqliteSaver supports check_same_thread=False and protects its
# own checkpoint operations with a lock. This synchronous saver is
# appropriate for the current lightweight local PP2 deployment.
_CONNECTION = sqlite3.connect(
    str(CHECKPOINT_DB_PATH),
    check_same_thread=False,
    timeout=30.0,
)
_CONNECTION.execute(
    "PRAGMA journal_mode=WAL"
)
_CONNECTION.execute(
    "PRAGMA synchronous=NORMAL"
)
_CONNECTION.execute(
    "PRAGMA busy_timeout=30000"
)
_CONNECTION.commit()


tutor_checkpointer = SqliteSaver(
    _CONNECTION
)
tutor_checkpointer.setup()


def get_checkpoint_db_path() -> Path:
    """Return the absolute LangGraph checkpoint database path."""
    return CHECKPOINT_DB_PATH


def check_persistence_ready() -> bool:
    """Verify that the persisted LangGraph SQLite store is readable."""
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


def close_persistence() -> None:
    """Close the process-local SQLite connection once."""
    global _CLOSED

    with _CLOSE_LOCK:
        if _CLOSED:
            return
        _CONNECTION.close()
        _CLOSED = True


atexit.register(
    close_persistence
)
