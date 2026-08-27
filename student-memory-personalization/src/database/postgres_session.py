"""SQLAlchemy engine, session, transaction, and health-check foundation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.database.postgres_config import (
    PostgresSettings,
    load_postgres_settings,
)


SessionFactory = sessionmaker[Session]


class PostgresConnectionError(RuntimeError):
    """Credential-safe PostgreSQL connection or health-check failure."""


@dataclass(frozen=True)
class PostgresHealthResult:
    """Safe diagnostic result returned by a live PostgreSQL health check."""

    server_version: str
    database_name: str
    database_user: str
    select_one: int

    @property
    def is_healthy(self) -> bool:
        return self.select_one == 1


def create_postgres_engine(
    settings: PostgresSettings | None = None,
    **engine_options: object,
) -> Engine:
    """Create a lazy PostgreSQL engine without opening a connection."""

    resolved = settings if settings is not None else load_postgres_settings()
    options: dict[str, object] = {
        "pool_pre_ping": True,
        "connect_args": {"options": f"-c search_path={resolved.schema},public"},
    }
    options.update(engine_options)
    return create_engine(resolved.database_url, **options)


def create_session_factory(engine: Engine) -> SessionFactory:
    """Create the shared SQLAlchemy session factory for an engine."""

    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )


_global_engine: Engine | None = None
_global_session_factory: SessionFactory | None = None


def get_session_factory() -> SessionFactory:
    """Return the active process-level SessionFactory, creating it lazily if needed."""
    global _global_engine, _global_session_factory
    if _global_session_factory is None:
        _global_engine = create_postgres_engine()
        _global_session_factory = create_session_factory(_global_engine)
    return _global_session_factory


def set_session_factory(factory: SessionFactory | None) -> None:
    """Override or reset the active process-level SessionFactory (e.g. for tests)."""
    global _global_session_factory
    _global_session_factory = factory


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    """Commit on success, roll back on failure, and always close the session."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_postgres_connection(engine: Engine) -> PostgresHealthResult:
    """Open one short connection and verify PostgreSQL identity and SELECT 1."""

    statement = text(
        """
        SELECT
            current_setting('server_version'),
            current_database(),
            current_user,
            1
        """
    )
    try:
        with engine.connect() as connection:
            row = connection.execute(statement).one()
    except SQLAlchemyError:
        raise PostgresConnectionError(
            "PostgreSQL connection health check failed."
        ) from None

    return PostgresHealthResult(
        server_version=str(row[0]),
        database_name=str(row[1]),
        database_user=str(row[2]),
        select_one=int(row[3]),
    )
