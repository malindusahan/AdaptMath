"""Tests for the PostgreSQL configuration and SQLAlchemy session foundation."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base, NAMING_CONVENTION
from src.database.postgres_config import (
    DATABASE_URL_ENV,
    PostgresConfigurationError,
    PostgresSettings,
    load_postgres_settings,
)
import src.database.postgres_session as postgres_session
from src.database.postgres_session import (
    PostgresConnectionError,
    check_postgres_connection,
    create_postgres_engine,
    create_session_factory,
    session_scope,
)


TEST_DATABASE_URL = (
    "postgresql+psycopg://student_memory_app:secret@"
    "127.0.0.1:5432/student_memory_db"
)


def test_declarative_base_has_stable_naming_convention():
    assert Base.metadata.naming_convention == NAMING_CONVENTION
    assert set(NAMING_CONVENTION) == {"pk", "fk", "uq", "ck", "ix"}


def test_settings_load_from_explicit_environment_mapping():
    settings = load_postgres_settings(
        {
            DATABASE_URL_ENV: TEST_DATABASE_URL,
            "MEMORY_DATABASE_SCHEMA": "student_memory",
        }
    )

    assert settings.database_url.drivername == "postgresql+psycopg"
    assert settings.database_url.database == "student_memory_db"
    assert settings.database_url.username == "student_memory_app"
    assert settings.schema == "student_memory"
    assert "secret" not in repr(settings)


def test_missing_database_configuration_fails_clearly():
    with pytest.raises(PostgresConfigurationError, match=DATABASE_URL_ENV):
        load_postgres_settings({})


@pytest.mark.parametrize(
    "database_url",
    [
        "not-a-database-url",
        "sqlite+pysqlite:///:memory:",
        "postgresql://student_memory_app:secret@localhost/student_memory_db",
        "postgresql+psycopg://student_memory_app:secret@localhost",
        (
            "postgresql+psycopg://student_memory_app:secret@part@"
            "127.0.0.1:5432/student_memory_db"
        ),
    ],
)
def test_invalid_database_url_is_rejected_without_exposing_password(database_url):
    with pytest.raises(PostgresConfigurationError) as error:
        load_postgres_settings({DATABASE_URL_ENV: database_url})

    assert "secret" not in str(error.value)


@pytest.mark.parametrize("schema", ["", "has-hyphen", "has space", "1invalid"])
def test_invalid_schema_is_rejected(schema):
    with pytest.raises(PostgresConfigurationError, match="MEMORY_DATABASE_SCHEMA"):
        load_postgres_settings(
            {
                DATABASE_URL_ENV: TEST_DATABASE_URL,
                "MEMORY_DATABASE_SCHEMA": schema,
            }
        )


def test_engine_creation_is_lazy_and_uses_pool_pre_ping():
    settings = load_postgres_settings({DATABASE_URL_ENV: TEST_DATABASE_URL})
    engine = create_postgres_engine(settings)

    try:
        assert isinstance(engine, Engine)
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.database == "student_memory_db"
        assert engine.pool._pre_ping is True
    finally:
        engine.dispose()


def test_importing_session_module_does_not_create_engine_or_connect():
    old_engine = postgres_session._global_engine
    old_factory = postgres_session._global_session_factory
    postgres_session._global_engine = None
    postgres_session._global_session_factory = None
    try:
        engine_instances = [
            value
            for value in vars(postgres_session).values()
            if isinstance(value, Engine)
        ]
        assert engine_instances == []
    finally:
        postgres_session._global_engine = old_engine
        postgres_session._global_session_factory = old_factory


def test_session_scope_commits_on_success():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE events (value INTEGER NOT NULL)"))

    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(text("INSERT INTO events (value) VALUES (1)"))

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM events")).scalar_one() == 1
    engine.dispose()


def test_session_scope_rolls_back_on_failure():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE events (value INTEGER NOT NULL)"))

    factory = create_session_factory(engine)
    with pytest.raises(RuntimeError, match="force rollback"):
        with session_scope(factory) as session:
            session.execute(text("INSERT INTO events (value) VALUES (1)"))
            raise RuntimeError("force rollback")

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM events")).scalar_one() == 0
    engine.dispose()


def test_session_scope_always_closes_session():
    class TrackingSession(Session):
        was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    engine = create_engine("sqlite+pysqlite:///:memory:")
    factory = sessionmaker(bind=engine, class_=TrackingSession)

    with session_scope(factory) as session:
        tracked_session = session

    assert tracked_session.was_closed is True
    engine.dispose()


def test_health_check_function_maps_database_response():
    class FakeResult:
        def one(self):
            return ("16.4", "student_memory_db", "student_memory_app", 1)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, statement):
            assert "server_version" in str(statement)
            return FakeResult()

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    result = check_postgres_connection(FakeEngine())

    assert result.server_version == "16.4"
    assert result.database_name == "student_memory_db"
    assert result.database_user == "student_memory_app"
    assert result.select_one == 1
    assert result.is_healthy is True


def test_health_check_failure_is_credential_safe():
    from sqlalchemy.exc import OperationalError

    class FailingEngine:
        def connect(self):
            raise OperationalError(
                "connect",
                {"password": "sensitive-value"},
                RuntimeError("sensitive-value"),
            )

    with pytest.raises(PostgresConnectionError) as error:
        check_postgres_connection(FailingEngine())

    assert str(error.value) == "PostgreSQL connection health check failed."
    assert "sensitive-value" not in str(error.value)


def test_live_postgresql_16_connection():
    """Run only when the private database URL is present in the environment."""

    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for the live test.")

    settings = load_postgres_settings()
    engine = create_postgres_engine(settings)
    try:
        result = check_postgres_connection(engine)
    finally:
        engine.dispose()

    assert result.server_version.startswith("16.")
    assert result.database_name == settings.database_url.database
    assert result.database_user == settings.database_url.username
    assert result.select_one == 1
    assert result.is_healthy is True
