"""Alembic environment for the Student Personalization Memory database."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.database.base import Base
from src.database import models  # noqa: F401 - registers model metadata
from src.database.postgres_config import (
    DEFAULT_POSTGRES_SCHEMA,
    PostgresConfigurationError,
    load_postgres_settings,
)


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


import os
from pathlib import Path


def _load_env_if_present() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v


def _configure_runtime_url() -> str:
    _load_env_if_present()
    settings = load_postgres_settings()
    if settings.schema != DEFAULT_POSTGRES_SCHEMA:
        raise PostgresConfigurationError(
            "Alembic migrations require MEMORY_DATABASE_SCHEMA=student_memory."
        )

    url = settings.database_url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    config.attributes["memory_schema"] = settings.schema
    return url


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""

    url = _configure_runtime_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a short-lived SQLAlchemy connection."""

    _configure_runtime_url()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        pool_pre_ping=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
