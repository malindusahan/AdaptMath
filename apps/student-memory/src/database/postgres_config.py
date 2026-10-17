"""Environment-based configuration for PostgreSQL persistence.

This module reads configuration only when ``load_postgres_settings`` is
called. Importing it never creates an engine or opens a database connection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import re

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError


DATABASE_URL_ENV = "MEMORY_DATABASE_URL"
DATABASE_SCHEMA_ENV = "MEMORY_DATABASE_SCHEMA"
DEFAULT_POSTGRES_SCHEMA = "student_memory"

_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REQUIRED_DRIVER = "postgresql+psycopg"


class PostgresConfigurationError(RuntimeError):
    """Raised when PostgreSQL runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class PostgresSettings:
    """Validated PostgreSQL settings with credential-safe representation."""

    database_url: URL
    schema: str = DEFAULT_POSTGRES_SCHEMA

    def __repr__(self) -> str:
        safe_url = self.database_url.render_as_string(hide_password=True)
        return f"PostgresSettings(database_url={safe_url!r}, schema={self.schema!r})"


def _parse_database_url(raw_url: str) -> URL:
    try:
        parsed = make_url(raw_url)
    except ArgumentError as exc:
        raise PostgresConfigurationError(
            f"{DATABASE_URL_ENV} must be a valid SQLAlchemy database URL."
        ) from exc

    if parsed.drivername != _REQUIRED_DRIVER:
        raise PostgresConfigurationError(
            f"{DATABASE_URL_ENV} must use the postgresql+psycopg driver."
        )
    if not parsed.username:
        raise PostgresConfigurationError(
            f"{DATABASE_URL_ENV} must include a database username."
        )
    if not parsed.host or "@" in parsed.host:
        raise PostgresConfigurationError(
            f"{DATABASE_URL_ENV} has an invalid host. URL-encode special "
            "characters in the password."
        )
    if not parsed.database:
        raise PostgresConfigurationError(
            f"{DATABASE_URL_ENV} must include a database name."
        )
    return parsed


def _validate_schema(raw_schema: str) -> str:
    schema = raw_schema.strip()
    if not schema:
        raise PostgresConfigurationError(
            f"{DATABASE_SCHEMA_ENV} must not be blank."
        )
    if not _SCHEMA_NAME_PATTERN.fullmatch(schema):
        raise PostgresConfigurationError(
            f"{DATABASE_SCHEMA_ENV} must be a valid unquoted PostgreSQL identifier."
        )
    return schema


def load_postgres_settings(
    environ: Mapping[str, str] | None = None,
) -> PostgresSettings:
    """Load and validate PostgreSQL settings from an environment mapping."""

    if environ is None:
        if not os.environ.get(DATABASE_URL_ENV):
            from pathlib import Path
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.is_file():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k and k not in os.environ:
                            os.environ[k] = v
        values = os.environ
    else:
        values = environ

    raw_url = values.get(DATABASE_URL_ENV, "").strip()
    if not raw_url:
        raise PostgresConfigurationError(
            f"Missing required environment variable: {DATABASE_URL_ENV}."
        )

    raw_schema = values.get(DATABASE_SCHEMA_ENV, DEFAULT_POSTGRES_SCHEMA)
    return PostgresSettings(
        database_url=_parse_database_url(raw_url),
        schema=_validate_schema(raw_schema),
    )
