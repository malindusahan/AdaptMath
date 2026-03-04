"""Shared credential-safe helpers for explicit PostgreSQL import utilities."""

from __future__ import annotations

import os


def database_url(explicit: str | None = None) -> str:
    value = (
        explicit
        or os.getenv("ADAPTMATH_DATABASE_URL")
        or os.getenv("MEMORY_DATABASE_URL")
        or ""
    ).strip()
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value.removeprefix("postgresql+psycopg://")
    if not value.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            "Provide --database-url or ADAPTMATH_DATABASE_URL with a PostgreSQL URL."
        )
    return value


def connect(explicit: str | None = None, *, schema: str | None = None):
    import psycopg

    connection = psycopg.connect(database_url(explicit), autocommit=False)
    if schema is not None:
        if schema not in {"auth", "student_memory", "tutor", "student_model", "research"}:
            connection.close()
            raise ValueError("Unsupported AdaptMath schema.")
        connection.execute(f"SET search_path TO {schema}, public")
        connection.commit()
    return connection
