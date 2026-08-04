"""Persistent student misconception storage and retrieval."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.database.connection import (
    DEFAULT_DATABASE_PATH,
    DatabasePath,
    get_connection,
    initialize_database,
)
from src.schemas.interaction import AssessmentMemoryUpdateRequest


class MisconceptionRepositoryError(Exception):
    """Base exception for misconception persistence errors."""


@dataclass(frozen=True)
class MisconceptionMemory:
    """One persistent misconception-memory record."""

    misconception_id: int
    student_id: str
    topic: str
    subtopic: str | None
    misconception: str
    occurrence_count: int
    first_seen_at: str
    last_seen_at: str


def normalize_misconception(misconception: str) -> str:
    """Conservatively normalize text for stable exact-match storage."""

    if not isinstance(misconception, str):
        raise MisconceptionRepositoryError("Misconception must be a string.")

    normalized = " ".join(misconception.strip().split()).lower()

    if not normalized:
        raise MisconceptionRepositoryError("Misconception cannot be empty.")

    return normalized


def collect_unique_misconceptions(
    request: AssessmentMemoryUpdateRequest,
) -> list[str]:
    """Collect and deduplicate top-level and question-level errors."""

    raw_errors = list(request.identified_errors)
    raw_errors.extend(
        question.identified_error
        for question in request.assessment_questions
        if question.identified_error is not None
    )

    unique_errors: list[str] = []
    seen: set[str] = set()

    for raw_error in raw_errors:
        normalized = normalize_misconception(raw_error)
        if normalized not in seen:
            seen.add(normalized)
            unique_errors.append(normalized)

    return unique_errors


def update_misconception_memory(
    request: AssessmentMemoryUpdateRequest,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> list[MisconceptionMemory]:
    """Insert or increment each unique misconception once per assessment."""

    misconceptions = collect_unique_misconceptions(request)
    if not misconceptions:
        return []

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        if owns_connection:
            connection.execute("BEGIN")

        for misconception in misconceptions:
            existing = connection.execute(
                """
                SELECT misconception_id
                FROM student_misconceptions
                WHERE student_id = ?
                  AND topic = ?
                  AND subtopic IS ?
                  AND misconception = ?
                """,
                (
                    request.student_id,
                    request.topic,
                    request.subtopic,
                    misconception,
                ),
            ).fetchone()

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO student_misconceptions (
                        student_id, topic, subtopic, misconception,
                        occurrence_count
                    )
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (
                        request.student_id,
                        request.topic,
                        request.subtopic,
                        misconception,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE student_misconceptions
                    SET occurrence_count = occurrence_count + 1,
                        last_seen_at = CURRENT_TIMESTAMP
                    WHERE misconception_id = ?
                    """,
                    (existing["misconception_id"],),
                )

        if owns_connection:
            connection.commit()

        return get_student_misconceptions(
            student_id=request.student_id,
            topic=request.topic,
            subtopic=request.subtopic,
            database_path=database_path,
            connection=connection,
        )

    except Exception as exc:
        if owns_connection:
            connection.rollback()
        if isinstance(exc, MisconceptionRepositoryError):
            raise
        raise MisconceptionRepositoryError(f"Misconception persistence failed: {exc}") from exc
    finally:
        if owns_connection:
            connection.close()


def get_student_misconceptions(
    student_id: str,
    topic: str,
    subtopic: str | None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> list[MisconceptionMemory]:
    """Retrieve a student's misconception memory, most frequent first."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        rows = connection.execute(
            """
            SELECT misconception_id, student_id, topic, subtopic,
                   misconception, occurrence_count, first_seen_at, last_seen_at
            FROM student_misconceptions
            WHERE student_id = ?
              AND topic = ?
              AND subtopic IS ?
            ORDER BY occurrence_count DESC,
                     last_seen_at DESC,
                     misconception ASC
            """,
            (student_id, topic, subtopic),
        ).fetchall()
        return [
            MisconceptionMemory(
            misconception_id=row["misconception_id"],
            student_id=row["student_id"],
            topic=row["topic"],
            subtopic=row["subtopic"],
            misconception=row["misconception"],
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            )
            for row in rows
        ]
    finally:
        if owns_connection:
            connection.close()
