"""Chronological, cutoff-aware historical interaction retrieval."""

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


@dataclass(frozen=True)
class HistoricalInteraction:
    interaction_id: int
    assessment_id: int
    student_id: str
    question_id: str
    topic: str
    subtopic: str | None
    is_correct: bool
    attempt_count: int | None
    hint_count: int | None
    hint_total: int | None
    response_time_ms: float | None
    attempt_data_available: bool
    hint_data_available: bool
    response_time_available: bool
    created_at: str


def _rows_to_history(rows) -> list[HistoricalInteraction]:
    return [
        HistoricalInteraction(
            interaction_id=row["interaction_id"],
            assessment_id=row["assessment_id"],
            student_id=row["student_id"],
            question_id=row["question_id"],
            topic=row["topic"],
            subtopic=row["subtopic"],
            is_correct=bool(row["is_correct"]),
            attempt_count=row["attempt_count"],
            hint_count=row["hint_count"],
            hint_total=row["hint_total"],
            response_time_ms=row["response_time_ms"],
            attempt_data_available=bool(row["attempt_data_available"]),
            hint_data_available=bool(row["hint_data_available"]),
            response_time_available=bool(row["response_time_available"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


_HISTORY_COLUMNS = """
    interaction_id,
    assessment_id,
    student_id,
    question_id,
    topic,
    subtopic,
    is_correct,
    attempt_count,
    hint_count,
    hint_total,
    response_time_ms,
    attempt_data_available,
    hint_data_available,
    response_time_available,
    created_at
"""


def get_student_history(
    student_id: str,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    before_assessment_id: int | None = None,
    connection: sqlite3.Connection | None = None,
) -> list[HistoricalInteraction]:
    """Retrieve all student interactions, optionally before an assessment."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    sql = f"""
        SELECT {_HISTORY_COLUMNS}
        FROM assessment_interactions
        WHERE student_id = ?
    """
    parameters: list[object] = [student_id]

    if before_assessment_id is not None:
        sql += " AND assessment_id < ?"
        parameters.append(before_assessment_id)

    sql += " ORDER BY assessment_id ASC, interaction_id ASC"

    try:
        rows = connection.execute(sql, parameters).fetchall()
        return _rows_to_history(rows)
    finally:
        if owns_connection:
            connection.close()


def get_student_topic_history(
    student_id: str,
    topic: str,
    subtopic: str | None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    before_assessment_id: int | None = None,
    connection: sqlite3.Connection | None = None,
) -> list[HistoricalInteraction]:
    """Retrieve a student's learning-area history with an optional cutoff."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    sql = f"""
        SELECT {_HISTORY_COLUMNS}
        FROM assessment_interactions
        WHERE student_id = ?
          AND topic = ?
          AND subtopic IS ?
    """
    parameters: list[object] = [student_id, topic, subtopic]

    if before_assessment_id is not None:
        sql += " AND assessment_id < ?"
        parameters.append(before_assessment_id)

    sql += " ORDER BY assessment_id ASC, interaction_id ASC"

    try:
        rows = connection.execute(sql, parameters).fetchall()
        return _rows_to_history(rows)
    finally:
        if owns_connection:
            connection.close()


def get_assessment_interactions(
    assessment_id: int,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> list[HistoricalInteraction]:
    """Retrieve one assessment's interactions in original insertion order."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        rows = connection.execute(
            f"""
            SELECT {_HISTORY_COLUMNS}
            FROM assessment_interactions
            WHERE assessment_id = ?
            ORDER BY interaction_id ASC
            """,
            (assessment_id,),
        ).fetchall()
        return _rows_to_history(rows)
    finally:
        if owns_connection:
            connection.close()
