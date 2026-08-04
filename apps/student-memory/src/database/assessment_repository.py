"""
Transactional persistence for completed assessments and question interactions.

Unavailable behavioural measurements are stored as NULL and accompanied by
explicit availability indicators. This module does not calculate features,
run predictions, persist state snapshots, or expose API routes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.database.connection import (
    DEFAULT_DATABASE_PATH,
    DatabasePath,
    get_connection,
    initialize_database,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    StoredAssessmentResult,
)


class AssessmentRepositoryError(Exception):
    """Base exception for assessment persistence errors."""


class DuplicateAssessmentQuestionError(AssessmentRepositoryError):
    """Raised when a question_id occurs more than once in an assessment."""


def _validate_unique_question_ids(request: AssessmentMemoryUpdateRequest) -> None:
    question_ids = [
        question.question_id for question in request.assessment_questions
    ]

    if len(question_ids) != len(set(question_ids)):
        raise DuplicateAssessmentQuestionError(
            "Duplicate question_id values are not allowed within one assessment."
        )


def _availability_flag(*values) -> int:
    """Return 1 if at least one supplied measurement is available."""

    return int(any(value is not None for value in values))


def store_assessment(
    request: AssessmentMemoryUpdateRequest,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> StoredAssessmentResult:
    """Store an assessment and all its question interactions atomically."""

    _validate_unique_question_ids(request)

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        if owns_connection:
            connection.execute("BEGIN")
        assessment_cursor = connection.execute(
            """
            INSERT INTO assessments (
                student_id, topic, subtopic, overall_feedback
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                request.student_id,
                request.topic,
                request.subtopic,
                request.overall_feedback,
            ),
        )
        assessment_id = assessment_cursor.lastrowid
        correct_count = 0
        wrong_count = 0

        for question in request.assessment_questions:
            if question.is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            connection.execute(
                """
                INSERT INTO assessment_interactions (
                    assessment_id,
                    student_id,
                    question_id,
                    topic,
                    subtopic,
                    question_text,
                    student_answer,
                    expected_answer,
                    is_correct,
                    identified_error,
                    attempt_count,
                    hint_count,
                    hint_total,
                    response_time_ms,
                    attempt_data_available,
                    hint_data_available,
                    response_time_available
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    assessment_id,
                    request.student_id,
                    question.question_id,
                    request.topic,
                    request.subtopic,
                    question.question,
                    question.student_answer,
                    question.expected_answer,
                    int(question.is_correct),
                    question.identified_error,
                    question.attempt_count,
                    question.hint_count,
                    question.hint_total,
                    question.response_time_ms,
                    _availability_flag(question.attempt_count),
                    _availability_flag(question.hint_count, question.hint_total),
                    _availability_flag(question.response_time_ms),
                ),
            )

        if owns_connection:
            connection.commit()

        return StoredAssessmentResult(
            assessment_id=assessment_id,
            student_id=request.student_id,
            topic=request.topic,
            subtopic=request.subtopic,
            total_questions=len(request.assessment_questions),
            correct_count=correct_count,
            wrong_count=wrong_count,
            stored=True,
        )

    except Exception as exc:
        if owns_connection:
            connection.rollback()
        if isinstance(exc, AssessmentRepositoryError):
            raise
        if isinstance(exc, sqlite3.IntegrityError):
            raise AssessmentRepositoryError(
                "Assessment storage failed due to "
                f"database integrity error: {exc}"
            ) from exc
        raise AssessmentRepositoryError(f"Assessment storage failed: {exc}") from exc
    finally:
        if owns_connection:
            connection.close()
