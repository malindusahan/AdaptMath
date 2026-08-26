"""Atomic learning-state snapshot and current-memory persistence."""

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
    CurrentStudentMemory,
    LearningStateHistoryItem,
    StoredLearningStateSnapshot,
    StudentHistoryPredictionResponse,
)


class StateRepositoryError(Exception):
    """Base exception for learning-state persistence failures."""


class AssessmentContextMismatchError(StateRepositoryError):
    """Raised when an assessment does not match the prediction context."""


def _normalize_subtopic_for_memory(subtopic: str | None) -> str:
    return "" if subtopic is None else subtopic


def _validate_assessment_context(
    connection: sqlite3.Connection,
    assessment_id: int,
    prediction: StudentHistoryPredictionResponse,
) -> None:
    assessment = connection.execute(
        """
        SELECT assessment_id, student_id, topic, subtopic
        FROM assessments
        WHERE assessment_id = ?
        """,
        (assessment_id,),
    ).fetchone()

    if assessment is None:
        raise AssessmentContextMismatchError(
            f"Assessment {assessment_id} does not exist."
        )
    if assessment["student_id"] != prediction.student_id:
        raise AssessmentContextMismatchError(
            "Assessment student_id does not match prediction student_id."
        )
    if assessment["topic"] != prediction.topic:
        raise AssessmentContextMismatchError(
            "Assessment topic does not match prediction topic."
        )
    if assessment["subtopic"] != prediction.subtopic:
        raise AssessmentContextMismatchError(
            "Assessment subtopic does not match prediction subtopic."
        )


def persist_learning_state(
    prediction: StudentHistoryPredictionResponse,
    assessment_id: int | None = None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> StoredLearningStateSnapshot:
    """Append a snapshot and upsert current memory in one transaction."""

    if not isinstance(prediction, StudentHistoryPredictionResponse):
        raise StateRepositoryError(
            "prediction must be a StudentHistoryPredictionResponse."
        )
    if assessment_id is not None and (
        not isinstance(assessment_id, int)
        or isinstance(assessment_id, bool)
        or assessment_id <= 0
    ):
        raise StateRepositoryError(
            "assessment_id must be a positive integer or None."
        )

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        if owns_connection:
            connection.execute("BEGIN")
        if assessment_id is not None:
            _validate_assessment_context(connection, assessment_id, prediction)

        snapshot_cursor = connection.execute(
            """
            INSERT INTO learning_state_snapshots (
                student_id, topic, subtopic, assessment_id,
                learning_state, evidence_level, evidence_strength, model_used,
                previous_interaction_count,
                previous_skill_interaction_count,
                behavioural_coverage, recent_interaction_count,
                attempt_observation_count, hint_observation_count,
                response_time_observation_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.student_id,
                prediction.topic,
                prediction.subtopic,
                assessment_id,
                prediction.learning_state,
                prediction.evidence_level,
                prediction.evidence_strength,
                int(prediction.model_used),
                prediction.previous_interaction_count,
                prediction.previous_skill_interaction_count,
                prediction.behavioural_coverage,
                prediction.recent_interaction_count,
                prediction.attempt_observation_count,
                prediction.hint_observation_count,
                prediction.response_time_observation_count,
            ),
        )
        snapshot_id = snapshot_cursor.lastrowid

        connection.execute(
            """
            INSERT INTO student_memory (
                student_id, topic, subtopic, current_learning_state,
                evidence_level, evidence_strength, model_used,
                behavioural_coverage, recent_interaction_count,
                attempt_observation_count, hint_observation_count,
                response_time_observation_count, last_assessment_id,
                last_snapshot_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (student_id, topic, subtopic)
            DO UPDATE SET
                current_learning_state = excluded.current_learning_state,
                evidence_level = excluded.evidence_level,
                evidence_strength = excluded.evidence_strength,
                model_used = excluded.model_used,
                behavioural_coverage = excluded.behavioural_coverage,
                recent_interaction_count = excluded.recent_interaction_count,
                attempt_observation_count = excluded.attempt_observation_count,
                hint_observation_count = excluded.hint_observation_count,
                response_time_observation_count =
                    excluded.response_time_observation_count,
                last_assessment_id = excluded.last_assessment_id,
                last_snapshot_id = excluded.last_snapshot_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                prediction.student_id,
                prediction.topic,
                _normalize_subtopic_for_memory(prediction.subtopic),
                prediction.learning_state,
                prediction.evidence_level,
                prediction.evidence_strength,
                int(prediction.model_used),
                prediction.behavioural_coverage,
                prediction.recent_interaction_count,
                prediction.attempt_observation_count,
                prediction.hint_observation_count,
                prediction.response_time_observation_count,
                assessment_id,
                snapshot_id,
            ),
        )
        if owns_connection:
            connection.commit()

        return StoredLearningStateSnapshot(
            snapshot_id=snapshot_id,
            assessment_id=assessment_id,
            student_id=prediction.student_id,
            topic=prediction.topic,
            subtopic=prediction.subtopic,
            learning_state=prediction.learning_state,
            evidence_level=prediction.evidence_level,
            evidence_strength=prediction.evidence_strength,
            behavioural_coverage=prediction.behavioural_coverage,
            model_used=prediction.model_used,
            previous_interaction_count=prediction.previous_interaction_count,
            previous_skill_interaction_count=(
                prediction.previous_skill_interaction_count
            ),
            recent_interaction_count=prediction.recent_interaction_count,
            attempt_observation_count=prediction.attempt_observation_count,
            hint_observation_count=prediction.hint_observation_count,
            response_time_observation_count=(
                prediction.response_time_observation_count
            ),
            persisted=True,
        )
    except Exception as exc:
        if owns_connection:
            connection.rollback()
        if isinstance(exc, AssessmentContextMismatchError):
            raise
        if isinstance(exc, StateRepositoryError):
            raise
        if isinstance(exc, sqlite3.IntegrityError):
            raise StateRepositoryError(
                "Learning-state persistence failed due to "
                f"database integrity error: {exc}"
            ) from exc
        raise StateRepositoryError(
            f"Unexpected learning-state persistence failure: {exc}"
        ) from exc
    finally:
        if owns_connection:
            connection.close()


def get_current_student_memory(
    student_id: str,
    topic: str,
    subtopic: str | None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> CurrentStudentMemory | None:
    """Retrieve current persisted memory without rerunning the model."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        row = connection.execute(
            """
            SELECT student_id, topic, subtopic, current_learning_state,
                   evidence_level, evidence_strength, model_used,
                   behavioural_coverage, recent_interaction_count,
                   attempt_observation_count, hint_observation_count,
                   response_time_observation_count, last_assessment_id,
                   last_snapshot_id, updated_at
            FROM student_memory
            WHERE student_id = ? AND topic = ? AND subtopic = ?
            """,
            (student_id, topic, _normalize_subtopic_for_memory(subtopic)),
        ).fetchone()
        if row is None:
            return None
        return CurrentStudentMemory(
            student_id=row["student_id"],
            topic=row["topic"],
            subtopic=None if row["subtopic"] == "" else row["subtopic"],
            learning_state=row["current_learning_state"],
            evidence_level=row["evidence_level"],
            evidence_strength=row["evidence_strength"],
            behavioural_coverage=row["behavioural_coverage"],
            model_used=bool(row["model_used"]),
            recent_interaction_count=row["recent_interaction_count"],
            attempt_observation_count=row["attempt_observation_count"],
            hint_observation_count=row["hint_observation_count"],
            response_time_observation_count=row["response_time_observation_count"],
            last_assessment_id=row["last_assessment_id"],
            last_snapshot_id=row["last_snapshot_id"],
            updated_at=row["updated_at"],
        )
    finally:
        if owns_connection:
            connection.close()


def get_learning_state_history(
    student_id: str,
    topic: str,
    subtopic: str | None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    connection: sqlite3.Connection | None = None,
) -> list[LearningStateHistoryItem]:
    """Retrieve immutable snapshots in deterministic creation order."""

    database_path = Path(database_path)
    owns_connection = connection is None
    if owns_connection:
        initialize_database(database_path)
        connection = get_connection(database_path)

    try:
        rows = connection.execute(
            """
            SELECT snapshot_id, assessment_id, student_id, topic, subtopic,
                   learning_state, evidence_level, evidence_strength,
                   model_used, previous_interaction_count,
                   previous_skill_interaction_count, behavioural_coverage,
                   recent_interaction_count, attempt_observation_count,
                   hint_observation_count, response_time_observation_count,
                   created_at
            FROM learning_state_snapshots
            WHERE student_id = ? AND topic = ? AND subtopic IS ?
            ORDER BY snapshot_id ASC
            """,
            (student_id, topic, subtopic),
        ).fetchall()
        return [
            LearningStateHistoryItem(
                snapshot_id=row["snapshot_id"],
                assessment_id=row["assessment_id"],
                student_id=row["student_id"],
                topic=row["topic"],
                subtopic=row["subtopic"],
                learning_state=row["learning_state"],
                evidence_level=row["evidence_level"],
                evidence_strength=row["evidence_strength"],
                behavioural_coverage=row["behavioural_coverage"],
                model_used=bool(row["model_used"]),
                previous_interaction_count=row["previous_interaction_count"],
                previous_skill_interaction_count=(
                    row["previous_skill_interaction_count"]
                ),
                recent_interaction_count=row["recent_interaction_count"],
                attempt_observation_count=row["attempt_observation_count"],
                hint_observation_count=row["hint_observation_count"],
                response_time_observation_count=(
                    row["response_time_observation_count"]
                ),
                created_at=row["created_at"],
            )
            for row in rows
        ]
    finally:
        if owns_connection:
            connection.close()
