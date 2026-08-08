"""
Student State Service.

Responsibilities
----------------
- Accept prepared historical learning-state features.
- Call the learning-state prediction engine.
- Return the typed prediction response.

This service does NOT yet:
- query SQLite,
- build historical features from raw interactions,
- save learning-state snapshots,
- expose FastAPI routes.

Those responsibilities will be added in later phases.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from src.database.connection import DEFAULT_DATABASE_PATH, DatabasePath
from src.database.history_repository import (
    get_student_history,
    get_student_topic_history,
)
from src.features.behavioural_coverage import classify_behavioural_coverage
from src.features.production_feature_builder import build_production_features
from src.models.learning_state_model import (
    FeatureValidationError,
    LearningStateModelError,
    predict_learning_state,
)
from src.schemas.interaction import (
    LearningStatePredictionResponse,
    StudentHistoryPredictionResponse,
)


class StudentStateServiceError(Exception):
    """Base exception for student-state service errors."""


class StudentStateService:
    """
    Service layer for student learning-state prediction.

    The service deliberately hides model-loading and inference details from
    higher layers such as database orchestration and FastAPI routes.
    """

    def predict_state(
        self,
        historical_features: Mapping[str, Any],
    ) -> LearningStatePredictionResponse:
        """
        Predict the student's current learning state from prepared history.

        Parameters
        ----------
        historical_features:
            Mapping containing the historical features required by the
            learning-state prediction engine.

        Returns
        -------
        LearningStatePredictionResponse
            Typed prediction result containing:

            - learning_state
            - evidence_level
            - evidence_strength
            - model_used
        """

        if not isinstance(historical_features, Mapping):
            raise StudentStateServiceError(
                "Historical features must be supplied as a mapping/dictionary."
            )

        try:
            return predict_learning_state(
                historical_features
            )

        except FeatureValidationError:
            # Preserve input-validation errors so callers can distinguish
            # invalid input from unexpected service/model failures.
            raise

        except LearningStateModelError as exc:
            raise StudentStateServiceError(
                f"Learning-state prediction failed: {exc}"
            ) from exc

    def predict_from_stored_history(
        self,
        student_id: str,
        topic: str,
        subtopic: str | None = None,
        database_path: DatabasePath = DEFAULT_DATABASE_PATH,
        before_assessment_id: int | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> StudentHistoryPredictionResponse:
        """Predict from persisted history without performing database writes."""

        if not isinstance(student_id, str) or not student_id.strip():
            raise StudentStateServiceError("student_id must be a non-empty string.")
        if not isinstance(topic, str) or not topic.strip():
            raise StudentStateServiceError("topic must be a non-empty string.")
        if before_assessment_id is not None and (
            not isinstance(before_assessment_id, int)
            or isinstance(before_assessment_id, bool)
            or before_assessment_id <= 0
        ):
            raise StudentStateServiceError(
                "before_assessment_id must be a positive integer when supplied."
            )

        student_id = student_id.strip()
        topic = topic.strip()

        if subtopic is not None:
            if not isinstance(subtopic, str):
                raise StudentStateServiceError("subtopic must be a string or None.")
            subtopic = subtopic.strip() or None

        overall_history = get_student_history(
            student_id=student_id,
            database_path=database_path,
            before_assessment_id=before_assessment_id,
            connection=connection,
        )
        skill_history = get_student_topic_history(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
            database_path=database_path,
            before_assessment_id=before_assessment_id,
            connection=connection,
        )
        features = build_production_features(
            overall_history=overall_history,
            skill_history=skill_history,
        )
        coverage = classify_behavioural_coverage(overall_history)
        prediction = self.predict_state(features)

        return StudentHistoryPredictionResponse(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
            learning_state=prediction.learning_state,
            evidence_level=prediction.evidence_level,
            evidence_strength=prediction.evidence_strength,
            behavioural_coverage=coverage.level,
            model_used=prediction.model_used,
            previous_interaction_count=features["previous_interaction_count"],
            previous_skill_interaction_count=features[
                "previous_skill_interaction_count"
            ],
            recent_interaction_count=coverage.recent_interaction_count,
            attempt_observation_count=coverage.attempt_observation_count,
            hint_observation_count=coverage.hint_observation_count,
            response_time_observation_count=coverage.response_time_observation_count,
        )


# =============================================================================
# Reusable service instance
# =============================================================================

_default_student_state_service: StudentStateService | None = None


def get_student_state_service() -> StudentStateService:
    """
    Return a reusable process-level StudentStateService instance.
    """

    global _default_student_state_service

    if _default_student_state_service is None:
        _default_student_state_service = StudentStateService()

    return _default_student_state_service


def predict_student_state(
    historical_features: Mapping[str, Any],
) -> LearningStatePredictionResponse:
    """
    Convenience service function for predicting a student's state.
    """

    service = get_student_state_service()

    return service.predict_state(
        historical_features
    )


def predict_student_state_from_history(
    student_id: str,
    topic: str,
    subtopic: str | None = None,
    database_path: DatabasePath = DEFAULT_DATABASE_PATH,
    before_assessment_id: int | None = None,
    connection: sqlite3.Connection | None = None,
) -> StudentHistoryPredictionResponse:
    """Convenience interface for predicting directly from stored history."""

    service = get_student_state_service()
    return service.predict_from_stored_history(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        database_path=database_path,
        before_assessment_id=before_assessment_id,
        connection=connection,
    )
