"""Orchestrate completed assessments through the PostgreSQL student-memory pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

from src.database.postgres_config import DATABASE_URL_ENV
from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.features.behavioural_coverage import classify_behavioural_coverage
from src.features.production_feature_builder import build_production_features
from src.models.learning_state_model import predict_learning_state
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    MemoryUpdateResponse,
)
from src.schemas.raw_interaction import RawInteractionCreate
from src.services.memory_aggregation_service import MemoryAggregationService
from src.services.postgres_memory_update_service import (
    HistoricalInteractionAdapter,
    _to_historical_adapter,
)
from src.services.skill_context_resolver import SkillContextResolver


class MemoryUpdateServiceError(Exception):
    """Raised when the complete memory-update workflow fails."""


class DuplicateAssessmentQuestionError(MemoryUpdateServiceError):
    """Raised when a question_id occurs more than once in an assessment."""


def _validate_unique_question_ids(request: AssessmentMemoryUpdateRequest) -> None:
    question_ids = [
        question.question_id for question in request.assessment_questions
    ]
    if len(question_ids) != len(set(question_ids)):
        raise DuplicateAssessmentQuestionError(
            "Duplicate question_id values are not allowed within one assessment."
        )


class MemoryUpdateService:
    """High-level application service for processing completed assessments."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        aggregation_service: MemoryAggregationService | None = None,
        state_service: Any = None,
    ):
        self.session_factory = session_factory
        self.aggregation_service = (
            aggregation_service
            if aggregation_service is not None
            else MemoryAggregationService()
        )
        self.state_service = state_service

    def update_memory(
        self,
        request: AssessmentMemoryUpdateRequest,
        database_path: Any = None,
        session_factory: SessionFactory | None = None,
        **kwargs: Any,
    ) -> MemoryUpdateResponse:
        """Process one completed assessment atomically."""
        if not isinstance(request, AssessmentMemoryUpdateRequest):
            raise MemoryUpdateServiceError(
                "request must be an AssessmentMemoryUpdateRequest."
            )

        _validate_unique_question_ids(request)

        # Check if database_path is actually a session factory callable
        if callable(database_path):
            session_factory = database_path
            database_path = None

        resolved_factory = (
            session_factory
            if session_factory is not None
            else self.session_factory
        )

        if resolved_factory is not None or database_path is None:
            return self._update_memory_postgres(
                request=request,
                session_factory=(
                    resolved_factory
                    if resolved_factory is not None
                    else get_session_factory()
                ),
            )

        # Fallback to SQLite execution for legacy test compatibility
        return self._update_memory_sqlite(
            request=request,
            database_path=database_path,
        )

    def _update_memory_postgres(
        self,
        request: AssessmentMemoryUpdateRequest,
        session_factory: SessionFactory,
    ) -> MemoryUpdateResponse:
        """Process assessment inside a single PostgreSQL transaction."""
        try:
            with UnitOfWork(session_factory) as uow:
                # 1. Lock student for serial update
                uow.identities.lock_student_update(request.student_id)

                # 2. Resolve Student & CanonicalSkill
                student = uow.identities.get_or_create_student(request.student_id)
                skill = SkillContextResolver.resolve_or_create_skill(
                    session=uow.session,
                    topic=request.topic,
                    subtopic=request.subtopic,
                )

                # 3. Obtain next assessment ID & Auto-create Session
                assessment_id = uow.identities.next_assessment_id()
                session = uow.identities.get_or_create_session(
                    student=student,
                    external_session_id=f"assessment_{assessment_id}",
                )

                # 4. Insert each question as a raw interaction log
                for question in request.assessment_questions:
                    interaction_data = RawInteractionCreate(
                        student_id=student.student_id,
                        session_id=session.session_id,
                        canonical_skill_id=skill.skill_id,
                        source="evaluator",
                        external_interaction_id=(
                            f"assess_{assessment_id}_{question.question_id}"
                        ),
                        problem_id=None,
                        question_id=question.question_id,
                        student_answer=question.student_answer,
                        expected_answer=question.expected_answer,
                        is_correct=question.is_correct,
                        identified_error=question.identified_error,
                        attempt_count=question.attempt_count,
                        hint_count=question.hint_count,
                        hint_total=question.hint_total,
                        response_time_ms=question.response_time_ms,
                    )
                    uow.raw_interactions.add(interaction_data)

                # 5. Collect and record distinct identified errors in this assessment
                assessment_errors: set[str] = set()
                for question in request.assessment_questions:
                    if question.identified_error and question.identified_error.strip():
                        assessment_errors.add(question.identified_error.strip())
                for error_text in request.identified_errors:
                    if error_text and error_text.strip():
                        assessment_errors.add(error_text.strip())

                for error_text in assessment_errors:
                    uow.supporting.upsert_misconception(
                        student_id=student.student_id,
                        skill_id=skill.skill_id,
                        text=error_text,
                    )

                # 6. Update Projections (STM, LTM, Concept Memory)
                self.aggregation_service.rebuild_student_projections(
                    student_id=student.student_id,
                    uow=uow,
                )

                # 7. Learning-State Prediction Pipeline
                all_student_interactions = (
                    uow.raw_interactions.get_student_interactions(
                        student.student_id
                    )
                )
                all_skill_interactions = (
                    uow.raw_interactions.get_student_skill_interactions(
                        student_id=student.student_id,
                        canonical_skill_id=skill.skill_id,
                    )
                )

                overall_adapters = [
                    _to_historical_adapter(item)
                    for item in all_student_interactions
                ]
                skill_adapters = [
                    _to_historical_adapter(item)
                    for item in all_skill_interactions
                ]

                features = build_production_features(
                    overall_history=overall_adapters,
                    skill_history=skill_adapters,
                )
                coverage = classify_behavioural_coverage(overall_adapters)
                prediction = predict_learning_state(features)

                # 8. Append Immutable Learning-State Snapshot
                snapshot = uow.supporting.add_snapshot(
                    assessment_id=assessment_id,
                    student_id=student.student_id,
                    session_id=session.session_id,
                    canonical_skill_id=skill.skill_id,
                    learning_state=prediction.learning_state,
                    evidence_level=prediction.evidence_level,
                    evidence_strength=prediction.evidence_strength,
                    behavioural_coverage=coverage.level,
                    model_used=prediction.model_used,
                    previous_interaction_count=features[
                        "previous_interaction_count"
                    ],
                    previous_skill_interaction_count=features[
                        "previous_skill_interaction_count"
                    ],
                    recent_interaction_count=coverage.recent_interaction_count,
                    attempt_observation_count=(
                        coverage.attempt_observation_count
                    ),
                    hint_observation_count=coverage.hint_observation_count,
                    response_time_observation_count=(
                        coverage.response_time_observation_count
                    ),
                    model_version="1.0",
                )

                # 9. Upsert Current Learning State
                uow.supporting.upsert_current(snapshot)

                # 10. Query distinct misconception count for context
                misconceptions = uow.supporting.get_misconceptions(
                    student_id=student.student_id,
                    skill_id=skill.skill_id,
                )
                misconception_count = len(misconceptions)

                # 11. Commit transaction
                uow.commit()

                return MemoryUpdateResponse(
                    student_id=request.student_id,
                    topic=request.topic,
                    subtopic=request.subtopic,
                    assessment_id=assessment_id,
                    snapshot_id=snapshot.snapshot_id,
                    learning_state=prediction.learning_state,
                    evidence_level=prediction.evidence_level,
                    evidence_strength=prediction.evidence_strength,
                    behavioural_coverage=coverage.level,
                    model_used=prediction.model_used,
                    previous_interaction_count=features[
                        "previous_interaction_count"
                    ],
                    previous_skill_interaction_count=features[
                        "previous_skill_interaction_count"
                    ],
                    recent_interaction_count=coverage.recent_interaction_count,
                    attempt_observation_count=(
                        coverage.attempt_observation_count
                    ),
                    hint_observation_count=coverage.hint_observation_count,
                    response_time_observation_count=(
                        coverage.response_time_observation_count
                    ),
                    misconception_count=misconception_count,
                    misconceptions=[m.display_error for m in misconceptions],
                    memory_updated=True,
                )

        except Exception as exc:
            if isinstance(exc, MemoryUpdateServiceError):
                raise
            raise MemoryUpdateServiceError(
                f"Memory update failed: {exc}"
            ) from exc

    def _update_memory_sqlite(
        self,
        request: AssessmentMemoryUpdateRequest,
        database_path: Any,
    ) -> MemoryUpdateResponse:
        """Legacy SQLite execution for isolated Phase 9 tests."""
        from src.database.assessment_repository import store_assessment
        from src.database.connection import (
            DEFAULT_DATABASE_PATH,
            get_connection,
            initialize_database,
        )
        from src.database.misconception_repository import (
            update_misconception_memory,
        )
        from src.database.state_repository import persist_learning_state
        from src.services.student_state_service import (
            get_student_state_service,
        )

        db_path = Path(database_path if database_path is not None else DEFAULT_DATABASE_PATH)
        initialize_database(db_path)
        conn = get_connection(db_path)
        try:
            conn.execute("BEGIN")
            stored = store_assessment(
                request=request, database_path=db_path, connection=conn
            )
            misconceptions = update_misconception_memory(
                request=request, database_path=db_path, connection=conn
            )
            state_service = (
                self.state_service
                if self.state_service is not None
                else get_student_state_service()
            )
            prediction = state_service.predict_from_stored_history(
                student_id=request.student_id,
                topic=request.topic,
                subtopic=request.subtopic,
                database_path=db_path,
                connection=conn,
            )
            snapshot = persist_learning_state(
                prediction=prediction,
                assessment_id=stored.assessment_id,
                database_path=db_path,
                connection=conn,
            )
            conn.commit()
            return MemoryUpdateResponse(
                student_id=prediction.student_id,
                topic=prediction.topic,
                subtopic=prediction.subtopic,
                assessment_id=stored.assessment_id,
                snapshot_id=snapshot.snapshot_id,
                learning_state=prediction.learning_state,
                evidence_level=prediction.evidence_level,
                evidence_strength=prediction.evidence_strength,
                behavioural_coverage=prediction.behavioural_coverage,
                model_used=prediction.model_used,
                previous_interaction_count=prediction.previous_interaction_count,
                previous_skill_interaction_count=prediction.previous_skill_interaction_count,
                recent_interaction_count=prediction.recent_interaction_count,
                attempt_observation_count=prediction.attempt_observation_count,
                hint_observation_count=prediction.hint_observation_count,
                response_time_observation_count=prediction.response_time_observation_count,
                misconception_count=len(misconceptions),
                memory_updated=True,
            )
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, MemoryUpdateServiceError):
                raise
            raise MemoryUpdateServiceError(
                f"Memory update failed: {exc}"
            ) from exc
        finally:
            conn.close()


_default_memory_update_service: MemoryUpdateService | None = None


def get_memory_update_service() -> MemoryUpdateService:
    global _default_memory_update_service
    if _default_memory_update_service is None:
        _default_memory_update_service = MemoryUpdateService()
    return _default_memory_update_service


def update_student_memory(
    request: AssessmentMemoryUpdateRequest,
    database_path: Any = None,
    session_factory: SessionFactory | None = None,
    **kwargs: Any,
) -> MemoryUpdateResponse:
    """Convenience interface for processing one completed assessment."""
    return get_memory_update_service().update_memory(
        request=request,
        database_path=database_path,
        session_factory=session_factory,
        **kwargs,
    )
