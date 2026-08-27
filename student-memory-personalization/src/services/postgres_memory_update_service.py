"""PostgreSQL transaction-safe memory update engine and learning-state pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import uuid

from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    StudentMisconception,
)
from src.database.postgres_session import SessionFactory
from src.database.unit_of_work import UnitOfWork
from src.features.behavioural_coverage import classify_behavioural_coverage
from src.features.production_feature_builder import build_production_features
from src.models.learning_state_model import predict_learning_state
from src.schemas.interaction import LearningStatePredictionResponse
from src.schemas.memory_projection import (
    ConceptMemoryRecord,
    LongTermMemoryRecord,
    ShortTermMemoryRecord,
)
from src.schemas.raw_interaction import RawInteractionCreate, RawInteractionRecord
from src.services.memory_aggregation_service import MemoryAggregationService


class PostgresMemoryUpdateError(Exception):
    """Base exception for PostgreSQL memory update failures."""


@dataclass(frozen=True)
class HistoricalInteractionAdapter:
    """Storage-neutral interaction adapter for feature and coverage builders."""

    is_correct: bool
    attempt_count: int | None
    attempt_data_available: bool
    hint_count: int | None
    hint_total: int | None
    hint_data_available: bool
    response_time_ms: float | None
    response_time_available: bool


def _to_historical_adapter(
    record: RawInteractionRecord | RawInteractionCreate,
) -> HistoricalInteractionAdapter:
    """Convert raw interaction log into historical interaction adapter."""
    return HistoricalInteractionAdapter(
        is_correct=bool(record.is_correct),
        attempt_count=record.attempt_count,
        attempt_data_available=(record.attempt_count is not None),
        hint_count=record.hint_count,
        hint_total=record.hint_total,
        hint_data_available=(
            record.hint_count is not None or record.hint_total is not None
        ),
        response_time_ms=record.response_time_ms,
        response_time_available=(record.response_time_ms is not None),
    )


@dataclass
class PostgresMemoryUpdateResult:
    """Structured result returned after an interaction updates PostgreSQL memory."""

    raw_interaction: RawInteractionRecord
    short_term_memory: ShortTermMemoryRecord
    long_term_memory: LongTermMemoryRecord
    concept_memory: ConceptMemoryRecord | None
    misconception: StudentMisconception | None
    snapshot: LearningStateSnapshot | None
    current_learning_state: CurrentLearningState | None
    features: dict[str, Any] | None
    coverage: str | None
    prediction: LearningStatePredictionResponse | None


class PostgresMemoryUpdateService:
    """Orchestrate completed interactions through the single-transaction memory pipeline."""

    def __init__(
        self,
        session_factory: SessionFactory,
        aggregation_service: MemoryAggregationService | None = None,
    ):
        self.session_factory = session_factory
        self.aggregation_service = (
            aggregation_service
            if aggregation_service is not None
            else MemoryAggregationService()
        )

    def update_interaction(
        self,
        interaction: RawInteractionCreate,
    ) -> PostgresMemoryUpdateResult:
        """Process one interaction through the complete memory pipeline in one transaction."""
        results = self.update_interactions([interaction])
        return results[0]

    def update_interactions(
        self,
        interactions: Sequence[RawInteractionCreate],
    ) -> list[PostgresMemoryUpdateResult]:
        """
        Process multiple interactions sequentially in a single PostgreSQL transaction.
        
        Rolls back all changes atomically if any step fails.
        """
        if not interactions:
            return []

        results: list[PostgresMemoryUpdateResult] = []

        try:
            with UnitOfWork(self.session_factory) as uow:
                for interaction in interactions:
                    result = self._process_single_interaction(interaction, uow)
                    results.append(result)

                uow.commit()
                return results

        except Exception as exc:
            if isinstance(exc, PostgresMemoryUpdateError):
                raise
            raise PostgresMemoryUpdateError(
                f"PostgreSQL memory pipeline failed: {exc}"
            ) from exc

    def _process_single_interaction(
        self,
        interaction: RawInteractionCreate,
        uow: UnitOfWork,
    ) -> PostgresMemoryUpdateResult:
        """Execute the complete memory pipeline for a single interaction within an active UoW."""
        # 1. Raw Interaction Log (append to interaction_logs)
        raw_record = uow.raw_interactions.add(interaction)

        # 2. STM, LTM, Concept Memory Aggregations
        stm_record, ltm_record, concept_record = (
            self.aggregation_service.update_projections_for_interaction(
                raw_record, uow
            )
        )

        # 3. Misconception Memory (if identified_error is present and canonical_skill_id is present)
        misconception_record: StudentMisconception | None = None
        if (
            raw_record.identified_error is not None
            and raw_record.identified_error.strip()
            and raw_record.canonical_skill_id is not None
        ):
            misconception_record = uow.supporting.upsert_misconception(
                student_id=raw_record.student_id,
                skill_id=raw_record.canonical_skill_id,
                text=raw_record.identified_error,
            )

        # 4. Learning-State Intelligence Pipeline (if canonical_skill_id is present)
        snapshot_record: LearningStateSnapshot | None = None
        current_state_record: CurrentLearningState | None = None
        features: dict[str, Any] | None = None
        coverage_level: str | None = None
        prediction: LearningStatePredictionResponse | None = None

        if raw_record.canonical_skill_id is not None:
            # Query full history up to current interaction
            all_student_interactions = (
                uow.raw_interactions.get_student_interactions(
                    raw_record.student_id
                )
            )
            all_skill_interactions = (
                uow.raw_interactions.get_student_skill_interactions(
                    student_id=raw_record.student_id,
                    canonical_skill_id=raw_record.canonical_skill_id,
                )
            )

            # Convert to adapter shape
            overall_history_adapters = [
                _to_historical_adapter(item)
                for item in all_student_interactions
            ]
            skill_history_adapters = [
                _to_historical_adapter(item)
                for item in all_skill_interactions
            ]

            # Build production features and classify coverage
            features = build_production_features(
                overall_history=overall_history_adapters,
                skill_history=skill_history_adapters,
            )
            coverage = classify_behavioural_coverage(overall_history_adapters)
            coverage_level = coverage.level

            # Model inference
            prediction = predict_learning_state(features)

            # Create immutable snapshot
            snapshot_record = uow.supporting.add_snapshot(
                student_id=raw_record.student_id,
                session_id=raw_record.session_id,
                canonical_skill_id=raw_record.canonical_skill_id,
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
                response_time_observation_count=(
                    coverage.response_time_observation_count
                ),
                model_version="1.0",
            )

            # Upsert current learning state
            current_state_record = uow.supporting.upsert_current(snapshot_record)

        return PostgresMemoryUpdateResult(
            raw_interaction=raw_record,
            short_term_memory=stm_record,
            long_term_memory=ltm_record,
            concept_memory=concept_record,
            misconception=misconception_record,
            snapshot=snapshot_record,
            current_learning_state=current_state_record,
            features=features,
            coverage=coverage_level,
            prediction=prediction,
        )

    def rebuild_projections(
        self,
        student_id: uuid.UUID,
    ) -> dict[str, object]:
        """Rebuild STM, LTM, and Concept Memory for a student in a single transaction."""
        try:
            with UnitOfWork(self.session_factory) as uow:
                result = self.aggregation_service.rebuild_student_projections(
                    student_id=student_id,
                    uow=uow,
                )
                uow.commit()
                return result
        except Exception as exc:
            raise PostgresMemoryUpdateError(
                f"Projection rebuild failed for student {student_id}: {exc}"
            ) from exc
