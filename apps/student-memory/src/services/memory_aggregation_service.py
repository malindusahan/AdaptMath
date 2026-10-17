"""Aggregation and projection service for Short-Term, Long-Term, and Concept Memory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
import uuid

from src.database.models.core import CanonicalSkill, Student
from src.database.unit_of_work import UnitOfWork
from src.schemas.memory_projection import (
    ConceptMemoryRecord,
    ConceptMemoryUpsert,
    LongTermMemoryRecord,
    LongTermMemoryUpsert,
    ShortTermMemoryRecord,
    ShortTermMemoryUpsert,
)
from src.schemas.raw_interaction import RawInteractionCreate, RawInteractionRecord


class MemoryAggregationService:
    """Compute and persist derived memory projections."""

    @staticmethod
    def aggregate_short_term_memory(
        student_id: uuid.UUID,
        session_id: uuid.UUID,
        interactions: Sequence[RawInteractionRecord | RawInteractionCreate],
        student_external_id: str | None = None,
        skill_name: str | None = None,
    ) -> ShortTermMemoryUpsert:
        """Calculate Short-Term Memory projection from session interactions."""
        interaction_count = len(interactions)
        correct_count = sum(1 for item in interactions if item.is_correct is True)
        incorrect_count = sum(1 for item in interactions if item.is_correct is False)
        evaluated_count = correct_count + incorrect_count
        recent_accuracy = (
            (correct_count / evaluated_count)
            if (evaluated_count > 0 and interaction_count > 0)
            else None
        )

        attempts = [
            item.attempt_count
            for item in interactions
            if item.attempt_count is not None
        ]
        attempt_observation_count = len(attempts)
        attempt_sum = sum(attempts) if attempts else 0

        hints = [
            item.hint_count
            for item in interactions
            if item.hint_count is not None
        ]
        hint_observation_count = len(hints)
        hint_sum = sum(hints) if hints else 0

        response_times = [
            item.response_time_ms
            for item in interactions
            if item.response_time_ms is not None
        ]
        response_time_observation_count = len(response_times)
        response_time_sum_ms = float(sum(response_times)) if response_times else 0.0

        current_skill_id = (
            interactions[-1].canonical_skill_id if interactions else None
        )
        last_interaction_at = (
            interactions[-1].created_at if interactions else None
        )

        return ShortTermMemoryUpsert(
            student_id=student_id,
            session_id=session_id,
            student_external_id=student_external_id,
            current_skill_id=current_skill_id,
            skill_name=skill_name,
            interaction_count=interaction_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            recent_accuracy=recent_accuracy,
            attempt_observation_count=attempt_observation_count,
            attempt_sum=attempt_sum,
            hint_observation_count=hint_observation_count,
            hint_sum=hint_sum,
            response_time_observation_count=response_time_observation_count,
            response_time_sum_ms=response_time_sum_ms,
            last_interaction_at=last_interaction_at,
        )

    @staticmethod
    def aggregate_long_term_memory(
        student_id: uuid.UUID,
        interactions: Sequence[RawInteractionRecord | RawInteractionCreate],
        student_external_id: str | None = None,
    ) -> LongTermMemoryUpsert:
        """Calculate Long-Term Memory projection from all student interactions."""
        interaction_count = len(interactions)
        distinct_sessions = {item.session_id for item in interactions}
        distinct_skills = {
            item.canonical_skill_id
            for item in interactions
            if item.canonical_skill_id is not None
        }

        correct_count = sum(1 for item in interactions if item.is_correct is True)
        incorrect_count = sum(1 for item in interactions if item.is_correct is False)
        evaluated_count = correct_count + incorrect_count
        overall_accuracy = (
            (correct_count / evaluated_count)
            if (evaluated_count > 0 and interaction_count > 0)
            else None
        )

        attempts = [
            item.attempt_count
            for item in interactions
            if item.attempt_count is not None
        ]
        attempt_observation_count = len(attempts)
        attempt_sum = sum(attempts) if attempts else 0

        hints = [
            item.hint_count
            for item in interactions
            if item.hint_count is not None
        ]
        hint_observation_count = len(hints)
        hint_sum = sum(hints) if hints else 0

        response_times = [
            item.response_time_ms
            for item in interactions
            if item.response_time_ms is not None
        ]
        response_time_observation_count = len(response_times)
        response_time_sum_ms = float(sum(response_times)) if response_times else 0.0

        last_interaction_at = (
            interactions[-1].created_at if interactions else None
        )

        return LongTermMemoryUpsert(
            student_id=student_id,
            student_external_id=student_external_id,
            total_sessions=len(distinct_sessions),
            concept_count=len(distinct_skills),
            interaction_count=interaction_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            overall_accuracy=overall_accuracy,
            attempt_observation_count=attempt_observation_count,
            attempt_sum=attempt_sum,
            hint_observation_count=hint_observation_count,
            hint_sum=hint_sum,
            response_time_observation_count=response_time_observation_count,
            response_time_sum_ms=response_time_sum_ms,
            last_interaction_at=last_interaction_at,
        )

    @staticmethod
    def aggregate_concept_memory(
        student_id: uuid.UUID,
        canonical_skill_id: uuid.UUID,
        interactions: Sequence[RawInteractionRecord | RawInteractionCreate],
        student_external_id: str | None = None,
        skill_name: str | None = None,
    ) -> ConceptMemoryUpsert:
        """Calculate Concept Memory projection from skill interactions."""
        interaction_count = len(interactions)
        correct_count = sum(1 for item in interactions if item.is_correct is True)
        incorrect_count = sum(1 for item in interactions if item.is_correct is False)
        evaluated_count = correct_count + incorrect_count
        accuracy = (
            (correct_count / evaluated_count)
            if (evaluated_count > 0 and interaction_count > 0)
            else None
        )

        attempts = [
            item.attempt_count
            for item in interactions
            if item.attempt_count is not None
        ]
        attempt_observation_count = len(attempts)
        attempt_sum = sum(attempts) if attempts else 0

        hints = [
            item.hint_count
            for item in interactions
            if item.hint_count is not None
        ]
        hint_observation_count = len(hints)
        hint_sum = sum(hints) if hints else 0

        response_times = [
            item.response_time_ms
            for item in interactions
            if item.response_time_ms is not None
        ]
        response_time_observation_count = len(response_times)
        response_time_sum_ms = float(sum(response_times)) if response_times else 0.0

        first_interaction_at = (
            interactions[0].created_at if interactions else None
        )
        last_interaction_at = (
            interactions[-1].created_at if interactions else None
        )

        return ConceptMemoryUpsert(
            student_id=student_id,
            canonical_skill_id=canonical_skill_id,
            student_external_id=student_external_id,
            skill_name=skill_name,
            interaction_count=interaction_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            accuracy=accuracy,
            attempt_observation_count=attempt_observation_count,
            attempt_sum=attempt_sum,
            hint_observation_count=hint_observation_count,
            hint_sum=hint_sum,
            response_time_observation_count=response_time_observation_count,
            response_time_sum_ms=response_time_sum_ms,
            first_interaction_at=first_interaction_at,
            last_interaction_at=last_interaction_at,
        )

    def update_projections_for_interaction(
        self,
        interaction: RawInteractionRecord,
        uow: UnitOfWork,
    ) -> tuple[
        ShortTermMemoryRecord,
        LongTermMemoryRecord,
        ConceptMemoryRecord | None,
    ]:
        """
        Incrementally or comprehensively update STM, LTM, and Concept Memory
        for a newly added interaction within the active Unit of Work.
        """
        # Resolve presentation-only names when a real SQLAlchemy UnitOfWork is
        # available. Repository fakes used by deterministic aggregation tests
        # intentionally have no session and should retain ``None`` names.
        session = getattr(uow, "session", None)
        student = (
            session.get(Student, interaction.student_id)
            if session is not None
            else None
        )
        student_external_id = student.external_student_id if student else None

        # Resolve skill name
        skill_name: str | None = None
        if interaction.canonical_skill_id is not None and session is not None:
            skill = session.get(CanonicalSkill, interaction.canonical_skill_id)
            skill_name = skill.display_name if skill else None

        # 1. Update Short-Term Memory
        session_interactions = uow.raw_interactions.get_session_interactions(
            interaction.session_id
        )
        stm_upsert = self.aggregate_short_term_memory(
            student_id=interaction.student_id,
            session_id=interaction.session_id,
            interactions=session_interactions,
            student_external_id=student_external_id,
            skill_name=skill_name,
        )
        stm_record = uow.projections.upsert_short_term_memory(stm_upsert)

        # 2. Update Long-Term Memory
        student_interactions = uow.raw_interactions.get_student_interactions(
            interaction.student_id
        )
        ltm_upsert = self.aggregate_long_term_memory(
            student_id=interaction.student_id,
            interactions=student_interactions,
            student_external_id=student_external_id,
        )
        ltm_record = uow.projections.upsert_long_term_memory(ltm_upsert)

        # 3. Update Concept Memory (if canonical_skill_id is present)
        concept_record: ConceptMemoryRecord | None = None
        if interaction.canonical_skill_id is not None:
            skill_interactions = uow.raw_interactions.get_student_skill_interactions(
                student_id=interaction.student_id,
                canonical_skill_id=interaction.canonical_skill_id,
            )
            concept_upsert = self.aggregate_concept_memory(
                student_id=interaction.student_id,
                canonical_skill_id=interaction.canonical_skill_id,
                interactions=skill_interactions,
                student_external_id=student_external_id,
                skill_name=skill_name,
            )
            concept_record = uow.projections.upsert_concept_memory(concept_upsert)

        return stm_record, ltm_record, concept_record

    def rebuild_student_projections(
        self,
        student_id: uuid.UUID,
        uow: UnitOfWork,
    ) -> dict[str, object]:
        """
        Recompute and overwrite all projections for a student directly from interaction_logs.

        Guarantees that rebuilt projections match incremental updates exactly.
        """
        all_interactions = uow.raw_interactions.get_student_interactions(
            student_id
        )

        if not all_interactions:
            return {
                "short_term_memories": [],
                "long_term_memory": None,
                "concept_memories": [],
            }

        session = getattr(uow, "session", None)
        student = session.get(Student, student_id) if session is not None else None
        student_external_id = student.external_student_id if student else None

        # 1. Rebuild Long-Term Memory
        ltm_upsert = self.aggregate_long_term_memory(
            student_id=student_id,
            interactions=all_interactions,
            student_external_id=student_external_id,
        )
        ltm_record = uow.projections.upsert_long_term_memory(ltm_upsert)

        # 2. Rebuild Short-Term Memories per session
        sessions_map: dict[uuid.UUID, list[RawInteractionRecord]] = defaultdict(list)
        for interaction in all_interactions:
            sessions_map[interaction.session_id].append(interaction)

        stm_records: list[ShortTermMemoryRecord] = []
        for session_id, s_interactions in sessions_map.items():
            last_skill_id = s_interactions[-1].canonical_skill_id if s_interactions else None
            last_skill = (
                session.get(CanonicalSkill, last_skill_id)
                if session is not None and last_skill_id is not None
                else None
            )
            s_skill_name = last_skill.display_name if last_skill else None

            stm_upsert = self.aggregate_short_term_memory(
                student_id=student_id,
                session_id=session_id,
                interactions=s_interactions,
                student_external_id=student_external_id,
                skill_name=s_skill_name,
            )
            stm_records.append(uow.projections.upsert_short_term_memory(stm_upsert))

        # 3. Rebuild Concept Memories per canonical_skill_id
        skills_map: dict[uuid.UUID, list[RawInteractionRecord]] = defaultdict(list)
        for interaction in all_interactions:
            if interaction.canonical_skill_id is not None:
                skills_map[interaction.canonical_skill_id].append(interaction)

        concept_records: list[ConceptMemoryRecord] = []
        for skill_id, k_interactions in skills_map.items():
            k_skill = (
                session.get(CanonicalSkill, skill_id)
                if session is not None
                else None
            )
            k_skill_name = k_skill.display_name if k_skill else None

            concept_upsert = self.aggregate_concept_memory(
                student_id=student_id,
                canonical_skill_id=skill_id,
                interactions=k_interactions,
                student_external_id=student_external_id,
                skill_name=k_skill_name,
            )
            concept_records.append(
                uow.projections.upsert_concept_memory(concept_upsert)
            )

        return {
            "short_term_memories": stm_records,
            "long_term_memory": ltm_record,
            "concept_memories": concept_records,
        }
