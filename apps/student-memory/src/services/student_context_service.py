"""Unified Student Memory Context Service aggregating multi-tier personalization data."""

from __future__ import annotations

import uuid
from typing import Any

from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.schemas.student_context import (
    RecentInteractionItem,
    RepairOutcomeItem,
    StudentContextResponse,
)


class StudentContextService:
    """Read-only aggregation service retrieving unified memory state for tutoring agents."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        ontology_lookup: OntologyLookupService | None = None,
    ):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        self.ontology_lookup = (
            ontology_lookup
            if ontology_lookup is not None
            else OntologyLookupService(session_factory=self.session_factory)
        )

    def get_student_context(
        self,
        student_id: str,
        session_id: str | None = None,
        skill_id: str | None = None,
        limit: int = 10,
    ) -> StudentContextResponse:
        """
        Aggregate complete student personalization memory:
        - STM (scoped to session if provided)
        - LTM (student-wide)
        - Concept Memory (scoped to skill if provided)
        - Learning State / Misconceptions (scoped to skill if provided)
        - Recent Raw Interactions (chronological, bounded by limit)
        - Recent Repair Outcomes (chronological, bounded by limit)
        """
        if limit < 1:
            limit = 10

        with UnitOfWork(self.session_factory) as uow:
            student = uow.identities.get_student(student_id)
            if student is None:
                # Student does not exist yet -> return safe empty context
                return StudentContextResponse(
                    student_id=student_id,
                    session_id=session_id,
                    skill_id=skill_id,
                    short_term_memory=None,
                    long_term_memory=None,
                    concept_memory=None,
                    learning_state=None,
                    misconceptions=[],
                    recent_interactions=[],
                    recent_repairs=[],
                )

            student_uuid = student.student_id

            # 1. Long-Term Memory (Student-wide)
            ltm_record = uow.projections.get_long_term_memory(student_uuid)
            ltm_data = ltm_record.model_dump(mode="json") if ltm_record else None

            # 2. Short-Term Memory (Session-scoped)
            stm_data: dict[str, Any] | None = None
            session_uuid: uuid.UUID | None = None
            if session_id:
                session_obj = uow.identities.get_session(student, session_id)
                if session_obj is not None:
                    session_uuid = session_obj.session_id
                    stm_record = uow.projections.get_short_term_memory(
                        student_uuid,
                        session_uuid,
                    )
                    if stm_record:
                        stm_data = stm_record.model_dump(mode="json")

            # 3. Resolve canonical skill UUID if skill_id provided
            canonical_skill_uuid: uuid.UUID | None = None
            if skill_id:
                try:
                    canonical_skill_uuid = uuid.UUID(skill_id)
                except ValueError:
                    # Attempt lookup by skill_code or canonical_name
                    skill_entry = (
                        self.ontology_lookup.get_by_skill_code(
                            skill_id,
                            session=uow.session,
                        )
                        or self.ontology_lookup.get_by_canonical_name(
                            skill_id,
                            session=uow.session,
                        )
                        or self.ontology_lookup.get_by_alias(
                            skill_id,
                            session=uow.session,
                        )
                    )
                    if skill_entry:
                        canonical_skill_uuid = skill_entry.skill_id

            # 4. Concept-scoped memory, learning state, and misconceptions
            concept_data: dict[str, Any] | None = None
            learning_state_data: dict[str, Any] | None = None
            misconceptions_list: list[dict[str, Any]] = []

            if canonical_skill_uuid:
                concept_record = uow.projections.get_concept_memory(student_uuid, canonical_skill_uuid)
                if concept_record:
                    concept_data = concept_record.model_dump(mode="json")

                current_state = uow.supporting.get_current(student_uuid, canonical_skill_uuid)
                if current_state:
                    learning_state_data = {
                        "learning_state": current_state.learning_state,
                        "evidence_level": current_state.evidence_level,
                        "evidence_strength": current_state.evidence_strength,
                        "behavioural_coverage": current_state.behavioural_coverage,
                        "model_used": current_state.model_used,
                        "recent_interaction_count": current_state.recent_interaction_count,
                        "attempt_observation_count": current_state.attempt_observation_count,
                        "hint_observation_count": current_state.hint_observation_count,
                        "response_time_observation_count": current_state.response_time_observation_count,
                        "updated_at": (
                            current_state.updated_at.isoformat()
                            if hasattr(current_state.updated_at, "isoformat")
                            else str(current_state.updated_at)
                        ),
                    }

                misc_records = uow.supporting.get_misconceptions(student_uuid, canonical_skill_uuid)
                misconceptions_list = [
                    {
                        "misconception_id": str(m.misconception_id),
                        "normalized_error": m.normalized_error,
                        "display_error": m.display_error,
                        "occurrence_count": m.occurrence_count,
                        "last_seen_at": (
                            m.last_seen_at.isoformat()
                            if hasattr(m.last_seen_at, "isoformat")
                            else str(m.last_seen_at)
                        ),
                    }
                    for m in misc_records
                ]

                raw_repairs = uow.repairs.for_student_skill(student_uuid, canonical_skill_uuid)
                raw_interactions = uow.raw_interactions.get_student_skill_interactions(
                    student_uuid, canonical_skill_uuid
                )
            else:
                # Student-wide supporting memory
                current_state = uow.supporting.get_current(student_uuid, None)
                if current_state:
                    learning_state_data = {
                        "learning_state": current_state.learning_state,
                        "evidence_level": current_state.evidence_level,
                        "evidence_strength": current_state.evidence_strength,
                        "behavioural_coverage": current_state.behavioural_coverage,
                        "model_used": current_state.model_used,
                        "recent_interaction_count": current_state.recent_interaction_count,
                        "attempt_observation_count": current_state.attempt_observation_count,
                        "hint_observation_count": current_state.hint_observation_count,
                        "response_time_observation_count": current_state.response_time_observation_count,
                        "updated_at": (
                            current_state.updated_at.isoformat()
                            if hasattr(current_state.updated_at, "isoformat")
                            else str(current_state.updated_at)
                        ),
                    }

                misc_records = uow.supporting.get_misconceptions(student_uuid, None)
                misconceptions_list = [
                    {
                        "misconception_id": str(m.misconception_id),
                        "normalized_error": m.normalized_error,
                        "display_error": m.display_error,
                        "occurrence_count": m.occurrence_count,
                        "last_seen_at": (
                            m.last_seen_at.isoformat()
                            if hasattr(m.last_seen_at, "isoformat")
                            else str(m.last_seen_at)
                        ),
                    }
                    for m in misc_records
                ]

                raw_repairs = uow.repairs.for_student(student_uuid)
                if session_uuid:
                    raw_interactions = uow.raw_interactions.get_session_interactions(session_uuid)
                elif session_id:
                    raw_interactions = []
                else:
                    raw_interactions = uow.raw_interactions.get_student_interactions(student_uuid)

            # Limit chronological interactions (latest N)
            latest_interactions = raw_interactions[-limit:] if len(raw_interactions) > limit else raw_interactions
            recent_interactions_items = [
                RecentInteractionItem(
                    interaction_id=str(i.interaction_id),
                    session_id=str(i.session_id),
                    canonical_skill_id=str(i.canonical_skill_id) if i.canonical_skill_id else None,
                    student_utterance=i.student_utterance,
                    identified_error=i.identified_error,
                    is_correct=i.is_correct,
                    attempt_count=i.attempt_count,
                    hint_count=i.hint_count,
                    response_time_ms=i.response_time_ms,
                    created_at=(
                        i.created_at.isoformat()
                        if hasattr(i.created_at, "isoformat")
                        else str(i.created_at)
                    ),
                )
                for i in latest_interactions
            ]

            # Limit chronological repairs (latest N)
            latest_repairs = raw_repairs[-limit:] if len(raw_repairs) > limit else raw_repairs
            recent_repairs_items = [
                RepairOutcomeItem(
                    repair_outcome_id=str(r.repair_outcome_id),
                    session_id=str(r.session_id),
                    canonical_skill_id=str(r.canonical_skill_id),
                    interaction_id=str(r.interaction_id) if r.interaction_id else None,
                    repair_action=r.repair_action,
                    outcome=r.outcome,
                    score=r.score,
                    notes=r.notes,
                    created_at=(
                        r.created_at.isoformat()
                        if hasattr(r.created_at, "isoformat")
                        else str(r.created_at)
                    ),
                )
                for r in latest_repairs
            ]

            return StudentContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                short_term_memory=stm_data,
                long_term_memory=ltm_data,
                concept_memory=concept_data,
                learning_state=learning_state_data,
                misconceptions=misconceptions_list,
                recent_interactions=recent_interactions_items,
                recent_repairs=recent_repairs_items,
            )

    get_full_student_context = get_student_context


# Singleton factory cache
_STUDENT_CONTEXT_SERVICE: StudentContextService | None = None


def get_student_context_service(
    session_factory: SessionFactory | None = None,
) -> StudentContextService:
    global _STUDENT_CONTEXT_SERVICE
    if _STUDENT_CONTEXT_SERVICE is None or session_factory is not None:
        _STUDENT_CONTEXT_SERVICE = StudentContextService(session_factory=session_factory)
    return _STUDENT_CONTEXT_SERVICE
