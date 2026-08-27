"""Service for resolving free-text questions to canonical skills and contextual personalized memory."""

from __future__ import annotations

import json
import uuid
from typing import Any

from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.schemas.question_context import (
    LearningStateItem,
    MisconceptionItem,
    QuestionContextRequest,
    QuestionContextResponse,
    TopicContextItem,
)
from src.services.topic_extraction_service import TopicExtractionService
from src.topic_extraction.topic_extractor import HybridTopicExtractor, get_topic_extractor


class QuestionContextService:
    """
    Retrieves unified student learning context for a free-text question:
    1. Extracts and audits canonical topic from question text.
    2. Retrieves current STM, LTM, Concept Memory, Learning State, and Misconceptions.
    """

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        topic_extraction_service: TopicExtractionService | None = None,
        topic_extractor: HybridTopicExtractor | None = None,
        ontology_lookup: OntologyLookupService | None = None,
    ):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        self.topic_extractor = (
            topic_extractor if topic_extractor is not None else get_topic_extractor()
        )
        self.ontology_lookup = (
            ontology_lookup
            if ontology_lookup is not None
            else OntologyLookupService(session_factory=self.session_factory)
        )
        self.topic_extraction_service = (
            topic_extraction_service
            if topic_extraction_service is not None
            else TopicExtractionService(
                session_factory=self.session_factory,
                topic_extractor=self.topic_extractor,
                ontology_lookup=self.ontology_lookup,
            )
        )

    def get_question_context(
        self,
        request: QuestionContextRequest,
    ) -> QuestionContextResponse:
        """Process incoming student question and return extracted skill with personalized memory."""
        # 1. Resolve / provision student and session identities
        with UnitOfWork(self.session_factory) as uow:
            student = uow.identities.get_or_create_student(request.student_id)
            session = uow.identities.get_or_create_session(student, request.session_id)
            student_uuid = student.student_id
            session_uuid = session.session_id
            uow.commit()

        # 2. Extract and log topic extraction audit
        topic_result = self.topic_extraction_service.extract_and_audit(
            student_id=student_uuid,
            session_id=session_uuid,
            text=request.question,
        )

        topic_item = TopicContextItem(
            skill_id=topic_result.skill_id,
            skill_code=topic_result.skill_code,
            canonical_skill_name=topic_result.canonical_skill_name,
            display_name=topic_result.display_name,
            confidence=round(topic_result.confidence, 4),
            method=topic_result.method,
            needs_review=topic_result.needs_review,
            top_candidates=[c.to_dict() for c in topic_result.top_candidates],
            model_version=topic_result.model_version,
        )

        # 3. Retrieve student personalized memory projections and supporting states
        with UnitOfWork(self.session_factory) as uow:
            stm_record = uow.projections.get_short_term_memory(
                student_id=student_uuid,
                session_id=session_uuid,
            )
            ltm_record = uow.projections.get_long_term_memory(
                student_id=student_uuid,
            )

            concept_data: dict[str, Any] | None = None
            learning_state_item: LearningStateItem | None = None
            misconceptions_list: list[MisconceptionItem] = []

            # If a valid canonical skill was identified, query concept-specific context
            if not topic_result.is_abstain and topic_result.skill_id:
                skill_uuid = uuid.UUID(str(topic_result.skill_id))

                concept_record = uow.projections.get_concept_memory(
                    student_id=student_uuid,
                    canonical_skill_id=skill_uuid,
                )
                if concept_record is not None:
                    concept_data = concept_record.model_dump(mode="json")

                current_state = uow.supporting.get_current(
                    student_id=student_uuid,
                    skill_id=skill_uuid,
                )
                if current_state is not None:
                    learning_state_item = LearningStateItem(
                        learning_state=current_state.learning_state,
                        evidence_level=current_state.evidence_level,
                        evidence_strength=current_state.evidence_strength,
                        behavioural_coverage=current_state.behavioural_coverage,
                        model_used=current_state.model_used,
                        recent_interaction_count=current_state.recent_interaction_count,
                        attempt_observation_count=current_state.attempt_observation_count,
                        hint_observation_count=current_state.hint_observation_count,
                        response_time_observation_count=current_state.response_time_observation_count,
                        updated_at=(
                            current_state.updated_at.isoformat()
                            if hasattr(current_state.updated_at, "isoformat")
                            else str(current_state.updated_at)
                        ),
                    )

                misconceptions = uow.supporting.get_misconceptions(
                    student_id=student_uuid,
                    skill_id=skill_uuid,
                )
                misconceptions_list = [
                    MisconceptionItem(
                        misconception_id=str(m.misconception_id),
                        normalized_error=m.normalized_error,
                        display_error=m.display_error,
                        occurrence_count=m.occurrence_count,
                        last_seen_at=(
                            m.last_seen_at.isoformat()
                            if hasattr(m.last_seen_at, "isoformat")
                            else str(m.last_seen_at)
                        ),
                    )
                    for m in misconceptions
                ]

            return QuestionContextResponse(
                student_id=request.student_id,
                session_id=request.session_id,
                question=request.question,
                topic=topic_item,
                short_term_memory=stm_record.model_dump(mode="json") if stm_record else None,
                long_term_memory=ltm_record.model_dump(mode="json") if ltm_record else None,
                concept_memory=concept_data,
                learning_state=learning_state_item,
                misconceptions=misconceptions_list,
            )


# Global singleton instance cache
_QUESTION_CONTEXT_SERVICE: QuestionContextService | None = None


def get_question_context_service(
    session_factory: SessionFactory | None = None,
) -> QuestionContextService:
    global _QUESTION_CONTEXT_SERVICE
    if _QUESTION_CONTEXT_SERVICE is None or session_factory is not None:
        _QUESTION_CONTEXT_SERVICE = QuestionContextService(session_factory=session_factory)
    return _QUESTION_CONTEXT_SERVICE
