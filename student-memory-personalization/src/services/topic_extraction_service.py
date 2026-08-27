"""Application service for Topic Extraction with PostgreSQL audit logging."""

from __future__ import annotations

import json
import uuid
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.core import CanonicalSkill
from src.database.models.supporting_memory import TopicExtractionLog
from src.database.postgres_session import SessionFactory, session_scope
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.topic_extraction.topic_extractor import (
    HybridTopicExtractor,
    TopicExtractionResult,
    get_topic_extractor,
)


class TopicExtractionService:
    """
    Topic Extraction Service that resolves student queries to canonical skills
    and persists immutable audit logs in PostgreSQL table `topic_extraction_logs`.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        topic_extractor: HybridTopicExtractor | None = None,
        ontology_lookup: OntologyLookupService | None = None,
    ):
        self.session_factory = session_factory
        self.topic_extractor = (
            topic_extractor if topic_extractor is not None else get_topic_extractor()
        )
        self.ontology_lookup = (
            ontology_lookup
            if ontology_lookup is not None
            else OntologyLookupService(session_factory=session_factory)
        )

    def extract_and_audit(
        self,
        *,
        student_id: uuid.UUID,
        session_id: uuid.UUID,
        text: str,
    ) -> TopicExtractionResult:
        """
        Extract canonical topic from student input text and log audit record in DB.
        """
        # 1. Run extraction through hybrid model
        result = self.topic_extractor.extract(text)

        # 2. Parse canonical skill ID if available
        canonical_uuid: uuid.UUID | None = None
        if not result.is_abstain and result.skill_id:
            try:
                canonical_uuid = uuid.UUID(str(result.skill_id))
            except (ValueError, TypeError):
                canonical_uuid = None

        # 3. Format alternatives JSON
        alternatives_data = [c.to_dict() for c in result.top_candidates]

        # 4. Atomic PostgreSQL audit insertion
        with UnitOfWork(self.session_factory) as uow:
            assert uow.session is not None
            if canonical_uuid is not None:
                existing_skill = uow.session.get(CanonicalSkill, canonical_uuid)
                if existing_skill is None and result.display_name:
                    canonical_name = result.canonical_skill_name or f"math :: {result.display_name.lower()}"
                    new_skill = CanonicalSkill(
                        skill_id=canonical_uuid,
                        canonical_name=canonical_name,
                        display_name=result.display_name,
                        is_active=True,
                    )
                    uow.session.add(new_skill)
                    uow.session.flush()

            uow.topic_audits.add(
                student_id=student_id,
                session_id=session_id,
                input_text=text,
                extraction_method=result.method,
                canonical_skill_id=canonical_uuid,
                confidence=result.confidence,
                needs_review=result.needs_review,
                alternatives=alternatives_data,
                model_version=result.model_version,
                ontology_version="phase13-canonical-v1",
            )
            uow.commit()

        return result

    def get_student_audit_logs(
        self,
        student_id: uuid.UUID,
    ) -> list[TopicExtractionLog]:
        """Fetch all topic extraction audit logs for a student."""
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(TopicExtractionLog)
                    .where(TopicExtractionLog.student_id == student_id)
                    .order_by(TopicExtractionLog.created_at.asc())
                )
            )
