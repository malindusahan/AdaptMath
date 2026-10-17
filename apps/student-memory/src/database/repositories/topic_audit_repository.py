"""Append-only topic extraction audit repository."""
import json, uuid
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database.models.supporting_memory import TopicExtractionLog

class TopicAuditRepository:
    def __init__(self, session: Session): self.session=session
    def add(self, *, student_id, session_id, input_text, extraction_method, canonical_skill_id=None, confidence=None, needs_review=False, alternatives=None, model_version=None, ontology_version=None):
        row=TopicExtractionLog(extraction_id=uuid.uuid4(),student_id=student_id,session_id=session_id,canonical_skill_id=canonical_skill_id,input_text=input_text,extraction_method=extraction_method,confidence=confidence,needs_review=needs_review,alternatives_json=json.dumps(alternatives) if alternatives is not None else None,model_version=model_version,ontology_version=ontology_version)
        self.session.add(row); self.session.flush(); return row
    def for_student(self, student_id):
        return list(self.session.scalars(select(TopicExtractionLog).where(TopicExtractionLog.student_id==student_id).order_by(TopicExtractionLog.created_at,TopicExtractionLog.extraction_id)))
