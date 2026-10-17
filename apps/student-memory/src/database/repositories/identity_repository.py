"""PostgreSQL identity, session, and compatibility-skill repository."""
import uuid
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from src.database.models.core import CanonicalSkill, LearningSession, Student

def normalize_skill_name(topic: str, subtopic: str | None) -> str:
    parts = [" ".join(topic.strip().split()).lower()]
    if subtopic and subtopic.strip():
        parts.append(" ".join(subtopic.strip().split()).lower())
    return " :: ".join(parts)

class IdentityRepository:
    def __init__(self, session: Session): self.session = session
    def get_or_create_student(self, external_id: str) -> Student:
        value = external_id.strip()
        # Try finding by external_student_id first
        row = self.session.scalar(select(Student).where(Student.external_student_id == value))
        if row: return row
        # Try finding by UUID student_id if value is a valid UUID
        try:
            val_uuid = uuid.UUID(value)
            row = self.session.scalar(select(Student).where(Student.student_id == val_uuid))
            if row: return row
        except (ValueError, TypeError):
            pass
        row = Student(student_id=uuid.uuid4(), external_student_id=value)
        self.session.add(row); self.session.flush(); return row
    def get_student(self, external_id: str):
        value = external_id.strip()
        row = self.session.scalar(select(Student).where(Student.external_student_id == value))
        if row: return row
        try:
            val_uuid = uuid.UUID(value)
            return self.session.scalar(select(Student).where(Student.student_id == val_uuid))
        except (ValueError, TypeError):
            return None
    def get_skill(self, topic: str, subtopic: str | None):
        return self.session.scalar(select(CanonicalSkill).where(CanonicalSkill.canonical_name == normalize_skill_name(topic, subtopic)))
    def lock_student_update(self, external_id: str) -> None:
        if self.session.bind is not None and self.session.bind.dialect.name == "postgresql":
            self.session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"), {"key": external_id})
    def lock_external_attempt(self, source: str, external_attempt_id: str) -> None:
        if self.session.bind is not None and self.session.bind.dialect.name == "postgresql":
            self.session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"),
                {"key": f"completed-attempt:{source}:{external_attempt_id}"},
            )
    def next_assessment_id(self) -> int:
        try:
            return int(self.session.scalar(text("SELECT nextval('student_memory.assessment_id_seq')")))
        except Exception:
            self.session.rollback()
            self.session.execute(text("CREATE SEQUENCE IF NOT EXISTS student_memory.assessment_id_seq START WITH 1000"))
            self.session.commit()
            return int(self.session.scalar(text("SELECT nextval('student_memory.assessment_id_seq')")))
    def get_or_create_session(self, student: Student, external_session_id: str) -> LearningSession:
        row = self.session.scalar(select(LearningSession).where(LearningSession.student_id == student.student_id, LearningSession.external_session_id == external_session_id))
        if row: return row
        row = LearningSession(session_id=uuid.uuid4(), student_id=student.student_id, external_session_id=external_session_id)
        self.session.add(row); self.session.flush(); return row
    def get_session(self, student: Student, session_identifier: str) -> LearningSession | None:
        value = session_identifier.strip()
        try:
            session_uuid = uuid.UUID(value)
        except (ValueError, TypeError):
            session_uuid = None
        identity_match = LearningSession.external_session_id == value
        if session_uuid is not None:
            identity_match = identity_match | (
                LearningSession.session_id == session_uuid
            )
        return self.session.scalar(
            select(LearningSession).where(
                LearningSession.student_id == student.student_id,
                identity_match,
            )
        )
    def get_or_create_skill(self, topic: str, subtopic: str | None) -> CanonicalSkill:
        name = normalize_skill_name(topic, subtopic)
        row = self.session.scalar(select(CanonicalSkill).where(CanonicalSkill.canonical_name == name))
        if row: return row
        display = topic.strip() + (f" / {subtopic.strip()}" if subtopic and subtopic.strip() else "")
        row = CanonicalSkill(skill_id=uuid.uuid4(), canonical_name=name, display_name=display)
        self.session.add(row); self.session.flush(); return row
