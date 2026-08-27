"""Supporting PostgreSQL memory models."""
from __future__ import annotations
from datetime import datetime
import uuid
from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from src.database.base import Base
from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA as SCHEMA

STATE = "learning_state IN ('NEEDS_SUPPORT','DEVELOPING','STRONG','UNAVAILABLE')"
LEVEL = "evidence_level IN ('COLD_START','OVERALL_ONLY','PARTIAL_SKILL','FULL_SKILL')"
STRENGTH = "evidence_strength IN ('NONE','LOW','MEDIUM','HIGH')"
COVERAGE = "behavioural_coverage IN ('FULL_BEHAVIOURAL_COVERAGE','PARTIAL_BEHAVIOURAL_COVERAGE','CORRECTNESS_ONLY_COVERAGE')"

class StudentMisconception(Base):
    __tablename__ = "student_misconceptions"
    __table_args__ = (UniqueConstraint("student_id", "canonical_skill_id", "normalized_error"), CheckConstraint("occurrence_count >= 1", name="occurrence_positive"), ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.student_id"], ondelete="CASCADE"), ForeignKeyConstraint(["canonical_skill_id"], [f"{SCHEMA}.canonical_skills.skill_id"], ondelete="CASCADE"), {"schema": SCHEMA})
    misconception_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    skill_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    normalized_error: Mapped[str] = mapped_column(String(500), nullable=False)
    display_error: Mapped[str] = mapped_column(Text, nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class LearningStateSnapshot(Base):
    __tablename__ = "learning_state_snapshots"
    __table_args__ = (CheckConstraint(STATE, name="state_domain"), CheckConstraint(LEVEL, name="level_domain"), CheckConstraint(STRENGTH, name="strength_domain"), CheckConstraint(COVERAGE, name="coverage_domain"), CheckConstraint("previous_interaction_count >= 0 AND previous_skill_interaction_count >= 0", name="history_counts_nonnegative"), ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.student_id"], ondelete="CASCADE"), ForeignKeyConstraint(["session_id", "student_id"], [f"{SCHEMA}.learning_sessions.session_id", f"{SCHEMA}.learning_sessions.student_id"], ondelete="RESTRICT"), ForeignKeyConstraint(["canonical_skill_id"], [f"{SCHEMA}.canonical_skills.skill_id"], ondelete="RESTRICT"), {"schema": SCHEMA})
    snapshot_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    assessment_id: Mapped[int | None] = mapped_column(BigInteger)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    skill_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    learning_state: Mapped[str] = mapped_column(String(30), nullable=False)
    evidence_level: Mapped[str] = mapped_column(String(30), nullable=False)
    evidence_strength: Mapped[str] = mapped_column(String(20), nullable=False)
    behavioural_coverage: Mapped[str] = mapped_column(String(40), nullable=False)
    model_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    previous_interaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_skill_interaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    recent_interaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hint_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class CurrentLearningState(Base):
    __tablename__ = "current_learning_state"
    __table_args__ = (CheckConstraint(STATE, name="state_domain"), CheckConstraint(LEVEL, name="level_domain"), CheckConstraint(STRENGTH, name="strength_domain"), CheckConstraint(COVERAGE, name="coverage_domain"), ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.student_id"], ondelete="CASCADE"), ForeignKeyConstraint(["canonical_skill_id"], [f"{SCHEMA}.canonical_skills.skill_id"], ondelete="RESTRICT"), ForeignKeyConstraint(["last_snapshot_id"], [f"{SCHEMA}.learning_state_snapshots.snapshot_id"], ondelete="RESTRICT"), {"schema": SCHEMA})
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    skill_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    learning_state: Mapped[str] = mapped_column(String(30), nullable=False)
    evidence_level: Mapped[str] = mapped_column(String(30), nullable=False)
    evidence_strength: Mapped[str] = mapped_column(String(20), nullable=False)
    behavioural_coverage: Mapped[str] = mapped_column(String(40), nullable=False)
    model_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    recent_interaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hint_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    last_assessment_id: Mapped[int | None] = mapped_column(BigInteger)
    last_snapshot_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class TopicExtractionLog(Base):
    __tablename__ = "topic_extraction_logs"
    __table_args__ = (CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="confidence_range"), ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.student_id"], ondelete="CASCADE"), ForeignKeyConstraint(["session_id", "student_id"], [f"{SCHEMA}.learning_sessions.session_id", f"{SCHEMA}.learning_sessions.student_id"], ondelete="RESTRICT"), ForeignKeyConstraint(["canonical_skill_id"], [f"{SCHEMA}.canonical_skills.skill_id"], ondelete="SET NULL"), {"schema": SCHEMA})
    extraction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_skill_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alternatives_json: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(100))
    ontology_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class RepairOutcome(Base):
    __tablename__ = "repair_outcomes"
    __table_args__ = (CheckConstraint("score IS NULL OR score BETWEEN 0 AND 1", name="score_range"), UniqueConstraint("student_id", "session_id", "canonical_skill_id", "interaction_id", "repair_action", "outcome", name="uq_repair_outcomes_durable_event", postgresql_nulls_not_distinct=True), ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.student_id"], ondelete="CASCADE"), ForeignKeyConstraint(["session_id", "student_id"], [f"{SCHEMA}.learning_sessions.session_id", f"{SCHEMA}.learning_sessions.student_id"], ondelete="RESTRICT"), ForeignKeyConstraint(["canonical_skill_id"], [f"{SCHEMA}.canonical_skills.skill_id"], ondelete="RESTRICT"), ForeignKeyConstraint(["interaction_id"], [f"{SCHEMA}.interaction_logs.interaction_id"], ondelete="SET NULL"), {"schema": SCHEMA})
    repair_outcome_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    interaction_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    repair_action: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

