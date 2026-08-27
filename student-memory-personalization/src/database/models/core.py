"""Core student, session, and canonical-skill PostgreSQL models."""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA


SCHEMA = DEFAULT_POSTGRES_SCHEMA


class Student(Base):
    """Stable internal identity for one externally identified student."""

    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint(
            "external_student_id",
            name="uq_students_external_student_id",
        ),
        {"schema": SCHEMA},
    )

    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    external_student_id: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="true",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class LearningSession(Base):
    """A bounded learning session owned by one student."""

    __tablename__ = "learning_sessions"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "student_id",
            name="uq_learning_sessions_session_student",
        ),
        UniqueConstraint(
            "student_id",
            "external_session_id",
            name="uq_learning_sessions_student_external_session",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'COMPLETED', 'ABANDONED')",
            name="status",
        ),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="time_order",
        ),
        Index("ix_learning_sessions_student_id", "student_id"),
        {"schema": SCHEMA},
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.students.student_id",
            name="fk_learning_sessions_student_id_students",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    external_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="ACTIVE",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class CanonicalSkill(Base):
    """Versionable canonical learning-skill identity."""

    __tablename__ = "canonical_skills"
    __table_args__ = (
        UniqueConstraint(
            "canonical_name",
            name="uq_canonical_skills_canonical_name",
        ),
        CheckConstraint(
            "length(trim(canonical_name)) > 0",
            name="name_not_blank",
        ),
        Index("ix_canonical_skills_parent_skill_id", "parent_skill_id"),
        {"schema": SCHEMA},
    )

    skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    parent_skill_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.canonical_skills.skill_id",
            name="fk_canonical_skills_parent_skill_id_canonical_skills",
            ondelete="SET NULL",
        ),
    )
    ontology_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default="1.0",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="true",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SkillAlias(Base):
    """Normalized surface form mapped to one canonical skill."""

    __tablename__ = "skill_aliases"
    __table_args__ = (
        UniqueConstraint(
            "normalized_alias",
            name="uq_skill_aliases_normalized_alias",
        ),
        CheckConstraint(
            "length(trim(normalized_alias)) > 0",
            name="normalized_not_blank",
        ),
        Index("ix_skill_aliases_skill_id", "skill_id"),
        {"schema": SCHEMA},
    )

    alias_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.canonical_skills.skill_id",
            name="fk_skill_aliases_skill_id_canonical_skills",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    alias: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_alias: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
