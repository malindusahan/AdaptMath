"""Derived Short-Term, Long-Term, and Concept Memory projection models."""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA


SCHEMA = DEFAULT_POSTGRES_SCHEMA


def _evidence_constraints(prefix: str) -> tuple[CheckConstraint, ...]:
    return (
        CheckConstraint(
            "interaction_count >= 0",
            name="interaction_count_nonnegative",
        ),
        CheckConstraint("correct_count >= 0", name="correct_count_nonnegative"),
        CheckConstraint(
            "incorrect_count >= 0",
            name="incorrect_count_nonnegative",
        ),
        CheckConstraint(
            "correct_count + incorrect_count <= interaction_count",
            name="outcomes_within_interactions",
        ),
        CheckConstraint(
            "attempt_observation_count >= 0",
            name="attempt_observations_nonnegative",
        ),
        CheckConstraint("attempt_sum >= 0", name="attempt_sum_nonnegative"),
        CheckConstraint(
            "attempt_observation_count > 0 OR attempt_sum = 0",
            name="attempt_zero_observation_sum",
        ),
        CheckConstraint(
            "hint_observation_count >= 0",
            name="hint_observations_nonnegative",
        ),
        CheckConstraint("hint_sum >= 0", name="hint_sum_nonnegative"),
        CheckConstraint(
            "hint_observation_count > 0 OR hint_sum = 0",
            name="hint_zero_observation_sum",
        ),
        CheckConstraint(
            "response_time_observation_count >= 0",
            name="response_observations_nonnegative",
        ),
        CheckConstraint(
            "response_time_sum_ms >= 0",
            name="response_sum_nonnegative",
        ),
        CheckConstraint(
            "response_time_observation_count > 0 OR response_time_sum_ms = 0",
            name="response_zero_observation_sum",
        ),
    )


class ShortTermMemory(Base):
    """Current derived memory for one student learning session."""

    __tablename__ = "short_term_memory"
    __table_args__ = (
        *_evidence_constraints("short_term_memory"),
        CheckConstraint(
            "recent_accuracy IS NULL OR "
            "(recent_accuracy >= 0 AND recent_accuracy <= 1)",
            name="accuracy_range",
        ),
        CheckConstraint(
            "interaction_count > 0 OR recent_accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_short_term_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_short_term_memory_session_student_learning_sessions",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["current_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_short_term_memory_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        Index("ix_short_term_memory_current_skill", "current_skill_id"),
        Index("ix_short_term_memory_updated", "updated_at"),
        {"schema": SCHEMA},
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_skill_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    skill_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    incorrect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recent_accuracy: Mapped[float | None] = mapped_column(Float)

    attempt_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    attempt_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hint_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    hint_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_time_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    response_time_sum_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0
    )

    last_interaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LongTermMemory(Base):
    """Current derived cross-session memory for one student."""

    __tablename__ = "long_term_memory"
    __table_args__ = (
        *_evidence_constraints("long_term_memory"),
        CheckConstraint("total_sessions >= 0", name="sessions_nonnegative"),
        CheckConstraint("concept_count >= 0", name="concept_count_nonnegative"),
        CheckConstraint(
            "overall_accuracy IS NULL OR "
            "(overall_accuracy >= 0 AND overall_accuracy <= 1)",
            name="accuracy_range",
        ),
        CheckConstraint(
            "interaction_count > 0 OR overall_accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_long_term_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        Index("ix_long_term_memory_last_interaction", "last_interaction_at"),
        Index("ix_long_term_memory_updated", "updated_at"),
        {"schema": SCHEMA},
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    total_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    incorrect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    overall_accuracy: Mapped[float | None] = mapped_column(Float)

    attempt_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    attempt_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hint_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    hint_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_time_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    response_time_sum_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0
    )

    concept_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_interaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ConceptMemory(Base):
    """Current derived memory for one student and canonical skill."""

    __tablename__ = "concept_memory"
    __table_args__ = (
        *_evidence_constraints("concept_memory"),
        CheckConstraint(
            "accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)",
            name="accuracy_range",
        ),
        CheckConstraint(
            "interaction_count > 0 OR accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        CheckConstraint(
            "first_interaction_at IS NULL OR last_interaction_at IS NULL OR "
            "first_interaction_at <= last_interaction_at",
            name="interaction_time_order",
        ),
        ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_concept_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_concept_memory_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_concept_memory_skill_last_interaction",
            "canonical_skill_id",
            "last_interaction_at",
        ),
        Index(
            "ix_concept_memory_student_last_interaction",
            "student_id",
            "last_interaction_at",
        ),
        {"schema": SCHEMA},
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    student_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    skill_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    incorrect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[float | None] = mapped_column(Float)

    attempt_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    attempt_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hint_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    hint_sum: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_time_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    response_time_sum_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0
    )

    first_interaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_interaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

