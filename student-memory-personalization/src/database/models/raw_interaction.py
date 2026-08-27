"""PostgreSQL source-of-truth interaction log model."""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
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


class InteractionLog(Base):
    """Immutable raw evidence for one student learning interaction."""

    __tablename__ = "interaction_logs"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "external_interaction_id",
            name="uq_interaction_logs_source_external_id",
        ),
        CheckConstraint(
            "length(trim(source)) > 0",
            name="source_not_blank",
        ),
        CheckConstraint(
            "length(trim(external_interaction_id)) > 0",
            name="external_id_not_blank",
        ),
        CheckConstraint(
            "attempt_count IS NULL OR attempt_count >= 0",
            name="attempt_count_nonnegative",
        ),
        CheckConstraint(
            "hint_count IS NULL OR hint_count >= 0",
            name="hint_count_nonnegative",
        ),
        CheckConstraint(
            "hint_total IS NULL OR hint_total >= 0",
            name="hint_total_nonnegative",
        ),
        CheckConstraint(
            "response_time_ms IS NULL OR response_time_ms >= 0",
            name="response_time_nonnegative",
        ),
        CheckConstraint(
            "hint_count IS NULL OR hint_total IS NULL OR hint_count <= hint_total",
            name="hint_count_within_total",
        ),
        ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_interaction_logs_student_id_students",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_interaction_logs_session_student_learning_sessions",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_interaction_logs_skill_id_canonical_skills",
            ondelete="SET NULL",
        ),
        Index(
            "ix_interaction_logs_student_created",
            "student_id",
            "created_at",
        ),
        Index(
            "ix_interaction_logs_session_created",
            "session_id",
            "created_at",
        ),
        Index(
            "ix_interaction_logs_student_skill_created",
            "student_id",
            "canonical_skill_id",
            "created_at",
        ),
        {"schema": SCHEMA},
    )

    interaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_skill_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True)
    )

    source: Mapped[str] = mapped_column(String(100), nullable=False)
    external_interaction_id: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_id: Mapped[str | None] = mapped_column(String(255))
    question_id: Mapped[str | None] = mapped_column(String(255))

    student_utterance: Mapped[str | None] = mapped_column(Text)
    tutor_response: Mapped[str | None] = mapped_column(Text)
    student_answer: Mapped[str | None] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text)

    is_correct: Mapped[bool | None] = mapped_column(Boolean)
    identified_error: Mapped[str | None] = mapped_column(Text)

    attempt_count: Mapped[int | None] = mapped_column(Integer)
    hint_count: Mapped[int | None] = mapped_column(Integer)
    hint_total: Mapped[int | None] = mapped_column(Integer)
    response_time_ms: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

