"""Durable receipts for idempotent completed adaptive-attempt ingestion."""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA as SCHEMA


class CompletedAttemptReceipt(Base):
    """One durable receipt for one externally completed adaptive attempt."""

    __tablename__ = "completed_attempt_receipts"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "external_attempt_id",
            name="uq_completed_attempt_receipts_source_attempt",
        ),
        CheckConstraint(
            "status IN ('INGESTED')",
            name="status_domain",
        ),
        CheckConstraint(
            "session_status IN ('ACTIVE', 'COMPLETED')",
            name="session_status_domain",
        ),
        CheckConstraint(
            "question_count > 0 AND correct_count >= 0 AND incorrect_count >= 0 "
            "AND correct_count + incorrect_count = question_count",
            name="question_counts_valid",
        ),
        ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_completed_attempt_receipts_student_id_students",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_completed_attempt_receipts_session_student_learning_sessions",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_completed_attempt_receipts_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_completed_attempt_receipts_student_created",
            "student_id",
            "created_at",
        ),
        {"schema": SCHEMA},
    )

    receipt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    external_attempt_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="INGESTED",
    )
    session_status: Mapped[str] = mapped_column(String(20), nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False)
    incorrect_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
