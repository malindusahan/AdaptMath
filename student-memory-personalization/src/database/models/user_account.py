"""User Account PostgreSQL model for authentication and student linking."""

from __future__ import annotations

from datetime import date, datetime
import uuid

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA

SCHEMA = DEFAULT_POSTGRES_SCHEMA


class UserAccount(Base):
    """User account authentication entity linked to student identity."""

    __tablename__ = "user_accounts"
    __table_args__ = (
        UniqueConstraint("username", name="uq_user_accounts_username"),
        CheckConstraint("role = 'STUDENT'", name="chk_user_accounts_role"),
        CheckConstraint("length(trim(username)) > 0", name="chk_user_accounts_username_not_blank"),
        CheckConstraint("age IS NULL OR age >= 0", name="chk_user_accounts_age_nonnegative"),
        Index("ix_user_accounts_username", "username", unique=True),
        Index("ix_user_accounts_student_id", "student_id"),
        {"schema": SCHEMA},
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="STUDENT")
    student_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.students.student_id",
            name="fk_user_accounts_student_id_students",
            ondelete="SET NULL",
        ),
        nullable=True,
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
        onupdate=func.now(),
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
