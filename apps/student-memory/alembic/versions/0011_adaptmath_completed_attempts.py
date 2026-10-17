"""Add durable AdaptMath attempt receipts and repair uniqueness.

Revision ID: 0011_adaptmath_ingestion
Revises: 0010_memory_skill_names
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0011_adaptmath_ingestion"
down_revision: str | None = "0010_memory_skill_names"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.create_table(
        "completed_attempt_receipts",
        sa.Column("receipt_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "canonical_skill_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("external_attempt_id", sa.String(255), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="INGESTED",
        ),
        sa.Column("session_status", sa.String(20), nullable=False),
        sa.Column("question_count", sa.Integer(), nullable=False),
        sa.Column("correct_count", sa.Integer(), nullable=False),
        sa.Column("incorrect_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("status IN ('INGESTED')", name="status_domain"),
        sa.CheckConstraint(
            "session_status IN ('ACTIVE', 'COMPLETED')",
            name="session_status_domain",
        ),
        sa.CheckConstraint(
            "question_count > 0 AND correct_count >= 0 AND incorrect_count >= 0 "
            "AND correct_count + incorrect_count = question_count",
            name="question_counts_valid",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_completed_attempt_receipts_student_id_students",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_completed_attempt_receipts_session_student_learning_sessions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_completed_attempt_receipts_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("receipt_id", name="pk_completed_attempt_receipts"),
        sa.UniqueConstraint(
            "source",
            "external_attempt_id",
            name="uq_completed_attempt_receipts_source_attempt",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_completed_attempt_receipts_student_created",
        "completed_attempt_receipts",
        ["student_id", "created_at"],
        schema=SCHEMA,
    )
    bind = op.get_bind()
    repair_constraint_exists = bind.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint c
                JOIN pg_namespace n ON n.oid = c.connamespace
                WHERE n.nspname = :schema
                  AND c.conname = :constraint_name
            )
            """
        ),
        {
            "schema": SCHEMA,
            "constraint_name": "uq_repair_outcomes_durable_event",
        },
    ).scalar_one()
    if not repair_constraint_exists:
        op.create_unique_constraint(
            "uq_repair_outcomes_durable_event",
            "repair_outcomes",
            [
                "student_id",
                "session_id",
                "canonical_skill_id",
                "interaction_id",
                "repair_action",
                "outcome",
            ],
            schema=SCHEMA,
            postgresql_nulls_not_distinct=True,
        )


def downgrade() -> None:
    op.drop_constraint(
        "uq_repair_outcomes_durable_event",
        "repair_outcomes",
        schema=SCHEMA,
        type_="unique",
    )
    op.drop_index(
        "ix_completed_attempt_receipts_student_created",
        table_name="completed_attempt_receipts",
        schema=SCHEMA,
    )
    op.drop_table("completed_attempt_receipts", schema=SCHEMA)
