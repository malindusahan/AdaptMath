"""Create the source-of-truth raw interaction log.

Revision ID: 0002_raw_interactions
Revises: 0001_core_identity
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_raw_interactions"
down_revision: str | None = "0001_core_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_learning_sessions_session_student",
        "learning_sessions",
        ["session_id", "student_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "interaction_logs",
        sa.Column("interaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "canonical_skill_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("external_interaction_id", sa.String(255), nullable=False),
        sa.Column("problem_id", sa.String(255), nullable=True),
        sa.Column("question_id", sa.String(255), nullable=True),
        sa.Column("student_utterance", sa.Text(), nullable=True),
        sa.Column("tutor_response", sa.Text(), nullable=True),
        sa.Column("student_answer", sa.Text(), nullable=True),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("identified_error", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=True),
        sa.Column("hint_count", sa.Integer(), nullable=True),
        sa.Column("hint_total", sa.Integer(), nullable=True),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "length(trim(source)) > 0",
            name="source_not_blank",
        ),
        sa.CheckConstraint(
            "length(trim(external_interaction_id)) > 0",
            name="external_id_not_blank",
        ),
        sa.CheckConstraint(
            "attempt_count IS NULL OR attempt_count >= 0",
            name="attempt_count_nonnegative",
        ),
        sa.CheckConstraint(
            "hint_count IS NULL OR hint_count >= 0",
            name="hint_count_nonnegative",
        ),
        sa.CheckConstraint(
            "hint_total IS NULL OR hint_total >= 0",
            name="hint_total_nonnegative",
        ),
        sa.CheckConstraint(
            "response_time_ms IS NULL OR response_time_ms >= 0",
            name="response_time_nonnegative",
        ),
        sa.CheckConstraint(
            "hint_count IS NULL OR hint_total IS NULL OR hint_count <= hint_total",
            name="hint_count_within_total",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_interaction_logs_student_id_students",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_interaction_logs_session_student_learning_sessions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_interaction_logs_skill_id_canonical_skills",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("interaction_id", name="pk_interaction_logs"),
        sa.UniqueConstraint(
            "source",
            "external_interaction_id",
            name="uq_interaction_logs_source_external_id",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_interaction_logs_student_created",
        "interaction_logs",
        ["student_id", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_interaction_logs_session_created",
        "interaction_logs",
        ["session_id", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_interaction_logs_student_skill_created",
        "interaction_logs",
        ["student_id", "canonical_skill_id", "created_at"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_interaction_logs_student_skill_created",
        table_name="interaction_logs",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_interaction_logs_session_created",
        table_name="interaction_logs",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_interaction_logs_student_created",
        table_name="interaction_logs",
        schema=SCHEMA,
    )
    op.drop_table("interaction_logs", schema=SCHEMA)
    op.drop_constraint(
        "uq_learning_sessions_session_student",
        "learning_sessions",
        schema=SCHEMA,
        type_="unique",
    )
