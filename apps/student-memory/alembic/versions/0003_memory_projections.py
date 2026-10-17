"""Create Short-Term, Long-Term, and Concept Memory projections.

Revision ID: 0003_memory_projections
Revises: 0002_raw_interactions
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_memory_projections"
down_revision: str | None = "0002_raw_interactions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def _evidence_columns() -> list[sa.Column]:
    return [
        sa.Column("interaction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("correct_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("incorrect_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "attempt_observation_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("attempt_sum", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "hint_observation_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("hint_sum", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "response_time_observation_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "response_time_sum_ms",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    ]


def _evidence_constraints(table: str) -> list[sa.CheckConstraint]:
    return [
        sa.CheckConstraint("interaction_count >= 0", name="interaction_count_nonnegative"),
        sa.CheckConstraint("correct_count >= 0", name="correct_count_nonnegative"),
        sa.CheckConstraint("incorrect_count >= 0", name="incorrect_count_nonnegative"),
        sa.CheckConstraint(
            "correct_count + incorrect_count <= interaction_count",
            name="outcomes_within_interactions",
        ),
        sa.CheckConstraint(
            "attempt_observation_count >= 0",
            name="attempt_observations_nonnegative",
        ),
        sa.CheckConstraint("attempt_sum >= 0", name="attempt_sum_nonnegative"),
        sa.CheckConstraint(
            "attempt_observation_count > 0 OR attempt_sum = 0",
            name="attempt_zero_observation_sum",
        ),
        sa.CheckConstraint(
            "hint_observation_count >= 0",
            name="hint_observations_nonnegative",
        ),
        sa.CheckConstraint("hint_sum >= 0", name="hint_sum_nonnegative"),
        sa.CheckConstraint(
            "hint_observation_count > 0 OR hint_sum = 0",
            name="hint_zero_observation_sum",
        ),
        sa.CheckConstraint(
            "response_time_observation_count >= 0",
            name="response_observations_nonnegative",
        ),
        sa.CheckConstraint(
            "response_time_sum_ms >= 0",
            name="response_sum_nonnegative",
        ),
        sa.CheckConstraint(
            "response_time_observation_count > 0 OR response_time_sum_ms = 0",
            name="response_zero_observation_sum",
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "short_term_memory",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_external_id", sa.String(255), nullable=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("current_skill_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("skill_name", sa.String(255), nullable=True),
        *_evidence_columns(),
        sa.Column("recent_accuracy", sa.Float(), nullable=True),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        *_evidence_constraints("short_term_memory"),
        sa.CheckConstraint(
            "recent_accuracy IS NULL OR (recent_accuracy >= 0 AND recent_accuracy <= 1)",
            name="accuracy_range",
        ),
        sa.CheckConstraint(
            "interaction_count > 0 OR recent_accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_short_term_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["session_id", "student_id"],
            [
                f"{SCHEMA}.learning_sessions.session_id",
                f"{SCHEMA}.learning_sessions.student_id",
            ],
            name="fk_short_term_memory_session_student_learning_sessions",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["current_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_short_term_memory_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "student_id", "session_id", name="pk_short_term_memory"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_short_term_memory_current_skill",
        "short_term_memory",
        ["current_skill_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_short_term_memory_updated",
        "short_term_memory",
        ["updated_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "long_term_memory",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_external_id", sa.String(255), nullable=True),
        sa.Column("total_sessions", sa.Integer(), nullable=False, server_default="0"),
        *_evidence_columns(),
        sa.Column("overall_accuracy", sa.Float(), nullable=True),
        sa.Column("concept_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        *_evidence_constraints("long_term_memory"),
        sa.CheckConstraint("total_sessions >= 0", name="sessions_nonnegative"),
        sa.CheckConstraint("concept_count >= 0", name="concept_count_nonnegative"),
        sa.CheckConstraint(
            "overall_accuracy IS NULL OR (overall_accuracy >= 0 AND overall_accuracy <= 1)",
            name="accuracy_range",
        ),
        sa.CheckConstraint(
            "interaction_count > 0 OR overall_accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_long_term_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("student_id", name="pk_long_term_memory"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_long_term_memory_last_interaction",
        "long_term_memory",
        ["last_interaction_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_long_term_memory_updated",
        "long_term_memory",
        ["updated_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "concept_memory",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_external_id", sa.String(255), nullable=True),
        sa.Column(
            "canonical_skill_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("skill_name", sa.String(255), nullable=True),
        *_evidence_columns(),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("first_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        *_evidence_constraints("concept_memory"),
        sa.CheckConstraint(
            "accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)",
            name="accuracy_range",
        ),
        sa.CheckConstraint(
            "interaction_count > 0 OR accuracy IS NULL",
            name="zero_interactions_no_accuracy",
        ),
        sa.CheckConstraint(
            "first_interaction_at IS NULL OR last_interaction_at IS NULL OR "
            "first_interaction_at <= last_interaction_at",
            name="interaction_time_order",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_concept_memory_student_id_students",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_concept_memory_skill_id_canonical_skills",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "student_id", "canonical_skill_id", name="pk_concept_memory"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_concept_memory_skill_last_interaction",
        "concept_memory",
        ["canonical_skill_id", "last_interaction_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_concept_memory_student_last_interaction",
        "concept_memory",
        ["student_id", "last_interaction_at"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_concept_memory_student_last_interaction",
        table_name="concept_memory",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_concept_memory_skill_last_interaction",
        table_name="concept_memory",
        schema=SCHEMA,
    )
    op.drop_table("concept_memory", schema=SCHEMA)

    op.drop_index(
        "ix_long_term_memory_updated",
        table_name="long_term_memory",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_long_term_memory_last_interaction",
        table_name="long_term_memory",
        schema=SCHEMA,
    )
    op.drop_table("long_term_memory", schema=SCHEMA)

    op.drop_index(
        "ix_short_term_memory_updated",
        table_name="short_term_memory",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_short_term_memory_current_skill",
        table_name="short_term_memory",
        schema=SCHEMA,
    )
    op.drop_table("short_term_memory", schema=SCHEMA)

