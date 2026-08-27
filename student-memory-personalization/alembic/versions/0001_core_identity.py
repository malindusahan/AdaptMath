"""Create the PostgreSQL schema and core identity/skill tables.

Revision ID: 0001_core_identity
Revises: None
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_core_identity"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(SCHEMA, if_not_exists=True))

    op.create_table(
        "students",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_student_id", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
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
        sa.PrimaryKeyConstraint("student_id", name="pk_students"),
        sa.UniqueConstraint(
            "external_student_id",
            name="uq_students_external_student_id",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "canonical_skills",
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_name", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parent_skill_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "ontology_version",
            sa.String(50),
            nullable=False,
            server_default="1.0",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
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
        sa.CheckConstraint(
            "length(trim(canonical_name)) > 0",
            name="name_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["parent_skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_canonical_skills_parent_skill_id_canonical_skills",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("skill_id", name="pk_canonical_skills"),
        sa.UniqueConstraint(
            "canonical_name",
            name="uq_canonical_skills_canonical_name",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_canonical_skills_parent_skill_id",
        "canonical_skills",
        ["parent_skill_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "learning_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_session_id", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'COMPLETED', 'ABANDONED')",
            name="status",
        ),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="time_order",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_learning_sessions_student_id_students",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_learning_sessions"),
        sa.UniqueConstraint(
            "student_id",
            "external_session_id",
            name="uq_learning_sessions_student_external_session",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_learning_sessions_student_id",
        "learning_sessions",
        ["student_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "skill_aliases",
        sa.Column("alias_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias", sa.String(255), nullable=False),
        sa.Column("normalized_alias", sa.String(255), nullable=False),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "length(trim(normalized_alias)) > 0",
            name="normalized_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            [f"{SCHEMA}.canonical_skills.skill_id"],
            name="fk_skill_aliases_skill_id_canonical_skills",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("alias_id", name="pk_skill_aliases"),
        sa.UniqueConstraint(
            "normalized_alias",
            name="uq_skill_aliases_normalized_alias",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_skill_aliases_skill_id",
        "skill_aliases",
        ["skill_id"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_skill_aliases_skill_id",
        table_name="skill_aliases",
        schema=SCHEMA,
    )
    op.drop_table("skill_aliases", schema=SCHEMA)

    op.drop_index(
        "ix_learning_sessions_student_id",
        table_name="learning_sessions",
        schema=SCHEMA,
    )
    op.drop_table("learning_sessions", schema=SCHEMA)

    op.drop_index(
        "ix_canonical_skills_parent_skill_id",
        table_name="canonical_skills",
        schema=SCHEMA,
    )
    op.drop_table("canonical_skills", schema=SCHEMA)
    op.drop_table("students", schema=SCHEMA)

    op.execute(sa.schema.DropSchema(SCHEMA))
