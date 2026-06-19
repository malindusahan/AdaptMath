"""Create user accounts table linked to student identity.

Revision ID: 0006_user_accounts
Revises: 0005_support_audit
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0006_user_accounts"
down_revision: str | None = "0005_support_audit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("username", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=True),
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
        sa.Column(
            "last_login_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_user_accounts"),
        sa.UniqueConstraint("username", name="uq_user_accounts_username"),
        sa.CheckConstraint(
            "role IN ('STUDENT', 'ADMIN')",
            name="chk_user_accounts_role",
        ),
        sa.CheckConstraint(
            "length(trim(username)) > 0",
            name="chk_user_accounts_username_not_blank",
        ),
        sa.CheckConstraint(
            "age IS NULL OR age >= 0",
            name="chk_user_accounts_age_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{SCHEMA}.students.student_id"],
            name="fk_user_accounts_student_id_students",
            ondelete="SET NULL",
        ),
        schema=SCHEMA,
    )

    op.create_index(
        "ix_user_accounts_username",
        "user_accounts",
        ["username"],
        unique=True,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_user_accounts_student_id",
        "user_accounts",
        ["student_id"],
        unique=False,
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_accounts_student_id",
        table_name="user_accounts",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_user_accounts_username",
        table_name="user_accounts",
        schema=SCHEMA,
    )
    op.drop_table("user_accounts", schema=SCHEMA)
