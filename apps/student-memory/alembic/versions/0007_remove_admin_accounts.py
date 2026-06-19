"""Remove administrator accounts and restrict authentication to students."""

from alembic import op


revision: str = "0007_remove_admin_accounts"
down_revision: str | None = "0006_user_accounts"
branch_labels = None
depends_on = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.execute(
        f"DELETE FROM {SCHEMA}.user_accounts WHERE role = 'ADMIN'"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.user_accounts "
        "DROP CONSTRAINT IF EXISTS chk_user_accounts_role"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.user_accounts "
        "ADD CONSTRAINT chk_user_accounts_role CHECK (role = 'STUDENT')"
    )


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.user_accounts "
        "DROP CONSTRAINT IF EXISTS chk_user_accounts_role"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.user_accounts "
        "ADD CONSTRAINT chk_user_accounts_role "
        "CHECK (role IN ('STUDENT', 'ADMIN'))"
    )
