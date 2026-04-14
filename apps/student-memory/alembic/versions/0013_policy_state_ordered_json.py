"""Preserve frozen policy mapping order in PostgreSQL JSON.

Revision ID: 0013_policy_state_ordered_json
Revises: 0012_postgres_unification

The frozen Turn-LinTS loader validates insertion order for arm-indexed
mappings. PostgreSQL JSONB intentionally normalizes object key order, whereas
JSON retains the serialized order supplied by the canonical state writer.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0013_policy_state_ordered_json"
down_revision: str | None = "0012_postgres_unification"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE research.policy_states "
            "ALTER COLUMN state_json TYPE json USING state_json::json"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE research.policy_states "
            "ALTER COLUMN state_json TYPE jsonb USING state_json::jsonb"
        )
    )
