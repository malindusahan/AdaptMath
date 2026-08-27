"""Create supporting memory and audit tables.

Revision ID: 0004_supporting_memory
Revises: 0003_memory_projections
"""
from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa
from src.database.models.supporting_memory import CurrentLearningState, LearningStateSnapshot, RepairOutcome, StudentMisconception, TopicExtractionLog

revision: str = "0004_supporting_memory"
down_revision: str | None = "0003_memory_projections"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
SCHEMA = "student_memory"

def upgrade() -> None:
    op.execute(sa.text(f"SET search_path TO {SCHEMA}, public"))
    op.execute(sa.text(f"CREATE SEQUENCE IF NOT EXISTS {SCHEMA}.assessment_id_seq"))
    bind = op.get_bind()
    for table in (StudentMisconception.__table__, LearningStateSnapshot.__table__, CurrentLearningState.__table__, TopicExtractionLog.__table__, RepairOutcome.__table__):
        table.create(bind=bind, checkfirst=True)

def downgrade() -> None:
    for table_name in ("repair_outcomes", "topic_extraction_logs", "current_learning_state", "learning_state_snapshots", "student_misconceptions"):
        op.execute(sa.text(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name} CASCADE"))
    op.execute(sa.text(f"DROP SEQUENCE IF EXISTS {SCHEMA}.assessment_id_seq CASCADE"))


