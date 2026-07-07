"""Add supporting-memory audit and session provenance fields."""
from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision="0005_support_audit"
down_revision="0004_supporting_memory"
branch_labels: str | Sequence[str] | None=None
depends_on: str | Sequence[str] | None=None
SCHEMA="student_memory"
def upgrade():
    op.execute("ALTER TABLE student_memory.student_misconceptions ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now()")
    op.execute("ALTER TABLE student_memory.student_misconceptions ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()")
    op.execute("ALTER TABLE student_memory.topic_extraction_logs ADD COLUMN IF NOT EXISTS session_id uuid")
    op.execute("ALTER TABLE student_memory.topic_extraction_logs ADD COLUMN IF NOT EXISTS model_version varchar(100)")
    op.execute("ALTER TABLE student_memory.topic_extraction_logs ADD COLUMN IF NOT EXISTS ontology_version varchar(100)")
    op.execute("ALTER TABLE student_memory.repair_outcomes ADD COLUMN IF NOT EXISTS session_id uuid")
    op.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_topic_extraction_logs_session_student_learning_sessions') THEN ALTER TABLE student_memory.topic_extraction_logs ADD CONSTRAINT fk_topic_extraction_logs_session_student_learning_sessions FOREIGN KEY (session_id,student_id) REFERENCES student_memory.learning_sessions(session_id,student_id) ON DELETE RESTRICT; END IF; END $$")
    op.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_repair_outcomes_session_student_learning_sessions') THEN ALTER TABLE student_memory.repair_outcomes ADD CONSTRAINT fk_repair_outcomes_session_student_learning_sessions FOREIGN KEY (session_id,student_id) REFERENCES student_memory.learning_sessions(session_id,student_id) ON DELETE RESTRICT; END IF; END $$")
    op.execute("ALTER TABLE student_memory.topic_extraction_logs ALTER COLUMN session_id SET NOT NULL")
    op.execute("ALTER TABLE student_memory.repair_outcomes ALTER COLUMN session_id SET NOT NULL")
def downgrade():
    op.execute("ALTER TABLE student_memory.repair_outcomes DROP CONSTRAINT IF EXISTS fk_repair_outcomes_session_student_learning_sessions")
    op.execute("ALTER TABLE student_memory.repair_outcomes DROP COLUMN IF EXISTS session_id")
    op.execute("ALTER TABLE student_memory.topic_extraction_logs DROP CONSTRAINT IF EXISTS fk_topic_extraction_logs_session_student_learning_sessions")
    for column in ("ontology_version","model_version","session_id"): op.execute(f"ALTER TABLE student_memory.topic_extraction_logs DROP COLUMN IF EXISTS {column}")
    op.execute("ALTER TABLE student_memory.student_misconceptions DROP COLUMN IF EXISTS updated_at")
    op.execute("ALTER TABLE student_memory.student_misconceptions DROP COLUMN IF EXISTS created_at")

