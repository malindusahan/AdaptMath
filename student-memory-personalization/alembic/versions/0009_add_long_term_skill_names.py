"""Add aggregated skill names to the long-term memory presentation view."""

from collections.abc import Sequence

from alembic import op


revision: str = "0009_add_long_term_skill_names"
down_revision: str | None = "0008_add_names_to_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.execute(
        f"""
        DROP VIEW IF EXISTS {SCHEMA}.v_long_term_memory;

        CREATE VIEW {SCHEMA}.v_long_term_memory AS
        SELECT
            ltm.student_id,
            ltm.student_external_id,
            ARRAY(
                SELECT DISTINCT cm.skill_name
                FROM {SCHEMA}.concept_memory cm
                WHERE cm.student_id = ltm.student_id
                  AND cm.skill_name IS NOT NULL
                ORDER BY cm.skill_name
            ) AS skill_names,
            ltm.overall_accuracy,
            ltm.total_sessions,
            ltm.concept_count,
            ltm.interaction_count,
            ltm.correct_count,
            ltm.incorrect_count,
            ltm.last_interaction_at,
            ltm.updated_at
        FROM {SCHEMA}.long_term_memory ltm;
        """
    )


def downgrade() -> None:
    op.execute(
        f"""
        DROP VIEW IF EXISTS {SCHEMA}.v_long_term_memory;

        CREATE VIEW {SCHEMA}.v_long_term_memory AS
        SELECT
            ltm.student_id,
            ltm.student_external_id,
            ltm.overall_accuracy,
            ltm.total_sessions,
            ltm.concept_count,
            ltm.interaction_count,
            ltm.correct_count,
            ltm.incorrect_count,
            ltm.last_interaction_at,
            ltm.updated_at
        FROM {SCHEMA}.long_term_memory ltm;
        """
    )
