"""Add readable skill names to all skill-bearing memory tables."""

from collections.abc import Sequence

from alembic import op


revision: str = "0010_memory_skill_names"
down_revision: str | None = "0009_add_long_term_skill_names"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE {SCHEMA}.long_term_memory
            ADD COLUMN IF NOT EXISTS skill_names TEXT[];

        ALTER TABLE {SCHEMA}.interaction_logs
            ADD COLUMN IF NOT EXISTS skill_name VARCHAR(255);

        ALTER TABLE {SCHEMA}.topic_extraction_logs
            ADD COLUMN IF NOT EXISTS skill_name VARCHAR(255);

        ALTER TABLE {SCHEMA}.repair_outcomes
            ADD COLUMN IF NOT EXISTS skill_name VARCHAR(255);

        UPDATE {SCHEMA}.interaction_logs il
        SET skill_name = cs.display_name
        FROM {SCHEMA}.canonical_skills cs
        WHERE il.canonical_skill_id = cs.skill_id;

        UPDATE {SCHEMA}.topic_extraction_logs tel
        SET skill_name = cs.display_name
        FROM {SCHEMA}.canonical_skills cs
        WHERE tel.canonical_skill_id = cs.skill_id;

        UPDATE {SCHEMA}.repair_outcomes ro
        SET skill_name = cs.display_name
        FROM {SCHEMA}.canonical_skills cs
        WHERE ro.canonical_skill_id = cs.skill_id;

        UPDATE {SCHEMA}.long_term_memory ltm
        SET skill_names = skills.names
        FROM (
            SELECT cm.student_id, array_agg(DISTINCT cm.skill_name ORDER BY cm.skill_name) AS names
            FROM {SCHEMA}.concept_memory cm
            WHERE cm.skill_name IS NOT NULL
            GROUP BY cm.student_id
        ) skills
        WHERE ltm.student_id = skills.student_id;

        CREATE OR REPLACE FUNCTION {SCHEMA}.refresh_long_term_skill_names()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            UPDATE {SCHEMA}.long_term_memory ltm
            SET skill_names = COALESCE(
                (
                    SELECT array_agg(DISTINCT cm.skill_name ORDER BY cm.skill_name)
                    FROM {SCHEMA}.concept_memory cm
                    WHERE cm.student_id = COALESCE(NEW.student_id, OLD.student_id)
                      AND cm.skill_name IS NOT NULL
                ),
                ARRAY[]::TEXT[]
            )
            WHERE ltm.student_id = COALESCE(NEW.student_id, OLD.student_id);
            RETURN COALESCE(NEW, OLD);
        END;
        $$;

        DROP TRIGGER IF EXISTS trg_refresh_long_term_skill_names
            ON {SCHEMA}.concept_memory;
        CREATE TRIGGER trg_refresh_long_term_skill_names
            AFTER INSERT OR UPDATE OR DELETE ON {SCHEMA}.concept_memory
            FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.refresh_long_term_skill_names();

        CREATE OR REPLACE FUNCTION {SCHEMA}.populate_skill_name()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.canonical_skill_id IS NOT NULL THEN
                SELECT cs.display_name
                INTO NEW.skill_name
                FROM {SCHEMA}.canonical_skills cs
                WHERE cs.skill_id = NEW.canonical_skill_id;
            ELSE
                NEW.skill_name := NULL;
            END IF;
            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS trg_populate_interaction_skill_name
            ON {SCHEMA}.interaction_logs;
        CREATE TRIGGER trg_populate_interaction_skill_name
            BEFORE INSERT OR UPDATE OF canonical_skill_id ON {SCHEMA}.interaction_logs
            FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.populate_skill_name();

        DROP TRIGGER IF EXISTS trg_populate_topic_extraction_skill_name
            ON {SCHEMA}.topic_extraction_logs;
        CREATE TRIGGER trg_populate_topic_extraction_skill_name
            BEFORE INSERT OR UPDATE OF canonical_skill_id ON {SCHEMA}.topic_extraction_logs
            FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.populate_skill_name();

        DROP TRIGGER IF EXISTS trg_populate_repair_skill_name
            ON {SCHEMA}.repair_outcomes;
        CREATE TRIGGER trg_populate_repair_skill_name
            BEFORE INSERT OR UPDATE OF canonical_skill_id ON {SCHEMA}.repair_outcomes
            FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.populate_skill_name();

        DROP VIEW IF EXISTS {SCHEMA}.v_long_term_memory;
        CREATE VIEW {SCHEMA}.v_long_term_memory AS
        SELECT
            ltm.student_id,
            ltm.student_external_id,
            ltm.skill_names,
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
        DROP TRIGGER IF EXISTS trg_refresh_long_term_skill_names
            ON {SCHEMA}.concept_memory;
        DROP TRIGGER IF EXISTS trg_populate_interaction_skill_name
            ON {SCHEMA}.interaction_logs;
        DROP TRIGGER IF EXISTS trg_populate_topic_extraction_skill_name
            ON {SCHEMA}.topic_extraction_logs;
        DROP TRIGGER IF EXISTS trg_populate_repair_skill_name
            ON {SCHEMA}.repair_outcomes;
        DROP FUNCTION IF EXISTS {SCHEMA}.refresh_long_term_skill_names();
        DROP FUNCTION IF EXISTS {SCHEMA}.populate_skill_name();

        ALTER TABLE {SCHEMA}.long_term_memory DROP COLUMN IF EXISTS skill_names;
        ALTER TABLE {SCHEMA}.interaction_logs DROP COLUMN IF EXISTS skill_name;
        ALTER TABLE {SCHEMA}.topic_extraction_logs DROP COLUMN IF EXISTS skill_name;
        ALTER TABLE {SCHEMA}.repair_outcomes DROP COLUMN IF EXISTS skill_name;

        -- Restore the 0009 view, whose skill_names value is computed from
        -- concept_memory rather than stored on long_term_memory.
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
