"""Recreate memory tables with student_external_id and skill_name near IDs and add presentation views.

Revision ID: 0008_add_names_to_memory
Revises: 0007_remove_admin_accounts
"""

from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa


revision: str = "0008_add_names_to_memory"
down_revision: str | None = "0007_remove_admin_accounts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "student_memory"


def upgrade() -> None:
    # 1. Concept Memory Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.concept_memory_new (
            student_id UUID NOT NULL REFERENCES {SCHEMA}.students(student_id) ON DELETE RESTRICT,
            student_external_id VARCHAR(255),
            canonical_skill_id UUID NOT NULL REFERENCES {SCHEMA}.canonical_skills(skill_id) ON DELETE RESTRICT,
            skill_name VARCHAR(255),
            interaction_count INTEGER DEFAULT 0 NOT NULL CHECK (interaction_count >= 0),
            correct_count INTEGER DEFAULT 0 NOT NULL CHECK (correct_count >= 0),
            incorrect_count INTEGER DEFAULT 0 NOT NULL CHECK (incorrect_count >= 0),
            accuracy DOUBLE PRECISION CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)),
            attempt_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (attempt_observation_count >= 0),
            attempt_sum INTEGER DEFAULT 0 NOT NULL CHECK (attempt_sum >= 0),
            hint_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (hint_observation_count >= 0),
            hint_sum INTEGER DEFAULT 0 NOT NULL CHECK (hint_sum >= 0),
            response_time_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (response_time_observation_count >= 0),
            response_time_sum_ms DOUBLE PRECISION DEFAULT 0 NOT NULL CHECK (response_time_sum_ms >= 0),
            first_interaction_at TIMESTAMPTZ,
            last_interaction_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            PRIMARY KEY (student_id, canonical_skill_id),
            CHECK (correct_count + incorrect_count <= interaction_count),
            CHECK (attempt_observation_count > 0 OR attempt_sum = 0),
            CHECK (hint_observation_count > 0 OR hint_sum = 0),
            CHECK (response_time_observation_count > 0 OR response_time_sum_ms = 0),
            CHECK (interaction_count > 0 OR accuracy IS NULL),
            CHECK (first_interaction_at IS NULL OR last_interaction_at IS NULL OR first_interaction_at <= last_interaction_at)
        );

        INSERT INTO {SCHEMA}.concept_memory_new (
            student_id, student_external_id, canonical_skill_id, skill_name,
            interaction_count, correct_count, incorrect_count, accuracy,
            attempt_observation_count, attempt_sum, hint_observation_count, hint_sum,
            response_time_observation_count, response_time_sum_ms,
            first_interaction_at, last_interaction_at, updated_at
        )
        SELECT
            cm.student_id, s.external_student_id, cm.canonical_skill_id, cs.display_name,
            cm.interaction_count, cm.correct_count, cm.incorrect_count, cm.accuracy,
            cm.attempt_observation_count, cm.attempt_sum, cm.hint_observation_count, cm.hint_sum,
            cm.response_time_observation_count, cm.response_time_sum_ms,
            cm.first_interaction_at, cm.last_interaction_at, cm.updated_at
        FROM {SCHEMA}.concept_memory cm
        LEFT JOIN {SCHEMA}.students s ON cm.student_id = s.student_id
        LEFT JOIN {SCHEMA}.canonical_skills cs ON cm.canonical_skill_id = cs.skill_id;

        DROP TABLE {SCHEMA}.concept_memory CASCADE;
        ALTER TABLE {SCHEMA}.concept_memory_new RENAME TO concept_memory;

        CREATE INDEX ix_concept_memory_skill_last_interaction ON {SCHEMA}.concept_memory (canonical_skill_id, last_interaction_at);
        CREATE INDEX ix_concept_memory_student_last_interaction ON {SCHEMA}.concept_memory (student_id, last_interaction_at);
        """
    )

    # 2. Short-Term Memory Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.short_term_memory_new (
            student_id UUID NOT NULL REFERENCES {SCHEMA}.students(student_id) ON DELETE RESTRICT,
            student_external_id VARCHAR(255),
            session_id UUID NOT NULL,
            current_skill_id UUID REFERENCES {SCHEMA}.canonical_skills(skill_id) ON DELETE RESTRICT,
            skill_name VARCHAR(255),
            interaction_count INTEGER DEFAULT 0 NOT NULL CHECK (interaction_count >= 0),
            correct_count INTEGER DEFAULT 0 NOT NULL CHECK (correct_count >= 0),
            incorrect_count INTEGER DEFAULT 0 NOT NULL CHECK (incorrect_count >= 0),
            recent_accuracy DOUBLE PRECISION CHECK (recent_accuracy IS NULL OR (recent_accuracy >= 0 AND recent_accuracy <= 1)),
            attempt_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (attempt_observation_count >= 0),
            attempt_sum INTEGER DEFAULT 0 NOT NULL CHECK (attempt_sum >= 0),
            hint_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (hint_observation_count >= 0),
            hint_sum INTEGER DEFAULT 0 NOT NULL CHECK (hint_sum >= 0),
            response_time_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (response_time_observation_count >= 0),
            response_time_sum_ms DOUBLE PRECISION DEFAULT 0 NOT NULL CHECK (response_time_sum_ms >= 0),
            last_interaction_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            PRIMARY KEY (student_id, session_id),
            FOREIGN KEY (session_id, student_id) REFERENCES {SCHEMA}.learning_sessions(session_id, student_id) ON DELETE RESTRICT,
            CHECK (correct_count + incorrect_count <= interaction_count),
            CHECK (attempt_observation_count > 0 OR attempt_sum = 0),
            CHECK (hint_observation_count > 0 OR hint_sum = 0),
            CHECK (response_time_observation_count > 0 OR response_time_sum_ms = 0),
            CHECK (interaction_count > 0 OR recent_accuracy IS NULL)
        );

        INSERT INTO {SCHEMA}.short_term_memory_new (
            student_id, student_external_id, session_id, current_skill_id, skill_name,
            interaction_count, correct_count, incorrect_count, recent_accuracy,
            attempt_observation_count, attempt_sum, hint_observation_count, hint_sum,
            response_time_observation_count, response_time_sum_ms,
            last_interaction_at, updated_at
        )
        SELECT
            stm.student_id, s.external_student_id, stm.session_id, stm.current_skill_id, cs.display_name,
            stm.interaction_count, stm.correct_count, stm.incorrect_count, stm.recent_accuracy,
            stm.attempt_observation_count, stm.attempt_sum, stm.hint_observation_count, stm.hint_sum,
            stm.response_time_observation_count, stm.response_time_sum_ms,
            stm.last_interaction_at, stm.updated_at
        FROM {SCHEMA}.short_term_memory stm
        LEFT JOIN {SCHEMA}.students s ON stm.student_id = s.student_id
        LEFT JOIN {SCHEMA}.canonical_skills cs ON stm.current_skill_id = cs.skill_id;

        DROP TABLE {SCHEMA}.short_term_memory CASCADE;
        ALTER TABLE {SCHEMA}.short_term_memory_new RENAME TO short_term_memory;

        CREATE INDEX ix_short_term_memory_current_skill ON {SCHEMA}.short_term_memory (current_skill_id);
        CREATE INDEX ix_short_term_memory_updated ON {SCHEMA}.short_term_memory (updated_at);
        """
    )

    # 3. Long-Term Memory Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.long_term_memory_new (
            student_id UUID NOT NULL PRIMARY KEY REFERENCES {SCHEMA}.students(student_id) ON DELETE RESTRICT,
            student_external_id VARCHAR(255),
            total_sessions INTEGER DEFAULT 0 NOT NULL CHECK (total_sessions >= 0),
            concept_count INTEGER DEFAULT 0 NOT NULL CHECK (concept_count >= 0),
            interaction_count INTEGER DEFAULT 0 NOT NULL CHECK (interaction_count >= 0),
            correct_count INTEGER DEFAULT 0 NOT NULL CHECK (correct_count >= 0),
            incorrect_count INTEGER DEFAULT 0 NOT NULL CHECK (incorrect_count >= 0),
            overall_accuracy DOUBLE PRECISION CHECK (overall_accuracy IS NULL OR (overall_accuracy >= 0 AND overall_accuracy <= 1)),
            attempt_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (attempt_observation_count >= 0),
            attempt_sum INTEGER DEFAULT 0 NOT NULL CHECK (attempt_sum >= 0),
            hint_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (hint_observation_count >= 0),
            hint_sum INTEGER DEFAULT 0 NOT NULL CHECK (hint_sum >= 0),
            response_time_observation_count INTEGER DEFAULT 0 NOT NULL CHECK (response_time_observation_count >= 0),
            response_time_sum_ms DOUBLE PRECISION DEFAULT 0 NOT NULL CHECK (response_time_sum_ms >= 0),
            last_interaction_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            CHECK (correct_count + incorrect_count <= interaction_count),
            CHECK (attempt_observation_count > 0 OR attempt_sum = 0),
            CHECK (hint_observation_count > 0 OR hint_sum = 0),
            CHECK (response_time_observation_count > 0 OR response_time_sum_ms = 0),
            CHECK (interaction_count > 0 OR overall_accuracy IS NULL)
        );

        INSERT INTO {SCHEMA}.long_term_memory_new (
            student_id, student_external_id, total_sessions, concept_count,
            interaction_count, correct_count, incorrect_count, overall_accuracy,
            attempt_observation_count, attempt_sum, hint_observation_count, hint_sum,
            response_time_observation_count, response_time_sum_ms,
            last_interaction_at, updated_at
        )
        SELECT
            ltm.student_id, s.external_student_id, ltm.total_sessions, ltm.concept_count,
            ltm.interaction_count, ltm.correct_count, ltm.incorrect_count, ltm.overall_accuracy,
            ltm.attempt_observation_count, ltm.attempt_sum, ltm.hint_observation_count, ltm.hint_sum,
            ltm.response_time_observation_count, ltm.response_time_sum_ms,
            ltm.last_interaction_at, ltm.updated_at
        FROM {SCHEMA}.long_term_memory ltm
        LEFT JOIN {SCHEMA}.students s ON ltm.student_id = s.student_id;

        DROP TABLE {SCHEMA}.long_term_memory CASCADE;
        ALTER TABLE {SCHEMA}.long_term_memory_new RENAME TO long_term_memory;

        CREATE INDEX ix_long_term_memory_last_interaction ON {SCHEMA}.long_term_memory (last_interaction_at);
        CREATE INDEX ix_long_term_memory_updated ON {SCHEMA}.long_term_memory (updated_at);
        """
    )

    # 4. Learning State Snapshots Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.learning_state_snapshots_new (
            snapshot_id BIGSERIAL PRIMARY KEY,
            student_id UUID NOT NULL REFERENCES {SCHEMA}.students(student_id) ON DELETE CASCADE,
            student_external_id VARCHAR(255),
            session_id UUID NOT NULL,
            canonical_skill_id UUID NOT NULL REFERENCES {SCHEMA}.canonical_skills(skill_id) ON DELETE RESTRICT,
            skill_name VARCHAR(255),
            assessment_id BIGINT,
            learning_state VARCHAR(30) NOT NULL CHECK (learning_state IN ('NEEDS_SUPPORT','DEVELOPING','STRONG','UNAVAILABLE')),
            evidence_level VARCHAR(30) NOT NULL CHECK (evidence_level IN ('COLD_START','OVERALL_ONLY','PARTIAL_SKILL','FULL_SKILL')),
            evidence_strength VARCHAR(20) NOT NULL CHECK (evidence_strength IN ('NONE','LOW','MEDIUM','HIGH')),
            behavioural_coverage VARCHAR(40) NOT NULL CHECK (behavioural_coverage IN ('FULL_BEHAVIOURAL_COVERAGE','PARTIAL_BEHAVIOURAL_COVERAGE','CORRECTNESS_ONLY_COVERAGE')),
            model_used BOOLEAN NOT NULL,
            previous_interaction_count INTEGER NOT NULL,
            previous_skill_interaction_count INTEGER NOT NULL,
            recent_interaction_count INTEGER NOT NULL,
            attempt_observation_count INTEGER NOT NULL,
            hint_observation_count INTEGER NOT NULL,
            response_time_observation_count INTEGER NOT NULL,
            model_version VARCHAR(100),
            created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            FOREIGN KEY (session_id, student_id) REFERENCES {SCHEMA}.learning_sessions(session_id, student_id) ON DELETE RESTRICT,
            CHECK (previous_interaction_count >= 0 AND previous_skill_interaction_count >= 0)
        );

        INSERT INTO {SCHEMA}.learning_state_snapshots_new (
            snapshot_id, student_id, student_external_id, session_id, canonical_skill_id, skill_name,
            assessment_id, learning_state, evidence_level, evidence_strength, behavioural_coverage,
            model_used, previous_interaction_count, previous_skill_interaction_count,
            recent_interaction_count, attempt_observation_count, hint_observation_count,
            response_time_observation_count, model_version, created_at
        )
        SELECT
            lss.snapshot_id, lss.student_id, s.external_student_id, lss.session_id, lss.canonical_skill_id, cs.display_name,
            lss.assessment_id, lss.learning_state, lss.evidence_level, lss.evidence_strength, lss.behavioural_coverage,
            lss.model_used, lss.previous_interaction_count, lss.previous_skill_interaction_count,
            lss.recent_interaction_count, lss.attempt_observation_count, lss.hint_observation_count,
            lss.response_time_observation_count, lss.model_version, lss.created_at
        FROM {SCHEMA}.learning_state_snapshots lss
        LEFT JOIN {SCHEMA}.students s ON lss.student_id = s.student_id
        LEFT JOIN {SCHEMA}.canonical_skills cs ON lss.canonical_skill_id = cs.skill_id;

        DROP TABLE {SCHEMA}.current_learning_state CASCADE;
        DROP TABLE {SCHEMA}.learning_state_snapshots CASCADE;
        ALTER TABLE {SCHEMA}.learning_state_snapshots_new RENAME TO learning_state_snapshots;

        SELECT setval(pg_get_serial_sequence('{SCHEMA}.learning_state_snapshots', 'snapshot_id'), coalesce(max(snapshot_id), 1)) FROM {SCHEMA}.learning_state_snapshots;
        """
    )

    # 5. Current Learning State Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.current_learning_state (
            student_id UUID NOT NULL REFERENCES {SCHEMA}.students(student_id) ON DELETE CASCADE,
            student_external_id VARCHAR(255),
            canonical_skill_id UUID NOT NULL REFERENCES {SCHEMA}.canonical_skills(skill_id) ON DELETE RESTRICT,
            skill_name VARCHAR(255),
            learning_state VARCHAR(30) NOT NULL CHECK (learning_state IN ('NEEDS_SUPPORT','DEVELOPING','STRONG','UNAVAILABLE')),
            evidence_level VARCHAR(30) NOT NULL CHECK (evidence_level IN ('COLD_START','OVERALL_ONLY','PARTIAL_SKILL','FULL_SKILL')),
            evidence_strength VARCHAR(20) NOT NULL CHECK (evidence_strength IN ('NONE','LOW','MEDIUM','HIGH')),
            behavioural_coverage VARCHAR(40) NOT NULL CHECK (behavioural_coverage IN ('FULL_BEHAVIOURAL_COVERAGE','PARTIAL_BEHAVIOURAL_COVERAGE','CORRECTNESS_ONLY_COVERAGE')),
            model_used BOOLEAN NOT NULL,
            recent_interaction_count INTEGER NOT NULL,
            attempt_observation_count INTEGER NOT NULL,
            hint_observation_count INTEGER NOT NULL,
            response_time_observation_count INTEGER NOT NULL,
            last_assessment_id BIGINT,
            last_snapshot_id BIGINT NOT NULL REFERENCES {SCHEMA}.learning_state_snapshots(snapshot_id) ON DELETE RESTRICT,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            PRIMARY KEY (student_id, canonical_skill_id)
        );
        """
    )

    # 6. Student Misconceptions Table Recreation
    op.execute(
        f"""
        CREATE TABLE {SCHEMA}.student_misconceptions_new (
            misconception_id UUID PRIMARY KEY,
            student_id UUID NOT NULL REFERENCES {SCHEMA}.students(student_id) ON DELETE CASCADE,
            student_external_id VARCHAR(255),
            canonical_skill_id UUID NOT NULL REFERENCES {SCHEMA}.canonical_skills(skill_id) ON DELETE CASCADE,
            skill_name VARCHAR(255),
            normalized_error VARCHAR(500) NOT NULL,
            display_error TEXT NOT NULL,
            occurrence_count INTEGER DEFAULT 1 NOT NULL CHECK (occurrence_count >= 1),
            first_seen_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            last_seen_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            UNIQUE (student_id, canonical_skill_id, normalized_error)
        );

        INSERT INTO {SCHEMA}.student_misconceptions_new (
            misconception_id, student_id, student_external_id, canonical_skill_id, skill_name,
            normalized_error, display_error, occurrence_count, first_seen_at, last_seen_at,
            created_at, updated_at
        )
        SELECT
            sm.misconception_id, sm.student_id, s.external_student_id, sm.canonical_skill_id, cs.display_name,
            sm.normalized_error, sm.display_error, sm.occurrence_count, sm.first_seen_at, sm.last_seen_at,
            sm.created_at, sm.updated_at
        FROM {SCHEMA}.student_misconceptions sm
        LEFT JOIN {SCHEMA}.students s ON sm.student_id = s.student_id
        LEFT JOIN {SCHEMA}.canonical_skills cs ON sm.canonical_skill_id = cs.skill_id;

        DROP TABLE {SCHEMA}.student_misconceptions CASCADE;
        ALTER TABLE {SCHEMA}.student_misconceptions_new RENAME TO student_misconceptions;
        """
    )

    # 7. Create SQL Presentation Views
    op.execute(
        f"""
        CREATE OR REPLACE VIEW {SCHEMA}.v_concept_memory AS
        SELECT
            cm.student_id,
            cm.student_external_id,
            cm.canonical_skill_id,
            cm.skill_name,
            cm.accuracy,
            cm.interaction_count,
            cm.correct_count,
            cm.incorrect_count,
            cm.attempt_sum,
            cm.hint_sum,
            cm.first_interaction_at,
            cm.last_interaction_at,
            cm.updated_at
        FROM {SCHEMA}.concept_memory cm;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE VIEW {SCHEMA}.v_short_term_memory AS
        SELECT
            stm.student_id,
            stm.student_external_id,
            stm.session_id,
            stm.current_skill_id,
            stm.skill_name,
            stm.recent_accuracy,
            stm.interaction_count,
            stm.correct_count,
            stm.incorrect_count,
            stm.attempt_sum,
            stm.hint_sum,
            stm.last_interaction_at,
            stm.updated_at
        FROM {SCHEMA}.short_term_memory stm;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE VIEW {SCHEMA}.v_long_term_memory AS
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

    op.execute(
        f"""
        CREATE OR REPLACE VIEW {SCHEMA}.v_current_learning_state AS
        SELECT
            cls.student_id,
            cls.student_external_id,
            cls.canonical_skill_id,
            cls.skill_name,
            cls.learning_state,
            cls.evidence_level,
            cls.evidence_strength,
            cls.behavioural_coverage,
            cls.recent_interaction_count,
            cls.updated_at
        FROM {SCHEMA}.current_learning_state cls;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE VIEW {SCHEMA}.v_student_misconceptions AS
        SELECT
            sm.misconception_id,
            sm.student_id,
            sm.student_external_id,
            sm.canonical_skill_id,
            sm.skill_name,
            sm.display_error,
            sm.occurrence_count,
            sm.first_seen_at,
            sm.last_seen_at
        FROM {SCHEMA}.student_misconceptions sm;
        """
    )


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {SCHEMA}.v_student_misconceptions;")
    op.execute(f"DROP VIEW IF EXISTS {SCHEMA}.v_current_learning_state;")
    op.execute(f"DROP VIEW IF EXISTS {SCHEMA}.v_long_term_memory;")
    op.execute(f"DROP VIEW IF EXISTS {SCHEMA}.v_short_term_memory;")
    op.execute(f"DROP VIEW IF EXISTS {SCHEMA}.v_concept_memory;")
