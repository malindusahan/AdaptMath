"""Provision schema-owned persistence for the AdaptMath services.

Revision ID: 0012_postgres_unification
Revises: 0011_adaptmath_ingestion

This is intentionally a storage-only migration.  The LangGraph checkpoint
tables below reproduce the official ``langgraph-checkpoint-postgres`` 3.1.2
schema at migration version 9; no Tutor state format is defined here.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0012_postgres_unification"
down_revision: str | None = "0011_adaptmath_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

AUTH = "auth"
MEMORY = "student_memory"
TUTOR = "tutor"
STUDENT_MODEL = "student_model"
RESEARCH = "research"


def _create_schemas() -> None:
    for schema in (AUTH, TUTOR, STUDENT_MODEL, RESEARCH):
        op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def _create_auth() -> None:
    op.create_table(
        "sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("session_id", name="pk_sessions"),
        sa.UniqueConstraint("token_hash", name="uq_sessions_token_hash"),
        schema=AUTH,
    )
    op.create_index(
        "ix_sessions_user_active",
        "sessions",
        ["user_id", "expires_at"],
        schema=AUTH,
    )
    op.create_index(
        "ix_sessions_expiry",
        "sessions",
        ["expires_at"],
        schema=AUTH,
    )
    # Phase A ownership bridge.  Accounts stay physically where their current
    # student FK is known-good; callers can use the auth-owned name without a
    # destructive table move or password rewrite.
    op.execute(
        sa.text(
            f"CREATE VIEW {AUTH}.user_accounts AS "
            f"SELECT * FROM {MEMORY}.user_accounts"
        )
    )


def _create_tutor() -> None:
    op.create_table(
        "checkpoint_migrations",
        sa.Column("v", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("v", name="pk_checkpoint_migrations"),
        schema=TUTOR,
    )
    op.create_table(
        "checkpoints",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.Text(), nullable=False),
        sa.Column("parent_checkpoint_id", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("checkpoint", postgresql.JSONB(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.PrimaryKeyConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            name="pk_checkpoints",
        ),
        schema=TUTOR,
    )
    op.create_table(
        "checkpoint_blobs",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False, server_default=""),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("blob", sa.LargeBinary(), nullable=True),
        sa.PrimaryKeyConstraint(
            "thread_id",
            "checkpoint_ns",
            "channel",
            "version",
            name="pk_checkpoint_blobs",
        ),
        schema=TUTOR,
    )
    op.create_table(
        "checkpoint_writes",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("task_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("blob", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name="pk_checkpoint_writes",
        ),
        schema=TUTOR,
    )
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        op.create_index(
            f"{table}_thread_id_idx",
            table,
            ["thread_id"],
            schema=TUTOR,
        )
    op.execute(
        sa.text(
            f"INSERT INTO {TUTOR}.checkpoint_migrations(v) "
            "SELECT generate_series(0, 9) ON CONFLICT DO NOTHING"
        )
    )
    op.create_table(
        "threads",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column("target_skill", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="ACTIVE"),
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
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'COMPLETED', 'ABORTED')",
            name="status_domain",
        ),
        sa.PrimaryKeyConstraint("thread_id", name="pk_threads"),
        schema=TUTOR,
    )
    op.create_index(
        "ix_threads_student_updated",
        "threads",
        ["student_id", "updated_at"],
        schema=TUTOR,
    )


def _create_student_model() -> None:
    op.create_table(
        "students",
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("profile_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("student_id", name="pk_students"),
        schema=STUDENT_MODEL,
    )
    op.create_table(
        "mastery",
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column("skill_name", sa.Text(), nullable=False),
        sa.Column("mastery_probability", sa.Float(), nullable=False),
        sa.Column("mastery_label", sa.Text(), nullable=False),
        sa.Column("previous_mastery_probability", sa.Float(), nullable=True),
        sa.Column(
            "last_updated",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{STUDENT_MODEL}.students.student_id"],
            name="fk_mastery_student",
        ),
        sa.PrimaryKeyConstraint("student_id", "skill_name", name="pk_mastery"),
        schema=STUDENT_MODEL,
    )
    op.create_table(
        "bkt_initial_priors",
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column("skill_name", sa.Text(), nullable=False),
        sa.Column("effective_initial_prior", sa.Float(), nullable=True),
        sa.Column("prior_source", sa.Text(), nullable=False),
        sa.Column(
            "selected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "prior_source IN ('population', 'transfer', "
            "'historical_population_rebase', 'legacy_unknown')",
            name="prior_source_domain",
        ),
        sa.CheckConstraint(
            "(prior_source = 'legacy_unknown' AND effective_initial_prior IS NULL) "
            "OR (prior_source <> 'legacy_unknown' "
            "AND effective_initial_prior BETWEEN 0.0 AND 1.0)",
            name="prior_value_domain",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{STUDENT_MODEL}.students.student_id"],
            name="fk_bkt_initial_priors_student",
        ),
        sa.PrimaryKeyConstraint(
            "student_id", "skill_name", name="pk_bkt_initial_priors"
        ),
        schema=STUDENT_MODEL,
    )
    op.create_table(
        "attempts",
        sa.Column(
            "attempt_id",
            sa.BigInteger(),
            sa.Identity(always=False),
            nullable=False,
        ),
        sa.Column("resolver_event_id", sa.Text(), nullable=True),
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column("skill_name", sa.Text(), nullable=False),
        sa.Column("correct", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("signal_type", sa.Text(), nullable=True),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("correct IN (0, 1)", name="correct_domain"),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{STUDENT_MODEL}.students.student_id"],
            name="fk_attempts_student",
        ),
        sa.PrimaryKeyConstraint("attempt_id", name="pk_attempts"),
        sa.UniqueConstraint("resolver_event_id", name="uq_attempts_resolver_event_id"),
        schema=STUDENT_MODEL,
    )
    op.create_table(
        "resolved_events",
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("source_action_event_id", sa.Text(), nullable=True),
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column("skill_name", sa.Text(), nullable=False),
        sa.Column("should_update", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Integer(), nullable=True),
        sa.Column("update_confidence", sa.Float(), nullable=False),
        sa.Column("observation_source", sa.Text(), nullable=False),
        sa.Column("primary_signal", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("mastery_before", sa.Float(), nullable=True),
        sa.Column("mastery_after", sa.Float(), nullable=True),
        sa.Column("delta_mastery", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("should_update IN (0, 1)", name="should_update_domain"),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN (0, 1)", name="outcome_domain"
        ),
        sa.CheckConstraint(
            "update_confidence BETWEEN 0.0 AND 1.0",
            name="update_confidence_domain",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{STUDENT_MODEL}.students.student_id"],
            name="fk_resolved_events_student",
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_resolved_events"),
        schema=STUDENT_MODEL,
    )
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("student_id", sa.Text(), nullable=False),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("concept_count", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["student_id"],
            [f"{STUDENT_MODEL}.students.student_id"],
            name="fk_sessions_student",
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_sessions"),
        schema=STUDENT_MODEL,
    )
    op.create_index(
        "ix_attempts_student_skill_created",
        "attempts",
        ["student_id", "skill_name", "created_at", "attempt_id"],
        schema=STUDENT_MODEL,
    )
    op.create_index(
        "ix_resolved_events_source_action",
        "resolved_events",
        ["source_action_event_id"],
        schema=STUDENT_MODEL,
    )
    op.create_index(
        "ix_mastery_student",
        "mastery",
        ["student_id"],
        schema=STUDENT_MODEL,
    )


def _create_research() -> None:
    op.create_table(
        "policy_states",
        sa.Column("policy_key", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("policy_class", sa.Text(), nullable=False),
        sa.Column("data_mode", sa.Text(), nullable=True),
        sa.Column("context_dimension", sa.Integer(), nullable=True),
        sa.Column("total_updates", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("state_sha256", sa.String(64), nullable=False),
        sa.Column("state_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("policy_key", name="pk_policy_states"),
        schema=RESEARCH,
    )
    op.create_table(
        "turn_actions",
        sa.Column("action_event_id", sa.Text(), nullable=False),
        sa.Column("attempt_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=True),
        sa.Column("student_id", sa.Text(), nullable=True),
        sa.Column("target_skill", sa.Text(), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="PENDING"),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'COMPLETED')", name="status_domain"
        ),
        sa.PrimaryKeyConstraint("action_event_id", name="pk_turn_actions"),
        schema=RESEARCH,
    )
    op.create_table(
        "selector_decisions",
        sa.Column("action_event_id", sa.Text(), nullable=False),
        sa.Column("attempt_id", sa.Text(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("selector_version", sa.Text(), nullable=True),
        sa.Column("p_generic", sa.Float(), nullable=True),
        sa.Column("p_probing", sa.Float(), nullable=True),
        sa.Column("p_focus", sa.Float(), nullable=True),
        sa.Column("p_telling", sa.Float(), nullable=True),
        sa.Column("selector_argmax", sa.Text(), nullable=True),
        sa.Column("selected_arm", sa.Text(), nullable=True),
        sa.Column("actual_move", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("action_event_id", name="pk_selector_decisions"),
        schema=RESEARCH,
    )
    op.create_table(
        "policy_contexts",
        sa.Column("action_event_id", sa.Text(), nullable=False),
        sa.Column("context_schema_version", sa.Text(), nullable=False),
        sa.Column("context_schema_id", sa.Text(), nullable=False),
        sa.Column("context_dimension", sa.Integer(), nullable=False),
        sa.Column("feature_names", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("context_vector", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("context_payload", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint(
            "cardinality(context_vector) = context_dimension",
            name="vector_matches_dimension",
        ),
        sa.ForeignKeyConstraint(
            ["action_event_id"],
            [f"{RESEARCH}.selector_decisions.action_event_id"],
            name="fk_policy_contexts_selector_decision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("action_event_id", name="pk_policy_contexts"),
        schema=RESEARCH,
    )
    op.create_table(
        "c3_contexts",
        sa.Column("action_event_id", sa.Text(), nullable=False),
        sa.Column("context_schema_version", sa.Text(), nullable=False),
        sa.Column("context_dimension", sa.Integer(), nullable=False),
        sa.Column("md6_p_generic", sa.Float(), nullable=False),
        sa.Column("md6_p_probing", sa.Float(), nullable=False),
        sa.Column("md6_p_focus", sa.Float(), nullable=False),
        sa.Column("md6_p_telling", sa.Float(), nullable=False),
        sa.Column("mrb1_mistake_identification", sa.Float(), nullable=False),
        sa.Column("mrb1_mistake_location", sa.Float(), nullable=False),
        sa.Column("mrb1_providing_guidance", sa.Float(), nullable=False),
        sa.Column("mrb1_actionability", sa.Float(), nullable=False),
        sa.Column("has_within_attempt_quality", sa.Float(), nullable=False),
        sa.Column("context_vector", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.CheckConstraint("context_dimension = 9", name="dimension_is_nine"),
        sa.ForeignKeyConstraint(
            ["action_event_id"],
            [f"{RESEARCH}.selector_decisions.action_event_id"],
            name="fk_c3_contexts_selector_decision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("action_event_id", name="pk_c3_contexts"),
        schema=RESEARCH,
    )
    op.create_table(
        "mrb1_scores",
        sa.Column("action_event_id", sa.Text(), nullable=False),
        sa.Column("attempt_id", sa.Text(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("mistake_identification", sa.Float(), nullable=False),
        sa.Column("mistake_location", sa.Float(), nullable=False),
        sa.Column("providing_guidance", sa.Float(), nullable=False),
        sa.Column("actionability", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["action_event_id"],
            [f"{RESEARCH}.selector_decisions.action_event_id"],
            name="fk_mrb1_scores_selector_decision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("action_event_id", name="pk_mrb1_scores"),
        schema=RESEARCH,
    )
    op.create_table(
        "policy_updates",
        sa.Column("update_id", sa.Text(), nullable=False),
        sa.Column("policy_key", sa.Text(), nullable=False),
        sa.Column("action_event_id", sa.Text(), nullable=True),
        sa.Column("attempt_id", sa.Text(), nullable=True),
        sa.Column("selected_arm", sa.Text(), nullable=True),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("sample_weight", sa.Float(), nullable=True),
        sa.Column("total_updates", sa.BigInteger(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("update_id", name="pk_policy_updates"),
        schema=RESEARCH,
    )
    op.create_table(
        "experience_records",
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("attempt_id", sa.Text(), nullable=True),
        sa.Column("thread_id", sa.Text(), nullable=True),
        sa.Column("student_id", sa.Text(), nullable=True),
        sa.Column("target_skill", sa.Text(), nullable=True),
        sa.Column("mastery_before", sa.Float(), nullable=True),
        sa.Column("mastery_after", sa.Float(), nullable=True),
        sa.Column("delta_mastery", sa.Float(), nullable=True),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_experience_records"),
        schema=RESEARCH,
    )
    for table, columns, name in (
        ("turn_actions", ["attempt_id", "turn_index"], "ix_turn_actions_attempt"),
        ("turn_actions", ["student_id", "created_at"], "ix_turn_actions_student"),
        ("selector_decisions", ["attempt_id", "turn_index"], "ix_selector_decisions_attempt"),
        ("policy_updates", ["policy_key", "created_at"], "ix_policy_updates_order"),
        ("experience_records", ["student_id", "created_at"], "ix_experience_records_student"),
    ):
        op.create_index(name, table, columns, schema=RESEARCH)
    op.execute(
        sa.text(
            f"CREATE VIEW {RESEARCH}.md6_decisions AS "
            "SELECT * FROM research.selector_decisions "
            "WHERE lower(coalesce(selector_version, '')) LIKE 'md6%'"
        )
    )


def upgrade() -> None:
    _create_schemas()
    _create_auth()
    _create_tutor()
    _create_student_model()
    _create_research()


def downgrade() -> None:
    op.execute(sa.text(f"DROP VIEW IF EXISTS {RESEARCH}.md6_decisions"))
    for table in (
        "experience_records",
        "policy_updates",
        "mrb1_scores",
        "policy_contexts",
        "c3_contexts",
        "selector_decisions",
        "turn_actions",
        "policy_states",
    ):
        op.drop_table(table, schema=RESEARCH)
    for table in (
        "sessions",
        "resolved_events",
        "attempts",
        "bkt_initial_priors",
        "mastery",
        "students",
    ):
        op.drop_table(table, schema=STUDENT_MODEL)
    for table in (
        "threads",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoints",
        "checkpoint_migrations",
    ):
        op.drop_table(table, schema=TUTOR)
    op.execute(sa.text(f"DROP VIEW IF EXISTS {AUTH}.user_accounts"))
    op.drop_table("sessions", schema=AUTH)
    for schema in (RESEARCH, STUDENT_MODEL, TUTOR, AUTH):
        op.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}"'))
