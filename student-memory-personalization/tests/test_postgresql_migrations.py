"""Alembic configuration and live PostgreSQL core-schema migration tests."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from src.database.base import Base
from src.database.postgres_config import (
    DATABASE_URL_ENV,
    DEFAULT_POSTGRES_SCHEMA,
    load_postgres_settings,
)
from src.database.postgres_session import create_postgres_engine
from src.database.models import core  # noqa: F401 - registers metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
EXPECTED_TABLES = {
    "students",
    "learning_sessions",
    "canonical_skills",
    "skill_aliases",
}
EXPECTED_HEAD_TABLES = EXPECTED_TABLES | {
    "completed_attempt_receipts",
    "concept_memory",
    "current_learning_state",
    "interaction_logs",
    "learning_state_snapshots",
    "long_term_memory",
    "repair_outcomes",
    "short_term_memory",
    "student_misconceptions",
    "topic_extraction_logs",
    "user_accounts",
}
FORBIDDEN_TABLE_FRAGMENTS = {
    "interaction",
    "assessment",
    "memory",
    "misconception",
    "learning_state",
    "repair",
}


def _alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


def test_alembic_configuration_loads():
    config = _alembic_config()

    assert Path(config.get_main_option("script_location")).resolve() == (
        PROJECT_ROOT / "alembic"
    ).resolve()
    assert (PROJECT_ROOT / "alembic" / "env.py").is_file()
    assert (PROJECT_ROOT / "alembic" / "script.py.mako").is_file()


def test_migration_chain_has_expected_head():
    scripts = ScriptDirectory.from_config(_alembic_config())

    assert scripts.get_heads() == ["0011_adaptmath_ingestion"]
    first_revision = scripts.get_revision("0001_core_identity")
    second_revision = scripts.get_revision("0002_raw_interactions")
    third_revision = scripts.get_revision("0003_memory_projections")
    fourth_revision = scripts.get_revision("0004_supporting_memory")
    fifth_revision = scripts.get_revision("0005_support_audit")
    sixth_revision = scripts.get_revision("0006_user_accounts")
    seventh_revision = scripts.get_revision("0007_remove_admin_accounts")
    eighth_revision = scripts.get_revision("0008_add_names_to_memory")
    ninth_revision = scripts.get_revision("0009_add_long_term_skill_names")
    tenth_revision = scripts.get_revision("0010_memory_skill_names")
    eleventh_revision = scripts.get_revision("0011_adaptmath_ingestion")
    assert first_revision is not None
    assert first_revision.down_revision is None
    assert second_revision is not None
    assert second_revision.down_revision == "0001_core_identity"
    assert third_revision is not None
    assert third_revision.down_revision == "0002_raw_interactions"
    assert fourth_revision is not None
    assert fourth_revision.down_revision == "0003_memory_projections"
    assert fifth_revision is not None
    assert fifth_revision.down_revision == "0004_supporting_memory"
    assert sixth_revision is not None
    assert sixth_revision.down_revision == "0005_support_audit"
    assert seventh_revision is not None
    assert seventh_revision.down_revision == "0006_user_accounts"
    assert eighth_revision is not None
    assert eighth_revision.down_revision == "0007_remove_admin_accounts"
    assert ninth_revision is not None
    assert ninth_revision.down_revision == "0008_add_names_to_memory"
    assert tenth_revision is not None
    assert tenth_revision.down_revision == "0009_add_long_term_skill_names"
    assert eleventh_revision is not None
    assert eleventh_revision.down_revision == "0010_memory_skill_names"


def test_core_metadata_tables_remain_registered():
    qualified_tables = set(Base.metadata.tables)

    assert {
        f"{DEFAULT_POSTGRES_SCHEMA}.{table}" for table in EXPECTED_TABLES
    }.issubset(qualified_tables)


@pytest.fixture(scope="module")
def migrated_postgres_engine():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for live migrations.")

    settings = load_postgres_settings()
    assert settings.schema == DEFAULT_POSTGRES_SCHEMA
    config = _alembic_config()
    engine = create_postgres_engine(settings)

    # This first migration owns only the student_memory schema. A failed
    # cleanup is surfaced rather than silently dropping unknown objects.
    command.downgrade(config, "base")
    command.upgrade(config, "0001_core_identity")

    inspector = inspect(engine)
    assert set(inspector.get_table_names(schema=settings.schema)) == EXPECTED_TABLES

    command.downgrade(config, "base")
    with engine.connect() as connection:
        schema_exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = :schema
                )
                """
            ),
            {"schema": settings.schema},
        ).scalar_one()
    assert schema_exists is False

    command.upgrade(config, "0001_core_identity")
    try:
        yield engine
    finally:
        command.upgrade(config, "head")
        engine.dispose()


def test_upgrade_downgrade_reupgrade_cycle(migrated_postgres_engine):
    inspector = inspect(migrated_postgres_engine)
    assert set(
        inspector.get_table_names(schema=DEFAULT_POSTGRES_SCHEMA)
    ) == EXPECTED_TABLES


def test_expected_primary_and_foreign_keys_exist(migrated_postgres_engine):
    inspector = inspect(migrated_postgres_engine)

    for table in EXPECTED_TABLES:
        primary_key = inspector.get_pk_constraint(
            table,
            schema=DEFAULT_POSTGRES_SCHEMA,
        )
        assert primary_key["constrained_columns"]

    session_foreign_keys = inspector.get_foreign_keys(
        "learning_sessions",
        schema=DEFAULT_POSTGRES_SCHEMA,
    )
    alias_foreign_keys = inspector.get_foreign_keys(
        "skill_aliases",
        schema=DEFAULT_POSTGRES_SCHEMA,
    )
    skill_foreign_keys = inspector.get_foreign_keys(
        "canonical_skills",
        schema=DEFAULT_POSTGRES_SCHEMA,
    )

    assert any(fk["referred_table"] == "students" for fk in session_foreign_keys)
    assert any(
        fk["referred_table"] == "canonical_skills" for fk in alias_foreign_keys
    )
    assert any(
        fk["referred_table"] == "canonical_skills" for fk in skill_foreign_keys
    )


def test_uniqueness_rules_are_enforced(migrated_postgres_engine):
    student_id = uuid.uuid4()

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text(
                f"""
                INSERT INTO {DEFAULT_POSTGRES_SCHEMA}.students
                    (student_id, external_student_id)
                VALUES (:student_id, :external_student_id)
                """
            ),
            {
                "student_id": student_id,
                "external_student_id": "migration-unique-student",
            },
        )

    with pytest.raises(IntegrityError):
        with migrated_postgres_engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {DEFAULT_POSTGRES_SCHEMA}.students
                        (student_id, external_student_id)
                    VALUES (:student_id, :external_student_id)
                    """
                ),
                {
                    "student_id": uuid.uuid4(),
                    "external_student_id": "migration-unique-student",
                },
            )


def test_no_interaction_or_memory_tables_are_created(migrated_postgres_engine):
    tables = set(
        inspect(migrated_postgres_engine).get_table_names(
            schema=DEFAULT_POSTGRES_SCHEMA
        )
    )

    assert tables == EXPECTED_TABLES
    assert not any(
        fragment in table
        for table in tables
        for fragment in FORBIDDEN_TABLE_FRAGMENTS
    )


def test_upgrade_to_full_head_is_complete_and_idempotent():
    """Verify all migrations and integration constraints on the test database."""
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for live migrations.")

    config = _alembic_config()
    command.upgrade(config, "head")
    settings = load_postgres_settings()
    engine = create_postgres_engine(settings)
    try:
        inspector = inspect(engine)
        assert set(
            inspector.get_table_names(schema=settings.schema)
        ) == EXPECTED_HEAD_TABLES

        receipt_uniques = {
            item["name"]
            for item in inspector.get_unique_constraints(
                "completed_attempt_receipts",
                schema=settings.schema,
            )
        }
        interaction_uniques = {
            item["name"]
            for item in inspector.get_unique_constraints(
                "interaction_logs",
                schema=settings.schema,
            )
        }
        repair_uniques = {
            item["name"]
            for item in inspector.get_unique_constraints(
                "repair_outcomes",
                schema=settings.schema,
            )
        }
        receipt_indexes = {
            item["name"]
            for item in inspector.get_indexes(
                "completed_attempt_receipts",
                schema=settings.schema,
            )
        }

        assert "uq_completed_attempt_receipts_source_attempt" in receipt_uniques
        assert "uq_interaction_logs_source_external_id" in interaction_uniques
        assert "uq_repair_outcomes_durable_event" in repair_uniques
        assert "ix_completed_attempt_receipts_student_created" in receipt_indexes

        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "0011_adaptmath_ingestion"

        # A second head upgrade must be an explicit no-op.
        command.upgrade(config, "head")
    finally:
        engine.dispose()


