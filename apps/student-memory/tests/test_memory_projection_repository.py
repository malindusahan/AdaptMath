"""Live PostgreSQL tests for the three derived memory projections."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

from alembic import command
from alembic.config import Config
import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from src.database.postgres_config import (
    DATABASE_URL_ENV,
    DEFAULT_POSTGRES_SCHEMA,
    load_postgres_settings,
)
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
    session_scope,
)
from src.database.repositories.memory_projection_repository import (
    MemoryProjectionRepository,
)
from src.schemas.memory_projection import (
    ConceptMemoryUpsert,
    LongTermMemoryUpsert,
    ShortTermMemoryUpsert,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA
PROJECTION_TABLES = {
    "short_term_memory",
    "long_term_memory",
    "concept_memory",
}


@pytest.fixture(scope="module")
def projection_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for projection tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "0003_memory_projections")
    assert PROJECTION_TABLES.issubset(
        inspect(engine).get_table_names(schema=SCHEMA)
    )

    command.downgrade(config, "0002_raw_interactions")
    tables_at_0002 = set(inspect(engine).get_table_names(schema=SCHEMA))
    assert not PROJECTION_TABLES.intersection(tables_at_0002)
    assert "interaction_logs" in tables_at_0002

    command.upgrade(config, "0003_memory_projections")
    factory = create_session_factory(engine)
    try:
        yield engine, factory
    finally:
        command.upgrade(config, "head")
        engine.dispose()


def _seed_context(engine, *, sessions=1, skills=1):
    student_id = uuid.uuid4()
    session_ids = [uuid.uuid4() for _ in range(sessions)]
    skill_ids = [uuid.uuid4() for _ in range(skills)]

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                INSERT INTO {SCHEMA}.students
                    (student_id, external_student_id)
                VALUES (:student_id, :external_student_id)
                """
            ),
            {
                "student_id": student_id,
                "external_student_id": f"projection-student-{student_id}",
            },
        )
        for session_id in session_ids:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {SCHEMA}.learning_sessions
                        (session_id, student_id, external_session_id)
                    VALUES (:session_id, :student_id, :external_session_id)
                    """
                ),
                {
                    "session_id": session_id,
                    "student_id": student_id,
                    "external_session_id": f"projection-session-{session_id}",
                },
            )
        for skill_id in skill_ids:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {SCHEMA}.canonical_skills
                        (skill_id, canonical_name, display_name)
                    VALUES (:skill_id, :canonical_name, :display_name)
                    """
                ),
                {
                    "skill_id": skill_id,
                    "canonical_name": f"projection-skill-{skill_id}",
                    "display_name": "Projection Skill",
                },
            )

    return student_id, session_ids, skill_ids


def _repository(factory):
    return factory(), None


def test_migration_upgrade_downgrade_reupgrade(projection_database):
    engine, _ = projection_database
    tables = set(inspect(engine).get_table_names(schema=SCHEMA))

    assert PROJECTION_TABLES.issubset(tables)
    assert "interaction_logs" in tables


def test_zero_evidence_is_preserved_without_fake_averages(projection_database):
    engine, factory = projection_database
    student_id, sessions, skills = _seed_context(engine)

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        stored = repository.upsert_short_term_memory(
            ShortTermMemoryUpsert(
                student_id=student_id,
                session_id=sessions[0],
                current_skill_id=skills[0],
            )
        )

    assert stored.interaction_count == 0
    assert stored.attempt_observation_count == 0
    assert stored.attempt_sum == 0
    assert stored.average_attempts is None
    assert stored.average_hints is None
    assert stored.average_response_time_ms is None
    assert stored.evidence_accuracy is None
    assert stored.recent_accuracy is None


def test_observed_evidence_averages_are_derived_from_counts_and_sums(
    projection_database,
):
    engine, factory = projection_database
    student_id, _, _ = _seed_context(engine)

    projection = LongTermMemoryUpsert(
        student_id=student_id,
        total_sessions=2,
        interaction_count=4,
        correct_count=3,
        incorrect_count=1,
        overall_accuracy=0.75,
        attempt_observation_count=2,
        attempt_sum=5,
        hint_observation_count=2,
        hint_sum=1,
        response_time_observation_count=2,
        response_time_sum_ms=3000,
        concept_count=2,
    )

    with session_scope(factory) as session:
        stored = MemoryProjectionRepository(session).upsert_long_term_memory(
            projection
        )

    assert stored.average_attempts == 2.5
    assert stored.average_hints == 0.5
    assert stored.average_response_time_ms == 1500
    assert stored.evidence_accuracy == 0.75


def test_short_term_memory_isolated_by_student_and_session(projection_database):
    engine, factory = projection_database
    student_id, sessions, skills = _seed_context(engine, sessions=2)

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        for index, session_id in enumerate(sessions, start=1):
            repository.upsert_short_term_memory(
                ShortTermMemoryUpsert(
                    student_id=student_id,
                    session_id=session_id,
                    current_skill_id=skills[0],
                    interaction_count=index,
                )
            )

    with factory() as session:
        repository = MemoryProjectionRepository(session)
        assert repository.get_short_term_memory(
            student_id, sessions[0]
        ).interaction_count == 1
        assert repository.get_short_term_memory(
            student_id, sessions[1]
        ).interaction_count == 2


def test_long_term_memory_isolated_by_student(projection_database):
    engine, factory = projection_database
    first_student, _, _ = _seed_context(engine)
    second_student, _, _ = _seed_context(engine)

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        repository.upsert_long_term_memory(
            LongTermMemoryUpsert(student_id=first_student, interaction_count=1)
        )
        repository.upsert_long_term_memory(
            LongTermMemoryUpsert(student_id=second_student, interaction_count=3)
        )

    with factory() as session:
        repository = MemoryProjectionRepository(session)
        assert repository.get_long_term_memory(first_student).interaction_count == 1
        assert repository.get_long_term_memory(second_student).interaction_count == 3


def test_concept_memory_isolated_by_student_and_skill(projection_database):
    engine, factory = projection_database
    student_id, _, skills = _seed_context(engine, skills=2)

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        repository.upsert_concept_memory(
            ConceptMemoryUpsert(
                student_id=student_id,
                canonical_skill_id=skills[0],
                interaction_count=2,
                correct_count=1,
                incorrect_count=1,
                accuracy=0.5,
            )
        )
        repository.upsert_concept_memory(
            ConceptMemoryUpsert(
                student_id=student_id,
                canonical_skill_id=skills[1],
                interaction_count=4,
                correct_count=4,
                accuracy=1,
            )
        )

    with factory() as session:
        repository = MemoryProjectionRepository(session)
        assert repository.get_concept_memory(
            student_id, skills[0]
        ).accuracy == 0.5
        assert repository.get_concept_memory(
            student_id, skills[1]
        ).accuracy == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("interaction_count", -1),
        ("correct_count", -1),
        ("incorrect_count", -1),
        ("attempt_observation_count", -1),
        ("attempt_sum", -1),
        ("hint_observation_count", -1),
        ("hint_sum", -1),
        ("response_time_observation_count", -1),
        ("response_time_sum_ms", -1),
    ],
)
def test_negative_evidence_values_are_rejected(field, value):
    values = {"student_id": uuid.uuid4(), field: value}
    with pytest.raises(ValidationError):
        LongTermMemoryUpsert(**values)


@pytest.mark.parametrize("accuracy", [-0.01, 1.01])
def test_accuracy_outside_unit_interval_is_rejected(accuracy):
    with pytest.raises(ValidationError):
        LongTermMemoryUpsert(
            student_id=uuid.uuid4(),
            interaction_count=1,
            overall_accuracy=accuracy,
        )


def test_nonzero_sum_requires_observation_count():
    with pytest.raises(ValidationError, match="attempt_sum must be zero"):
        LongTermMemoryUpsert(
            student_id=uuid.uuid4(),
            attempt_observation_count=0,
            attempt_sum=1,
        )


def test_database_constraints_reject_bypassed_negative_counts(projection_database):
    engine, _ = projection_database
    student_id, _, _ = _seed_context(engine)

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {SCHEMA}.long_term_memory
                        (student_id, interaction_count)
                    VALUES (:student_id, -1)
                    """
                ),
                {"student_id": student_id},
            )


def test_foreign_keys_are_restrictive_and_enforced(projection_database):
    engine, factory = projection_database
    unknown_student = uuid.uuid4()

    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            MemoryProjectionRepository(session).upsert_long_term_memory(
                LongTermMemoryUpsert(student_id=unknown_student)
            )

    foreign_keys = inspect(engine).get_foreign_keys(
        "long_term_memory", schema=SCHEMA
    )
    assert foreign_keys[0]["options"].get("ondelete") == "RESTRICT"


def test_upsert_updates_without_creating_duplicate_rows(projection_database):
    engine, factory = projection_database
    student_id, sessions, skills = _seed_context(engine)

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        repository.upsert_short_term_memory(
            ShortTermMemoryUpsert(
                student_id=student_id,
                session_id=sessions[0],
                current_skill_id=skills[0],
                interaction_count=1,
            )
        )
        updated = repository.upsert_short_term_memory(
            ShortTermMemoryUpsert(
                student_id=student_id,
                session_id=sessions[0],
                current_skill_id=skills[0],
                interaction_count=3,
            )
        )

    with engine.connect() as connection:
        count = connection.execute(
            text(
                f"""
                SELECT COUNT(*) FROM {SCHEMA}.short_term_memory
                WHERE student_id = :student_id AND session_id = :session_id
                """
            ),
            {"student_id": student_id, "session_id": sessions[0]},
        ).scalar_one()

    assert updated.interaction_count == 3
    assert count == 1


def test_transaction_rollback_removes_projection_write(projection_database):
    engine, factory = projection_database
    student_id, _, _ = _seed_context(engine)

    with pytest.raises(RuntimeError, match="rollback projection"):
        with session_scope(factory) as session:
            MemoryProjectionRepository(session).upsert_long_term_memory(
                LongTermMemoryUpsert(student_id=student_id, interaction_count=2)
            )
            raise RuntimeError("rollback projection")

    with factory() as session:
        assert (
            MemoryProjectionRepository(session).get_long_term_memory(student_id)
            is None
        )


def test_projection_upserts_do_not_modify_raw_interactions(projection_database):
    engine, factory = projection_database
    student_id, sessions, skills = _seed_context(engine)
    interaction_id = uuid.uuid4()

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                INSERT INTO {SCHEMA}.interaction_logs
                    (interaction_id, student_id, session_id,
                     canonical_skill_id, source, external_interaction_id)
                VALUES (:interaction_id, :student_id, :session_id,
                        :skill_id, 'projection-test', :external_id)
                """
            ),
            {
                "interaction_id": interaction_id,
                "student_id": student_id,
                "session_id": sessions[0],
                "skill_id": skills[0],
                "external_id": str(interaction_id),
            },
        )

    with session_scope(factory) as session:
        repository = MemoryProjectionRepository(session)
        repository.upsert_short_term_memory(
            ShortTermMemoryUpsert(
                student_id=student_id,
                session_id=sessions[0],
                current_skill_id=skills[0],
            )
        )
        repository.upsert_long_term_memory(
            LongTermMemoryUpsert(student_id=student_id)
        )
        repository.upsert_concept_memory(
            ConceptMemoryUpsert(
                student_id=student_id,
                canonical_skill_id=skills[0],
            )
        )

    with engine.connect() as connection:
        count = connection.execute(
            text(
                f"""
                SELECT COUNT(*) FROM {SCHEMA}.interaction_logs
                WHERE interaction_id = :interaction_id
                """
            ),
            {"interaction_id": interaction_id},
        ).scalar_one()

    assert count == 1

