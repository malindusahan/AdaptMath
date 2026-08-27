"""Live PostgreSQL tests for raw interaction migration and repository behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
)
from src.schemas.raw_interaction import RawInteractionCreate
from src.services.raw_interaction_service import RawInteractionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA


@pytest.fixture(scope="module")
def raw_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for live repository tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "0002_raw_interactions")
    assert "interaction_logs" in inspect(engine).get_table_names(schema=SCHEMA)

    command.downgrade(config, "0001_core_identity")
    assert "interaction_logs" not in inspect(engine).get_table_names(schema=SCHEMA)

    command.upgrade(config, "0002_raw_interactions")
    service = RawInteractionService(create_session_factory(engine))
    try:
        yield engine, service
    finally:
        command.upgrade(config, "head")
        engine.dispose()


def _seed_context(engine):
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    first_skill_id = uuid.uuid4()
    second_skill_id = uuid.uuid4()

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
                "external_student_id": f"student-{student_id}",
            },
        )
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
                "external_session_id": f"session-{session_id}",
            },
        )
        for skill_id, suffix in (
            (first_skill_id, "first"),
            (second_skill_id, "second"),
        ):
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
                    "canonical_name": f"skill-{suffix}-{skill_id}",
                    "display_name": f"Skill {suffix}",
                },
            )

    return student_id, session_id, first_skill_id, second_skill_id


def _interaction(
    student_id,
    session_id,
    skill_id,
    *,
    external_id=None,
    created_at=None,
    **overrides,
):
    values = {
        "student_id": student_id,
        "session_id": session_id,
        "canonical_skill_id": skill_id,
        "source": "repository-test",
        "external_interaction_id": external_id or f"interaction-{uuid.uuid4()}",
        "problem_id": "problem-1",
        "question_id": "question-1",
        "student_utterance": "I think the answer is five.",
        "tutor_response": "Show your reasoning.",
        "student_answer": "5",
        "expected_answer": "5",
        "is_correct": True,
        "created_at": created_at,
    }
    values.update(overrides)
    return RawInteractionCreate(**values)


def test_raw_migration_upgrade_downgrade_reupgrade(raw_database):
    engine, _ = raw_database
    assert set(inspect(engine).get_table_names(schema=SCHEMA)) == {
        "students",
        "learning_sessions",
        "canonical_skills",
        "skill_aliases",
        "interaction_logs",
    }


def test_raw_table_has_expected_foreign_keys_and_checks(raw_database):
    engine, _ = raw_database
    inspector = inspect(engine)
    foreign_keys = inspector.get_foreign_keys("interaction_logs", schema=SCHEMA)
    checks = inspector.get_check_constraints("interaction_logs", schema=SCHEMA)

    assert {foreign_key["name"] for foreign_key in foreign_keys} == {
        "fk_interaction_logs_student_id_students",
        "fk_interaction_logs_session_student_learning_sessions",
        "fk_interaction_logs_skill_id_canonical_skills",
    }
    assert {
        "ck_interaction_logs_attempt_count_nonnegative",
        "ck_interaction_logs_hint_count_nonnegative",
        "ck_interaction_logs_hint_total_nonnegative",
        "ck_interaction_logs_response_time_nonnegative",
        "ck_interaction_logs_hint_count_within_total",
    }.issubset({check["name"] for check in checks})


def test_valid_insert_and_get_by_id(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)

    created = service.insert(
        _interaction(
            student_id,
            session_id,
            skill_id,
            attempt_count=2,
            hint_count=1,
            hint_total=3,
            response_time_ms=12500,
        )
    )
    retrieved = service.get_by_id(created.interaction_id)

    assert retrieved == created
    assert retrieved.student_id == student_id
    assert retrieved.canonical_skill_id == skill_id


def test_null_behavioural_values_remain_null(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)

    record = service.insert(_interaction(student_id, session_id, skill_id))

    assert record.attempt_count is None
    assert record.hint_count is None
    assert record.hint_total is None
    assert record.response_time_ms is None


def test_real_zero_behavioural_values_remain_zero(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)

    record = service.insert(
        _interaction(
            student_id,
            session_id,
            skill_id,
            attempt_count=0,
            hint_count=0,
            hint_total=0,
            response_time_ms=0,
        )
    )

    assert record.attempt_count == 0
    assert record.hint_count == 0
    assert record.hint_total == 0
    assert record.response_time_ms == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_count", -1),
        ("hint_count", -1),
        ("hint_total", -1),
        ("response_time_ms", -0.1),
    ],
)
def test_negative_behavioural_values_are_rejected(field, value):
    values = {
        "student_id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "source": "validation-test",
        "external_interaction_id": str(uuid.uuid4()),
        field: value,
    }

    with pytest.raises(ValidationError):
        RawInteractionCreate(**values)


def test_hint_count_cannot_exceed_hint_total():
    with pytest.raises(ValidationError, match="hint_count must not exceed"):
        RawInteractionCreate(
            student_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            source="validation-test",
            external_interaction_id=str(uuid.uuid4()),
            hint_count=2,
            hint_total=1,
        )


def test_database_checks_reject_bypassed_invalid_values(raw_database):
    engine, _ = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {SCHEMA}.interaction_logs
                        (interaction_id, student_id, session_id,
                         canonical_skill_id, source, external_interaction_id,
                         hint_count, hint_total)
                    VALUES
                        (:interaction_id, :student_id, :session_id,
                         :skill_id, :source, :external_id, 2, 1)
                    """
                ),
                {
                    "interaction_id": uuid.uuid4(),
                    "student_id": student_id,
                    "session_id": session_id,
                    "skill_id": skill_id,
                    "source": "constraint-test",
                    "external_id": str(uuid.uuid4()),
                },
            )


def test_interaction_cannot_reference_another_students_session(raw_database):
    engine, service = raw_database
    first_student, _, skill_id, _ = _seed_context(engine)
    _, second_session, _, _ = _seed_context(engine)

    with pytest.raises(IntegrityError):
        service.insert(_interaction(first_student, second_session, skill_id))


def test_skill_and_session_queries_are_isolated(raw_database):
    engine, service = raw_database
    student_id, session_id, first_skill, second_skill = _seed_context(engine)
    other_student, other_session, other_skill, _ = _seed_context(engine)

    first = service.insert(_interaction(student_id, session_id, first_skill))
    second = service.insert(_interaction(student_id, session_id, second_skill))
    service.insert(_interaction(other_student, other_session, other_skill))

    student_records = service.get_student_interactions(student_id)
    session_records = service.get_session_interactions(session_id)
    skill_records = service.get_student_skill_interactions(student_id, first_skill)

    assert {record.interaction_id for record in student_records} == {
        first.interaction_id,
        second.interaction_id,
    }
    assert {record.interaction_id for record in session_records} == {
        first.interaction_id,
        second.interaction_id,
    }
    assert [record.interaction_id for record in skill_records] == [
        first.interaction_id
    ]


def test_chronological_retrieval_uses_timestamp_then_id(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    late = service.insert(
        _interaction(
            student_id,
            session_id,
            skill_id,
            external_id=f"late-{uuid.uuid4()}",
            created_at=start + timedelta(minutes=10),
        )
    )
    early = service.insert(
        _interaction(
            student_id,
            session_id,
            skill_id,
            external_id=f"early-{uuid.uuid4()}",
            created_at=start,
        )
    )

    records = service.get_student_interactions(student_id)
    assert [record.interaction_id for record in records] == [
        early.interaction_id,
        late.interaction_id,
    ]


def test_duplicate_source_identity_is_rejected(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)
    external_id = f"duplicate-{uuid.uuid4()}"

    service.insert(
        _interaction(
            student_id,
            session_id,
            skill_id,
            external_id=external_id,
        )
    )
    with pytest.raises(IntegrityError):
        service.insert(
            _interaction(
                student_id,
                session_id,
                skill_id,
                external_id=external_id,
            )
        )


def test_atomic_batch_rolls_back_all_rows_on_failure(raw_database):
    engine, service = raw_database
    student_id, session_id, skill_id, _ = _seed_context(engine)
    external_id = f"rollback-{uuid.uuid4()}"

    batch = [
        _interaction(
            student_id,
            session_id,
            skill_id,
            external_id=external_id,
        ),
        _interaction(
            student_id,
            session_id,
            skill_id,
            external_id=external_id,
        ),
    ]

    with pytest.raises(IntegrityError):
        service.insert_many(batch)

    assert all(
        record.external_interaction_id != external_id
        for record in service.get_student_interactions(student_id)
    )
