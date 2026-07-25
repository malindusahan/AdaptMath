"""Live PostgreSQL verification for supporting memory and audit persistence."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    RepairOutcome,
    StudentMisconception,
    TopicExtractionLog,
)
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
from src.database.repositories.repair_outcome_repository import (
    RepairOutcomeRepository,
)
from src.database.repositories.supporting_memory_repository import (
    SupportingMemoryRepository,
)
from src.database.repositories.topic_audit_repository import TopicAuditRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA
SUPPORTING_TABLES = {
    "student_misconceptions",
    "learning_state_snapshots",
    "current_learning_state",
    "topic_extraction_logs",
    "repair_outcomes",
}


@pytest.fixture(scope="module")
def supporting_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for supporting tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    assert SUPPORTING_TABLES.issubset(
        inspect(engine).get_table_names(schema=SCHEMA)
    )

    factory = create_session_factory(engine)
    try:
        yield engine, factory, config
    finally:
        # Preserve the project's expected live migration head after verification.
        command.upgrade(config, "head")
        engine.dispose()


def _seed_context(engine, *, students=1, skills=1):
    student_ids = [uuid.uuid4() for _ in range(students)]
    session_ids = [uuid.uuid4() for _ in range(students)]
    skill_ids = [uuid.uuid4() for _ in range(skills)]

    with engine.begin() as connection:
        for student_id, session_id in zip(student_ids, session_ids, strict=True):
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
                    "external_student_id": f"support-student-{student_id}",
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
                    "external_session_id": f"support-session-{session_id}",
                },
            )

        for skill_id in skill_ids:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {SCHEMA}.canonical_skills
                        (skill_id, canonical_name, display_name)
                    VALUES (:skill_id, :canonical_name, 'Supporting Skill')
                    """
                ),
                {
                    "skill_id": skill_id,
                    "canonical_name": f"support-skill-{skill_id}",
                },
            )

    return student_ids, session_ids, skill_ids


def _snapshot_values(student_id, session_id, skill_id, *, state="DEVELOPING"):
    return {
        "student_id": student_id,
        "session_id": session_id,
        "canonical_skill_id": skill_id,
        "learning_state": state,
        "evidence_level": "PARTIAL_SKILL",
        "evidence_strength": "MEDIUM",
        "behavioural_coverage": "CORRECTNESS_ONLY_COVERAGE",
        "model_used": True,
        "previous_interaction_count": 2,
        "previous_skill_interaction_count": 2,
        "recent_interaction_count": 2,
        "attempt_observation_count": 0,
        "hint_observation_count": 0,
        "response_time_observation_count": 0,
        "model_version": "support-test",
    }


def test_0004_and_0005_upgrade_downgrade_reupgrade(supporting_database):
    engine, _, config = supporting_database

    command.downgrade(config, "0003_memory_projections")
    tables_at_0003 = set(inspect(engine).get_table_names(schema=SCHEMA))
    assert not SUPPORTING_TABLES.intersection(tables_at_0003)

    command.upgrade(config, "0005_support_audit")
    tables_at_0005 = set(inspect(engine).get_table_names(schema=SCHEMA))
    assert SUPPORTING_TABLES.issubset(tables_at_0005)

    command.upgrade(config, "head")


def test_supporting_tables_have_expected_foreign_keys_and_checks(
    supporting_database,
):
    engine, _, _ = supporting_database
    inspector = inspect(engine)

    for table in SUPPORTING_TABLES:
        assert inspector.get_pk_constraint(table, schema=SCHEMA)[
            "constrained_columns"
        ]
        assert inspector.get_foreign_keys(table, schema=SCHEMA)

    misconception_checks = inspector.get_check_constraints(
        "student_misconceptions", schema=SCHEMA
    )
    snapshot_checks = inspector.get_check_constraints(
        "learning_state_snapshots", schema=SCHEMA
    )
    topic_checks = inspector.get_check_constraints(
        "topic_extraction_logs", schema=SCHEMA
    )
    repair_checks = inspector.get_check_constraints(
        "repair_outcomes", schema=SCHEMA
    )

    assert any("occurrence_count" in check["sqltext"] for check in misconception_checks)
    assert any("learning_state" in check["sqltext"] for check in snapshot_checks)
    assert any("confidence" in check["sqltext"] for check in topic_checks)
    assert any("score" in check["sqltext"] for check in repair_checks)


def test_misconception_upsert_normalizes_counts_and_isolates_context(
    supporting_database,
):
    engine, factory, _ = supporting_database
    students, _, skills = _seed_context(engine, students=2, skills=2)

    with session_scope(factory) as session:
        repository = SupportingMemoryRepository(session)
        first = repository.upsert_misconception(
            students[0], skills[0], "  Addition   Error "
        )
        repeated = repository.upsert_misconception(
            students[0], skills[0], "addition error"
        )
        other_skill = repository.upsert_misconception(
            students[0], skills[1], "addition error"
        )
        other_student = repository.upsert_misconception(
            students[1], skills[0], "addition error"
        )

    assert first.misconception_id == repeated.misconception_id
    assert repeated.normalized_error == "addition error"
    assert repeated.occurrence_count == 2
    assert other_skill.occurrence_count == 1
    assert other_student.occurrence_count == 1


def test_learning_state_snapshots_append_and_current_state_upserts(
    supporting_database,
):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine)

    with session_scope(factory) as session:
        repository = SupportingMemoryRepository(session)
        first = repository.add_snapshot(
            **_snapshot_values(students[0], sessions[0], skills[0])
        )
        current_first = repository.upsert_current(first)
        second = repository.add_snapshot(
            **_snapshot_values(
                students[0], sessions[0], skills[0], state="STRONG"
            )
        )
        current_second = repository.upsert_current(second)

    with factory() as session:
        repository = SupportingMemoryRepository(session)
        history = repository.get_history(students[0], skills[0])
        current = repository.get_current(students[0], skills[0])

    assert first.snapshot_id != second.snapshot_id
    assert [row.learning_state for row in history] == ["DEVELOPING", "STRONG"]
    assert current_first.student_id == current_second.student_id
    assert current.learning_state == "STRONG"
    assert current.last_snapshot_id == second.snapshot_id


def test_learning_state_history_isolated_by_student_and_skill(supporting_database):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine, students=2, skills=2)

    with session_scope(factory) as session:
        repository = SupportingMemoryRepository(session)
        repository.add_snapshot(
            **_snapshot_values(students[0], sessions[0], skills[0])
        )
        repository.add_snapshot(
            **_snapshot_values(students[0], sessions[0], skills[1])
        )
        repository.add_snapshot(
            **_snapshot_values(students[1], sessions[1], skills[0])
        )

    with factory() as session:
        repository = SupportingMemoryRepository(session)
        assert len(repository.get_history(students[0], skills[0])) == 1
        assert len(repository.get_history(students[0], skills[1])) == 1
        assert len(repository.get_history(students[1], skills[0])) == 1


def test_topic_extraction_audit_insert_and_student_isolation(supporting_database):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine, students=2)

    with session_scope(factory) as session:
        repository = TopicAuditRepository(session)
        first = repository.add(
            student_id=students[0],
            session_id=sessions[0],
            canonical_skill_id=skills[0],
            input_text="Solve 2x = 10",
            extraction_method="RULE",
            confidence=0.9,
            alternatives=[{"skill": "linear equations", "score": 0.9}],
            model_version="rules-1",
            ontology_version="1.0",
        )
        repository.add(
            student_id=students[1],
            session_id=sessions[1],
            input_text="Another prompt",
            extraction_method="ABSTAIN",
            needs_review=True,
        )

    with factory() as session:
        first_student_rows = TopicAuditRepository(session).for_student(students[0])

    assert [row.extraction_id for row in first_student_rows] == [first.extraction_id]
    assert first_student_rows[0].model_version == "rules-1"


def test_repair_outcome_insert_retrieval_and_context_isolation(supporting_database):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine, students=2, skills=2)

    with session_scope(factory) as session:
        repository = RepairOutcomeRepository(session)
        first = repository.add(
            student_id=students[0],
            session_id=sessions[0],
            canonical_skill_id=skills[0],
            repair_action="worked example",
            outcome="IMPROVED",
            score=0.8,
        )
        repository.add(
            student_id=students[0],
            session_id=sessions[0],
            canonical_skill_id=skills[1],
            repair_action="hint",
            outcome="UNCHANGED",
        )
        repository.add(
            student_id=students[1],
            session_id=sessions[1],
            canonical_skill_id=skills[0],
            repair_action="retry",
            outcome="IMPROVED",
        )

    with factory() as session:
        rows = RepairOutcomeRepository(session).for_student_skill(
            students[0], skills[0]
        )

    assert [row.repair_outcome_id for row in rows] == [first.repair_outcome_id]


@pytest.mark.parametrize(
    "model,values",
    [
        (
            TopicExtractionLog,
            {
                "extraction_id": uuid.uuid4(),
                "input_text": "invalid confidence",
                "extraction_method": "TEST",
                "confidence": 1.1,
                "needs_review": False,
            },
        ),
        (
            RepairOutcome,
            {
                "repair_outcome_id": uuid.uuid4(),
                "repair_action": "test",
                "outcome": "test",
                "score": -0.1,
            },
        ),
    ],
)
def test_database_constraints_reject_invalid_audit_values(
    supporting_database, model, values
):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine)
    values.update(
        student_id=students[0],
        session_id=sessions[0],
        canonical_skill_id=skills[0],
    )

    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.add(model(**values))


def test_cross_student_session_foreign_keys_are_rejected(supporting_database):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine, students=2)

    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            TopicAuditRepository(session).add(
                student_id=students[0],
                session_id=sessions[1],
                canonical_skill_id=skills[0],
                input_text="cross-student session",
                extraction_method="TEST",
            )


def test_transaction_rollback_removes_all_supporting_writes(supporting_database):
    engine, factory, _ = supporting_database
    students, sessions, skills = _seed_context(engine)
    normalized_error = f"rollback-{uuid.uuid4()}"

    with pytest.raises(RuntimeError, match="rollback supporting"):
        with session_scope(factory) as session:
            SupportingMemoryRepository(session).upsert_misconception(
                students[0], skills[0], normalized_error
            )
            TopicAuditRepository(session).add(
                student_id=students[0],
                session_id=sessions[0],
                canonical_skill_id=skills[0],
                input_text="rollback input",
                extraction_method="TEST",
            )
            raise RuntimeError("rollback supporting")

    with factory() as session:
        misconception = session.scalar(
            select(StudentMisconception).where(
                StudentMisconception.student_id == students[0],
                StudentMisconception.canonical_skill_id == skills[0],
                StudentMisconception.normalized_error == normalized_error,
            )
        )
        audit_count = session.scalar(
            select(TopicExtractionLog).where(
                TopicExtractionLog.student_id == students[0],
                TopicExtractionLog.input_text == "rollback input",
            ).with_only_columns(text("count(*)"))
        )

    assert misconception is None
    assert audit_count == 0


def test_supporting_models_are_exactly_the_intended_five():
    assert {
        StudentMisconception.__tablename__,
        LearningStateSnapshot.__tablename__,
        CurrentLearningState.__tablename__,
        TopicExtractionLog.__tablename__,
        RepairOutcome.__tablename__,
    } == SUPPORTING_TABLES
