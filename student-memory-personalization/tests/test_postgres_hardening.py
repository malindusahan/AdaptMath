"""Hardening tests for PostgreSQL concurrency, migrations, restart persistence, and failure sanitization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any
import uuid

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, text

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.connection import DEFAULT_DATABASE_PATH
from src.database.postgres_config import (
    DATABASE_URL_ENV,
    DEFAULT_POSTGRES_SCHEMA,
    load_postgres_settings,
)
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
    set_session_factory,
)
from src.database.unit_of_work import UnitOfWork
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)
from src.services.memory_update_service import (
    MemoryUpdateService,
    MemoryUpdateServiceError,
    update_student_memory,
)
from src.services.student_memory_query_service import StudentMemoryQueryService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA

EXPECTED_TABLES = {
    "students",
    "learning_sessions",
    "canonical_skills",
    "skill_aliases",
    "interaction_logs",
    "short_term_memory",
    "long_term_memory",
    "concept_memory",
    "student_misconceptions",
    "learning_state_snapshots",
    "current_learning_state",
    "topic_extraction_logs",
    "repair_outcomes",
    "user_accounts",
}


# =============================================================================
# Pure Unit Tests (Deterministic without live PostgreSQL)
# =============================================================================

def test_credential_sanitization_on_database_connection_failure(monkeypatch):
    """Verify that credentials and raw database traces are never exposed in error responses."""
    bad_dsn_secret = "postgresql+psycopg://superadmin:TopSecretPassword123@db.internal:5432/proddb"

    def bad_update(*args, **kwargs):
        raise RuntimeError(f"Connection to {bad_dsn_secret} timed out: password authentication failed")

    monkeypatch.setattr(memory_routes, "update_student_memory", bad_update)

    client = TestClient(app)
    res = client.post(
        "/memory/update",
        json={
            "student_id": "s1",
            "topic": "Algebra",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )

    assert res.status_code == 500
    assert res.json()["detail"] == "Memory update failed."
    assert "TopSecretPassword123" not in res.text
    assert "superadmin" not in res.text
    assert "db.internal" not in res.text


def test_query_failure_sanitization(monkeypatch):
    """Verify GET endpoints return sanitized 500 when query service encounters an internal exception."""
    def bad_get_current(*args, **kwargs):
        raise RuntimeError("SQL syntax error near SELECT * FROM secret_table")

    monkeypatch.setattr(memory_routes, "get_current_student_memory", bad_get_current)

    client = TestClient(app)
    res = client.get("/memory/s1/current?topic=Algebra")
    assert res.status_code == 500
    assert res.json()["detail"] == "Failed to retrieve student memory."
    assert "secret_table" not in res.text


# =============================================================================
# Live PostgreSQL Hardening Integration Tests
# =============================================================================

@pytest.fixture(scope="module")
def hardening_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for hardening tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)
    client = TestClient(app)

    try:
        yield engine, session_factory, client, config
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()


def test_fresh_database_migration_lifecycle(hardening_database):
    """Verify full migration cycle from base to head, verifying all 13 tables and sequence."""
    engine, _, _, config = hardening_database

    # Downgrade to base
    command.downgrade(config, "base")

    # Upgrade to head
    command.upgrade(config, "head")

    inspector = inspect(engine)
    tables = set(inspector.get_table_names(schema=SCHEMA))
    assert EXPECTED_TABLES.issubset(tables), f"Missing tables: {EXPECTED_TABLES - tables}"

    # Verify sequence exists in schema
    with engine.connect() as conn:
        seq_exists = conn.scalar(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_sequences WHERE schemaname = :schema AND sequencename = 'assessment_id_seq')"
            ),
            {"schema": SCHEMA},
        )
        assert seq_exists is True


def test_concurrency_same_student_multiple_updates(hardening_database):
    """Verify concurrent updates for the same student serialize cleanly via advisory lock."""
    engine, session_factory, client, _ = hardening_database
    student_id = f"conc_same_{uuid.uuid4().hex[:8]}"
    topic = "Algebra"
    subtopic = "Linear equations"

    num_concurrent = 4

    def post_update(worker_idx: int) -> dict[str, Any]:
        update_client = TestClient(app)
        payload = {
            "student_id": student_id,
            "topic": topic,
            "subtopic": subtopic,
            "assessment_questions": [
                {
                    "question_id": f"q_w{worker_idx}_1",
                    "student_answer": "4",
                    "expected_answer": "4",
                    "is_correct": True,
                    "attempt_count": 1,
                    "response_time_ms": 3000.0,
                },
                {
                    "question_id": f"q_w{worker_idx}_2",
                    "student_answer": "x=2",
                    "expected_answer": "x=3",
                    "is_correct": False,
                    "identified_error": "sign arithmetic error",
                    "attempt_count": 2,
                    "response_time_ms": 5000.0,
                },
            ],
            "identified_errors": ["sign arithmetic error"],
        }
        res = update_client.post("/memory/update", json=payload, headers={"X-Service-Key": os.environ.get("SERVICE_KEY", "dev-service-key")})
        return res.json()

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [
            executor.submit(post_update, i) for i in range(num_concurrent)
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == num_concurrent

    # Verify that all interactions and projections accumulated without loss
    with UnitOfWork(session_factory) as uow:
        student = uow.identities.get_student(student_id)
        assert student is not None
        skill = uow.identities.get_skill(topic, subtopic)
        assert skill is not None

        # 4 workers * 2 questions = 8 total interactions
        logs = uow.raw_interactions.get_student_interactions(student.student_id)
        assert len(logs) == 8

        # Verify STM, LTM, Concept projections accumulated correctly
        ltm = uow.projections.get_long_term_memory(student.student_id)
        assert ltm is not None
        assert ltm.interaction_count == 8
        assert ltm.correct_count == 4
        assert ltm.incorrect_count == 4

        concept = uow.projections.get_concept_memory(
            student.student_id, skill.skill_id
        )
        assert concept is not None
        assert concept.interaction_count == 8

        # Misconception occurrence accumulated
        misconceptions = uow.supporting.get_misconceptions(
            student.student_id, skill.skill_id
        )
        assert len(misconceptions) == 1
        assert misconceptions[0].occurrence_count == 4


def test_concurrency_distinct_students(hardening_database):
    """Verify concurrent updates for distinct students proceed simultaneously without deadlock."""
    _, session_factory, _, _ = hardening_database
    num_students = 4
    students = [f"conc_diff_{i}_{uuid.uuid4().hex[:6]}" for i in range(num_students)]
    topic = "Calculus"

    service = MemoryUpdateService(session_factory=session_factory)

    def update_for_student(std_id: str) -> dict[str, Any]:
        req = AssessmentMemoryUpdateRequest(
            student_id=std_id,
            topic=topic,
            assessment_questions=[
                AssessmentQuestionResult(question_id="q1", is_correct=True)
            ],
        )
        return service.update_memory(req).model_dump()

    with ThreadPoolExecutor(max_workers=num_students) as executor:
        futures = [
            executor.submit(update_for_student, std) for std in students
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == num_students

    # Verify each student has exactly 1 interaction
    with UnitOfWork(session_factory) as uow:
        for std_id in students:
            student = uow.identities.get_student(std_id)
            assert student is not None
            logs = uow.raw_interactions.get_student_interactions(student.student_id)
            assert len(logs) == 1


def test_restart_persistence_across_engine_reset(hardening_database):
    """Verify that persisted memory survives complete client and engine reset."""
    engine, _, _, _ = hardening_database
    student_id = f"restart_{uuid.uuid4().hex[:8]}"
    topic = "Physics"
    subtopic = "Kinematics"

    # Step 1: Write through client 1
    client_1 = TestClient(app)
    post_res = client_1.post(
        "/memory/update",
        json={
            "student_id": student_id,
            "topic": topic,
            "subtopic": subtopic,
            "assessment_questions": [
                {
                    "question_id": "q1",
                    "student_answer": "v = 10",
                    "expected_answer": "v = 10",
                    "is_correct": True,
                }
            ],
        },
        headers={"X-Service-Key": os.environ.get("SERVICE_KEY", "dev-service-key")},
    )
    assert post_res.status_code == 200
    posted_state = post_res.json()["learning_state"]

    # Step 2: Simulate process restart by rebuilding session factory and client
    settings = load_postgres_settings()
    fresh_engine = create_postgres_engine(settings)
    fresh_session_factory = create_session_factory(fresh_engine)
    set_session_factory(fresh_session_factory)
    client_2 = TestClient(app)

    try:
        # Query GET current through fresh instance
        current_res = client_2.get(
            f"/memory/{student_id}/current?topic={topic}&subtopic={subtopic}",
            headers={"X-Service-Key": os.environ.get("SERVICE_KEY", "dev-service-key")},
        )
        assert current_res.status_code == 200
        assert current_res.json()["learning_state"] == posted_state
        assert current_res.json()["student_id"] == student_id

        # Query GET history through fresh instance
        history_res = client_2.get(
            f"/memory/{student_id}/history?topic={topic}&subtopic={subtopic}",
            headers={"X-Service-Key": os.environ.get("SERVICE_KEY", "dev-service-key")},
        )
        assert history_res.status_code == 200
        assert len(history_res.json()) == 1
        assert history_res.json()[0]["learning_state"] == posted_state
    finally:
        set_session_factory(None)
        fresh_engine.dispose()


def test_sqlite_file_remains_untouched_by_hardening_suite(hardening_database):
    """Verify that student_memory.db is never accessed or generated."""
    if DEFAULT_DATABASE_PATH.exists():
        DEFAULT_DATABASE_PATH.unlink()

    client = TestClient(app)
    student_id = f"sqlite_check_{uuid.uuid4().hex[:8]}"
    client.post(
        "/memory/update",
        json={
            "student_id": student_id,
            "topic": "Chemistry",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )
    client.get(f"/memory/{student_id}/current?topic=Chemistry")
    client.get(f"/memory/{student_id}/history?topic=Chemistry")

    assert not DEFAULT_DATABASE_PATH.exists()
