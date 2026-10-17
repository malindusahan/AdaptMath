"""Comprehensive integration and unit tests for the PostgreSQL-backed FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text

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
from src.schemas.interaction import MemoryUpdateResponse
from src.services.memory_update_service import MemoryUpdateServiceError
from src.services.skill_context_resolver import (
    SkillContextResolver,
    format_skill_display_name,
    normalize_skill_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA
client = TestClient(app)


# =============================================================================
# Pure Unit Tests (Deterministic without external DB)
# =============================================================================

def test_skill_context_resolver_normalization():
    """Verify deterministic skill normalization and display formatting."""
    assert normalize_skill_name("  Algebra  ", " Linear  equations ") == "algebra :: linear equations"
    assert normalize_skill_name("Calculus", None) == "calculus"
    assert normalize_skill_name("Calculus", "   ") == "calculus"

    assert format_skill_display_name("Algebra", "Linear equations") == "Algebra / Linear equations"
    assert format_skill_display_name("Calculus", None) == "Calculus"
    assert format_skill_display_name("Calculus", "   ") == "Calculus"


def test_api_post_update_validation_errors():
    """Verify 422 response for invalid request structures."""
    # Blank student_id
    res = client.post(
        "/memory/update",
        json={
            "student_id": "   ",
            "topic": "Algebra",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )
    assert res.status_code == 422

    # Blank topic
    res = client.post(
        "/memory/update",
        json={
            "student_id": "s1",
            "topic": "   ",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )
    assert res.status_code == 422

    # Empty assessment questions
    res = client.post(
        "/memory/update",
        json={
            "student_id": "s1",
            "topic": "Algebra",
            "assessment_questions": [],
        },
    )
    assert res.status_code == 422


def test_api_get_endpoints_validation_errors():
    """Verify 422 for blank parameters on GET /current and GET /history."""
    # Blank student_id on current
    res = client.get("/memory/%20%20/current?topic=Algebra")
    assert res.status_code == 422

    # Blank topic on current
    res = client.get("/memory/s1/current?topic=%20%20")
    assert res.status_code == 422

    # Blank student_id on history
    res = client.get("/memory/%20%20/history?topic=Algebra")
    assert res.status_code == 422

    # Blank topic on history
    res = client.get("/memory/s1/history?topic=%20%20")
    assert res.status_code == 422


def test_api_sanitized_500_on_internal_service_failure(monkeypatch):
    """Verify internal exception details are not exposed in 500 responses."""
    def failing_update(request, **kwargs):
        raise MemoryUpdateServiceError("Secret database credentials / internal trace")

    monkeypatch.setattr(memory_routes, "update_student_memory", failing_update)

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
    assert "Secret" not in res.text


# =============================================================================
# Live PostgreSQL Pipeline Integration Tests
# =============================================================================

@pytest.fixture(scope="module")
def api_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for API integration tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)
    test_client = TestClient(app)

    try:
        yield engine, session_factory, test_client
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()


def test_post_update_and_get_current_and_history_workflow(api_database):
    """Verify complete POST update, GET current, and GET history API flow against PostgreSQL."""
    engine, session_factory, test_client = api_database
    student_id = f"api_s1_{uuid.uuid4().hex[:8]}"
    topic = "Algebra"
    subtopic = "Linear equations"

    payload = {
        "student_id": student_id,
        "topic": topic,
        "subtopic": subtopic,
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 2x = 10",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "hint_total": 2,
                "response_time_ms": 4500.0,
            },
            {
                "question_id": "q2",
                "question": "Solve x - 3 = 7",
                "student_answer": "x = 9",
                "expected_answer": "x = 10",
                "is_correct": False,
                "identified_error": "addition arithmetic error",
                "attempt_count": 2,
                "hint_count": 1,
                "hint_total": 2,
                "response_time_ms": 12000.0,
            },
        ],
        "identified_errors": ["addition arithmetic error"],
        "overall_feedback": "Review linear addition.",
    }

    # 1. POST /memory/update
    post_res = test_client.post("/memory/update", json=payload)
    assert post_res.status_code == 200
    post_data = post_res.json()

    assert post_data["student_id"] == student_id
    assert post_data["topic"] == topic
    assert post_data["subtopic"] == subtopic
    assert post_data["memory_updated"] is True
    assert post_data["learning_state"] in (
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
        "UNAVAILABLE",
    )
    assert post_data["misconception_count"] == 1
    assessment_id = post_data["assessment_id"]
    snapshot_id = post_data["snapshot_id"]

    # Verify PostgreSQL records exist directly in database
    with UnitOfWork(session_factory) as uow:
        student = uow.identities.get_student(student_id)
        assert student is not None
        skill = uow.identities.get_skill(topic, subtopic)
        assert skill is not None

        # Verify raw interactions created
        raw_logs = uow.raw_interactions.get_student_interactions(
            student.student_id
        )
        assert len(raw_logs) == 2

        # Verify STM, LTM, Concept projections
        ltm = uow.projections.get_long_term_memory(student.student_id)
        assert ltm is not None
        assert ltm.interaction_count == 2
        assert ltm.correct_count == 1
        assert ltm.incorrect_count == 1

        concept = uow.projections.get_concept_memory(
            student.student_id, skill.skill_id
        )
        assert concept is not None
        assert concept.interaction_count == 2

        # Verify misconception
        misconceptions = uow.supporting.get_misconceptions(
            student.student_id, skill.skill_id
        )
        assert len(misconceptions) == 1
        assert misconceptions[0].occurrence_count == 1

    # 2. GET /memory/{student_id}/current
    current_res = test_client.get(
        f"/memory/{student_id}/current?topic={topic}&subtopic={subtopic}"
    )
    assert current_res.status_code == 200
    current_data = current_res.json()
    assert current_data["student_id"] == student_id
    assert current_data["topic"] == topic
    assert current_data["subtopic"] == subtopic
    assert current_data["learning_state"] == post_data["learning_state"]
    assert current_data["last_assessment_id"] == assessment_id
    assert current_data["last_snapshot_id"] == snapshot_id

    # 3. GET /memory/{student_id}/history
    history_res = test_client.get(
        f"/memory/{student_id}/history?topic={topic}&subtopic={subtopic}"
    )
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert len(history_data) == 1
    assert history_data[0]["snapshot_id"] == snapshot_id
    assert history_data[0]["assessment_id"] == assessment_id
    assert history_data[0]["learning_state"] == post_data["learning_state"]


def test_second_post_accumulates_history_and_increments_misconceptions(api_database):
    """Verify multiple assessments accumulate history chronologically and increment error counts."""
    _, session_factory, test_client = api_database
    student_id = f"api_multi_{uuid.uuid4().hex[:8]}"
    topic = "Algebra"
    subtopic = "Linear equations"

    # Assessment 1
    payload_1 = {
        "student_id": student_id,
        "topic": topic,
        "subtopic": subtopic,
        "assessment_questions": [
            {
                "question_id": "q1",
                "is_correct": False,
                "identified_error": "algebra sign error",
            }
        ],
    }
    res_1 = test_client.post("/memory/update", json=payload_1)
    assert res_1.status_code == 200
    snap_1 = res_1.json()["snapshot_id"]

    # Assessment 2
    payload_2 = {
        "student_id": student_id,
        "topic": topic,
        "subtopic": subtopic,
        "assessment_questions": [
            {
                "question_id": "q2",
                "is_correct": False,
                "identified_error": "ALGEBRA SIGN ERROR",  # Normalized exact match
            }
        ],
    }
    res_2 = test_client.post("/memory/update", json=payload_2)
    assert res_2.status_code == 200
    snap_2 = res_2.json()["snapshot_id"]
    assert snap_2 > snap_1
    assert res_2.json()["misconception_count"] == 1

    # Check history contains both chronologically
    history_res = test_client.get(
        f"/memory/{student_id}/history?topic={topic}&subtopic={subtopic}"
    )
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert len(history_data) == 2
    assert history_data[0]["snapshot_id"] == snap_1
    assert history_data[1]["snapshot_id"] == snap_2

    # Check misconception count = 2 in PostgreSQL
    with UnitOfWork(session_factory) as uow:
        student = uow.identities.get_student(student_id)
        skill = uow.identities.get_skill(topic, subtopic)
        misconceptions = uow.supporting.get_misconceptions(
            student.student_id, skill.skill_id
        )
        assert len(misconceptions) == 1
        assert misconceptions[0].occurrence_count == 2


def test_student_and_topic_isolation(api_database):
    """Verify students and topics do not cross-contaminate in API queries."""
    _, _, test_client = api_database
    s1 = f"iso_s1_{uuid.uuid4().hex[:8]}"
    s2 = f"iso_s2_{uuid.uuid4().hex[:8]}"

    # S1 in Algebra
    test_client.post(
        "/memory/update",
        json={
            "student_id": s1,
            "topic": "Algebra",
            "subtopic": "Linear",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )

    # S2 in Geometry
    test_client.post(
        "/memory/update",
        json={
            "student_id": s2,
            "topic": "Geometry",
            "subtopic": "Angles",
            "assessment_questions": [{"question_id": "q1", "is_correct": False}],
        },
    )

    # S1 should have no memory in Geometry
    res = test_client.get(f"/memory/{s1}/current?topic=Geometry&subtopic=Angles")
    assert res.status_code == 404

    # S2 should have no memory in Algebra
    res = test_client.get(f"/memory/{s2}/current?topic=Algebra&subtopic=Linear")
    assert res.status_code == 404

    # S1 history in Geometry is empty list
    hist_res = test_client.get(f"/memory/{s1}/history?topic=Geometry&subtopic=Angles")
    assert hist_res.status_code == 200
    assert hist_res.json() == []


def test_none_subtopic_round_trip(api_database):
    """Verify topic-only assessments (None subtopic) function cleanly."""
    _, _, test_client = api_database
    student_id = f"none_sub_{uuid.uuid4().hex[:8]}"
    topic = "Calculus"

    payload = {
        "student_id": student_id,
        "topic": topic,
        "subtopic": None,
        "assessment_questions": [
            {"question_id": "q1", "is_correct": True}
        ],
    }

    res = test_client.post("/memory/update", json=payload)
    assert res.status_code == 200
    assert res.json()["subtopic"] is None

    # Query current without subtopic param
    curr_res = test_client.get(f"/memory/{student_id}/current?topic={topic}")
    assert curr_res.status_code == 200
    assert curr_res.json()["subtopic"] is None

    # Query history without subtopic param
    hist_res = test_client.get(f"/memory/{student_id}/history?topic={topic}")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) == 1
    assert hist_res.json()[0]["subtopic"] is None


def test_api_unknown_resources_and_atomic_rollback(api_database):
    """Verify 404 for unknown memory, 200 [] for unknown history, and rollback on error."""
    _, session_factory, test_client = api_database

    # 404 on unknown current memory
    res = test_client.get("/memory/nonexistent_student_xyz/current?topic=Algebra")
    assert res.status_code == 404
    assert res.json()["detail"] == "Student memory not found."

    # 200 [] on unknown history
    res = test_client.get("/memory/nonexistent_student_xyz/history?topic=Algebra")
    assert res.status_code == 200
    assert res.json() == []

    # Duplicate question_ids causes 500 and complete transaction rollback
    student_id = f"rollback_std_{uuid.uuid4().hex[:8]}"
    res = test_client.post(
        "/memory/update",
        json={
            "student_id": student_id,
            "topic": "Algebra",
            "assessment_questions": [
                {"question_id": "dup_q1", "is_correct": True},
                {"question_id": "dup_q1", "is_correct": False},
            ],
        },
    )
    assert res.status_code == 500
    assert res.json()["detail"] == "Memory update failed."

    # Verify no student or raw interactions were saved in DB
    with UnitOfWork(session_factory) as uow:
        student = uow.identities.get_student(student_id)
        assert student is None


def test_sqlite_database_is_never_created_by_api(api_database):
    """Verify that PostgreSQL API operations do NOT create or access student_memory.db."""
    _, _, test_client = api_database

    # Ensure SQLite file does not exist before test
    if DEFAULT_DATABASE_PATH.exists():
        DEFAULT_DATABASE_PATH.unlink()

    student_id = f"sqlite_check_{uuid.uuid4().hex[:8]}"
    test_client.post(
        "/memory/update",
        json={
            "student_id": student_id,
            "topic": "Physics",
            "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        },
    )

    test_client.get(f"/memory/{student_id}/current?topic=Physics")
    test_client.get(f"/memory/{student_id}/history?topic=Physics")

    # Assert SQLite database file remains non-existent
    assert not DEFAULT_DATABASE_PATH.exists()
