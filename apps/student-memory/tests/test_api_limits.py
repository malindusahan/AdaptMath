"""Tests for API pagination, limits, and request-size safeguards."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.supporting_memory import LearningStateSnapshot
from src.services.student_context_service import StudentContextService


@pytest.fixture
def limits_test_setup(monkeypatch):
    """Configure SQLite in-memory database with realistic snapshots for pagination tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    student_ext = "limit_stud_01"
    topic = "Algebra"

    with factory() as session:
        skill_uuid = uuid.uuid4()
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_lim")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="algebra", display_name="Algebra")
        session.add_all([st, sess, skill])
        session.flush()

        # Seed 25 history snapshots
        for idx in range(25):
            snap = LearningStateSnapshot(
                snapshot_id=idx + 1,
                student_id=st.student_id,
                session_id=sess.session_id,
                canonical_skill_id=skill_uuid,
                learning_state="DEVELOPING",
                evidence_level="FULL_SKILL",
                evidence_strength="MEDIUM",
                behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                model_used=True,
                previous_interaction_count=idx,
                previous_skill_interaction_count=idx,
                recent_interaction_count=idx + 1,
                attempt_observation_count=1,
                hint_observation_count=0,
                response_time_observation_count=1,
                created_at=datetime(2026, 1, 1, 10, idx, tzinfo=timezone.utc),
            )
            session.add(snap)
        session.commit()

    # Wire services to shared factory
    from src.services.repair_outcome_service import RepairOutcomeService
    from src.services.student_memory_query_service import StudentMemoryQueryService
    query_svc = StudentMemoryQueryService(session_factory=factory)
    rep_svc = RepairOutcomeService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_student_memory_query_service", lambda: query_svc)
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: rep_svc)

    stud_svc = StudentContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: stud_svc)

    client = TestClient(app, raise_server_exceptions=False)
    return client, factory, student_ext, topic


def test_default_limit_applied_to_history(limits_test_setup):
    """Verify omitting limit on /history applies default limit of 20."""
    client, _, student_id, topic = limits_test_setup

    response = client.get(f"/memory/{student_id}/history?topic={topic}")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) == 20  # Total 25 seeded, default 20 returned


def test_custom_limit_and_pagination_slices(limits_test_setup):
    """Verify offset and limit paginate deterministically without overlapping or dropping records."""
    client, _, student_id, topic = limits_test_setup

    # Page 1: limit=10, offset=0
    r1 = client.get(f"/memory/{student_id}/history?topic={topic}&limit=10&offset=0")
    assert r1.status_code == 200
    page1 = r1.json()
    assert len(page1) == 10

    # Page 2: limit=10, offset=10
    r2 = client.get(f"/memory/{student_id}/history?topic={topic}&limit=10&offset=10")
    assert r2.status_code == 200
    page2 = r2.json()
    assert len(page2) == 10

    # Page 3: limit=10, offset=20 (remaining 5)
    r3 = client.get(f"/memory/{student_id}/history?topic={topic}&limit=10&offset=20")
    assert r3.status_code == 200
    page3 = r3.json()
    assert len(page3) == 5

    # Verify no overlapping snapshot timestamps between pages
    t_page1 = {item["created_at"] for item in page1}
    t_page2 = {item["created_at"] for item in page2}
    t_page3 = {item["created_at"] for item in page3}
    assert len(t_page1.intersection(t_page2)) == 0
    assert len(t_page2.intersection(t_page3)) == 0


def test_max_limit_accepted(limits_test_setup):
    """Verify limit=100 is accepted on /history."""
    client, _, student_id, topic = limits_test_setup

    response = client.get(f"/memory/{student_id}/history?topic={topic}&limit=100")
    assert response.status_code == 200
    assert len(response.json()) == 25


def test_limit_greater_than_max_rejected(limits_test_setup):
    """Verify limit=101 is rejected with standardized HTTP 422 VALIDATION_ERROR."""
    client, _, student_id, topic = limits_test_setup

    response = client.get(f"/memory/{student_id}/history?topic={topic}&limit=101")
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "limit" in data["details"][0]["field"]


def test_limit_less_than_one_rejected(limits_test_setup):
    """Verify limit=0 is rejected with standardized HTTP 422 VALIDATION_ERROR."""
    client, _, student_id, topic = limits_test_setup

    response = client.get(f"/memory/{student_id}/history?topic={topic}&limit=0")
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "limit" in data["details"][0]["field"]


def test_negative_offset_rejected(limits_test_setup):
    """Verify negative offset is rejected with standardized HTTP 422 VALIDATION_ERROR."""
    client, _, student_id, topic = limits_test_setup

    response = client.get(f"/memory/{student_id}/history?topic={topic}&offset=-1")
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "offset" in data["details"][0]["field"]


def test_oversized_post_payload_returns_413(monkeypatch, limits_test_setup):
    """Verify payload exceeding MEMORY_MAX_REQUEST_BYTES returns HTTP 413 PAYLOAD_TOO_LARGE."""
    client, _, student_id, topic = limits_test_setup
    monkeypatch.setenv("MEMORY_MAX_REQUEST_BYTES", "500")  # Set small limit of 500 bytes

    # Create oversized payload (> 500 bytes)
    large_notes = "X" * 1000
    payload = {
        "student_id": student_id,
        "session_id": "sess_1",
        "skill_id": str(uuid.uuid4()),
        "repair_action": "step_hint",
        "outcome": "RESOLVED",
        "notes": large_notes,
    }

    response = client.post(
        "/memory/repair-outcome",
        json=payload,
        headers={"X-Request-ID": "req-overflow-1"},
    )
    assert response.status_code == 413

    data = response.json()
    assert data["error_code"] == "PAYLOAD_TOO_LARGE"
    assert "exceeds limit" in data["message"].lower() or "too large" in data["message"].lower()
    assert data["request_id"] == "req-overflow-1"

    # Verify no raw body is echoed
    assert large_notes not in response.text


def test_normal_post_payload_under_size_limit_accepted(monkeypatch, limits_test_setup):
    """Verify standard payload within size limit is accepted normally."""
    client, _, student_id, topic = limits_test_setup
    monkeypatch.setenv("MEMORY_MAX_REQUEST_BYTES", "1048576")  # 1 MB

    payload = {
        "student_id": student_id,
        "session_id": "sess_lim",
        "skill_id": str(uuid.uuid4()),
        "repair_action": "step_hint",
        "outcome": "RESOLVED",
        "score": 1.0,
    }

    # Posting repair outcome with unknown skill should return 404 (not 413)
    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code in (201, 400, 404)
    assert response.status_code != 413
