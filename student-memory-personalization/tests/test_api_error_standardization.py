"""Tests for API Error Standardization and Request Tracing Telemetry."""

from __future__ import annotations

from unittest.mock import MagicMock
import uuid
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.memory_projection import ShortTermMemory
from src.services.student_context_service import StudentContextService


@pytest.fixture
def api_test_client(monkeypatch):
    """Create test client with in-memory SQLite backend."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    from src.services.fapr_context_service import FAPRContextService
    from src.services.meta_signal_service import MetaSignalService
    from src.services.planner_context_service import PlannerContextService
    from src.services.repair_outcome_service import RepairOutcomeService
    from src.services.student_context_service import StudentContextService
    from src.services.support_preference_service import SupportPreferenceService
    from src.services.tutor_context_service import TutorContextService

    stud_svc = StudentContextService(session_factory=factory)
    tutor_svc = TutorContextService(session_factory=factory, student_context_service=stud_svc)
    planner_svc = PlannerContextService(session_factory=factory, student_context_service=stud_svc)
    fapr_svc = FAPRContextService(session_factory=factory, student_context_service=stud_svc)
    rep_svc = RepairOutcomeService(session_factory=factory)
    supp_svc = SupportPreferenceService(session_factory=factory)
    meta_svc = MetaSignalService(session_factory=factory, student_context_service=stud_svc)

    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: stud_svc)
    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: tutor_svc)
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: planner_svc)
    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: fapr_svc)
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: rep_svc)
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: supp_svc)
    monkeypatch.setattr(memory_routes, "get_meta_signal_service", lambda: meta_svc)

    client = TestClient(app, raise_server_exceptions=False)
    return client, factory


def test_422_validation_error_envelope(api_test_client):
    """Verify HTTP 422 errors return standardized VALIDATION_ERROR envelope with granular details."""
    client, _ = api_test_client

    # Send invalid limit (0 is < 1)
    response = client.get("/memory/stud1/context?limit=0")
    assert response.status_code == 422

    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert data["message"] == "Request validation failed."
    assert "request_id" in data and len(data["request_id"]) > 0
    assert "timestamp" in data
    assert len(data["details"]) > 0
    assert any("limit" in d["field"] for d in data["details"])

    # Headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers
    assert response.headers["X-Request-ID"] == data["request_id"]


def test_404_not_found_envelope(api_test_client):
    """Verify HTTP 404 errors return standardized NOT_FOUND envelope."""
    client, factory = api_test_client

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="stud1")
        session.add(st)
        session.commit()

    response = client.get("/memory/stud1/support-preference?skill_id=non_existent_skill")
    assert response.status_code == 404

    data = response.json()
    assert data["error_code"] == "NOT_FOUND"
    assert "Canonical skill" in data["message"]
    assert data["detail"] == data["message"]
    assert "request_id" in data
    assert "timestamp" in data


def test_400_bad_request_envelope(api_test_client):
    """Verify HTTP 400 errors return standardized BAD_REQUEST envelope."""
    client, factory = api_test_client

    skill_uuid = uuid.uuid4()
    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id="stud_A")
        stB = Student(student_id=uuid.uuid4(), external_student_id="stud_B")
        sessA = LearningSession(session_id=uuid.uuid4(), student_id=stA.student_id, external_session_id="sess_A")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: general", display_name="General")
        session.add_all([stA, stB, sessA, skill])
        session.commit()

    # Attempt to post repair outcome for student B using session belonging to student A
    payload = {
        "student_id": "stud_B",
        "session_id": "sess_A",
        "skill_id": str(skill_uuid),
        "repair_action": "hint",
        "outcome": "RESOLVED",
    }
    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["error_code"] == "BAD_REQUEST"
    assert "does not belong to student" in data["message"]
    assert "request_id" in data
    assert "timestamp" in data


def test_500_internal_error_envelope_sanitization(monkeypatch):
    """Verify 500 exceptions return sanitized INTERNAL_ERROR envelope without leaking secrets."""
    failing_service = MagicMock()
    # Mock error containing sensitive strings
    failing_service.get_student_context.side_effect = RuntimeError(
        "FATAL: connection to postgresql://postgres:super_secret_password@db.example.com:5432 failed; SELECT * FROM credentials"
    )
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: failing_service)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/memory/stud_err/context")
    assert response.status_code == 500

    data = response.json()
    assert data["error_code"] == "INTERNAL_ERROR"
    assert "request_id" in data
    assert "timestamp" in data

    # Verify complete sanitization
    raw_text = response.text
    assert "postgresql://" not in raw_text
    assert "super_secret_password" not in raw_text
    assert "SELECT *" not in raw_text
    assert "Traceback" not in raw_text


def test_incoming_request_id_preserved(api_test_client):
    """Verify incoming X-Request-ID header is propagated and returned in error envelope and headers."""
    client, _ = api_test_client
    custom_req_id = "custom-trace-uuid-12345"

    response = client.get(
        "/memory/stud1/context?limit=0",
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == custom_req_id

    data = response.json()
    assert data["request_id"] == custom_req_id


def test_telemetry_headers_present_on_successful_responses(api_test_client):
    """Verify X-Request-ID and X-Response-Time-MS headers are present on valid 200 responses."""
    client, factory = api_test_client

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="stud_ok_1")
        session.add(st)
        session.commit()

    response = client.get("/memory/stud_ok_1/context")
    assert response.status_code == 200

    # Headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

    # Latency is a float formatted as string
    ms_val = float(response.headers["X-Response-Time-MS"])
    assert ms_val >= 0.0

    # Existing response schema is preserved
    data = response.json()
    assert data["student_id"] == "stud_ok_1"
    assert "short_term_memory" in data
    assert "long_term_memory" in data
