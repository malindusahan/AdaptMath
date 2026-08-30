"""End-to-end API tests for POST /memory/repair-outcome endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import RepairOutcome
from src.services.repair_outcome_service import RepairOutcomeService


@pytest.fixture
def test_setup(monkeypatch):
    """Create shared in-memory SQLite database and configure FastAPI test client."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    service = RepairOutcomeService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_valid_repair_outcome_with_interaction(test_setup):
    """Verify recording a valid repair outcome linked to an interaction succeeds."""
    client, service, factory = test_setup

    student_ext = "rep_stud_1"
    session_ext = "rep_sess_1"
    skill_uuid = uuid.uuid4()
    inter_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: geometry :: pythagoras", display_name="Pythagoras")
        session.add_all([st, sess, skill])
        session.flush()

        inter = InteractionLog(
            interaction_id=inter_uuid,
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="ext_inter_rep_1",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
        )
        session.add(inter)
        session.commit()

    payload = {
        "student_id": student_ext,
        "session_id": session_ext,
        "skill_id": str(skill_uuid),
        "interaction_id": str(inter_uuid),
        "repair_action": "inverse_operation_prompt",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Student applied square root correctly.",
    }

    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["session_id"] == session_ext
    assert data["canonical_skill_id"] == str(skill_uuid)
    assert data["interaction_id"] == str(inter_uuid)
    assert data["repair_action"] == "inverse_operation_prompt"
    assert data["outcome"] == "RESOLVED"
    assert data["score"] == 1.0
    assert data["notes"] == "Student applied square root correctly."

    with factory() as session:
        count = session.scalar(select(func.count(RepairOutcome.repair_outcome_id)))
        assert count == 1


def test_valid_repair_outcome_without_interaction(test_setup):
    """Verify recording a repair outcome without interaction_id succeeds."""
    client, service, factory = test_setup

    student_ext = "rep_stud_2"
    session_ext = "rep_sess_2"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: fractions :: simplify", display_name="Simplify")
        session.add_all([st, sess, skill])
        session.commit()

    payload = {
        "student_id": student_ext,
        "session_id": session_ext,
        "skill_id": str(skill_uuid),
        "repair_action": "visual_fraction_bar",
        "outcome": "PARTIALLY_RESOLVED",
        "score": 0.5,
    }

    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["interaction_id"] is None
    assert data["repair_action"] == "visual_fraction_bar"
    assert data["outcome"] == "PARTIALLY_RESOLVED"
    assert data["score"] == 0.5
    assert data["notes"] is None


def test_invalid_student_or_session_ownership(test_setup):
    """Verify mismatch between session and student returns HTTP 400."""
    client, service, factory = test_setup

    skill_uuid = uuid.uuid4()
    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id="rep_user_A")
        stB = Student(student_id=uuid.uuid4(), external_student_id="rep_user_B")
        sessA = LearningSession(session_id=uuid.uuid4(), student_id=stA.student_id, external_session_id="sess_A")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: algebra", display_name="Algebra")
        session.add_all([stA, stB, sessA, skill])
        session.commit()

    # Try assigning sessA to stB
    payload = {
        "student_id": "rep_user_B",
        "session_id": "sess_A",
        "skill_id": str(skill_uuid),
        "repair_action": "hint",
        "outcome": "RESOLVED",
    }

    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 400
    assert "does not belong to student" in response.json()["detail"]


def test_invalid_skill_returns_404(test_setup):
    """Verify non-existent skill returns HTTP 404."""
    client, service, factory = test_setup

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="rep_user_sk")
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_sk")
        session.add_all([st, sess])
        session.commit()

    payload = {
        "student_id": "rep_user_sk",
        "session_id": "sess_sk",
        "skill_id": "non_existent_skill_uuid",
        "repair_action": "hint",
        "outcome": "RESOLVED",
    }

    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 404
    assert "Canonical skill" in response.json()["detail"]


def test_invalid_interaction_ownership(test_setup):
    """Verify interaction belonging to another session/student returns HTTP 400."""
    client, service, factory = test_setup

    skill_uuid = uuid.uuid4()
    inter_uuid = uuid.uuid4()
    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id="stud_A")
        stB = Student(student_id=uuid.uuid4(), external_student_id="stud_B")
        sessA = LearningSession(session_id=uuid.uuid4(), student_id=stA.student_id, external_session_id="sess_A")
        sessB = LearningSession(session_id=uuid.uuid4(), student_id=stB.student_id, external_session_id="sess_B")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: general", display_name="General")
        session.add_all([stA, stB, sessA, sessB, skill])
        session.flush()

        interA = InteractionLog(
            interaction_id=inter_uuid,
            student_id=stA.student_id,
            session_id=sessA.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_A",
            is_correct=False,
            attempt_count=1,
            hint_count=0,
        )
        session.add(interA)
        session.commit()

    payload = {
        "student_id": "stud_B",
        "session_id": "sess_B",
        "skill_id": str(skill_uuid),
        "interaction_id": str(inter_uuid),
        "repair_action": "hint",
        "outcome": "RESOLVED",
    }

    response = client.post("/memory/repair-outcome", json=payload)
    assert response.status_code == 400
    assert "does not belong" in response.json()["detail"]


def test_score_boundaries():
    """Verify score < 0.0 or score > 1.0 returns HTTP 422."""
    client = TestClient(app)

    r1 = client.post(
        "/memory/repair-outcome",
        json={
            "student_id": "s1",
            "session_id": "sess1",
            "skill_id": "sk1",
            "repair_action": "hint",
            "outcome": "RESOLVED",
            "score": -0.1,
        },
    )
    assert r1.status_code == 422

    r2 = client.post(
        "/memory/repair-outcome",
        json={
            "student_id": "s1",
            "session_id": "sess1",
            "skill_id": "sk1",
            "repair_action": "hint",
            "outcome": "RESOLVED",
            "score": 1.5,
        },
    )
    assert r2.status_code == 422


def test_duplicate_retry_behavior(test_setup):
    """Verify repeating the exact same repair outcome request returns existing record idempotently."""
    client, service, factory = test_setup

    student_ext = "rep_dup_stud"
    session_ext = "rep_dup_sess"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: stats", display_name="Stats")
        session.add_all([st, sess, skill])
        session.commit()

    payload = {
        "student_id": student_ext,
        "session_id": session_ext,
        "skill_id": str(skill_uuid),
        "repair_action": "worked_example_step",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "First attempt at repair.",
    }

    # 1st call -> creates row
    r1 = client.post("/memory/repair-outcome", json=payload)
    assert r1.status_code == 201
    rep_id_1 = r1.json()["repair_outcome_id"]

    # 2nd call (retry) -> returns same row
    r2 = client.post("/memory/repair-outcome", json=payload)
    assert r2.status_code == 201
    rep_id_2 = r2.json()["repair_outcome_id"]

    assert rep_id_1 == rep_id_2

    with factory() as session:
        count = session.scalar(select(func.count(RepairOutcome.repair_outcome_id)))
        assert count == 1


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.record_repair_outcome.side_effect = RuntimeError("DB write error")
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: failing_service)

    client = TestClient(app)
    response = client.post(
        "/memory/repair-outcome",
        json={
            "student_id": "s1",
            "session_id": "sess1",
            "skill_id": "sk1",
            "repair_action": "hint",
            "outcome": "RESOLVED",
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to record repair outcome."
