"""End-to-end API tests for GET /memory/{student_id}/support-preference endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock
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
from src.database.models.supporting_memory import RepairOutcome
from src.services.support_preference_service import SupportPreferenceService


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

    service = SupportPreferenceService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_no_repair_history_returns_insufficient_evidence(test_setup):
    """Verify fresh student without repair logs returns INSUFFICIENT_EVIDENCE safely."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_supp_stud/support-preference")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_supp_stud"
    assert data["preferred_support_style"] is None
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert data["evidence_count"] is None
    assert data["success_rate"] is None
    assert data["strategies"] == []


def test_insufficient_evidence_under_threshold(test_setup):
    """Verify strategies with < 3 observations do not trigger preference inference."""
    client, service, factory = test_setup

    student_ext = "stud_few_repairs"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_1")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: trig", display_name="Trig")
        session.add_all([st, sess, skill])
        session.flush()

        # Only 2 observations for worked_example
        r1 = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="worked_example",
            outcome="RESOLVED",
            score=1.0,
        )
        r2 = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="worked_example",
            outcome="RESOLVED",
            score=1.0,
        )
        session.add_all([r1, r2])
        session.commit()

    response = client.get(f"/memory/{student_ext}/support-preference")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert data["preferred_support_style"] is None
    assert len(data["strategies"]) == 1
    assert data["strategies"][0]["repair_action"] == "worked_example"
    assert data["strategies"][0]["observation_count"] == 2
    assert data["strategies"][0]["successful_count"] == 2


def test_clear_preferred_strategy(test_setup):
    """Verify strategy with >= 3 observations and highest success rate is inferred as preferred."""
    client, service, factory = test_setup

    student_ext = "stud_pref_clear"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_1")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: algebra", display_name="Algebra")
        session.add_all([st, sess, skill])
        session.flush()

        # 4 observations for worked_example (3 resolved, 1 partial) -> success_rate = 0.75
        repairs = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="worked_example", outcome="RESOLVED", score=1.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="worked_example", outcome="RESOLVED", score=1.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="worked_example", outcome="RESOLVED", score=0.9),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="worked_example", outcome="PARTIALLY_RESOLVED", score=0.5),
            # 3 observations for visual_diagram (1 resolved, 2 failed) -> success_rate = 0.33
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="visual_diagram", outcome="RESOLVED", score=1.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="visual_diagram", outcome="UNRESOLVED", score=0.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="visual_diagram", outcome="UNRESOLVED", score=0.0),
        ]
        session.add_all(repairs)
        session.commit()

    response = client.get(f"/memory/{student_ext}/support-preference")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "SUPPORTED_BY_HISTORY"
    assert data["preferred_support_style"] == "worked_example"
    assert data["evidence_count"] == 4
    assert data["success_rate"] == 0.75

    assert len(data["strategies"]) == 2
    assert data["strategies"][0]["repair_action"] == "worked_example"
    assert data["strategies"][0]["success_rate"] == 0.75
    assert data["strategies"][0]["average_score"] == 0.85
    assert data["strategies"][1]["repair_action"] == "visual_diagram"
    assert data["strategies"][1]["success_rate"] == 0.33


def test_tie_breaking_by_score_and_count(test_setup):
    """Verify tie between two strategies with equal success rate breaks deterministically by score/count."""
    client, service, factory = test_setup

    student_ext = "stud_pref_tie"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_1")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: geometry", display_name="Geometry")
        session.add_all([st, sess, skill])
        session.flush()

        # Strategy A: 3 observations, 3 resolved, avg_score = 1.0
        rep_A = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="formula_prompt", outcome="RESOLVED", score=1.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="formula_prompt", outcome="RESOLVED", score=1.0),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="formula_prompt", outcome="RESOLVED", score=1.0),
        ]
        # Strategy B: 3 observations, 3 resolved, avg_score = 0.8
        rep_B = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="hint_sequence", outcome="RESOLVED", score=0.8),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="hint_sequence", outcome="RESOLVED", score=0.8),
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill_uuid, repair_action="hint_sequence", outcome="RESOLVED", score=0.8),
        ]
        session.add_all(rep_A + rep_B)
        session.commit()

    response = client.get(f"/memory/{student_ext}/support-preference")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "SUPPORTED_BY_HISTORY"
    assert data["preferred_support_style"] == "formula_prompt"


def test_skill_specific_preference(test_setup):
    """Verify skill_id filters repairs to that specific skill."""
    client, service, factory = test_setup

    student_ext = "stud_pref_skill"
    skill1_uuid = uuid.uuid4()
    skill2_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id="sess_1")
        skill1 = CanonicalSkill(skill_id=skill1_uuid, canonical_name="math :: skill_1", display_name="Skill 1")
        skill2 = CanonicalSkill(skill_id=skill2_uuid, canonical_name="math :: skill_2", display_name="Skill 2")
        session.add_all([st, sess, skill1, skill2])
        session.flush()

        # Skill 1: 3 visual_aid repairs
        rep1 = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill1_uuid, repair_action="visual_aid", outcome="RESOLVED", score=1.0)
            for _ in range(3)
        ]
        # Skill 2: 3 text_prompt repairs
        rep2 = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=st.student_id, session_id=sess.session_id, canonical_skill_id=skill2_uuid, repair_action="text_prompt", outcome="RESOLVED", score=1.0)
            for _ in range(3)
        ]
        session.add_all(rep1 + rep2)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/support-preference?skill_id={skill1_uuid}")
    assert r1.status_code == 200
    assert r1.json()["preferred_support_style"] == "visual_aid"

    r2 = client.get(f"/memory/{student_ext}/support-preference?skill_id={skill2_uuid}")
    assert r2.status_code == 200
    assert r2.json()["preferred_support_style"] == "text_prompt"


def test_student_isolation(test_setup):
    """Verify distinct students do not share repair outcome statistics."""
    client, service, factory = test_setup

    sA_ext = "pref_iso_A"
    sB_ext = "pref_iso_B"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        sessA = LearningSession(session_id=uuid.uuid4(), student_id=stA.student_id, external_session_id="sess_A")
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: isol", display_name="Isol")
        session.add_all([stA, stB, sessA, skill])
        session.flush()

        repA = [
            RepairOutcome(repair_outcome_id=uuid.uuid4(), student_id=stA.student_id, session_id=sessA.session_id, canonical_skill_id=skill_uuid, repair_action="action_A", outcome="RESOLVED", score=1.0)
            for _ in range(4)
        ]
        session.add_all(repA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/support-preference")
    rB = client.get(f"/memory/{sB_ext}/support-preference")

    assert rA.status_code == 200
    assert rA.json()["status"] == "SUPPORTED_BY_HISTORY"
    assert rA.json()["preferred_support_style"] == "action_A"

    assert rB.status_code == 200
    assert rB.json()["status"] == "INSUFFICIENT_EVIDENCE"
    assert rB.json()["preferred_support_style"] is None


def test_unknown_skill_returns_404(test_setup):
    """Verify querying support preference with non-existent skill returns HTTP 404."""
    client, service, factory = test_setup

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="stud_uk_sk")
        session.add(st)
        session.commit()

    response = client.get("/memory/stud_uk_sk/support-preference?skill_id=unknown_skill_xyz")
    assert response.status_code == 404
    assert "Canonical skill" in response.json()["detail"]


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_support_preference.side_effect = RuntimeError("DB error")
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/stud_err/support-preference")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to evaluate support preference."
