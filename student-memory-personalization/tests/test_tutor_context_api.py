"""End-to-end API tests for GET /memory/{student_id}/tutor-context endpoint."""

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
from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import CurrentLearningState, RepairOutcome, StudentMisconception
from src.services.tutor_context_service import TutorContextService


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

    service = TutorContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_no_history_student_returns_safe_defaults(test_setup):
    """Verify non-existent student or brand new student returns clean, safe default fields."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_stud_999/tutor-context")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_stud_999"
    assert data["session_id"] is None
    assert data["skill_id"] is None
    assert data["current_learning_state"] is None
    assert data["evidence_strength"] is None
    assert data["behavioural_coverage"] is None
    assert data["recent_accuracy"] is None
    assert data["recent_correct_count"] == 0
    assert data["recent_incorrect_count"] == 0
    assert data["attempt_evidence"]["observation_count"] == 0
    assert data["attempt_evidence"]["average_value"] is None
    assert data["hint_evidence"]["observation_count"] == 0
    assert data["response_time_evidence"]["observation_count"] == 0
    assert data["misconceptions"] == []
    assert data["recent_interactions"] == []
    assert data["recent_repairs"] == []


def test_full_tutor_context(test_setup):
    """Verify student with full memory history returns complete, structured tutor context."""
    client, service, factory = test_setup

    student_ext = "tutor_stud_full"
    session_ext = "tutor_sess_full"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(
            skill_id=skill_uuid,
            canonical_name="math :: geometry :: pythagorean theorem",
            display_name="Pythagorean Theorem",
        )
        session.add_all([st, sess, skill])
        session.flush()

        # Concept Memory
        concept = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            interaction_count=10,
            correct_count=7,
            incorrect_count=3,
            attempt_observation_count=10,
            attempt_sum=15,
            hint_observation_count=10,
            hint_sum=5,
            response_time_observation_count=10,
            response_time_sum_ms=50000.0,
            accuracy=0.7,
        )
        # Learning State
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="DEVELOPING",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=10,
            attempt_observation_count=10,
            hint_observation_count=10,
            response_time_observation_count=10,
            last_snapshot_id=1,
        )
        # Misconception
        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="forgot to take square root of c squared",
            display_error="Forgot to square root c^2 at the end of calculation",
            occurrence_count=2,
        )
        # Raw Interaction
        inter = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_pyth_01",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
            response_time_ms=6200.0,
        )
        # Repair Outcome
        repair = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="formula_step_breakdown",
            outcome="RESOLVED",
            score=0.9,
            notes="Student successfully executed square root step.",
        )

        session.add_all([concept, state, misc, inter, repair])
        session.commit()

    url = f"/memory/{student_ext}/tutor-context?session_id={session_ext}&skill_id={skill_uuid}&limit=5"
    response = client.get(url)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["session_id"] == session_ext
    assert data["skill_id"] == str(skill_uuid)

    assert data["current_learning_state"] == "DEVELOPING"
    assert data["evidence_strength"] == "MEDIUM"
    assert data["behavioural_coverage"] == "FULL_BEHAVIOURAL_COVERAGE"
    assert data["recent_accuracy"] == 0.7
    assert data["recent_correct_count"] == 7
    assert data["recent_incorrect_count"] == 3

    assert data["attempt_evidence"]["observation_count"] == 10
    assert data["attempt_evidence"]["average_value"] == 1.5
    assert data["hint_evidence"]["observation_count"] == 10
    assert data["hint_evidence"]["average_value"] == 0.5
    assert data["response_time_evidence"]["observation_count"] == 10
    assert data["response_time_evidence"]["average_value"] == 5000.0

    assert len(data["misconceptions"]) == 1
    assert data["misconceptions"][0]["display_error"] == "Forgot to square root c^2 at the end of calculation"
    assert len(data["recent_interactions"]) == 1
    assert data["recent_interactions"][0]["is_correct"] is False
    assert len(data["recent_repairs"]) == 1
    assert data["recent_repairs"][0]["repair_action"] == "formula_step_breakdown"


def test_session_filtering(test_setup):
    """Verify session_id query param isolates session interactions and STM."""
    client, service, factory = test_setup

    student_ext = "tutor_stud_sess"
    s1_ext = "tutor_s1"
    s2_ext = "tutor_s2"

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s1_ext)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s2_ext)
        session.add_all([st, sess1, sess2])
        session.flush()

        stm1 = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess1.session_id,
            interaction_count=4,
            correct_count=3,
            incorrect_count=1,
            attempt_observation_count=4,
            attempt_sum=5,
            hint_observation_count=4,
            hint_sum=1,
            response_time_observation_count=4,
            response_time_sum_ms=16000.0,
            recent_accuracy=0.75,
        )
        session.add(stm1)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/tutor-context?session_id={s1_ext}")
    assert r1.status_code == 200
    assert r1.json()["recent_accuracy"] == 0.75
    assert r1.json()["recent_correct_count"] == 3

    r2 = client.get(f"/memory/{student_ext}/tutor-context?session_id={s2_ext}")
    assert r2.status_code == 200
    assert r2.json()["recent_accuracy"] is None
    assert r2.json()["recent_correct_count"] == 0


def test_skill_filtering(test_setup):
    """Verify skill_id query param isolates concept memory and misconceptions."""
    client, service, factory = test_setup

    student_ext = "tutor_stud_skill"
    skill1_uuid = uuid.uuid4()
    skill2_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        session.add(st)
        session.flush()

        concept1 = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill1_uuid,
            interaction_count=5,
            correct_count=5,
            incorrect_count=0,
            attempt_observation_count=5,
            attempt_sum=5,
            hint_observation_count=5,
            hint_sum=0,
            response_time_observation_count=5,
            response_time_sum_ms=15000.0,
            accuracy=1.0,
        )
        session.add(concept1)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/tutor-context?skill_id={skill1_uuid}")
    assert r1.status_code == 200
    assert r1.json()["recent_accuracy"] == 1.0

    r2 = client.get(f"/memory/{student_ext}/tutor-context?skill_id={skill2_uuid}")
    assert r2.status_code == 200
    assert r2.json()["recent_accuracy"] is None


def test_student_isolation(test_setup):
    """Verify Tutor context never leaks data across students."""
    client, service, factory = test_setup

    sA_ext = "tutor_iso_A"
    sB_ext = "tutor_iso_B"

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        session.add_all([stA, stB])
        session.flush()

        ltmA = LongTermMemory(
            student_id=stA.student_id,
            total_sessions=5,
            interaction_count=20,
            correct_count=18,
            incorrect_count=2,
            attempt_observation_count=20,
            attempt_sum=22,
            hint_observation_count=20,
            hint_sum=2,
            response_time_observation_count=20,
            response_time_sum_ms=80000.0,
            overall_accuracy=0.9,
            concept_count=2,
        )
        session.add(ltmA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/tutor-context")
    rB = client.get(f"/memory/{sB_ext}/tutor-context")

    assert rA.status_code == 200
    assert rB.status_code == 200

    assert rA.json()["recent_accuracy"] == 0.9
    assert rB.json()["recent_accuracy"] is None


def test_invalid_limit_returns_422():
    """Verify limit < 1 or limit > 100 returns HTTP 422."""
    client = TestClient(app)

    r1 = client.get("/memory/tutor_stud_1/tutor-context?limit=0")
    assert r1.status_code == 422

    r2 = client.get("/memory/tutor_stud_1/tutor-context?limit=-1")
    assert r2.status_code == 422

    r3 = client.get("/memory/tutor_stud_1/tutor-context?limit=200")
    assert r3.status_code == 422


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_tutor_context.side_effect = RuntimeError("DB connection failure")
    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/tutor_err_1/tutor-context")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve tutor context."
