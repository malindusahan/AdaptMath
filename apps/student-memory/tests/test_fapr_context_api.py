"""End-to-end API tests for GET /memory/{student_id}/fapr-context endpoint."""

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
from src.services.fapr_context_service import FAPRContextService


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

    service = FAPRContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_no_history_student_returns_safe_defaults(test_setup):
    """Verify non-existent or fresh student returns clean default fields without crashing."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_fapr_stud/fapr-context")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_fapr_stud"
    assert data["session_id"] is None
    assert data["skill_id"] is None
    assert data["current_learning_state"] is None
    assert data["evidence_strength"] is None
    assert data["behavioural_coverage"] is None
    assert data["recent_accuracy"] is None
    assert data["recent_incorrect_count"] == 0
    assert data["attempt_evidence"]["observation_count"] == 0
    assert data["attempt_evidence"]["average_value"] is None
    assert data["hint_evidence"]["observation_count"] == 0
    assert data["response_time_evidence"]["observation_count"] == 0
    assert data["misconceptions"] == []
    assert data["recent_interactions"] == []
    assert data["previous_repairs"] == []
    assert data["latest_student_utterance"] is None


def test_full_fapr_context(test_setup):
    """Verify student with full memory history returns complete, structured FAPR context."""
    client, service, factory = test_setup

    student_ext = "fapr_stud_full"
    session_ext = "fapr_sess_full"
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
            interaction_count=8,
            correct_count=5,
            incorrect_count=3,
            attempt_observation_count=8,
            attempt_sum=12,
            hint_observation_count=8,
            hint_sum=4,
            response_time_observation_count=8,
            response_time_sum_ms=40000.0,
            accuracy=0.625,
        )
        # Learning State
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="NEEDS_SUPPORT",
            evidence_level="FULL_SKILL",
            evidence_strength="HIGH",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=8,
            attempt_observation_count=8,
            hint_observation_count=8,
            response_time_observation_count=8,
            last_snapshot_id=1,
        )
        # Misconception
        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="forgot to square root hypotenuse",
            display_error="Forgot to square root hypotenuse",
            occurrence_count=3,
        )
        # Raw Interactions
        from datetime import datetime, timezone
        inter1 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_fapr_01",
            student_utterance="I think c squared is 25 so c is 25",
            identified_error="forgot to square root hypotenuse",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
            response_time_ms=5500.0,
            created_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        inter2 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_fapr_02",
            student_utterance="Why isn't 25 the hypotenuse length?",
            identified_error="confused area with side length",
            is_correct=False,
            attempt_count=3,
            hint_count=2,
            response_time_ms=7500.0,
            created_at=datetime(2026, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
        )
        # Repair Outcomes
        repair1 = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="geometric_area_analogy",
            outcome="PARTIALLY_RESOLVED",
            score=0.5,
            notes="Student grasped squares on legs but stumbled on root.",
            created_at=datetime(2026, 1, 1, 10, 2, 0, tzinfo=timezone.utc),
        )
        repair2 = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="inverse_operation_prompt",
            outcome="RESOLVED",
            score=1.0,
            notes="Student applied square root to isolate c.",
            created_at=datetime(2026, 1, 1, 10, 6, 0, tzinfo=timezone.utc),
        )

        session.add_all([concept, state, misc, inter1, inter2, repair1, repair2])
        session.commit()

    url = f"/memory/{student_ext}/fapr-context?session_id={session_ext}&skill_id={skill_uuid}&limit=5"
    response = client.get(url)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["session_id"] == session_ext
    assert data["skill_id"] == str(skill_uuid)

    assert data["current_learning_state"] == "NEEDS_SUPPORT"
    assert data["evidence_strength"] == "HIGH"
    assert data["behavioural_coverage"] == "FULL_BEHAVIOURAL_COVERAGE"

    assert data["recent_accuracy"] == 0.625
    assert data["recent_incorrect_count"] == 3

    assert data["attempt_evidence"]["observation_count"] == 8
    assert data["attempt_evidence"]["average_value"] == 1.5
    assert data["hint_evidence"]["observation_count"] == 8
    assert data["hint_evidence"]["average_value"] == 0.5
    assert data["response_time_evidence"]["observation_count"] == 8
    assert data["response_time_evidence"]["average_value"] == 5000.0

    assert len(data["misconceptions"]) == 1
    assert data["misconceptions"][0]["display_error"] == "Forgot to square root hypotenuse"

    assert len(data["recent_interactions"]) == 2
    assert data["recent_interactions"][-1]["student_utterance"] == "Why isn't 25 the hypotenuse length?"

    assert len(data["previous_repairs"]) == 2
    assert data["previous_repairs"][0]["repair_action"] == "geometric_area_analogy"
    assert data["previous_repairs"][1]["repair_action"] == "inverse_operation_prompt"
    assert data["previous_repairs"][1]["outcome"] == "RESOLVED"

    # Latest student utterance extracted properly
    assert data["latest_student_utterance"] == "Why isn't 25 the hypotenuse length?"


def test_session_filtering(test_setup):
    """Verify session_id isolates session-specific recent interactions."""
    client, service, factory = test_setup

    student_ext = "fapr_stud_sess"
    s1_ext = "fapr_s1"
    s2_ext = "fapr_s2"

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s1_ext)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s2_ext)
        session.add_all([st, sess1, sess2])
        session.flush()

        inter1 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess1.session_id,
            source="ASSESSMENT",
            external_interaction_id="inter_fapr_s1",
            student_utterance="Question from session 1",
            is_correct=False,
            attempt_count=1,
            hint_count=0,
            response_time_ms=3000.0,
        )
        inter2 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess2.session_id,
            source="ASSESSMENT",
            external_interaction_id="inter_fapr_s2",
            student_utterance="Question from session 2",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=2000.0,
        )
        session.add_all([inter1, inter2])
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/fapr-context?session_id={s1_ext}")
    assert r1.status_code == 200
    assert len(r1.json()["recent_interactions"]) == 1
    assert r1.json()["latest_student_utterance"] == "Question from session 1"

    r2 = client.get(f"/memory/{student_ext}/fapr-context?session_id={s2_ext}")
    assert r2.status_code == 200
    assert len(r2.json()["recent_interactions"]) == 1
    assert r2.json()["latest_student_utterance"] == "Question from session 2"


def test_skill_filtering(test_setup):
    """Verify skill_id isolates concept metrics and skill-scoped misconceptions."""
    client, service, factory = test_setup

    student_ext = "fapr_stud_skill"
    skill1_uuid = uuid.uuid4()
    skill2_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        session.add(st)
        session.flush()

        concept1 = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill1_uuid,
            interaction_count=6,
            correct_count=4,
            incorrect_count=2,
            attempt_observation_count=6,
            attempt_sum=8,
            hint_observation_count=6,
            hint_sum=2,
            response_time_observation_count=6,
            response_time_sum_ms=24000.0,
            accuracy=0.6667,
        )
        session.add(concept1)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/fapr-context?skill_id={skill1_uuid}")
    assert r1.status_code == 200
    assert r1.json()["recent_incorrect_count"] == 2
    assert r1.json()["recent_accuracy"] == 0.6667

    r2 = client.get(f"/memory/{student_ext}/fapr-context?skill_id={skill2_uuid}")
    assert r2.status_code == 200
    assert r2.json()["recent_incorrect_count"] == 0
    assert r2.json()["recent_accuracy"] is None


def test_student_isolation(test_setup):
    """Verify FAPR context never leaks across distinct students."""
    client, service, factory = test_setup

    sA_ext = "fapr_iso_A"
    sB_ext = "fapr_iso_B"

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        session.add_all([stA, stB])
        session.flush()

        ltmA = LongTermMemory(
            student_id=stA.student_id,
            total_sessions=4,
            interaction_count=25,
            correct_count=20,
            incorrect_count=5,
            attempt_observation_count=25,
            attempt_sum=30,
            hint_observation_count=25,
            hint_sum=5,
            response_time_observation_count=25,
            response_time_sum_ms=100000.0,
            overall_accuracy=0.8,
            concept_count=3,
        )
        session.add(ltmA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/fapr-context")
    rB = client.get(f"/memory/{sB_ext}/fapr-context")

    assert rA.status_code == 200
    assert rB.status_code == 200

    assert rA.json()["recent_incorrect_count"] == 5
    assert rB.json()["recent_incorrect_count"] == 0


def test_invalid_limit_returns_422():
    """Verify limit < 1 or limit > 100 returns HTTP 422."""
    client = TestClient(app)

    r1 = client.get("/memory/fapr_stud_1/fapr-context?limit=0")
    assert r1.status_code == 422

    r2 = client.get("/memory/fapr_stud_1/fapr-context?limit=-3")
    assert r2.status_code == 422

    r3 = client.get("/memory/fapr_stud_1/fapr-context?limit=300")
    assert r3.status_code == 422


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_fapr_context.side_effect = RuntimeError("DB connection failure")
    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/fapr_err_1/fapr-context")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve FAPR context."
