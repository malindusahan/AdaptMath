"""End-to-end API tests for GET /memory/{student_id}/planner-context endpoint."""

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
from src.database.models.supporting_memory import CurrentLearningState, StudentMisconception
from src.services.planner_context_service import PlannerContextService


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

    service = PlannerContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_no_history_student_returns_safe_defaults(test_setup):
    """Verify non-existent or fresh student returns clean default fields without crashing."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_planner_stud/planner-context")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_planner_stud"
    assert data["session_id"] is None
    assert data["skill_id"] is None
    assert data["session_interaction_count"] == 0
    assert data["session_accuracy"] is None
    assert data["concept_interaction_count"] == 0
    assert data["concept_accuracy"] is None
    assert data["long_term_interaction_count"] == 0
    assert data["overall_accuracy"] is None
    assert data["total_sessions"] == 0
    assert data["concept_count"] == 0
    assert data["current_learning_state"] is None
    assert data["attempt_evidence"]["observation_count"] == 0
    assert data["misconceptions"] == []
    assert data["recent_interactions"] == []


def test_full_planner_context(test_setup):
    """Verify student with complete longitudinal and concept history returns rich planner context."""
    client, service, factory = test_setup

    student_ext = "planner_stud_full"
    session_ext = "planner_sess_full"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(
            skill_id=skill_uuid,
            canonical_name="math :: algebra :: slope",
            display_name="Slope",
        )
        session.add_all([st, sess, skill])
        session.flush()

        # STM
        stm = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess.session_id,
            interaction_count=6,
            correct_count=5,
            incorrect_count=1,
            attempt_observation_count=6,
            attempt_sum=8,
            hint_observation_count=6,
            hint_sum=2,
            response_time_observation_count=6,
            response_time_sum_ms=30000.0,
            recent_accuracy=0.8333,
        )
        # Concept Memory
        concept = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            interaction_count=12,
            correct_count=9,
            incorrect_count=3,
            attempt_observation_count=12,
            attempt_sum=18,
            hint_observation_count=12,
            hint_sum=4,
            response_time_observation_count=12,
            response_time_sum_ms=60000.0,
            accuracy=0.75,
        )
        # LTM
        ltm = LongTermMemory(
            student_id=st.student_id,
            total_sessions=4,
            interaction_count=45,
            correct_count=36,
            incorrect_count=9,
            attempt_observation_count=45,
            attempt_sum=55,
            hint_observation_count=45,
            hint_sum=15,
            response_time_observation_count=45,
            response_time_sum_ms=225000.0,
            overall_accuracy=0.8,
            concept_count=5,
        )
        # Learning State
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="DEVELOPING",
            evidence_level="FULL_SKILL",
            evidence_strength="HIGH",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=12,
            attempt_observation_count=12,
            hint_observation_count=12,
            response_time_observation_count=12,
            last_snapshot_id=1,
        )
        # Misconception
        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="inverted slope delta x over delta y",
            display_error="Inverted delta x and delta y",
            occurrence_count=3,
        )
        # Raw Interaction
        inter = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_pl_01",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=4500.0,
        )

        session.add_all([stm, concept, ltm, state, misc, inter])
        session.commit()

    url = f"/memory/{student_ext}/planner-context?session_id={session_ext}&skill_id={skill_uuid}&limit=5"
    response = client.get(url)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["session_id"] == session_ext
    assert data["skill_id"] == str(skill_uuid)

    # Session stats
    assert data["session_interaction_count"] == 6
    assert data["session_accuracy"] == 0.8333
    assert data["session_correct_count"] == 5
    assert data["session_incorrect_count"] == 1

    # Concept stats
    assert data["concept_interaction_count"] == 12
    assert data["concept_accuracy"] == 0.75
    assert data["concept_correct_count"] == 9
    assert data["concept_incorrect_count"] == 3

    # Long term stats
    assert data["long_term_interaction_count"] == 45
    assert data["overall_accuracy"] == 0.8
    assert data["total_sessions"] == 4
    assert data["concept_count"] == 5

    # Learning state
    assert data["current_learning_state"] == "DEVELOPING"
    assert data["evidence_strength"] == "HIGH"
    assert data["behavioural_coverage"] == "FULL_BEHAVIOURAL_COVERAGE"

    # Behavioral averages from concept tier
    assert data["attempt_evidence"]["observation_count"] == 12
    assert data["attempt_evidence"]["average_value"] == 1.5
    assert data["hint_evidence"]["observation_count"] == 12
    assert data["hint_evidence"]["average_value"] == 0.33
    assert data["response_time_evidence"]["observation_count"] == 12
    assert data["response_time_evidence"]["average_value"] == 5000.0

    # Misconceptions and interactions
    assert len(data["misconceptions"]) == 1
    assert data["misconceptions"][0]["display_error"] == "Inverted delta x and delta y"
    assert len(data["recent_interactions"]) == 1
    assert data["recent_interactions"][0]["is_correct"] is True


def test_session_filtering(test_setup):
    """Verify session_id isolates session-level metrics."""
    client, service, factory = test_setup

    student_ext = "planner_stud_sess"
    s1_ext = "planner_s1"
    s2_ext = "planner_s2"

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s1_ext)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s2_ext)
        session.add_all([st, sess1, sess2])
        session.flush()

        stm1 = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess1.session_id,
            interaction_count=5,
            correct_count=4,
            incorrect_count=1,
            attempt_observation_count=5,
            attempt_sum=6,
            hint_observation_count=5,
            hint_sum=1,
            response_time_observation_count=5,
            response_time_sum_ms=20000.0,
            recent_accuracy=0.8,
        )
        session.add(stm1)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/planner-context?session_id={s1_ext}")
    assert r1.status_code == 200
    assert r1.json()["session_interaction_count"] == 5
    assert r1.json()["session_accuracy"] == 0.8

    r2 = client.get(f"/memory/{student_ext}/planner-context?session_id={s2_ext}")
    assert r2.status_code == 200
    assert r2.json()["session_interaction_count"] == 0
    assert r2.json()["session_accuracy"] is None


def test_skill_filtering(test_setup):
    """Verify skill_id isolates concept-level metrics."""
    client, service, factory = test_setup

    student_ext = "planner_stud_skill"
    skill1_uuid = uuid.uuid4()
    skill2_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        session.add(st)
        session.flush()

        concept1 = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill1_uuid,
            interaction_count=7,
            correct_count=6,
            incorrect_count=1,
            attempt_observation_count=7,
            attempt_sum=8,
            hint_observation_count=7,
            hint_sum=2,
            response_time_observation_count=7,
            response_time_sum_ms=28000.0,
            accuracy=0.8571,
        )
        session.add(concept1)
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/planner-context?skill_id={skill1_uuid}")
    assert r1.status_code == 200
    assert r1.json()["concept_interaction_count"] == 7
    assert r1.json()["concept_accuracy"] == 0.8571

    r2 = client.get(f"/memory/{student_ext}/planner-context?skill_id={skill2_uuid}")
    assert r2.status_code == 200
    assert r2.json()["concept_interaction_count"] == 0
    assert r2.json()["concept_accuracy"] is None


def test_student_isolation(test_setup):
    """Verify planner context never leaks data across distinct students."""
    client, service, factory = test_setup

    sA_ext = "planner_iso_A"
    sB_ext = "planner_iso_B"

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        session.add_all([stA, stB])
        session.flush()

        ltmA = LongTermMemory(
            student_id=stA.student_id,
            total_sessions=8,
            interaction_count=40,
            correct_count=35,
            incorrect_count=5,
            attempt_observation_count=40,
            attempt_sum=45,
            hint_observation_count=40,
            hint_sum=5,
            response_time_observation_count=40,
            response_time_sum_ms=160000.0,
            overall_accuracy=0.875,
            concept_count=6,
        )
        session.add(ltmA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/planner-context")
    rB = client.get(f"/memory/{sB_ext}/planner-context")

    assert rA.status_code == 200
    assert rB.status_code == 200

    assert rA.json()["total_sessions"] == 8
    assert rA.json()["overall_accuracy"] == 0.875

    assert rB.json()["total_sessions"] == 0
    assert rB.json()["overall_accuracy"] is None


def test_invalid_limit_returns_422():
    """Verify limit < 1 or limit > 100 returns HTTP 422."""
    client = TestClient(app)

    r1 = client.get("/memory/planner_stud_1/planner-context?limit=0")
    assert r1.status_code == 422

    r2 = client.get("/memory/planner_stud_1/planner-context?limit=-10")
    assert r2.status_code == 422

    r3 = client.get("/memory/planner_stud_1/planner-context?limit=150")
    assert r3.status_code == 422


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_planner_context.side_effect = RuntimeError("DB connection failure")
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/planner_err_1/planner-context")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve planner context."
