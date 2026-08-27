"""End-to-end API tests for GET /memory/{student_id}/context endpoint."""

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
from src.schemas.raw_interaction import RawInteractionCreate
from src.services.student_context_service import StudentContextService


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

    service = StudentContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_unknown_student_and_no_history_returns_safe_defaults(test_setup):
    """Verify non-existent student or student with no history returns safe null/empty sections."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_student_999/context")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_student_999"
    assert data["session_id"] is None
    assert data["skill_id"] is None
    assert data["short_term_memory"] is None
    assert data["long_term_memory"] is None
    assert data["concept_memory"] is None
    assert data["learning_state"] is None
    assert data["misconceptions"] == []
    assert data["recent_interactions"] == []
    assert data["recent_repairs"] == []


def test_student_with_full_history(test_setup):
    """Verify student with complete STM, LTM, Concept, State, Misconceptions, Interactions, and Repairs."""
    client, service, factory = test_setup

    student_ext = "student_full_1"
    session_ext = "session_full_1"
    skill_uuid = uuid.uuid4()
    skill_code = "SKILL_221"

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
            attempt_sum=7,
            hint_observation_count=6,
            hint_sum=2,
            response_time_observation_count=6,
            response_time_sum_ms=30000.0,
            recent_accuracy=0.8333,
        )
        # LTM
        ltm = LongTermMemory(
            student_id=st.student_id,
            total_sessions=3,
            interaction_count=35,
            correct_count=28,
            incorrect_count=7,
            attempt_observation_count=35,
            attempt_sum=40,
            hint_observation_count=35,
            hint_sum=12,
            response_time_observation_count=35,
            response_time_sum_ms=175000.0,
            overall_accuracy=0.8,
            concept_count=4,
        )
        # Concept Memory
        concept = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            interaction_count=10,
            correct_count=8,
            incorrect_count=2,
            attempt_observation_count=10,
            attempt_sum=12,
            hint_observation_count=10,
            hint_sum=3,
            response_time_observation_count=10,
            response_time_sum_ms=50000.0,
            accuracy=0.8,
        )
        # Learning State
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="STRONG",
            evidence_level="FULL_SKILL",
            evidence_strength="HIGH",
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
            normalized_error="slope formula delta x over delta y",
            display_error="Inverted delta x and delta y in slope formula",
            occurrence_count=3,
        )
        # Raw Interaction
        inter = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="ext_inter_001",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=4500.0,
        )
        # Repair Outcome
        repair = RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            repair_action="visual_graphical_hint",
            outcome="RESOLVED",
            score=1.0,
            notes="Student correctly recognized vertical change over horizontal change.",
        )

        session.add_all([stm, ltm, concept, state, misc, inter, repair])
        session.commit()

    url = f"/memory/{student_ext}/context?session_id={session_ext}&skill_id={skill_uuid}&limit=5"
    response = client.get(url)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == student_ext
    assert data["session_id"] == session_ext
    assert data["skill_id"] == str(skill_uuid)

    assert data["short_term_memory"]["interaction_count"] == 6
    assert data["long_term_memory"]["total_sessions"] == 3
    assert data["concept_memory"]["accuracy"] == 0.8
    assert data["learning_state"]["learning_state"] == "STRONG"
    assert len(data["misconceptions"]) == 1
    assert data["misconceptions"][0]["display_error"] == "Inverted delta x and delta y in slope formula"
    assert len(data["recent_interactions"]) == 1
    assert data["recent_interactions"][0]["is_correct"] is True
    assert len(data["recent_repairs"]) == 1
    assert data["recent_repairs"][0]["repair_action"] == "visual_graphical_hint"


def test_session_scoped_stm(test_setup):
    """Verify STM is only retrieved for the specific session requested."""
    client, service, factory = test_setup

    student_ext = "student_stm_1"
    s1_ext = "session_1"
    s2_ext = "session_2"

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s1_ext)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s2_ext)
        session.add_all([st, sess1, sess2])
        session.flush()

        stm1 = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess1.session_id,
            interaction_count=3,
            correct_count=3,
            incorrect_count=0,
            attempt_observation_count=3,
            attempt_sum=3,
            hint_observation_count=3,
            hint_sum=0,
            response_time_observation_count=3,
            response_time_sum_ms=10000.0,
            recent_accuracy=1.0,
        )
        session.add(stm1)
        session.commit()

    # Querying session 1 returns STM1
    r1 = client.get(f"/memory/{student_ext}/context?session_id={s1_ext}")
    assert r1.status_code == 200
    assert r1.json()["short_term_memory"]["interaction_count"] == 3

    # Querying session 2 returns None for STM
    r2 = client.get(f"/memory/{student_ext}/context?session_id={s2_ext}")
    assert r2.status_code == 200
    assert r2.json()["short_term_memory"] is None

    # Querying without session_id returns None for STM
    r3 = client.get(f"/memory/{student_ext}/context")
    assert r3.status_code == 200
    assert r3.json()["short_term_memory"] is None


def test_skill_scoped_concept_memory(test_setup):
    """Verify Concept memory, learning state, and misconceptions are scoped to the requested skill."""
    client, service, factory = test_setup

    student_ext = "student_skill_1"
    skill1_uuid = uuid.uuid4()
    skill2_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        session.add(st)
        session.flush()

        concept1 = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill1_uuid,
            interaction_count=4,
            correct_count=4,
            incorrect_count=0,
            attempt_observation_count=4,
            attempt_sum=4,
            hint_observation_count=4,
            hint_sum=0,
            response_time_observation_count=4,
            response_time_sum_ms=12000.0,
            accuracy=1.0,
        )
        session.add(concept1)
        session.commit()

    # Querying skill 1
    r1 = client.get(f"/memory/{student_ext}/context?skill_id={skill1_uuid}")
    assert r1.status_code == 200
    assert r1.json()["concept_memory"]["interaction_count"] == 4

    # Querying skill 2
    r2 = client.get(f"/memory/{student_ext}/context?skill_id={skill2_uuid}")
    assert r2.status_code == 200
    assert r2.json()["concept_memory"] is None

    # Querying without skill_id
    r3 = client.get(f"/memory/{student_ext}/context")
    assert r3.status_code == 200
    assert r3.json()["concept_memory"] is None


def test_student_isolation(test_setup):
    """Verify records of student A never leak into student B's context."""
    client, service, factory = test_setup

    sA_ext = "student_A"
    sB_ext = "student_B"

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        session.add_all([stA, stB])
        session.flush()

        ltmA = LongTermMemory(
            student_id=stA.student_id,
            total_sessions=10,
            interaction_count=50,
            correct_count=45,
            incorrect_count=5,
            attempt_observation_count=50,
            attempt_sum=55,
            hint_observation_count=50,
            hint_sum=5,
            response_time_observation_count=50,
            response_time_sum_ms=200000.0,
            overall_accuracy=0.9,
            concept_count=5,
        )
        session.add(ltmA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/context")
    rB = client.get(f"/memory/{sB_ext}/context")

    assert rA.status_code == 200
    assert rB.status_code == 200

    assert rA.json()["long_term_memory"]["total_sessions"] == 10
    assert rB.json()["long_term_memory"] is None


def test_invalid_limit_returns_422():
    """Verify limit < 1 or limit > 100 returns HTTP 422 Unprocessable Content."""
    client = TestClient(app)

    r1 = client.get("/memory/stud_1/context?limit=0")
    assert r1.status_code == 422

    r2 = client.get("/memory/stud_1/context?limit=-5")
    assert r2.status_code == 422

    r3 = client.get("/memory/stud_1/context?limit=500")
    assert r3.status_code == 422


def test_db_failure_sanitized_500(monkeypatch):
    """Verify database exceptions return a sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_student_context.side_effect = RuntimeError("DB connection dropped")
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/stud_err_1/context")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve student context."
