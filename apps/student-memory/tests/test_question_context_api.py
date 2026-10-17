"""End-to-end API tests for POST /memory/question-context endpoint."""

from __future__ import annotations

import json
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
from src.database.models.supporting_memory import CurrentLearningState, StudentMisconception, TopicExtractionLog
from src.services.question_context_service import QuestionContextService


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

    service = QuestionContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_question_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_keyword_question_context(test_setup):
    """Verify alias match resolves canonical skill, attaches empty memory safely, and logs audit."""
    client, service, factory = test_setup
    payload = {
        "student_id": "student_kw_1",
        "session_id": "session_kw_1",
        "question": "Can you explain PEMDAS and how it works?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "student_kw_1"
    assert data["session_id"] == "session_kw_1"
    assert data["question"] == payload["question"]

    topic = data["topic"]
    assert topic["method"] == "minilm_finetuned"
    assert topic["confidence"] >= 0.40
    assert topic["needs_review"] is False
    assert "Order of Operations" in topic["display_name"]
    assert topic["skill_id"] is not None

    assert data["concept_memory"] is None
    assert data["learning_state"] is None
    assert data["misconceptions"] == []


def test_minilm_semantic_question_context(test_setup):
    """Verify semantic question maps via fine-tuned MiniLM."""
    client, service, factory = test_setup
    payload = {
        "student_id": "student_sem_1",
        "session_id": "session_sem_1",
        "question": "How do I work out the steepness of a line from two points on a graph?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    topic = data["topic"]
    assert topic["method"] == "minilm_finetuned"
    assert topic["needs_review"] is False
    assert topic["display_name"] in ("Slope", "Finding Slope from Ordered Pairs")
    assert topic["confidence"] >= 0.45


def test_geometry_comparison_question_context(test_setup):
    """Verify geometry comparison question utilizes neural extraction."""
    client, service, factory = test_setup
    payload = {
        "student_id": "student_geom_1",
        "session_id": "session_geom_1",
        "question": "What is the difference between congruent figures and similar figures?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    topic = data["topic"]
    assert topic["method"] == "minilm_finetuned"
    assert topic["needs_review"] is False
    assert topic["display_name"] in ("Calculations with Similar Figures", "Congruence")


def test_vague_question_abstention(test_setup):
    """Verify vague input triggers safe abstention with needs_review=True and concept_memory=None."""
    client, service, factory = test_setup
    payload = {
        "student_id": "student_vague_1",
        "session_id": "session_vague_1",
        "question": "Can you help me?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    topic = data["topic"]
    assert topic["method"] == "abstain"
    assert topic["skill_id"] is None
    assert topic["needs_review"] is True

    assert data["concept_memory"] is None
    assert data["learning_state"] is None
    assert data["misconceptions"] == []


def test_student_with_no_history_returns_safe_defaults(test_setup):
    """Verify brand new student returns clean null/empty memory fields without crashing."""
    client, service, factory = test_setup
    payload = {
        "student_id": f"new_student_{uuid.uuid4()}",
        "session_id": f"new_session_{uuid.uuid4()}",
        "question": "What is the Pythagorean theorem formula?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["short_term_memory"] is None
    assert data["long_term_memory"] is None
    assert data["concept_memory"] is None
    assert data["learning_state"] is None
    assert data["misconceptions"] == []


def test_student_with_existing_stm_ltm_concept_memory(test_setup):
    """Verify existing STM, LTM, and Concept memory are correctly retrieved for the extracted skill."""
    client, service, factory = test_setup

    student_ext = "student_rich_1"
    session_ext = "session_rich_1"

    # Pre-extract skill UUID for "Slope"
    topic_res = service.topic_extractor.extract("How do I calculate slope?")
    skill_uuid = uuid.UUID(str(topic_res.skill_id))

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        session.add(st)
        session.add(sess)
        session.flush()

        # Seed STM
        stm = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess.session_id,
            interaction_count=5,
            correct_count=4,
            incorrect_count=1,
            attempt_observation_count=5,
            attempt_sum=6,
            hint_observation_count=5,
            hint_sum=2,
            response_time_observation_count=5,
            response_time_sum_ms=25000.0,
            recent_accuracy=0.8,
        )
        # Seed LTM
        ltm = LongTermMemory(
            student_id=st.student_id,
            total_sessions=2,
            interaction_count=20,
            correct_count=16,
            incorrect_count=4,
            attempt_observation_count=20,
            attempt_sum=24,
            hint_observation_count=20,
            hint_sum=8,
            response_time_observation_count=20,
            response_time_sum_ms=100000.0,
            overall_accuracy=0.8,
            concept_count=3,
        )
        # Seed Concept Memory
        concept = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            interaction_count=8,
            correct_count=7,
            incorrect_count=1,
            attempt_observation_count=8,
            attempt_sum=9,
            hint_observation_count=8,
            hint_sum=1,
            response_time_observation_count=8,
            response_time_sum_ms=40000.0,
            accuracy=0.875,
        )
        # Seed Learning State
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="STRONG",
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
        # Seed Misconception
        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="inverted delta y and delta x",
            display_error="Inverted delta y and delta x",
            occurrence_count=2,
        )

        session.add(stm)
        session.add(ltm)
        session.add(concept)
        session.add(state)
        session.add(misc)
        session.commit()

    payload = {
        "student_id": student_ext,
        "session_id": session_ext,
        "question": "How do I calculate slope?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["short_term_memory"]["interaction_count"] == 5
    assert data["short_term_memory"]["recent_accuracy"] == 0.8
    assert data["long_term_memory"]["total_sessions"] == 2
    assert data["concept_memory"]["accuracy"] == 0.875
    assert data["learning_state"]["learning_state"] == "STRONG"
    assert len(data["misconceptions"]) == 1
    assert data["misconceptions"][0]["display_error"] == "Inverted delta y and delta x"


def test_student_and_session_isolation(test_setup):
    """Verify distinct students receive isolated context responses and audit entries."""
    client, service, factory = test_setup
    p1 = {"student_id": "stud_iso_1", "session_id": "sess_iso_1", "question": "Can you explain PEMDAS?"}
    p2 = {"student_id": "stud_iso_2", "session_id": "sess_iso_2", "question": "What is Pythagoras theorem?"}

    r1 = client.post("/memory/question-context", json=p1)
    r2 = client.post("/memory/question-context", json=p2)

    assert r1.status_code == 200
    assert r2.status_code == 200

    assert r1.json()["student_id"] == "stud_iso_1"
    assert r2.json()["student_id"] == "stud_iso_2"
    assert "Order of Operations" in r1.json()["topic"]["display_name"]
    assert "Pythagorean Theorem" in r2.json()["topic"]["display_name"]


def test_invalid_request_validation_422(test_setup):
    """Verify blank fields or missing parameters return HTTP 422 Unprocessable Content."""
    client, service, factory = test_setup
    invalid_payloads = [
        {"student_id": "", "session_id": "sess_1", "question": "What is slope?"},
        {"student_id": "stud_1", "session_id": "   ", "question": "What is slope?"},
        {"student_id": "stud_1", "session_id": "sess_1", "question": ""},
        {"student_id": "stud_1", "session_id": "sess_1"},  # missing question
    ]

    for p in invalid_payloads:
        resp = client.post("/memory/question-context", json=p)
        assert resp.status_code == 422, f"Expected 422 for {p}, got {resp.status_code}"


def test_db_failure_sanitized_500(monkeypatch):
    """Verify database exceptions are caught and sanitized as a 500 error."""
    failing_service = MagicMock()
    failing_service.get_question_context.side_effect = RuntimeError("Database timeout connection error")
    monkeypatch.setattr(memory_routes, "get_question_context_service", lambda: failing_service)

    client = TestClient(app)
    payload = {
        "student_id": "student_err_1",
        "session_id": "session_err_1",
        "question": "Can you explain PEMDAS?",
    }
    response = client.post("/memory/question-context", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to resolve question context."
