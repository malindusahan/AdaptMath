"""End-to-end API tests for GET /memory/{student_id}/meta-signals endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
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
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import StudentMisconception
from src.services.meta_signal_service import MetaSignalService


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

    service = MetaSignalService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_meta_signal_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)
    return client, service, factory


def test_no_history_student_returns_empty_signals(test_setup):
    """Verify fresh or unknown student returns zero signals safely."""
    client, service, factory = test_setup

    response = client.get("/memory/unknown_meta_stud/meta-signals")
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "unknown_meta_stud"
    assert data["total_signals"] == 0
    assert data["signals"] == []


def test_correct_and_incorrect_answer_signals(test_setup):
    """Verify interaction correctness emits correct_answer and incorrect_answer signals."""
    client, service, factory = test_setup

    student_ext = "meta_stud_answers"
    session_ext = "meta_sess_answers"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: fractions", display_name="Fractions")
        session.add_all([st, sess, skill])
        session.flush()

        inter1 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_m1",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=3000.0,
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )
        inter2 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_m2",
            identified_error="added numerators and denominators directly",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
            response_time_ms=5000.0,
            created_at=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        )
        session.add_all([inter1, inter2])
        session.commit()

    response = client.get(f"/memory/{student_ext}/meta-signals")
    assert response.status_code == 200

    data = response.json()
    assert data["total_signals"] == 2
    types = [s["signal_type"] for s in data["signals"]]
    assert "correct_answer" in types
    assert "incorrect_answer" in types

    corr_sig = next(s for s in data["signals"] if s["signal_type"] == "correct_answer")
    assert corr_sig["confidence"] == 1.0
    assert corr_sig["evidence"]["attempt_count"] == 1

    incorr_sig = next(s for s in data["signals"] if s["signal_type"] == "incorrect_answer")
    assert incorr_sig["confidence"] == 1.0
    assert incorr_sig["evidence"]["identified_error"] == "added numerators and denominators directly"


def test_clarification_request_and_confusion_signals(test_setup):
    """Verify utterances with confusion and clarification phrases emit appropriate signals."""
    client, service, factory = test_setup

    student_ext = "meta_stud_linguistic"
    session_ext = "meta_sess_linguistic"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: geometry", display_name="Geometry")
        session.add_all([st, sess, skill])
        session.flush()

        inter1 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_ling_1",
            student_utterance="I don't understand how this formula works",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )
        inter2 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_ling_2",
            student_utterance="Can you explain why the hypotenuse is opposite the right angle?",
            is_correct=False,
            attempt_count=1,
            hint_count=0,
            created_at=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        )
        session.add_all([inter1, inter2])
        session.commit()

    response = client.get(f"/memory/{student_ext}/meta-signals")
    assert response.status_code == 200

    data = response.json()
    types = [s["signal_type"] for s in data["signals"]]
    assert "confusion" in types
    assert "clarification_request" in types

    conf_sig = next(s for s in data["signals"] if s["signal_type"] == "confusion")
    assert conf_sig["confidence"] == 0.95
    assert "don't understand" in conf_sig["evidence"]["matched_pattern"].lower()

    clar_sig = next(s for s in data["signals"] if s["signal_type"] == "clarification_request")
    assert clar_sig["confidence"] == 0.90
    assert "can you explain" in clar_sig["evidence"]["matched_pattern"].lower()


def test_repeated_misunderstanding_signal(test_setup):
    """Verify misconceptions with occurrence_count >= 2 emit repeated_misunderstanding signal."""
    client, service, factory = test_setup

    student_ext = "meta_stud_misc"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: slope", display_name="Slope")
        session.add_all([st, skill])
        session.flush()

        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="inverted delta x and delta y in slope",
            display_error="Inverted delta x and delta y in slope formula",
            occurrence_count=3,
        )
        session.add(misc)
        session.commit()

    response = client.get(f"/memory/{student_ext}/meta-signals?skill_id={skill_uuid}")
    assert response.status_code == 200

    data = response.json()
    assert data["total_signals"] == 1
    assert data["signals"][0]["signal_type"] == "repeated_misunderstanding"
    assert data["signals"][0]["confidence"] == 1.0
    assert data["signals"][0]["evidence"]["occurrence_count"] == 3


def test_session_and_skill_filtering(test_setup):
    """Verify session_id and skill_id query parameters isolate signals."""
    client, service, factory = test_setup

    student_ext = "meta_stud_filter"
    s1_ext = "meta_s1"
    s2_ext = "meta_s2"
    skill_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s1_ext)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=s2_ext)
        skill = CanonicalSkill(skill_id=skill_uuid, canonical_name="math :: trig", display_name="Trig")
        session.add_all([st, sess1, sess2, skill])
        session.flush()

        inter1 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess1.session_id,
            source="ASSESSMENT",
            external_interaction_id="int_s1",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )
        inter2 = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=st.student_id,
            session_id=sess2.session_id,
            source="ASSESSMENT",
            external_interaction_id="int_s2",
            is_correct=False,
            attempt_count=1,
            hint_count=0,
            created_at=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        )
        session.add_all([inter1, inter2])
        session.commit()

    r1 = client.get(f"/memory/{student_ext}/meta-signals?session_id={s1_ext}")
    assert r1.status_code == 200
    assert r1.json()["total_signals"] == 1
    assert r1.json()["signals"][0]["signal_type"] == "correct_answer"

    r2 = client.get(f"/memory/{student_ext}/meta-signals?session_id={s2_ext}")
    assert r2.status_code == 200
    assert r2.json()["total_signals"] == 1
    assert r2.json()["signals"][0]["signal_type"] == "incorrect_answer"


def test_student_isolation(test_setup):
    """Verify signals are strictly isolated between students."""
    client, service, factory = test_setup

    sA_ext = "meta_iso_A"
    sB_ext = "meta_iso_B"

    with factory() as session:
        stA = Student(student_id=uuid.uuid4(), external_student_id=sA_ext)
        stB = Student(student_id=uuid.uuid4(), external_student_id=sB_ext)
        sessA = LearningSession(session_id=uuid.uuid4(), student_id=stA.student_id, external_session_id="sess_A")
        session.add_all([stA, stB, sessA])
        session.flush()

        interA = InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stA.student_id,
            session_id=sessA.session_id,
            source="ASSESSMENT",
            external_interaction_id="int_iso_A",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )
        session.add(interA)
        session.commit()

    rA = client.get(f"/memory/{sA_ext}/meta-signals")
    rB = client.get(f"/memory/{sB_ext}/meta-signals")

    assert rA.status_code == 200
    assert rA.json()["total_signals"] == 1

    assert rB.status_code == 200
    assert rB.json()["total_signals"] == 0


def test_limit_handling(test_setup):
    """Verify limit parameter bounds returned signals."""
    client, service, factory = test_setup

    student_ext = "meta_stud_limit"
    session_ext = "meta_sess_limit"

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_ext)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_ext)
        session.add_all([st, sess])
        session.flush()

        for idx in range(10):
            inter = InteractionLog(
                interaction_id=uuid.uuid4(),
                student_id=st.student_id,
                session_id=sess.session_id,
                source="ASSESSMENT",
                external_interaction_id=f"int_lim_{idx}",
                is_correct=True,
                attempt_count=1,
                hint_count=0,
                created_at=datetime(2026, 1, 1, 10, idx, tzinfo=timezone.utc),
            )
            session.add(inter)
        session.commit()

    response = client.get(f"/memory/{student_ext}/meta-signals?limit=3")
    assert response.status_code == 200
    assert response.json()["total_signals"] == 3


def test_db_failure_sanitized_500(monkeypatch):
    """Verify unexpected database exceptions return sanitized 500 error."""
    failing_service = MagicMock()
    failing_service.get_meta_signals.side_effect = RuntimeError("DB failure")
    monkeypatch.setattr(memory_routes, "get_meta_signal_service", lambda: failing_service)

    client = TestClient(app)
    response = client.get("/memory/meta_err/meta-signals")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate meta signals."
