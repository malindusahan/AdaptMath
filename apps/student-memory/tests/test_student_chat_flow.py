"""Focused tests for Student Chat flow, automatic student identity, session management, and cognitive memory retrieval."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
from src.database.base import Base
import src.database.postgres_session as postgres_session
from src.demo.seed_demo_data import DEFAULT_DEMO_PASSWORD, seed_demo_environment
from src.services.auth_service import AuthService, set_auth_service
import src.services.fapr_context_service as fapr_context_module
import src.services.meta_signal_service as meta_signal_module
import src.services.planner_context_service as planner_context_module
import src.services.question_context_service as question_context_module
import src.services.repair_outcome_service as repair_outcome_module
import src.services.student_context_service as student_context_module
import src.services.support_preference_service as support_preference_module
import src.services.tutor_context_service as tutor_context_module


@pytest.fixture
def chat_test_env(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Seed standard demo accounts into in-memory engine
    seed_demo_environment(session_factory=factory)

    postgres_session.set_session_factory(factory)
    monkeypatch.setattr(postgres_session, "get_session_factory", lambda: factory)

    # Clear singleton caches
    auth_svc = AuthService(session_factory=factory)
    set_auth_service(auth_svc)
    student_context_module._STUDENT_CONTEXT_SERVICE = student_context_module.StudentContextService(session_factory=factory)
    tutor_context_module._TUTOR_CONTEXT_SERVICE = tutor_context_module.TutorContextService(session_factory=factory)
    planner_context_module._PLANNER_CONTEXT_SERVICE = planner_context_module.PlannerContextService(session_factory=factory)
    fapr_context_module._FAPR_CONTEXT_SERVICE = fapr_context_module.FAPRContextService(session_factory=factory)
    support_preference_module._SUPPORT_PREFERENCE_SERVICE = support_preference_module.SupportPreferenceService(session_factory=factory)
    meta_signal_module._META_SIGNAL_SERVICE = meta_signal_module.MetaSignalService(session_factory=factory)
    question_context_module._QUESTION_CONTEXT_SERVICE = question_context_module.QuestionContextService(session_factory=factory)
    repair_outcome_module._REPAIR_OUTCOME_SERVICE = repair_outcome_module.RepairOutcomeService(session_factory=factory)

    client = TestClient(app, raise_server_exceptions=True)
    yield {"client": client, "factory": factory, "auth_service": auth_svc}
    set_auth_service(None)
    postgres_session.set_session_factory(None)


def test_new_student_cold_start_flow(chat_test_env):
    """Verify new student asks question and receives safe empty memory with no fake history."""
    client = chat_test_env["client"]

    # 1. Login demo_new
    res_login = client.post("/auth/login", json={"username": "demo_new", "password": DEFAULT_DEMO_PASSWORD})
    assert res_login.status_code == 200
    token = res_login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create learning session
    res_sess = client.post("/ui/sessions", headers=headers)
    assert res_sess.status_code == 201
    sess_id = res_sess.json()["session_id"]

    # 3. Ask a mathematics question without sending student_id
    res_qc = client.post(
        "/ui/question-context",
        json={"session_id": sess_id, "question": "Solve the linear equation 3x + 5 = 20"},
        headers=headers,
    )
    assert res_qc.status_code == 200
    data = res_qc.json()

    # Topic detected
    assert data["student_id"] == "demo_new"
    topic_name = (data["topic"].get("canonical_skill_name") or data["topic"].get("display_name") or "").lower()
    assert "linear" in topic_name or "equation" in topic_name

    # Cold start: Safe empty memory
    assert data["concept_memory"] is None
    assert data["learning_state"] is None or data["learning_state"]["learning_state"] == "UNAVAILABLE"
    assert len(data["misconceptions"]) == 0


def test_returning_developing_student_history_reuse(chat_test_env):
    """Verify returning developing student automatically retrieves persistent DEVELOPING learning state and misconceptions."""
    client = chat_test_env["client"]

    # 1. Login demo_developing
    res_login = client.post("/auth/login", json={"username": "demo_developing", "password": DEFAULT_DEMO_PASSWORD})
    assert res_login.status_code == 200
    token = res_login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get sessions
    res_sess_list = client.get("/ui/sessions", headers=headers)
    assert res_sess_list.status_code == 200
    sessions = res_sess_list.json()
    assert len(sessions) > 0
    sess_id = sessions[0]["session_id"]

    # 3. Ask question
    res_qc = client.post(
        "/ui/question-context",
        json={"session_id": sess_id, "question": "Solve the linear equation 3x + 5 = 20"},
        headers=headers,
    )
    assert res_qc.status_code == 200
    data = res_qc.json()

    # Verify history reuse
    assert data["student_id"] == "demo_developing"
    assert data["learning_state"] is not None
    assert data["learning_state"]["learning_state"] == "DEVELOPING"
    assert data["concept_memory"] is not None
    assert data["concept_memory"]["interaction_count"] == 2
    assert len(data["misconceptions"]) >= 1
    assert "Subtracted coefficient" in data["misconceptions"][0]["display_error"]


def test_returning_strong_student_history_reuse(chat_test_env):
    """Verify returning strong student automatically retrieves STRONG learning state."""
    client = chat_test_env["client"]

    # 1. Login demo_strong
    res_login = client.post("/auth/login", json={"username": "demo_strong", "password": DEFAULT_DEMO_PASSWORD})
    assert res_login.status_code == 200
    token = res_login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create session
    res_sess = client.post("/ui/sessions", headers=headers)
    assert res_sess.status_code == 201
    sess_id = res_sess.json()["session_id"]

    # 3. Ask Pythagorean question
    res_qc = client.post(
        "/ui/question-context",
        json={"session_id": sess_id, "question": "What is the Pythagorean theorem hypotenuse if legs are 3 and 4?"},
        headers=headers,
    )
    assert res_qc.status_code == 200
    data = res_qc.json()

    assert data["student_id"] == "demo_strong"
    topic_name = (data["topic"].get("canonical_skill_name") or data["topic"].get("display_name") or "").lower()
    assert "pythagorean" in topic_name
    assert data["learning_state"] is not None
    assert data["learning_state"]["learning_state"] == "STRONG"
    assert data["concept_memory"]["accuracy"] == 1.0


def test_vague_question_abstention(chat_test_env):
    """Verify non-math general question triggers Topic Extractor abstention flag."""
    client = chat_test_env["client"]

    res_login = client.post("/auth/login", json={"username": "demo_new", "password": DEFAULT_DEMO_PASSWORD})
    token = res_login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    res_sess = client.post("/ui/sessions", headers=headers)
    sess_id = res_sess.json()["session_id"]

    res_qc = client.post(
        "/ui/question-context",
        json={"session_id": sess_id, "question": "Can you help me with this?"},
        headers=headers,
    )
    assert res_qc.status_code == 200
    data = res_qc.json()
    assert data["topic"]["needs_review"] is True


def test_session_isolation_between_students(chat_test_env):
    """Verify Student B cannot access or send question context using Student A's session."""
    client = chat_test_env["client"]

    # Student A session
    res_a = client.post("/auth/login", json={"username": "demo_developing", "password": DEFAULT_DEMO_PASSWORD})
    token_a = res_a.json()["token"]
    res_sess_a = client.post("/ui/sessions", headers={"Authorization": f"Bearer {token_a}"})
    sess_a_id = res_sess_a.json()["session_id"]

    # Student B login
    res_b = client.post("/auth/login", json={"username": "demo_strong", "password": DEFAULT_DEMO_PASSWORD})
    token_b = res_b.json()["token"]

    # Student B attempts to query using Student A's session
    res_cross = client.post(
        "/ui/question-context",
        json={"session_id": sess_a_id, "question": "Solve the linear equation 2x = 8"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert res_cross.status_code == 403
