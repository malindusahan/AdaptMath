from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import tutor as tutor_api
from app.clients.memory.memory_client import MemoryAuthenticatedUser
from app.main import app
import app.core.user_auth as user_auth


STUDENT_A = MemoryAuthenticatedUser(
    user_id="user-a",
    username="student-a",
    role="STUDENT",
    student_id="student-a",
    age=15,
)


class AuthClient:
    def __init__(self, user: MemoryAuthenticatedUser | None):
        self.user = user
        self.tokens: list[str] = []

    def authenticate_user(self, token: str):
        self.tokens.append(token)
        return self.user


def _start_payload(**overrides):
    payload = {
        "age": 15,
        "question": "Solve 2x = 8.",
        "topic": "Algebra",
        "target_skill": "Equation Solving Two or Fewer Steps",
        "relevant_history": [],
        "previous_errors": [],
        "previous_strategies": [],
    }
    payload.update(overrides)
    return payload


def test_tutor_start_without_token_returns_401():
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/tutor/start", json=_start_payload())

    assert response.status_code == 401


def test_tutor_start_with_invalid_token_returns_401(monkeypatch):
    auth_client = AuthClient(None)
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: auth_client)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/tutor/start",
        headers={"Authorization": "Bearer invalid-token"},
        json=_start_payload(),
    )

    assert response.status_code == 401
    assert auth_client.tokens == ["invalid-token"]


def test_authenticated_student_is_authoritative_for_tutor_bkt_and_memory(
    monkeypatch,
):
    auth_client = AuthClient(STUDENT_A)
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: auth_client)

    memory_calls: list[dict[str, str]] = []

    class MemoryClient:
        def retrieve_tutor_context(self, *, student_id, target_skill):
            memory_calls.append(
                {"student_id": student_id, "target_skill": target_skill}
            )
            return None

    class Graph:
        def __init__(self):
            self.initial_state = None

        def invoke(self, state, *, config):
            del config
            self.initial_state = dict(state)
            return dict(state)

    graph = Graph()
    monkeypatch.setattr(tutor_api, "get_memory_client", lambda: MemoryClient())
    monkeypatch.setattr(tutor_api, "_find_active_student_session", lambda _: None)
    monkeypatch.setattr(tutor_api, "tutor_graph", graph)
    monkeypatch.setattr(
        tutor_api.adaptive_coordinator,
        "validate_target_skill",
        lambda skill: skill,
    )
    monkeypatch.setattr(
        tutor_api.adaptive_coordinator,
        "derive_attempt_id",
        lambda thread_id, index: f"{thread_id}:{index}",
    )

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/tutor/start",
        headers={"Authorization": "Bearer valid-a"},
        json=_start_payload(),
    )

    assert response.status_code == 200, response.text
    assert graph.initial_state["student_id"] == "student-a"
    assert memory_calls == [
        {
            "student_id": "student-a",
            "target_skill": "Equation Solving Two or Fewer Steps",
        }
    ]


def test_student_cannot_impersonate_another_student(monkeypatch):
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: AuthClient(STUDENT_A))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/tutor/start",
        headers={"Authorization": "Bearer valid-a"},
        json=_start_payload(student_id="student-b"),
    )

    assert response.status_code == 403


def test_student_cannot_resume_another_students_thread(monkeypatch):
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: AuthClient(STUDENT_A))
    snapshot = SimpleNamespace(
        values={"student_id": "student-b"},
        next=("unfinished_node",),
        tasks=(),
    )
    monkeypatch.setattr(tutor_api, "_get_persisted_snapshot", lambda _: snapshot)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/tutor/thread-owned-by-b/resume",
        headers={"Authorization": "Bearer valid-a"},
    )

    assert response.status_code == 403
