from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import tutor as tutor_api
from app.clients.memory.memory_client import MemoryAuthenticatedUser
from app.main import app
from app.services.practice_problem_service import (
    GeneratedPracticeProblem,
    PracticeProblemGenerationError,
)
import app.core.user_auth as user_auth


USER = MemoryAuthenticatedUser(
    user_id="user-profile",
    username="profile-student",
    role="STUDENT",
    student_id="student-profile",
    age=14,
)


class AuthClient:
    def authenticate_user(self, token: str):
        return USER if token == "valid-token" else None


class Generator:
    def __init__(self, *, fails: bool = False):
        self.fails = fails
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fails:
            raise PracticeProblemGenerationError("no validated candidate")
        return GeneratedPracticeProblem(
            question="Which fraction is equivalent to 3/5: 6/10 or 6/15?",
            expected_answer="6/10",
        )


def _auth(monkeypatch):
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: AuthClient())


def test_profile_practice_start_requires_authentication():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/tutor/start-practice",
        json={"target_skill": "Equivalent Fractions"},
    )
    assert response.status_code == 401


def test_profile_practice_generates_then_reuses_normal_start(monkeypatch):
    _auth(monkeypatch)
    generator = Generator()
    monkeypatch.setattr(
        tutor_api,
        "get_practice_problem_generator",
        lambda: generator,
    )
    monkeypatch.setattr(tutor_api, "get_memory_client", lambda: object())
    monkeypatch.setattr(tutor_api, "_find_active_student_session", lambda _: None)
    monkeypatch.setattr(
        tutor_api.adaptive_coordinator,
        "validate_target_skill",
        lambda skill: skill,
    )
    start_calls = []

    def fake_start(request, authenticated_user):
        start_calls.append((request, authenticated_user))
        return tutor_api.TutorSessionResponse(
            thread_id="thread-profile",
            status="student_response_required",
            tutor_response="What relationship do you notice?",
            target_skill=request.target_skill,
            turn_count=1,
        )

    monkeypatch.setattr(tutor_api, "_start_tutor_session", fake_start)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/tutor/start-practice",
        headers={"Authorization": "Bearer valid-token"},
        json={"target_skill": "Equivalent Fractions"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target_skill"] == "Equivalent Fractions"
    assert body["problem"].startswith("Which fraction")
    assert body["session"]["thread_id"] == "thread-profile"
    request, authenticated_user = start_calls[0]
    assert authenticated_user.student_id == "student-profile"
    assert request.age == 14
    assert request.question == body["problem"]
    assert request.topic == "Equivalent Fractions"
    assert request.subtopic == "Equivalent Fractions"
    assert request.target_skill == "Equivalent Fractions"
    assert generator.calls[0]["target_skill"] == "Equivalent Fractions"


def test_generation_failure_does_not_start_tutor(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(
        tutor_api,
        "get_practice_problem_generator",
        lambda: Generator(fails=True),
    )
    monkeypatch.setattr(tutor_api, "get_memory_client", lambda: object())
    monkeypatch.setattr(tutor_api, "_find_active_student_session", lambda _: None)
    monkeypatch.setattr(
        tutor_api.adaptive_coordinator,
        "validate_target_skill",
        lambda skill: skill,
    )

    def should_not_start(*args, **kwargs):
        raise AssertionError("Tutor start must not run for an unverified problem")

    monkeypatch.setattr(tutor_api, "_start_tutor_session", should_not_start)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/tutor/start-practice",
        headers={"Authorization": "Bearer valid-token"},
        json={"target_skill": "Equivalent Fractions"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "We couldn't prepare a practice problem right now."
