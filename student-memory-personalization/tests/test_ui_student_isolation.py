"""Authorization regression tests for browser-facing student routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth_routes import get_current_user_optional


def test_unauthenticated_ui_student_route_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/ui/student-a/tutor-context")
    assert response.status_code == 401


def test_student_a_cannot_read_or_mutate_student_b(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    app.dependency_overrides[get_current_user_optional] = lambda: {
        "role": "STUDENT",
        "student_id": "student-a",
    }
    client = TestClient(app, raise_server_exceptions=False)
    try:
        read_response = client.get("/ui/student-b/tutor-context")
        write_response = client.post(
            "/ui/memory-update",
            json={
                "student_id": "student-b",
                "topic": "Algebra",
                "assessment_questions": [
                    {"question_id": "q1", "is_correct": True}
                ],
            },
        )
        repair_response = client.post(
            "/ui/repair-outcome",
            json={
                "student_id": "student-b",
                "session_id": "thread-b",
                "skill_id": "Linear Equations",
                "repair_action": "hint",
                "outcome": "RESOLVED",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert read_response.status_code == 403
    assert write_response.status_code == 403
    assert repair_response.status_code == 403
