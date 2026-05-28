import pytest

from fastapi.testclient import TestClient

import src.api.memory_routes as memory_routes

from src.api.app import app
from src.database.state_repository import (
    get_current_student_memory,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)
from src.services.memory_update_service import (
    update_student_memory,
)


client = TestClient(app)


def make_request(
    *,
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
    question_id="q1",
    is_correct=True,
):
    return AssessmentMemoryUpdateRequest(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        assessment_questions=[
            AssessmentQuestionResult(
                question_id=question_id,
                is_correct=is_correct,
            )
        ],
    )


@pytest.fixture
def temporary_current_memory_route(
    monkeypatch,
    tmp_path,
):
    database_path = tmp_path / "student_memory.db"
    temporary_current_memory_route_path = database_path

    def temporary_get_current_memory(
        student_id,
        topic,
        subtopic,
        database_path=None,
        connection=None,
    ):
        return get_current_student_memory(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
            database_path=temporary_current_memory_route_path,
            connection=connection,
        )

    monkeypatch.setattr(
        memory_routes,
        "get_current_student_memory",
        temporary_get_current_memory,
    )
    return database_path


def test_current_memory_success(temporary_current_memory_route):
    database_path = temporary_current_memory_route
    update = update_student_memory(make_request(), database_path=database_path)
    response = client.get(
        "/memory/s1/current",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["student_id"] == "s1"
    assert payload["topic"] == "Algebra"
    assert payload["subtopic"] == "Linear equations"
    assert payload["learning_state"] == update.learning_state
    assert payload["last_assessment_id"] == update.assessment_id
    assert payload["last_snapshot_id"] == update.snapshot_id


def test_unknown_memory_returns_404(temporary_current_memory_route):
    response = client.get(
        "/memory/unknown/current",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Student memory not found."


def test_topic_is_required(temporary_current_memory_route):
    response = client.get("/memory/s1/current")
    assert response.status_code == 422


def test_subtopic_is_optional(temporary_current_memory_route):
    database_path = temporary_current_memory_route
    update_student_memory(make_request(subtopic=None), database_path=database_path)
    response = client.get("/memory/s1/current", params={"topic": "Algebra"})
    assert response.status_code == 200
    assert response.json()["subtopic"] is None


def test_current_retrieval_is_context_specific(temporary_current_memory_route):
    database_path = temporary_current_memory_route
    update_student_memory(
        make_request(
            topic="Algebra",
            subtopic="Linear equations",
            question_id="alg-q1",
        ),
        database_path=database_path,
    )
    update_student_memory(
        make_request(
            topic="Geometry",
            subtopic="Angles",
            question_id="geo-q1",
        ),
        database_path=database_path,
    )
    algebra = client.get(
        "/memory/s1/current",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    geometry = client.get(
        "/memory/s1/current",
        params={"topic": "Geometry", "subtopic": "Angles"},
    )
    assert algebra.status_code == 200
    assert geometry.status_code == 200
    assert algebra.json()["topic"] == "Algebra"
    assert geometry.json()["topic"] == "Geometry"


def test_current_endpoint_is_read_only(temporary_current_memory_route):
    database_path = temporary_current_memory_route
    update = update_student_memory(make_request(), database_path=database_path)
    params = {"topic": "Algebra", "subtopic": "Linear equations"}
    first = client.get("/memory/s1/current", params=params)
    second = client.get("/memory/s1/current", params=params)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["last_snapshot_id"] == update.snapshot_id


def test_openapi_contains_current_memory_endpoint():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    path = "/memory/{student_id}/current"
    assert path in paths
    assert "get" in paths[path]
