import pytest

from fastapi.testclient import TestClient

import src.api.memory_routes as memory_routes

from src.api.app import app
from src.database.state_repository import get_learning_state_history
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)
from src.services.memory_update_service import update_student_memory


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
def temporary_history_route(monkeypatch, tmp_path):
    database_path = tmp_path / "student_memory.db"
    temporary_history_route_path = database_path

    def temporary_get_history(
        student_id,
        topic,
        subtopic,
        database_path=None,
        connection=None,
    ):
        return get_learning_state_history(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
            database_path=temporary_history_route_path,
            connection=connection,
        )

    monkeypatch.setattr(
        memory_routes,
        "get_learning_state_history",
        temporary_get_history,
    )
    return database_path


def test_history_success(temporary_history_route):
    database_path = temporary_history_route
    first = update_student_memory(
        make_request(question_id="q1", is_correct=False),
        database_path=database_path,
    )
    second = update_student_memory(
        make_request(question_id="q2", is_correct=True),
        database_path=database_path,
    )
    response = client.get(
        "/memory/s1/history",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["snapshot_id"] == first.snapshot_id
    assert payload[1]["snapshot_id"] == second.snapshot_id


def test_history_is_chronological(temporary_history_route):
    database_path = temporary_history_route
    updates = []
    for index, correct in enumerate([False, True, True], start=1):
        updates.append(
            update_student_memory(
                make_request(question_id=f"q{index}", is_correct=correct),
                database_path=database_path,
            )
        )
    response = client.get(
        "/memory/s1/history",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    assert response.status_code == 200
    assert [item["snapshot_id"] for item in response.json()] == [
        update.snapshot_id for update in updates
    ]


def test_unknown_history_returns_empty_list(temporary_history_route):
    response = client.get(
        "/memory/unknown/history",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    assert response.status_code == 200
    assert response.json() == []


def test_topic_is_required(temporary_history_route):
    assert client.get("/memory/s1/history").status_code == 422


def test_subtopic_is_optional(temporary_history_route):
    update_student_memory(
        make_request(subtopic=None),
        database_path=temporary_history_route,
    )
    response = client.get("/memory/s1/history", params={"topic": "Algebra"})
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["subtopic"] is None


def test_history_is_student_specific(temporary_history_route):
    database_path = temporary_history_route
    update_student_memory(
        make_request(student_id="s1", question_id="s1-q1"),
        database_path=database_path,
    )
    update_student_memory(
        make_request(student_id="s2", question_id="s2-q1"),
        database_path=database_path,
    )
    params = {"topic": "Algebra", "subtopic": "Linear equations"}
    s1 = client.get("/memory/s1/history", params=params)
    s2 = client.get("/memory/s2/history", params=params)
    assert s1.status_code == 200
    assert s2.status_code == 200
    assert len(s1.json()) == 1
    assert len(s2.json()) == 1
    assert s1.json()[0]["student_id"] == "s1"
    assert s2.json()[0]["student_id"] == "s2"


def test_history_is_context_specific(temporary_history_route):
    database_path = temporary_history_route
    update_student_memory(
        make_request(
            topic="Algebra", subtopic="Linear equations", question_id="alg-q1"
        ),
        database_path=database_path,
    )
    update_student_memory(
        make_request(topic="Geometry", subtopic="Angles", question_id="geo-q1"),
        database_path=database_path,
    )
    algebra = client.get(
        "/memory/s1/history",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    geometry = client.get(
        "/memory/s1/history",
        params={"topic": "Geometry", "subtopic": "Angles"},
    )
    assert algebra.status_code == 200
    assert geometry.status_code == 200
    assert len(algebra.json()) == 1
    assert len(geometry.json()) == 1
    assert algebra.json()[0]["topic"] == "Algebra"
    assert geometry.json()[0]["topic"] == "Geometry"


def test_history_endpoint_is_read_only(temporary_history_route):
    update_student_memory(make_request(), database_path=temporary_history_route)
    params = {"topic": "Algebra", "subtopic": "Linear equations"}
    first = client.get("/memory/s1/history", params=params)
    second = client.get("/memory/s1/history", params=params)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_history_preserves_state_context(temporary_history_route):
    update = update_student_memory(
        make_request(),
        database_path=temporary_history_route,
    )
    response = client.get(
        "/memory/s1/history",
        params={"topic": "Algebra", "subtopic": "Linear equations"},
    )
    item = response.json()[0]
    assert item["learning_state"] == update.learning_state
    assert item["evidence_level"] == update.evidence_level
    assert item["evidence_strength"] == update.evidence_strength
    assert item["behavioural_coverage"] == update.behavioural_coverage


def test_openapi_contains_history_endpoint():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    path = "/memory/{student_id}/history"
    assert path in paths
    assert "get" in paths[path]
