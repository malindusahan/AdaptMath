import pytest

from fastapi.testclient import TestClient

import src.api.memory_routes as memory_routes

from src.api.app import app
from src.services.memory_update_service import (
    MemoryUpdateServiceError,
    update_student_memory,
)


client = TestClient(app)


def valid_payload():
    return {
        "student_id": "s1",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "What is x if 3x = 15?",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "identified_error": None,
            },
            {
                "question_id": "q2",
                "question": "Solve x - 2 = 7.",
                "student_answer": "x = 8",
                "expected_answer": "x = 9",
                "is_correct": False,
                "identified_error": "addition error",
            },
        ],
        "identified_errors": ["addition error"],
        "overall_feedback": (
            "Mostly correct, but review addition "
            "in linear equations."
        ),
    }


@pytest.fixture
def temporary_memory_route(monkeypatch, tmp_path):
    """Run the complete pipeline against a temporary SQLite database."""
    database_path = tmp_path / "student_memory.db"

    def temporary_update(request):
        return update_student_memory(
            request=request,
            database_path=database_path,
        )

    monkeypatch.setattr(
        memory_routes,
        "update_student_memory",
        temporary_update,
    )
    return database_path


def test_update_memory_success(temporary_memory_route):
    response = client.post("/memory/update", json=valid_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["student_id"] == "s1"
    assert payload["topic"] == "Algebra"
    assert payload["subtopic"] == "Linear equations"
    assert payload["assessment_id"] > 0
    assert payload["snapshot_id"] > 0
    assert payload["learning_state"] in {
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG",
    }
    assert payload["evidence_level"] in {
        "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL",
    }
    assert payload["evidence_strength"] in {"LOW", "MEDIUM", "HIGH"}
    assert payload["behavioural_coverage"] == "CORRECTNESS_ONLY_COVERAGE"
    assert payload["model_used"] is True
    assert payload["memory_updated"] is True


def test_update_includes_current_assessment(temporary_memory_route):
    response = client.post("/memory/update", json=valid_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["previous_interaction_count"] == 2
    assert payload["previous_skill_interaction_count"] == 2


def test_optional_behavioural_fields_can_be_omitted(temporary_memory_route):
    payload = valid_payload()
    for question in payload["assessment_questions"]:
        assert "attempt_count" not in question
        assert "hint_count" not in question
        assert "hint_total" not in question
        assert "response_time_ms" not in question
    response = client.post("/memory/update", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["behavioural_coverage"] == "CORRECTNESS_ONLY_COVERAGE"
    assert result["attempt_observation_count"] == 0
    assert result["hint_observation_count"] == 0
    assert result["response_time_observation_count"] == 0


def test_behavioural_fields_are_accepted(temporary_memory_route):
    payload = valid_payload()
    for index, question in enumerate(payload["assessment_questions"]):
        question["attempt_count"] = index + 1
        question["hint_count"] = 0
        question["hint_total"] = 2
        question["response_time_ms"] = 1500 + index * 500
    response = client.post("/memory/update", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["behavioural_coverage"] == "FULL_BEHAVIOURAL_COVERAGE"
    assert result["attempt_observation_count"] == 2
    assert result["hint_observation_count"] == 2
    assert result["response_time_observation_count"] == 2


def test_missing_student_id_returns_422():
    payload = valid_payload()
    del payload["student_id"]
    assert client.post("/memory/update", json=payload).status_code == 422


def test_missing_topic_returns_422():
    payload = valid_payload()
    del payload["topic"]
    assert client.post("/memory/update", json=payload).status_code == 422


def test_missing_assessment_questions_returns_422():
    payload = valid_payload()
    del payload["assessment_questions"]
    assert client.post("/memory/update", json=payload).status_code == 422


def test_invalid_is_correct_returns_422():
    payload = valid_payload()
    payload["assessment_questions"][0]["is_correct"] = "not-a-boolean"
    assert client.post("/memory/update", json=payload).status_code == 422


def test_service_failure_returns_500(monkeypatch):
    def fail_update(request):
        raise MemoryUpdateServiceError("Simulated internal failure")

    monkeypatch.setattr(memory_routes, "update_student_memory", fail_update)
    response = client.post("/memory/update", json=valid_payload())
    assert response.status_code == 500
    assert response.json()["detail"] == "Memory update failed."


def test_internal_error_is_not_exposed(monkeypatch):
    secret_message = "internal database details that must not reach the caller"

    def fail_update(request):
        raise MemoryUpdateServiceError(secret_message)

    monkeypatch.setattr(memory_routes, "update_student_memory", fail_update)
    response = client.post("/memory/update", json=valid_payload())
    assert response.status_code == 500
    assert secret_message not in response.text


def test_openapi_contains_memory_update():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/memory/update" in paths
    assert "post" in paths["/memory/update"]
