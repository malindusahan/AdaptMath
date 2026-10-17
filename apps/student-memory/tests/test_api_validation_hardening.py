import pytest

from fastapi.testclient import TestClient

import src.api.memory_routes as memory_routes
from src.api.app import app


client = TestClient(app)


def valid_payload():
    return {
        "student_id": "s1",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [{"question_id": "q1", "is_correct": True}],
        "identified_errors": [],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("student_id", ""), ("student_id", "   "), ("topic", ""), ("topic", "   ")],
)
def test_blank_required_update_fields_return_422(field, value):
    payload = valid_payload()
    payload[field] = value
    assert client.post("/memory/update", json=payload).status_code == 422


@pytest.mark.parametrize("question_id", ["", "   "])
def test_blank_question_id_returns_422(question_id):
    payload = valid_payload()
    payload["assessment_questions"][0]["question_id"] = question_id
    assert client.post("/memory/update", json=payload).status_code == 422


def test_blank_identified_error_returns_422():
    payload = valid_payload()
    payload["assessment_questions"][0]["identified_error"] = "   "
    assert client.post("/memory/update", json=payload).status_code == 422


def test_blank_top_level_identified_error_returns_422():
    payload = valid_payload()
    payload["identified_errors"] = ["addition error", "   "]
    assert client.post("/memory/update", json=payload).status_code == 422


def test_hint_count_cannot_exceed_hint_total():
    payload = valid_payload()
    question = payload["assessment_questions"][0]
    question["hint_count"] = 3
    question["hint_total"] = 2
    assert client.post("/memory/update", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_count", -1),
        ("hint_count", -1),
        ("hint_total", -1),
        ("response_time_ms", -1),
    ],
)
def test_negative_behavioural_values_return_422(field, value):
    payload = valid_payload()
    payload["assessment_questions"][0][field] = value
    assert client.post("/memory/update", json=payload).status_code == 422


def test_blank_subtopic_is_normalized_to_none(monkeypatch):
    captured = {}

    def fake_update(request):
        captured["subtopic"] = request.subtopic
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(memory_routes, "update_student_memory", fake_update)
    payload = valid_payload()
    payload["subtopic"] = "   "
    response = client.post("/memory/update", json=payload)
    assert response.status_code == 500
    assert captured["subtopic"] is None


@pytest.mark.parametrize(
    "path", ["/memory/%20%20/current", "/memory/%20%20/history"]
)
def test_blank_path_student_id_returns_422(path):
    response = client.get(path, params={"topic": "Algebra"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "path", ["/memory/s1/current", "/memory/s1/history"]
)
def test_blank_query_topic_returns_422(path):
    response = client.get(path, params={"topic": "   "})
    assert response.status_code == 422


def test_unexpected_update_error_is_sanitized(monkeypatch):
    secret = "database password or internal path"

    def fail(request):
        raise RuntimeError(secret)

    monkeypatch.setattr(memory_routes, "update_student_memory", fail)
    response = client.post("/memory/update", json=valid_payload())
    assert response.status_code == 500
    assert response.json()["detail"] == "Memory update failed."
    assert secret not in response.text


def test_current_repository_failure_is_sanitized(monkeypatch):
    secret = "sqlite internal details"

    def fail(**kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(memory_routes, "get_current_student_memory", fail)
    response = client.get("/memory/s1/current", params={"topic": "Algebra"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve student memory."
    assert secret not in response.text


def test_history_repository_failure_is_sanitized(monkeypatch):
    secret = "private repository failure"

    def fail(**kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(memory_routes, "get_learning_state_history", fail)
    response = client.get("/memory/s1/history", params={"topic": "Algebra"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve learning-state history."
    assert secret not in response.text


def test_whitespace_identifiers_are_trimmed(monkeypatch):
    captured = {}

    def fail_after_capture(request):
        captured["student_id"] = request.student_id
        captured["topic"] = request.topic
        captured["question_id"] = request.assessment_questions[0].question_id
        raise RuntimeError("stop")

    monkeypatch.setattr(memory_routes, "update_student_memory", fail_after_capture)
    payload = valid_payload()
    payload["student_id"] = " s1 "
    payload["topic"] = " Algebra "
    payload["assessment_questions"][0]["question_id"] = " q1 "
    response = client.post("/memory/update", json=payload)
    assert response.status_code == 500
    assert captured == {
        "student_id": "s1",
        "topic": "Algebra",
        "question_id": "q1",
    }
