import pytest
from fastapi.testclient import TestClient

import src.api.memory_routes as memory_routes

from src.api.app import app
from src.database.state_repository import (
    get_current_student_memory,
    get_learning_state_history,
)
from src.services.memory_update_service import update_student_memory


client = TestClient(app)


@pytest.fixture
def temporary_api_database(monkeypatch, tmp_path):
    database_path = tmp_path / "student_memory.db"
    temporary_api_database_path = database_path

    def temporary_update(request):
        return update_student_memory(request=request, database_path=database_path)

    def temporary_current(
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
            database_path=temporary_api_database_path,
            connection=connection,
        )

    def temporary_history(
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
            database_path=temporary_api_database_path,
            connection=connection,
        )

    monkeypatch.setattr(memory_routes, "update_student_memory", temporary_update)
    monkeypatch.setattr(
        memory_routes, "get_current_student_memory", temporary_current
    )
    monkeypatch.setattr(
        memory_routes, "get_learning_state_history", temporary_history
    )
    return database_path


def assessment_payload(
    *,
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
    question_id="q1",
    is_correct=True,
    identified_error=None,
):
    return {
        "student_id": student_id,
        "topic": topic,
        "subtopic": subtopic,
        "assessment_questions": [
            {
                "question_id": question_id,
                "question": "Example assessment question",
                "student_answer": "correct answer" if is_correct else "wrong answer",
                "expected_answer": "correct answer",
                "is_correct": is_correct,
                "identified_error": identified_error,
            }
        ],
        "identified_errors": [identified_error] if identified_error else [],
        "overall_feedback": "Assessment completed.",
    }


def get_current(
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
):
    params = {"topic": topic}
    if subtopic is not None:
        params["subtopic"] = subtopic
    return client.get(f"/memory/{student_id}/current", params=params)


def get_history(
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
):
    params = {"topic": topic}
    if subtopic is not None:
        params["subtopic"] = subtopic
    return client.get(f"/memory/{student_id}/history", params=params)


def test_first_assessment_complete_http_flow(temporary_api_database):
    update = client.post(
        "/memory/update",
        json=assessment_payload(
            question_id="q1",
            is_correct=False,
            identified_error="addition error",
        ),
    )
    assert update.status_code == 200
    update_data = update.json()
    current = get_current()
    assert current.status_code == 200
    current_data = current.json()
    assert current_data["last_assessment_id"] == update_data["assessment_id"]
    assert current_data["last_snapshot_id"] == update_data["snapshot_id"]
    assert current_data["learning_state"] == update_data["learning_state"]
    history = get_history()
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["snapshot_id"] == update_data["snapshot_id"]


def test_second_assessment_replaces_current_memory(temporary_api_database):
    first = client.post(
        "/memory/update",
        json=assessment_payload(
            question_id="q1", is_correct=False, identified_error="addition error"
        ),
    )
    second = client.post(
        "/memory/update",
        json=assessment_payload(question_id="q2", is_correct=True),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_data = first.json()
    second_data = second.json()
    assert second_data["snapshot_id"] > first_data["snapshot_id"]
    current = get_current()
    assert current.status_code == 200
    current_data = current.json()
    assert current_data["last_snapshot_id"] == second_data["snapshot_id"]
    assert current_data["last_assessment_id"] == second_data["assessment_id"]
    assert current_data["learning_state"] == second_data["learning_state"]


def test_history_grows_after_multiple_updates(temporary_api_database):
    first = client.post(
        "/memory/update",
        json=assessment_payload(question_id="q1", is_correct=False),
    )
    second = client.post(
        "/memory/update",
        json=assessment_payload(question_id="q2", is_correct=True),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    history = get_history()
    assert history.status_code == 200
    assert [item["snapshot_id"] for item in history.json()] == [
        first.json()["snapshot_id"],
        second.json()["snapshot_id"],
    ]


def test_http_flow_preserves_topic_isolation(temporary_api_database):
    algebra = client.post(
        "/memory/update",
        json=assessment_payload(
            topic="Algebra",
            subtopic="Linear equations",
            question_id="alg-q1",
            is_correct=True,
        ),
    )
    geometry = client.post(
        "/memory/update",
        json=assessment_payload(
            topic="Geometry",
            subtopic="Angles",
            question_id="geo-q1",
            is_correct=False,
        ),
    )
    assert algebra.status_code == 200
    assert geometry.status_code == 200
    algebra_history = get_history(
        topic="Algebra", subtopic="Linear equations"
    )
    geometry_history = get_history(topic="Geometry", subtopic="Angles")
    assert len(algebra_history.json()) == 1
    assert len(geometry_history.json()) == 1
    assert algebra_history.json()[0]["topic"] == "Algebra"
    assert geometry_history.json()[0]["topic"] == "Geometry"


def test_http_flow_preserves_student_isolation(temporary_api_database):
    s1 = client.post(
        "/memory/update",
        json=assessment_payload(student_id="s1", question_id="s1-q1"),
    )
    s2 = client.post(
        "/memory/update",
        json=assessment_payload(student_id="s2", question_id="s2-q1"),
    )
    assert s1.status_code == 200
    assert s2.status_code == 200
    s1_history = get_history(student_id="s1")
    s2_history = get_history(student_id="s2")
    assert len(s1_history.json()) == 1
    assert len(s2_history.json()) == 1
    assert s1_history.json()[0]["student_id"] == "s1"
    assert s2_history.json()[0]["student_id"] == "s2"


def test_http_flow_supports_null_subtopic(temporary_api_database):
    update = client.post(
        "/memory/update",
        json=assessment_payload(subtopic=None),
    )
    assert update.status_code == 200
    assert update.json()["subtopic"] is None
    current = get_current(subtopic=None)
    assert current.status_code == 200
    assert current.json()["subtopic"] is None
    history = get_history(subtopic=None)
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["subtopic"] is None


def test_later_assessment_uses_accumulated_history(temporary_api_database):
    first = client.post(
        "/memory/update",
        json=assessment_payload(question_id="q1", is_correct=False),
    )
    second = client.post(
        "/memory/update",
        json=assessment_payload(question_id="q2", is_correct=True),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_data = first.json()
    second_data = second.json()
    assert (
        second_data["previous_interaction_count"]
        > first_data["previous_interaction_count"]
    )
    assert (
        second_data["previous_skill_interaction_count"]
        > first_data["previous_skill_interaction_count"]
    )
