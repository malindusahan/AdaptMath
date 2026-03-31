"""Tests for frontend-safe /ui/memory-update endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    MemoryUpdateResponse,
)
from src.services.memory_update_service import MemoryUpdateServiceError


@pytest.fixture
def memory_update_client(monkeypatch):
    import src.api.ui_routes as ui_routes

    def mock_update(request: AssessmentMemoryUpdateRequest) -> MemoryUpdateResponse:
        if request.student_id == "force_fail_student":
            raise MemoryUpdateServiceError("Database persistence failed on write lock.")

        correct_count = sum(1 for q in request.assessment_questions if q.is_correct)
        total = len(request.assessment_questions)
        learning_state = "STRONG" if correct_count == total else "DEVELOPING" if correct_count > 0 else "NEEDS_SUPPORT"

        return MemoryUpdateResponse(
            student_id=request.student_id,
            topic=request.topic,
            subtopic=request.subtopic,
            assessment_id=901,
            snapshot_id=1201,
            learning_state=learning_state,
            evidence_level="PARTIAL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE"
            if all(q.attempt_count is not None for q in request.assessment_questions)
            else "CORRECTNESS_ONLY_COVERAGE",
            model_used=True,
            previous_interaction_count=0,
            previous_skill_interaction_count=0,
            recent_interaction_count=total,
            attempt_observation_count=sum(1 for q in request.assessment_questions if q.attempt_count is not None),
            hint_observation_count=sum(1 for q in request.assessment_questions if q.hint_count is not None),
            response_time_observation_count=sum(1 for q in request.assessment_questions if q.response_time_ms is not None),
            misconception_count=len(request.identified_errors),
            memory_updated=True,
        )

    monkeypatch.setattr(ui_routes, "update_student_memory", mock_update)
    return TestClient(app, raise_server_exceptions=False)


def test_ui_memory_update_single_question_success(memory_update_client):
    """Verify single correct question assessment update succeeds."""
    payload = {
        "student_id": "stud_ui_test",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 4200.0,
            }
        ],
        "identified_errors": [],
        "overall_feedback": "Assessment submitted from React demo.",
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "stud_ui_test"
    assert data["learning_state"] == "STRONG"
    assert data["recent_interaction_count"] == 1
    assert data["behavioural_coverage"] == "FULL_BEHAVIOURAL_COVERAGE"


def test_ui_memory_update_multiple_questions_success(memory_update_client):
    """Verify multi-question assessment update succeeds."""
    payload = {
        "student_id": "stud_ui_multi",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 3500.0,
            },
            {
                "question_id": "q2",
                "question": "Solve 2x + 4 = 10",
                "student_answer": "3",
                "expected_answer": "3",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 4100.0,
            },
        ],
        "identified_errors": [],
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recent_interaction_count"] == 2
    assert data["learning_state"] == "STRONG"


def test_ui_memory_update_incorrect_with_misconceptions(memory_update_client):
    """Verify incorrect question with identified errors records misconceptions."""
    payload = {
        "student_id": "stud_ui_err",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "student_answer": "12",
                "expected_answer": "5",
                "is_correct": False,
                "identified_error": "Subtracted instead of dividing",
                "attempt_count": 2,
                "hint_count": 1,
                "response_time_ms": 7800.0,
            }
        ],
        "identified_errors": ["Subtracted instead of dividing"],
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["learning_state"] == "NEEDS_SUPPORT"
    assert data["misconception_count"] == 1


def test_ui_memory_update_optional_behavioral_fields(memory_update_client):
    """Verify omitted/null behavioral values are preserved without converting to fake zeros."""
    payload = {
        "student_id": "stud_ui_nulls",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "is_correct": True,
            }
        ],
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["behavioural_coverage"] == "CORRECTNESS_ONLY_COVERAGE"
    assert data["attempt_observation_count"] == 0
    assert data["hint_observation_count"] == 0
    assert data["response_time_observation_count"] == 0


def test_ui_memory_update_validation_error(memory_update_client):
    """Verify missing required fields return standardized 422 error."""
    payload = {
        "student_id": "",
        "topic": "",
        "assessment_questions": [],
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


def test_ui_memory_update_database_failure_sanitized(memory_update_client):
    """Verify backend database failure returns standardized 500 error without leaking secrets."""
    payload = {
        "student_id": "force_fail_student",
        "topic": "Algebra",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "is_correct": True,
            }
        ],
    }
    response = memory_update_client.post("/ui/memory-update", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == "INTERNAL_ERROR"
    assert data["message"] == "Memory update failed."
