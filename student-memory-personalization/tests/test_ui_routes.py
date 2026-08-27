"""Tests for frontend-safe /ui endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.schemas.question_context import (
    QuestionContextRequest,
    QuestionContextResponse,
    TopicContextItem,
)


class DummyQuestionContextService:
    def get_question_context(self, request: QuestionContextRequest) -> QuestionContextResponse:
        return QuestionContextResponse(
            student_id=request.student_id,
            session_id=request.session_id,
            question=request.question,
            topic=TopicContextItem(
                skill_id="skill-alg-101",
                canonical_skill_name="math :: algebra :: linear equations",
                display_name="Linear Equations",
                confidence=0.95,
                method="fine_tuned_minilm",
                needs_review=False,
            ),
            short_term_memory={"recent_skill_ids": ["skill-alg-101"]},
            long_term_memory={"mastery_level": 0.8},
            concept_memory={"interaction_count": 5},
            misconceptions=[],
        )


@pytest.fixture
def ui_client(monkeypatch):
    import src.api.ui_routes as ui_routes
    monkeypatch.setattr(ui_routes, "get_question_context_service", lambda: DummyQuestionContextService())
    return TestClient(app, raise_server_exceptions=False)


def test_ui_question_context_allows_request_without_service_key(ui_client):
    """Verify /ui/question-context does not require X-Service-Key header."""
    payload = {
        "student_id": "stud_ui_001",
        "session_id": "sess_ui_001",
        "question": "How to solve 2x + 4 = 10?",
    }
    response = ui_client.post("/ui/question-context", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "stud_ui_001"
    assert data["topic"]["display_name"] == "Linear Equations"
    assert data["topic"]["confidence"] == 0.95


def test_ui_question_context_validates_payload(ui_client):
    """Verify validation errors return standardized 422 error response."""
    payload = {
        "student_id": "",
        "session_id": "sess_ui_001",
        "question": "",
    }
    response = ui_client.post("/ui/question-context", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
