"""Tests for frontend-safe /ui/repair-outcome endpoint."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.api.app import app
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)


class DummyRepairOutcomeService:
    def record_repair_outcome(self, request: RepairOutcomeCreateRequest) -> RepairOutcomeResponse:
        if request.student_id == "unknown_student":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Referenced student not found.",
            )
        if request.student_id == "mismatched_student":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session belongs to a different student.",
            )
        if request.student_id == "db_fail_student":
            raise RuntimeError("Database connection reset during repair write.")

        return RepairOutcomeResponse(
            repair_outcome_id="rep-101-uuid",
            student_id=request.student_id,
            session_id=request.session_id,
            canonical_skill_id=request.skill_id,
            interaction_id=request.interaction_id,
            repair_action=request.repair_action,
            outcome=request.outcome,
            score=request.score,
            notes=request.notes,
            created_at="2026-08-18T16:40:00Z",
        )


@pytest.fixture
def repair_client(monkeypatch):
    import src.api.ui_routes as ui_routes
    monkeypatch.setattr(ui_routes, "get_repair_outcome_service", lambda: DummyRepairOutcomeService())
    return TestClient(app, raise_server_exceptions=False)


def test_ui_repair_outcome_valid_resolved(repair_client):
    """Verify valid resolved repair outcome is stored with HTTP 201."""
    payload = {
        "student_id": "stud_101",
        "session_id": "sess_202",
        "skill_id": "skill_303",
        "repair_action": "inverse_operation_prompt",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Student applied inverse operations correctly.",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["repair_outcome_id"] == "rep-101-uuid"
    assert data["outcome"] == "RESOLVED"
    assert data["score"] == 1.0


def test_ui_repair_outcome_partial_resolution(repair_client):
    """Verify partially resolved repair outcome with partial score."""
    payload = {
        "student_id": "stud_101",
        "session_id": "sess_202",
        "skill_id": "skill_303",
        "repair_action": "step_by_step_hint",
        "outcome": "PARTIALLY_RESOLVED",
        "score": 0.5,
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["outcome"] == "PARTIALLY_RESOLVED"
    assert data["score"] == 0.5
    assert data["notes"] is None


def test_ui_repair_outcome_optional_notes(repair_client):
    """Verify notes can be omitted or provided."""
    payload = {
        "student_id": "stud_101",
        "session_id": "sess_202",
        "skill_id": "skill_303",
        "repair_action": "worked_example",
        "outcome": "RESOLVED",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["repair_action"] == "worked_example"


def test_ui_repair_outcome_invalid_ownership(repair_client):
    """Verify cross-student session mismatch returns standardized 400 error."""
    payload = {
        "student_id": "mismatched_student",
        "session_id": "sess_other",
        "skill_id": "skill_303",
        "repair_action": "hint",
        "outcome": "UNRESOLVED",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "BAD_REQUEST"


def test_ui_repair_outcome_invalid_skill_or_student_not_found(repair_client):
    """Verify unknown student returns standardized 404 error."""
    payload = {
        "student_id": "unknown_student",
        "session_id": "sess_202",
        "skill_id": "skill_303",
        "repair_action": "hint",
        "outcome": "UNRESOLVED",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "NOT_FOUND"


def test_ui_repair_outcome_validation_failure(repair_client):
    """Verify missing required fields return standardized 422 error."""
    payload = {
        "student_id": "",
        "session_id": "",
        "skill_id": "",
        "repair_action": "",
        "outcome": "",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


def test_ui_repair_outcome_db_failure_sanitized(repair_client):
    """Verify database runtime error is sanitized to standardized 500 without leaking secrets."""
    payload = {
        "student_id": "db_fail_student",
        "session_id": "sess_202",
        "skill_id": "skill_303",
        "repair_action": "hint",
        "outcome": "RESOLVED",
    }
    response = repair_client.post("/ui/repair-outcome", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == "INTERNAL_ERROR"
    assert data["message"] == "Failed to record repair outcome."
