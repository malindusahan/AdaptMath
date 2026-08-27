"""Tests for frontend-safe /ui inspector endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.schemas.fapr_context import FAPRContextResponse
from src.schemas.meta_signals import MetaSignalItem, MetaSignalsResponse
from src.schemas.planner_context import PlannerContextResponse
from src.schemas.student_context import StudentContextResponse
from src.schemas.support_preference import SupportPreferenceResponse, SupportStrategySummary
from src.schemas.tutor_context import TutorContextResponse


class DummyStudentContextService:
    def get_student_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20):
        return StudentContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            learning_state={"learning_state": "DEVELOPING"},
            short_term_memory={"recent_interaction_count": 3},
            long_term_memory={"mastery_level": 0.75},
            concept_memory={"interaction_count": 8},
            misconceptions=[],
            recent_interactions=[],
            repair_history=[],
        )


class DummyTutorContextService:
    def get_tutor_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20):
        return TutorContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            current_learning_state="STRONG",
            evidence_strength="0.85",
            behavioural_coverage="HIGH",
            recent_accuracy=0.9,
            recent_correct_count=4,
            recent_incorrect_count=1,
            misconceptions=[],
            recent_interactions=[],
            recent_repairs=[],
        )


class DummyPlannerContextService:
    def get_planner_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20):
        return PlannerContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            session_interaction_count=10,
            session_accuracy=0.8,
            current_learning_state="STRONG",
            concept_interaction_count=5,
            concept_accuracy=0.8,
            long_term_interaction_count=20,
            overall_accuracy=0.75,
            total_sessions=3,
            concept_count=2,
            misconceptions=[],
            recent_interactions=[],
        )


class DummyFAPRContextService:
    def get_fapr_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20):
        return FAPRContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            current_learning_state="NEEDS_SUPPORT",
            recent_accuracy=0.4,
            recent_incorrect_count=2,
            misconceptions=[],
            recent_interactions=[],
            previous_repairs=[],
            latest_student_utterance="I am confused about this step",
        )


class DummySupportPreferenceService:
    def get_support_preference(self, student_id: str, skill_id: str | None = None):
        return SupportPreferenceResponse(
            student_id=student_id,
            skill_id=skill_id,
            preferred_support_style="HINT",
            status="SUPPORTED_BY_HISTORY",
            evidence_count=5,
            success_rate=0.8,
            strategies=[
                SupportStrategySummary(
                    repair_action="HINT",
                    observation_count=5,
                    successful_count=4,
                    partial_count=1,
                    failed_count=0,
                    success_rate=0.8,
                    average_score=0.85,
                )
            ],
        )


class DummyMetaSignalService:
    def get_meta_signals(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20):
        return MetaSignalsResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            signals=[
                MetaSignalItem(
                    signal_type="correct_answer",
                    student_id=student_id,
                    session_id=session_id,
                    timestamp="2026-08-18T10:00:00Z",
                    confidence=1.0,
                    evidence={"score": 1.0},
                )
            ],
            total_signals=1,
        )


@pytest.fixture
def inspector_client(monkeypatch):
    import src.api.ui_routes as ui_routes
    monkeypatch.setattr(ui_routes, "get_student_context_service", lambda: DummyStudentContextService())
    monkeypatch.setattr(ui_routes, "get_tutor_context_service", lambda: DummyTutorContextService())
    monkeypatch.setattr(ui_routes, "get_planner_context_service", lambda: DummyPlannerContextService())
    monkeypatch.setattr(ui_routes, "get_fapr_context_service", lambda: DummyFAPRContextService())
    monkeypatch.setattr(ui_routes, "get_support_preference_service", lambda: DummySupportPreferenceService())
    monkeypatch.setattr(ui_routes, "get_meta_signal_service", lambda: DummyMetaSignalService())
    return TestClient(app, raise_server_exceptions=False)


def test_ui_get_student_context(inspector_client):
    res = inspector_client.get("/ui/stud_01/context?session_id=sess_01&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert data["session_id"] == "sess_01"
    assert data["long_term_memory"]["mastery_level"] == 0.75


def test_ui_get_tutor_context(inspector_client):
    res = inspector_client.get("/ui/stud_01/tutor-context")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert data["current_learning_state"] == "STRONG"
    assert data["recent_accuracy"] == 0.9


def test_ui_get_planner_context(inspector_client):
    res = inspector_client.get("/ui/stud_01/planner-context")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert data["current_learning_state"] == "STRONG"
    assert data["session_interaction_count"] == 10


def test_ui_get_fapr_context(inspector_client):
    res = inspector_client.get("/ui/stud_01/fapr-context")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert data["recent_incorrect_count"] == 2
    assert data["latest_student_utterance"] == "I am confused about this step"


def test_ui_get_support_preference(inspector_client):
    res = inspector_client.get("/ui/stud_01/support-preference")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert data["preferred_support_style"] == "HINT"
    assert data["status"] == "SUPPORTED_BY_HISTORY"


def test_ui_get_meta_signals(inspector_client):
    res = inspector_client.get("/ui/stud_01/meta-signals")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "stud_01"
    assert len(data["signals"]) == 1
    assert data["signals"][0]["signal_type"] == "correct_answer"
