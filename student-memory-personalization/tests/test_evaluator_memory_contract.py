"""Comprehensive integration tests for the Evaluator -> Memory update contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=True)


def test_evaluator_multi_question_update_flow(client: TestClient):
    """Verify that an Evaluator assessment update processes multiple questions and returns updated learning state."""
    payload = {
        "student_id": "eval_test_student_01",
        "topic": "Linear Equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "What is x if 3x = 15?",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "identified_error": None,
                "attempt_count": 1,
                "hint_count": 0,
                "hint_total": 2,
                "response_time_ms": 3500.0,
            },
            {
                "question_id": "q2",
                "question": "Solve x - 2 = 7.",
                "student_answer": "x = 8",
                "expected_answer": "x = 9",
                "is_correct": False,
                "identified_error": "addition sign error",
                "attempt_count": 2,
                "hint_count": 1,
                "hint_total": 2,
                "response_time_ms": 6200.0,
            },
        ],
        "identified_errors": ["addition sign error"],
        "overall_feedback": "Mostly correct, but review addition in linear equations.",
    }

    # 1. Post memory update
    response = client.post(
        "/memory/update",
        json=payload,
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["student_id"] == "eval_test_student_01"
    assert data["topic"] == "Linear Equations"
    assert data["learning_state"] in ("NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE")
    assert data["evidence_level"] in ("COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL")
    assert data["misconception_count"] >= 1
    assert "addition sign error" in data["misconceptions"]
    assert data["memory_updated"] is True

    # 2. Retrieve tutor context to verify persistence
    tutor_resp = client.get(
        f"/memory/eval_test_student_01/tutor-context",
        headers={"X-Service-Key": "test-service-key"},
    )
    assert tutor_resp.status_code == 200
    tutor_data = tutor_resp.json()
    assert tutor_data["student_id"] == "eval_test_student_01"
    assert len(tutor_data["misconceptions"]) >= 1


def test_evaluator_string_student_id_resolution(client: TestClient):
    """Verify arbitrary external student IDs (e.g., '999001', 's1') resolve automatically."""
    for ext_id in ["999001", "s1", "STU-ALPHA-99"]:
        payload = {
            "student_id": ext_id,
            "topic": "Pythagorean Theorem",
            "assessment_questions": [
                {
                    "question_id": "q_pyth_1",
                    "question": "What is hypotenuse of legs 3 and 4?",
                    "student_answer": "5",
                    "expected_answer": "5",
                    "is_correct": True,
                }
            ],
            "identified_errors": [],
        }
        resp = client.post(
            "/memory/update",
            json=payload,
            headers={"X-Service-Key": "test-service-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["student_id"] == ext_id
        assert resp.json()["memory_updated"] is True


def test_database_memory_tables_have_readable_names(client: TestClient):
    """Verify that ConceptMemory, STM, LTM, and CurrentLearningState store readable skill and student names."""
    from sqlalchemy import select
    from src.database.models.core import Student
    from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
    from src.database.models.supporting_memory import CurrentLearningState
    from src.database.postgres_session import get_session_factory
    from src.database.unit_of_work import UnitOfWork

    student_id = "test_readable_names_student"
    payload = {
        "student_id": student_id,
        "topic": "Linear Equations",
        "assessment_questions": [
            {
                "question_id": "q_read_1",
                "question": "Solve 2x = 10",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
            }
        ],
        "identified_errors": [],
    }

    resp = client.post(
        "/memory/update",
        json=payload,
        headers={"X-Service-Key": "test-service-key"},
    )
    assert resp.status_code == 200

    with UnitOfWork(get_session_factory()) as uow:
        student = uow.identities.get_student(student_id)
        assert student is not None

        # Check Concept Memory
        concept_stmt = select(ConceptMemory).where(ConceptMemory.student_id == student.student_id)
        concept = uow.session.scalars(concept_stmt).first()
        assert concept is not None
        assert concept.student_external_id == student_id
        assert concept.skill_name == "Linear Equations"

        # Check Short-Term Memory
        stm_stmt = select(ShortTermMemory).where(ShortTermMemory.student_id == student.student_id)
        stm = uow.session.scalars(stm_stmt).first()
        assert stm is not None
        assert stm.student_external_id == student_id
        assert stm.skill_name == "Linear Equations"

        # Check Long-Term Memory
        ltm = uow.session.get(LongTermMemory, student.student_id)
        assert ltm is not None
        assert ltm.student_external_id == student_id

        # Check Current Learning State
        state_stmt = select(CurrentLearningState).where(CurrentLearningState.student_id == student.student_id)
        curr_state = uow.session.scalars(state_stmt).first()
        assert curr_state is not None
        assert curr_state.student_external_id == student_id
        assert curr_state.skill_name == "Linear Equations"

