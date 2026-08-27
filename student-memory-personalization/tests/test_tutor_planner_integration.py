"""Phase 18 — Step 3: Mock Tutor + Planner Consumer Integration Test.

Validates that external Tutor and Planner agents consume their dedicated context projections
(GET /memory/{student_id}/tutor-context and GET /memory/{student_id}/planner-context)
under production authentication and parameter filtering policies.
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    RepairOutcome,
    StudentMisconception,
)
from src.schemas.planner_context import PlannerContextResponse
from src.schemas.tutor_context import TutorContextResponse
from src.services.planner_context_service import PlannerContextService
from src.services.tutor_context_service import TutorContextService

TEST_SERVICE_KEY = "tutor_planner_test_service_key_7788"


@pytest.fixture
def tutor_planner_env(monkeypatch):
    """Set up in-memory SQLite schema, populate realistic student history, and inject services."""
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", TEST_SERVICE_KEY)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Populate Student A and B
    stud_a_uuid = uuid.uuid4()
    stud_a_ext = "stud_tp_alice"
    stud_b_uuid = uuid.uuid4()
    stud_b_ext = "stud_tp_bob"

    skill_1_uuid = uuid.uuid4()
    skill_2_uuid = uuid.uuid4()

    sess_1_uuid = uuid.uuid4()
    sess_2_uuid = uuid.uuid4()
    sess_b_uuid = uuid.uuid4()

    with factory() as session:
        # Students
        session.add(Student(student_id=stud_a_uuid, external_student_id=stud_a_ext, display_name="Alice"))
        session.add(Student(student_id=stud_b_uuid, external_student_id=stud_b_ext, display_name="Bob"))

        # Skills
        session.add(CanonicalSkill(
            skill_id=skill_1_uuid,
            canonical_name="math :: algebra :: linear equations",
            display_name="Linear equations",
            description="Solving one and two-step linear equations",
        ))
        session.add(CanonicalSkill(
            skill_id=skill_2_uuid,
            canonical_name="math :: statistics :: median",
            display_name="Median",
            description="Calculating statistical median",
        ))

        # Sessions for Alice & Bob
        session.add(LearningSession(session_id=sess_1_uuid, student_id=stud_a_uuid, external_session_id="session_alice_1"))
        session.add(LearningSession(session_id=sess_2_uuid, student_id=stud_a_uuid, external_session_id="session_alice_2"))
        session.add(LearningSession(session_id=sess_b_uuid, student_id=stud_b_uuid, external_session_id="session_bob_1"))

        # Raw Interactions for Alice (Session 1: Skill 1)
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            source="ASSESSMENT",
            external_interaction_id="alice_act_1",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=3200.0,
        ))
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            source="ASSESSMENT",
            external_interaction_id="alice_act_2",
            is_correct=False,
            identified_error="Subtracted instead of dividing",
            attempt_count=2,
            hint_count=1,
            response_time_ms=6400.0,
        ))
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_2_uuid,
            canonical_skill_id=skill_2_uuid,
            source="ASSESSMENT",
            external_interaction_id="alice_act_3",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            response_time_ms=2900.0,
        ))

        # Short Term Memory for Alice in Session 1
        session.add(ShortTermMemory(
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            interaction_count=2,
            correct_count=1,
            incorrect_count=1,
            recent_accuracy=0.5,
            attempt_observation_count=2,
            attempt_sum=3,
            hint_observation_count=2,
            hint_sum=1,
            response_time_observation_count=2,
            response_time_sum_ms=9600.0,
        ))

        # Concept Memory for Alice on Skill 1
        session.add(ConceptMemory(
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            interaction_count=2,
            correct_count=1,
            incorrect_count=1,
            accuracy=0.5,
            attempt_observation_count=2,
            attempt_sum=3,
            hint_observation_count=2,
            hint_sum=1,
            response_time_observation_count=2,
            response_time_sum_ms=9600.0,
        ))

        # Long Term Memory for Alice
        session.add(LongTermMemory(
            student_id=stud_a_uuid,
            total_sessions=2,
            concept_count=2,
            interaction_count=3,
            correct_count=2,
            incorrect_count=1,
            overall_accuracy=0.667,
            attempt_observation_count=3,
            attempt_sum=4,
            hint_observation_count=3,
            hint_sum=1,
            response_time_observation_count=3,
            response_time_sum_ms=12500.0,
        ))

        # Learning State Snapshot for Alice
        snapshot = LearningStateSnapshot(
            snapshot_id=1,
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            learning_state="DEVELOPING",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            previous_interaction_count=0,
            previous_skill_interaction_count=0,
            recent_interaction_count=2,
            attempt_observation_count=2,
            hint_observation_count=2,
            response_time_observation_count=2,
        )
        session.add(snapshot)

        # Current Learning State for Alice on Skill 1
        session.add(CurrentLearningState(
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            learning_state="DEVELOPING",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=2,
            attempt_observation_count=2,
            hint_observation_count=2,
            response_time_observation_count=2,
            last_snapshot_id=1,
        ))

        # Misconception for Alice
        session.add(StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            normalized_error="subtracted_instead_of_dividing",
            display_error="Subtracted instead of dividing",
            occurrence_count=1,
        ))

        # Repair Outcome for Alice
        session.add(RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            repair_action="HINT",
            outcome="RESOLVED",
            score=1.0,
            notes="Student solved equation after reminder.",
        ))

        # Bob's separate interaction (Student Isolation Test)
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_b_uuid,
            session_id=sess_b_uuid,
            canonical_skill_id=skill_1_uuid,
            source="ASSESSMENT",
            external_interaction_id="bob_act_1",
            is_correct=False,
            identified_error="Bob specific error",
            attempt_count=3,
            hint_count=2,
            response_time_ms=9000.0,
        ))
        session.commit()

    tutor_service = TutorContextService(session_factory=factory)
    planner_service = PlannerContextService(session_factory=factory)

    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: tutor_service)
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: planner_service)

    client = TestClient(app, raise_server_exceptions=False)
    return {
        "client": client,
        "alice_id": stud_a_ext,
        "bob_id": stud_b_ext,
        "session_1_id": "session_alice_1",
        "skill_1_id": str(skill_1_uuid),
        "skill_2_id": str(skill_2_uuid),
        "service_key": TEST_SERVICE_KEY,
    }


# ---------------------------------------------------------------------------
# Tutor Consumer Flow Tests
# ---------------------------------------------------------------------------

def test_tutor_consumer_flow(tutor_planner_env):
    """Verify Mock Tutor queries /tutor-context and receives structured evidence."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    skill_1_id = tutor_planner_env["skill_1_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    res = client.get(f"/memory/{alice_id}/tutor-context?skill_id={skill_1_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()

    # Validate against Schema
    parsed = TutorContextResponse.model_validate(data)
    assert parsed.student_id == alice_id
    assert parsed.current_learning_state == "DEVELOPING"
    assert parsed.evidence_strength == "MEDIUM"
    assert parsed.behavioural_coverage == "FULL_BEHAVIOURAL_COVERAGE"
    assert parsed.recent_correct_count == 1
    assert parsed.recent_incorrect_count == 1
    assert parsed.recent_accuracy == 0.5

    # Behavioral Evidence
    assert parsed.attempt_evidence.observation_count == 2
    assert parsed.attempt_evidence.total_value == 3.0
    assert parsed.hint_evidence.observation_count == 2
    assert parsed.response_time_evidence.observation_count == 2

    # Misconceptions & Repairs
    assert len(parsed.misconceptions) == 1
    assert parsed.misconceptions[0].display_error == "Subtracted instead of dividing"
    assert len(parsed.recent_repairs) == 1
    assert parsed.recent_repairs[0].repair_action == "HINT"


# ---------------------------------------------------------------------------
# Planner Consumer Flow Tests
# ---------------------------------------------------------------------------

def test_planner_consumer_flow(tutor_planner_env):
    """Verify Mock Planner queries /planner-context and receives longitudinal statistics."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    skill_1_id = tutor_planner_env["skill_1_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    res = client.get(f"/memory/{alice_id}/planner-context?skill_id={skill_1_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()

    # Validate against Schema
    parsed = PlannerContextResponse.model_validate(data)
    assert parsed.student_id == alice_id
    assert parsed.total_sessions == 2
    assert parsed.concept_count == 2
    assert parsed.long_term_interaction_count == 3
    assert parsed.overall_accuracy == pytest.approx(0.667, rel=1e-2)
    assert parsed.current_learning_state == "DEVELOPING"
    assert parsed.concept_interaction_count == 2
    assert parsed.concept_accuracy == 0.5
    assert len(parsed.misconceptions) == 1
    assert len(parsed.recent_interactions) == 2


# ---------------------------------------------------------------------------
# Session & Skill Filtering Tests
# ---------------------------------------------------------------------------

def test_session_filtering(tutor_planner_env):
    """Verify query parameter session_id filters context to only that session."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    session_1_id = tutor_planner_env["session_1_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    # Filter Tutor Context
    res_tutor = client.get(
        f"/memory/{alice_id}/tutor-context?session_id={session_1_id}",
        headers=headers,
    )
    assert res_tutor.status_code == 200
    tutor_data = res_tutor.json()
    assert len(tutor_data["recent_interactions"]) == 2
    assert tutor_data["recent_correct_count"] == 1
    assert tutor_data["recent_incorrect_count"] == 1

    # Filter Planner Context
    res_plan = client.get(
        f"/memory/{alice_id}/planner-context?session_id={session_1_id}",
        headers=headers,
    )
    assert res_plan.status_code == 200
    plan_data = res_plan.json()
    assert plan_data["session_interaction_count"] == 2
    assert plan_data["session_correct_count"] == 1
    assert plan_data["session_incorrect_count"] == 1
    assert plan_data["session_accuracy"] == 0.5


def test_skill_filtering(tutor_planner_env):
    """Verify query parameter skill_id filters concept-level statistics to that skill."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    skill_1_id = tutor_planner_env["skill_1_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    res = client.get(
        f"/memory/{alice_id}/planner-context?skill_id={skill_1_id}",
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["concept_interaction_count"] == 2
    assert data["concept_correct_count"] == 1
    assert data["concept_incorrect_count"] == 1
    assert data["concept_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Cold-Start, Isolation, and Auth Invariant Tests
# ---------------------------------------------------------------------------

def test_cold_start_no_history_student(tutor_planner_env):
    """Verify cold start student with zero records returns safe empty context with 200."""
    client = tutor_planner_env["client"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    # Tutor Context
    res_t = client.get("/memory/cold_start_newbie/tutor-context", headers=headers)
    assert res_t.status_code == 200
    t_data = res_t.json()
    assert t_data["current_learning_state"] is None
    assert t_data["recent_correct_count"] == 0
    assert t_data["misconceptions"] == []
    assert t_data["recent_repairs"] == []

    # Planner Context
    res_p = client.get("/memory/cold_start_newbie/planner-context", headers=headers)
    assert res_p.status_code == 200
    p_data = res_p.json()
    assert p_data["total_sessions"] == 0
    assert p_data["concept_count"] == 0
    assert p_data["long_term_interaction_count"] == 0


def test_student_isolation(tutor_planner_env):
    """Verify Student A's context never leaks Student B's data."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    bob_id = tutor_planner_env["bob_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    res_alice = client.get(f"/memory/{alice_id}/tutor-context", headers=headers)
    assert res_alice.status_code == 200
    alice_errors = [m["display_error"] for m in res_alice.json()["misconceptions"]]
    assert "Bob specific error" not in alice_errors

    res_bob = client.get(f"/memory/{bob_id}/tutor-context", headers=headers)
    assert res_bob.status_code == 200
    bob_interactions = res_bob.json()["recent_interactions"]
    assert len(bob_interactions) == 1
    assert bob_interactions[0]["is_correct"] is False
    assert bob_interactions[0]["attempt_count"] == 3


def test_auth_rejection(tutor_planner_env):
    """Verify endpoints reject requests with missing or invalid service keys (401)."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]

    # Missing Key
    res_t_no_key = client.get(f"/memory/{alice_id}/tutor-context")
    assert res_t_no_key.status_code == 401
    assert res_t_no_key.json()["error_code"] == "UNAUTHORIZED"

    res_p_no_key = client.get(f"/memory/{alice_id}/planner-context")
    assert res_p_no_key.status_code == 401
    assert res_p_no_key.json()["error_code"] == "UNAUTHORIZED"

    # Invalid Key
    bad_headers = {"X-Service-Key": "invalid_tutor_key"}
    res_t_bad = client.get(f"/memory/{alice_id}/tutor-context", headers=bad_headers)
    assert res_t_bad.status_code == 401

    res_p_bad = client.get(f"/memory/{alice_id}/planner-context", headers=bad_headers)
    assert res_p_bad.status_code == 401


def test_invalid_limit_validation(tutor_planner_env):
    """Verify limit outside bounds (1-100) returns standardized 422 VALIDATION_ERROR."""
    client = tutor_planner_env["client"]
    alice_id = tutor_planner_env["alice_id"]
    headers = {"X-Service-Key": tutor_planner_env["service_key"]}

    # Limit = 0 (below minimum 1)
    res_zero = client.get(f"/memory/{alice_id}/tutor-context?limit=0", headers=headers)
    assert res_zero.status_code == 422
    assert res_zero.json()["error_code"] == "VALIDATION_ERROR"

    # Limit = 500 (above maximum 100)
    res_large = client.get(f"/memory/{alice_id}/planner-context?limit=500", headers=headers)
    assert res_large.status_code == 422
    assert res_large.json()["error_code"] == "VALIDATION_ERROR"
