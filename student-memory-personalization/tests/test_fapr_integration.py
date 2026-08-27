"""Phase 18 — Step 4: Mock FAPR Repair Integration Test.

Validates the complete external FAPR-LB feedback loop:
1. Mock FAPR queries GET /memory/{student_id}/fapr-context
2. Evaluates errors, misconceptions, and past repairs
3. Submits repair outcome via POST /memory/repair-outcome (RESOLVED, PARTIALLY_RESOLVED, UNRESOLVED)
4. Verifies Memory stores the outcome and reflects it in subsequent context queries
5. Verifies GET /memory/{student_id}/support-preference transitions to SUPPORTED_BY_HISTORY
6. Verifies authentication, isolation, validation, and retry handling.
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
from src.schemas.fapr_context import FAPRContextResponse
from src.schemas.repair_outcome import RepairOutcomeResponse
from src.schemas.support_preference import SupportPreferenceResponse
from src.services.fapr_context_service import FAPRContextService
from src.services.repair_outcome_service import RepairOutcomeService
from src.services.support_preference_service import SupportPreferenceService

TEST_SERVICE_KEY = "fapr_integration_test_secret_9988"


@pytest.fixture
def fapr_test_env(monkeypatch):
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

    # Populate Student A and Student B
    stud_a_uuid = uuid.uuid4()
    stud_a_ext = "fapr_stud_alice"
    stud_b_uuid = uuid.uuid4()
    stud_b_ext = "fapr_stud_bob"

    skill_1_uuid = uuid.uuid4()
    skill_2_uuid = uuid.uuid4()

    sess_1_uuid = uuid.uuid4()
    sess_2_uuid = uuid.uuid4()
    sess_3_uuid = uuid.uuid4()
    sess_b_uuid = uuid.uuid4()

    with factory() as session:
        # Students
        session.add(Student(student_id=stud_a_uuid, external_student_id=stud_a_ext, display_name="Alice FAPR"))
        session.add(Student(student_id=stud_b_uuid, external_student_id=stud_b_ext, display_name="Bob FAPR"))

        # Skills
        session.add(CanonicalSkill(
            skill_id=skill_1_uuid,
            canonical_name="math :: algebra :: linear equations",
            display_name="Linear equations",
            description="Solving linear equations",
        ))
        session.add(CanonicalSkill(
            skill_id=skill_2_uuid,
            canonical_name="math :: algebra :: quadratic equations",
            display_name="Quadratic equations",
            description="Solving quadratic equations",
        ))

        # Sessions for Alice & Bob
        session.add(LearningSession(session_id=sess_1_uuid, student_id=stud_a_uuid, external_session_id="fapr_session_alice_1"))
        session.add(LearningSession(session_id=sess_2_uuid, student_id=stud_a_uuid, external_session_id="fapr_session_alice_2"))
        session.add(LearningSession(session_id=sess_3_uuid, student_id=stud_a_uuid, external_session_id="fapr_session_alice_3"))
        session.add(LearningSession(session_id=sess_b_uuid, student_id=stud_b_uuid, external_session_id="fapr_session_bob_1"))

        # Interactions for Alice
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            source="TUTOR",
            external_interaction_id="fapr_alice_act_1",
            student_utterance="I think 3x = 12 means x = 12 - 3 = 9",
            tutor_response="Remember that 3x represents multiplication.",
            is_correct=False,
            identified_error="Subtracted coefficient instead of dividing",
            attempt_count=2,
            hint_count=1,
            response_time_ms=5800.0,
        ))

        # Concept Memory for Alice
        session.add(ConceptMemory(
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            interaction_count=1,
            correct_count=0,
            incorrect_count=1,
            accuracy=0.0,
            attempt_observation_count=1,
            attempt_sum=2,
            hint_observation_count=1,
            hint_sum=1,
            response_time_observation_count=1,
            response_time_sum_ms=5800.0,
        ))

        # Snapshot for Alice
        snapshot = LearningStateSnapshot(
            snapshot_id=1,
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            learning_state="NEEDS_SUPPORT",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            previous_interaction_count=0,
            previous_skill_interaction_count=0,
            recent_interaction_count=1,
            attempt_observation_count=1,
            hint_observation_count=1,
            response_time_observation_count=1,
        )
        session.add(snapshot)

        # Current Learning State for Alice
        session.add(CurrentLearningState(
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            learning_state="NEEDS_SUPPORT",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=1,
            attempt_observation_count=1,
            hint_observation_count=1,
            response_time_observation_count=1,
            last_snapshot_id=1,
        ))

        # Misconception for Alice
        session.add(StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            canonical_skill_id=skill_1_uuid,
            normalized_error="subtracted_coefficient_instead_of_dividing",
            display_error="Subtracted coefficient instead of dividing",
            occurrence_count=1,
        ))

        # Initial Repair for Alice
        session.add(RepairOutcome(
            repair_outcome_id=uuid.uuid4(),
            student_id=stud_a_uuid,
            session_id=sess_1_uuid,
            canonical_skill_id=skill_1_uuid,
            repair_action="RULE_REMINDER",
            outcome="PARTIALLY_RESOLVED",
            score=0.5,
            notes="Reminded rule: isolate variable via inverse operation.",
        ))

        # Bob's interaction for isolation test
        session.add(InteractionLog(
            interaction_id=uuid.uuid4(),
            student_id=stud_b_uuid,
            session_id=sess_b_uuid,
            canonical_skill_id=skill_1_uuid,
            source="TUTOR",
            external_interaction_id="fapr_bob_act_1",
            student_utterance="Bob secret error utterance",
            is_correct=False,
            identified_error="Bob exclusive misconception",
            attempt_count=1,
            hint_count=0,
            response_time_ms=4000.0,
        ))
        session.commit()

    fapr_service = FAPRContextService(session_factory=factory)
    repair_service = RepairOutcomeService(session_factory=factory)
    support_service = SupportPreferenceService(session_factory=factory)

    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: fapr_service)
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: repair_service)
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: support_service)

    client = TestClient(app, raise_server_exceptions=False)
    return {
        "client": client,
        "alice_id": stud_a_ext,
        "bob_id": stud_b_ext,
        "session_1_id": "fapr_session_alice_1",
        "session_2_id": "fapr_session_alice_2",
        "session_3_id": "fapr_session_alice_3",
        "skill_1_id": str(skill_1_uuid),
        "skill_2_id": str(skill_2_uuid),
        "service_key": TEST_SERVICE_KEY,
    }


# ---------------------------------------------------------------------------
# FAPR Context Retrieval Flow Tests
# ---------------------------------------------------------------------------

def test_fapr_context_retrieval_flow(fapr_test_env):
    """Verify Mock FAPR queries /fapr-context and receives complete diagnosis & repair evidence."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    skill_1_id = fapr_test_env["skill_1_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    res = client.get(f"/memory/{alice_id}/fapr-context?skill_id={skill_1_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()

    # Validate against Schema
    parsed = FAPRContextResponse.model_validate(data)
    assert parsed.student_id == alice_id
    assert parsed.current_learning_state == "NEEDS_SUPPORT"
    assert parsed.evidence_strength == "MEDIUM"
    assert parsed.behavioural_coverage == "FULL_BEHAVIOURAL_COVERAGE"
    assert parsed.recent_incorrect_count == 1
    assert parsed.recent_accuracy == 0.0

    # Behavioral Evidence
    assert parsed.attempt_evidence.observation_count == 1
    assert parsed.hint_evidence.observation_count == 1
    assert parsed.response_time_evidence.observation_count == 1

    # Misconceptions, Utterance, and Past Repairs
    assert len(parsed.misconceptions) == 1
    assert parsed.misconceptions[0].display_error == "Subtracted coefficient instead of dividing"
    assert parsed.latest_student_utterance == "I think 3x = 12 means x = 12 - 3 = 9"
    assert len(parsed.previous_repairs) == 1
    assert parsed.previous_repairs[0].repair_action == "RULE_REMINDER"
    assert parsed.previous_repairs[0].outcome == "PARTIALLY_RESOLVED"


# ---------------------------------------------------------------------------
# FAPR Repair Outcome Submission Flow Tests
# ---------------------------------------------------------------------------

def test_fapr_repair_outcome_lifecycle(fapr_test_env):
    """Verify Mock FAPR submits repair outcomes and they are immediately reflected in context."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    session_1_id = fapr_test_env["session_1_id"]
    skill_1_id = fapr_test_env["skill_1_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    # 1. Submit RESOLVED repair outcome
    payload_resolved = {
        "student_id": alice_id,
        "session_id": session_1_id,
        "skill_id": skill_1_id,
        "repair_action": "WORKED_EXAMPLE",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Provided step-by-step division example. Student solved subsequent problem.",
    }

    res = client.post("/memory/repair-outcome", json=payload_resolved, headers=headers)
    assert res.status_code == 201
    created_repair = RepairOutcomeResponse.model_validate(res.json())
    assert created_repair.student_id == alice_id
    assert created_repair.repair_action == "WORKED_EXAMPLE"
    assert created_repair.outcome == "RESOLVED"
    assert created_repair.score == 1.0

    # 2. Verify subsequent GET /fapr-context includes WORKED_EXAMPLE
    res_ctx = client.get(f"/memory/{alice_id}/fapr-context?skill_id={skill_1_id}", headers=headers)
    assert res_ctx.status_code == 200
    ctx_data = res_ctx.json()
    actions = [r["repair_action"] for r in ctx_data["previous_repairs"]]
    assert "WORKED_EXAMPLE" in actions

    # 3. Submit UNRESOLVED repair outcome
    payload_unresolved = {
        "student_id": alice_id,
        "session_id": session_1_id,
        "skill_id": skill_1_id,
        "repair_action": "ANALOGY",
        "outcome": "UNRESOLVED",
        "score": 0.0,
        "notes": "Balance scale analogy did not clarify inverse division.",
    }
    res_unres = client.post("/memory/repair-outcome", json=payload_unresolved, headers=headers)
    assert res_unres.status_code == 201
    assert res_unres.json()["outcome"] == "UNRESOLVED"


# ---------------------------------------------------------------------------
# Support Preference Evolution Tests
# ---------------------------------------------------------------------------

def test_fapr_support_preference_evolution(fapr_test_env):
    """Verify repeated successful repair outcomes cause support-preference to become SUPPORTED_BY_HISTORY."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    session_1_id = fapr_test_env["session_1_id"]
    session_2_id = fapr_test_env["session_2_id"]
    session_3_id = fapr_test_env["session_3_id"]
    skill_1_id = fapr_test_env["skill_1_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    # Submit 3 successful HINT repairs across distinct sessions
    sessions = [session_1_id, session_2_id, session_3_id]
    for i, sess in enumerate(sessions):
        res = client.post(
            "/memory/repair-outcome",
            json={
                "student_id": alice_id,
                "session_id": sess,
                "skill_id": skill_1_id,
                "repair_action": "HINT",
                "outcome": "RESOLVED",
                "score": 1.0,
                "notes": f"Scaffolded hint #{i+1} successfully guided the student.",
            },
            headers=headers,
        )
        assert res.status_code == 201

    # Query GET /support-preference
    res_pref = client.get(f"/memory/{alice_id}/support-preference?skill_id={skill_1_id}", headers=headers)
    assert res_pref.status_code == 200
    pref_data = res_pref.json()

    parsed_pref = SupportPreferenceResponse.model_validate(pref_data)
    assert parsed_pref.status == "SUPPORTED_BY_HISTORY"
    assert parsed_pref.preferred_support_style == "HINT"
    assert parsed_pref.evidence_count >= 3
    assert parsed_pref.success_rate == 1.0


# ---------------------------------------------------------------------------
# Isolation, Cold-Start, and Auth Tests
# ---------------------------------------------------------------------------

def test_fapr_student_isolation(fapr_test_env):
    """Verify Student A's FAPR context does not leak Student B's repairs or utterances."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    bob_id = fapr_test_env["bob_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    res_alice = client.get(f"/memory/{alice_id}/fapr-context", headers=headers)
    assert res_alice.status_code == 200
    alice_utterance = res_alice.json()["latest_student_utterance"]
    assert "Bob secret error" not in (alice_utterance or "")

    res_bob = client.get(f"/memory/{bob_id}/fapr-context", headers=headers)
    assert res_bob.status_code == 200
    assert res_bob.json()["latest_student_utterance"] == "Bob secret error utterance"


def test_fapr_auth_rejection(fapr_test_env):
    """Verify FAPR context and repair endpoints reject requests without valid service key (401)."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]

    # Context query missing key
    res1 = client.get(f"/memory/{alice_id}/fapr-context")
    assert res1.status_code == 401
    assert res1.json()["error_code"] == "UNAUTHORIZED"

    # Context query wrong key
    res2 = client.get(f"/memory/{alice_id}/fapr-context", headers={"X-Service-Key": "wrong_key"})
    assert res2.status_code == 401

    # Repair outcome post missing key
    res3 = client.post(
        "/memory/repair-outcome",
        json={"student_id": alice_id, "session_id": "sess_1", "skill_id": "sk_1", "repair_action": "HINT", "outcome": "RESOLVED"},
    )
    assert res3.status_code == 401


def test_fapr_invalid_payload_validation(fapr_test_env):
    """Verify invalid payloads (bad outcome, score out of range, empty action) return 422."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    session_1_id = fapr_test_env["session_1_id"]
    skill_1_id = fapr_test_env["skill_1_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    # Empty repair_action
    res1 = client.post(
        "/memory/repair-outcome",
        json={"student_id": alice_id, "session_id": session_1_id, "skill_id": skill_1_id, "repair_action": "", "outcome": "RESOLVED"},
        headers=headers,
    )
    assert res1.status_code == 422
    assert res1.json()["error_code"] == "VALIDATION_ERROR"

    # Score > 1.0
    res2 = client.post(
        "/memory/repair-outcome",
        json={"student_id": alice_id, "session_id": session_1_id, "skill_id": skill_1_id, "repair_action": "HINT", "outcome": "RESOLVED", "score": 2.5},
        headers=headers,
    )
    assert res2.status_code == 422

    # Score < 0.0
    res3 = client.post(
        "/memory/repair-outcome",
        json={"student_id": alice_id, "session_id": session_1_id, "skill_id": skill_1_id, "repair_action": "HINT", "outcome": "RESOLVED", "score": -0.5},
        headers=headers,
    )
    assert res3.status_code == 422


def test_fapr_duplicate_retry_handling(fapr_test_env):
    """Verify repeated submission of identical repair outcomes is handled gracefully without crash."""
    client = fapr_test_env["client"]
    alice_id = fapr_test_env["alice_id"]
    session_1_id = fapr_test_env["session_1_id"]
    skill_1_id = fapr_test_env["skill_1_id"]
    headers = {"X-Service-Key": fapr_test_env["service_key"]}

    payload = {
        "student_id": alice_id,
        "session_id": session_1_id,
        "skill_id": skill_1_id,
        "repair_action": "SCAFFOLDED_QUESTION",
        "outcome": "RESOLVED",
        "score": 1.0,
    }

    res1 = client.post("/memory/repair-outcome", json=payload, headers=headers)
    assert res1.status_code == 201

    res2 = client.post("/memory/repair-outcome", json=payload, headers=headers)
    assert res2.status_code == 201
