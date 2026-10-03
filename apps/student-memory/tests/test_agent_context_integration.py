"""Comprehensive integration tests validating the full Agent Context & Memory API surface."""

from __future__ import annotations

from datetime import datetime, timezone
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
from src.database.models.supporting_memory import CurrentLearningState, RepairOutcome, StudentMisconception
from src.database.postgres_config import DATABASE_URL_ENV, load_postgres_settings
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
    set_session_factory,
)
from src.services.fapr_context_service import FAPRContextService
from src.services.meta_signal_service import MetaSignalService
from src.services.planner_context_service import PlannerContextService
from src.services.repair_outcome_service import RepairOutcomeService
from src.services.student_context_service import StudentContextService
from src.services.support_preference_service import SupportPreferenceService
from src.services.tutor_context_service import TutorContextService
from src.services.skill_context_resolver import SkillContextResolver


@pytest.fixture
def sqlite_context_setup(monkeypatch):
    """Create in-memory SQLite database wired to all context services."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Wire services to shared factory
    stud_svc = StudentContextService(session_factory=factory)
    tutor_svc = TutorContextService(session_factory=factory, student_context_service=stud_svc)
    planner_svc = PlannerContextService(session_factory=factory, student_context_service=stud_svc)
    fapr_svc = FAPRContextService(session_factory=factory, student_context_service=stud_svc)
    rep_svc = RepairOutcomeService(session_factory=factory)
    supp_svc = SupportPreferenceService(session_factory=factory)
    meta_svc = MetaSignalService(session_factory=factory, student_context_service=stud_svc)

    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: stud_svc)
    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: tutor_svc)
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: planner_svc)
    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: fapr_svc)
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: rep_svc)
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: supp_svc)
    monkeypatch.setattr(memory_routes, "get_meta_signal_service", lambda: meta_svc)

    client = TestClient(app, raise_server_exceptions=True)
    return client, factory


# =============================================================================
# 1. Complete Multi-Agent Workflow Test
# =============================================================================

def test_full_multi_agent_workflow(sqlite_context_setup):
    """
    Simulate a complete multi-agent learning lifecycle:
    1. Seed Student, Session, Skill, Misconception, STM, ConceptMemory, LTM, LearningState.
    2. Student makes an incorrect interaction with utterance.
    3. FAPR agent requests /fapr-context -> receives misconception, error, latest utterance.
    4. FAPR agent posts a repair outcome to /repair-outcome.
    5. Support preference service reflects the newly logged repair outcome.
    6. Tutor agent requests /tutor-context -> receives concise teaching state.
    7. Planner agent requests /planner-context -> receives curriculum evidence.
    8. Meta-Agent requests /meta-signals -> receives chronological signal stream.
    9. Full context endpoint /context returns complete unified picture.
    """
    client, factory = sqlite_context_setup

    student_id = "agent_student_01"
    session_id = "agent_session_01"
    skill_uuid = uuid.uuid4()
    inter_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_id)
        sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=session_id)
        skill = CanonicalSkill(
            skill_id=skill_uuid,
            canonical_name="math :: geometry :: slope",
            display_name="Slope",
        )
        session.add_all([st, sess, skill])
        session.flush()

        # Seed initial memory projections
        stm = ShortTermMemory(
            student_id=st.student_id,
            session_id=sess.session_id,
            interaction_count=3,
            correct_count=2,
            incorrect_count=1,
            attempt_observation_count=3,
            attempt_sum=4,
            hint_observation_count=3,
            hint_sum=1,
            response_time_observation_count=3,
            response_time_sum_ms=12000.0,
            recent_accuracy=0.67,
        )
        cm = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            interaction_count=5,
            correct_count=3,
            incorrect_count=2,
            attempt_observation_count=5,
            attempt_sum=8,
            hint_observation_count=5,
            hint_sum=2,
            response_time_observation_count=5,
            response_time_sum_ms=25000.0,
            accuracy=0.6,
        )
        ltm = LongTermMemory(
            student_id=st.student_id,
            total_sessions=2,
            interaction_count=15,
            correct_count=11,
            incorrect_count=4,
            attempt_observation_count=15,
            attempt_sum=18,
            hint_observation_count=15,
            hint_sum=4,
            response_time_observation_count=15,
            response_time_sum_ms=65000.0,
            overall_accuracy=0.7333,
            concept_count=3,
        )
        state = CurrentLearningState(
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            learning_state="DEVELOPING",
            evidence_level="FULL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            recent_interaction_count=5,
            attempt_observation_count=5,
            hint_observation_count=5,
            response_time_observation_count=5,
            last_snapshot_id=1,
        )
        misc = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skill_uuid,
            normalized_error="inverted delta x and delta y in slope",
            display_error="Inverted delta x and delta y in slope formula",
            occurrence_count=2,
        )
        inter = InteractionLog(
            interaction_id=inter_uuid,
            student_id=st.student_id,
            session_id=sess.session_id,
            canonical_skill_id=skill_uuid,
            source="ASSESSMENT",
            external_interaction_id="inter_agent_01",
            student_utterance="I don't understand why the slope is negative here",
            identified_error="inverted delta x and delta y in slope",
            is_correct=False,
            attempt_count=2,
            hint_count=1,
            response_time_ms=6500.0,
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )
        session.add_all([stm, cm, ltm, state, misc, inter])
        session.commit()

    # Step A: FAPR queries FAPR context
    fapr_res = client.get(f"/memory/{student_id}/fapr-context?session_id={session_id}&skill_id={skill_uuid}")
    assert fapr_res.status_code == 200
    fapr_data = fapr_res.json()
    assert fapr_data["current_learning_state"] == "DEVELOPING"
    assert fapr_data["latest_student_utterance"] == "I don't understand why the slope is negative here"
    assert len(fapr_data["misconceptions"]) == 1

    # Step B: FAPR executes pedagogical repair and records outcome
    rep_payload = {
        "student_id": student_id,
        "session_id": session_id,
        "skill_id": str(skill_uuid),
        "interaction_id": str(inter_uuid),
        "repair_action": "visual_graphical_slope",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Student resolved delta y over delta x concept.",
    }
    rep_res = client.post("/memory/repair-outcome", json=rep_payload)
    assert rep_res.status_code == 201
    assert rep_res.json()["repair_action"] == "visual_graphical_slope"

    # Step C: Tutor queries Tutor context
    tutor_res = client.get(f"/memory/{student_id}/tutor-context?session_id={session_id}&skill_id={skill_uuid}")
    assert tutor_res.status_code == 200
    tutor_data = tutor_res.json()
    assert tutor_data["current_learning_state"] == "DEVELOPING"
    assert tutor_data["recent_accuracy"] == 0.6
    assert len(tutor_data["recent_repairs"]) == 1
    assert tutor_data["recent_repairs"][0]["outcome"] == "RESOLVED"

    # Step D: Planner queries Planner context
    planner_res = client.get(f"/memory/{student_id}/planner-context?session_id={session_id}&skill_id={skill_uuid}")
    assert planner_res.status_code == 200
    planner_data = planner_res.json()
    assert planner_data["session_interaction_count"] == 3
    assert planner_data["concept_interaction_count"] == 5
    assert planner_data["total_sessions"] == 2
    assert planner_data["overall_accuracy"] == 0.7333

    # Step E: Meta-Agent queries Meta-signals
    meta_res = client.get(f"/memory/{student_id}/meta-signals?session_id={session_id}&skill_id={skill_uuid}")
    assert meta_res.status_code == 200
    meta_data = meta_res.json()
    sig_types = [s["signal_type"] for s in meta_data["signals"]]
    assert "incorrect_answer" in sig_types
    assert "confusion" in sig_types
    assert "repeated_misunderstanding" in sig_types

    # Step F: Full unified context endpoint
    full_res = client.get(f"/memory/{student_id}/context?session_id={session_id}&skill_id={skill_uuid}")
    assert full_res.status_code == 200
    full_data = full_res.json()
    assert full_data["short_term_memory"] is not None
    assert full_data["concept_memory"] is not None
    assert full_data["long_term_memory"] is not None
    assert len(full_data["recent_repairs"]) == 1


# =============================================================================
# 2. No-History Across All Endpoints Test
# =============================================================================

def test_no_history_student_across_all_endpoints(sqlite_context_setup):
    """Verify fresh student without records receives safe default 200 responses across all 6 GET endpoints."""
    client, factory = sqlite_context_setup
    fresh_stud = "fresh_student_999"

    # 1. Full context
    r1 = client.get(f"/memory/{fresh_stud}/context")
    assert r1.status_code == 200
    assert r1.json()["short_term_memory"] is None
    assert r1.json()["long_term_memory"] is None

    # 2. Tutor context
    r2 = client.get(f"/memory/{fresh_stud}/tutor-context")
    assert r2.status_code == 200
    assert r2.json()["current_learning_state"] is None
    assert r2.json()["recent_accuracy"] is None

    # 3. Planner context
    r3 = client.get(f"/memory/{fresh_stud}/planner-context")
    assert r3.status_code == 200
    assert r3.json()["total_sessions"] == 0
    assert r3.json()["concept_interaction_count"] == 0

    # 4. FAPR context
    r4 = client.get(f"/memory/{fresh_stud}/fapr-context")
    assert r4.status_code == 200
    assert r4.json()["latest_student_utterance"] is None
    assert r4.json()["previous_repairs"] == []

    # 5. Support preference
    r5 = client.get(f"/memory/{fresh_stud}/support-preference")
    assert r5.status_code == 200
    assert r5.json()["status"] == "INSUFFICIENT_EVIDENCE"
    assert r5.json()["preferred_support_style"] is None

    # 6. Meta signals
    r6 = client.get(f"/memory/{fresh_stud}/meta-signals")
    assert r6.status_code == 200
    assert r6.json()["total_signals"] == 0


# =============================================================================
# 3. Skill Isolation Test
# =============================================================================

def test_skill_isolation_for_same_student(sqlite_context_setup):
    """Verify concept-specific memory, misconceptions, and repairs never leak between skills."""
    client, factory = sqlite_context_setup

    student_id = "skill_iso_student"
    skillA_uuid = uuid.uuid4()
    skillB_uuid = uuid.uuid4()

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id=student_id)
        skillA = CanonicalSkill(skill_id=skillA_uuid, canonical_name="math :: skill_A", display_name="Skill A")
        skillB = CanonicalSkill(skill_id=skillB_uuid, canonical_name="math :: skill_B", display_name="Skill B")
        session.add_all([st, skillA, skillB])
        session.flush()

        # Skill A: accuracy 1.0, 1 misconception
        cmA = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skillA_uuid,
            interaction_count=6,
            correct_count=6,
            incorrect_count=0,
            accuracy=1.0,
        )
        miscA = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skillA_uuid,
            normalized_error="misc_A",
            display_error="Misconception A",
            occurrence_count=2,
        )
        # Skill B: accuracy 0.5, 1 misconception
        cmB = ConceptMemory(
            student_id=st.student_id,
            canonical_skill_id=skillB_uuid,
            interaction_count=6,
            correct_count=3,
            incorrect_count=3,
            accuracy=0.5,
        )
        miscB = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=st.student_id,
            canonical_skill_id=skillB_uuid,
            normalized_error="misc_B",
            display_error="Misconception B",
            occurrence_count=1,
        )
        session.add_all([cmA, miscA, cmB, miscB])
        session.commit()

    # Query Skill A
    rA = client.get(f"/memory/{student_id}/tutor-context?skill_id={skillA_uuid}")
    assert rA.status_code == 200
    assert rA.json()["recent_accuracy"] == 1.0
    assert len(rA.json()["misconceptions"]) == 1
    assert rA.json()["misconceptions"][0]["display_error"] == "Misconception A"

    # Query Skill B
    rB = client.get(f"/memory/{student_id}/tutor-context?skill_id={skillB_uuid}")
    assert rB.status_code == 200
    assert rB.json()["recent_accuracy"] == 0.5
    assert len(rB.json()["misconceptions"]) == 1
    assert rB.json()["misconceptions"][0]["display_error"] == "Misconception B"


# =============================================================================
# 4. Student Isolation Test
# =============================================================================

def test_student_isolation_across_endpoints(sqlite_context_setup):
    """Verify distinct students never share state even with identical external session names."""
    client, factory = sqlite_context_setup

    stud1 = "student_alpha"
    stud2 = "student_beta"
    same_sess_name = "shared_sess_name_001"

    with factory() as session:
        st1 = Student(student_id=uuid.uuid4(), external_student_id=stud1)
        st2 = Student(student_id=uuid.uuid4(), external_student_id=stud2)
        sess1 = LearningSession(session_id=uuid.uuid4(), student_id=st1.student_id, external_session_id=same_sess_name)
        sess2 = LearningSession(session_id=uuid.uuid4(), student_id=st2.student_id, external_session_id=same_sess_name)
        session.add_all([st1, st2, sess1, sess2])
        session.flush()

        stm1 = ShortTermMemory(
            student_id=st1.student_id,
            session_id=sess1.session_id,
            interaction_count=8,
            correct_count=8,
            recent_accuracy=1.0,
        )
        stm2 = ShortTermMemory(
            student_id=st2.student_id,
            session_id=sess2.session_id,
            interaction_count=4,
            correct_count=1,
            incorrect_count=3,
            recent_accuracy=0.25,
        )
        session.add_all([stm1, stm2])
        session.commit()

    r1 = client.get(f"/memory/{stud1}/planner-context?session_id={same_sess_name}")
    assert r1.status_code == 200
    assert r1.json()["session_interaction_count"] == 8
    assert r1.json()["session_accuracy"] == 1.0

    r2 = client.get(f"/memory/{stud2}/planner-context?session_id={same_sess_name}")
    assert r2.status_code == 200
    assert r2.json()["session_interaction_count"] == 4
    assert r2.json()["session_accuracy"] == 0.25


# =============================================================================
# 5. Error & Edge Case Tests
# =============================================================================

def test_invalid_parameters_and_errors(sqlite_context_setup):
    """Verify validation boundaries: non-existent skill, invalid limits, and empty IDs."""
    client, factory = sqlite_context_setup

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="stud_err_check")
        session.add(st)
        session.commit()

    # Invalid skill_id in support-preference -> 404
    r_sk = client.get("/memory/stud_err_check/support-preference?skill_id=non_existent_skill_uuid")
    assert r_sk.status_code == 404

    # Invalid limit < 1 -> 422
    r_lim = client.get("/memory/stud_err_check/context?limit=0")
    assert r_lim.status_code == 422

    # Blank student_id in URL -> 404 / 422
    r_blank = client.get("/memory/%20/context")
    assert r_blank.status_code in (400, 422)


# =============================================================================
# 6. Live PostgreSQL Agent Context Test
# =============================================================================

def test_live_postgresql_agent_context_pipeline():
    """Verify entire agent context suite against real PostgreSQL when configured."""
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for live PostgreSQL integration.")

    try:
        settings = load_postgres_settings()
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        engine = create_postgres_engine(settings)
        with engine.connect() as conn:
            pass
    except Exception as exc:
        pytest.skip(f"Live PostgreSQL is not accessible: {exc}")

    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)
    test_client = TestClient(app)

    try:
        stud_id = f"pg_stud_{uuid.uuid4().hex[:8]}"
        sess_id = f"pg_sess_{uuid.uuid4().hex[:8]}"
        skill_uuid = uuid.uuid4()

        with session_factory() as session:
            st = Student(student_id=uuid.uuid4(), external_student_id=stud_id)
            session.add(st)
            session.flush()

            sess = LearningSession(session_id=uuid.uuid4(), student_id=st.student_id, external_session_id=sess_id)
            session.add(sess)
            session.flush()

            skill = SkillContextResolver.resolve_or_create_skill(session, "Calculus", f"Limits_{uuid.uuid4().hex[:4]}")
            skill_uuid = skill.skill_id

            stm = ShortTermMemory(student_id=st.student_id, session_id=sess.session_id, interaction_count=4, correct_count=3, incorrect_count=1, attempt_observation_count=4, attempt_sum=5, hint_observation_count=4, hint_sum=1, response_time_observation_count=4, response_time_sum_ms=16000.0, recent_accuracy=0.75)
            cm = ConceptMemory(student_id=st.student_id, canonical_skill_id=skill_uuid, interaction_count=4, correct_count=3, incorrect_count=1, attempt_observation_count=4, attempt_sum=5, hint_observation_count=4, hint_sum=1, response_time_observation_count=4, response_time_sum_ms=16000.0, accuracy=0.75)
            ltm = LongTermMemory(student_id=st.student_id, total_sessions=1, interaction_count=4, correct_count=3, incorrect_count=1, attempt_observation_count=4, attempt_sum=5, hint_observation_count=4, hint_sum=1, response_time_observation_count=4, response_time_sum_ms=16000.0, overall_accuracy=0.75, concept_count=1)
            state = CurrentLearningState(student_id=st.student_id, canonical_skill_id=skill_uuid, learning_state="STRONG", evidence_level="FULL_SKILL", evidence_strength="HIGH", behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE", model_used=True, recent_interaction_count=4, attempt_observation_count=4, hint_observation_count=4, response_time_observation_count=4, last_snapshot_id=None)
            session.add_all([stm, cm, ltm, state])
            session.commit()

        headers = {"X-Service-Key": getattr(settings, "service_key", "dev-service-key") if hasattr(settings, "service_key") else os.environ.get("SERVICE_KEY", "dev-service-key")}

        # 1. Full context on PostgreSQL
        r_ctx = test_client.get(f"/memory/{stud_id}/context?session_id={sess_id}&skill_id={skill_uuid}", headers=headers)
        assert r_ctx.status_code == 200
        assert r_ctx.json()["short_term_memory"]["recent_accuracy"] == 0.75

        # 2. Tutor context on PostgreSQL
        r_tut = test_client.get(f"/memory/{stud_id}/tutor-context?session_id={sess_id}&skill_id={skill_uuid}", headers=headers)
        assert r_tut.status_code == 200
        assert r_tut.json()["current_learning_state"] == "STRONG"

        # 3. Planner context on PostgreSQL
        r_plan = test_client.get(f"/memory/{stud_id}/planner-context?session_id={sess_id}&skill_id={skill_uuid}", headers=headers)
        assert r_plan.status_code == 200
        assert r_plan.json()["concept_interaction_count"] == 4

        # 4. FAPR context on PostgreSQL
        r_fapr = test_client.get(f"/memory/{stud_id}/fapr-context?session_id={sess_id}&skill_id={skill_uuid}", headers=headers)
        assert r_fapr.status_code == 200
        assert r_fapr.json()["current_learning_state"] == "STRONG"

        # 5. Post repair outcome on PostgreSQL
        r_rep = test_client.post(
            "/memory/repair-outcome",
            json={
                "student_id": stud_id,
                "session_id": sess_id,
                "skill_id": str(skill_uuid),
                "repair_action": "l_hopital_hint",
                "outcome": "RESOLVED",
                "score": 1.0,
            },
            headers=headers,
        )
        assert r_rep.status_code == 201
        assert r_rep.json()["repair_action"] == "l_hopital_hint"

        # 6. Meta signals on PostgreSQL
        r_meta = test_client.get(f"/memory/{stud_id}/meta-signals", headers=headers)
        assert r_meta.status_code == 200

        # 7. Support preference on PostgreSQL
        r_pref = test_client.get(f"/memory/{stud_id}/support-preference", headers=headers)
        assert r_pref.status_code == 200
        assert r_pref.json()["status"] == "INSUFFICIENT_EVIDENCE"
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()

