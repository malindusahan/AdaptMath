"""Final Backend Integration Acceptance Test Suite (Phase 16 - Step 8).

Validates all 13 API endpoints, security, CORS, limits, error envelopes,
topic extraction ML model, learning state classifier ML model,
multi-agent contexts, repair loops, meta-signals, and live PostgreSQL persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from src.api.app import app
from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, SkillAlias, Student
from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    RepairOutcome,
    StudentMisconception,
    TopicExtractionLog,
)
from src.database.postgres_config import DATABASE_URL_ENV, load_postgres_settings
from src.database.postgres_session import create_postgres_engine, create_session_factory, set_session_factory
from src.ontology.ontology_seed_service import OntologySeedService
from src.services.fapr_context_service import FAPRContextService
from src.services.meta_signal_service import MetaSignalService
from src.services.planner_context_service import PlannerContextService
from src.services.question_context_service import QuestionContextService
from src.services.readiness_service import ReadinessService
from src.services.repair_outcome_service import RepairOutcomeService
from src.services.skill_context_resolver import SkillContextResolver
from src.services.student_context_service import StudentContextService
from src.services.student_memory_query_service import StudentMemoryQueryService
from src.services.support_preference_service import SupportPreferenceService
from src.services.tutor_context_service import TutorContextService
import src.api.memory_routes as memory_routes

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_into_environ() -> None:
    """Load key-value pairs from .env if present and not already in os.environ."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val


@pytest.fixture(scope="module")
def final_acceptance_client():
    """Configure live PostgreSQL environment, run migrations, seed ontology, and wire services."""
    old_env = dict(os.environ)
    _load_dotenv_into_environ()

    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for final acceptance testing.")

    engine = None
    try:
        from alembic import command
        from alembic.config import Config

        settings = load_postgres_settings()

        for model_cls in (
            Student, LearningSession, CanonicalSkill, SkillAlias,
            InteractionLog, ShortTermMemory, LongTermMemory, ConceptMemory,
            CurrentLearningState, LearningStateSnapshot, TopicExtractionLog,
            StudentMisconception, RepairOutcome
        ):
            model_cls.__table__.schema = settings.schema

        for table in Base.metadata.tables.values():
            table.schema = settings.schema

        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        engine = create_postgres_engine(settings)

        # 1. Upgrade schema to head
        command.upgrade(config, "head")
        session_factory = create_session_factory(engine)
        set_session_factory(session_factory)

        # 2. Seed canonical ontology
        OntologySeedService(session_factory=session_factory).seed_ontology()

        # 3. Wire services to PostgreSQL session factory
        memory_routes.get_student_context_service = lambda: StudentContextService(session_factory=session_factory)
        memory_routes.get_tutor_context_service = lambda: TutorContextService(session_factory=session_factory)
        memory_routes.get_planner_context_service = lambda: PlannerContextService(session_factory=session_factory)
        memory_routes.get_fapr_context_service = lambda: FAPRContextService(session_factory=session_factory)
        memory_routes.get_repair_outcome_service = lambda: RepairOutcomeService(session_factory=session_factory)
        memory_routes.get_support_preference_service = lambda: SupportPreferenceService(session_factory=session_factory)
        memory_routes.get_meta_signal_service = lambda: MetaSignalService(session_factory=session_factory)
        memory_routes.get_student_memory_query_service = lambda: StudentMemoryQueryService(session_factory=session_factory)
        memory_routes.get_question_context_service = lambda: QuestionContextService(session_factory=session_factory)

        test_service_key = os.environ.get("MEMORY_SERVICE_API_KEY", "pg-acceptance-secret-888")
        os.environ["MEMORY_SERVICE_API_KEY"] = test_service_key
        os.environ["MEMORY_AUTH_ENABLED"] = "true"

        client = TestClient(app, raise_server_exceptions=False)

        yield client, session_factory, test_service_key
    finally:
        set_session_factory(None)
        if engine is not None:
            engine.dispose()
        os.environ.clear()
        os.environ.update(old_env)


def test_final_acceptance_system_probes_and_openapi(final_acceptance_client):
    """Verify public health & readiness probes, dependency status, and OpenAPI schema."""
    client, _, _ = final_acceptance_client

    # 1. Liveness check (public)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "ok"
    assert h_data["service"] == "student-personalization-memory"

    # 2. Readiness check (public)
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    r_data = r_ready.json()
    assert r_data["status"] == "ready"
    assert r_data["database"] == "ready"
    assert r_data["migrations"] == "ready"
    assert r_data["topic_extractor"] == "ready"
    assert r_data["learning_state_model"] == "ready"

    # 3. OpenAPI contract
    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    openapi_doc = r_openapi.json()
    assert "paths" in openapi_doc
    assert "/memory/update" in openapi_doc["paths"]
    assert "/memory/question-context" in openapi_doc["paths"]
    assert "/ready" in openapi_doc["paths"]
    assert "/health" in openapi_doc["paths"]
    assert "components" in openapi_doc
    assert "APIKeyHeader" in openapi_doc["components"]["securitySchemes"]


def test_final_acceptance_auth_cors_and_limits(final_acceptance_client):
    """Verify security, authentication, CORS preflight, pagination, limits, and request-size protection."""
    client, _, service_key = final_acceptance_client
    headers = {"X-Service-Key": service_key, "X-Request-ID": "final-accept-req-001"}

    # 1. Auth failure: Missing key -> 401 with standard ErrorResponse envelope
    r_no_key = client.get("/memory/any_stud/context")
    assert r_no_key.status_code == 401
    err_data = r_no_key.json()
    assert err_data["error_code"] == "UNAUTHORIZED"
    assert "request_id" in err_data
    assert "timestamp" in err_data

    # 2. Auth failure: Wrong key -> 401
    r_wrong_key = client.get("/memory/any_stud/context", headers={"X-Service-Key": "wrong-secret-key"})
    assert r_wrong_key.status_code == 401
    assert r_wrong_key.json()["error_code"] == "UNAUTHORIZED"

    # 3. Payload size limit (oversized body -> 413)
    large_payload = {
        "student_id": "stud_large",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": f"q_oversized_{i}",
                "question": "x" * 5000,
                "student_answer": "y" * 5000,
                "expected_answer": "y" * 5000,
                "is_correct": True,
            }
            for i in range(120)  # ~1.2 MB > 1.0 MB limit
        ],
    }
    r_oversized = client.post("/memory/update", json=large_payload, headers=headers)
    assert r_oversized.status_code == 413
    assert r_oversized.json()["error_code"] == "PAYLOAD_TOO_LARGE"

    # 4. Pagination limits bounds check: limit=0 or limit=101 -> 422
    r_invalid_limit = client.get("/memory/stud_any/history?limit=101", headers=headers)
    assert r_invalid_limit.status_code == 422
    assert r_invalid_limit.json()["error_code"] == "VALIDATION_ERROR"

    r_negative_offset = client.get("/memory/stud_any/history?offset=-1", headers=headers)
    assert r_negative_offset.status_code == 422

    # 5. CORS headers verification
    r_cors = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r_cors.status_code == 200
    assert r_cors.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_final_acceptance_complete_multi_agent_workflow(final_acceptance_client):
    """Execute complete 13-operation student lifecycle against live PostgreSQL database."""
    client, session_factory, service_key = final_acceptance_client
    headers = {"X-Service-Key": service_key, "X-Request-ID": "final-accept-e2e-001"}

    uid = uuid.uuid4().hex[:8]
    student_id = f"final_accept_stud_{uid}"
    session_id = f"final_accept_sess_{uid}"

    # Op 1: POST /memory/question-context (Topic Extractor fine-tuned inference)
    q_payload = {
        "student_id": student_id,
        "session_id": session_id,
        "question": "How do I solve the linear equation 3x + 9 = 24?",
    }
    r_qctx = client.post("/memory/question-context", json=q_payload, headers=headers)
    assert r_qctx.status_code == 200
    qctx_data = r_qctx.json()
    assert qctx_data["student_id"] == student_id
    assert qctx_data["topic"]["canonical_skill_name"] is not None
    assert qctx_data["topic"]["confidence"] > 0.5
    assert "request-id" in r_qctx.headers or "x-request-id" in r_qctx.headers
    assert "x-response-time-ms" in r_qctx.headers

    # Op 2: POST /memory/update (Raw Interactions, STM, LTM, Concept Memory, ML Learning State Classifier)
    update_payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": f"q_linear_{uid}_1",
                "question": "Solve for x: 3x + 9 = 24",
                "student_answer": "x = 5",
                "expected_answer": "x = 5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "hint_total": 2,
                "response_time_ms": 4100.0,
            },
            {
                "question_id": f"q_linear_{uid}_2",
                "question": "Solve for x: 5x - 10 = 15",
                "student_answer": "x = 5",
                "expected_answer": "x = 5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "hint_total": 2,
                "response_time_ms": 3900.0,
            },
            {
                "question_id": f"q_linear_{uid}_3",
                "question": "Solve for x: 2x + 6 = 16",
                "student_answer": "x = 4",
                "expected_answer": "x = 5",
                "is_correct": False,
                "identified_error": "sign_flip_error",
                "attempt_count": 2,
                "hint_count": 1,
                "hint_total": 2,
                "response_time_ms": 7500.0,
            },
        ],
        "identified_errors": ["sign_flip_error"],
        "overall_feedback": "Strong baseline performance with one minor sign computation error.",
    }
    r_update = client.post("/memory/update", json=update_payload, headers=headers)
    assert r_update.status_code == 200
    update_data = r_update.json()
    assert update_data["student_id"] == student_id
    assert update_data["learning_state"] in ("DEVELOPING", "STRONG", "NEEDS_SUPPORT")
    assert update_data["model_used"] is True
    assert update_data["recent_interaction_count"] == 3

    # Op 3: GET /memory/{student_id}/current
    r_curr = client.get(f"/memory/{student_id}/current?topic=Algebra&subtopic=Linear equations", headers=headers)
    assert r_curr.status_code == 200
    curr_data = r_curr.json()
    assert curr_data["student_id"] == student_id
    assert curr_data["recent_interaction_count"] == 3

    # Op 4: GET /memory/{student_id}/history
    r_hist = client.get(f"/memory/{student_id}/history?topic=Algebra&subtopic=Linear equations&limit=10&offset=0", headers=headers)
    assert r_hist.status_code == 200
    hist_data = r_hist.json()
    assert len(hist_data) >= 1
    assert hist_data[0]["student_id"] == student_id

    # Resolve skill UUID used during memory update
    with session_factory() as session:
        skill_row = SkillContextResolver.resolve_skill(session, "Algebra", "Linear equations")
        skill_id_str = str(skill_row.skill_id) if skill_row else qctx_data["topic"]["skill_id"]

    # Op 5: GET /memory/{student_id}/context (Full Unified Agent Context)
    r_full = client.get(f"/memory/{student_id}/context?session_id={session_id}&skill_id={skill_id_str}", headers=headers)
    assert r_full.status_code == 200
    full_data = r_full.json()
    assert full_data["student_id"] == student_id
    assert full_data["long_term_memory"] is not None
    assert full_data["concept_memory"] is not None
    assert full_data["learning_state"] is not None

    # Op 6: GET /memory/{student_id}/tutor-context
    r_tutor = client.get(f"/memory/{student_id}/tutor-context?session_id={session_id}&skill_id={skill_id_str}", headers=headers)
    assert r_tutor.status_code == 200
    tutor_data = r_tutor.json()
    assert tutor_data["student_id"] == student_id
    assert tutor_data["recent_accuracy"] is not None
    assert tutor_data["current_learning_state"] is not None

    # Op 7: GET /memory/{student_id}/planner-context
    r_planner = client.get(f"/memory/{student_id}/planner-context?session_id={session_id}&skill_id={skill_id_str}", headers=headers)
    assert r_planner.status_code == 200
    planner_data = r_planner.json()
    assert planner_data["student_id"] == student_id
    assert planner_data["current_learning_state"] is not None
    assert planner_data["concept_interaction_count"] >= 3

    # Op 8: GET /memory/{student_id}/fapr-context
    r_fapr = client.get(f"/memory/{student_id}/fapr-context?session_id={session_id}&skill_id={skill_id_str}", headers=headers)
    assert r_fapr.status_code == 200
    fapr_data = r_fapr.json()
    assert fapr_data["student_id"] == student_id
    assert fapr_data["recent_incorrect_count"] >= 1
    assert any(m["normalized_error"] == "sign_flip_error" for m in fapr_data["misconceptions"])

    # Op 9: POST /memory/repair-outcome (Record 3 distinct repair episodes)
    with session_factory() as session:
        st_obj = session.scalar(select(Student).where(Student.external_student_id == student_id))
        interaction_rows = session.scalars(
            select(InteractionLog)
            .where(InteractionLog.student_id == st_obj.student_id)
            .order_by(InteractionLog.created_at)
        ).all()
        interactions_info = []
        for r in interaction_rows:
            sess_obj = session.get(LearningSession, r.session_id)
            sess_id_val = sess_obj.external_session_id if sess_obj else str(r.session_id)
            interactions_info.append((str(r.interaction_id), sess_id_val))

    for i in range(min(3, len(interactions_info))):
        rep_payload = {
            "student_id": student_id,
            "session_id": interactions_info[i][1],
            "skill_id": skill_id_str,
            "interaction_id": interactions_info[i][0],
            "repair_action": "inverse_operation_prompt",
            "outcome": "RESOLVED",
            "score": 1.0,
            "notes": f"Pedagogical repair attempt {i + 1} resolved error.",
        }
        r_rep = client.post("/memory/repair-outcome", json=rep_payload, headers=headers)
        assert r_rep.status_code == 201

    # Op 10: GET /memory/{student_id}/support-preference
    r_pref = client.get(f"/memory/{student_id}/support-preference", headers=headers)
    assert r_pref.status_code == 200
    pref_data = r_pref.json()
    assert pref_data["student_id"] == student_id
    assert pref_data["status"] == "SUPPORTED_BY_HISTORY"
    assert pref_data["preferred_support_style"] == "inverse_operation_prompt"
    assert pref_data["evidence_count"] >= 3
    assert pref_data["success_rate"] == 1.0

    # Op 11: GET /memory/{student_id}/meta-signals
    r_meta = client.get(f"/memory/{student_id}/meta-signals", headers=headers)
    assert r_meta.status_code == 200
    meta_data = r_meta.json()
    assert meta_data["student_id"] == student_id
    assert meta_data["total_signals"] >= 3
    signal_types = {s["signal_type"] for s in meta_data["signals"]}
    assert "correct_answer" in signal_types
    assert "incorrect_answer" in signal_types

    # Direct PostgreSQL relational database inspection across all 9 schema tables
    with session_factory() as session:
        st_row = session.scalar(select(Student).where(Student.external_student_id == student_id))
        assert st_row is not None, "Student record missing in PostgreSQL students table."
        db_student_id = st_row.student_id

        # 1. interaction_logs
        interactions = session.scalars(select(InteractionLog).where(InteractionLog.student_id == db_student_id)).all()
        assert len(interactions) >= 3

        # 2. short_term_memory
        stm = session.scalar(select(ShortTermMemory).where(ShortTermMemory.student_id == db_student_id))
        assert stm is not None

        # 3. long_term_memory
        ltm = session.scalar(select(LongTermMemory).where(LongTermMemory.student_id == db_student_id))
        assert ltm is not None

        # 4. concept_memory
        cm = session.scalar(select(ConceptMemory).where(ConceptMemory.student_id == db_student_id))
        assert cm is not None

        # 5. learning_state_snapshots
        snapshots = session.scalars(select(LearningStateSnapshot).where(LearningStateSnapshot.student_id == db_student_id)).all()
        assert len(snapshots) >= 1

        # 6. current_learning_state
        cls_row = session.scalar(select(CurrentLearningState).where(CurrentLearningState.student_id == db_student_id))
        assert cls_row is not None

        # 7. topic_extraction_logs
        extraction_logs = session.scalars(select(TopicExtractionLog).where(TopicExtractionLog.student_id == db_student_id)).all()
        assert len(extraction_logs) >= 1

        # 8. repair_outcomes
        repairs = session.scalars(select(RepairOutcome).where(RepairOutcome.student_id == db_student_id)).all()
        assert len(repairs) >= 3

        # 9. student_misconceptions
        misconceptions = session.scalars(select(StudentMisconception).where(StudentMisconception.student_id == db_student_id)).all()
        assert len(misconceptions) >= 1

    # Clean up test student's records in FK dependency order
    with session_factory() as session:
        st_to_delete = session.scalar(select(Student).where(Student.external_student_id == student_id))
        if st_to_delete:
            db_id = st_to_delete.student_id
            schema_name = load_postgres_settings().schema
            session.execute(text(f"DELETE FROM {schema_name}.repair_outcomes WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.topic_extraction_logs WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.current_learning_state WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.student_misconceptions WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.learning_state_snapshots WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.concept_memory WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.long_term_memory WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.short_term_memory WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.interaction_logs WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.learning_sessions WHERE student_id = :sid"), {"sid": db_id})
            session.execute(text(f"DELETE FROM {schema_name}.students WHERE student_id = :sid"), {"sid": db_id})
            session.commit()
