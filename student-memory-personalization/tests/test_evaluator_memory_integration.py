"""Phase 18 — Step 2: Evaluator → Memory Integration Contract Validation Test.

Simulates the external Evaluator component posting completed, evaluated assessment data
to `POST /memory/update` with `X-Service-Key` authentication.
Verifies all downstream PostgreSQL state transitions across interaction_logs, STM, LTM,
Concept Memory, misconceptions, learning-state snapshots, and current learning state.
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
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.raw_interaction import InteractionLog
from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    StudentMisconception,
)
from src.database.postgres_config import DATABASE_SCHEMA_ENV, DATABASE_URL_ENV, load_postgres_settings
from src.database.postgres_session import create_postgres_engine, create_session_factory, set_session_factory
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_seed_service import OntologySeedService
from src.schemas.interaction import MemoryUpdateResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SERVICE_KEY = "evaluator_integration_secret_key_8899"


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
def evaluator_test_env():
    """Configure PostgreSQL environment, seed ontology, and provide TestClient."""
    old_env = dict(os.environ)
    _load_dotenv_into_environ()

    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for Evaluator live integration.")

    os.environ["MEMORY_AUTH_ENABLED"] = "true"
    os.environ["MEMORY_SERVICE_API_KEY"] = TEST_SERVICE_KEY

    engine = None
    try:
        from alembic import command
        from alembic.config import Config

        settings = load_postgres_settings()
        engine = create_postgres_engine(settings)
        factory = create_session_factory(engine)
        set_session_factory(factory)

        alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        command.upgrade(alembic_cfg, "head")

        OntologySeedService(session_factory=factory).seed_ontology()

        eval_student_id = f"eval_student_{uuid.uuid4().hex[:8]}"

        with factory() as session:
            student = Student(
                student_id=uuid.uuid4(),
                external_student_id=eval_student_id,
                display_name="Evaluator Test Student",
            )
            session.add(student)
            session.commit()

        client = TestClient(app, raise_server_exceptions=False)
        yield {
            "client": client,
            "engine": engine,
            "factory": factory,
            "schema": settings.schema,
            "student_id": eval_student_id,
            "service_key": TEST_SERVICE_KEY,
        }

    finally:
        if engine is not None:
            engine.dispose()

        os.environ.clear()
        os.environ.update(old_env)


# ---------------------------------------------------------------------------
# Evaluator Integration Tests
# ---------------------------------------------------------------------------

def test_evaluator_auth_rejection(evaluator_test_env):
    """Verify Evaluator request is rejected when X-Service-Key is missing or wrong."""
    client = evaluator_test_env["client"]
    student_id = evaluator_test_env["student_id"]
    payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
            }
        ],
    }

    # Missing header
    res_no_key = client.post("/memory/update", json=payload)
    assert res_no_key.status_code == 401
    assert res_no_key.json()["error_code"] == "UNAUTHORIZED"

    # Invalid header
    res_bad_key = client.post("/memory/update", json=payload, headers={"X-Service-Key": "wrong_key"})
    assert res_bad_key.status_code == 401
    assert res_bad_key.json()["error_code"] == "UNAUTHORIZED"


def test_evaluator_validation_failure(evaluator_test_env):
    """Verify invalid payload returns standardized 422 VALIDATION_ERROR."""
    client = evaluator_test_env["client"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}

    # Empty student_id
    res_empty_student = client.post(
        "/memory/update",
        json={"student_id": "", "topic": "Algebra", "assessment_questions": []},
        headers=headers,
    )
    assert res_empty_student.status_code == 422
    assert res_empty_student.json()["error_code"] == "VALIDATION_ERROR"


def test_evaluator_single_correct_question_update(evaluator_test_env):
    """Verify Evaluator posts one correct question with complete behavioral metrics."""
    client = evaluator_test_env["client"]
    factory = evaluator_test_env["factory"]
    student_id = evaluator_test_env["student_id"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}

    payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q_eval_01",
                "question": "Solve 3x = 15",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 3400.0,
            }
        ],
        "identified_errors": [],
        "overall_feedback": "Evaluator assessment: Perfect answer on linear equations.",
    }

    res = client.post("/memory/update", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    # Verify Response Contract
    parsed_response = MemoryUpdateResponse.model_validate(data)
    assert parsed_response.student_id == student_id
    assert parsed_response.topic == "Algebra"
    assert parsed_response.subtopic == "Linear equations"
    assert parsed_response.learning_state in ("DEVELOPING", "STRONG")
    assert parsed_response.behavioural_coverage == "FULL_BEHAVIOURAL_COVERAGE"
    assert parsed_response.recent_interaction_count >= 1
    assert parsed_response.memory_updated is True

    # Verify direct database state in PostgreSQL
    with factory() as session:
        # 1. raw interaction logs
        interactions = session.execute(select(InteractionLog)).scalars().all()
        assert len(interactions) >= 1
        latest_interaction = interactions[-1]
        assert latest_interaction.is_correct is True

        # 2. STM
        stms = session.execute(select(ShortTermMemory)).scalars().all()
        assert len(stms) >= 1

        # 3. LTM
        ltms = session.execute(select(LongTermMemory)).scalars().all()
        assert len(ltms) >= 1

        # 4. Concept Memory
        cms = session.execute(select(ConceptMemory)).scalars().all()
        assert len(cms) >= 1

        # 5. Snapshots & Current State
        snapshots = session.execute(select(LearningStateSnapshot)).scalars().all()
        assert len(snapshots) >= 1

        curr_states = session.execute(select(CurrentLearningState)).scalars().all()
        assert len(curr_states) >= 1


def test_evaluator_incorrect_with_misconception(evaluator_test_env):
    """Verify Evaluator posts incorrect question with identified error and logs misconception."""
    client = evaluator_test_env["client"]
    factory = evaluator_test_env["factory"]
    student_id = evaluator_test_env["student_id"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}

    payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q_eval_02",
                "question": "Solve 4x = 20",
                "student_answer": "16",
                "expected_answer": "5",
                "is_correct": False,
                "identified_error": "Subtracted instead of dividing",
                "attempt_count": 2,
                "hint_count": 1,
                "response_time_ms": 7100.0,
            }
        ],
        "identified_errors": ["Subtracted instead of dividing"],
        "overall_feedback": "Evaluator: Error identified in inverse operations.",
    }

    res = client.post("/memory/update", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["student_id"] == student_id
    assert data["misconception_count"] >= 1
    assert data["memory_updated"] is True

    # Verify Misconception table in PostgreSQL
    with factory() as session:
        misconceptions = session.execute(select(StudentMisconception)).scalars().all()
        assert len(misconceptions) >= 1
        error_names = [m.display_error for m in misconceptions]
        assert "Subtracted instead of dividing" in error_names


def test_evaluator_multi_question_batch(evaluator_test_env):
    """Verify Evaluator posts multi-question assessment in a single atomic transaction."""
    client = evaluator_test_env["client"]
    student_id = evaluator_test_env["student_id"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}

    payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q_batch_1",
                "question": "Solve 2x = 8",
                "student_answer": "4",
                "expected_answer": "4",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 2800.0,
            },
            {
                "question_id": "q_batch_2",
                "question": "Solve 5x = 25",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 3100.0,
            },
        ],
        "identified_errors": [],
    }

    res = client.post("/memory/update", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["recent_interaction_count"] >= 2
    assert data["learning_state"] in ("DEVELOPING", "STRONG")


def test_evaluator_optional_behavioral_omitted(evaluator_test_env):
    """Verify Evaluator can omit behavioral metrics and backend does not inject fake zeros."""
    client = evaluator_test_env["client"]
    factory = evaluator_test_env["factory"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}
    corr_student_id = f"eval_corr_{uuid.uuid4().hex[:8]}"

    with factory() as session:
        session.add(
            Student(
                student_id=uuid.uuid4(),
                external_student_id=corr_student_id,
                display_name="Correctness Only Student",
            )
        )
        session.commit()

    payload = {
        "student_id": corr_student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q_no_behavior",
                "question": "Solve 6x = 36",
                "student_answer": "6",
                "expected_answer": "6",
                "is_correct": True,
                # attempt_count, hint_count, response_time_ms omitted
            }
        ],
    }

    res = client.post("/memory/update", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["behavioural_coverage"] == "CORRECTNESS_ONLY_COVERAGE"
    assert data["attempt_observation_count"] == 0
    assert data["hint_observation_count"] == 0
    assert data["response_time_observation_count"] == 0


def test_evaluator_retry_idempotent_behavior(evaluator_test_env):
    """Verify Evaluator retrying assessment submission succeeds idempotently."""
    client = evaluator_test_env["client"]
    student_id = evaluator_test_env["student_id"]
    headers = {"X-Service-Key": evaluator_test_env["service_key"]}

    payload = {
        "student_id": student_id,
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q_retry_1",
                "question": "Solve 7x = 49",
                "student_answer": "7",
                "expected_answer": "7",
                "is_correct": True,
            }
        ],
    }

    res1 = client.post("/memory/update", json=payload, headers=headers)
    assert res1.status_code == 200

    res2 = client.post("/memory/update", json=payload, headers=headers)
    assert res2.status_code == 200
    assert res2.json()["memory_updated"] is True
