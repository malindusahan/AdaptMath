"""Regression tests for the evidence-only AdaptMath integration boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.models.completed_attempt import CompletedAttemptReceipt
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.memory_projection import LongTermMemory
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import LearningStateSnapshot
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.schemas.adaptmath import AdaptMathCompletedAttemptRequest
from src.schemas.student_context import StudentContextResponse
from src.services.adaptmath_ingestion_service import (
    AdaptMathAttemptConflictError,
    AdaptMathIngestionService,
    AdaptMathSkillNotFoundError,
)
from src.services.student_context_service import StudentContextService
from src.services.skill_context_resolver import SkillContextResolver
from src.services.tutor_context_service import TutorContextService
from src.database.postgres_config import DATABASE_URL_ENV, load_postgres_settings
from src.database.postgres_session import create_postgres_engine, create_session_factory


SCHEMA = "student_memory"


@pytest.fixture
def integration_database():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        execution_options={"schema_translate_map": {SCHEMA: None}},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    skill_id = uuid.uuid4()
    with factory.begin() as session:
        session.add(
            CanonicalSkill(
                skill_id=skill_id,
                canonical_name="linear equations",
                display_name="Linear Equations",
                ontology_version="1.0",
                is_active=True,
            )
        )
    try:
        yield engine, factory, skill_id
    finally:
        engine.dispose()


def _request(
    attempt_id: str = "thread-abc:1",
    *,
    session_status: str = "ACTIVE",
    second_correct: bool = False,
) -> AdaptMathCompletedAttemptRequest:
    return AdaptMathCompletedAttemptRequest.model_validate(
        {
            "external_student_id": "student-a",
            "external_session_id": "thread-abc",
            "external_attempt_id": attempt_id,
            "source": "adaptmath",
            "target_skill": "Linear Equations",
            "session_status": session_status,
            "questions": [
                {
                    "external_interaction_id": f"{attempt_id}:q1",
                    "question_id": "q1",
                    "problem_id": "thread-abc",
                    "student_answer": "4",
                    "expected_answer": "4",
                    "is_correct": True,
                },
                {
                    "external_interaction_id": f"{attempt_id}:q2",
                    "question_id": "q2",
                    "problem_id": "thread-abc",
                    "student_answer": "5",
                    "expected_answer": "6",
                    "is_correct": second_correct,
                    "identified_error": (
                        None if second_correct else "Added instead of dividing"
                    ),
                },
            ],
        }
    )


def test_contract_forbids_mastery_and_learning_state_fields():
    payload = _request().model_dump(mode="json")
    payload["learning_state"] = "STRONG"
    payload["mastery_after"] = 0.9
    with pytest.raises(ValidationError):
        AdaptMathCompletedAttemptRequest.model_validate(payload)


def test_identical_retry_is_exactly_once(integration_database):
    _, factory, _ = integration_database
    service = AdaptMathIngestionService(session_factory=factory)

    first = service.ingest_completed_attempt(_request())
    replay = service.ingest_completed_attempt(_request())

    assert first.status == "STORED"
    assert replay.status == "ALREADY_STORED"
    assert first.receipt_id == replay.receipt_id
    with factory() as session:
        assert session.scalar(select(func.count(CompletedAttemptReceipt.receipt_id))) == 1
        assert session.scalar(select(func.count(InteractionLog.interaction_id))) == 2
        interactions = list(session.scalars(select(InteractionLog)))
        assert [item.is_correct for item in interactions] == [True, False]
        assert session.scalar(select(func.count(LearningStateSnapshot.snapshot_id))) == 0


def test_same_attempt_key_with_different_evidence_conflicts(integration_database):
    _, factory, _ = integration_database
    service = AdaptMathIngestionService(session_factory=factory)
    service.ingest_completed_attempt(_request())

    with pytest.raises(AdaptMathAttemptConflictError):
        service.ingest_completed_attempt(_request(second_correct=True))


def test_reteaching_attempts_share_one_session(integration_database):
    _, factory, _ = integration_database
    service = AdaptMathIngestionService(session_factory=factory)
    service.ingest_completed_attempt(_request("thread-abc:1"))
    service.ingest_completed_attempt(
        _request(
            "thread-abc:2",
            session_status="COMPLETED",
            second_correct=True,
        )
    )

    with factory() as session:
        sessions = list(session.scalars(select(LearningSession)))
        assert len(sessions) == 1
        assert sessions[0].external_session_id == "thread-abc"
        assert sessions[0].status == "COMPLETED"
        assert session.scalar(select(func.count(CompletedAttemptReceipt.receipt_id))) == 2
        ltm = session.scalar(select(LongTermMemory))
        assert ltm is not None
        assert ltm.total_sessions == 1
        assert ltm.interaction_count == 4


def test_unknown_skill_rolls_back_without_creating_identity(integration_database):
    _, factory, _ = integration_database
    service = AdaptMathIngestionService(session_factory=factory)
    payload = _request().model_copy(update={"target_skill": "Unknown BKT Skill"})
    with pytest.raises(AdaptMathSkillNotFoundError):
        service.ingest_completed_attempt(payload)
    with factory() as session:
        assert session.scalar(select(func.count(Student.student_id))) == 0


def test_student_context_retrieval_does_not_create_unknown_session(
    integration_database,
):
    _, factory, _ = integration_database
    with factory.begin() as session:
        session.add(
            Student(
                student_id=uuid.uuid4(),
                external_student_id="read-only-student",
            )
        )
    service = StudentContextService(session_factory=factory)
    context = service.get_student_context(
        "read-only-student",
        session_id="new-thread-must-not-be-created",
        skill_id="Linear Equations",
    )
    assert context.session_id == "new-thread-must-not-be-created"
    with factory() as session:
        assert session.scalar(select(func.count(LearningSession.session_id))) == 0


def test_zero_accuracy_is_not_replaced_by_fallback_value():
    context = StudentContextResponse(
        student_id="student-zero",
        concept_memory={
            "accuracy": 0.0,
            "recent_accuracy": 0.5,
            "overall_accuracy": 0.75,
        },
    )
    fake_context_service = SimpleNamespace(
        get_student_context=lambda **_: context
    )
    service = TutorContextService(
        session_factory=lambda: None,
        student_context_service=fake_context_service,
    )
    response = service.get_tutor_context(
        student_id="student-zero",
        skill_id="Linear Equations",
    )
    assert response.recent_accuracy == 0.0


def test_all_bkt_skills_resolve_without_changing_vocabulary():
    project_root = Path(__file__).resolve().parents[2]
    bkt_path = Path(
        os.environ.get(
            "BKT_SKILLS_PATH",
            project_root / "student-modeling" / "models" / "bkt_params.json",
        )
    )
    bkt_skills = list(json.loads(bkt_path.read_text(encoding="utf-8")))
    lookup = OntologyLookupService()
    unresolved = [
        skill for skill in bkt_skills if lookup.get_by_alias(skill) is None
    ]
    assert len(bkt_skills) == 95
    assert unresolved == []


def test_live_postgres_concurrent_retry_is_one_logical_attempt():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured.")

    engine = create_postgres_engine(load_postgres_settings())
    factory = create_session_factory(engine)
    token = uuid.uuid4().hex
    attempt_id = f"concurrent-thread-{token}:1"
    request = AdaptMathCompletedAttemptRequest.model_validate(
        {
            "external_student_id": f"concurrent-student-{token}",
            "external_session_id": f"concurrent-thread-{token}",
            "external_attempt_id": attempt_id,
            "source": "adaptmath",
            "target_skill": "Linear Equations",
            "session_status": "COMPLETED",
            "questions": [
                {
                    "external_interaction_id": f"{attempt_id}:q1",
                    "question_id": "q1",
                    "problem_id": f"concurrent-thread-{token}",
                    "student_answer": "4",
                    "expected_answer": "4",
                    "is_correct": True,
                }
            ],
        }
    )

    def ingest():
        return AdaptMathIngestionService(
            session_factory=factory
        ).ingest_completed_attempt(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(), range(2)))

    assert {result.status for result in results} == {
        "STORED",
        "ALREADY_STORED",
    }
    with factory() as session:
        receipt_count = session.scalar(
            select(func.count(CompletedAttemptReceipt.receipt_id)).where(
                CompletedAttemptReceipt.source == "adaptmath",
                CompletedAttemptReceipt.external_attempt_id == attempt_id,
            )
        )
        interaction_count = session.scalar(
            select(func.count(InteractionLog.interaction_id)).where(
                InteractionLog.source == "adaptmath",
                InteractionLog.external_interaction_id == f"{attempt_id}:q1",
            )
        )
    engine.dispose()
    assert receipt_count == 1
    assert interaction_count == 1


def test_all_bkt_skills_resolve_in_seeded_postgres_ontology():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured.")

    project_root = Path(__file__).resolve().parents[2]
    bkt_path = Path(
        os.environ.get(
            "BKT_SKILLS_PATH",
            project_root / "student-modeling" / "models" / "bkt_params.json",
        )
    )
    bkt_skills = list(json.loads(bkt_path.read_text(encoding="utf-8")))
    engine = create_postgres_engine(load_postgres_settings())
    factory = create_session_factory(engine)
    try:
        with factory() as session:
            unresolved = [
                skill
                for skill in bkt_skills
                if SkillContextResolver.resolve_skill(
                    session,
                    topic=skill,
                    subtopic=None,
                )
                is None
            ]
    finally:
        engine.dispose()

    assert len(bkt_skills) == 95
    assert unresolved == []
