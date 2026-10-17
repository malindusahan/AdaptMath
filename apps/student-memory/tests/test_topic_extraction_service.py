"""Tests for TopicExtractionService and PostgreSQL audit logging."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.supporting_memory import TopicExtractionLog
from src.services.topic_extraction_service import TopicExtractionService


@pytest.fixture
def mock_session_factory():
    """Create an in-memory SQLite database session factory for fast transactional testing."""
    engine = create_engine("sqlite:///:memory:")
    # Remove schema qualification for SQLite compatibility
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def topic_service(mock_session_factory):
    """Instantiate TopicExtractionService with real extractor and in-memory audit DB."""
    return TopicExtractionService(session_factory=mock_session_factory)


def _seed_student_and_session(session_factory, student_id: uuid.UUID, session_id: uuid.UUID):
    with session_factory() as session:
        st = Student(
            student_id=student_id,
            external_student_id=f"ext-student-{student_id}",
        )
        sess = LearningSession(
            session_id=session_id,
            student_id=student_id,
            external_session_id=f"ext-session-{session_id}",
        )
        session.add(st)
        session.add(sess)
        session.commit()


def test_keyword_prediction_logged(topic_service, mock_session_factory):
    """Verify that an exact alias keyword query is extracted and properly audited."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    _seed_student_and_session(mock_session_factory, student_id, session_id)

    query = "Can you explain PEMDAS and how it works?"
    result = topic_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text=query,
    )

    assert result.method == "minilm_finetuned"
    assert result.confidence >= 0.40
    assert result.needs_review is False
    assert result.skill_id is not None

    logs = topic_service.get_student_audit_logs(student_id)
    assert len(logs) == 1
    log = logs[0]
    assert log.student_id == student_id
    assert log.session_id == session_id
    assert log.input_text == query
    assert log.extraction_method == "minilm_finetuned"
    assert log.canonical_skill_id == uuid.UUID(result.skill_id)
    assert log.confidence == result.confidence
    assert log.needs_review is False
    assert log.model_version == result.model_version

    alternatives = json.loads(log.alternatives_json)
    assert len(alternatives) >= 1
    assert alternatives[0]["skill_code"] == result.skill_code


def test_minilm_semantic_prediction_logged(topic_service, mock_session_factory):
    """Verify that a paraphrased semantic query is classified via MiniLM and audited."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    _seed_student_and_session(mock_session_factory, student_id, session_id)

    query = "How do I work out the steepness of a line from two points on a graph?"
    result = topic_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text=query,
    )

    assert result.method == "minilm_finetuned"
    assert result.needs_review is False
    assert result.skill_id is not None

    logs = topic_service.get_student_audit_logs(student_id)
    assert len(logs) == 1
    log = logs[0]
    assert log.extraction_method == "minilm_finetuned"
    assert log.canonical_skill_id == uuid.UUID(result.skill_id)
    assert log.confidence == result.confidence
    assert log.needs_review is False


def test_geometry_comparison_prediction_logged(topic_service, mock_session_factory):
    """Verify that a geometry query is audited accurately."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    _seed_student_and_session(mock_session_factory, student_id, session_id)

    query = "What is the difference between congruent figures and similar figures?"
    result = topic_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text=query,
    )

    assert result.method == "minilm_finetuned"
    assert result.needs_review is False
    assert result.skill_id is not None

    logs = topic_service.get_student_audit_logs(student_id)
    assert len(logs) == 1
    log = logs[0]
    assert log.extraction_method == "minilm_finetuned"
    assert log.canonical_skill_id == uuid.UUID(result.skill_id)
    assert log.confidence == result.confidence


def test_abstention_logged_with_null_skill(topic_service, mock_session_factory):
    """Verify that a vague non-educational input logs an audit entry with NULL skill_id and needs_review=True."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    _seed_student_and_session(mock_session_factory, student_id, session_id)

    vague_query = "Can you help me?"
    result = topic_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text=vague_query,
    )

    assert result.method == "abstain"
    assert result.skill_id is None
    assert result.needs_review is True

    logs = topic_service.get_student_audit_logs(student_id)
    assert len(logs) == 1
    log = logs[0]
    assert log.extraction_method == "abstain"
    assert log.canonical_skill_id is None
    assert log.needs_review is True
    assert log.input_text == vague_query


def test_student_and_session_isolation(topic_service, mock_session_factory):
    """Verify multiple extractions across students maintain strict student_id and session_id isolation."""
    student_1, session_1 = uuid.uuid4(), uuid.uuid4()
    student_2, session_2 = uuid.uuid4(), uuid.uuid4()

    _seed_student_and_session(mock_session_factory, student_1, session_1)
    _seed_student_and_session(mock_session_factory, student_2, session_2)

    topic_service.extract_and_audit(
        student_id=student_1,
        session_id=session_1,
        text="Can you explain PEMDAS?",
    )
    topic_service.extract_and_audit(
        student_id=student_1,
        session_id=session_1,
        text="What is Pythagoras theorem?",
    )
    topic_service.extract_and_audit(
        student_id=student_2,
        session_id=session_2,
        text="How do I calculate percentage discount?",
    )

    logs_1 = topic_service.get_student_audit_logs(student_1)
    logs_2 = topic_service.get_student_audit_logs(student_2)

    assert len(logs_1) == 2
    assert len(logs_2) == 1

    assert all(l.student_id == student_1 for l in logs_1)
    assert all(l.student_id == student_2 for l in logs_2)


def test_rollback_on_db_failure(mock_session_factory):
    """Verify that an unexpected DB failure triggers rollback without corrupting state."""
    broken_factory = MagicMock(side_effect=RuntimeError("Database connection lost"))
    failing_service = TopicExtractionService(session_factory=broken_factory)

    with pytest.raises(RuntimeError, match="Database connection lost"):
        failing_service.extract_and_audit(
            student_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            text="Can you explain PEMDAS?",
        )
