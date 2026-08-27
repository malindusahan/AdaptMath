"""Runtime latency, singleton caching, isolation, and system validation tests for Phase 14."""

from __future__ import annotations

import time
import uuid
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.supporting_memory import TopicExtractionLog
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.services.question_context_service import QuestionContextService
from src.topic_extraction.topic_extractor import HybridTopicExtractor, get_topic_extractor


@pytest.fixture
def runtime_test_db():
    """Shared in-memory database setup for runtime testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory


def test_singleton_reuse():
    """Verify that get_topic_extractor() reuses the same instance without reloading weights."""
    ext1 = get_topic_extractor()
    ext2 = get_topic_extractor()
    assert ext1 is ext2


def test_deterministic_identical_inference():
    """Verify repeated identical queries return identical scores, candidates, and paths."""
    extractor = get_topic_extractor()
    queries = [
        "Can you explain PEMDAS?",
        "How do I find the steepness of a linear equation?",
        "What is the Pythagorean theorem?",
        "Can you help me?",
    ]
    for q in queries:
        r1 = extractor.extract(q)
        r2 = extractor.extract(q)
        assert r1.skill_id == r2.skill_id
        assert r1.skill_code == r2.skill_code
        assert r1.method == r2.method
        assert r1.confidence == pytest.approx(r2.confidence, abs=1e-5)
        assert len(r1.top_candidates) == len(r2.top_candidates)


def test_ontology_ids_validity():
    """Verify all non-abstain extracted skills resolve to valid ontology entries."""
    extractor = get_topic_extractor()
    sample_queries = [
        "What is slope?",
        "Explain median and mode",
        "How do I calculate volume of a cylinder?",
        "What is the quadratic formula?",
        "Graph linear inequalities on coordinate plane",
    ]
    for q in sample_queries:
        res = extractor.extract(q)
        if not res.is_abstain:
            skill = extractor.ontology_lookup.get_by_skill_code(res.skill_code)
            assert skill is not None
            assert skill.canonical_name == res.canonical_skill_name
            assert str(skill.skill_id) == str(res.skill_id)


def test_audit_logging_persistence(runtime_test_db):
    """Verify topic extraction events are persisted to topic_extraction_logs in DB."""
    factory = runtime_test_db
    service = QuestionContextService(session_factory=factory)

    student_id = uuid.uuid4()
    session_id = uuid.uuid4()

    # 1. Successful prediction
    r1 = service.topic_extraction_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text="Can you explain PEMDAS?",
    )
    assert not r1.is_abstain

    # 2. Abstention prediction
    r2 = service.topic_extraction_service.extract_and_audit(
        student_id=student_id,
        session_id=session_id,
        text="Can you help me with something?",
    )
    assert r2.is_abstain

    with factory() as session:
        logs = list(session.scalars(select(TopicExtractionLog).order_by(TopicExtractionLog.created_at.asc())))
        assert len(logs) == 2
        assert logs[0].extraction_method == "minilm_finetuned"
        assert logs[0].canonical_skill_id is not None
        assert logs[0].needs_review is False

        assert logs[1].extraction_method == "abstain"
        assert logs[1].canonical_skill_id is None
        assert logs[1].needs_review is True


def test_real_question_context_http_endpoint(runtime_test_db, monkeypatch):
    """Verify full end-to-end question-context HTTP endpoint with different query paths."""
    factory = runtime_test_db
    service = QuestionContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_question_context_service", lambda: service)
    client = TestClient(app, raise_server_exceptions=True)

    # Neural path
    resp_kw = client.post("/memory/question-context", json={
        "student_id": "stud_rt_1",
        "session_id": "sess_rt_1",
        "question": "Can you explain PEMDAS?",
    })
    assert resp_kw.status_code == 200
    assert resp_kw.json()["topic"]["method"] == "minilm_finetuned"

    # Semantic MiniLM path
    resp_sem = client.post("/memory/question-context", json={
        "student_id": "stud_rt_1",
        "session_id": "sess_rt_1",
        "question": "How do I work out the steepness of a line from two points on a graph?",
    })
    assert resp_sem.status_code == 200
    assert resp_sem.json()["topic"]["method"] == "minilm_finetuned"

    # Abstention path
    resp_abs = client.post("/memory/question-context", json={
        "student_id": "stud_rt_1",
        "session_id": "sess_rt_1",
        "question": "Can you help me?",
    })
    assert resp_abs.status_code == 200
    assert resp_abs.json()["topic"]["method"] == "abstain"
    assert resp_abs.json()["topic"]["skill_id"] is None
    assert resp_abs.json()["topic"]["needs_review"] is True


def test_runtime_latency_benchmarks():
    """Measure inference latency across 100 questions and path-specific routes."""
    extractor = get_topic_extractor()

    # Warmup
    extractor.extract("What is PEMDAS?")

    # Keyword queries
    kw_queries = [
        "Can you explain PEMDAS?",
        "What is Pythagorean theorem?",
        "How to find prime factorization?",
        "Order of operations rules",
    ]
    t0 = time.perf_counter()
    for q in kw_queries * 25:  # 100 calls
        extractor.extract(q)
    kw_lat_ms = (time.perf_counter() - t0) / 100.0 * 1000.0

    # Semantic MiniLM queries
    sem_queries = [
        "How steep is this line when plotted on a graph?",
        "How do I determine the center value of an ordered list of test scores?",
        "What is the mathematical definition of a circle circumference?",
        "How to calculate the probability of rolling a 6 on a die?",
    ]
    t0 = time.perf_counter()
    for q in sem_queries * 25:  # 100 calls
        extractor.extract(q)
    sem_lat_ms = (time.perf_counter() - t0) / 100.0 * 1000.0

    # Abstention queries
    abs_queries = [
        "Can you help me?",
        "I have a question",
        "What should I do next?",
        "Hi there",
    ]
    t0 = time.perf_counter()
    for q in abs_queries * 25:  # 100 calls
        extractor.extract(q)
    abs_lat_ms = (time.perf_counter() - t0) / 100.0 * 1000.0

    assert kw_lat_ms < 150.0, f"Keyword latency {kw_lat_ms:.2f}ms exceeds 150ms"
    assert sem_lat_ms < 300.0, f"MiniLM latency {sem_lat_ms:.2f}ms exceeds 300ms"
    assert abs_lat_ms < 300.0, f"Abstention latency {abs_lat_ms:.2f}ms exceeds 300ms"
