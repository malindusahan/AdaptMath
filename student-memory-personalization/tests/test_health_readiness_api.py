"""Tests for /health and /ready service status endpoints."""

from __future__ import annotations

from pathlib import Path
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.app as app_module
import src.services.readiness_service as readiness_module
from src.database.base import Base
from src.database.models.core import Student
from src.services.readiness_service import ReadinessService


@pytest.fixture
def health_client():
    """Create test client for system endpoints."""
    return TestClient(app, raise_server_exceptions=False)


def test_health_returns_200_and_telemetry(health_client):
    """Verify /health is always alive, fast, and does not require database."""
    response = health_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "student-personalization-memory"

    # Tracing headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_health_works_when_db_fails(monkeypatch, health_client):
    """Verify /health succeeds even when database is completely down."""
    # Break readiness / db factory
    monkeypatch.setattr(app_module, "get_readiness_service", lambda: None)

    response = health_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_200_when_all_dependencies_ready(monkeypatch):
    """Verify /ready returns 200 when DB and all model artifacts are valid."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    svc = ReadinessService(session_factory=factory)
    monkeypatch.setattr(app_module, "get_readiness_service", lambda: svc)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "student-personalization-memory"
    assert data["database"] == "ready"
    assert data["migrations"] == "ready"
    assert data["topic_extractor"] == "ready"
    assert data["learning_state_model"] == "ready"

    # Headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_ready_returns_503_when_database_unavailable(monkeypatch):
    """Verify /ready returns 503 with standardized envelope when database is offline."""
    # Provide broken engine/factory
    broken_engine = create_engine("sqlite:///non_existent_dir_123/bad.db")
    broken_factory = sessionmaker(bind=broken_engine)

    svc = ReadinessService(session_factory=broken_factory)
    monkeypatch.setattr(app_module, "get_readiness_service", lambda: svc)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/ready")
    assert response.status_code == 503

    data = response.json()
    assert data["error_code"] == "SERVICE_NOT_READY"
    assert data["message"] == "Memory service is not ready."
    assert "request_id" in data
    assert "timestamp" in data
    assert len(data["details"]) > 0
    assert any("database" in d["field"] for d in data["details"])

    # Headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_ready_returns_503_when_learning_state_model_missing(monkeypatch):
    """Verify /ready returns 503 when learning state model artifacts are missing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with tempfile.TemporaryDirectory() as tmpdir:
        svc = ReadinessService(session_factory=factory, artifacts_dir=Path(tmpdir))
        monkeypatch.setattr(app_module, "get_readiness_service", lambda: svc)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/ready")
        assert response.status_code == 503

        data = response.json()
        assert data["error_code"] == "SERVICE_NOT_READY"
        fields = [d["field"] for d in data["details"]]
        assert "learning_state_model" in fields
        assert "topic_extractor" in fields


def test_ready_returns_503_when_topic_extractor_missing(monkeypatch):
    """Verify /ready returns 503 when topic extractor artifacts are missing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create dummy learning state model files
        (tmp_path / "learning_state_model.joblib").write_text("dummy")
        (tmp_path / "learning_state_imputer.joblib").write_text("dummy")
        (tmp_path / "learning_state_model_metadata.json").write_text("{}")

        svc = ReadinessService(session_factory=factory, artifacts_dir=tmp_path)
        monkeypatch.setattr(app_module, "get_readiness_service", lambda: svc)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/ready")
        assert response.status_code == 503

        data = response.json()
        assert data["error_code"] == "SERVICE_NOT_READY"
        fields = [d["field"] for d in data["details"]]
        assert "topic_extractor" in fields
        assert "learning_state_model" not in fields


def test_ready_returns_503_when_topic_runtime_dependency_is_missing(monkeypatch):
    """Artifacts alone must not make readiness pass without its runtime."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    monkeypatch.setattr(
        readiness_module,
        "_topic_runtime_available",
        lambda: False,
    )
    svc = ReadinessService(session_factory=factory)
    monkeypatch.setattr(app_module, "get_readiness_service", lambda: svc)

    response = TestClient(app, raise_server_exceptions=False).get("/ready")
    assert response.status_code == 503
    fields = [item["field"] for item in response.json()["details"]]
    assert "topic_extractor" in fields


def test_no_credential_leakage_on_503(monkeypatch):
    """Verify raw database connection strings or passwords never leak into 503 payloads."""
    secret_url = "postgresql://user:SecretPass999@secret-db.internal:5432/proddb"

    def fail_service():
        raise RuntimeError(f"Connection to {secret_url} failed")

    monkeypatch.setattr(app_module, "get_readiness_service", fail_service)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/ready")
    assert response.status_code == 500  # unhandled service init is caught as 500

    raw_text = response.text
    assert "SecretPass999" not in raw_text
    assert "secret-db.internal" not in raw_text
    assert "postgresql://" not in raw_text
