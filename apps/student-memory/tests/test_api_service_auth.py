"""Tests for Agent-to-Agent Service Authentication."""

from __future__ import annotations

import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
import src.api.memory_routes as memory_routes
from src.database.base import Base
from src.database.models.core import Student
from src.services.student_context_service import StudentContextService


@pytest.fixture
def auth_test_setup(monkeypatch):
    """Configure SQLite in-memory and mock service context for authenticated endpoints."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with factory() as session:
        st = Student(student_id=uuid.uuid4(), external_student_id="stud_auth_1")
        session.add(st)
        session.commit()

    service = StudentContextService(session_factory=factory)
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: service)

    client = TestClient(app, raise_server_exceptions=False)
    return client, factory


def test_public_health_and_ready_endpoints_accessible_without_key(monkeypatch, auth_test_setup):
    """Verify /health and /ready are public and never require X-Service-Key."""
    client, _ = auth_test_setup
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", "super-secret-key-12345")

    # /health without key
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # /ready without key
    r_ready = client.get("/ready")
    # /ready returns 200 or 503 depending on model artifact existence, but NOT 401
    assert r_ready.status_code in (200, 503)


def test_protected_endpoint_without_key_returns_401(monkeypatch, auth_test_setup):
    """Verify accessing protected routes without X-Service-Key returns 401 UNAUTHORIZED."""
    client, _ = auth_test_setup
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", "super-secret-key-12345")

    response = client.get("/memory/stud_auth_1/context")
    assert response.status_code == 401

    data = response.json()
    assert data["error_code"] == "UNAUTHORIZED"
    assert data["message"] == "Service API key is missing."
    assert "request_id" in data
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


def test_protected_endpoint_with_wrong_key_returns_401(monkeypatch, auth_test_setup):
    """Verify accessing protected routes with invalid X-Service-Key returns 401 UNAUTHORIZED."""
    client, _ = auth_test_setup
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", "super-secret-key-12345")

    response = client.get(
        "/memory/stud_auth_1/context",
        headers={"X-Service-Key": "wrong-invalid-secret-key"},
    )
    assert response.status_code == 401

    data = response.json()
    assert data["error_code"] == "UNAUTHORIZED"
    assert data["message"] == "Invalid service API key."
    assert "request_id" in data


def test_protected_endpoint_with_correct_key_succeeds(monkeypatch, auth_test_setup):
    """Verify valid X-Service-Key successfully grants access to protected routes."""
    client, _ = auth_test_setup
    test_key = "valid-agent-secret-key-999"
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", test_key)

    response = client.get(
        "/memory/stud_auth_1/context",
        headers={"X-Service-Key": test_key},
    )
    assert response.status_code == 200
    assert response.json()["student_id"] == "stud_auth_1"


def test_secret_key_never_appears_in_error_responses(monkeypatch, auth_test_setup):
    """Verify expected secret API key is never reflected or leaked in 401 payloads."""
    client, _ = auth_test_setup
    secret_key = "ultra_classified_agent_token_888"
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", secret_key)

    response = client.get(
        "/memory/stud_auth_1/context",
        headers={"X-Service-Key": "bad-key"},
    )
    assert response.status_code == 401
    assert secret_key not in response.text


def test_auth_disabled_mode_allows_unauthenticated_requests(monkeypatch, auth_test_setup):
    """Verify MEMORY_AUTH_ENABLED=false permits unauthenticated requests."""
    client, _ = auth_test_setup
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "false")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", "some-key")

    response = client.get("/memory/stud_auth_1/context")
    assert response.status_code == 200
    assert response.json()["student_id"] == "stud_auth_1"


def test_incoming_request_id_preserved_on_401(monkeypatch, auth_test_setup):
    """Verify X-Request-ID is preserved when 401 UNAUTHORIZED is returned."""
    client, _ = auth_test_setup
    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", "expected-secret")
    custom_trace = "trace-auth-fail-777"

    response = client.get(
        "/memory/stud_auth_1/context",
        headers={"X-Request-ID": custom_trace},
    )
    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == custom_trace
    assert response.json()["request_id"] == custom_trace
