"""Tests for Runtime Configuration and CORS Middleware Hardening."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from src.api.app import app
from src.api.runtime_config import (
    RuntimeConfig,
    RuntimeConfigError,
    load_runtime_config,
)


def test_default_development_config():
    """Verify default runtime settings fall back to local dev origin with credentials."""
    config = load_runtime_config(env={})
    assert config.environment == "development"
    assert config.allowed_origins == ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"]
    assert config.allow_credentials is True
    assert "GET" in config.allowed_methods
    assert "POST" in config.allowed_methods


def test_multiple_configured_origins_parsed():
    """Verify comma-separated MEMORY_ALLOWED_ORIGINS environment variable is cleanly parsed."""
    env = {
        "MEMORY_ALLOWED_ORIGINS": "http://localhost:3000, https://tutor.example.edu , http://127.0.0.1:8080",
        "MEMORY_ENVIRONMENT": "production",
    }
    config = load_runtime_config(env=env)
    assert config.environment == "production"
    assert config.allowed_origins == [
        "http://localhost:3000",
        "https://tutor.example.edu",
        "http://127.0.0.1:8080",
    ]


def test_invalid_wildcard_with_credentials_rejected():
    """Verify combining '*' wildcard origin with credentials=True is strictly rejected."""
    env = {
        "MEMORY_ALLOWED_ORIGINS": "*",
        "MEMORY_ALLOW_CREDENTIALS": "true",
    }
    with pytest.raises(RuntimeConfigError, match=r"wildcard origin '\*' cannot be combined with allow_credentials=True"):
        load_runtime_config(env=env)


def test_empty_origins_rejected():
    """Verify empty origin string raises clear RuntimeConfigError."""
    env = {"MEMORY_ALLOWED_ORIGINS": "   ,   "}
    with pytest.raises(RuntimeConfigError, match="allowed_origins cannot be empty"):
        load_runtime_config(env=env)


def test_allowed_origin_receives_cors_headers():
    """Verify requests originating from allowed origin receive Access-Control-Allow-Origin header."""
    client = TestClient(app)

    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_unauthorized_origin_does_not_receive_cors_headers():
    """Verify requests from disallowed origins do NOT receive Access-Control-Allow-Origin header."""
    client = TestClient(app)

    response = client.get(
        "/health",
        headers={"Origin": "https://malicious-site.attacker.com"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_options_request():
    """Verify browser preflight OPTIONS request returns 200 with appropriate CORS headers."""
    client = TestClient(app)

    response = client.options(
        "/memory/s1/context",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Request-ID, Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_existing_api_routes_functional_with_cors():
    """Verify standard routes continue operating seamlessly with CORS middleware attached."""
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers
