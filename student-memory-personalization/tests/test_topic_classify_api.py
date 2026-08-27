"""End-to-end API tests for POST /topic/classify dedicated topic extraction endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=True)


def test_classify_linear_equation(client: TestClient):
    """Verify standard math equation extracts Linear Equations topic."""
    response = client.post(
        "/topic/classify",
        json={"question": "Solve the equation 3x + 5 = 20 for x."},
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_math"] is True
    assert data["topic"] in ("Linear Equations", "Algebraic Solving", "Equation Solving Two or Fewer Steps")
    assert data["skill_id"] is not None
    assert data["confidence"] >= 0.25
    assert data["model_version"] == "phase16-minilm-ft-v2"


def test_classify_pythagorean_theorem(client: TestClient):
    """Verify geometry word problem extracts Pythagorean Theorem."""
    response = client.post(
        "/topic/classify",
        json={"question": "A ladder 13m long leans on a wall 5m away. How high is the wall?"},
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_math"] is True
    assert data["topic"] == "Pythagorean Theorem"
    assert data["skill_id"] == "SKILL_27"
    assert data["confidence"] >= 0.35


def test_classify_real_world_train_story_problem(client: TestClient):
    """Verify narrative elapsed time story problem extracts arithmetic topic."""
    response = client.post(
        "/topic/classify",
        json={"question": "A train leaves Colombo at 8:35 AM and reaches its destination at 11:20 AM. How long was the journey?"},
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_math"] is True
    assert data["topic"] in ("Subtraction Whole Numbers", "Rate", "Computation with Real Numbers")
    assert data["confidence"] >= 0.25


def test_classify_non_math_greeting_abstains(client: TestClient):
    """Verify off-topic greeting returns topic=null and is_math=false."""
    response = client.post(
        "/topic/classify",
        json={"question": "Hello there, how is the weather today?"},
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_math"] is False
    assert data["topic"] is None
    assert data["skill_id"] is None
    assert data["confidence"] < 0.30


def test_classify_empty_string_validation_error(client: TestClient):
    """Verify empty string triggers 422 schema validation error."""
    response = client.post(
        "/topic/classify",
        json={"question": ""},
        headers={"X-Service-Key": "test-service-key"},
    )
    assert response.status_code == 422
