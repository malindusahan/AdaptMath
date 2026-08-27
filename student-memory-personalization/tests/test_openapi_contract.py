"""Tests for OpenAPI contract schema, endpoint consistency, and documentation integrity."""

from __future__ import annotations

import pytest
from src.api.app import app


@pytest.fixture(scope="module")
def openapi_schema():
    """Retrieve generated OpenAPI schema dictionary."""
    return app.openapi()


def test_openapi_schema_loads_successfully(openapi_schema):
    """Verify OpenAPI schema can be generated and contains title and version."""
    assert openapi_schema is not None
    assert "paths" in openapi_schema
    assert openapi_schema["info"]["title"] == "Student Personalization Memory Service"
    assert openapi_schema["info"]["version"] == "1.0.0"


def test_all_expected_endpoints_are_present(openapi_schema):
    """Verify all 13 system and agent memory endpoints are present in OpenAPI spec."""
    paths = openapi_schema["paths"]

    expected_routes = [
        "/health",
        "/ready",
        "/memory/update",
        "/memory/{student_id}/current",
        "/memory/{student_id}/history",
        "/memory/question-context",
        "/memory/{student_id}/context",
        "/memory/{student_id}/tutor-context",
        "/memory/{student_id}/planner-context",
        "/memory/{student_id}/fapr-context",
        "/memory/repair-outcome",
        "/memory/{student_id}/support-preference",
        "/memory/{student_id}/meta-signals",
    ]

    for route in expected_routes:
        assert route in paths, f"Expected route {route} missing from OpenAPI paths."


def test_public_endpoints_have_no_security_requirement(openapi_schema):
    """Verify /health and /ready do not require security credentials in OpenAPI."""
    paths = openapi_schema["paths"]
    for public_path in ("/health", "/ready"):
        get_op = paths[public_path]["get"]
        assert "security" not in get_op or len(get_op["security"]) == 0


def test_protected_endpoints_expose_service_key_security(openapi_schema):
    """Verify /memory/... endpoints declare X-Service-Key security requirements."""
    paths = openapi_schema["paths"]
    assert "components" in openapi_schema
    assert "securitySchemes" in openapi_schema["components"]
    
    sec_schemes = openapi_schema["components"]["securitySchemes"]
    found_key = any(s.get("name") == "X-Service-Key" for s in sec_schemes.values())
    assert found_key, f"X-Service-Key scheme not found in securitySchemes: {sec_schemes}"

    for path, methods in paths.items():
        if path.startswith("/memory"):
            for method, operation in methods.items():
                if method.lower() in ("get", "post", "put", "delete"):
                    # Security defined at operation or router level
                    assert "security" in operation or "security" in openapi_schema, (
                        f"Route {method.upper()} {path} missing security declaration"
                    )


def test_no_duplicate_operation_ids(openapi_schema):
    """Verify every endpoint in the OpenAPI schema has a unique operation_id."""
    paths = openapi_schema["paths"]
    operation_ids: list[str] = []

    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.lower() in ("get", "post", "put", "delete", "patch"):
                op_id = operation.get("operationId")
                assert op_id is not None, f"Route {method.upper()} {path} is missing operationId."
                operation_ids.append(op_id)

    duplicates = [op for op in operation_ids if operation_ids.count(op) > 1]
    assert len(duplicates) == 0, f"Found duplicate operationIds: {set(duplicates)}"
    assert len(operation_ids) >= 13


def test_post_routes_have_request_schema_examples(openapi_schema):
    """Verify POST routes (/update, /question-context, /repair-outcome) expose schema examples."""
    components = openapi_schema.get("components", {}).get("schemas", {})

    for schema_name in (
        "AssessmentMemoryUpdateRequest",
        "QuestionContextRequest",
        "RepairOutcomeCreateRequest",
    ):
        assert schema_name in components, f"Schema {schema_name} not found in components."
        schema = components[schema_name]
        assert "example" in schema or "examples" in schema, (
            f"Schema {schema_name} is missing OpenAPI example documentation."
        )


def test_error_responses_documented(openapi_schema):
    """Verify 401/422/500 responses are documented across memory routes."""
    paths = openapi_schema["paths"]

    for path, methods in paths.items():
        if path.startswith("/memory"):
            for method, operation in methods.items():
                if method.lower() in ("get", "post"):
                    responses = operation.get("responses", {})
                    assert "401" in responses or "422" in responses or "500" in responses, (
                        f"Route {method.upper()} {path} missing documented error responses."
                    )
