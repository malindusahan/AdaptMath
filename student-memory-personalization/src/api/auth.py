"""Service authentication middleware and dependency injection for agent-to-agent communication."""

from __future__ import annotations

import hmac
import os
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

SERVICE_KEY_HEADER = "X-Service-Key"
AUTH_ENABLED_ENV = "MEMORY_AUTH_ENABLED"
SERVICE_API_KEY_ENV = "MEMORY_SERVICE_API_KEY"

service_key_security_scheme = APIKeyHeader(
    name=SERVICE_KEY_HEADER,
    auto_error=False,
    description="Agent-to-agent shared service secret key.",
)


def is_auth_enabled() -> bool:
    """Check whether service authentication is active (default: True)."""
    val = os.environ.get(AUTH_ENABLED_ENV, "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def get_configured_service_key() -> str | None:
    """Retrieve the expected service API key from environment."""
    key = os.environ.get(SERVICE_API_KEY_ENV)
    return key.strip() if key and key.strip() else None


async def verify_service_api_key(
    x_service_key: str | None = Security(service_key_security_scheme),
) -> str | None:
    """
    FastAPI dependency verifying agent-to-agent secret key.
    - If MEMORY_AUTH_ENABLED=false: allows request through.
    - If key is missing: raises HTTP 401 UNAUTHORIZED.
    - If key does not match MEMORY_SERVICE_API_KEY: raises HTTP 401 UNAUTHORIZED.
    - Secret keys are compared using constant-time comparison to prevent timing attacks.
    """
    if not is_auth_enabled():
        return None

    if not x_service_key or not x_service_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service API key is missing.",
        )

    expected_key = get_configured_service_key()
    if not expected_key or not hmac.compare_digest(x_service_key.strip(), expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service API key.",
        )

    return x_service_key.strip()
