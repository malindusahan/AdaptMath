"""Runtime configuration and environment settings for the Student Memory API."""

from __future__ import annotations

import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuntimeConfigError(ValueError):
    """Raised when runtime environment configuration is invalid."""


class RuntimeConfig(BaseModel):
    """Immutable runtime configuration parsed strictly from environment variables."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production, test).",
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
        description="List of allowed CORS origins.",
    )
    allow_credentials: bool = Field(
        default=True,
        description="Whether to support CORS credentials (cookies, authorization headers).",
    )
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        description="Allowed HTTP methods.",
    )
    allowed_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed HTTP headers.",
    )

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, origins: list[str]) -> list[str]:
        cleaned = [o.strip() for o in origins if o and o.strip()]
        if not cleaned:
            raise RuntimeConfigError("allowed_origins cannot be empty.")
        return cleaned

    def model_post_init(self, __context: object) -> None:
        """Enforce strict security invariant: wildcard origin with credentials is forbidden."""
        if "*" in self.allowed_origins and self.allow_credentials:
            raise RuntimeConfigError(
                "Security violation: wildcard origin '*' cannot be combined with allow_credentials=True."
            )


def load_runtime_config(env: dict[str, str] | None = None) -> RuntimeConfig:
    """
    Load runtime configuration from explicit dict or os.environ.
    Environment variables:
    - MEMORY_ENVIRONMENT
    - MEMORY_ALLOWED_ORIGINS (comma-separated)
    - MEMORY_ALLOW_CREDENTIALS (true/false)
    - MEMORY_ALLOWED_METHODS (comma-separated)
    - MEMORY_ALLOWED_HEADERS (comma-separated)
    """
    environ = os.environ if env is None else env

    env_name = environ.get("MEMORY_ENVIRONMENT", "development").strip()

    # Parse origins
    raw_origins = environ.get("MEMORY_ALLOWED_ORIGINS")
    if raw_origins is not None:
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        origins = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"]

    # Parse credentials
    raw_cred = environ.get("MEMORY_ALLOW_CREDENTIALS", "true").strip().lower()
    allow_credentials = raw_cred in ("true", "1", "yes", "on")

    # Parse methods
    raw_methods = environ.get("MEMORY_ALLOWED_METHODS")
    if raw_methods is not None:
        methods = [m.strip().upper() for m in raw_methods.split(",") if m.strip()]
    else:
        methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]

    # Parse headers
    raw_headers = environ.get("MEMORY_ALLOWED_HEADERS")
    if raw_headers is not None:
        headers = [h.strip() for h in raw_headers.split(",") if h.strip()]
    else:
        headers = ["*"]

    try:
        return RuntimeConfig(
            environment=env_name,
            allowed_origins=origins,
            allow_credentials=allow_credentials,
            allowed_methods=methods,
            allowed_headers=headers,
        )
    except Exception as exc:
        raise RuntimeConfigError(f"Invalid runtime configuration: {exc}") from exc


_RUNTIME_CONFIG: RuntimeConfig | None = None


def get_runtime_config() -> RuntimeConfig:
    """Global cached runtime configuration loader."""
    global _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = load_runtime_config()
    return _RUNTIME_CONFIG


def set_runtime_config(config: RuntimeConfig | None) -> None:
    """Override cached runtime config (primarily for tests)."""
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config
