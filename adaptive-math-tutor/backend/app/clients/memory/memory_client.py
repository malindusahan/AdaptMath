"""Authenticated, fail-open HTTP client for the external Memory service."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Literal, Mapping
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)


class MemoryMisconception(BaseModel):
    """Allowlisted misconception fields from Memory."""

    model_config = ConfigDict(extra="ignore")
    display_error: str = Field(min_length=1, max_length=1000)
    occurrence_count: int = Field(default=1, ge=1)
    last_seen_at: datetime | None = None


class MemoryInteractionEvidence(BaseModel):
    """Allowlisted objective interaction summary; no answer text is accepted."""

    model_config = ConfigDict(extra="ignore")
    is_correct: bool | None = None
    attempt_count: int | None = Field(default=None, ge=0)
    hint_count: int | None = Field(default=None, ge=0)
    response_time_ms: float | None = Field(default=None, ge=0)
    created_at: datetime | None = None


class MemoryTutorContextResponse(BaseModel):
    """Strict retrieval projection that discards state/mastery/policy fields."""

    model_config = ConfigDict(extra="ignore")
    student_id: str = Field(min_length=1, max_length=255)
    misconceptions: list[MemoryMisconception] = Field(
        default_factory=list,
        max_length=20,
    )
    recent_interactions: list[MemoryInteractionEvidence] = Field(
        default_factory=list,
        max_length=20,
    )


class MemoryTopicClassification(BaseModel):
    """Allowlisted result from Memory's stateless topic classifier."""

    model_config = ConfigDict(extra="ignore")
    topic: str | None = Field(default=None, max_length=255)
    skill_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    is_math: bool
    model_version: str = Field(min_length=1, max_length=128)


class MemoryAuthenticatedUser(BaseModel):
    """Allowlisted non-secret learner identity returned by Memory ``/auth/me``."""

    model_config = ConfigDict(extra="ignore")
    user_id: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=255)
    role: Literal["STUDENT"]
    student_id: str = Field(min_length=1, max_length=255)
    age: int | None = Field(default=None, ge=0)


class CompletedAttemptAcknowledgement(BaseModel):
    """Safe acknowledgement from the evidence-only ingestion endpoint."""

    model_config = ConfigDict(extra="forbid")
    receipt_id: str
    external_student_id: str
    external_session_id: str
    external_attempt_id: str
    canonical_skill_id: str
    source: Literal["adaptmath"]
    status: Literal["STORED", "ALREADY_STORED"]
    session_status: Literal["ACTIVE", "COMPLETED"]
    question_count: int = Field(ge=1)
    correct_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
    created_at: str


class MemoryClient:
    """Small synchronous client used only at graph integration boundaries."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.settings = settings or get_settings()
        self._http_client = http_client

    def _headers(self) -> dict[str, str] | None:
        if not self.settings.memory_enabled:
            return None
        secret = self.settings.memory_service_api_key
        key = secret.get_secret_value().strip() if secret is not None else ""
        if not key:
            logger.warning("Memory integration skipped: service key is not configured.")
            return None
        return {"X-Service-Key": key}

    def _client(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[httpx.Client, bool]:
        if self._http_client is not None:
            return self._http_client, False
        return (
            httpx.Client(
                timeout=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self.settings.memory_timeout_seconds
                )
            ),
            True,
        )

    def classify_question(
        self,
        *,
        question: str,
    ) -> MemoryTopicClassification | None:
        """Classify a question, or return None when Memory is unavailable."""
        headers = self._headers()
        if headers is None:
            return None
        client, owns_client = self._client(
            timeout_seconds=self.settings.memory_topic_timeout_seconds,
        )
        try:
            response = client.post(
                f"{self.settings.memory_api_url.rstrip('/')}/topic/classify",
                json={"question": question},
                headers=headers,
            )
            response.raise_for_status()
            return MemoryTopicClassification.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError):
            logger.warning(
                "Memory topic classification failed; no skill was selected."
            )
            return None
        finally:
            if owns_client:
                client.close()

    def authenticate_user(self, token: str) -> MemoryAuthenticatedUser | None:
        """Validate one browser session token against Memory, failing closed."""
        normalized_token = token.strip()
        if not normalized_token:
            return None
        client, owns_client = self._client()
        try:
            response = client.get(
                f"{self.settings.memory_api_url.rstrip('/')}/auth/me",
                headers={"Authorization": f"Bearer {normalized_token}"},
            )
            response.raise_for_status()
            return MemoryAuthenticatedUser.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError):
            logger.warning("Memory user authentication failed closed.")
            return None
        finally:
            if owns_client:
                client.close()

    def retrieve_tutor_context(
        self,
        *,
        student_id: str,
        target_skill: str,
        limit: int | None = None,
    ) -> MemoryTutorContextResponse | None:
        """Return bounded skill context, or None on an optional dependency failure."""
        headers = self._headers()
        if headers is None:
            return None
        bounded_limit = max(
            1,
            min(limit or self.settings.memory_history_limit, 10),
        )
        client, owns_client = self._client()
        try:
            response = client.get(
                f"{self.settings.memory_api_url.rstrip('/')}/memory/"
                f"{quote(student_id, safe='')}/tutor-context",
                params={"skill_id": target_skill, "limit": bounded_limit},
                headers=headers,
            )
            response.raise_for_status()
            parsed = MemoryTutorContextResponse.model_validate(response.json())
            if parsed.student_id.strip() != student_id.strip():
                logger.warning(
                    "Memory retrieval returned a mismatched learner identity; "
                    "tutoring will continue without history."
                )
                return None
            return parsed
        except (httpx.HTTPError, ValueError, ValidationError):
            logger.warning("Memory retrieval failed; tutoring will continue without history.")
            return None
        finally:
            if owns_client:
                client.close()

    def write_completed_attempt(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write a stable attempt once; return retry metadata instead of raising."""
        headers = self._headers()
        if headers is None:
            return {
                "status": "disabled",
                "retryable": bool(self.settings.memory_enabled),
            }
        client, owns_client = self._client()
        try:
            response = client.post(
                f"{self.settings.memory_api_url.rstrip('/')}/memory/"
                "adaptmath/completed-attempt",
                json=dict(payload),
                headers=headers,
            )
            response.raise_for_status()
            acknowledgement = CompletedAttemptAcknowledgement.model_validate(
                response.json()
            )
            return {
                "status": acknowledgement.status,
                "receipt_id": acknowledgement.receipt_id,
                "external_attempt_id": acknowledgement.external_attempt_id,
                "retryable": False,
            }
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 409:
                logger.warning(
                    "Memory rejected a reused attempt ID with different evidence."
                )
                status = "conflict"
            else:
                logger.warning(
                    "Memory rejected the completed-attempt write (HTTP %s).",
                    status_code,
                )
                status = "unavailable" if status_code >= 500 else "rejected"
            return {
                "status": status,
                "external_attempt_id": payload.get("external_attempt_id"),
                "retryable": status_code >= 500 or status_code in {408, 429},
            }
        except httpx.RequestError:
            logger.warning(
                "Memory write failed after adaptive completion; stable retry is pending."
            )
            return {
                "status": "unavailable",
                "external_attempt_id": payload.get("external_attempt_id"),
                "retryable": True,
            }
        except (ValueError, ValidationError):
            logger.warning("Memory returned an invalid write acknowledgement.")
            return {
                "status": "invalid_acknowledgement",
                "external_attempt_id": payload.get("external_attempt_id"),
                "retryable": False,
            }
        finally:
            if owns_client:
                client.close()


_MEMORY_CLIENT: MemoryClient | None = None


def get_memory_client() -> MemoryClient:
    global _MEMORY_CLIENT
    if _MEMORY_CLIENT is None:
        _MEMORY_CLIENT = MemoryClient()
    return _MEMORY_CLIENT
