import logging
from typing import Any

from fastapi import HTTPException


logger = logging.getLogger(
    "adaptmath.model_service"
)


def _retry_after_headers(
    exc: Exception,
) -> dict[str, str] | None:
    """
    Preserve an upstream Retry-After hint when one is
    available without exposing the provider response body.
    """

    response: Any = getattr(
        exc,
        "response",
        None,
    )

    headers: Any = getattr(
        response,
        "headers",
        None,
    )

    if headers is None:
        return None

    retry_after = None

    try:
        retry_after = (
            headers.get("retry-after")
            or headers.get("Retry-After")
        )
    except AttributeError:
        return None

    if retry_after is None:
        return None

    return {
        "Retry-After": str(
            retry_after
        )
    }


def _log_model_failure(
    *,
    category: str,
    exc: Exception,
    upstream_status: Any,
) -> None:
    """
    Record only safe diagnostic metadata.

    Never log str(exc): provider exceptions can contain
    organization identifiers, quota information, request
    payloads, or other details that should not be persisted.
    """

    logger.warning(
        "model_service_failure category=%s "
        "exception_type=%s upstream_status=%s",
        category,
        type(exc).__name__,
        upstream_status,
    )


def model_service_http_exception(
    exc: Exception,
) -> HTTPException:
    """
    Convert model-provider/pipeline failures into safe public
    HTTP errors.

    Raw provider messages can contain account identifiers,
    quota details, request IDs, or other implementation data,
    so they must not be returned directly to frontend clients
    or written to application logs.
    """

    exception_name = type(exc).__name__

    upstream_status = getattr(
        exc,
        "status_code",
        None,
    )

    if (
        upstream_status == 429
        or exception_name
        == "RateLimitError"
    ):
        _log_model_failure(
            category="rate_limit",
            exc=exc,
            upstream_status=upstream_status,
        )

        return HTTPException(
            status_code=503,
            detail=(
                "The AI model service is temporarily "
                "rate-limited. Please retry later."
            ),
            headers=(
                _retry_after_headers(
                    exc
                )
            ),
        )

    if exception_name in {
        "APITimeoutError",
        "TimeoutError",
    }:
        _log_model_failure(
            category="timeout",
            exc=exc,
            upstream_status=upstream_status,
        )

        return HTTPException(
            status_code=504,
            detail=(
                "The AI model service timed out. "
                "Please retry."
            ),
        )

    if exception_name in {
        "APIConnectionError",
        "ConnectionError",
    }:
        _log_model_failure(
            category="connection",
            exc=exc,
            upstream_status=upstream_status,
        )

        return HTTPException(
            status_code=503,
            detail=(
                "The AI model service is temporarily "
                "unavailable. Please retry later."
            ),
        )

    if isinstance(
        upstream_status,
        int,
    ):
        _log_model_failure(
            category="upstream_status",
            exc=exc,
            upstream_status=upstream_status,
        )

        return HTTPException(
            status_code=502,
            detail=(
                "The AI model provider could not complete "
                "the backend request."
            ),
        )

    _log_model_failure(
        category="pipeline",
        exc=exc,
        upstream_status=upstream_status,
    )

    return HTTPException(
        status_code=502,
        detail=(
            "The AI tutoring pipeline could not complete "
            "the request."
        ),
    )