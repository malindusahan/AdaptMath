"""Global exception handlers providing standardized and sanitized error envelopes."""

from __future__ import annotations

import re
import uuid
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.schemas.api_common import ErrorDetail, ErrorResponse

# Sensitive patterns that must never leak into responses
SENSITIVE_PATTERNS = [
    re.compile(r"postgresql://[^\s'\"]+", re.IGNORECASE),
    re.compile(r"password=[^\s'\"]+", re.IGNORECASE),
    re.compile(r"secret=[^\s'\"]+", re.IGNORECASE),
]


def _sanitize_message(message: str) -> str:
    """Strip database URLs, passwords, and sensitive credentials from error text."""
    sanitized = message
    for pat in SENSITIVE_PATTERNS:
        sanitized = pat.sub("[REDACTED]", sanitized)
    return sanitized


def _get_request_id(request: Request) -> str:
    """Safely extract the active request_id from state or generate a fallback."""
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def _status_to_error_code(status_code: int) -> str:
    """Map standard HTTP status codes to standardized error code enums."""
    mapping = {
        status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
        status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
        status.HTTP_403_FORBIDDEN: "FORBIDDEN",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_409_CONFLICT: "CONFLICT",
        status.HTTP_413_CONTENT_TOO_LARGE: "PAYLOAD_TOO_LARGE",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "VALIDATION_ERROR",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_ERROR",
        status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, f"HTTP_ERROR_{status_code}")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle explicit HTTPExceptions and return standardized ErrorResponse envelope."""
    request_id = _get_request_id(request)
    error_code = _status_to_error_code(exc.status_code)
    details: list[ErrorDetail] = []

    if isinstance(exc.detail, dict):
        if "error_code" in exc.detail:
            error_code = str(exc.detail["error_code"])
        raw_message = str(exc.detail.get("message", "An error occurred."))
        raw_details = exc.detail.get("details", [])
        if isinstance(raw_details, list):
            for d in raw_details:
                if isinstance(d, dict):
                    details.append(
                        ErrorDetail(
                            field=d.get("field") or d.get("component"),
                            message=d.get("message") or d.get("status") or "error",
                            error_type=d.get("error_type"),
                        )
                    )
                elif isinstance(d, str):
                    details.append(ErrorDetail(message=d))
    else:
        raw_message = str(exc.detail) if exc.detail is not None else "An error occurred."

    message = _sanitize_message(raw_message)

    envelope = ErrorResponse(
        error_code=error_code,
        message=message,
        detail=message,
        details=details,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic/FastAPI request validation errors (HTTP 422)."""
    request_id = _get_request_id(request)
    details: list[ErrorDetail] = []

    for err in exc.errors():
        loc_parts = err.get("loc", [])
        field_str = ".".join(str(p) for p in loc_parts) if loc_parts else "unknown"
        msg = err.get("msg", "Invalid value.")
        err_type = str(err.get("type", "validation_error"))
        details.append(
            ErrorDetail(
                field=field_str,
                message=msg,
                error_type=err_type,
            )
        )

    message = "Request validation failed."
    envelope = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message=message,
        detail=message,
        details=details,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=envelope.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled server exceptions (HTTP 500) ensuring total redaction."""
    request_id = _get_request_id(request)
    message = "An internal server error occurred."

    envelope = ErrorResponse(
        error_code="INTERNAL_ERROR",
        message=message,
        detail=message,
        details=[],
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=envelope.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all global error handlers onto the FastAPI application."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
