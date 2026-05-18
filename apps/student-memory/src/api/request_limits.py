"""Request limit policies, pagination defaults, and request size protection middleware."""

from __future__ import annotations

import os
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from fastapi import status

from src.schemas.api_common import ErrorResponse

MAX_REQUEST_BYTES_ENV = "MEMORY_MAX_REQUEST_BYTES"
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024  # 1 MB
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100


def get_max_request_bytes() -> int:
    """Read maximum permitted request body size in bytes from environment."""
    raw_val = os.environ.get(MAX_REQUEST_BYTES_ENV)
    if raw_val is not None:
        try:
            return int(raw_val.strip())
        except ValueError:
            return DEFAULT_MAX_REQUEST_BYTES
    return DEFAULT_MAX_REQUEST_BYTES


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Guards the service against denial-of-service or memory bloat from oversized request bodies.
    Enforces MEMORY_MAX_REQUEST_BYTES (default: 1 MB).
    Returns HTTP 413 with standardized ErrorResponse.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        max_bytes = get_max_request_bytes()

        # Check Content-Length header if present
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
                if content_length > max_bytes:
                    request_id = getattr(request.state, "request_id", None) or "req-overflow"
                    envelope = ErrorResponse(
                        error_code="PAYLOAD_TOO_LARGE",
                        message=f"Request payload of {content_length} bytes exceeds limit of {max_bytes} bytes.",
                        detail="Request payload exceeds maximum allowed size.",
                        details=[],
                        request_id=request_id,
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content=envelope.model_dump(mode="json"),
                        headers={"X-Request-ID": request_id},
                    )
            except ValueError:
                pass

        return await call_next(request)
