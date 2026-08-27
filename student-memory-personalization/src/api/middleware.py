"""ASGI middleware for request tracing and latency telemetry."""

from __future__ import annotations

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-MS"


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware ensuring every HTTP request carries an auditable X-Request-ID
    and calculates elapsed processing time in X-Response-Time-MS.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract incoming X-Request-ID or generate new UUID
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id or not request_id.strip():
            request_id = uuid.uuid4().hex
        else:
            request_id = request_id.strip()

        # Attach to request state
        request.state.request_id = request_id

        # Measure wall-clock execution time
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Let global exception handlers capture exception, but ensure request_id is attached
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Inject tracing headers into response
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{duration_ms:.2f}"

        return response
