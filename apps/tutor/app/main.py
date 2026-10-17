import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.complexity import (
    router as complexity_router,
)
from app.api.planner import (
    router as planner_router,
)
from app.api.profile import (
    router as profile_router,
)
from app.api.tutor import (
    router as tutor_router,
)
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.readiness import get_readiness_report
from app.schemas.tutor_context import (
    TutorRequestContext,
)


settings = get_settings()

configure_logging(
    settings.log_level
)

logger = logging.getLogger(
    "adaptmath.api"
)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Research backend for the AdaptMath "
        "adaptive mathematics tutor."
    ),
    version="0.1.0",
)


# CORS is infrastructure configuration only. Browser origins are explicit and
# environment-configurable so the frontend can move between local development
# and deployment without opening the API to every website.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Accept",
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    ],
    expose_headers=[
        "X-Request-ID",
    ],
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    """
    Log safe request metadata without recording request bodies,
    learner answers, prompts, expected answers, or memory data.
    """

    request_id = str(
        uuid4()
    )

    started = time.perf_counter()

    try:
        response = await call_next(
            request
        )

    except Exception:
        duration_ms = (
            time.perf_counter()
            - started
        ) * 1000

        logger.exception(
            "request_failed request_id=%s method=%s path=%s "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        raise

    duration_ms = (
        time.perf_counter()
        - started
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    logger.info(
        "request_complete request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


app.include_router(
    complexity_router
)

app.include_router(
    planner_router
)

app.include_router(
    tutor_router
)

app.include_router(
    profile_router
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {
        "status": "ok",
        "service": "adaptmath-backend",
        "environment": settings.app_env,
    }


@app.get("/ready")
def readiness_check() -> dict:
    """Readiness probe for local dependencies required to serve tutoring."""
    report = get_readiness_report()

    if report["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=report,
        )

    return report


@app.post("/test/context")
def test_context(
    context: TutorRequestContext,
) -> TutorRequestContext:
    return context
