"""
FastAPI application entry point for the Student Memory service.
"""

from fastapi import Depends, FastAPI

from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import verify_service_api_key
from src.api.error_handlers import register_error_handlers
from src.api.memory_routes import router as memory_router
from src.api.middleware import RequestTracingMiddleware
from src.api.request_limits import RequestSizeLimitMiddleware
from src.api.runtime_config import get_runtime_config


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Database/schema provisioning is intentionally explicit via Alembic and
    # scripts/provision_local_database.py. Normal process startup never creates
    # or alters persistence resources.
    del app
    yield


app = FastAPI(
    title="Student Personalization Memory Service",
    version="1.0.0",
    description=(
        "Learning-state memory service for assessment updates, "
        "current-state retrieval, and state-history retrieval."
    ),
    lifespan=lifespan,
)

config = get_runtime_config()

# Register Request Size Limit Middleware (MEMORY_MAX_REQUEST_BYTES)
app.add_middleware(RequestSizeLimitMiddleware)

# Register Request Tracing Middleware (X-Request-ID & X-Response-Time-MS)
app.add_middleware(RequestTracingMiddleware)

# Register CORS Middleware (outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=config.allow_credentials,
    allow_methods=config.allowed_methods,
    allow_headers=config.allowed_headers,
)

# Register Standardized Global Exception Handlers
register_error_handlers(app)

from src.api.auth_routes import router as auth_router
from src.api.ui_routes import router as ui_router
from src.api.topic_routes import router as topic_router

app.include_router(
    auth_router,
)

app.include_router(
    topic_router,
)

app.include_router(
    memory_router,
    dependencies=[Depends(verify_service_api_key)],
)

app.include_router(
    ui_router,
)


from src.schemas.api_common import HealthResponse, ReadinessResponse
from src.services.readiness_service import get_readiness_service


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Liveness Probe",
    operation_id="check_service_health",
    tags=["system"],
    responses={
        200: {"description": "Service process is alive and healthy."},
    },
)
def health_check() -> HealthResponse:
    """
    Lightweight service liveness probe.
    Does not depend on external databases or model assets.
    """
    return HealthResponse(
        status="ok",
        service="student-personalization-memory",
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Service Readiness & Dependency Check",
    operation_id="check_service_readiness",
    tags=["system"],
    responses={
        200: {"description": "All dependencies and ML model artifacts are ready."},
        503: {"description": "One or more core dependencies (PostgreSQL, models) are unavailable."},
    },
)
def readiness_check() -> ReadinessResponse:
    """
    Deep service readiness probe checking:
    - PostgreSQL database connection
    - Migration / schema health
    - Topic extractor model artifacts
    - Learning state classification model artifacts
    """
    service = get_readiness_service()
    return service.check_readiness()
