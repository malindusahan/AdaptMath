"""
HTTP routes for the Student Memory service.

The API layer is intentionally thin. Business logic remains in the
service and repository layers.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)

from src.database.postgres_config import DATABASE_URL_ENV
from src.schemas.api_common import ErrorResponse
from src.schemas.adaptmath import (
    AdaptMathCompletedAttemptRequest,
    AdaptMathCompletedAttemptResponse,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    CurrentStudentMemory,
    LearningStateHistoryItem,
    MemoryUpdateResponse,
)
from src.schemas.question_context import (
    QuestionContextRequest,
    QuestionContextResponse,
)
from src.schemas.fapr_context import (
    FAPRContextResponse,
)
from src.schemas.meta_signals import (
    MetaSignalsResponse,
)
from src.schemas.planner_context import (
    PlannerContextResponse,
)
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)
from src.schemas.student_context import (
    StudentContextResponse,
)
from src.schemas.support_preference import (
    SupportPreferenceResponse,
)
from src.schemas.tutor_context import (
    TutorContextResponse,
)
from src.services.fapr_context_service import (
    get_fapr_context_service,
)
from src.services.adaptmath_ingestion_service import (
    AdaptMathAttemptConflictError,
    AdaptMathIngestionError,
    AdaptMathSkillNotFoundError,
    get_adaptmath_ingestion_service,
)
from src.services.memory_update_service import (
    MemoryUpdateServiceError,
    update_student_memory,
)
from src.services.meta_signal_service import (
    get_meta_signal_service,
)
from src.services.planner_context_service import (
    get_planner_context_service,
)
from src.services.question_context_service import (
    get_question_context_service,
)
from src.services.repair_outcome_service import (
    get_repair_outcome_service,
)
from src.services.student_context_service import (
    get_student_context_service,
)
from src.services.student_memory_query_service import (
    get_student_memory_query_service,
)
from src.services.support_preference_service import (
    get_support_preference_service,
)
from src.services.tutor_context_service import (
    get_tutor_context_service,
)


router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)


@router.post(
    "/adaptmath/completed-attempt",
    response_model=AdaptMathCompletedAttemptResponse,
    status_code=status.HTTP_200_OK,
    summary="Idempotently ingest one completed AdaptMath attempt",
    operation_id="ingest_adaptmath_completed_attempt",
    responses={
        200: {"description": "Attempt evidence stored or replay acknowledged."},
        401: {"model": ErrorResponse, "description": "Missing or invalid service key."},
        404: {"model": ErrorResponse, "description": "Canonical skill not found."},
        409: {"model": ErrorResponse, "description": "Stable attempt key conflicts."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Transactional ingestion failed."},
    },
)
def ingest_adaptmath_completed_attempt(
    request: AdaptMathCompletedAttemptRequest,
) -> AdaptMathCompletedAttemptResponse:
    """Store objective evidence only; BKT and policy state are never accepted."""
    try:
        return get_adaptmath_ingestion_service().ingest_completed_attempt(request)
    except AdaptMathSkillNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical skill not found.",
        ) from exc
    except AdaptMathAttemptConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The completed attempt key conflicts with stored evidence.",
        ) from exc
    except AdaptMathIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Completed attempt ingestion failed.",
        ) from exc


def _require_non_blank(
    value: str,
    field_name: str,
) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} must not be blank.",
        )
    return value


def _normalize_optional_text(
    value: str | None,
) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_current_student_memory(
    student_id: str,
    topic: str,
    subtopic: str | None = None,
    database_path: Any = None,
    connection: Any = None,
) -> CurrentStudentMemory | None:
    """Retrieve current student memory from PostgreSQL with SQLite legacy fallback."""
    if os.environ.get(DATABASE_URL_ENV) or database_path is None:
        return get_student_memory_query_service().get_current_memory(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
        )
    from src.database.state_repository import (
        get_current_student_memory as sqlite_get_current,
    )

    return sqlite_get_current(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        database_path=database_path,
        connection=connection,
    )


def get_learning_state_history(
    student_id: str,
    topic: str,
    subtopic: str | None = None,
    database_path: Any = None,
    connection: Any = None,
) -> list[LearningStateHistoryItem]:
    """Retrieve learning state history from PostgreSQL with SQLite legacy fallback."""
    if os.environ.get(DATABASE_URL_ENV) or database_path is None:
        return get_student_memory_query_service().get_state_history(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
        )
    from src.database.state_repository import (
        get_learning_state_history as sqlite_get_history,
    )

    return sqlite_get_history(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        database_path=database_path,
        connection=connection,
    )


@router.post(
    "/update",
    response_model=MemoryUpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Assessment & Update Memory State",
    operation_id="process_memory_update",
    responses={
        200: {"description": "Assessment processed and dynamic learning state updated."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Internal memory update error."},
    },
)
def update_memory(
    request: AssessmentMemoryUpdateRequest,
) -> MemoryUpdateResponse:
    """
    Process one completed assessment and update the student's memory in PostgreSQL.

    The resulting learning state includes the completed assessment
    and is intended to be carried forward to the student's next
    learning activity.
    """

    try:
        return update_student_memory(
            request=request,
        )

    except MemoryUpdateServiceError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Memory update failed.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory update failed.",
        ) from exc


@router.get(
    "/{student_id}/current",
    response_model=CurrentStudentMemory,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Current Learning-State Memory",
    operation_id="get_current_memory_state",
    responses={
        200: {"description": "Current learning state memory record."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        404: {"model": ErrorResponse, "description": "Student memory record not found."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve student memory."},
    },
)
def get_current_memory(
    student_id: str,
    topic: str,
    subtopic: str | None = None,
) -> CurrentStudentMemory:
    """
    Retrieve the latest persisted learning-state memory for one
    student/topic/subtopic context from PostgreSQL.

    This endpoint is read-only and does not rerun the model, rebuild
    historical features, or modify database state.
    """

    student_id = _require_non_blank(student_id, "student_id")
    topic = _require_non_blank(topic, "topic")
    subtopic = _normalize_optional_text(subtopic)

    try:
        memory = get_current_student_memory(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve student memory.",
        ) from exc

    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student memory not found.",
        )

    return memory


@router.get(
    "/{student_id}/history",
    response_model=list[LearningStateHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Retrieve Learning-State History",
    operation_id="get_state_history_snapshots",
    responses={
        200: {"description": "List of historical learning state snapshots."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve learning-state history."},
    },
)
def get_state_history(
    student_id: str,
    topic: str,
    subtopic: str | None = None,
    limit: int = Query(default=20, ge=1, le=100, description="Max history snapshots returned."),
    offset: int = Query(default=0, ge=0, description="Number of history snapshots to skip."),
) -> list[LearningStateHistoryItem]:
    """
    Retrieve the student's persisted learning-state history for one
    topic/subtopic context from PostgreSQL with limit/offset pagination.

    This endpoint is read-only and does not rerun the model, rebuild
    features, or modify memory.
    """

    student_id = _require_non_blank(student_id, "student_id")
    topic = _require_non_blank(topic, "topic")
    subtopic = _normalize_optional_text(subtopic)

    try:
        history = get_learning_state_history(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
        )
        return history[offset : offset + limit]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve learning-state history.",
        ) from exc


@router.post(
    "/question-context",
    response_model=QuestionContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Topic & Retrieve Question Context",
    operation_id="extract_question_context",
    responses={
        200: {"description": "Resolved canonical skill and personalized memory context."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to resolve question context."},
    },
)
def get_question_context_endpoint(
    request: QuestionContextRequest,
) -> QuestionContextResponse:
    """
    Extract canonical skill from student free-text question, persist extraction audit log,
    and return current personalized memory context for tutoring adaptation.
    """
    try:
        service = get_question_context_service()
        return service.get_question_context(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve question context.",
        ) from exc


@router.get(
    "/{student_id}/context",
    response_model=StudentContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Unified Full Student Memory Context",
    operation_id="get_full_student_context",
    responses={
        200: {"description": "Unified full student memory context."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve student context."},
    },
)
def get_student_context_endpoint(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope for STM."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions / repairs."),
) -> StudentContextResponse:
    """
    Retrieve unified full student memory context combining STM, LTM, Concept Memory,
    Learning State, Misconceptions, Recent Interactions, and Repair Outcomes.
    """
    student_id = _require_non_blank(student_id, "student_id")
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_student_context_service()
        return service.get_student_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve student context.",
        ) from exc


@router.get(
    "/{student_id}/tutor-context",
    response_model=TutorContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Tutor Agent Personalization Context",
    operation_id="get_tutor_personalization_context",
    responses={
        200: {"description": "Focused tutor personalization context."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve tutor context."},
    },
)
def get_tutor_context_endpoint(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope for STM."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions / repairs."),
) -> TutorContextResponse:
    """
    Retrieve focused tutor personalization context:
    - Current learning state and evidence strength
    - Recent accuracy, correct count, incorrect count
    - Structured behavioral evidence (attempts, hints, response times)
    - Active misconceptions and past repair outcomes
    """
    student_id = _require_non_blank(student_id, "student_id")
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_tutor_context_service()
        return service.get_tutor_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tutor context.",
        ) from exc


@router.get(
    "/{student_id}/planner-context",
    response_model=PlannerContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Planner Agent Memory Context",
    operation_id="get_planner_memory_context",
    responses={
        200: {"description": "Focused planner memory context."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve planner context."},
    },
)
def get_planner_context_endpoint(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions."),
) -> PlannerContextResponse:
    """
    Retrieve focused planner context containing session performance, concept mastery metrics,
    longitudinal learning evidence, behavioral patterns, and active misconceptions.
    """
    student_id = _require_non_blank(student_id, "student_id")
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_planner_context_service()
        return service.get_planner_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve planner context.",
        ) from exc


@router.get(
    "/{student_id}/fapr-context",
    response_model=FAPRContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve FAPR-LB Pedagogical Repair Context",
    operation_id="get_fapr_repair_context",
    responses={
        200: {"description": "Focused FAPR-LB pedagogical repair context."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve FAPR context."},
    },
)
def get_fapr_context_endpoint(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions / repairs."),
) -> FAPRContextResponse:
    """
    Retrieve focused FAPR-LB context containing recent error counts, behavioral latency patterns,
    active misconceptions, previous repair outcomes, and latest student utterance.
    """
    student_id = _require_non_blank(student_id, "student_id")
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_fapr_context_service()
        return service.get_fapr_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve FAPR context.",
        ) from exc


@router.post(
    "/repair-outcome",
    response_model=RepairOutcomeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record FAPR Pedagogical Repair Outcome",
    operation_id="record_repair_outcome",
    responses={
        201: {"description": "Pedagogical repair outcome logged successfully."},
        400: {"model": ErrorResponse, "description": "Foreign key constraint or validation failure."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        404: {"model": ErrorResponse, "description": "Referenced student, session, or skill not found."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to record repair outcome."},
    },
)
def record_repair_outcome_endpoint(
    request: RepairOutcomeCreateRequest,
) -> RepairOutcomeResponse:
    """
    Store an append-only pedagogical repair outcome event from FAPR-LB:
    - Validates student, session, and skill existence & ownership
    - Optionally links to a specific interaction
    - Prevents duplicate writes on retry
    - Memory stores evidence objectively without judging pedagogical strategy
    """
    try:
        service = get_repair_outcome_service()
        return service.record_repair_outcome(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record repair outcome.",
        ) from exc


@router.get(
    "/{student_id}/support-preference",
    response_model=SupportPreferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Empirical Support Style Preference",
    operation_id="evaluate_support_preference",
    responses={
        200: {"description": "Evaluated empirical support preference evidence."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        404: {"model": ErrorResponse, "description": "Student or skill not found."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to evaluate support preference."},
    },
)
def get_support_preference_endpoint(
    student_id: str,
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
) -> SupportPreferenceResponse:
    """
    Evaluate empirical repair outcome history to estimate which pedagogical support style
    has proven most effective for this student (and optionally skill).
    Requires >= 3 observations to assert a preferred strategy.
    """
    student_id = _require_non_blank(student_id, "student_id")
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_support_preference_service()
        return service.get_support_preference(
            student_id=student_id,
            skill_id=skill_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate support preference.",
        ) from exc


@router.get(
    "/{student_id}/meta-signals",
    response_model=MetaSignalsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Meta-Agent Learning Evidence Signals",
    operation_id="get_meta_agent_signals",
    responses={
        200: {"description": "Chronological learning evidence signals generated."},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-Service-Key."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to generate meta signals."},
    },
)
def get_meta_signals_endpoint(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max evidence signals returned."),
) -> MetaSignalsResponse:
    """
    Generate transparent, chronological evidence signals for the Meta-Agent:
    - correct_answer: empirical success marker
    - incorrect_answer: empirical failure marker
    - confusion: transparent linguistic indicator from utterance
    - clarification_request: explicit student question or request for elaboration
    - repeated_misunderstanding: persistent misconception with >= 2 occurrences
    """
    student_id = _require_non_blank(student_id, "student_id")
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)

    try:
        service = get_meta_signal_service()
        return service.get_meta_signals(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate meta signals.",
        ) from exc







