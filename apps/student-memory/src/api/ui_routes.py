"""UI/Frontend demo routes for browser client without requiring backend service API key."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from src.api.auth_routes import (
    get_current_user_optional,
    get_current_user_required,
)
from src.api.auth import is_auth_enabled
from src.database.models.core import LearningSession, Student
from src.database.postgres_session import get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.api_common import ErrorResponse
from src.schemas.fapr_context import FAPRContextResponse
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    MemoryUpdateResponse,
)
from src.schemas.meta_signals import MetaSignalsResponse
from src.schemas.planner_context import PlannerContextResponse
from src.schemas.question_context import (
    QuestionContextRequest,
    QuestionContextResponse,
    UIQuestionContextRequest,
)
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)
from src.schemas.sessions import SessionItemResponse
from src.schemas.student_context import StudentContextResponse
from src.schemas.support_preference import SupportPreferenceResponse
from src.schemas.tutor_context import TutorContextResponse
from src.services.fapr_context_service import get_fapr_context_service
from src.services.memory_update_service import (
    MemoryUpdateServiceError,
    update_student_memory,
)
from src.services.meta_signal_service import get_meta_signal_service
from src.services.planner_context_service import get_planner_context_service
from src.services.question_context_service import get_question_context_service
from src.services.repair_outcome_service import get_repair_outcome_service
from src.services.student_context_service import get_student_context_service
from src.services.support_preference_service import get_support_preference_service
from src.services.tutor_context_service import get_tutor_context_service

router = APIRouter(
    prefix="/ui",
    tags=["ui"],
)


def get_ui_user_required(
    current_user: dict[str, Any] | None = Depends(get_current_user_optional),
) -> dict[str, Any]:
    """Require a browser identity; permit only explicit auth-disabled test mode."""
    if current_user is not None:
        return current_user
    if not is_auth_enabled():
        return {"role": "AUTH_DISABLED_TEST", "student_id": None}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please log in.",
    )


def _require_ui_student_access(
    requested_student_id: str | None,
    current_user: dict[str, Any],
) -> str:
    """Return the authenticated student ID or reject cross-student access."""
    if current_user.get("role") == "AUTH_DISABLED_TEST":
        if not requested_student_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Student identity is required in auth-disabled test mode.",
            )
        return requested_student_id

    authenticated_student_id = current_user.get("student_id")
    if not authenticated_student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated account is not linked to a student.",
        )
    if (
        requested_student_id is not None
        and requested_student_id.strip() != authenticated_student_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-student access is forbidden.",
        )
    return authenticated_student_id


def _require_non_blank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} must not be blank.",
        )
    return value


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@router.post(
    "/sessions",
    response_model=SessionItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new learning session for the logged-in student",
)
def create_session(
    current_user: dict[str, Any] = Depends(get_current_user_required),
) -> SessionItemResponse:
    """Automatically create an active learning session for the authenticated student."""
    student_id_str = current_user.get("student_id")
    if not student_id_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is not linked to a student identity.",
        )
    factory = get_session_factory()
    with UnitOfWork(factory) as uow:
        assert uow.session is not None
        student = uow.session.scalar(
            select(Student).where(Student.external_student_id == student_id_str)
        )
        if not student:
            try:
                stud_uuid = uuid.UUID(student_id_str)
                student = uow.session.get(Student, stud_uuid)
            except ValueError:
                pass
        if not student:
            raise HTTPException(status_code=404, detail="Student record not found.")

        sess_uuid = uuid.uuid4()
        ext_sess_id = f"chat_{uuid.uuid4().hex[:8]}"
        new_sess = LearningSession(
            session_id=sess_uuid,
            student_id=student.student_id,
            external_session_id=ext_sess_id,
            status="ACTIVE",
        )
        uow.session.add(new_sess)
        uow.session.commit()
        return SessionItemResponse(
            session_id=str(new_sess.session_id),
            external_session_id=new_sess.external_session_id,
            student_id=str(student.external_student_id),
            status=new_sess.status,
            started_at=(
                new_sess.started_at.isoformat()
                if hasattr(new_sess.started_at, "isoformat")
                else str(new_sess.started_at)
            ),
        )


@router.get(
    "/sessions",
    response_model=list[SessionItemResponse],
    status_code=status.HTTP_200_OK,
    summary="List recent learning sessions for the logged-in student",
)
def list_sessions(
    current_user: dict[str, Any] = Depends(get_current_user_required),
    limit: int = Query(default=30, ge=1, le=100),
) -> list[SessionItemResponse]:
    """List historical learning sessions scoped strictly to the authenticated student, excluding background assessment logs."""
    student_id_str = current_user.get("student_id")
    if not student_id_str:
        return []
    factory = get_session_factory()
    with UnitOfWork(factory) as uow:
        assert uow.session is not None
        student = uow.session.scalar(
            select(Student).where(Student.external_student_id == student_id_str)
        )
        if not student:
            try:
                stud_uuid = uuid.UUID(student_id_str)
                student = uow.session.get(Student, stud_uuid)
            except ValueError:
                pass
        if not student:
            return []

        sessions = uow.session.scalars(
            select(LearningSession)
            .where(
                LearningSession.student_id == student.student_id,
                ~LearningSession.external_session_id.startswith("assessment_"),
            )
            .order_by(
                func.coalesce(LearningSession.updated_at, LearningSession.started_at).desc(),
                LearningSession.started_at.desc(),
            )
            .limit(limit)
        ).all()

        return [
            SessionItemResponse(
                session_id=str(s.session_id),
                external_session_id=s.external_session_id,
                student_id=str(student.external_student_id),
                status=s.status,
                started_at=(
                    s.started_at.isoformat()
                    if hasattr(s.started_at, "isoformat")
                    else str(s.started_at)
                ),
            )
            for s in sessions
        ]


class SessionUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionItemResponse,
    status_code=status.HTTP_200_OK,
    summary="Rename / update learning session title",
)
def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user_required),
) -> SessionItemResponse:
    """Update a learning session's display title / topic name."""
    student_id_str = current_user.get("student_id")
    factory = get_session_factory()
    with UnitOfWork(factory) as uow:
        assert uow.session is not None
        sess: LearningSession | None = None
        try:
            s_uuid = uuid.UUID(session_id.strip())
            sess = uow.session.get(LearningSession, s_uuid)
        except ValueError:
            sess = uow.session.scalar(
                select(LearningSession).where(LearningSession.external_session_id == session_id.strip())
            )

        if not sess:
            raise HTTPException(status_code=404, detail="Learning session not found.")

        student = uow.session.scalar(
            select(Student).where(
                Student.external_student_id == student_id_str,
            )
        )
        if student is None or sess.student_id != student.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Learning session belongs to another student.",
            )

        # Update external_session_id with the new clean topic title and touch timestamp
        new_title = payload.title.strip()[:250]
        sess.external_session_id = new_title
        sess.updated_at = datetime.now(timezone.utc)
        uow.session.commit()

        return SessionItemResponse(
            session_id=str(sess.session_id),
            external_session_id=sess.external_session_id,
            student_id=student_id_str or "",
            status=sess.status,
            started_at=(
                sess.started_at.isoformat()
                if hasattr(sess.started_at, "isoformat")
                else str(sess.started_at)
            ),
        )


@router.post(
    "/question-context",
    response_model=QuestionContextResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Question Context Probe",
    description="Frontend/UI safe endpoint for interactive question analysis and memory retrieval.",
    operation_id="ui_post_question_context",
    responses={
        200: {"description": "Extracted topic and student personalized memory context."},
        403: {"model": ErrorResponse, "description": "Session does not belong to logged-in student."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to resolve question context."},
    },
)
def ui_question_context_endpoint(
    request: UIQuestionContextRequest,
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> QuestionContextResponse:
    """
    Extract topic and retrieve memory context for frontend demo without requiring X-Service-Key.
    Automatically resolves student_id from active login session if available.
    """
    target_student_id = _require_ui_student_access(
        request.student_id,
        current_user,
    )

    # Check session ownership if student is logged in and session exists
    if current_user.get("role") != "AUTH_DISABLED_TEST" and request.session_id:
        factory = get_session_factory()
        with UnitOfWork(factory) as uow:
            assert uow.session is not None
            sess_raw = request.session_id.strip()
            sess: LearningSession | None = None
            try:
                s_uuid = uuid.UUID(sess_raw)
                sess = uow.session.scalar(
                    select(LearningSession).where(
                        (LearningSession.session_id == s_uuid)
                        | (LearningSession.external_session_id == sess_raw)
                    )
                )
            except ValueError:
                sess = uow.session.scalar(
                    select(LearningSession).where(LearningSession.external_session_id == sess_raw)
                )

            if sess is not None:
                stud = uow.session.scalar(
                    select(Student).where(Student.external_student_id == target_student_id)
                )
                if not stud:
                    try:
                        stud = uow.session.get(Student, uuid.UUID(target_student_id))
                    except ValueError:
                        pass
                if stud and sess.student_id != stud.student_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Session does not belong to the logged-in student.",
                    )
                # Mark session as recently active
                sess.updated_at = datetime.now(timezone.utc)
                uow.session.commit()

    full_req = QuestionContextRequest(
        student_id=target_student_id,
        session_id=request.session_id,
        question=request.question,
    )
    service = get_question_context_service()
    try:
        return service.get_question_context(full_req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve question context.",
        ) from exc


@router.post(
    "/memory-update",
    response_model=MemoryUpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Assessment Memory Update",
    description="Frontend/UI safe endpoint for submitting completed assessments and updating cognitive memory.",
    operation_id="ui_post_memory_update",
    responses={
        200: {"description": "Assessment processed and dynamic learning state updated."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Internal memory update error."},
    },
)
def ui_memory_update_endpoint(
    request: AssessmentMemoryUpdateRequest,
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> MemoryUpdateResponse:
    """
    Process one completed assessment and update student memory in PostgreSQL for frontend demo.
    Enforces student identity when student is logged in.
    """
    _require_ui_student_access(request.student_id, current_user)
    try:
        return update_student_memory(request=request)
    except HTTPException:
        raise
    except MemoryUpdateServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory update failed.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory update failed.",
        ) from exc


@router.post(
    "/repair-outcome",
    response_model=RepairOutcomeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="UI Record FAPR Pedagogical Repair Outcome",
    description="Frontend/UI safe endpoint for logging pedagogical repair outcomes without requiring service key.",
    operation_id="ui_post_repair_outcome",
    responses={
        201: {"description": "Pedagogical repair outcome logged successfully."},
        400: {"model": ErrorResponse, "description": "Foreign key constraint or validation failure."},
        404: {"model": ErrorResponse, "description": "Referenced student, session, or skill not found."},
        413: {"model": ErrorResponse, "description": "Request body exceeds maximum size limit."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to record repair outcome."},
    },
)
def ui_repair_outcome_endpoint(
    request: RepairOutcomeCreateRequest,
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> RepairOutcomeResponse:
    """
    Store an append-only pedagogical repair outcome event from frontend demo without X-Service-Key.
    """
    _require_ui_student_access(request.student_id, current_user)
    service = get_repair_outcome_service()
    try:
        return service.record_repair_outcome(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record repair outcome.",
        ) from exc


@router.get(
    "/{student_id}/context",
    response_model=StudentContextResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Unified Full Student Memory Context",
    operation_id="ui_get_full_student_context",
    responses={
        200: {"description": "Unified full student memory context."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve student context."},
    },
)
def ui_get_student_context(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions / repairs."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> StudentContextResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)
    service = get_student_context_service()
    try:
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
            detail="Failed to retrieve full student context.",
        ) from exc


@router.get(
    "/{student_id}/tutor-context",
    response_model=TutorContextResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Tutor Context",
    operation_id="ui_get_tutor_context",
    responses={
        200: {"description": "Focused Tutor teaching context."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve tutor context."},
    },
)
def ui_get_tutor_context(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=10, ge=1, le=50, description="Max recent interactions / repairs."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> TutorContextResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)
    service = get_tutor_context_service()
    try:
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
    summary="UI Planner Context",
    operation_id="ui_get_planner_context",
    responses={
        200: {"description": "Focused Planner curriculum context."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve planner context."},
    },
)
def ui_get_planner_context(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=20, ge=1, le=100, description="Max recent interactions."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> PlannerContextResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)
    service = get_planner_context_service()
    try:
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
    summary="UI FAPR-LB Context",
    operation_id="ui_get_fapr_context",
    responses={
        200: {"description": "Focused FAPR-LB diagnostic repair context."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to retrieve FAPR context."},
    },
)
def ui_get_fapr_context(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    limit: int = Query(default=10, ge=1, le=50, description="Max recent interactions / repairs."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> FAPRContextResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)
    service = get_fapr_context_service()
    try:
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


@router.get(
    "/{student_id}/support-preference",
    response_model=SupportPreferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Support Preference Analysis",
    operation_id="ui_get_support_preference",
    responses={
        200: {"description": "Ranked support style preference."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to calculate support preference."},
    },
)
def ui_get_support_preference(
    student_id: str,
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> SupportPreferenceResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    skill_id = _normalize_optional_text(skill_id)
    service = get_support_preference_service()
    try:
        return service.get_support_preference(
            student_id=student_id,
            skill_id=skill_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate support preference.",
        ) from exc


@router.get(
    "/{student_id}/meta-signals",
    response_model=MetaSignalsResponse,
    status_code=status.HTTP_200_OK,
    summary="UI Meta-Agent Cognitive Signals",
    operation_id="ui_get_meta_signals",
    responses={
        200: {"description": "Cognitive and behavioral reliability signals."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Failed to compute meta signals."},
    },
)
def ui_get_meta_signals(
    student_id: str,
    session_id: str | None = Query(default=None, description="Optional session scope."),
    skill_id: str | None = Query(default=None, description="Optional canonical skill scope."),
    current_user: dict[str, Any] = Depends(get_ui_user_required),
) -> MetaSignalsResponse:
    student_id = _require_non_blank(student_id, "student_id")
    _require_ui_student_access(student_id, current_user)
    session_id = _normalize_optional_text(session_id)
    skill_id = _normalize_optional_text(skill_id)
    service = get_meta_signal_service()
    try:
        return service.get_meta_signals(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute meta signals.",
        ) from exc


@router.get(
    "/skills",
    status_code=status.HTTP_200_OK,
    summary="List all canonical math skills and topics",
    operation_id="ui_list_skills",
)
def ui_list_skills() -> list[dict[str, Any]]:
    """Return all available canonical math skills for dropdown and search selectors."""
    from src.ontology.ontology_lookup_service import get_ontology_lookup_service
    lookup = get_ontology_lookup_service()
    skills = lookup.list_all_skills()
    return [
        {
            "skill_id": str(s.skill_id),
            "skill_code": s.skill_code,
            "display_name": s.display_name,
            "canonical_name": s.canonical_name,
            "category": s.category,
            "description": s.description,
        }
        for s in skills
    ]
