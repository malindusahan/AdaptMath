import logging
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.general_chat.general_chat_agent import get_general_chat_agent
from app.core.input_validation import (
    ANSWER_MAX_LENGTH,
    MAX_CONTEXT_ITEMS,
    MAX_HISTORY_ITEMS,
    QUESTION_MAX_LENGTH,
    STUDENT_ID_MAX_LENGTH,
    SUBTOPIC_MAX_LENGTH,
    TOPIC_MAX_LENGTH,
    clean_optional_text,
    clean_required_text,
    clean_text_list,
)
from app.core.model_errors import model_service_http_exception
from app.core.persistence import (
    list_checkpoint_thread_ids,
    set_thread_status,
    tutor_checkpointer,
    upsert_thread_metadata,
)
from app.graph.state import TutorState
from app.graph.workflow import adaptive_coordinator, tutor_graph
from app.clients.memory.memory_client import MemoryAuthenticatedUser, get_memory_client
from app.core.user_auth import require_authenticated_student
from app.integrations.adaptive_component_coordinator import (
    AdaptiveConcurrencyError,
    AdaptiveIntegrationError,
    AdaptiveRestartRequiredError,
)
from app.integrations.memory_context_adapter import adapt_memory_context
from app.schemas.evaluator import StudentAnswer
from app.schemas.memory import LearnerHistoryItem
from app.schemas.pedagogical_move import PedagogicalMove
from app.services.practice_problem_service import (
    PracticeProblemGenerationError,
    get_practice_problem_generator,
)


router = APIRouter(prefix="/tutor", tags=["tutor"])
logger = logging.getLogger(__name__)
_START_GUARD = Lock()

# Memory's topic classifier has a wider ontology than the trained 95-skill BKT
# model. Keep deliberate semantic fallbacks explicit and auditable rather than
# silently accepting an untrained label. A discount problem applies a percent
# to an original quantity, which is the supported ``Percent Of`` BKT skill.
MEMORY_TO_BKT_SKILL_COMPATIBILITY = {
    "Percent Discount": "Percent Of",
}


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class TutorStartRequest(BaseModel):
    """
    Temporary integration input until teammate APIs are connected.

    Final ownership:
    - Malindu supplies student_id and age;
    - Hamooth supplies topic/learner context.

    Pedagogical moves are NOT accepted from the learner-facing frontend. The
    workflow pauses for Omash's external move selector before the first teacher
    turn and before every later teacher turn.
    """

    model_config = ConfigDict(extra="forbid")

    # Retained as an optional legacy field. HTTP callers are authenticated and
    # the server-derived Memory identity is authoritative.
    student_id: str | None = Field(default=None, max_length=STUDENT_ID_MAX_LENGTH)
    age: int = Field(..., ge=8, le=18)
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LENGTH)
    topic: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)
    subtopic: str | None = Field(default=None, max_length=SUBTOPIC_MAX_LENGTH)
    target_skill: str | None = Field(default=None, max_length=TOPIC_MAX_LENGTH)
    relevant_history: list[LearnerHistoryItem] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
    )
    previous_errors: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )
    previous_strategies: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )

    @field_validator("question", "topic")
    @classmethod
    def normalize_required_fields(cls, value: str, info) -> str:
        return clean_required_text(
            value,
            label=info.field_name.replace("_", " ").title(),
        )

    @field_validator("student_id")
    @classmethod
    def normalize_legacy_student_id(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Student ID")

    @field_validator("subtopic")
    @classmethod
    def normalize_subtopic(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Subtopic")

    @field_validator("target_skill")
    @classmethod
    def normalize_target_skill(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Target Skill")

    @field_validator("previous_errors", "previous_strategies")
    @classmethod
    def normalize_context_lists(cls, values: list[str], info) -> list[str]:
        return clean_text_list(
            values,
            label=info.field_name.replace("_", " ").title(),
        )


class TutorPracticeStartRequest(BaseModel):
    """The profile supplies only the canonical skill selected for practice."""

    model_config = ConfigDict(extra="forbid")
    target_skill: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)

    @field_validator("target_skill")
    @classmethod
    def normalize_target_skill(cls, value: str) -> str:
        return clean_required_text(value, label="Target Skill")


class TutorStudentResponseRequest(BaseModel):
    """One student reply to the latest interactive Tutor turn."""

    model_config = ConfigDict(extra="forbid")
    response: str = Field(..., min_length=1, max_length=ANSWER_MAX_LENGTH)

    @field_validator("response")
    @classmethod
    def normalize_response(cls, value: str) -> str:
        return clean_required_text(value, label="Student response")


class TutorAnswerRequest(BaseModel):
    """Exactly three answers to the Evaluator-generated assessment."""

    model_config = ConfigDict(extra="forbid")
    answers: list[StudentAnswer] = Field(..., min_length=3, max_length=3)

    @model_validator(mode="after")
    def reject_duplicate_question_ids(self):
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Each assessment question may be answered only once.")
        return self


class TutorPedagogicalMoveRequest(BaseModel):
    """Next move selected externally by Omash's move-selector component."""

    model_config = ConfigDict(extra="forbid")
    pedagogical_move: PedagogicalMove


class PublicAssessmentQuestion(BaseModel):
    question_id: str
    question: str


class TutorSessionResponse(BaseModel):
    thread_id: str
    status: Literal[
        "student_response_required",
        "pedagogical_move_required",
        "assessment_required",
        "recovery_required",
        "complete",
    ]

    tutor_response: str | None = None
    route: str | None = None
    route_reason: str | None = None
    complexity_score: float | None = None
    target_skill: str | None = None

    student_response_message: str | None = None
    assessment_message: str | None = None
    pedagogical_move_message: str | None = None
    allowed_pedagogical_moves: list[PedagogicalMove] = Field(default_factory=list)
    questions: list[PublicAssessmentQuestion] = Field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    reteach_round: int = 0
    turn_count: int = 0


class TutorPracticeStartResponse(BaseModel):
    """A validated problem together with the normal Tutor start response."""

    target_skill: str
    problem: str
    session: TutorSessionResponse


# ============================================================
# SAFE RESPONSE HELPERS
# ============================================================

def _extract_interrupt_payload(
    graph_result: dict[str, Any],
) -> dict[str, Any] | None:
    interrupts = graph_result.get("__interrupt__", [])
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else None


def _safe_answer_evidence(answer: dict[str, Any]) -> dict[str, Any]:
    safe_fields = (
        "question_id",
        "question",
        "student_answer",
        "is_correct",
        "feedback",
        "identified_error",
    )
    return {field: answer[field] for field in safe_fields if field in answer}


def _safe_evaluation(
    graph_result: dict[str, Any],
) -> dict[str, Any] | None:
    evaluation = graph_result.get("evaluation_result")
    if not isinstance(evaluation, dict):
        return None

    return {
        "correct_answers": [
            _safe_answer_evidence(answer)
            for answer in evaluation.get("correct_answers", [])
        ],
        "wrong_answers": [
            _safe_answer_evidence(answer)
            for answer in evaluation.get("wrong_answers", [])
        ],
        "identified_errors": evaluation.get("identified_errors", []),
        "needs_reteaching": evaluation.get("needs_reteaching", False),
        "overall_feedback": evaluation.get("overall_feedback"),
    }


def _base_response_fields(graph_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "tutor_response": graph_result.get("tutor_response"),
        "route": graph_result.get("route"),
        "route_reason": graph_result.get("route_reason"),
        "complexity_score": graph_result.get("complexity_score"),
        "target_skill": graph_result.get("target_skill"),
        "evaluation": _safe_evaluation(graph_result),
        "reteach_round": graph_result.get("reteach_round", 0),
        "turn_count": graph_result.get("turn_count", 0),
    }


def _build_public_response(
    thread_id: str,
    graph_result: dict[str, Any],
) -> TutorSessionResponse:
    payload = _extract_interrupt_payload(graph_result)
    base = _base_response_fields(graph_result)

    if payload:
        interrupt_type = payload.get("type")

        if interrupt_type == "student_response_required":
            return TutorSessionResponse(
                thread_id=thread_id,
                status="student_response_required",
                student_response_message=payload.get("message"),
                questions=[],
                **base,
            )

        if interrupt_type == "pedagogical_move_required":
            return TutorSessionResponse(
                thread_id=thread_id,
                status="pedagogical_move_required",
                pedagogical_move_message=payload.get("message"),
                allowed_pedagogical_moves=payload.get("allowed_moves", []),
                questions=[],
                **base,
            )

        if interrupt_type == "assessment_required":
            raw_questions = payload.get("questions", [])
            public_questions = [
                PublicAssessmentQuestion(
                    question_id=question["question_id"],
                    question=question["question"],
                )
                for question in raw_questions
            ]
            return TutorSessionResponse(
                thread_id=thread_id,
                status="assessment_required",
                assessment_message=payload.get("message"),
                questions=public_questions,
                **base,
            )

    set_thread_status(thread_id, "COMPLETED")
    return TutorSessionResponse(
        thread_id=thread_id,
        status="complete",
        questions=[],
        **base,
    )


# ============================================================
# PERSISTED SESSION HELPERS
# ============================================================

def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot_has_interrupt(snapshot: Any) -> bool:
    tasks = getattr(snapshot, "tasks", ()) or ()
    return any(bool(getattr(task, "interrupts", ())) for task in tasks)


def _get_persisted_snapshot(thread_id: str) -> Any:
    config = _thread_config(thread_id)
    checkpoint = tutor_checkpointer.get_tuple(config)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Tutoring session was not found.")
    return tutor_graph.get_state(config)


def _resolved_authenticated_user(
    value: object,
) -> MemoryAuthenticatedUser | None:
    """Return a DI-resolved user while preserving direct internal test calls."""
    return value if isinstance(value, MemoryAuthenticatedUser) else None


def _authoritative_student_id(
    requested_student_id: str | None,
    authenticated_user: object,
) -> str:
    """Use authenticated identity for HTTP calls and reject impersonation."""
    user = _resolved_authenticated_user(authenticated_user)
    if user is not None:
        if (
            requested_student_id is not None
            and requested_student_id != user.student_id
        ):
            raise HTTPException(
                status_code=403,
                detail="The supplied student_id does not match the authenticated learner.",
            )
        return user.student_id

    # Route dependencies always provide a user. This compatibility branch is
    # solely for existing in-process unit tests and trusted internal callers.
    if requested_student_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return requested_student_id


def _require_thread_owner(snapshot: Any, authenticated_user: object) -> None:
    """Reject access when a persisted Tutor thread belongs to another learner."""
    user = _resolved_authenticated_user(authenticated_user)
    if user is None:
        return
    values = dict(getattr(snapshot, "values", {}) or {})
    if values.get("student_id") != user.student_id:
        raise HTTPException(
            status_code=403,
            detail="This tutoring session belongs to another learner.",
        )


def _find_active_student_session(
    student_id: str,
) -> TutorSessionResponse | None:
    """Return the newest durable, unfinished adaptive lesson for a student."""

    for thread_id in list_checkpoint_thread_ids():
        snapshot = tutor_graph.get_state(_thread_config(thread_id))
        values = dict(getattr(snapshot, "values", {}) or {})
        if values.get("student_id") != student_id:
            continue
        if values.get("adaptive_lifecycle_status") != "active":
            continue

        if _persisted_session_phase(snapshot) == "complete":
            continue
        return _build_snapshot_response(thread_id, snapshot)

    return None


def _persisted_session_phase(
    snapshot: Any,
) -> Literal[
    "student_response_required",
    "pedagogical_move_required",
    "assessment_required",
    "recovery_required",
    "complete",
]:
    next_nodes = getattr(snapshot, "next", ()) or ()

    if _snapshot_has_interrupt(snapshot):
        if "await_student_response" in next_nodes:
            return "student_response_required"
        if "await_pedagogical_move" in next_nodes:
            return "pedagogical_move_required"
        if "await_answers" in next_nodes:
            return "assessment_required"

    if not next_nodes:
        return "complete"
    return "recovery_required"


def _build_snapshot_response(
    thread_id: str,
    snapshot: Any,
) -> TutorSessionResponse:
    values = dict(getattr(snapshot, "values", {}) or {})
    phase = _persisted_session_phase(snapshot)
    base = _base_response_fields(values)

    if phase == "student_response_required":
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            student_response_message="Reply to your tutor to continue the lesson.",
            questions=[],
            **base,
        )

    if phase == "pedagogical_move_required":
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            pedagogical_move_message=(
                "An externally selected pedagogical move is required for the "
                "next teacher turn."
            ),
            allowed_pedagogical_moves=["telling", "focus", "generic", "probing"],
            questions=[],
            **base,
        )

    if phase == "assessment_required":
        raw_questions = values.get("assessment_questions", [])
        public_questions = [
            PublicAssessmentQuestion(
                question_id=question["question_id"],
                question=question["question"],
            )
            for question in raw_questions
        ]
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            assessment_message="Please answer all three assessment questions.",
            questions=public_questions,
            **base,
        )

    if phase == "recovery_required":
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            questions=[],
            **base,
        )

    return TutorSessionResponse(
        thread_id=thread_id,
        status="complete",
        questions=[],
        **base,
    )


def _adaptive_http_exception(exc: AdaptiveIntegrationError) -> HTTPException:
    logger.warning(
        "adaptive_integration_failure exception_type=%s detail=%s",
        type(exc).__name__,
        exc,
    )
    if isinstance(exc, AdaptiveConcurrencyError):
        detail = (
            "Another adaptive tutoring session is currently active. "
            "Please retry shortly."
        )
    elif isinstance(exc, AdaptiveRestartRequiredError):
        detail = (
            "This tutoring session was interrupted by a server restart and "
            "cannot be safely continued. Start a new session."
        )
    else:
        detail = "Tutor and adaptive attempt state are inconsistent."
    return HTTPException(status_code=409, detail=detail)


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/active", response_model=TutorSessionResponse | None)
def get_active_tutor_session(
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse | None:
    user = _resolved_authenticated_user(authenticated_user)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return _find_active_student_session(user.student_id)


@router.get("/{thread_id}", response_model=TutorSessionResponse)
def get_tutor_session(
    thread_id: str,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    return _build_snapshot_response(
        thread_id,
        snapshot,
    )


@router.get(
    "/student/{student_id}/active",
    response_model=TutorSessionResponse | None,
)
def get_active_tutor_session_for_student(
    student_id: str,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse | None:
    normalized_student_id = clean_required_text(
        student_id,
        label="Student ID",
    )
    if len(normalized_student_id) > STUDENT_ID_MAX_LENGTH:
        raise HTTPException(status_code=422, detail="Student ID is too long.")
    authoritative_student_id = _authoritative_student_id(
        normalized_student_id,
        authenticated_user,
    )
    return _find_active_student_session(authoritative_student_id)


@router.post("/{thread_id}/abandon", response_model=TutorSessionResponse)
def abandon_tutor_session(
    thread_id: str,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    """End an unfinished lesson while retaining its durable research record."""

    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    values = dict(getattr(snapshot, "values", {}) or {})
    lifecycle_status = values.get("adaptive_lifecycle_status")

    if lifecycle_status == "active":
        try:
            aborted = adaptive_coordinator.abort_attempt(values)
        except AdaptiveRestartRequiredError:
            # A restarted non-learning deployment may be unable to rebuild the
            # process-local controller. No live lease exists in that case, so
            # close the durable lesson without fabricating a policy update.
            logger.warning(
                "abandoning_unrestorable_attempt thread_id=%s",
                thread_id,
            )
            aborted = {"adaptive_lifecycle_status": "aborted"}
        except AdaptiveIntegrationError as exc:
            raise _adaptive_http_exception(exc) from exc
        tutor_graph.update_state(
            _thread_config(thread_id),
            aborted,
        )
        values.update(aborted)
    elif lifecycle_status not in {"aborted", "completed"}:
        aborted = {"adaptive_lifecycle_status": "aborted"}
        tutor_graph.update_state(
            _thread_config(thread_id),
            aborted,
        )
        values.update(aborted)

    set_thread_status(thread_id, "ABORTED")
    return TutorSessionResponse(
        thread_id=thread_id,
        status="complete",
        questions=[],
        **_base_response_fields(values),
    )


@router.post("/{thread_id}/resume", response_model=TutorSessionResponse)
def resume_tutor_session(
    thread_id: str,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    phase = _persisted_session_phase(snapshot)

    if phase != "recovery_required":
        return _build_snapshot_response(thread_id, snapshot)

    config = _thread_config(thread_id)
    try:
        graph_result = tutor_graph.invoke(None, config=config)
    except AdaptiveIntegrationError as exc:
        raise _adaptive_http_exception(exc) from exc
    except Exception as exc:
        raise model_service_http_exception(exc) from exc

    return _build_public_response(thread_id, graph_result)


def _start_tutor_session(
    request: TutorStartRequest,
    authenticated_user: MemoryAuthenticatedUser,
) -> TutorSessionResponse:
    """
    Start one Tutor thread and its first assessment-cycle adaptive attempt.

    An explicitly supplied ``target_skill`` remains authoritative. When it is
    omitted, Memory's trained topic classifier identifies a candidate from the
    question and Repository B must validate it before the attempt can start.
    Topic and subtopic are never silently reinterpreted as BKT identifiers.
    Classifier abstentions receive one Gemini general-chat response and exit
    before any adaptive lesson, BKT, or move-selector state is created.
    """

    student_id = _authoritative_student_id(
        request.student_id,
        authenticated_user,
    )
    target_skill: str | None = None
    if request.target_skill is not None:
        try:
            target_skill = adaptive_coordinator.validate_target_skill(
                request.target_skill
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = _find_active_student_session(student_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "An adaptive tutoring session is currently active for this student. "
                "Recover it before starting another one."
            ),
        )

    if not _START_GUARD.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail=(
                "Another tutoring session is currently being prepared. "
                "Please wait for it to finish."
            ),
        )
    try:
        # Close the small race between the first durable-session lookup and
        # acquiring the non-blocking preparation guard.
        existing = _find_active_student_session(student_id)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An adaptive tutoring session is currently active for this student. "
                    "Recover it before starting another one."
                ),
            )

        memory_client = get_memory_client()
        if target_skill is None:
            classification = memory_client.classify_question(
                question=request.question,
            )
            if classification is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Automatic BKT skill identification is temporarily "
                        "unavailable. Retry, or provide an explicit target_skill."
                    ),
                )
            if not classification.is_math or not classification.topic:
                try:
                    gateway_decision = get_general_chat_agent().route(
                        message=request.question,
                    )
                except Exception as exc:
                    raise model_service_http_exception(exc) from exc

                if gateway_decision.kind == "general_chat":
                    response_id = f"general-{uuid4()}"
                    logger.info(
                        "gemini_general_chat_response response_id=%s "
                        "classifier_confidence=%.4f classifier_model=%s "
                        "gateway_confidence=%.4f",
                        response_id,
                        classification.confidence,
                        classification.model_version,
                        gateway_decision.confidence,
                    )
                    return TutorSessionResponse(
                        thread_id=response_id,
                        status="complete",
                        tutor_response=gateway_decision.response,
                        route="gemini_general_chat",
                        route_reason=(
                            "Memory abstained; Gemini identified general "
                            "conversation outside the adaptive lesson lifecycle."
                        ),
                        target_skill=None,
                        questions=[],
                        turn_count=1,
                    )

                candidate = gateway_decision.target_skill
                try:
                    target_skill = adaptive_coordinator.validate_target_skill(
                        candidate
                    )
                except (KeyError, ValueError) as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Gemini identified a math question but did not "
                            "return a supported BKT target skill."
                        ),
                    ) from exc
                logger.info(
                    "gemini_target_skill_selected bkt_skill=%s "
                    "classifier_confidence=%.4f classifier_model=%s "
                    "gateway_confidence=%.4f",
                    target_skill,
                    classification.confidence,
                    classification.model_version,
                    gateway_decision.confidence,
                )
            else:
                memory_skill = classification.topic.strip()
                candidate = MEMORY_TO_BKT_SKILL_COMPATIBILITY.get(
                    memory_skill,
                    memory_skill,
                )
                try:
                    target_skill = adaptive_coordinator.validate_target_skill(
                        candidate
                    )
                except (KeyError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Memory identified {candidate!r}, but that label is "
                            "not in the trained BKT vocabulary. Provide an explicit "
                            "supported target_skill."
                        ),
                    ) from exc
                logger.info(
                    "automatic_target_skill_selected memory_skill=%s bkt_skill=%s "
                    "confidence=%.4f model=%s",
                    memory_skill,
                    target_skill,
                    classification.confidence,
                    classification.model_version,
                )

        assert target_skill is not None

        thread_id = str(uuid4())
        config = _thread_config(thread_id)
        adaptive_attempt_index = 1
        attempt_id = adaptive_coordinator.derive_attempt_id(
            thread_id,
            adaptive_attempt_index,
        )

        retrieved_memory = adapt_memory_context(
            memory_client.retrieve_tutor_context(
                student_id=student_id,
                target_skill=target_skill,
            ),
            target_skill=target_skill,
        )

        initial_state: TutorState = {
            "student_id": student_id,
            "thread_id": thread_id,
            "attempt_id": attempt_id,
            "adaptive_attempt_index": adaptive_attempt_index,
            "target_skill": target_skill,
            "adaptive_lifecycle_status": "not_started",
            "age": request.age,
            "question": request.question,
            "topic": request.topic,
            "subtopic": request.subtopic,
            "pedagogical_move": None,
            "relevant_history": retrieved_memory.relevant_history,
            "previous_errors": retrieved_memory.previous_errors,
            "previous_strategies": retrieved_memory.previous_strategies,
            "conversation_history": [],
            "teaching_phase": "initial",
            "teaching_status": "continue_teaching",
            "turn_count": 0,
            "reteach_round": 0,
        }

        try:
            graph_result = tutor_graph.invoke(initial_state, config=config)
        except AdaptiveIntegrationError as exc:
            if not adaptive_coordinator.has_active_attempt(thread_id):
                tutor_checkpointer.delete_thread(thread_id)
            raise _adaptive_http_exception(exc) from exc
        except Exception as exc:
            # Once frozen v3 has started, deleting the persisted thread would
            # orphan an active policy attempt. Preserve it for explicit recovery.
            if not adaptive_coordinator.has_active_attempt(thread_id):
                tutor_checkpointer.delete_thread(thread_id)
            raise model_service_http_exception(exc) from exc

        upsert_thread_metadata(
            thread_id=thread_id,
            student_id=student_id,
            target_skill=target_skill,
        )
        return _build_public_response(thread_id, graph_result)
    finally:
        _START_GUARD.release()


@router.post("/start", response_model=TutorSessionResponse)
def start_tutor_session(
    request: TutorStartRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    """Start a lesson from the learner's manually supplied question."""
    return _start_tutor_session(request, authenticated_user)


@router.post("/start-practice", response_model=TutorPracticeStartResponse)
def start_recommended_practice(
    request: TutorPracticeStartRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorPracticeStartResponse:
    """Prepare a verified problem for one skill, then use the normal start path."""
    try:
        target_skill = adaptive_coordinator.validate_target_skill(
            request.target_skill
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _find_active_student_session(authenticated_user.student_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "An adaptive tutoring session is currently active for this student. "
                "Recover it before starting another one."
            ),
        )

    # Current registration captures an 8-18 age. The fallback keeps migrated
    # prototype accounts usable without letting the browser assert identity.
    student_age = authenticated_user.age or 15
    if not 8 <= student_age <= 18:
        student_age = 15

    try:
        generated = get_practice_problem_generator().generate(
            target_skill=target_skill,
            student_age=student_age,
            classifier=get_memory_client(),
        )
    except PracticeProblemGenerationError as exc:
        raise HTTPException(
            status_code=503,
            detail="We couldn't prepare a practice problem right now.",
        ) from exc

    session = _start_tutor_session(
        TutorStartRequest(
            age=student_age,
            question=generated.question,
            topic=target_skill,
            subtopic=target_skill,
            target_skill=target_skill,
            relevant_history=[],
            previous_errors=[],
            previous_strategies=[],
        ),
        authenticated_user,
    )
    return TutorPracticeStartResponse(
        target_skill=target_skill,
        problem=generated.question,
        session=session,
    )


@router.post("/{thread_id}/turn", response_model=TutorSessionResponse)
def submit_student_response(
    thread_id: str,
    request: TutorStudentResponseRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    """
    Submit one student's conversational reply to the latest teacher turn.

    The graph then uses the GenAI progress judge. If teaching should continue,
    it pauses for a fresh Omash-selected pedagogical move. If the dialogue is
    ready, the Evaluator generates the three-question assessment.
    """

    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    phase = _persisted_session_phase(snapshot)

    if phase == "complete":
        raise HTTPException(status_code=409, detail="This tutoring session has completed.")
    if phase != "student_response_required":
        raise HTTPException(
            status_code=409,
            detail="This tutoring session is not awaiting a student dialogue response.",
        )

    try:
        graph_result = tutor_graph.invoke(
            Command(resume={"response": request.response}),
            config=_thread_config(thread_id),
        )
    except AdaptiveIntegrationError as exc:
        raise _adaptive_http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise model_service_http_exception(exc) from exc

    return _build_public_response(thread_id, graph_result)


@router.post("/{thread_id}/move", response_model=TutorSessionResponse)
def submit_pedagogical_move(
    thread_id: str,
    request: TutorPedagogicalMoveRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    """Legacy/manual seam; integrated adaptive turns never consume this."""

    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    phase = _persisted_session_phase(snapshot)
    values = dict(getattr(snapshot, "values", {}) or {})

    if values.get("adaptive_lifecycle_status") == "active":
        raise HTTPException(
            status_code=409,
            detail=(
                "The active adaptive pipeline selects pedagogical moves; "
                "the legacy endpoint cannot mutate this thread."
            ),
        )

    if phase == "complete":
        raise HTTPException(status_code=409, detail="This tutoring session has completed.")
    if phase != "pedagogical_move_required":
        raise HTTPException(
            status_code=409,
            detail="This tutoring session is not awaiting a pedagogical move.",
        )

    try:
        graph_result = tutor_graph.invoke(
            Command(resume={"move": request.pedagogical_move}),
            config=_thread_config(thread_id),
        )
    except AdaptiveIntegrationError as exc:
        raise _adaptive_http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise model_service_http_exception(exc) from exc

    return _build_public_response(thread_id, graph_result)


@router.post("/{thread_id}/answers", response_model=TutorSessionResponse)
def submit_tutor_answers(
    thread_id: str,
    request: TutorAnswerRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorSessionResponse:
    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    phase = _persisted_session_phase(snapshot)

    if phase == "complete":
        raise HTTPException(status_code=409, detail="This tutoring session has completed.")
    if phase != "assessment_required":
        raise HTTPException(
            status_code=409,
            detail="This tutoring session is not awaiting assessment answers.",
        )

    resume_answers = [answer.model_dump() for answer in request.answers]

    try:
        graph_result = tutor_graph.invoke(
            Command(resume=resume_answers),
            config=_thread_config(thread_id),
        )
    except AdaptiveIntegrationError as exc:
        raise _adaptive_http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise model_service_http_exception(exc) from exc

    return _build_public_response(thread_id, graph_result)
