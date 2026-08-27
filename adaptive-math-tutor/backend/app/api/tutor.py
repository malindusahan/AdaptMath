from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from app.core.persistence import tutor_checkpointer
from app.graph.state import TutorState
from app.graph.workflow import adaptive_coordinator, tutor_graph
from app.clients.memory.memory_client import get_memory_client
from app.integrations.adaptive_component_coordinator import (
    AdaptiveConcurrencyError,
    AdaptiveIntegrationError,
    AdaptiveRestartRequiredError,
)
from app.integrations.memory_context_adapter import adapt_memory_context
from app.schemas.evaluator import StudentAnswer
from app.schemas.memory import LearnerHistoryItem
from app.schemas.pedagogical_move import PedagogicalMove


router = APIRouter(prefix="/tutor", tags=["tutor"])


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

    student_id: str = Field(..., min_length=1, max_length=STUDENT_ID_MAX_LENGTH)
    age: int = Field(..., ge=8, le=18)
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LENGTH)
    topic: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)
    subtopic: str | None = Field(default=None, max_length=SUBTOPIC_MAX_LENGTH)
    target_skill: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)
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

    @field_validator("student_id", "question", "topic", "target_skill")
    @classmethod
    def normalize_required_fields(cls, value: str, info) -> str:
        return clean_required_text(
            value,
            label=info.field_name.replace("_", " ").title(),
        )

    @field_validator("subtopic")
    @classmethod
    def normalize_subtopic(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Subtopic")

    @field_validator("previous_errors", "previous_strategies")
    @classmethod
    def normalize_context_lists(cls, values: list[str], info) -> list[str]:
        return clean_text_list(
            values,
            label=info.field_name.replace("_", " ").title(),
        )


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

    student_response_message: str | None = None
    assessment_message: str | None = None
    pedagogical_move_message: str | None = None
    allowed_pedagogical_moves: list[PedagogicalMove] = Field(default_factory=list)
    questions: list[PublicAssessmentQuestion] = Field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    reteach_round: int = 0
    turn_count: int = 0


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

@router.get("/{thread_id}", response_model=TutorSessionResponse)
def get_tutor_session(thread_id: str) -> TutorSessionResponse:
    return _build_snapshot_response(
        thread_id,
        _get_persisted_snapshot(thread_id),
    )


@router.post("/{thread_id}/resume", response_model=TutorSessionResponse)
def resume_tutor_session(thread_id: str) -> TutorSessionResponse:
    snapshot = _get_persisted_snapshot(thread_id)
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


@router.post("/start", response_model=TutorSessionResponse)
def start_tutor_session(request: TutorStartRequest) -> TutorSessionResponse:
    """
    Start one Tutor thread and its first assessment-cycle adaptive attempt.

    ``target_skill`` is an explicit canonical Repository-B skill. Topic and
    subtopic are never silently reinterpreted as BKT skill identifiers.
    """

    thread_id = str(uuid4())
    config = _thread_config(thread_id)

    try:
        target_skill = adaptive_coordinator.validate_target_skill(
            request.target_skill
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    adaptive_attempt_index = 1
    attempt_id = adaptive_coordinator.derive_attempt_id(
        thread_id,
        adaptive_attempt_index,
    )

    retrieved_memory = adapt_memory_context(
        get_memory_client().retrieve_tutor_context(
            student_id=request.student_id,
            target_skill=target_skill,
        ),
        target_skill=target_skill,
    )

    initial_state: TutorState = {
        "student_id": request.student_id,
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

    return _build_public_response(thread_id, graph_result)


@router.post("/{thread_id}/turn", response_model=TutorSessionResponse)
def submit_student_response(
    thread_id: str,
    request: TutorStudentResponseRequest,
) -> TutorSessionResponse:
    """
    Submit one student's conversational reply to the latest teacher turn.

    The graph then uses the GenAI progress judge. If teaching should continue,
    it pauses for a fresh Omash-selected pedagogical move. If the dialogue is
    ready, the Evaluator generates the three-question assessment.
    """

    snapshot = _get_persisted_snapshot(thread_id)
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
) -> TutorSessionResponse:
    """Legacy/manual seam; integrated adaptive turns never consume this."""

    snapshot = _get_persisted_snapshot(thread_id)
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
) -> TutorSessionResponse:
    snapshot = _get_persisted_snapshot(thread_id)
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
