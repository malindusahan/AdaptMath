from typing import Any, Literal, Mapping

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.complexity.service import get_complexity_service
from app.agents.evaluator.evaluator_agent import EvaluatorAgent
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.progress.teaching_progress_agent import TeachingProgressAgent
from app.agents.router.router_agent import RouterAgent
from app.agents.tutor.tutor_agent import TutorAgent
from app.clients.memory.memory_client import get_memory_client
from app.core.input_validation import ANSWER_MAX_LENGTH, clean_required_text
from app.core.persistence import tutor_checkpointer
from app.graph.state import TutorState
from app.integrations.adaptive_component_coordinator import (
    build_default_adaptive_component_coordinator,
)
from app.integrations.memory_event_adapter import build_completed_attempt_evidence
from app.schemas.assessment import AssessmentGenerationInput, AssessmentQuestion
from app.schemas.dialogue import DialogueTurn
from app.schemas.evaluator import EvaluatorInput, StudentAnswer
from app.schemas.memory import MemoryContext
from app.schemas.pedagogical_move import PedagogicalMoveSelection
from app.schemas.planner import PlannerInput, PlannerOutput
from app.schemas.profile import StudentProfile
from app.schemas.progress import TeachingProgressInput
from app.schemas.router import RouterInput
from app.schemas.tutor import TutorInput, TutorMathInput


# ============================================================
# AGENTS
# ============================================================

router_agent = RouterAgent()
planner_agent = PlannerAgent()
tutor_agent = TutorAgent()
progress_agent = TeachingProgressAgent()
evaluator_agent = EvaluatorAgent()
adaptive_coordinator = build_default_adaptive_component_coordinator()


# ============================================================
# OPTIONAL MEMORY INTEGRATION CLIENT
# ============================================================

# Hamooth owns cross-session learner evidence. Failures at this boundary never
# invalidate an already completed BKT/policy attempt.
memory_client = get_memory_client()


# ============================================================
# COMPLEXITY
# ============================================================

def complexity_node(state: TutorState) -> dict:
    score = get_complexity_service().predict(state["question"])
    return {"complexity_score": score}


# ============================================================
# ROUTER
# ============================================================

def router_node(state: TutorState) -> dict:
    profile = StudentProfile(
        student_id=state["student_id"],
        age=state["age"],
    )
    memory = MemoryContext(
        student_id=state["student_id"],
        topic=state["topic"],
        subtopic=state.get("subtopic"),
        relevant_history=state.get("relevant_history", []),
        previous_errors=state.get("previous_errors", []),
        previous_strategies=state.get("previous_strategies", []),
    )
    result = router_agent.route(
        RouterInput(
            question=state["question"],
            profile=profile,
            memory=memory,
            complexity_score=state["complexity_score"],
        )
    )
    return {
        "route": result.route,
        "route_reason": result.reason,
    }


def choose_initial_workflow(
    state: TutorState,
) -> Literal["direct_tutor", "planned_tutor"]:
    return state["route"]


# ============================================================
# PLANNER
# ============================================================

def planner_node(state: TutorState) -> dict:
    result = planner_agent.create_plan(
        PlannerInput(
            question=state["question"],
            topic=state["topic"],
            subtopic=state.get("subtopic"),
            student_age=state["age"],
            complexity_score=state["complexity_score"],
            previous_errors=state.get("previous_errors", []),
            previous_strategies=state.get("previous_strategies", []),
        )
    )
    return {"planner_output": result.model_dump()}


# ============================================================
# ONE-TIME VERIFIED MATHEMATICAL PREPARATION
# ============================================================

def _planner_output_from_state(state: TutorState) -> PlannerOutput | None:
    raw = state.get("planner_output")
    if not raw:
        return None
    return PlannerOutput.model_validate(raw)


def math_preparation_node(state: TutorState) -> dict:
    """
    Solve/check the mathematical problem internally once and preserve verified
    evidence for all later interactive teacher turns.

    This avoids recomputing the entire mathematics solution on every student
    message while retaining the response-verification layer.
    """

    math_input = TutorMathInput(
        question=state["question"],
        topic=state["topic"],
        subtopic=state.get("subtopic"),
        student_age=state["age"],
        complexity_score=state["complexity_score"],
        planner_output=_planner_output_from_state(state),
        previous_errors=state.get("previous_errors", []),
        reteaching=False,
    )
    evidence = tutor_agent.prepare_math_evidence(math_input)
    return {"verified_math_evidence": evidence}


# ============================================================
# ADAPTIVE ATTEMPT / TEACHER TURN
# ============================================================

def start_adaptive_attempt_node(state: TutorState) -> dict:
    """Start Repository B and frozen v3 with one shared cycle identity."""

    return adaptive_coordinator.start_attempt(state)


def adaptive_tutor_turn_node(state: TutorState) -> dict:
    """Delegate the complete MD6 -> LinTS -> Tutor -> MRB1 operation."""

    result = adaptive_coordinator.run_tutor_turn(
        state,
        tutor_agent=tutor_agent,
    )
    diagnostics = {
        key: result[key]
        for key in (
            "attempt_id",
            "turn_index",
            "pedagogical_move",
            "selected_arm",
            "base_move",
            "overridden",
            "eligible_arms",
            "context",
            "md6_probabilities",
            "mrb1_scores",
        )
        if key in result
    }
    return {
        "tutor_response": result["tutor_response"],
        # The response was appended exactly once by TutorStateMemoryAdapter.
        "conversation_history": state["conversation_history"],
        "pedagogical_move": result["pedagogical_move"],
        "adaptive_turn_diagnostics": diagnostics,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ============================================================
# ONE INTERACTIVE TEACHER TURN
# ============================================================

def tutor_node(state: TutorState) -> dict:
    """
    Generate ONE teacher utterance using exactly the pedagogical move supplied
    by Omash's external move selector.

    The Tutor never chooses the move and never generates the formal assessment.
    """

    selected_move = state.get("pedagogical_move")
    if selected_move is None:
        raise ValueError(
            "Tutor cannot run without an externally selected pedagogical move."
        )

    conversation = [
        DialogueTurn.model_validate(turn)
        for turn in state.get("conversation_history", [])
    ]

    tutor_input = TutorInput(
        question=state["question"],
        topic=state["topic"],
        subtopic=state.get("subtopic"),
        student_age=state["age"],
        complexity_score=state["complexity_score"],
        planner_output=_planner_output_from_state(state),
        pedagogical_move=selected_move,
        conversation_history=conversation,
        previous_errors=state.get("previous_errors", []),
        reteaching=state.get("teaching_phase", "initial") == "reteaching",
    )

    result = tutor_agent.teach_turn(
        tutor_input,
        state.get("verified_math_evidence", []),
    )

    teacher_turn = DialogueTurn(
        role="teacher",
        content=result.teaching_response,
        pedagogical_move=selected_move,
    )

    updated_history = [
        *state.get("conversation_history", []),
        teacher_turn.model_dump(),
    ]

    return {
        "tutor_response": result.teaching_response,
        "conversation_history": updated_history,
        "pedagogical_move": None,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ============================================================
# WAIT FOR THE STUDENT'S NEXT DIALOGUE TURN
# ============================================================

def await_student_response_node(state: TutorState) -> dict:
    """Pause after each teacher utterance and wait for one student response."""

    submitted = interrupt(
        {
            "type": "student_response_required",
            "message": "Reply to your tutor to continue the lesson.",
            "teacher_message": state.get("tutor_response"),
        }
    )

    if isinstance(submitted, dict):
        submitted = submitted.get("response")

    if not isinstance(submitted, str):
        raise ValueError("Student response must be supplied as text.")

    cleaned = clean_required_text(submitted, label="Student response")
    if len(cleaned) > ANSWER_MAX_LENGTH:
        raise ValueError(
            f"Student response must be at most {ANSWER_MAX_LENGTH} characters."
        )

    student_turn = DialogueTurn(
        role="student",
        content=cleaned,
    )

    return {
        "latest_student_response": cleaned,
        "conversation_history": [
            *state.get("conversation_history", []),
            student_turn.model_dump(),
        ],
    }


# ============================================================
# GENAI TEACHING-PROGRESS JUDGMENT
# ============================================================

def teaching_progress_node(state: TutorState) -> dict:
    """
    Decide whether interactive teaching should continue or whether the learner
    is ready for the formal 3-question assessment.

    This node does NOT choose the next pedagogical move.
    """

    result = progress_agent.judge(
        TeachingProgressInput(
            original_question=state["question"],
            topic=state["topic"],
            subtopic=state.get("subtopic"),
            student_age=state["age"],
            planner_output=_planner_output_from_state(state),
            conversation_history=[
                DialogueTurn.model_validate(turn)
                for turn in state.get("conversation_history", [])
            ],
            previous_errors=state.get("previous_errors", []),
            reteaching=state.get("teaching_phase", "initial") == "reteaching",
        )
    )

    return {
        "teaching_status": result.status,
        "teaching_progress_reason": result.reason,
    }


def after_teaching_progress(
    state: TutorState,
) -> Literal["adaptive_turn", "generate_assessment"]:
    if state["teaching_status"] == "continue_teaching":
        return "adaptive_turn"
    return "generate_assessment"


# ============================================================
# WAIT FOR OMASH'S NEXT MOVE
# ============================================================

def await_pedagogical_move_node(state: TutorState) -> dict:
    """
    Pause until Omash's component supplies the move for the NEXT teacher turn.

    The real HTTP/service contract is intentionally not invented here. Once
    Omash's API is available this integration point can be replaced by a client
    call without changing Tutor ownership.
    """

    submitted_move = interrupt(
        {
            "type": "pedagogical_move_required",
            "message": (
                "An externally selected pedagogical move is required for the "
                "next teacher turn."
            ),
            "allowed_moves": ["telling", "focus", "generic", "probing"],
        }
    )

    if isinstance(submitted_move, str):
        submitted_move = {"move": submitted_move}

    selection = PedagogicalMoveSelection.model_validate(submitted_move)
    return {"pedagogical_move": selection.move}


# ============================================================
# EVALUATOR-GENERATED THREE-QUESTION ASSESSMENT
# ============================================================

def assessment_generation_node(state: TutorState) -> dict:
    result = evaluator_agent.generate_assessment(
        AssessmentGenerationInput(
            original_question=state["question"],
            topic=state["topic"],
            subtopic=state.get("subtopic"),
            student_age=state["age"],
            conversation_history=[
                DialogueTurn.model_validate(turn)
                for turn in state.get("conversation_history", [])
            ],
            planner_output=_planner_output_from_state(state),
            previous_errors=state.get("previous_errors", []),
            reteaching=state.get("teaching_phase", "initial") == "reteaching",
        )
    )
    return {
        "assessment_questions": [
            question.model_dump() for question in result.assessment_questions
        ],
        "student_answers": [],
        "evaluation_result": {},
        "correct_answers": [],
        "wrong_answers": [],
        "needs_reteaching": False,
    }


# ============================================================
# WAIT FOR THREE ASSESSMENT ANSWERS
# ============================================================

def await_answers_node(state: TutorState) -> dict:
    assessment_questions = state.get("assessment_questions", [])

    public_questions = [
        {
            "question_id": question["question_id"],
            "question": question["question"],
        }
        for question in assessment_questions
    ]

    submitted_answers = interrupt(
        {
            "type": "assessment_required",
            "message": "Please answer all three assessment questions.",
            "questions": public_questions,
        }
    )

    if not isinstance(submitted_answers, list):
        raise ValueError("Student answers must be provided as a list.")

    validated_answers = [
        StudentAnswer.model_validate(answer)
        for answer in submitted_answers
    ]

    question_ids = {
        question["question_id"] for question in assessment_questions
    }
    answer_ids = [answer.question_id for answer in validated_answers]

    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Each assessment question must be answered once.")
    if set(answer_ids) != question_ids or len(answer_ids) != 3:
        raise ValueError("Answers must be supplied for all three assessment questions.")

    answer_by_id = {answer.question_id: answer.answer for answer in validated_answers}
    updated_history = list(state.get("conversation_history", []))
    for question in assessment_questions:
        updated_history.append(
            DialogueTurn(
                role="teacher",
                content=f"Assessment: {question['question']}",
            ).model_dump()
        )
        updated_history.append(
            DialogueTurn(
                role="student",
                content=answer_by_id[question["question_id"]],
            ).model_dump()
        )

    return {
        "student_answers": [answer.model_dump() for answer in validated_answers],
        "conversation_history": updated_history,
    }


# ============================================================
# EVALUATE THE THREE ANSWERS
# ============================================================

def evaluator_node(state: TutorState) -> dict:
    assessment_questions = [
        AssessmentQuestion.model_validate(question)
        for question in state.get("assessment_questions", [])
    ]
    student_answers = [
        StudentAnswer.model_validate(answer)
        for answer in state.get("student_answers", [])
    ]

    result = evaluator_agent.evaluate(
        EvaluatorInput(
            original_question=state["question"],
            topic=state["topic"],
            subtopic=state.get("subtopic"),
            assessment_questions=assessment_questions,
            student_answers=student_answers,
        )
    )

    return {
        "evaluation_result": result.model_dump(),
        "correct_answers": [answer.model_dump() for answer in result.correct_answers],
        "wrong_answers": [answer.model_dump() for answer in result.wrong_answers],
        "needs_reteaching": result.needs_reteaching,
    }


# ============================================================
# AUTHORITATIVE STUDENT-MODEL + FROZEN-V3 COMPLETION
# ============================================================

def complete_adaptive_attempt_node(state: TutorState) -> dict:
    """Complete one assessment cycle once and atomically start reteaching."""

    return adaptive_coordinator.finish_attempt(state)


# ============================================================
# SEND EVALUATION EVIDENCE TO HAMOOTH-MEMORY INTEGRATION
# ============================================================

def _memory_write_candidates(
    state: TutorState,
    current_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return a stable, de-duplicated queue without trusting external state."""
    candidates: list[dict[str, Any]] = []
    raw_queue = state.get("memory_pending_writes", [])
    if isinstance(raw_queue, list):
        for item in raw_queue:
            if isinstance(item, Mapping):
                candidate = dict(item)
                if candidate not in candidates:
                    candidates.append(candidate)

    legacy = state.get("memory_pending_write")
    if isinstance(legacy, Mapping):
        candidate = dict(legacy)
        if candidate not in candidates:
            candidates.append(candidate)

    if current_payload is not None and current_payload not in candidates:
        candidates.append(current_payload)
    return candidates

def memory_update_node(state: TutorState) -> dict:
    evaluation_result = state.get("evaluation_result", {})
    identified_errors = evaluation_result.get("identified_errors", [])

    payload = build_completed_attempt_evidence(state)
    if payload is None:
        current_payload = None
        current_result = {
            "status": "skipped_incomplete",
            "retryable": False,
        }
    else:
        current_payload = payload.model_dump(mode="json")
        current_result = None

    pending_payloads: list[dict[str, Any]] = []
    for candidate in _memory_write_candidates(state, current_payload):
        result = memory_client.write_completed_attempt(candidate)
        if current_payload is not None and candidate == current_payload:
            current_result = result
        if bool(result.get("retryable", False)):
            pending_payloads.append(candidate)

    if current_result is None:
        current_result = {
            "status": "skipped_incomplete",
            "retryable": False,
        }

    updated_errors = list(
        dict.fromkeys(
            state.get("previous_errors", []) + identified_errors
        )
    )

    return {
        "memory_update_result": current_result,
        "memory_write_pending": bool(pending_payloads),
        "memory_pending_write": pending_payloads[0] if pending_payloads else None,
        "memory_pending_writes": pending_payloads,
        "previous_errors": updated_errors,
    }


def after_memory(
    state: TutorState,
) -> Literal["begin_reteaching", "finish"]:
    if state.get("needs_reteaching", False):
        return "begin_reteaching"
    return "finish"


# ============================================================
# START ANOTHER INTERACTIVE RETEACHING CYCLE
# ============================================================

def begin_reteaching_node(state: TutorState) -> dict:
    """
    Reset only cycle-specific assessment state. The completion node has already
    finished the prior adaptive attempt and started the next indexed attempt.
    Dialogue history and the new authoritative mastery snapshot are preserved.
    """

    return {
        "teaching_phase": "reteaching",
        "teaching_status": "continue_teaching",
        "teaching_progress_reason": "",
        "assessment_questions": [],
        "student_answers": [],
        "reteach_round": state.get("reteach_round", 0) + 1,
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_tutor_graph():
    builder = StateGraph(TutorState)

    builder.add_node("complexity", complexity_node)
    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("prepare_math", math_preparation_node)
    builder.add_node("start_adaptive_attempt", start_adaptive_attempt_node)
    builder.add_node("adaptive_tutor_turn", adaptive_tutor_turn_node)
    builder.add_node("await_student_response", await_student_response_node)
    builder.add_node("teaching_progress", teaching_progress_node)
    builder.add_node("generate_assessment", assessment_generation_node)
    builder.add_node("await_answers", await_answers_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("complete_adaptive_attempt", complete_adaptive_attempt_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("begin_reteaching", begin_reteaching_node)

    builder.add_edge(START, "complexity")
    builder.add_edge("complexity", "router")

    builder.add_conditional_edges(
        "router",
        choose_initial_workflow,
        {
            "direct_tutor": "prepare_math",
            "planned_tutor": "planner",
        },
    )
    builder.add_edge("planner", "prepare_math")

    # The first adaptive attempt starts only after verified math evidence is
    # cached, then frozen v3 owns the complete teacher-turn operation.
    builder.add_edge("prepare_math", "start_adaptive_attempt")
    builder.add_edge("start_adaptive_attempt", "adaptive_tutor_turn")

    # One teacher turn -> one student response -> GenAI progress judgment.
    builder.add_edge("adaptive_tutor_turn", "await_student_response")
    builder.add_edge("await_student_response", "teaching_progress")
    builder.add_conditional_edges(
        "teaching_progress",
        after_teaching_progress,
        {
            "adaptive_turn": "adaptive_tutor_turn",
            "generate_assessment": "generate_assessment",
        },
    )

    # Formal assessment exists only after the interactive teaching phase is
    # judged ready. The Evaluator generates exactly three questions.
    builder.add_edge("generate_assessment", "await_answers")
    builder.add_edge("await_answers", "evaluator")
    builder.add_edge("evaluator", "complete_adaptive_attempt")
    builder.add_edge("complete_adaptive_attempt", "memory_update")

    builder.add_conditional_edges(
        "memory_update",
        after_memory,
        {
            "begin_reteaching": "begin_reteaching",
            "finish": END,
        },
    )

    # Wrong assessment answers already rolled over to a new adaptive attempt.
    builder.add_edge("begin_reteaching", "adaptive_tutor_turn")

    return builder.compile(checkpointer=tutor_checkpointer)


tutor_graph = build_tutor_graph()
