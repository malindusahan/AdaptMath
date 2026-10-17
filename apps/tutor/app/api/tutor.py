import logging
from collections.abc import Mapping
from math import isfinite
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
from app.graph.workflow import (
    adaptive_coordinator,
    complexity_node,
    router_node,
    tutor_graph,
)
from app.clients.memory.memory_client import MemoryAuthenticatedUser, get_memory_client
from app.core.user_auth import require_authenticated_student
from app.integrations.adaptive_component_coordinator import (
    AdaptiveConcurrencyError,
    AdaptiveIntegrationError,
    AdaptiveRestartRequiredError,
)
from app.integrations.memory_context_adapter import adapt_memory_context
from app.schemas.evaluator import StudentAnswer
from app.schemas.dialogue import DialogueTurn
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
    continuation: bool = False
    conversation_history: list[DialogueTurn] = Field(
        default_factory=list,
        max_length=100,
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

    @model_validator(mode="after")
    def validate_continuation_history(self):
        if self.conversation_history and not self.continuation:
            raise ValueError(
                "Conversation history is accepted only for an explicit continuation."
            )
        if self.continuation:
            if not self.conversation_history:
                raise ValueError("A continuation requires conversation history.")
            if self.conversation_history[-1].role != "student":
                raise ValueError(
                    "A continuation must end with the student's latest response."
                )
        return self


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
    memory_classified_skill: str | None = None
    memory_skill_source: Literal[
        "memory_classifier",
        "explicit_target",
        "gemini_fallback",
    ] | None = None
    memory_learner_state: Literal[
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
        "UNAVAILABLE",
    ] | None = None

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


class MoveSelectorContextFeature(BaseModel):
    """One public feature from the pre-action Turn-LinTS context."""

    index: int
    block: Literal["S", "K", "L", "H", "Q", "unknown"]
    name: str
    description: str
    value: float
    missing_indicator: bool = False


class MoveSelectorRewardUpdate(BaseModel):
    """Safe feedback and posterior-update diagnostics for one resolved action."""

    action_event_id: str | None = None
    resolution_status: str | None = None
    outcome_observed: bool = False
    mastery_before: float | None = None
    mastery_after: float | None = None
    raw_delta: float | None = None
    reward_mode: str | None = None
    reward_value: float | None = None
    attributed_move: str | None = None
    posterior_updated: bool = False
    posterior_update_weight: float | None = None
    posterior_total_updates_before: int | None = None
    posterior_total_updates_after: int | None = None
    update_id: str | None = None
    update_timestamp: str | None = None


class MoveSelectorTrace(BaseModel):
    """Public input -> decision -> feedback boundary for Turn-LinTS."""

    action_event_id: str | None = None
    action_turn_index: int | None = None
    policy_mode: str | None = None
    behavior_policy: str | None = None
    decision_source: str | None = None
    context_schema_id: str | None = None
    context_dimension: int = 0
    enabled_blocks: list[str] = Field(default_factory=list)
    context_features: list[MoveSelectorContextFeature] = Field(default_factory=list)
    md7_probabilities: dict[str, float] = Field(default_factory=dict)
    available_arms: list[str] = Field(default_factory=list)
    sampled_scores: dict[str, float] = Field(default_factory=dict)
    policy_scores: dict[str, float] = Field(default_factory=dict)
    base_model_move: str | None = None
    selected_move: str | None = None
    tutor_move: str | None = None
    post_selection_override: bool = False
    response_quality_scores: dict[str, float] = Field(default_factory=dict)
    reward_mode: str | None = None
    latest_reward_update: MoveSelectorRewardUpdate | None = None


class BKTParametersTrace(BaseModel):
    """Trained per-skill parameters used by the authoritative predictor."""

    population_prior: float | None = None
    learn: float | None = None
    guess: float | None = None
    slip: float | None = None
    forget: float | None = None


class BKTHistoryTrace(BaseModel):
    """Observation ledger counts around the displayed transaction."""

    observations_before_attempt: int | None = None
    dialogue_turns_before_assessment: int | None = None
    dialogue_observations_before_assessment: int | None = None
    observations_before_update: int | None = None
    observations_applied: int = 0
    observations_after_update: int | None = None
    persisted_suffix_matches: bool | None = None


class BKTMasteryTrace(BaseModel):
    """Recorded and independently recomputed mastery for one transaction."""

    before: float | None = None
    after: float | None = None
    delta: float | None = None
    recomputed_before: float | None = None
    before_assessment: float | None = None
    dialogue_delta_before_assessment: float | None = None
    assessment_delta: float | None = None
    recomputed_after: float | None = None
    before_consistent: bool | None = None
    after_consistent: bool | None = None


class BKTEvidenceTrace(BaseModel):
    """Safe evidence metadata; raw dialogue and evaluator reasoning are omitted."""

    correctness: str | None = None
    evidence_category: str | None = None
    evaluator_source: str | None = None
    reported_evaluator_confidence: float | None = None
    applied_evaluator_confidence: float | None = None
    evaluator_confidence_cap: float | None = None


class BKTBehaviourTrace(BaseModel):
    """Behaviour detector values that can modulate resolver confidence."""

    reasoning_probability: float | None = None
    reasoning_present: bool | None = None
    uncertainty_probability: float | None = None
    uncertainty_present: bool | None = None
    clarification_probability: float | None = None
    clarification_present: bool | None = None
    repeated_misunderstanding: bool | None = None


class BKTCalculationTrace(BaseModel):
    """One exact confidence-weighted BKT calculation."""

    mastery_before: float
    likelihood_if_known: float
    likelihood_if_not_known: float
    bayes_numerator: float
    bayes_denominator: float
    full_observation_posterior: float
    confidence_weighted_posterior: float
    effective_learn: float
    after_learning: float
    after_forgetting: float
    recorded_mastery_after: float
    matches_recorded_mastery: bool


class BKTObservationTrace(BaseModel):
    """Resolver decision, observation persistence, and optional BKT transition."""

    sequence: int
    event_id: str | None = None
    source_action_event_id: str | None = None
    skill: str | None = None
    primary_signal: str | None = None
    resolver_version: str | None = None
    should_update: bool = False
    outcome: int | None = None
    evidence_weight: float | None = None
    evaluator_confidence: float | None = None
    behavioural_confidence: float | None = None
    behaviour_factor: float | None = None
    update_confidence: float | None = None
    observation_source: str | None = None
    contributors: list[str] = Field(default_factory=list)
    evidence: BKTEvidenceTrace = Field(default_factory=BKTEvidenceTrace)
    behaviour: BKTBehaviourTrace = Field(default_factory=BKTBehaviourTrace)
    persistence_status: str | None = None
    mastery_before: float | None = None
    mastery_after: float | None = None
    delta_mastery: float | None = None
    calculation: BKTCalculationTrace | None = None


class BKTUpdateTrace(BaseModel):
    """Safe input -> resolution -> Bayesian update trace for BKT."""

    schema_version: str | None = None
    event_type: str | None = None
    recorded_at: str | None = None
    attempt_id: str | None = None
    session_id: str | None = None
    target_skill: str | None = None
    turn_index: int | None = None
    update_timing: str | None = None
    assessment_answer_count: int | None = None
    assessment_correct_count: int | None = None
    assessment_incorrect_count: int | None = None
    parameters: BKTParametersTrace = Field(default_factory=BKTParametersTrace)
    effective_initial_prior: float | None = None
    effective_initial_prior_source: str | None = None
    history: BKTHistoryTrace = Field(default_factory=BKTHistoryTrace)
    mastery: BKTMasteryTrace = Field(default_factory=BKTMasteryTrace)
    observations: list[BKTObservationTrace] = Field(default_factory=list)
    persisted_mastery_probability: float | None = None
    persisted_mastery_label: str | None = None
    persisted_initial_prior: float | None = None
    persisted_initial_prior_source: str | None = None
    compatibility_rebased: bool | None = None


class TutorDemoTraceResponse(BaseModel):
    """Safe, panel-facing diagnostics for one persisted Tutor thread."""

    thread_id: str
    phase: str
    question: str
    age: int
    topic: str
    subtopic: str | None = None
    target_skill: str | None = None

    complexity_model: str = "Word + Character TF-IDF + Ridge regression"
    complexity_input_scope: str = "question_text_only"
    complexity_score: float | None = None

    relevant_history_count: int = 0
    previous_error_count: int = 0
    previous_strategy_count: int = 0

    route: str | None = None
    route_reason: str | None = None
    planner_used: bool = False
    planner_output_available: bool = False

    pedagogical_move: str | None = None
    selector: str | None = None
    selected_arm: str | None = None
    base_move: str | None = None
    move_overridden: bool | None = None

    verified_math_ready: bool = False
    verified_math_evidence_count: int = 0
    tutor_generation_fallback_used: bool | None = None

    teaching_status: str | None = None
    teaching_progress_reason: str | None = None

    assessment_question_count: int = 0
    evaluation_available: bool = False
    correct_answer_count: int = 0
    wrong_answer_count: int = 0
    needs_reteaching: bool = False

    turn_count: int = 0
    reteach_round: int = 0
    attempt_id: str | None = None
    adaptive_attempt_index: int | None = None
    checkpoint_saved: bool = True
    next_nodes: list[str] = Field(default_factory=list)
    move_selector: MoveSelectorTrace | None = None
    bkt_update: BKTUpdateTrace | None = None


class DemoRouterScenario(BaseModel):
    """One controlled learner context for the PP2 Router comparison."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, max_length=80)
    age: int = Field(..., ge=8, le=18)
    previous_errors: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )
    previous_strategies: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        return clean_required_text(value, label="Scenario label")

    @field_validator("previous_errors", "previous_strategies")
    @classmethod
    def normalize_demo_context_lists(cls, values: list[str], info) -> list[str]:
        return clean_text_list(
            values,
            label=info.field_name.replace("_", " ").title(),
        )


class DemoRouterComparisonRequest(BaseModel):
    """Run the Complexity model once, then reuse its score across contexts."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LENGTH)
    topic: str = Field(..., min_length=1, max_length=TOPIC_MAX_LENGTH)
    subtopic: str | None = Field(default=None, max_length=SUBTOPIC_MAX_LENGTH)
    scenarios: list[DemoRouterScenario] = Field(..., min_length=2, max_length=5)

    @field_validator("question", "topic")
    @classmethod
    def normalize_demo_required_fields(cls, value: str, info) -> str:
        return clean_required_text(
            value,
            label=info.field_name.replace("_", " ").title(),
        )

    @field_validator("subtopic")
    @classmethod
    def normalize_demo_subtopic(cls, value: str | None) -> str | None:
        return clean_optional_text(value, label="Subtopic")


class DemoRouterScenarioResult(BaseModel):
    label: str
    age: int
    previous_error_count: int
    previous_strategy_count: int
    route: str
    route_reason: str


class DemoRouterComparisonResponse(BaseModel):
    question: str
    complexity_score: float
    complexity_input_scope: str = "question_text_only"
    scenarios: list[DemoRouterScenarioResult]


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


def _safe_demo_text(value: object) -> str | None:
    """Return only short scalar diagnostics suitable for the panel UI."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text[:240] if text else None
    return None


def _safe_demo_selector(value: object) -> str | None:
    """Summarize selector diagnostics without returning probability vectors."""
    if not isinstance(value, dict):
        return _safe_demo_text(value)

    raw = value.get("raw")
    effective = value.get("effective")
    raw_argmax = raw.get("argmax") if isinstance(raw, dict) else None
    effective_argmax = (
        effective.get("argmax") if isinstance(effective, dict) else None
    )
    if raw_argmax and effective_argmax:
        if raw_argmax == effective_argmax:
            return str(raw_argmax)[:240]
        return (
            f"legacy adjustment: raw {raw_argmax} -> effective {effective_argmax}"
        )[:240]
    return _safe_demo_text(effective_argmax or raw_argmax)


_MOVE_SELECTOR_FEATURE_META: dict[str, tuple[str, str]] = {
    "selector_p_generic": ("S", "MD7 probability for generic"),
    "selector_p_probing": ("S", "MD7 probability for probing"),
    "selector_p_focus": ("S", "MD7 probability for focus"),
    "selector_p_telling": ("S", "MD7 probability for telling"),
    "mastery_before": ("K", "BKT mastery available before this action"),
    "previous_mastery_delta": ("K", "Mastery change from the previous resolved action"),
    "previous_mastery_delta_missing": ("K", "1 when previous mastery change is unavailable"),
    "previous_reasoning_probability": ("L", "Previous learner reasoning signal"),
    "previous_uncertainty_probability": ("L", "Previous learner uncertainty signal"),
    "previous_clarification_probability": ("L", "Previous learner clarification signal"),
    "previous_learner_signals_missing": ("L", "1 when previous learner signals are unavailable"),
}


def _safe_demo_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _safe_demo_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _safe_float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        number = _safe_demo_float(raw_value)
        if isinstance(raw_key, str) and number is not None:
            result[raw_key[:80]] = number
    return result


def _safe_demo_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item[:120] for item in value if isinstance(item, str)]


def _bkt_calculation_trace(
    parameters: BKTParametersTrace,
    update: Mapping[str, object],
    transition: Mapping[str, object],
) -> BKTCalculationTrace | None:
    """Reproduce the predictor's two-stage update from recorded inputs."""

    should_update = update.get("should_update") is True
    outcome = update.get("outcome")
    confidence = _safe_demo_float(update.get("update_confidence"))
    before = _safe_demo_float(transition.get("mastery_before_observation"))
    recorded_after = _safe_demo_float(
        transition.get("mastery_after_observation")
    )
    if (
        not should_update
        or isinstance(outcome, bool)
        or outcome not in (0, 1)
        or confidence is None
        or before is None
        or recorded_after is None
        or parameters.learn is None
        or parameters.guess is None
        or parameters.slip is None
        or parameters.forget is None
    ):
        return None

    if outcome == 1:
        likelihood_known = 1.0 - parameters.slip
        likelihood_not_known = parameters.guess
    else:
        likelihood_known = parameters.slip
        likelihood_not_known = 1.0 - parameters.guess
    numerator = before * likelihood_known
    denominator = numerator + (1.0 - before) * likelihood_not_known
    full_posterior = numerator / denominator if denominator > 0.0 else before
    weighted_posterior = (
        confidence * full_posterior + (1.0 - confidence) * before
    )
    effective_learn = confidence * parameters.learn
    after_learning = weighted_posterior + (
        1.0 - weighted_posterior
    ) * effective_learn
    after_forgetting = after_learning * (1.0 - parameters.forget)
    return BKTCalculationTrace(
        mastery_before=before,
        likelihood_if_known=likelihood_known,
        likelihood_if_not_known=likelihood_not_known,
        bayes_numerator=numerator,
        bayes_denominator=denominator,
        full_observation_posterior=full_posterior,
        confidence_weighted_posterior=weighted_posterior,
        effective_learn=effective_learn,
        after_learning=after_learning,
        after_forgetting=after_forgetting,
        recorded_mastery_after=recorded_after,
        matches_recorded_mastery=(
            abs(after_forgetting - recorded_after) <= 1e-9
        ),
    )


def _bkt_update_trace(raw_trace: object) -> BKTUpdateTrace | None:
    if not isinstance(raw_trace, Mapping) or not raw_trace:
        return None

    raw_parameters = raw_trace.get("bkt_parameters")
    raw_parameters = raw_parameters if isinstance(raw_parameters, Mapping) else {}
    parameters = BKTParametersTrace(
        population_prior=_safe_demo_float(raw_parameters.get("prior")),
        learn=_safe_demo_float(raw_parameters.get("learns")),
        guess=_safe_demo_float(raw_parameters.get("guesses")),
        slip=_safe_demo_float(raw_parameters.get("slips")),
        forget=_safe_demo_float(raw_parameters.get("forgets")),
    )

    raw_prior = raw_trace.get("effective_initial_prior")
    raw_prior = raw_prior if isinstance(raw_prior, Mapping) else {}
    raw_history = raw_trace.get("history")
    raw_history = raw_history if isinstance(raw_history, Mapping) else {}
    event_type = _safe_demo_text(raw_trace.get("event"))
    is_assessment = event_type == "assessment_cycle_bkt_update"
    observations_before = _safe_demo_int(
        raw_history.get(
            "observation_count_before_assessment"
            if is_assessment
            else "observation_count_before_dialogue_turn"
        )
    )
    observations_applied = _safe_demo_int(
        raw_history.get(
            "observations_applied_in_assessment"
            if is_assessment
            else "observations_applied_in_dialogue_turn"
        )
    )
    observations_after = _safe_demo_int(
        raw_history.get(
            "observation_count_after_assessment"
            if is_assessment
            else "observation_count_after_dialogue_turn"
        )
    )
    history = BKTHistoryTrace(
        observations_before_attempt=_safe_demo_int(
            raw_history.get("observation_count_before_attempt")
        ),
        dialogue_turns_before_assessment=_safe_demo_int(
            raw_history.get("dialogue_turns_processed_before_assessment")
        ),
        dialogue_observations_before_assessment=_safe_demo_int(
            raw_history.get("dialogue_observations_applied_before_assessment")
        ),
        observations_before_update=observations_before,
        observations_applied=observations_applied or 0,
        observations_after_update=observations_after,
        persisted_suffix_matches=_safe_demo_bool(
            raw_history.get("persisted_suffix_matches_resolved_updates")
        ),
    )

    raw_mastery = raw_trace.get("mastery")
    raw_mastery = raw_mastery if isinstance(raw_mastery, Mapping) else {}
    mastery = BKTMasteryTrace(
        before=_safe_demo_float(raw_mastery.get("before")),
        after=_safe_demo_float(raw_mastery.get("after")),
        delta=_safe_demo_float(raw_mastery.get("delta")),
        recomputed_before=_safe_demo_float(raw_mastery.get("recomputed_before")),
        before_assessment=_safe_demo_float(raw_mastery.get("before_assessment")),
        dialogue_delta_before_assessment=_safe_demo_float(
            raw_mastery.get("dialogue_delta_before_assessment")
        ),
        assessment_delta=_safe_demo_float(raw_mastery.get("assessment_delta")),
        recomputed_after=_safe_demo_float(raw_mastery.get("recomputed_after")),
        before_consistent=_safe_demo_bool(raw_mastery.get("before_consistent")),
        after_consistent=_safe_demo_bool(raw_mastery.get("after_consistent")),
    )

    observations: list[BKTObservationTrace] = []
    raw_observations = raw_trace.get("observations")
    if isinstance(raw_observations, list):
        for index, raw_observation in enumerate(raw_observations, start=1):
            if not isinstance(raw_observation, Mapping):
                continue
            update = raw_observation.get("bkt_update")
            update = update if isinstance(update, Mapping) else {}
            evidence = raw_observation.get("evidence")
            evidence = evidence if isinstance(evidence, Mapping) else {}
            behaviour = raw_observation.get("behaviour")
            behaviour = behaviour if isinstance(behaviour, Mapping) else {}
            transition = raw_observation.get("mastery_transition")
            transition = transition if isinstance(transition, Mapping) else {}
            raw_sequence = _safe_demo_int(raw_observation.get("sequence"))
            raw_outcome = update.get("outcome")
            outcome = (
                int(raw_outcome)
                if not isinstance(raw_outcome, bool) and raw_outcome in (0, 1)
                else None
            )
            observations.append(
                BKTObservationTrace(
                    sequence=raw_sequence or index,
                    event_id=_safe_demo_text(raw_observation.get("event_id")),
                    source_action_event_id=_safe_demo_text(
                        raw_observation.get("source_action_event_id")
                    ),
                    skill=_safe_demo_text(raw_observation.get("skill_id")),
                    primary_signal=_safe_demo_text(
                        raw_observation.get("primary_signal")
                    ),
                    resolver_version=_safe_demo_text(
                        raw_observation.get("resolver_version")
                    ),
                    should_update=update.get("should_update") is True,
                    outcome=outcome,
                    evidence_weight=_safe_demo_float(update.get("evidence_weight")),
                    evaluator_confidence=_safe_demo_float(
                        update.get("evaluator_confidence")
                    ),
                    behavioural_confidence=_safe_demo_float(
                        update.get("behavioural_confidence")
                    ),
                    behaviour_factor=_safe_demo_float(
                        update.get("behaviour_factor")
                    ),
                    update_confidence=_safe_demo_float(
                        update.get("update_confidence")
                    ),
                    observation_source=_safe_demo_text(
                        update.get("observation_source")
                    ),
                    contributors=_safe_string_list(update.get("contributors")),
                    evidence=BKTEvidenceTrace(
                        correctness=_safe_demo_text(evidence.get("correctness")),
                        evidence_category=_safe_demo_text(
                            evidence.get("evidence_category")
                        ),
                        evaluator_source=_safe_demo_text(
                            evidence.get("evaluator_source")
                        ),
                        reported_evaluator_confidence=_safe_demo_float(
                            evidence.get("reported_evaluator_confidence")
                        ),
                        applied_evaluator_confidence=_safe_demo_float(
                            evidence.get("applied_evaluator_confidence")
                        ),
                        evaluator_confidence_cap=_safe_demo_float(
                            evidence.get("evaluator_confidence_cap")
                        ),
                    ),
                    behaviour=BKTBehaviourTrace(
                        reasoning_probability=_safe_demo_float(
                            behaviour.get("reasoning_probability")
                        ),
                        reasoning_present=_safe_demo_bool(
                            behaviour.get("reasoning_present")
                        ),
                        uncertainty_probability=_safe_demo_float(
                            behaviour.get("uncertainty_probability")
                        ),
                        uncertainty_present=_safe_demo_bool(
                            behaviour.get("uncertainty_present")
                        ),
                        clarification_probability=_safe_demo_float(
                            behaviour.get("clarification_probability")
                        ),
                        clarification_present=_safe_demo_bool(
                            behaviour.get("clarification_present")
                        ),
                        repeated_misunderstanding=_safe_demo_bool(
                            raw_observation.get("repeated_misunderstanding")
                        ),
                    ),
                    persistence_status=_safe_demo_text(
                        raw_observation.get("persistence_status")
                    ),
                    mastery_before=_safe_demo_float(
                        transition.get("mastery_before_observation")
                    ),
                    mastery_after=_safe_demo_float(
                        transition.get("mastery_after_observation")
                    ),
                    delta_mastery=_safe_demo_float(
                        transition.get("delta_mastery")
                    ),
                    calculation=_bkt_calculation_trace(
                        parameters,
                        update,
                        transition,
                    ),
                )
            )

    raw_assessment = raw_trace.get("assessment")
    raw_assessment = raw_assessment if isinstance(raw_assessment, Mapping) else {}
    raw_persisted = raw_trace.get("persisted_skill_update")
    raw_persisted = raw_persisted if isinstance(raw_persisted, Mapping) else {}
    return BKTUpdateTrace(
        schema_version=_safe_demo_text(raw_trace.get("schema_version")),
        event_type=event_type,
        recorded_at=_safe_demo_text(raw_trace.get("recorded_at_utc")),
        attempt_id=_safe_demo_text(raw_trace.get("attempt_id")),
        session_id=_safe_demo_text(raw_trace.get("session_id")),
        target_skill=_safe_demo_text(raw_trace.get("target_skill")),
        turn_index=_safe_demo_int(raw_trace.get("turn_index")),
        update_timing=_safe_demo_text(raw_trace.get("update_timing")),
        assessment_answer_count=_safe_demo_int(raw_assessment.get("answer_count")),
        assessment_correct_count=_safe_demo_int(
            raw_assessment.get("correct_count")
        ),
        assessment_incorrect_count=_safe_demo_int(
            raw_assessment.get("incorrect_count")
        ),
        parameters=parameters,
        effective_initial_prior=_safe_demo_float(raw_prior.get("probability")),
        effective_initial_prior_source=_safe_demo_text(raw_prior.get("source")),
        history=history,
        mastery=mastery,
        observations=observations,
        persisted_mastery_probability=_safe_demo_float(
            raw_persisted.get("probability")
        ),
        persisted_mastery_label=_safe_demo_text(raw_persisted.get("label")),
        persisted_initial_prior=_safe_demo_float(
            raw_persisted.get("effective_initial_prior")
        ),
        persisted_initial_prior_source=_safe_demo_text(
            raw_persisted.get("effective_initial_prior_source")
        ),
        compatibility_rebased=_safe_demo_bool(
            raw_persisted.get("compatibility_rebased")
        ),
    )


def _move_selector_trace(
    diagnostics: Mapping[str, object],
    latest_outcome: object,
) -> MoveSelectorTrace | None:
    adaptive = diagnostics.get("adaptive_decision")
    if not isinstance(adaptive, Mapping):
        return None
    policy_context = adaptive.get("policy_context")
    if not isinstance(policy_context, Mapping):
        policy_context = adaptive

    names = policy_context.get("feature_names")
    vector = policy_context.get("vector")
    context_features: list[MoveSelectorContextFeature] = []
    if (
        isinstance(names, list)
        and isinstance(vector, list)
        and len(names) == len(vector)
    ):
        for index, (raw_name, raw_value) in enumerate(zip(names, vector, strict=True)):
            if not isinstance(raw_name, str):
                continue
            value = _safe_demo_float(raw_value)
            if value is None:
                continue
            block, description = _MOVE_SELECTOR_FEATURE_META.get(
                raw_name,
                ("unknown", raw_name.replace("_", " ")),
            )
            context_features.append(
                MoveSelectorContextFeature(
                    index=index + 1,
                    block=block,  # type: ignore[arg-type]
                    name=raw_name[:120],
                    description=description[:240],
                    value=value,
                    missing_indicator=raw_name.endswith("_missing"),
                )
            )

    selector = diagnostics.get("selector")
    raw_selector = selector.get("raw") if isinstance(selector, Mapping) else None
    probabilities = (
        raw_selector.get("probabilities")
        if isinstance(raw_selector, Mapping)
        else diagnostics.get("md7_probabilities")
    )
    if not isinstance(probabilities, Mapping):
        probabilities = diagnostics.get("md6_probabilities")

    selected_move = _safe_demo_text(
        adaptive.get("selected_arm") or diagnostics.get("selected_arm")
    )
    tutor_move = _safe_demo_text(diagnostics.get("pedagogical_move"))
    base_move = _safe_demo_text(
        adaptive.get("base_move") or diagnostics.get("base_move")
    )
    expected_move = selected_move or base_move

    reward_update: MoveSelectorRewardUpdate | None = None
    if isinstance(latest_outcome, Mapping) and latest_outcome:
        reward_update = MoveSelectorRewardUpdate(
            action_event_id=_safe_demo_text(latest_outcome.get("action_event_id")),
            resolution_status=_safe_demo_text(latest_outcome.get("resolution_status")),
            outcome_observed=latest_outcome.get("outcome_observed") is True,
            mastery_before=_safe_demo_float(latest_outcome.get("mastery_before")),
            mastery_after=_safe_demo_float(latest_outcome.get("mastery_after")),
            raw_delta=_safe_demo_float(latest_outcome.get("raw_delta")),
            reward_mode=_safe_demo_text(latest_outcome.get("configured_reward_mode")),
            reward_value=_safe_demo_float(latest_outcome.get("configured_reward_value")),
            attributed_move=_safe_demo_text(latest_outcome.get("reward_attributed_to_move")),
            posterior_updated=latest_outcome.get("posterior_updated") is True,
            posterior_update_weight=_safe_demo_float(
                latest_outcome.get("posterior_update_weight")
            ),
            posterior_total_updates_before=_safe_demo_int(
                latest_outcome.get("posterior_total_updates_before")
            ),
            posterior_total_updates_after=_safe_demo_int(
                latest_outcome.get("posterior_total_updates_after")
            ),
            update_id=_safe_demo_text(latest_outcome.get("update_id")),
            update_timestamp=_safe_demo_text(latest_outcome.get("update_timestamp")),
        )

    enabled_blocks = policy_context.get("enabled_blocks")
    available_arms = adaptive.get("available_arms")
    return MoveSelectorTrace(
        action_event_id=_safe_demo_text(diagnostics.get("action_event_id")),
        action_turn_index=_safe_demo_int(diagnostics.get("action_turn_index")),
        policy_mode=_safe_demo_text(adaptive.get("policy_mode")),
        behavior_policy=_safe_demo_text(adaptive.get("behavior_policy")),
        decision_source=_safe_demo_text(adaptive.get("treatment_assignment_source")),
        context_schema_id=_safe_demo_text(policy_context.get("schema_id")),
        context_dimension=len(context_features),
        enabled_blocks=(
            [str(item)[:20] for item in enabled_blocks]
            if isinstance(enabled_blocks, list)
            else []
        ),
        context_features=context_features,
        md7_probabilities=_safe_float_mapping(probabilities),
        available_arms=(
            [str(item)[:40] for item in available_arms]
            if isinstance(available_arms, list)
            else []
        ),
        sampled_scores=_safe_float_mapping(adaptive.get("sampled_scores")),
        policy_scores=_safe_float_mapping(adaptive.get("policy_scores")),
        base_model_move=base_move,
        selected_move=expected_move,
        tutor_move=tutor_move,
        post_selection_override=(
            expected_move is not None and tutor_move != expected_move
        ),
        response_quality_scores=_safe_float_mapping(diagnostics.get("mrb1_scores")),
        reward_mode=_safe_demo_text(adaptive.get("reward_mode")),
        latest_reward_update=reward_update,
    )


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


def _active_memory_display_fields(graph_result: dict[str, Any]) -> dict[str, Any]:
    """Return UI-only Memory metadata for an active lesson response."""
    return {
        "memory_classified_skill": graph_result.get("memory_classified_skill"),
        "memory_skill_source": graph_result.get("memory_skill_source"),
        "memory_learner_state": graph_result.get("memory_learner_state"),
    }


def _build_public_response(
    thread_id: str,
    graph_result: dict[str, Any],
) -> TutorSessionResponse:
    payload = _extract_interrupt_payload(graph_result)
    base = _base_response_fields(graph_result)
    memory_display = _active_memory_display_fields(graph_result)

    if payload:
        interrupt_type = payload.get("type")

        if interrupt_type == "student_response_required":
            return TutorSessionResponse(
                thread_id=thread_id,
                status="student_response_required",
                student_response_message=payload.get("message"),
                questions=[],
                **memory_display,
                **base,
            )

        if interrupt_type == "pedagogical_move_required":
            return TutorSessionResponse(
                thread_id=thread_id,
                status="pedagogical_move_required",
                pedagogical_move_message=payload.get("message"),
                allowed_pedagogical_moves=payload.get("allowed_moves", []),
                questions=[],
                **memory_display,
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
                **memory_display,
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
    memory_display = _active_memory_display_fields(values)

    if phase == "student_response_required":
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            student_response_message="Reply to your tutor to continue the lesson.",
            questions=[],
            **memory_display,
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
            **memory_display,
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
            **memory_display,
            **base,
        )

    if phase == "recovery_required":
        return TutorSessionResponse(
            thread_id=thread_id,
            status=phase,
            questions=[],
            **memory_display,
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

@router.get("/{thread_id}/demo-trace", response_model=TutorDemoTraceResponse)
def get_tutor_demo_trace(
    thread_id: str,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> TutorDemoTraceResponse:
    """Expose safe workflow decisions without leaking hidden answer content."""

    snapshot = _get_persisted_snapshot(thread_id)
    _require_thread_owner(snapshot, authenticated_user)
    values = dict(getattr(snapshot, "values", {}) or {})

    diagnostics = values.get("adaptive_turn_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    evaluation = values.get("evaluation_result")
    if not isinstance(evaluation, dict):
        evaluation = {}

    verified_math = values.get("verified_math_evidence")
    verified_math_count = len(verified_math) if isinstance(verified_math, list) else 0

    assessment_questions = values.get("assessment_questions")
    assessment_count = (
        len(assessment_questions)
        if isinstance(assessment_questions, list)
        else 0
    )

    relevant_history = values.get("relevant_history")
    previous_errors = values.get("previous_errors")
    previous_strategies = values.get("previous_strategies")
    correct_answers = evaluation.get("correct_answers")
    wrong_answers = evaluation.get("wrong_answers")
    move_selector = _move_selector_trace(
        diagnostics,
        values.get("move_selector_last_outcome"),
    )
    bkt_update = _bkt_update_trace(values.get("bkt_last_update"))

    return TutorDemoTraceResponse(
        thread_id=thread_id,
        phase=_persisted_session_phase(snapshot),
        question=str(values.get("question", "")),
        age=int(values.get("age", 0)),
        topic=str(values.get("topic", "")),
        subtopic=_safe_demo_text(values.get("subtopic")),
        target_skill=_safe_demo_text(values.get("target_skill")),
        complexity_score=values.get("complexity_score"),
        relevant_history_count=(
            len(relevant_history) if isinstance(relevant_history, list) else 0
        ),
        previous_error_count=(
            len(previous_errors) if isinstance(previous_errors, list) else 0
        ),
        previous_strategy_count=(
            len(previous_strategies) if isinstance(previous_strategies, list) else 0
        ),
        route=_safe_demo_text(values.get("route")),
        route_reason=_safe_demo_text(values.get("route_reason")),
        planner_used=values.get("route") == "planned_tutor",
        planner_output_available=isinstance(values.get("planner_output"), dict),
        pedagogical_move=_safe_demo_text(
            diagnostics.get("pedagogical_move")
            or values.get("pedagogical_move")
        ),
        selector=_safe_demo_selector(diagnostics.get("selector")),
        selected_arm=_safe_demo_text(diagnostics.get("selected_arm")),
        base_move=_safe_demo_text(diagnostics.get("base_move")),
        move_overridden=(
            move_selector.post_selection_override
            if move_selector is not None
            else None
        ),
        verified_math_ready=verified_math_count > 0,
        verified_math_evidence_count=verified_math_count,
        tutor_generation_fallback_used=(
            bool(diagnostics.get("tutor_generation_fallback_used"))
            if "tutor_generation_fallback_used" in diagnostics
            else None
        ),
        teaching_status=_safe_demo_text(values.get("teaching_status")),
        teaching_progress_reason=_safe_demo_text(
            values.get("teaching_progress_reason")
        ),
        assessment_question_count=assessment_count,
        evaluation_available=bool(evaluation),
        correct_answer_count=(
            len(correct_answers) if isinstance(correct_answers, list) else 0
        ),
        wrong_answer_count=(
            len(wrong_answers) if isinstance(wrong_answers, list) else 0
        ),
        needs_reteaching=bool(
            evaluation.get(
                "needs_reteaching",
                values.get("needs_reteaching", False),
            )
        ),
        turn_count=int(values.get("turn_count", 0)),
        reteach_round=int(values.get("reteach_round", 0)),
        attempt_id=_safe_demo_text(values.get("attempt_id")),
        adaptive_attempt_index=(
            int(values["adaptive_attempt_index"])
            if isinstance(values.get("adaptive_attempt_index"), int)
            else None
        ),
        checkpoint_saved=True,
        next_nodes=[str(node) for node in (getattr(snapshot, "next", ()) or ())],
        move_selector=move_selector,
        bkt_update=bkt_update,
    )


@router.post(
    "/demo/compare-router",
    response_model=DemoRouterComparisonResponse,
)
def compare_router_contexts(
    request: DemoRouterComparisonRequest,
    authenticated_user: MemoryAuthenticatedUser = Depends(
        require_authenticated_student
    ),
) -> DemoRouterComparisonResponse:
    """Compare real Router decisions without creating a lesson or learner data."""

    user = _resolved_authenticated_user(authenticated_user)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        complexity_score = float(
            complexity_node({"question": request.question})["complexity_score"]
        )
        results: list[DemoRouterScenarioResult] = []

        for index, scenario in enumerate(request.scenarios, start=1):
            demo_state: TutorState = {
                "student_id": f"{user.student_id}-demo-{index}",
                "age": scenario.age,
                "question": request.question,
                "topic": request.topic,
                "subtopic": request.subtopic,
                "complexity_score": complexity_score,
                "relevant_history": [],
                "previous_errors": scenario.previous_errors,
                "previous_strategies": scenario.previous_strategies,
            }
            routed = router_node(demo_state)
            results.append(
                DemoRouterScenarioResult(
                    label=scenario.label,
                    age=scenario.age,
                    previous_error_count=len(scenario.previous_errors),
                    previous_strategy_count=len(scenario.previous_strategies),
                    route=str(routed["route"]),
                    route_reason=str(routed["route_reason"]),
                )
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise model_service_http_exception(exc) from exc

    return DemoRouterComparisonResponse(
        question=request.question,
        complexity_score=complexity_score,
        scenarios=results,
    )


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
    memory_classified_skill: str | None = None
    memory_skill_source: Literal[
        "memory_classifier",
        "explicit_target",
        "gemini_fallback",
    ] = "explicit_target"
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
                memory_skill_source = "gemini_fallback"
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
                memory_classified_skill = memory_skill
                memory_skill_source = "memory_classifier"
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

        memory_response = memory_client.retrieve_tutor_context(
            student_id=student_id,
            target_skill=target_skill,
            session_id=thread_id,
        )
        retrieved_memory = adapt_memory_context(
            memory_response,
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
            "move_selector_last_outcome": {},
            "bkt_last_update": {},
            "relevant_history": retrieved_memory.relevant_history,
            "previous_errors": retrieved_memory.previous_errors,
            "previous_strategies": retrieved_memory.previous_strategies,
            "memory_classified_skill": memory_classified_skill,
            "memory_skill_source": memory_skill_source,
            "memory_learner_state": (
                memory_response.current_learning_state
                if memory_response is not None
                else None
            ),
            "conversation_history": [
                turn.model_dump(mode="json")
                for turn in request.conversation_history
            ],
            "teaching_phase": "initial",
            "teaching_status": "continue_teaching",
            "turn_count": sum(
                turn.role == "teacher"
                for turn in request.conversation_history
            ),
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
