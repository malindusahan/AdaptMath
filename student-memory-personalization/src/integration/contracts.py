"""External Component Integration Contracts for Multi-Agent Tutoring Architecture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

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
)
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)
from src.schemas.student_context import StudentContextResponse
from src.schemas.support_preference import SupportPreferenceResponse
from src.schemas.tutor_context import TutorContextResponse


class ConsumerComponent(str, Enum):
    """External agent or orchestrator components consuming Student Personalization Memory."""

    EVALUATOR = "Evaluator"
    TUTOR = "Tutor"
    PLANNER = "Planner"
    FAPR_LB = "FAPR-LB"
    META_AGENT = "Meta-Agent"
    ORCHESTRATOR = "Orchestrator"


class HttpMethod(str, Enum):
    """Supported HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass(frozen=True)
class IntegrationContract:
    """Formal specification of an external integration interface with the Memory Component."""

    contract_id: str
    component: ConsumerComponent
    method: HttpMethod
    path: str
    request_model: type[Any] | None
    response_model: type[Any]
    auth_required: bool
    purpose: str
    query_params: tuple[str, ...] = ()
    expected_status: int = 200


INTEGRATION_CONTRACTS: dict[str, IntegrationContract] = {
    "evaluator_update": IntegrationContract(
        contract_id="evaluator_update",
        component=ConsumerComponent.EVALUATOR,
        method=HttpMethod.POST,
        path="/memory/update",
        request_model=AssessmentMemoryUpdateRequest,
        response_model=MemoryUpdateResponse,
        auth_required=True,
        purpose="Evaluator posts evaluated assessment results to update cognitive memory and calculate learning-state transition.",
        expected_status=200,
    ),
    "tutor_context": IntegrationContract(
        contract_id="tutor_context",
        component=ConsumerComponent.TUTOR,
        method=HttpMethod.GET,
        path="/memory/{student_id}/tutor-context",
        request_model=None,
        response_model=TutorContextResponse,
        auth_required=True,
        purpose="Tutor retrieves immediate pedagogical state, behavioral observations, and active misconceptions for scaffolding adaptation.",
        query_params=("session_id", "skill_id", "limit"),
        expected_status=200,
    ),
    "planner_context": IntegrationContract(
        contract_id="planner_context",
        component=ConsumerComponent.PLANNER,
        method=HttpMethod.GET,
        path="/memory/{student_id}/planner-context",
        request_model=None,
        response_model=PlannerContextResponse,
        auth_required=True,
        purpose="Planner retrieves cross-concept learning history, longitudinal retention, and session progression for curriculum sequencing.",
        query_params=("session_id", "skill_id", "limit"),
        expected_status=200,
    ),
    "fapr_context": IntegrationContract(
        contract_id="fapr_context",
        component=ConsumerComponent.FAPR_LB,
        method=HttpMethod.GET,
        path="/memory/{student_id}/fapr-context",
        request_model=None,
        response_model=FAPRContextResponse,
        auth_required=True,
        purpose="FAPR-LB retrieves error patterns, prior repair attempts, and latest student utterance to diagnose failure and select repair strategies.",
        query_params=("session_id", "skill_id", "limit"),
        expected_status=200,
    ),
    "repair_outcome": IntegrationContract(
        contract_id="repair_outcome",
        component=ConsumerComponent.FAPR_LB,
        method=HttpMethod.POST,
        path="/memory/repair-outcome",
        request_model=RepairOutcomeCreateRequest,
        response_model=RepairOutcomeResponse,
        auth_required=True,
        purpose="FAPR-LB logs pedagogical repair intervention outcomes to build empirical support strategy preferences.",
        expected_status=201,
    ),
    "support_preference": IntegrationContract(
        contract_id="support_preference",
        component=ConsumerComponent.FAPR_LB,
        method=HttpMethod.GET,
        path="/memory/{student_id}/support-preference",
        request_model=None,
        response_model=SupportPreferenceResponse,
        auth_required=True,
        purpose="FAPR-LB and Tutor retrieve empirical support strategy efficacy rankings based on historical repair success.",
        query_params=("skill_id",),
        expected_status=200,
    ),
    "meta_signals": IntegrationContract(
        contract_id="meta_signals",
        component=ConsumerComponent.META_AGENT,
        method=HttpMethod.GET,
        path="/memory/{student_id}/meta-signals",
        request_model=None,
        response_model=MetaSignalsResponse,
        auth_required=True,
        purpose="Meta-Agent audits behavioral signals, latency anomalies, and cognitive overload indicators.",
        query_params=("session_id", "skill_id", "limit"),
        expected_status=200,
    ),
    "question_context": IntegrationContract(
        contract_id="question_context",
        component=ConsumerComponent.ORCHESTRATOR,
        method=HttpMethod.POST,
        path="/memory/question-context",
        request_model=QuestionContextRequest,
        response_model=QuestionContextResponse,
        auth_required=True,
        purpose="Orchestrator submits free-text mathematics question for topic extraction and immediate epistemic context retrieval.",
        expected_status=200,
    ),
    "full_context": IntegrationContract(
        contract_id="full_context",
        component=ConsumerComponent.ORCHESTRATOR,
        method=HttpMethod.GET,
        path="/memory/{student_id}/context",
        request_model=None,
        response_model=StudentContextResponse,
        auth_required=True,
        purpose="Orchestrator retrieves complete unified cognitive memory projection across all tiers.",
        query_params=("session_id", "skill_id", "limit"),
        expected_status=200,
    ),
}


def get_contract(contract_id: str) -> IntegrationContract:
    """Retrieve an integration contract by unique identifier."""
    if contract_id not in INTEGRATION_CONTRACTS:
        raise KeyError(f"Integration contract '{contract_id}' not found in registry.")
    return INTEGRATION_CONTRACTS[contract_id]


def list_contracts() -> list[IntegrationContract]:
    """List all registered external integration contracts."""
    return list(INTEGRATION_CONTRACTS.values())


def get_contracts_for_component(component: ConsumerComponent | str) -> list[IntegrationContract]:
    """Retrieve all contracts associated with a specific external consumer component."""
    target = ConsumerComponent(component) if isinstance(component, str) else component
    return [c for c in INTEGRATION_CONTRACTS.values() if c.component == target]
