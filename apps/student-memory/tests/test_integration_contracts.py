"""Phase 18 — Step 1: Tests for External Component Integration Contracts and Mock Flows."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.integration.contracts import (
    INTEGRATION_CONTRACTS,
    ConsumerComponent,
    HttpMethod,
    IntegrationContract,
    get_contract,
    get_contracts_for_component,
    list_contracts,
)
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
    TopicContextItem,
)
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)
from src.schemas.student_context import StudentContextResponse
from src.schemas.support_preference import SupportPreferenceResponse
from src.schemas.tutor_context import TutorContextResponse


TEST_SERVICE_KEY = "test_integration_service_key_secret_12345"


@pytest.fixture
def integration_client(monkeypatch):
    """Test client configured with valid service key auth and mocked service handlers."""
    import src.api.auth as auth_module
    import src.api.memory_routes as memory_routes

    monkeypatch.setenv("MEMORY_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEMORY_SERVICE_API_KEY", TEST_SERVICE_KEY)

    # Mock Evaluator update
    def mock_update(request: AssessmentMemoryUpdateRequest) -> MemoryUpdateResponse:
        return MemoryUpdateResponse(
            student_id=request.student_id,
            topic=request.topic,
            subtopic=request.subtopic,
            assessment_id=8801,
            snapshot_id=9901,
            learning_state="DEVELOPING",
            evidence_level="PARTIAL_SKILL",
            evidence_strength="MEDIUM",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            previous_interaction_count=0,
            previous_skill_interaction_count=0,
            recent_interaction_count=len(request.assessment_questions),
            attempt_observation_count=len(request.assessment_questions),
            hint_observation_count=len(request.assessment_questions),
            response_time_observation_count=len(request.assessment_questions),
            misconception_count=len(request.identified_errors),
            memory_updated=True,
        )

    # Mock Tutor context
    class MockTutorService:
        def get_tutor_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> TutorContextResponse:
            return TutorContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                current_learning_state="DEVELOPING",
                evidence_strength="MEDIUM",
                behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                recent_accuracy=0.75,
                recent_correct_count=3,
                recent_incorrect_count=1,
                misconceptions=[],
                recent_interactions=[],
                recent_repairs=[],
            )

    # Mock Planner context
    class MockPlannerService:
        def get_planner_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> PlannerContextResponse:
            return PlannerContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                session_interaction_count=4,
                session_accuracy=0.75,
                session_correct_count=3,
                session_incorrect_count=1,
                concept_interaction_count=4,
                concept_accuracy=0.75,
                concept_correct_count=3,
                concept_incorrect_count=1,
                long_term_interaction_count=8,
                overall_accuracy=0.75,
                total_sessions=2,
                concept_count=1,
                misconceptions=[],
                recent_interactions=[],
            )

    # Mock FAPR context
    class MockFaprService:
        def get_fapr_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 10) -> FAPRContextResponse:
            return FAPRContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                recent_accuracy=0.5,
                recent_incorrect_count=1,
                misconceptions=[],
                recent_interactions=[],
                previous_repairs=[],
            )

    # Mock Repair outcome service
    class MockRepairService:
        def record_repair_outcome(self, request: RepairOutcomeCreateRequest) -> RepairOutcomeResponse:
            return RepairOutcomeResponse(
                repair_outcome_id="rep-outcome-uuid-99",
                student_id=request.student_id,
                session_id=request.session_id,
                canonical_skill_id=request.skill_id,
                interaction_id=request.interaction_id,
                repair_action=request.repair_action,
                outcome=request.outcome,
                score=request.score,
                notes=request.notes,
                created_at="2026-08-18T18:00:00Z",
            )

    # Mock Support preference service
    class MockSupportPrefService:
        def get_support_preference(self, student_id: str, skill_id: str | None = None) -> SupportPreferenceResponse:
            return SupportPreferenceResponse(
                student_id=student_id,
                skill_id=skill_id,
                preferred_support_style="HINT",
                status="SUPPORTED_BY_HISTORY",
                evidence_count=3,
                success_rate=1.0,
                strategies=[],
            )

    # Mock Meta signal service
    class MockMetaSignalService:
        def get_meta_signals(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 50) -> MetaSignalsResponse:
            return MetaSignalsResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                total_signals=0,
                signals=[],
            )

    # Mock Question context service
    class MockQuestionContextService:
        def get_question_context(self, request: QuestionContextRequest) -> QuestionContextResponse:
            return QuestionContextResponse(
                student_id=request.student_id,
                session_id=request.session_id,
                question=request.question,
                topic=TopicContextItem(
                    skill_id="alg-01-uuid",
                    skill_code="ALG_LIN_EQ",
                    canonical_skill_name="math :: algebra :: linear equations",
                    display_name="Linear Equations",
                    confidence=0.95,
                    method="MINILM_EMBEDDING_CENTROID",
                    needs_review=False,
                    top_candidates=[],
                ),
                learning_state=None,
                misconceptions=[],
            )

    # Mock Student full context service
    class MockStudentContextService:
        def get_student_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> StudentContextResponse:
            return StudentContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                misconceptions=[],
                recent_interactions=[],
                repair_history=[],
            )

    monkeypatch.setattr(memory_routes, "update_student_memory", mock_update)
    monkeypatch.setattr(memory_routes, "get_tutor_context_service", lambda: MockTutorService())
    monkeypatch.setattr(memory_routes, "get_planner_context_service", lambda: MockPlannerService())
    monkeypatch.setattr(memory_routes, "get_fapr_context_service", lambda: MockFaprService())
    monkeypatch.setattr(memory_routes, "get_repair_outcome_service", lambda: MockRepairService())
    monkeypatch.setattr(memory_routes, "get_support_preference_service", lambda: MockSupportPrefService())
    monkeypatch.setattr(memory_routes, "get_meta_signal_service", lambda: MockMetaSignalService())
    monkeypatch.setattr(memory_routes, "get_question_context_service", lambda: MockQuestionContextService())
    monkeypatch.setattr(memory_routes, "get_student_context_service", lambda: MockStudentContextService())

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Contract Registry Integrity & Route Mapping Tests
# ---------------------------------------------------------------------------

def test_all_expected_contracts_registered():
    """Verify all 9 integration contracts are properly declared in registry."""
    expected_ids = {
        "evaluator_update",
        "tutor_context",
        "planner_context",
        "fapr_context",
        "repair_outcome",
        "support_preference",
        "meta_signals",
        "question_context",
        "full_context",
    }
    assert set(INTEGRATION_CONTRACTS.keys()) == expected_ids
    assert len(list_contracts()) == 9


def test_contracts_have_valid_components_and_methods():
    """Verify every contract has a valid component owner, HTTP method, and purpose."""
    for contract_id, contract in INTEGRATION_CONTRACTS.items():
        assert isinstance(contract.component, ConsumerComponent)
        assert isinstance(contract.method, HttpMethod)
        assert contract.path.startswith("/memory/")
        assert contract.purpose and len(contract.purpose) > 15
        assert contract.auth_required is True
        assert contract.expected_status in (200, 201)


def test_contract_registry_query_helpers():
    """Verify get_contract and get_contracts_for_component query helpers."""
    evaluator_contracts = get_contracts_for_component(ConsumerComponent.EVALUATOR)
    assert len(evaluator_contracts) == 1
    assert evaluator_contracts[0].contract_id == "evaluator_update"

    fapr_contracts = get_contracts_for_component(ConsumerComponent.FAPR_LB)
    assert len(fapr_contracts) == 3
    assert {c.contract_id for c in fapr_contracts} == {"fapr_context", "repair_outcome", "support_preference"}

    with pytest.raises(KeyError):
        get_contract("non_existent_contract")


def test_contracts_map_to_registered_fastapi_routes():
    """Verify every contract path and HTTP method maps to an actual FastAPI endpoint in OpenAPI."""
    openapi = app.openapi()
    paths = openapi.get("paths", {})

    for contract_id, contract in INTEGRATION_CONTRACTS.items():
        assert contract.path in paths, f"Contract {contract_id} path {contract.path} not found in OpenAPI paths!"
        method_lower = contract.method.value.lower()
        assert method_lower in paths[contract.path], (
            f"Contract {contract_id} method {contract.method} not found under {contract.path} in OpenAPI!"
        )


def test_health_and_ready_excluded_from_agent_contracts():
    """Verify infrastructure routes /health and /ready are NOT declared as agent contracts."""
    contract_paths = {c.path for c in INTEGRATION_CONTRACTS.values()}
    assert "/health" not in contract_paths
    assert "/ready" not in contract_paths


def test_no_duplicate_contract_ids():
    """Verify uniqueness of contract IDs."""
    ids = [c.contract_id for c in list_contracts()]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Authentication Invariant Tests for All Contracts
# ---------------------------------------------------------------------------

def test_all_contracts_reject_missing_service_key(integration_client):
    """Verify every external contract endpoint rejects requests with missing X-Service-Key (401)."""
    for contract in INTEGRATION_CONTRACTS.values():
        path = contract.path.replace("{student_id}", "student_test_401")
        if contract.method == HttpMethod.POST:
            res = integration_client.post(path, json={})
        elif contract.method == HttpMethod.GET:
            res = integration_client.get(path)
        else:
            continue

        assert res.status_code == 401, f"Contract {contract.contract_id} did not enforce 401 on missing auth!"
        data = res.json()
        assert data["error_code"] == "UNAUTHORIZED"


def test_all_contracts_reject_invalid_service_key(integration_client):
    """Verify every external contract endpoint rejects requests with invalid X-Service-Key (401)."""
    bad_headers = {"X-Service-Key": "wrong_key_xyz"}
    for contract in INTEGRATION_CONTRACTS.values():
        path = contract.path.replace("{student_id}", "student_test_401")
        if contract.method == HttpMethod.POST:
            res = integration_client.post(path, json={}, headers=bad_headers)
        elif contract.method == HttpMethod.GET:
            res = integration_client.get(path, headers=bad_headers)
        else:
            continue

        assert res.status_code == 401, f"Contract {contract.contract_id} did not enforce 401 on bad key!"


# ---------------------------------------------------------------------------
# Mock External Consumer Directional Flow Tests
# ---------------------------------------------------------------------------

def test_mock_evaluator_consumer_flow(integration_client):
    """Mock Evaluator agent posts completed assessment and receives MemoryUpdateResponse."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}
    payload = {
        "student_id": "stud_eval_01",
        "topic": "Algebra",
        "subtopic": "Linear equations",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "Solve 3x = 15",
                "student_answer": "5",
                "expected_answer": "5",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 3200.0,
            }
        ],
        "identified_errors": [],
    }
    contract = get_contract("evaluator_update")
    res = integration_client.post(contract.path, json=payload, headers=headers)
    assert res.status_code == contract.expected_status
    data = res.json()
    assert data["student_id"] == "stud_eval_01"
    assert data["learning_state"] == "DEVELOPING"
    assert data["memory_updated"] is True


def test_mock_tutor_consumer_flow(integration_client):
    """Mock Tutor agent queries tutor-context and receives TutorContextResponse."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}
    contract = get_contract("tutor_context")
    path = contract.path.replace("{student_id}", "stud_tutor_01")
    res = integration_client.get(path, headers=headers)
    assert res.status_code == contract.expected_status
    data = res.json()
    assert data["student_id"] == "stud_tutor_01"
    assert data["current_learning_state"] == "DEVELOPING"
    assert data["recent_accuracy"] == 0.75


def test_mock_planner_consumer_flow(integration_client):
    """Mock Planner agent queries planner-context and receives PlannerContextResponse."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}
    contract = get_contract("planner_context")
    path = contract.path.replace("{student_id}", "stud_planner_01")
    res = integration_client.get(path, headers=headers)
    assert res.status_code == contract.expected_status
    data = res.json()
    assert data["student_id"] == "stud_planner_01"
    assert data["overall_accuracy"] == 0.75
    assert data["total_sessions"] == 2


def test_mock_fapr_consumer_flow(integration_client):
    """Mock FAPR-LB agent queries fapr-context and logs repair outcome."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}

    # 1. Query FAPR Context
    ctx_contract = get_contract("fapr_context")
    ctx_path = ctx_contract.path.replace("{student_id}", "stud_fapr_01")
    res_ctx = integration_client.get(ctx_path, headers=headers)
    assert res_ctx.status_code == ctx_contract.expected_status
    assert res_ctx.json()["recent_incorrect_count"] == 1

    # 2. Record Repair Outcome
    rep_contract = get_contract("repair_outcome")
    rep_payload = {
        "student_id": "stud_fapr_01",
        "session_id": "sess_fapr_01",
        "skill_id": "alg-01-uuid",
        "repair_action": "HINT",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Targeted hint resolved student misconception.",
    }
    res_rep = integration_client.post(rep_contract.path, json=rep_payload, headers=headers)
    assert res_rep.status_code == rep_contract.expected_status
    assert res_rep.json()["outcome"] == "RESOLVED"

    # 3. Query Support Preference
    pref_contract = get_contract("support_preference")
    pref_path = pref_contract.path.replace("{student_id}", "stud_fapr_01")
    res_pref = integration_client.get(pref_path, headers=headers)
    assert res_pref.status_code == pref_contract.expected_status
    assert res_pref.json()["preferred_support_style"] == "HINT"


def test_mock_meta_agent_consumer_flow(integration_client):
    """Mock Meta-Agent queries meta-signals and receives MetaSignalsResponse."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}
    contract = get_contract("meta_signals")
    path = contract.path.replace("{student_id}", "stud_meta_01")
    res = integration_client.get(path, headers=headers)
    assert res.status_code == contract.expected_status
    data = res.json()
    assert data["student_id"] == "stud_meta_01"
    assert data["total_signals"] == 0


def test_mock_orchestrator_consumer_flow(integration_client):
    """Mock Orchestrator queries question-context and full unified context."""
    headers = {"X-Service-Key": TEST_SERVICE_KEY}

    # 1. Question Context
    q_contract = get_contract("question_context")
    q_payload = {
        "student_id": "stud_orch_01",
        "session_id": "sess_orch_01",
        "question": "Solve 3x + 2 = 11",
    }
    res_q = integration_client.post(q_contract.path, json=q_payload, headers=headers)
    assert res_q.status_code == q_contract.expected_status
    assert res_q.json()["topic"]["display_name"] == "Linear Equations"

    # 2. Full Unified Context
    full_contract = get_contract("full_context")
    full_path = full_contract.path.replace("{student_id}", "stud_orch_01")
    res_full = integration_client.get(full_path, headers=headers)
    assert res_full.status_code == full_contract.expected_status
    assert res_full.json()["student_id"] == "stud_orch_01"
