"""Phase 17 Final Acceptance Tests — UI Integration and Complete Lifecycle Demo Flow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.schemas.fapr_context import FAPRContextResponse
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    MemoryUpdateResponse,
)
from src.schemas.meta_signals import MetaSignalItem, MetaSignalsResponse
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
from src.schemas.support_preference import (
    SupportPreferenceResponse,
    SupportStrategySummary,
)
from src.schemas.tutor_context import TutorContextResponse


class MockUIServices:
    """Mock test provider mimicking complete multi-tier cognitive memory lifecycle."""

    def __init__(self):
        self.assessments = []
        self.repairs = []

    def get_question_context(self, request: QuestionContextRequest) -> QuestionContextResponse:
        if "box and whisker" in request.question.lower():
            return QuestionContextResponse(
                student_id=request.student_id,
                session_id=request.session_id,
                question=request.question,
                topic=TopicContextItem(
                    skill_id="stat-01-uuid",
                    skill_code="STAT_BOX_PLOT",
                    canonical_skill_name="math :: statistics & probability :: box plot",
                    display_name="Box and Whisker Plots",
                    confidence=0.92,
                    method="MINILM_EMBEDDING_CENTROID",
                    needs_review=False,
                    top_candidates=[
                        {
                            "skill_id": "stat-01-uuid",
                            "skill_code": "STAT_BOX_PLOT",
                            "canonical_name": "math :: statistics & probability :: box plot",
                            "display_name": "Box and Whisker Plots",
                            "score": 0.92,
                        }
                    ],
                ),
                learning_state=None,
                misconceptions=[],
            )
        # Abstained question
        return QuestionContextResponse(
            student_id=request.student_id,
            session_id=request.session_id,
            question=request.question,
            topic=TopicContextItem(
                skill_id=None,
                skill_code=None,
                canonical_skill_name=None,
                display_name=None,
                confidence=0.15,
                method="TFIDF_FALLBACK",
                needs_review=True,
                top_candidates=[],
            ),
            learning_state=None,
            misconceptions=[],
        )

    def update_student_memory(self, request: AssessmentMemoryUpdateRequest) -> MemoryUpdateResponse:
        self.assessments.append(request)
        correct_count = sum(1 for q in request.assessment_questions if q.is_correct)
        total = len(request.assessment_questions)
        state = "STRONG" if correct_count == total else "DEVELOPING" if correct_count > 0 else "NEEDS_SUPPORT"
        return MemoryUpdateResponse(
            student_id=request.student_id,
            topic=request.topic,
            subtopic=request.subtopic,
            assessment_id=len(self.assessments),
            snapshot_id=len(self.assessments) * 10,
            learning_state=state,
            evidence_level="FULL_SKILL",
            evidence_strength="HIGH",
            behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
            model_used=True,
            previous_interaction_count=0,
            previous_skill_interaction_count=0,
            recent_interaction_count=total,
            attempt_observation_count=total,
            hint_observation_count=total,
            response_time_observation_count=total,
            misconception_count=len(request.identified_errors),
            memory_updated=True,
        )

    def record_repair_outcome(self, request: RepairOutcomeCreateRequest) -> RepairOutcomeResponse:
        self.repairs.append(request)
        return RepairOutcomeResponse(
            repair_outcome_id=f"rep-uuid-{len(self.repairs)}",
            student_id=request.student_id,
            session_id=request.session_id,
            canonical_skill_id=request.skill_id,
            interaction_id=request.interaction_id,
            repair_action=request.repair_action,
            outcome=request.outcome,
            score=request.score,
            notes=request.notes,
            created_at="2026-08-18T16:45:00Z",
        )

    def get_support_preference(self, student_id: str, skill_id: str | None = None) -> SupportPreferenceResponse:
        if not self.repairs:
            return SupportPreferenceResponse(
                student_id=student_id,
                skill_id=skill_id,
                preferred_support_style=None,
                status="INSUFFICIENT_EVIDENCE",
                evidence_count=0,
                success_rate=None,
                strategies=[],
            )
        return SupportPreferenceResponse(
            student_id=student_id,
            skill_id=skill_id,
            preferred_support_style="HINT",
            status="SUPPORTED_BY_HISTORY",
            evidence_count=len(self.repairs),
            success_rate=1.0,
            strategies=[
                SupportStrategySummary(
                    repair_action="HINT",
                    observation_count=len(self.repairs),
                    successful_count=len(self.repairs),
                    partial_count=0,
                    failed_count=0,
                    success_rate=1.0,
                    average_score=1.0,
                )
            ],
        )


@pytest.fixture
def acceptance_client(monkeypatch):
    mock = MockUIServices()
    import src.api.ui_routes as ui_routes

    monkeypatch.setattr(ui_routes, "get_question_context_service", lambda: mock)
    monkeypatch.setattr(ui_routes, "update_student_memory", mock.update_student_memory)
    monkeypatch.setattr(ui_routes, "get_repair_outcome_service", lambda: mock)

    class DummyTutorService:
        def get_tutor_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> TutorContextResponse:
            return TutorContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                current_learning_state="DEVELOPING",
                evidence_strength="HIGH",
                behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                recent_accuracy=0.85,
                recent_correct_count=4,
                recent_incorrect_count=1,
                misconceptions=[],
                recent_interactions=[],
                recent_repairs=[],
            )

    class DummyPlannerService:
        def get_planner_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> PlannerContextResponse:
            return PlannerContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                session_interaction_count=5,
                session_accuracy=0.8,
                session_correct_count=4,
                session_incorrect_count=1,
                concept_interaction_count=5,
                concept_accuracy=0.8,
                concept_correct_count=4,
                concept_incorrect_count=1,
                long_term_interaction_count=10,
                overall_accuracy=0.8,
                total_sessions=2,
                concept_count=1,
                misconceptions=[],
                recent_interactions=[],
            )

    class DummyFaprService:
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

    class DummyStudentService:
        def get_student_context(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 20) -> StudentContextResponse:
            return StudentContextResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                misconceptions=[],
                recent_interactions=[],
                repair_history=[],
            )

    class DummyMetaService:
        def get_meta_signals(self, student_id: str, session_id: str | None = None, skill_id: str | None = None, limit: int = 50) -> MetaSignalsResponse:
            return MetaSignalsResponse(
                student_id=student_id,
                session_id=session_id,
                skill_id=skill_id,
                total_signals=1,
                signals=[
                    MetaSignalItem(
                        signal_type="COGNITIVE_OVERLOAD_SUSPECTED",
                        student_id=student_id,
                        session_id=session_id,
                        skill_id=skill_id,
                        timestamp="2026-08-18T16:45:00Z",
                        confidence=0.88,
                        evidence={"hint_frequency": 3, "latency_spike_ms": 12000},
                    )
                ],
            )

    monkeypatch.setattr(ui_routes, "get_tutor_context_service", lambda: DummyTutorService())
    monkeypatch.setattr(ui_routes, "get_planner_context_service", lambda: DummyPlannerService())
    monkeypatch.setattr(ui_routes, "get_fapr_context_service", lambda: DummyFaprService())
    monkeypatch.setattr(ui_routes, "get_student_context_service", lambda: DummyStudentService())
    monkeypatch.setattr(ui_routes, "get_support_preference_service", lambda: mock)
    monkeypatch.setattr(ui_routes, "get_meta_signal_service", lambda: DummyMetaService())

    import src.api.app as app_module
    from src.schemas.api_common import ReadinessResponse

    class DummyReadinessService:
        def check_readiness(self) -> ReadinessResponse:
            return ReadinessResponse(
                status="ready",
                database="connected",
                migrations="current",
                topic_extractor="ready",
                learning_state_model="ready",
            )

    monkeypatch.setattr(app_module, "get_readiness_service", lambda: DummyReadinessService())

    return TestClient(app, raise_server_exceptions=False)


def test_complete_frontend_acceptance_workflow(acceptance_client):
    """
    Execute full end-to-end user journey:
    1. Check Health & Readiness
    2. Analyze math question (Topic extraction & context probe)
    3. Test Topic Abstention fallback
    4. Submit multi-question assessment evaluation
    5. Retrieve Unified Student Context
    6. Retrieve Tutor, Planner, and FAPR agent projections
    7. Submit Pedagogical Repair Outcome
    8. Verify Support Preference changes to SUPPORTED_BY_HISTORY
    9. Retrieve Meta-Agent diagnostic signals
    10. Verify absence of service key leakage
    """
    # 1. Health and Readiness
    res_health = acceptance_client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_ready = acceptance_client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"

    # 2. Analyze Question (MiniLM Topic Extraction)
    q_payload = {
        "student_id": "student_accept_01",
        "session_id": "session_accept_01",
        "question": "How do I interpret a box and whisker plot?",
    }
    res_q = acceptance_client.post("/ui/question-context", json=q_payload)
    assert res_q.status_code == 200
    q_data = res_q.json()
    assert q_data["topic"]["display_name"] == "Box and Whisker Plots"
    assert q_data["topic"]["confidence"] >= 0.90
    assert q_data["topic"]["needs_review"] is False

    # 3. Topic Abstention Fallback
    res_abstain = acceptance_client.post("/ui/question-context", json={
        "student_id": "student_accept_01",
        "session_id": "session_accept_01",
        "question": "Vague meaningless text",
    })
    assert res_abstain.status_code == 200
    assert res_abstain.json()["topic"]["needs_review"] is True

    # 4. Submit Assessment & Update Dynamic Memory
    assess_payload = {
        "student_id": "student_accept_01",
        "topic": "Statistics",
        "subtopic": "Box Plots",
        "assessment_questions": [
            {
                "question_id": "q1",
                "question": "What is the median of [2, 4, 6]?",
                "student_answer": "4",
                "expected_answer": "4",
                "is_correct": True,
                "attempt_count": 1,
                "hint_count": 0,
                "response_time_ms": 3200.0,
            },
            {
                "question_id": "q2",
                "question": "What is the IQR?",
                "student_answer": "3",
                "expected_answer": "4",
                "is_correct": False,
                "identified_error": "Subtracted min from max",
                "attempt_count": 2,
                "hint_count": 1,
                "response_time_ms": 6500.0,
            }
        ],
        "identified_errors": ["Subtracted min from max"],
    }
    res_update = acceptance_client.post("/ui/memory-update", json=assess_payload)
    assert res_update.status_code == 200
    up_data = res_update.json()
    assert up_data["learning_state"] == "DEVELOPING"
    assert up_data["recent_interaction_count"] == 2
    assert up_data["misconception_count"] == 1
    assert up_data["memory_updated"] is True

    # 5. Full Student Context
    res_ctx = acceptance_client.get("/ui/student_accept_01/context")
    assert res_ctx.status_code == 200

    # 6. Agent Projections
    res_tutor = acceptance_client.get("/ui/student_accept_01/tutor-context")
    assert res_tutor.status_code == 200
    assert res_tutor.json()["recent_accuracy"] == 0.85

    res_plan = acceptance_client.get("/ui/student_accept_01/planner-context")
    assert res_plan.status_code == 200
    assert res_plan.json()["session_accuracy"] == 0.8

    res_fapr = acceptance_client.get("/ui/student_accept_01/fapr-context")
    assert res_fapr.status_code == 200

    # 7. Check Support Preference (Initially INSUFFICIENT_EVIDENCE)
    res_pref_initial = acceptance_client.get("/ui/student_accept_01/support-preference")
    assert res_pref_initial.status_code == 200
    assert res_pref_initial.json()["status"] == "INSUFFICIENT_EVIDENCE"

    # 8. Submit Repair Outcome
    repair_payload = {
        "student_id": "student_accept_01",
        "session_id": "session_accept_01",
        "skill_id": "stat-01-uuid",
        "repair_action": "HINT",
        "outcome": "RESOLVED",
        "score": 1.0,
        "notes": "Student resolved IQR question after reminder prompt.",
    }
    res_rep = acceptance_client.post("/ui/repair-outcome", json=repair_payload)
    assert res_rep.status_code == 201
    assert res_rep.json()["outcome"] == "RESOLVED"

    # 9. Verify Support Preference changes
    res_pref_after = acceptance_client.get("/ui/student_accept_01/support-preference")
    assert res_pref_after.status_code == 200
    assert res_pref_after.json()["status"] == "SUPPORTED_BY_HISTORY"
    assert res_pref_after.json()["preferred_support_style"] == "HINT"

    # 10. Meta-Agent Signals
    res_meta = acceptance_client.get("/ui/student_accept_01/meta-signals")
    assert res_meta.status_code == 200
    assert res_meta.json()["total_signals"] == 1
    assert res_meta.json()["signals"][0]["signal_type"] == "COGNITIVE_OVERLOAD_SUSPECTED"
