"""Planner Personalization Context Service providing longitudinal and performance evidence."""

from __future__ import annotations

from src.database.postgres_session import SessionFactory, get_session_factory
from src.schemas.planner_context import (
    BehaviouralEvidenceSummary,
    PlannerContextResponse,
    PlannerInteractionSummary,
    PlannerMisconceptionSummary,
)
from src.services.student_context_service import (
    StudentContextService,
    get_student_context_service,
)


class PlannerContextService:
    """Provides structured performance, behavioral, and longitudinal context for pedagogical planning."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        student_context_service: StudentContextService | None = None,
    ):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        self.student_context_service = (
            student_context_service
            if student_context_service is not None
            else get_student_context_service(session_factory=self.session_factory)
        )

    def get_planner_context(
        self,
        student_id: str,
        session_id: str | None = None,
        skill_id: str | None = None,
        limit: int = 10,
    ) -> PlannerContextResponse:
        """
        Assemble comprehensive evidence context for the Planner agent:
        - Session metrics (interactions, accuracy, counts)
        - Concept metrics (skill-scoped interactions & mastery)
        - Longitudinal metrics (overall sessions, concepts touched, lifetime accuracy)
        - Behavioral evidence (attempts, hints, response latency)
        - Active learning state & misconceptions
        """
        full_context = self.student_context_service.get_student_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )

        # 1. Learning State & Evidence Reliability
        learning_state_str: str | None = None
        evidence_strength_str: str | None = None
        behavioural_coverage_str: str | None = None

        if full_context.learning_state:
            learning_state_str = full_context.learning_state.get("learning_state")
            evidence_strength_str = full_context.learning_state.get("evidence_strength")
            behavioural_coverage_str = full_context.learning_state.get("behavioural_coverage")

        # 2. Session-Level Performance
        session_interaction_count = 0
        session_accuracy: float | None = None
        session_correct_count = 0
        session_incorrect_count = 0

        if full_context.short_term_memory:
            stm = full_context.short_term_memory
            session_interaction_count = stm.get("interaction_count", 0)
            session_accuracy = stm.get("recent_accuracy")
            session_correct_count = stm.get("correct_count", 0)
            session_incorrect_count = stm.get("incorrect_count", 0)

        # 3. Concept-Level Performance (Skill-scoped)
        concept_interaction_count = 0
        concept_accuracy: float | None = None
        concept_correct_count = 0
        concept_incorrect_count = 0

        if full_context.concept_memory:
            cm = full_context.concept_memory
            concept_interaction_count = cm.get("interaction_count", 0)
            concept_accuracy = cm.get("accuracy")
            concept_correct_count = cm.get("correct_count", 0)
            concept_incorrect_count = cm.get("incorrect_count", 0)

        # 4. Long-Term / Longitudinal Metrics
        long_term_interaction_count = 0
        overall_accuracy: float | None = None
        total_sessions = 0
        concept_count = 0

        if full_context.long_term_memory:
            ltm = full_context.long_term_memory
            long_term_interaction_count = ltm.get("interaction_count", 0)
            overall_accuracy = ltm.get("overall_accuracy")
            total_sessions = ltm.get("total_sessions", 0)
            concept_count = ltm.get("concept_count", 0)

        # 5. Behavioral Evidence Summary
        active_memory = (
            full_context.concept_memory
            or full_context.short_term_memory
            or full_context.long_term_memory
        )

        attempt_evidence = BehaviouralEvidenceSummary()
        hint_evidence = BehaviouralEvidenceSummary()
        response_time_evidence = BehaviouralEvidenceSummary()

        if active_memory:
            # Attempts
            att_obs = active_memory.get("attempt_observation_count", 0)
            att_sum = float(active_memory.get("attempt_sum", 0))
            attempt_evidence = BehaviouralEvidenceSummary(
                observation_count=att_obs,
                total_value=att_sum,
                average_value=round(att_sum / att_obs, 2) if att_obs > 0 else None,
            )

            # Hints
            hint_obs = active_memory.get("hint_observation_count", 0)
            hint_sum = float(active_memory.get("hint_sum", 0))
            hint_evidence = BehaviouralEvidenceSummary(
                observation_count=hint_obs,
                total_value=hint_sum,
                average_value=round(hint_sum / hint_obs, 2) if hint_obs > 0 else None,
            )

            # Response Times
            rt_obs = active_memory.get("response_time_observation_count", 0)
            rt_sum = float(active_memory.get("response_time_sum_ms", 0.0))
            response_time_evidence = BehaviouralEvidenceSummary(
                observation_count=rt_obs,
                total_value=rt_sum,
                average_value=round(rt_sum / rt_obs, 2) if rt_obs > 0 else None,
            )

        # 6. Misconceptions
        misconceptions_summary = [
            PlannerMisconceptionSummary(
                misconception_id=m["misconception_id"],
                normalized_error=m["normalized_error"],
                display_error=m["display_error"],
                occurrence_count=m["occurrence_count"],
                last_seen_at=m["last_seen_at"],
            )
            for m in full_context.misconceptions
        ]

        # 7. Recent Interactions
        interactions_summary = [
            PlannerInteractionSummary(
                interaction_id=i.interaction_id,
                session_id=i.session_id,
                canonical_skill_id=i.canonical_skill_id,
                is_correct=i.is_correct,
                attempt_count=i.attempt_count,
                hint_count=i.hint_count,
                response_time_ms=i.response_time_ms,
                created_at=i.created_at,
            )
            for i in full_context.recent_interactions
        ]

        return PlannerContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            session_interaction_count=session_interaction_count,
            session_accuracy=session_accuracy,
            session_correct_count=session_correct_count,
            session_incorrect_count=session_incorrect_count,
            attempt_evidence=attempt_evidence,
            hint_evidence=hint_evidence,
            response_time_evidence=response_time_evidence,
            current_learning_state=learning_state_str,
            evidence_strength=evidence_strength_str,
            behavioural_coverage=behavioural_coverage_str,
            concept_interaction_count=concept_interaction_count,
            concept_accuracy=concept_accuracy,
            concept_correct_count=concept_correct_count,
            concept_incorrect_count=concept_incorrect_count,
            long_term_interaction_count=long_term_interaction_count,
            overall_accuracy=overall_accuracy,
            total_sessions=total_sessions,
            concept_count=concept_count,
            misconceptions=misconceptions_summary,
            recent_interactions=interactions_summary,
        )


# Singleton factory cache
_PLANNER_CONTEXT_SERVICE: PlannerContextService | None = None


def get_planner_context_service(
    session_factory: SessionFactory | None = None,
) -> PlannerContextService:
    global _PLANNER_CONTEXT_SERVICE
    if _PLANNER_CONTEXT_SERVICE is None or session_factory is not None:
        _PLANNER_CONTEXT_SERVICE = PlannerContextService(session_factory=session_factory)
    return _PLANNER_CONTEXT_SERVICE
