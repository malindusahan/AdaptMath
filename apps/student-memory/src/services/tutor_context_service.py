"""Tutor Personalization Context Service providing pedagogical decision context."""

from __future__ import annotations

from src.database.postgres_session import SessionFactory, get_session_factory
from src.schemas.tutor_context import (
    BehaviouralEvidenceSummary,
    TutorContextResponse,
    TutorInteractionSummary,
    TutorMisconceptionSummary,
    TutorRepairSummary,
)
from src.services.student_context_service import (
    StudentContextService,
    get_student_context_service,
)


class TutorContextService:
    """Provides focused, low-noise context tailored specifically for Tutor personalization."""

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

    def get_tutor_context(
        self,
        student_id: str,
        session_id: str | None = None,
        skill_id: str | None = None,
        limit: int = 10,
    ) -> TutorContextResponse:
        """
        Assemble concise, structured context for the Tutor agent:
        - Current dynamic learning state & reliability markers
        - Recent accuracy & counts (concept-scoped > session-scoped > student-wide)
        - Behavioral evidence (attempts, hints, response times)
        - Active misconceptions & recent repair history
        """
        full_context = self.student_context_service.get_student_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )

        # 1. Learning State & Evidence Metadata
        learning_state_str: str | None = None
        evidence_strength_str: str | None = None
        behavioural_coverage_str: str | None = None

        if full_context.learning_state:
            learning_state_str = full_context.learning_state.get("learning_state")
            evidence_strength_str = full_context.learning_state.get("evidence_strength")
            behavioural_coverage_str = full_context.learning_state.get("behavioural_coverage")

        # 2. Select best available evidence tier for performance metrics
        active_memory = (
            full_context.concept_memory
            or full_context.short_term_memory
            or full_context.long_term_memory
        )

        recent_accuracy: float | None = None
        recent_correct_count = 0
        recent_incorrect_count = 0

        attempt_evidence = BehaviouralEvidenceSummary()
        hint_evidence = BehaviouralEvidenceSummary()
        response_time_evidence = BehaviouralEvidenceSummary()

        if active_memory:
            for accuracy_field in (
                "accuracy",
                "recent_accuracy",
                "overall_accuracy",
            ):
                candidate = active_memory.get(accuracy_field)
                if candidate is not None:
                    recent_accuracy = candidate
                    break
            recent_correct_count = active_memory.get("correct_count", 0)
            recent_incorrect_count = active_memory.get("incorrect_count", 0)

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

        # 3. Misconceptions
        misconceptions_summary = [
            TutorMisconceptionSummary(
                misconception_id=m["misconception_id"],
                normalized_error=m["normalized_error"],
                display_error=m["display_error"],
                occurrence_count=m["occurrence_count"],
                last_seen_at=m["last_seen_at"],
            )
            for m in full_context.misconceptions
        ]

        # 4. Recent Interactions
        interactions_summary = [
            TutorInteractionSummary(
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

        # 5. Recent Repairs
        repairs_summary = [
            TutorRepairSummary(
                repair_outcome_id=r.repair_outcome_id,
                session_id=r.session_id,
                canonical_skill_id=r.canonical_skill_id,
                repair_action=r.repair_action,
                outcome=r.outcome,
                score=r.score,
                notes=r.notes,
                created_at=r.created_at,
            )
            for r in full_context.recent_repairs
        ]

        return TutorContextResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            current_learning_state=learning_state_str,
            evidence_strength=evidence_strength_str,
            behavioural_coverage=behavioural_coverage_str,
            recent_accuracy=recent_accuracy,
            recent_correct_count=recent_correct_count,
            recent_incorrect_count=recent_incorrect_count,
            attempt_evidence=attempt_evidence,
            hint_evidence=hint_evidence,
            response_time_evidence=response_time_evidence,
            misconceptions=misconceptions_summary,
            recent_interactions=interactions_summary,
            recent_repairs=repairs_summary,
        )


# Singleton factory cache
_TUTOR_CONTEXT_SERVICE: TutorContextService | None = None


def get_tutor_context_service(
    session_factory: SessionFactory | None = None,
) -> TutorContextService:
    global _TUTOR_CONTEXT_SERVICE
    if _TUTOR_CONTEXT_SERVICE is None or session_factory is not None:
        _TUTOR_CONTEXT_SERVICE = TutorContextService(session_factory=session_factory)
    return _TUTOR_CONTEXT_SERVICE
