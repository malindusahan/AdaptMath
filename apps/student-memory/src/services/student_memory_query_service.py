"""PostgreSQL-backed query service for current student memory and learning-state history."""

from __future__ import annotations

from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.interaction import CurrentStudentMemory, LearningStateHistoryItem
from src.services.skill_context_resolver import SkillContextResolver


class StudentMemoryQueryService:
    """Read-only query service retrieving persisted memory directly from PostgreSQL."""

    def __init__(self, session_factory: SessionFactory | None = None):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )

    def get_current_memory(
        self,
        student_id: str,
        topic: str,
        subtopic: str | None = None,
    ) -> CurrentStudentMemory | None:
        """
        Retrieve latest persisted CurrentLearningState for student and skill context.

        Does not trigger feature rebuilding or model inference.
        """
        with UnitOfWork(self.session_factory) as uow:
            student = uow.identities.get_student(student_id)
            if student is None:
                return None

            skill = SkillContextResolver.resolve_skill(
                session=uow.session,
                topic=topic,
                subtopic=subtopic,
            )
            if skill is None:
                return None

            current = uow.supporting.get_current(
                student_id=student.student_id,
                skill_id=skill.skill_id,
            )
            if current is None:
                return None

            return CurrentStudentMemory(
                student_id=student_id,
                topic=topic,
                subtopic=subtopic,
                learning_state=current.learning_state,
                evidence_level=current.evidence_level,
                evidence_strength=current.evidence_strength,
                behavioural_coverage=current.behavioural_coverage,
                model_used=current.model_used,
                recent_interaction_count=current.recent_interaction_count,
                attempt_observation_count=current.attempt_observation_count,
                hint_observation_count=current.hint_observation_count,
                response_time_observation_count=(
                    current.response_time_observation_count
                ),
                last_assessment_id=current.last_assessment_id,
                last_snapshot_id=current.last_snapshot_id,
                updated_at=(
                    current.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(current.updated_at, "strftime")
                    else str(current.updated_at)
                ),
            )

    def get_state_history(
        self,
        student_id: str,
        topic: str,
        subtopic: str | None = None,
    ) -> list[LearningStateHistoryItem]:
        """
        Retrieve chronological LearningStateSnapshots for student and skill context.

        Returns empty list if student or skill is not found.
        """
        with UnitOfWork(self.session_factory) as uow:
            student = uow.identities.get_student(student_id)
            if student is None:
                return []

            skill = SkillContextResolver.resolve_skill(
                session=uow.session,
                topic=topic,
                subtopic=subtopic,
            )
            if skill is None:
                return []

            snapshots = uow.supporting.get_history(
                student_id=student.student_id,
                skill_id=skill.skill_id,
            )

            return [
                LearningStateHistoryItem(
                    snapshot_id=s.snapshot_id,
                    assessment_id=s.assessment_id,
                    student_id=student_id,
                    topic=topic,
                    subtopic=subtopic,
                    learning_state=s.learning_state,
                    evidence_level=s.evidence_level,
                    evidence_strength=s.evidence_strength,
                    behavioural_coverage=s.behavioural_coverage,
                    model_used=s.model_used,
                    previous_interaction_count=s.previous_interaction_count,
                    previous_skill_interaction_count=(
                        s.previous_skill_interaction_count
                    ),
                    recent_interaction_count=s.recent_interaction_count,
                    attempt_observation_count=s.attempt_observation_count,
                    hint_observation_count=s.hint_observation_count,
                    response_time_observation_count=(
                        s.response_time_observation_count
                    ),
                    created_at=(
                        s.created_at.strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(s.created_at, "strftime")
                        else str(s.created_at)
                    ),
                )
                for s in snapshots
            ]


_default_query_service: StudentMemoryQueryService | None = None


def get_student_memory_query_service() -> StudentMemoryQueryService:
    global _default_query_service
    if _default_query_service is None:
        _default_query_service = StudentMemoryQueryService()
    return _default_query_service
