"""Support Preference Evidence Service analyzing empirical repair efficacy."""

from __future__ import annotations

from collections import defaultdict
import uuid
from fastapi import HTTPException, status
from sqlalchemy import select

from src.database.models.core import CanonicalSkill, Student
from src.database.models.supporting_memory import RepairOutcome
from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.support_preference import (
    SupportPreferenceResponse,
    SupportStrategySummary,
)

MIN_OBSERVATIONS_FOR_PREFERENCE = 3


class SupportPreferenceService:
    """Estimates historical pedagogical support preferences from past repair outcomes."""

    def __init__(self, session_factory: SessionFactory | None = None):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )

    def get_support_preference(
        self,
        student_id: str,
        skill_id: str | None = None,
    ) -> SupportPreferenceResponse:
        """
        Analyze repair history for a student (and optionally canonical skill)
        to identify evidence-backed support strategy preferences.
        """
        student_raw = student_id.strip()

        with UnitOfWork(self.session_factory) as uow:
            assert uow.session is not None

            # 1. Resolve Student
            student: Student | None = None
            try:
                stud_uuid = uuid.UUID(student_raw)
                student = uow.session.scalar(
                    select(Student).where(
                        (Student.student_id == stud_uuid)
                        | (Student.external_student_id == student_raw)
                    )
                )
            except ValueError:
                student = uow.session.scalar(
                    select(Student).where(Student.external_student_id == student_raw)
                )

            # If student does not exist, return safe default insufficient evidence
            if student is None:
                return SupportPreferenceResponse(
                    student_id=student_id,
                    skill_id=skill_id,
                    preferred_support_style=None,
                    status="INSUFFICIENT_EVIDENCE",
                    evidence_count=None,
                    success_rate=None,
                    strategies=[],
                )

            # 2. Resolve Skill (if specified)
            skill: CanonicalSkill | None = None
            if skill_id and skill_id.strip():
                skill_raw = skill_id.strip()
                try:
                    sk_uuid = uuid.UUID(skill_raw)
                    skill = uow.session.scalar(
                        select(CanonicalSkill).where(
                            (CanonicalSkill.skill_id == sk_uuid)
                            | (CanonicalSkill.canonical_name == skill_raw)
                            | (CanonicalSkill.display_name == skill_raw)
                        )
                    )
                except ValueError:
                    skill = uow.session.scalar(
                        select(CanonicalSkill).where(
                            (CanonicalSkill.canonical_name == skill_raw)
                            | (CanonicalSkill.display_name == skill_raw)
                        )
                    )

                if skill is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Canonical skill '{skill_id}' not found.",
                    )

            # 3. Retrieve Repair Outcomes
            if skill is not None:
                records = uow.repairs.for_student_skill(student.student_id, skill.skill_id)
            else:
                records = uow.repairs.for_student(student.student_id)

            if not records:
                return SupportPreferenceResponse(
                    student_id=student_id,
                    skill_id=str(skill.skill_id) if skill else None,
                    preferred_support_style=None,
                    status="INSUFFICIENT_EVIDENCE",
                    evidence_count=None,
                    success_rate=None,
                    strategies=[],
                )

            # 4. Group by repair_action
            grouped: dict[str, dict[str, list]] = defaultdict(
                lambda: {"outcomes": [], "scores": []}
            )

            for r in records:
                action = r.repair_action.strip()
                grouped[action]["outcomes"].append(r.outcome.strip().upper())
                if r.score is not None:
                    grouped[action]["scores"].append(float(r.score))

            strategies: list[SupportStrategySummary] = []
            for action, data in grouped.items():
                outcomes = data["outcomes"]
                scores = data["scores"]
                obs_count = len(outcomes)
                succ_count = sum(1 for o in outcomes if o in ("RESOLVED", "SUCCESS", "SUCCESSFUL"))
                part_count = sum(1 for o in outcomes if o in ("PARTIALLY_RESOLVED", "PARTIAL"))
                fail_count = sum(1 for o in outcomes if o in ("UNRESOLVED", "FAILED", "FAIL"))
                succ_rate = round(succ_count / obs_count, 2) if obs_count > 0 else 0.0
                avg_score = round(sum(scores) / len(scores), 2) if scores else None

                strategies.append(
                    SupportStrategySummary(
                        repair_action=action,
                        observation_count=obs_count,
                        successful_count=succ_count,
                        partial_count=part_count,
                        failed_count=fail_count,
                        success_rate=succ_rate,
                        average_score=avg_score,
                    )
                )

            # 5. Evaluate Preference Evidence
            candidates = [
                s for s in strategies if s.observation_count >= MIN_OBSERVATIONS_FOR_PREFERENCE
            ]

            preferred_style: str | None = None
            evidence_status = "INSUFFICIENT_EVIDENCE"
            evidence_count: int | None = None
            success_rate: float | None = None

            if candidates:
                # Rank candidates: higher success rate > higher average score > higher observations
                candidates.sort(
                    key=lambda s: (
                        s.success_rate,
                        s.average_score if s.average_score is not None else -1.0,
                        s.observation_count,
                    ),
                    reverse=True,
                )
                top = candidates[0]
                preferred_style = top.repair_action
                evidence_status = "SUPPORTED_BY_HISTORY"
                evidence_count = top.observation_count
                success_rate = top.success_rate

            # Sort strategies for clean display (success_rate desc, observation_count desc)
            strategies.sort(
                key=lambda s: (s.success_rate, s.observation_count),
                reverse=True,
            )

            return SupportPreferenceResponse(
                student_id=student_id,
                skill_id=str(skill.skill_id) if skill else None,
                preferred_support_style=preferred_style,
                status=evidence_status,
                evidence_count=evidence_count,
                success_rate=success_rate,
                strategies=strategies,
            )


# Singleton factory cache
_SUPPORT_PREFERENCE_SERVICE: SupportPreferenceService | None = None


def get_support_preference_service(
    session_factory: SessionFactory | None = None,
) -> SupportPreferenceService:
    global _SUPPORT_PREFERENCE_SERVICE
    if _SUPPORT_PREFERENCE_SERVICE is None or session_factory is not None:
        _SUPPORT_PREFERENCE_SERVICE = SupportPreferenceService(
            session_factory=session_factory
        )
    return _SUPPORT_PREFERENCE_SERVICE
