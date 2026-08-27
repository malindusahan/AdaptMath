"""Repositories for misconceptions and persisted learning states."""

from __future__ import annotations

import uuid
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models.core import CanonicalSkill, Student
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    StudentMisconception,
)


def normalize_error(value: str) -> str:
    return " ".join(value.strip().split()).lower()


class SupportingMemoryRepository:
    """PostgreSQL repository for misconceptions, learning state snapshots, and current state."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_misconception(
        self,
        student_id: uuid.UUID,
        skill_id: uuid.UUID,
        text: str,
        student_external_id: str | None = None,
        skill_name: str | None = None,
    ) -> StudentMisconception:
        if student_external_id is None:
            student = self.session.get(Student, student_id)
            student_external_id = student.external_student_id if student else None
        if skill_name is None:
            skill = self.session.get(CanonicalSkill, skill_id)
            skill_name = skill.display_name if skill else None

        normalized = normalize_error(text)
        stmt = (
            insert(StudentMisconception)
            .values(
                misconception_id=uuid.uuid4(),
                student_id=student_id,
                student_external_id=student_external_id,
                canonical_skill_id=skill_id,
                skill_name=skill_name,
                normalized_error=normalized,
                display_error=" ".join(text.strip().split()),
                occurrence_count=1,
            )
            .on_conflict_do_update(
                index_elements=[
                    "student_id",
                    "canonical_skill_id",
                    "normalized_error",
                ],
                set_={
                    "occurrence_count": StudentMisconception.occurrence_count + 1,
                    "student_external_id": student_external_id,
                    "skill_name": skill_name,
                    "last_seen_at": func.now(),
                },
            )
            .returning(StudentMisconception)
            .execution_options(populate_existing=True)
        )
        return self.session.scalars(stmt).one()

    def get_misconceptions(
        self,
        student_id: uuid.UUID,
        skill_id: uuid.UUID | None = None,
    ) -> list[StudentMisconception]:
        stmt = select(StudentMisconception).where(StudentMisconception.student_id == student_id)
        if skill_id is not None:
            stmt = stmt.where(StudentMisconception.canonical_skill_id == skill_id)
        stmt = stmt.order_by(
            StudentMisconception.occurrence_count.desc(),
            StudentMisconception.last_seen_at.desc(),
        )
        return list(self.session.scalars(stmt))

    def add_snapshot(self, **values) -> LearningStateSnapshot:
        if "student_external_id" not in values or values["student_external_id"] is None:
            student_id = values.get("student_id")
            if student_id:
                student = self.session.get(Student, student_id)
                values["student_external_id"] = student.external_student_id if student else None

        if "skill_name" not in values or values["skill_name"] is None:
            skill_id = values.get("canonical_skill_id")
            if skill_id:
                skill = self.session.get(CanonicalSkill, skill_id)
                values["skill_name"] = skill.display_name if skill else None

        row = LearningStateSnapshot(**values)
        self.session.add(row)
        self.session.flush()
        self.session.refresh(row)
        return row

    def upsert_current(
        self,
        snapshot: LearningStateSnapshot,
    ) -> CurrentLearningState:
        values = {
            k: getattr(snapshot, k)
            for k in (
                "student_id",
                "student_external_id",
                "canonical_skill_id",
                "skill_name",
                "learning_state",
                "evidence_level",
                "evidence_strength",
                "behavioural_coverage",
                "model_used",
                "recent_interaction_count",
                "attempt_observation_count",
                "hint_observation_count",
                "response_time_observation_count",
            )
        }
        values.update(
            last_assessment_id=snapshot.assessment_id,
            last_snapshot_id=snapshot.snapshot_id,
        )
        stmt = (
            insert(CurrentLearningState)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["student_id", "canonical_skill_id"],
                set_={**values, "updated_at": func.now()},
            )
            .returning(CurrentLearningState)
            .execution_options(populate_existing=True)
        )
        return self.session.scalars(stmt).one()

    def get_current(
        self,
        student_id: uuid.UUID,
        skill_id: uuid.UUID | None = None,
    ) -> CurrentLearningState | None:
        if skill_id is not None:
            return self.session.get(CurrentLearningState, (student_id, skill_id))
        return self.session.scalar(
            select(CurrentLearningState)
            .where(CurrentLearningState.student_id == student_id)
            .order_by(CurrentLearningState.updated_at.desc())
        )

    def get_history(
        self,
        student_id: uuid.UUID,
        skill_id: uuid.UUID,
    ) -> list[LearningStateSnapshot]:
        return list(
            self.session.scalars(
                select(LearningStateSnapshot)
                .where(
                    LearningStateSnapshot.student_id == student_id,
                    LearningStateSnapshot.canonical_skill_id == skill_id,
                )
                .order_by(
                    LearningStateSnapshot.created_at,
                    LearningStateSnapshot.snapshot_id,
                )
            )
        )
