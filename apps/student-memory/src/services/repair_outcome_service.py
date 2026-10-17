"""Repair Outcome Service for persisting and validating repair events."""

from __future__ import annotations

import uuid
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import RepairOutcome
from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.repair_outcome import (
    RepairOutcomeCreateRequest,
    RepairOutcomeResponse,
)


class RepairOutcomeServiceError(Exception):
    """Custom exception raised during repair outcome recording."""


class RepairOutcomeService:
    """Handles validation, idempotency, and transactional persistence of repair outcome logs."""

    def __init__(self, session_factory: SessionFactory | None = None):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )

    def record_repair_outcome(
        self,
        request: RepairOutcomeCreateRequest,
    ) -> RepairOutcomeResponse:
        """
        Validate and store a repair outcome event:
        1. Confirm student exists.
        2. Confirm session exists and belongs to the student.
        3. Confirm canonical skill exists.
        4. If interaction_id is provided, confirm it matches student, session, and skill.
        5. Check for duplicate/retry event; if duplicate exists, return existing row.
        6. Append new RepairOutcome row and commit atomically.
        """
        with UnitOfWork(self.session_factory) as uow:
            assert uow.session is not None

            # 1. Resolve Student
            student_raw = request.student_id.strip()
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

            if student is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Student '{request.student_id}' not found.",
                )

            # 2. Resolve LearningSession
            session_raw = request.session_id.strip()
            learning_session: LearningSession | None = None
            try:
                sess_uuid = uuid.UUID(session_raw)
                learning_session = uow.session.scalar(
                    select(LearningSession).where(
                        (
                            (LearningSession.session_id == sess_uuid)
                            | (LearningSession.external_session_id == session_raw)
                        ),
                    )
                )
            except ValueError:
                learning_session = uow.session.scalar(
                    select(LearningSession).where(
                        LearningSession.external_session_id == session_raw,
                    )
                )

            if learning_session is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Learning session '{request.session_id}' not found.",
                )

            # Validate Session ownership
            if learning_session.student_id != student.student_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Learning session '{request.session_id}' does not belong to student '{request.student_id}'.",
                )

            # 3. Resolve CanonicalSkill
            skill_raw = request.skill_id.strip()
            skill: CanonicalSkill | None = None
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
                    detail=f"Canonical skill '{request.skill_id}' not found.",
                )

            # 4. Resolve & Validate Interaction (if provided)
            interaction_uuid: uuid.UUID | None = None
            if request.interaction_id and request.interaction_id.strip():
                inter_raw = request.interaction_id.strip()
                try:
                    interaction_uuid = uuid.UUID(inter_raw)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid interaction_id format: '{request.interaction_id}'.",
                    )

                interaction = uow.session.get(InteractionLog, interaction_uuid)
                if interaction is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Interaction '{request.interaction_id}' not found.",
                    )

                if interaction.student_id != student.student_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Interaction '{request.interaction_id}' does not belong to student '{request.student_id}'.",
                    )

                if interaction.session_id != learning_session.session_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Interaction '{request.interaction_id}' does not belong to session '{request.session_id}'.",
                    )

                if (
                    interaction.canonical_skill_id is not None
                    and interaction.canonical_skill_id != skill.skill_id
                ):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Interaction '{request.interaction_id}' does not match canonical skill '{request.skill_id}'.",
                    )

            # 5. Idempotency Check / Duplicate Prevention
            existing_repair = uow.session.scalar(
                select(RepairOutcome).where(
                    RepairOutcome.student_id == student.student_id,
                    RepairOutcome.session_id == learning_session.session_id,
                    RepairOutcome.canonical_skill_id == skill.skill_id,
                    RepairOutcome.interaction_id == interaction_uuid,
                    RepairOutcome.repair_action == request.repair_action.strip(),
                    RepairOutcome.outcome == request.outcome.strip(),
                )
            )

            if existing_repair is not None:
                return RepairOutcomeResponse(
                    repair_outcome_id=str(existing_repair.repair_outcome_id),
                    student_id=str(student.external_student_id or student.student_id),
                    session_id=str(learning_session.external_session_id or learning_session.session_id),
                    canonical_skill_id=str(skill.skill_id),
                    interaction_id=str(existing_repair.interaction_id) if existing_repair.interaction_id else None,
                    repair_action=existing_repair.repair_action,
                    outcome=existing_repair.outcome,
                    score=existing_repair.score,
                    notes=existing_repair.notes,
                    created_at=(
                        existing_repair.created_at.isoformat()
                        if hasattr(existing_repair.created_at, "isoformat")
                        else str(existing_repair.created_at)
                    ),
                )

            # 6. Insert new RepairOutcome
            try:
                new_repair = uow.repairs.add(
                    student_id=student.student_id,
                    session_id=learning_session.session_id,
                    canonical_skill_id=skill.skill_id,
                    interaction_id=interaction_uuid,
                    repair_action=request.repair_action.strip(),
                    outcome=request.outcome.strip(),
                    score=request.score,
                    notes=request.notes.strip() if request.notes else None,
                )
                uow.commit()

                return RepairOutcomeResponse(
                    repair_outcome_id=str(new_repair.repair_outcome_id),
                    student_id=str(student.external_student_id or student.student_id),
                    session_id=str(learning_session.external_session_id or learning_session.session_id),
                    canonical_skill_id=str(skill.skill_id),
                    interaction_id=str(new_repair.interaction_id) if new_repair.interaction_id else None,
                    repair_action=new_repair.repair_action,
                    outcome=new_repair.outcome,
                    score=new_repair.score,
                    notes=new_repair.notes,
                    created_at=(
                        new_repair.created_at.isoformat()
                        if hasattr(new_repair.created_at, "isoformat")
                        else str(new_repair.created_at)
                    ),
                )
            except IntegrityError:
                uow.rollback()
                existing_repair = uow.session.scalar(
                    select(RepairOutcome).where(
                        RepairOutcome.student_id == student.student_id,
                        RepairOutcome.session_id == learning_session.session_id,
                        RepairOutcome.canonical_skill_id == skill.skill_id,
                        RepairOutcome.interaction_id == interaction_uuid,
                        RepairOutcome.repair_action == request.repair_action.strip(),
                        RepairOutcome.outcome == request.outcome.strip(),
                    )
                )
                if existing_repair is None:
                    raise
                return RepairOutcomeResponse(
                    repair_outcome_id=str(existing_repair.repair_outcome_id),
                    student_id=str(student.external_student_id or student.student_id),
                    session_id=str(learning_session.external_session_id),
                    canonical_skill_id=str(skill.skill_id),
                    interaction_id=(
                        str(existing_repair.interaction_id)
                        if existing_repair.interaction_id
                        else None
                    ),
                    repair_action=existing_repair.repair_action,
                    outcome=existing_repair.outcome,
                    score=existing_repair.score,
                    notes=existing_repair.notes,
                    created_at=(
                        existing_repair.created_at.isoformat()
                        if hasattr(existing_repair.created_at, "isoformat")
                        else str(existing_repair.created_at)
                    ),
                )


# Singleton factory cache
_REPAIR_OUTCOME_SERVICE: RepairOutcomeService | None = None


def get_repair_outcome_service(
    session_factory: SessionFactory | None = None,
) -> RepairOutcomeService:
    global _REPAIR_OUTCOME_SERVICE
    if _REPAIR_OUTCOME_SERVICE is None or session_factory is not None:
        _REPAIR_OUTCOME_SERVICE = RepairOutcomeService(session_factory=session_factory)
    return _REPAIR_OUTCOME_SERVICE
