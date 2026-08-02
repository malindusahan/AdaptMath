"""SQLAlchemy repository for append-only raw interaction evidence."""

from __future__ import annotations

from collections.abc import Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.raw_interaction import InteractionLog
from src.schemas.raw_interaction import RawInteractionCreate, RawInteractionRecord


class RawInteractionRepository:
    """Persist and chronologically retrieve raw interaction records."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, interaction: RawInteractionCreate) -> RawInteractionRecord:
        values = interaction.model_dump(exclude={"created_at"})
        model = InteractionLog(
            interaction_id=uuid.uuid4(),
            **values,
        )
        if interaction.created_at is not None:
            model.created_at = interaction.created_at

        self.session.add(model)
        self.session.flush()
        self.session.refresh(model)
        return RawInteractionRecord.model_validate(model)

    def add_many(
        self,
        interactions: Sequence[RawInteractionCreate],
    ) -> list[RawInteractionRecord]:
        return [self.add(interaction) for interaction in interactions]

    def get_by_id(
        self,
        interaction_id: uuid.UUID,
    ) -> RawInteractionRecord | None:
        model = self.session.get(InteractionLog, interaction_id)
        return (
            RawInteractionRecord.model_validate(model)
            if model is not None
            else None
        )

    def get_student_interactions(
        self,
        student_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        return self._records(
            select(InteractionLog)
            .where(InteractionLog.student_id == student_id)
            .order_by(InteractionLog.created_at, InteractionLog.interaction_id)
        )

    def get_session_interactions(
        self,
        session_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        return self._records(
            select(InteractionLog)
            .where(InteractionLog.session_id == session_id)
            .order_by(InteractionLog.created_at, InteractionLog.interaction_id)
        )

    def get_student_skill_interactions(
        self,
        student_id: uuid.UUID,
        canonical_skill_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        return self._records(
            select(InteractionLog)
            .where(
                InteractionLog.student_id == student_id,
                InteractionLog.canonical_skill_id == canonical_skill_id,
            )
            .order_by(InteractionLog.created_at, InteractionLog.interaction_id)
        )

    def _records(self, statement) -> list[RawInteractionRecord]:
        models = self.session.scalars(statement).all()
        return [RawInteractionRecord.model_validate(model) for model in models]

