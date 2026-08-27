"""Transaction-safe application service for raw interaction evidence."""

from __future__ import annotations

from collections.abc import Sequence
import uuid

from src.database.postgres_session import SessionFactory, session_scope
from src.database.repositories.raw_interaction_repository import (
    RawInteractionRepository,
)
from src.schemas.raw_interaction import RawInteractionCreate, RawInteractionRecord


class RawInteractionService:
    """Append and query source-of-truth interactions through SQLAlchemy."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory

    def insert(self, interaction: RawInteractionCreate) -> RawInteractionRecord:
        with session_scope(self.session_factory) as session:
            return RawInteractionRepository(session).add(interaction)

    def insert_many(
        self,
        interactions: Sequence[RawInteractionCreate],
    ) -> list[RawInteractionRecord]:
        with session_scope(self.session_factory) as session:
            return RawInteractionRepository(session).add_many(interactions)

    def get_by_id(
        self,
        interaction_id: uuid.UUID,
    ) -> RawInteractionRecord | None:
        with self.session_factory() as session:
            return RawInteractionRepository(session).get_by_id(interaction_id)

    def get_student_interactions(
        self,
        student_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        with self.session_factory() as session:
            return RawInteractionRepository(session).get_student_interactions(
                student_id
            )

    def get_session_interactions(
        self,
        session_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        with self.session_factory() as session:
            return RawInteractionRepository(session).get_session_interactions(
                session_id
            )

    def get_student_skill_interactions(
        self,
        student_id: uuid.UUID,
        canonical_skill_id: uuid.UUID,
    ) -> list[RawInteractionRecord]:
        with self.session_factory() as session:
            return RawInteractionRepository(session).get_student_skill_interactions(
                student_id,
                canonical_skill_id,
            )

