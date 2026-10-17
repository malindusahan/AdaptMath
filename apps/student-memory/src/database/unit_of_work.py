"""PostgreSQL Unit of Work pattern for managing transactional boundaries."""

from __future__ import annotations

from typing import Protocol
from sqlalchemy.orm import Session

from src.database.postgres_session import SessionFactory
from src.database.repositories.identity_repository import IdentityRepository
from src.database.repositories.completed_attempt_repository import (
    CompletedAttemptRepository,
)
from src.database.repositories.memory_projection_repository import (
    MemoryProjectionRepository,
)
from src.database.repositories.raw_interaction_repository import (
    RawInteractionRepository,
)
from src.database.repositories.repair_outcome_repository import (
    RepairOutcomeRepository,
)
from src.database.repositories.supporting_memory_repository import (
    SupportingMemoryRepository,
)
from src.database.repositories.topic_audit_repository import TopicAuditRepository


class UnitOfWork:
    """Context manager owning a single database session and transaction."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory
        self.session: Session | None = None
        self.identities: IdentityRepository | None = None
        self.completed_attempts: CompletedAttemptRepository | None = None
        self.raw_interactions: RawInteractionRepository | None = None
        self.projections: MemoryProjectionRepository | None = None
        self.supporting: SupportingMemoryRepository | None = None
        self.repairs: RepairOutcomeRepository | None = None
        self.topic_audits: TopicAuditRepository | None = None

    def __enter__(self) -> UnitOfWork:
        self.session = self.session_factory()
        self.identities = IdentityRepository(self.session)
        self.completed_attempts = CompletedAttemptRepository(self.session)
        self.raw_interactions = RawInteractionRepository(self.session)
        self.projections = MemoryProjectionRepository(self.session)
        self.supporting = SupportingMemoryRepository(self.session)
        self.repairs = RepairOutcomeRepository(self.session)
        self.topic_audits = TopicAuditRepository(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session is not None:
            if exc_type is not None:
                self.session.rollback()
            self.session.close()

    def commit(self) -> None:
        """Explicitly commit the current session transaction."""
        if self.session is not None:
            self.session.commit()

    def rollback(self) -> None:
        """Explicitly roll back the current session transaction."""
        if self.session is not None:
            self.session.rollback()

    def flush(self) -> None:
        """Flush pending changes to the database without committing."""
        if self.session is not None:
            self.session.flush()
