"""Repository for durable completed-attempt ingestion receipts."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.completed_attempt import CompletedAttemptReceipt


class CompletedAttemptRepository:
    """Persist and retrieve exactly-once external attempt receipts."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_external_key(
        self,
        source: str,
        external_attempt_id: str,
    ) -> CompletedAttemptReceipt | None:
        return self.session.scalar(
            select(CompletedAttemptReceipt).where(
                CompletedAttemptReceipt.source == source,
                CompletedAttemptReceipt.external_attempt_id == external_attempt_id,
            )
        )

    def add(
        self,
        *,
        student_id: uuid.UUID,
        session_id: uuid.UUID,
        canonical_skill_id: uuid.UUID,
        source: str,
        external_attempt_id: str,
        payload_hash: str,
        session_status: str,
        question_count: int,
        correct_count: int,
        incorrect_count: int,
    ) -> CompletedAttemptReceipt:
        receipt = CompletedAttemptReceipt(
            receipt_id=uuid.uuid4(),
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=canonical_skill_id,
            source=source,
            external_attempt_id=external_attempt_id,
            payload_hash=payload_hash,
            status="INGESTED",
            session_status=session_status,
            question_count=question_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
        )
        self.session.add(receipt)
        self.session.flush()
        self.session.refresh(receipt)
        return receipt
