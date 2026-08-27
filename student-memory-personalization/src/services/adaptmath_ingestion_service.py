"""Transactional, idempotent ingestion of completed AdaptMath attempts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

from sqlalchemy.exc import IntegrityError

from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.adaptmath import (
    AdaptMathCompletedAttemptRequest,
    AdaptMathCompletedAttemptResponse,
)
from src.schemas.raw_interaction import RawInteractionCreate
from src.services.memory_aggregation_service import MemoryAggregationService
from src.services.skill_context_resolver import SkillContextResolver


class AdaptMathIngestionError(RuntimeError):
    """Base error for the completed-attempt ingestion boundary."""


class AdaptMathAttemptConflictError(AdaptMathIngestionError):
    """The stable attempt key already exists with different content."""


class AdaptMathSkillNotFoundError(AdaptMathIngestionError):
    """The requested BKT skill is absent from the Memory ontology."""


def _payload_hash(request: AdaptMathCompletedAttemptRequest) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AdaptMathIngestionService:
    """Store objective evidence without invoking Memory's alternate state model."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        aggregation_service: MemoryAggregationService | None = None,
    ):
        self.session_factory = session_factory or get_session_factory()
        self.aggregation_service = aggregation_service or MemoryAggregationService()

    @staticmethod
    def _response(request, receipt, status_value):
        return AdaptMathCompletedAttemptResponse(
            receipt_id=str(receipt.receipt_id),
            external_student_id=request.external_student_id,
            external_session_id=request.external_session_id,
            external_attempt_id=request.external_attempt_id,
            canonical_skill_id=str(receipt.canonical_skill_id),
            source="adaptmath",
            status=status_value,
            session_status=receipt.session_status,
            question_count=receipt.question_count,
            correct_count=receipt.correct_count,
            incorrect_count=receipt.incorrect_count,
            created_at=(
                receipt.created_at.isoformat()
                if hasattr(receipt.created_at, "isoformat")
                else str(receipt.created_at)
            ),
        )

    def _existing_response(self, request, digest):
        with UnitOfWork(self.session_factory) as uow:
            existing = uow.completed_attempts.get_by_external_key(
                request.source,
                request.external_attempt_id,
            )
            if existing is None:
                return None
            if existing.payload_hash != digest:
                raise AdaptMathAttemptConflictError(
                    "The external attempt ID is already bound to different evidence."
                )
            return self._response(request, existing, "ALREADY_STORED")

    def ingest_completed_attempt(
        self,
        request: AdaptMathCompletedAttemptRequest,
    ) -> AdaptMathCompletedAttemptResponse:
        digest = _payload_hash(request)

        try:
            with UnitOfWork(self.session_factory) as uow:
                assert uow.session is not None
                uow.identities.lock_external_attempt(
                    request.source,
                    request.external_attempt_id,
                )

                existing = uow.completed_attempts.get_by_external_key(
                    request.source,
                    request.external_attempt_id,
                )
                if existing is not None:
                    if existing.payload_hash != digest:
                        raise AdaptMathAttemptConflictError(
                            "The external attempt ID is already bound to different evidence."
                        )
                    return self._response(request, existing, "ALREADY_STORED")

                uow.identities.lock_student_update(request.external_student_id)
                student = uow.identities.get_or_create_student(
                    request.external_student_id
                )
                skill = SkillContextResolver.resolve_skill(
                    session=uow.session,
                    topic=request.target_skill,
                    subtopic=None,
                )
                if skill is None:
                    raise AdaptMathSkillNotFoundError(
                        f"Unknown canonical skill: {request.target_skill}"
                    )

                learning_session = uow.identities.get_or_create_session(
                    student,
                    request.external_session_id,
                )
                now = datetime.now(timezone.utc)
                if request.session_status == "COMPLETED":
                    learning_session.status = "COMPLETED"
                    learning_session.ended_at = learning_session.ended_at or now
                elif learning_session.status != "COMPLETED":
                    learning_session.status = "ACTIVE"
                    learning_session.ended_at = None
                learning_session.updated_at = now

                correct_count = sum(item.is_correct for item in request.questions)
                receipt = uow.completed_attempts.add(
                    student_id=student.student_id,
                    session_id=learning_session.session_id,
                    canonical_skill_id=skill.skill_id,
                    source=request.source,
                    external_attempt_id=request.external_attempt_id,
                    payload_hash=digest,
                    session_status=request.session_status,
                    question_count=len(request.questions),
                    correct_count=correct_count,
                    incorrect_count=len(request.questions) - correct_count,
                )

                for item in request.questions:
                    uow.raw_interactions.add(
                        RawInteractionCreate(
                            student_id=student.student_id,
                            session_id=learning_session.session_id,
                            canonical_skill_id=skill.skill_id,
                            source=request.source,
                            external_interaction_id=item.external_interaction_id,
                            problem_id=item.problem_id,
                            question_id=item.question_id,
                            student_answer=item.student_answer,
                            expected_answer=item.expected_answer,
                            is_correct=item.is_correct,
                            identified_error=item.identified_error,
                            attempt_count=None,
                            hint_count=None,
                            hint_total=None,
                            response_time_ms=None,
                        )
                    )
                    if item.identified_error:
                        uow.supporting.upsert_misconception(
                            student_id=student.student_id,
                            skill_id=skill.skill_id,
                            text=item.identified_error,
                        )

                self.aggregation_service.rebuild_student_projections(
                    student_id=student.student_id,
                    uow=uow,
                )
                uow.commit()
                return self._response(request, receipt, "STORED")
        except (AdaptMathAttemptConflictError, AdaptMathSkillNotFoundError):
            raise
        except IntegrityError as exc:
            replay = self._existing_response(request, digest)
            if replay is not None:
                return replay
            raise AdaptMathIngestionError(
                "Completed attempt could not be stored transactionally."
            ) from exc
        except Exception as exc:
            raise AdaptMathIngestionError(
                "Completed attempt could not be stored transactionally."
            ) from exc


_SERVICE: AdaptMathIngestionService | None = None


def get_adaptmath_ingestion_service() -> AdaptMathIngestionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AdaptMathIngestionService()
    return _SERVICE
