"""Comprehensive tests for the PostgreSQL Aggregation, Transaction Engine, and Memory Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
import uuid

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import text

from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    StudentMisconception,
)
from src.database.postgres_config import (
    DATABASE_URL_ENV,
    DEFAULT_POSTGRES_SCHEMA,
    load_postgres_settings,
)
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
    session_scope,
)
from src.database.unit_of_work import UnitOfWork
from src.schemas.memory_projection import (
    ConceptMemoryRecord,
    ConceptMemoryUpsert,
    LongTermMemoryRecord,
    LongTermMemoryUpsert,
    ShortTermMemoryRecord,
    ShortTermMemoryUpsert,
)
from src.schemas.raw_interaction import RawInteractionCreate, RawInteractionRecord
from src.services.memory_aggregation_service import MemoryAggregationService
from src.services.postgres_memory_update_service import (
    HistoricalInteractionAdapter,
    PostgresMemoryUpdateError,
    PostgresMemoryUpdateService,
    _to_historical_adapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA


# =============================================================================
# Pure Unit Tests for MemoryAggregationService
# =============================================================================

def test_aggregation_missing_vs_real_zero_behavioural_values():
    """Verify that None (unavailable) and 0 (observed zero) are kept distinct."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    # Interaction with missing behavioural values (None)
    missing_interaction = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="test_evaluator",
        external_interaction_id="inter_none",
        is_correct=True,
        attempt_count=None,
        hint_count=None,
        hint_total=None,
        response_time_ms=None,
    )

    stm_none = MemoryAggregationService.aggregate_short_term_memory(
        student_id=student_id,
        session_id=session_id,
        interactions=[missing_interaction],
    )
    assert stm_none.interaction_count == 1
    assert stm_none.correct_count == 1
    assert stm_none.incorrect_count == 0
    assert stm_none.recent_accuracy == 1.0
    assert stm_none.attempt_observation_count == 0
    assert stm_none.attempt_sum == 0
    assert stm_none.hint_observation_count == 0
    assert stm_none.hint_sum == 0
    assert stm_none.response_time_observation_count == 0
    assert stm_none.response_time_sum_ms == 0.0

    # Interaction with real zero behavioural values (0)
    zero_interaction = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="test_evaluator",
        external_interaction_id="inter_zero",
        is_correct=False,
        attempt_count=0,
        hint_count=0,
        hint_total=3,
        response_time_ms=0.0,
    )

    stm_zero = MemoryAggregationService.aggregate_short_term_memory(
        student_id=student_id,
        session_id=session_id,
        interactions=[zero_interaction],
    )
    assert stm_zero.interaction_count == 1
    assert stm_zero.correct_count == 0
    assert stm_zero.incorrect_count == 1
    assert stm_zero.recent_accuracy == 0.0
    assert stm_zero.attempt_observation_count == 1
    assert stm_zero.attempt_sum == 0
    assert stm_zero.hint_observation_count == 1
    assert stm_zero.hint_sum == 0
    assert stm_zero.response_time_observation_count == 1
    assert stm_zero.response_time_sum_ms == 0.0


def test_aggregation_multi_session_and_multi_skill_calculations():
    """Verify LTM correctly computes distinct session count and distinct concept count."""
    student_id = uuid.uuid4()
    session_1 = uuid.uuid4()
    session_2 = uuid.uuid4()
    skill_1 = uuid.uuid4()
    skill_2 = uuid.uuid4()

    interactions = [
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_1,
            canonical_skill_id=skill_1,
            source="test",
            external_interaction_id="i1",
            is_correct=True,
            attempt_count=1,
            response_time_ms=1000.0,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_1,
            canonical_skill_id=skill_2,
            source="test",
            external_interaction_id="i2",
            is_correct=False,
            attempt_count=2,
            response_time_ms=2000.0,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_2,
            canonical_skill_id=skill_1,
            source="test",
            external_interaction_id="i3",
            is_correct=True,
            attempt_count=1,
            response_time_ms=1500.0,
        ),
    ]

    ltm = MemoryAggregationService.aggregate_long_term_memory(
        student_id=student_id,
        interactions=interactions,
    )
    assert ltm.interaction_count == 3
    assert ltm.total_sessions == 2
    assert ltm.concept_count == 2
    assert ltm.correct_count == 2
    assert ltm.incorrect_count == 1
    assert pytest.approx(ltm.overall_accuracy) == 2 / 3
    assert ltm.attempt_observation_count == 3
    assert ltm.attempt_sum == 4
    assert ltm.response_time_observation_count == 3
    assert ltm.response_time_sum_ms == 4500.0


def test_concept_memory_aggregation():
    """Verify Concept Memory correctly computes skill-specific accuracy and timestamps."""
    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    t1 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 1, 10, 5, 0, tzinfo=timezone.utc)

    interactions = [
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="test",
            external_interaction_id="i1",
            is_correct=True,
            attempt_count=1,
            response_time_ms=5000.0,
            created_at=t1,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="test",
            external_interaction_id="i2",
            is_correct=True,
            attempt_count=1,
            response_time_ms=4000.0,
            created_at=t2,
        ),
    ]

    concept = MemoryAggregationService.aggregate_concept_memory(
        student_id=student_id,
        canonical_skill_id=skill_id,
        interactions=interactions,
    )
    assert concept.interaction_count == 2
    assert concept.correct_count == 2
    assert concept.incorrect_count == 0
    assert concept.accuracy == 1.0
    assert concept.first_interaction_at == t1
    assert concept.last_interaction_at == t2
    assert concept.attempt_observation_count == 2
    assert concept.attempt_sum == 2
    assert concept.response_time_sum_ms == 9000.0


def test_historical_interaction_adapter():
    """Verify conversion to adapter for production feature builder and behavioural coverage."""
    raw = RawInteractionRecord(
        interaction_id=uuid.uuid4(),
        student_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        canonical_skill_id=uuid.uuid4(),
        source="evaluator",
        external_interaction_id="e1",
        problem_id="p1",
        question_id="q1",
        student_utterance=None,
        tutor_response=None,
        student_answer="4",
        expected_answer="4",
        is_correct=True,
        identified_error=None,
        attempt_count=1,
        hint_count=0,
        hint_total=2,
        response_time_ms=3500.0,
        created_at=datetime.now(timezone.utc),
    )

    adapter = _to_historical_adapter(raw)
    assert adapter.is_correct is True
    assert adapter.attempt_data_available is True
    assert adapter.attempt_count == 1
    assert adapter.hint_data_available is True
    assert adapter.hint_count == 0
    assert adapter.hint_total == 2
    assert adapter.response_time_available is True
    assert adapter.response_time_ms == 3500.0


# =============================================================================
# In-Memory / Mock Unit of Work Pipeline Tests
# =============================================================================

class FakeRawRepo:
    def __init__(self):
        self.logs: list[RawInteractionRecord] = []

    def add(self, interaction: RawInteractionCreate) -> RawInteractionRecord:
        record = RawInteractionRecord(
            interaction_id=uuid.uuid4(),
            student_id=interaction.student_id,
            session_id=interaction.session_id,
            canonical_skill_id=interaction.canonical_skill_id,
            source=interaction.source,
            external_interaction_id=interaction.external_interaction_id,
            problem_id=interaction.problem_id,
            question_id=interaction.question_id,
            student_utterance=interaction.student_utterance,
            tutor_response=interaction.tutor_response,
            student_answer=interaction.student_answer,
            expected_answer=interaction.expected_answer,
            is_correct=interaction.is_correct,
            identified_error=interaction.identified_error,
            attempt_count=interaction.attempt_count,
            hint_count=interaction.hint_count,
            hint_total=interaction.hint_total,
            response_time_ms=interaction.response_time_ms,
            created_at=interaction.created_at or datetime.now(timezone.utc),
        )
        self.logs.append(record)
        return record

    def get_student_interactions(self, student_id: uuid.UUID) -> list[RawInteractionRecord]:
        return [item for item in self.logs if item.student_id == student_id]

    def get_session_interactions(self, session_id: uuid.UUID) -> list[RawInteractionRecord]:
        return [item for item in self.logs if item.session_id == session_id]

    def get_student_skill_interactions(
        self, student_id: uuid.UUID, canonical_skill_id: uuid.UUID
    ) -> list[RawInteractionRecord]:
        return [
            item
            for item in self.logs
            if item.student_id == student_id
            and item.canonical_skill_id == canonical_skill_id
        ]


class FakeProjectionRepo:
    def __init__(self):
        self.stm: dict[tuple[uuid.UUID, uuid.UUID], ShortTermMemoryRecord] = {}
        self.ltm: dict[uuid.UUID, LongTermMemoryRecord] = {}
        self.concept: dict[tuple[uuid.UUID, uuid.UUID], ConceptMemoryRecord] = {}

    def get_short_term_memory(self, student_id, session_id):
        return self.stm.get((student_id, session_id))

    def upsert_short_term_memory(self, projection: ShortTermMemoryUpsert) -> ShortTermMemoryRecord:
        record = ShortTermMemoryRecord(
            **projection.model_dump(),
            updated_at=datetime.now(timezone.utc),
        )
        self.stm[(projection.student_id, projection.session_id)] = record
        return record

    def get_long_term_memory(self, student_id):
        return self.ltm.get(student_id)

    def upsert_long_term_memory(self, projection: LongTermMemoryUpsert) -> LongTermMemoryRecord:
        record = LongTermMemoryRecord(
            **projection.model_dump(),
            updated_at=datetime.now(timezone.utc),
        )
        self.ltm[projection.student_id] = record
        return record

    def get_concept_memory(self, student_id, canonical_skill_id):
        return self.concept.get((student_id, canonical_skill_id))

    def upsert_concept_memory(self, projection: ConceptMemoryUpsert) -> ConceptMemoryRecord:
        record = ConceptMemoryRecord(
            **projection.model_dump(),
            updated_at=datetime.now(timezone.utc),
        )
        self.concept[(projection.student_id, projection.canonical_skill_id)] = record
        return record


class FakeSupportingRepo:
    def __init__(self):
        self.misconceptions: dict[tuple[uuid.UUID, uuid.UUID, str], StudentMisconception] = {}
        self.snapshots: list[LearningStateSnapshot] = []
        self.current_state: dict[tuple[uuid.UUID, uuid.UUID], CurrentLearningState] = {}
        self._snapshot_id_seq = 1

    def upsert_misconception(self, student_id, skill_id, text):
        norm = " ".join(text.strip().split()).lower()
        key = (student_id, skill_id, norm)
        if key in self.misconceptions:
            m = self.misconceptions[key]
            m.occurrence_count += 1
            m.last_seen_at = datetime.now(timezone.utc)
            return m

        m = StudentMisconception(
            misconception_id=uuid.uuid4(),
            student_id=student_id,
            canonical_skill_id=skill_id,
            normalized_error=norm,
            display_error=" ".join(text.strip().split()),
            occurrence_count=1,
            first_seen_at=datetime.now(timezone.utc),
            last_seen_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.misconceptions[key] = m
        return m

    def add_snapshot(self, **values):
        snap = LearningStateSnapshot(
            snapshot_id=self._snapshot_id_seq,
            created_at=datetime.now(timezone.utc),
            **values,
        )
        self._snapshot_id_seq += 1
        self.snapshots.append(snap)
        return snap

    def upsert_current(self, snapshot):
        key = (snapshot.student_id, snapshot.canonical_skill_id)
        current = CurrentLearningState(
            student_id=snapshot.student_id,
            canonical_skill_id=snapshot.canonical_skill_id,
            learning_state=snapshot.learning_state,
            evidence_level=snapshot.evidence_level,
            evidence_strength=snapshot.evidence_strength,
            behavioural_coverage=snapshot.behavioural_coverage,
            model_used=snapshot.model_used,
            recent_interaction_count=snapshot.recent_interaction_count,
            attempt_observation_count=snapshot.attempt_observation_count,
            hint_observation_count=snapshot.hint_observation_count,
            response_time_observation_count=snapshot.response_time_observation_count,
            last_assessment_id=snapshot.assessment_id,
            last_snapshot_id=snapshot.snapshot_id,
            updated_at=datetime.now(timezone.utc),
        )
        self.current_state[key] = current
        return current


class FakeUnitOfWork:
    def __init__(self, raw_repo, proj_repo, supp_repo):
        self.raw_interactions = raw_repo
        self.projections = proj_repo
        self.supporting = supp_repo
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rolled_back = True

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_fake_uow_pipeline_execution_and_learning_state():
    """Verify complete pipeline orchestration, prediction, and state persistence with UnitOfWork."""
    raw_repo = FakeRawRepo()
    proj_repo = FakeProjectionRepo()
    supp_repo = FakeSupportingRepo()
    fake_uow = FakeUnitOfWork(raw_repo, proj_repo, supp_repo)

    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    interaction = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="i_1",
        is_correct=False,
        identified_error="sign inversion error",
        attempt_count=2,
        hint_count=1,
        hint_total=2,
        response_time_ms=12500.0,
    )

    # Use service's internal processor directly on fake UoW
    service = PostgresMemoryUpdateService(session_factory=lambda: None)
    result = service._process_single_interaction(interaction, fake_uow)

    assert result.raw_interaction.student_id == student_id
    assert result.short_term_memory.interaction_count == 1
    assert result.short_term_memory.correct_count == 0
    assert result.short_term_memory.incorrect_count == 1
    assert result.short_term_memory.recent_accuracy == 0.0

    assert result.long_term_memory.interaction_count == 1
    assert result.long_term_memory.total_sessions == 1
    assert result.long_term_memory.concept_count == 1

    assert result.concept_memory.interaction_count == 1
    assert result.concept_memory.accuracy == 0.0

    assert result.misconception is not None
    assert result.misconception.occurrence_count == 1
    assert result.misconception.normalized_error == "sign inversion error"

    assert result.snapshot is not None
    assert result.snapshot.learning_state in ("NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE")
    assert result.current_learning_state is not None
    assert result.current_learning_state.learning_state == result.snapshot.learning_state


def test_fake_uow_misconception_increment():
    """Verify misconception occurrences increment upon repeated errors."""
    raw_repo = FakeRawRepo()
    proj_repo = FakeProjectionRepo()
    supp_repo = FakeSupportingRepo()
    fake_uow = FakeUnitOfWork(raw_repo, proj_repo, supp_repo)

    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    service = PostgresMemoryUpdateService(session_factory=lambda: None)

    # Error 1
    i1 = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="i1",
        is_correct=False,
        identified_error="algebra addition error",
    )
    res1 = service._process_single_interaction(i1, fake_uow)
    assert res1.misconception.occurrence_count == 1

    # Error 2 (with different casing and whitespace)
    i2 = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="i2",
        is_correct=False,
        identified_error="  ALGEBRA   addition  ERROR  ",
    )
    res2 = service._process_single_interaction(i2, fake_uow)
    assert res2.misconception.occurrence_count == 2


def test_fake_uow_rebuild_equality():
    """Verify incremental projection equals rebuilt projection exactly."""
    raw_repo = FakeRawRepo()
    proj_repo = FakeProjectionRepo()
    supp_repo = FakeSupportingRepo()
    fake_uow = FakeUnitOfWork(raw_repo, proj_repo, supp_repo)

    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    service = PostgresMemoryUpdateService(session_factory=lambda: None)

    # Feed 3 interactions
    for idx, (corr, att, hints, rt) in enumerate([
        (True, 1, 0, 4000.0),
        (False, 2, 1, 10000.0),
        (True, None, None, 6000.0),
    ]):
        i = RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="evaluator",
            external_interaction_id=f"reb_{idx}",
            is_correct=corr,
            attempt_count=att,
            hint_count=hints,
            hint_total=2 if hints is not None else None,
            response_time_ms=rt,
        )
        service._process_single_interaction(i, fake_uow)

    incremental_stm = proj_repo.get_short_term_memory(student_id, session_id)
    incremental_ltm = proj_repo.get_long_term_memory(student_id)
    incremental_concept = proj_repo.get_concept_memory(student_id, skill_id)

    # Rebuild
    rebuilt = service.aggregation_service.rebuild_student_projections(student_id, fake_uow)

    rebuilt_stm = proj_repo.get_short_term_memory(student_id, session_id)
    rebuilt_ltm = proj_repo.get_long_term_memory(student_id)
    rebuilt_concept = proj_repo.get_concept_memory(student_id, skill_id)

    assert incremental_stm.interaction_count == rebuilt_stm.interaction_count == 3
    assert incremental_stm.correct_count == rebuilt_stm.correct_count == 2
    assert incremental_stm.incorrect_count == rebuilt_stm.incorrect_count == 1
    assert incremental_stm.recent_accuracy == rebuilt_stm.recent_accuracy
    assert incremental_stm.attempt_observation_count == rebuilt_stm.attempt_observation_count == 2
    assert incremental_stm.attempt_sum == rebuilt_stm.attempt_sum == 3

    assert incremental_ltm.total_sessions == rebuilt_ltm.total_sessions == 1
    assert incremental_ltm.concept_count == rebuilt_ltm.concept_count == 1
    assert incremental_ltm.interaction_count == rebuilt_ltm.interaction_count == 3

    assert incremental_concept.interaction_count == rebuilt_concept.interaction_count == 3
    assert incremental_concept.accuracy == rebuilt_concept.accuracy


def test_fake_uow_complete_rollback_on_forced_failure(monkeypatch):
    """Verify that if an error occurs during update_interactions, UoW rolls back and raises PostgresMemoryUpdateError."""
    raw_repo = FakeRawRepo()
    proj_repo = FakeProjectionRepo()
    supp_repo = FakeSupportingRepo()
    fake_uow = FakeUnitOfWork(raw_repo, proj_repo, supp_repo)

    # Monkeypatch UnitOfWork constructor to return our fake_uow
    import src.services.postgres_memory_update_service as service_mod

    monkeypatch.setattr(service_mod, "UnitOfWork", lambda factory: fake_uow)

    service = PostgresMemoryUpdateService(session_factory=lambda: None)

    student_id = uuid.uuid4()
    session_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    # Create an interaction that triggers an error during processing
    bad_interaction = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="bad_1",
        is_correct=True,
    )

    # Inject an intentional failure in aggregation service
    def failing_update(*args, **kwargs):
        raise RuntimeError("Simulated database failure mid-transaction")

    monkeypatch.setattr(
        service.aggregation_service,
        "update_projections_for_interaction",
        failing_update,
    )

    with pytest.raises(PostgresMemoryUpdateError) as exc_info:
        service.update_interaction(bad_interaction)

    assert "Simulated database failure mid-transaction" in str(exc_info.value)
    assert fake_uow.rolled_back is True
    assert fake_uow.committed is False


def test_fake_uow_student_isolation():
    """Verify students are strictly isolated across all projection and misconception stores."""
    raw_repo = FakeRawRepo()
    proj_repo = FakeProjectionRepo()
    supp_repo = FakeSupportingRepo()
    fake_uow = FakeUnitOfWork(raw_repo, proj_repo, supp_repo)

    service = PostgresMemoryUpdateService(session_factory=lambda: None)

    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    s1_session = uuid.uuid4()
    s2_session = uuid.uuid4()
    skill_id = uuid.uuid4()

    # S1 interaction (correct)
    i1 = RawInteractionCreate(
        student_id=s1_id,
        session_id=s1_session,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="s1_i1",
        is_correct=True,
        identified_error="error_s1",
    )
    service._process_single_interaction(i1, fake_uow)

    # S2 interaction (incorrect)
    i2 = RawInteractionCreate(
        student_id=s2_id,
        session_id=s2_session,
        canonical_skill_id=skill_id,
        source="evaluator",
        external_interaction_id="s2_i1",
        is_correct=False,
        identified_error="error_s2",
    )
    service._process_single_interaction(i2, fake_uow)

    # Assert S1 projections
    s1_stm = proj_repo.get_short_term_memory(s1_id, s1_session)
    assert s1_stm.correct_count == 1
    assert s1_stm.incorrect_count == 0
    assert s1_stm.recent_accuracy == 1.0

    # Assert S2 projections
    s2_stm = proj_repo.get_short_term_memory(s2_id, s2_session)
    assert s2_stm.correct_count == 0
    assert s2_stm.incorrect_count == 1
    assert s2_stm.recent_accuracy == 0.0

    # Assert misconceptions are separated by student
    assert (s1_id, skill_id, "error_s1") in supp_repo.misconceptions
    assert (s2_id, skill_id, "error_s2") in supp_repo.misconceptions
    assert (s1_id, skill_id, "error_s2") not in supp_repo.misconceptions


# =============================================================================
# Live PostgreSQL Pipeline Integration Tests
# =============================================================================

@pytest.fixture(scope="module")
def live_database():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for live pipeline tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    try:
        yield engine, session_factory
    finally:
        command.upgrade(config, "head")
        engine.dispose()


def _seed_student_session_skill(session_factory, suffix=""):
    with session_scope(session_factory) as session:
        student = Student(
            student_id=uuid.uuid4(),
            external_student_id=f"student_{suffix}_{uuid.uuid4().hex[:8]}",
        )
        session.add(student)
        session.flush()

        learning_session = LearningSession(
            session_id=uuid.uuid4(),
            student_id=student.student_id,
            external_session_id=f"session_{suffix}_{uuid.uuid4().hex[:8]}",
        )
        session.add(learning_session)

        skill = CanonicalSkill(
            skill_id=uuid.uuid4(),
            canonical_name=f"math :: algebra :: linear_{suffix}_{uuid.uuid4().hex[:8]}",
            display_name=f"Linear Equations {suffix}",
        )
        session.add(skill)
        session.flush()

        return student.student_id, learning_session.session_id, skill.skill_id


def test_single_interaction_pipeline_execution(live_database):
    """Verify single interaction update across all memory layers."""
    _, session_factory = live_database
    student_id, session_id, skill_id = _seed_student_session_skill(
        session_factory, "single"
    )

    service = PostgresMemoryUpdateService(session_factory)
    interaction = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator_v1",
        external_interaction_id=f"ext_{uuid.uuid4().hex}",
        problem_id="prob_1",
        question_id="q_1",
        student_answer="x=4",
        expected_answer="x=5",
        is_correct=False,
        identified_error="sign inversion error",
        attempt_count=2,
        hint_count=1,
        hint_total=2,
        response_time_ms=12500.0,
    )

    result = service.update_interaction(interaction)

    # 1. Raw Interaction Log
    assert result.raw_interaction.student_id == student_id
    assert result.raw_interaction.is_correct is False
    assert result.raw_interaction.identified_error == "sign inversion error"

    # 2. Short-Term Memory
    assert result.short_term_memory.interaction_count == 1
    assert result.short_term_memory.correct_count == 0
    assert result.short_term_memory.incorrect_count == 1
    assert result.short_term_memory.recent_accuracy == 0.0
    assert result.short_term_memory.attempt_sum == 2
    assert result.short_term_memory.hint_sum == 1
    assert result.short_term_memory.response_time_sum_ms == 12500.0

    # 3. Long-Term Memory
    assert result.long_term_memory.interaction_count == 1
    assert result.long_term_memory.total_sessions == 1
    assert result.long_term_memory.concept_count == 1
    assert result.long_term_memory.overall_accuracy == 0.0

    # 4. Concept Memory
    assert result.concept_memory is not None
    assert result.concept_memory.interaction_count == 1
    assert result.concept_memory.accuracy == 0.0

    # 5. Misconceptions
    assert result.misconception is not None
    assert result.misconception.occurrence_count == 1
    assert result.misconception.normalized_error == "sign inversion error"

    # 6. Learning-State Snapshot & Current State
    assert result.snapshot is not None
    assert result.snapshot.learning_state in (
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
        "UNAVAILABLE",
    )
    assert result.current_learning_state is not None
    assert (
        result.current_learning_state.learning_state
        == result.snapshot.learning_state
    )


def test_multiple_interactions_and_misconception_increment(live_database):
    """Verify occurrence count increments and statistics accumulate accurately."""
    _, session_factory = live_database
    student_id, session_id, skill_id = _seed_student_session_skill(
        session_factory, "multi"
    )

    service = PostgresMemoryUpdateService(session_factory)

    # Interaction 1: Incorrect with misconception
    i1 = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator_v1",
        external_interaction_id=f"ext_{uuid.uuid4().hex}",
        is_correct=False,
        identified_error="algebra sign error",
        attempt_count=1,
    )
    res1 = service.update_interaction(i1)
    assert res1.misconception.occurrence_count == 1
    assert res1.short_term_memory.interaction_count == 1
    assert res1.short_term_memory.recent_accuracy == 0.0

    # Interaction 2: Correct, no misconception
    i2 = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator_v1",
        external_interaction_id=f"ext_{uuid.uuid4().hex}",
        is_correct=True,
        attempt_count=1,
    )
    res2 = service.update_interaction(i2)
    assert res2.misconception is None
    assert res2.short_term_memory.interaction_count == 2
    assert res2.short_term_memory.recent_accuracy == 0.5

    # Interaction 3: Incorrect with same misconception (normalized check)
    i3 = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator_v1",
        external_interaction_id=f"ext_{uuid.uuid4().hex}",
        is_correct=False,
        identified_error="  ALGEBRA   SIGN  ERROR  ",
        attempt_count=2,
    )
    res3 = service.update_interaction(i3)
    assert res3.misconception is not None
    assert res3.misconception.occurrence_count == 2
    assert res3.short_term_memory.interaction_count == 3
    assert res3.short_term_memory.correct_count == 1
    assert res3.short_term_memory.incorrect_count == 2
    assert pytest.approx(res3.short_term_memory.recent_accuracy) == 1 / 3


def test_complete_rollback_on_forced_failure(live_database):
    """Verify that if any step in the pipeline fails, the transaction rolls back completely."""
    _, session_factory = live_database
    student_id, session_id, skill_id = _seed_student_session_skill(
        session_factory, "rollback"
    )

    service = PostgresMemoryUpdateService(session_factory)

    # Valid interaction first
    valid_i = RawInteractionCreate(
        student_id=student_id,
        session_id=session_id,
        canonical_skill_id=skill_id,
        source="evaluator_v1",
        external_interaction_id=f"valid_{uuid.uuid4().hex}",
        is_correct=True,
    )
    service.update_interaction(valid_i)

    # Verify 1 interaction exists
    with UnitOfWork(session_factory) as uow:
        assert len(uow.raw_interactions.get_student_interactions(student_id)) == 1

    # Attempt a batch where second interaction violates a foreign key (non-existent session)
    invalid_session_id = uuid.uuid4()
    batch = [
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"batch_1_{uuid.uuid4().hex}",
            is_correct=True,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=invalid_session_id,  # Invalid session FK constraint failure
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"batch_2_{uuid.uuid4().hex}",
            is_correct=True,
        ),
    ]

    with pytest.raises(PostgresMemoryUpdateError):
        service.update_interactions(batch)

    # Verify that batch_1 was rolled back and still only 1 interaction exists!
    with UnitOfWork(session_factory) as uow:
        logs = uow.raw_interactions.get_student_interactions(student_id)
        assert len(logs) == 1
        stm = uow.projections.get_short_term_memory(student_id, session_id)
        assert stm.interaction_count == 1


def test_rebuild_student_projections_equality(live_database):
    """Verify that incremental projection results match rebuilt projection results exactly."""
    _, session_factory = live_database
    student_id, session_id, skill_id = _seed_student_session_skill(
        session_factory, "rebuild"
    )

    service = PostgresMemoryUpdateService(session_factory)

    # Sequence of varying interactions
    interactions = [
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"reb_1_{uuid.uuid4().hex}",
            is_correct=True,
            attempt_count=1,
            hint_count=0,
            hint_total=2,
            response_time_ms=5000.0,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"reb_2_{uuid.uuid4().hex}",
            is_correct=False,
            attempt_count=3,
            hint_count=2,
            hint_total=2,
            response_time_ms=15000.0,
        ),
        RawInteractionCreate(
            student_id=student_id,
            session_id=session_id,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"reb_3_{uuid.uuid4().hex}",
            is_correct=True,
            attempt_count=None,  # Missing attempt
            hint_count=None,    # Missing hint
            response_time_ms=8000.0,
        ),
    ]

    for item in interactions:
        service.update_interaction(item)

    # Check incremental projections
    with UnitOfWork(session_factory) as uow:
        incremental_stm = uow.projections.get_short_term_memory(
            student_id, session_id
        )
        incremental_ltm = uow.projections.get_long_term_memory(student_id)
        incremental_concept = uow.projections.get_concept_memory(
            student_id, skill_id
        )

    # Run full rebuild
    rebuilt = service.rebuild_projections(student_id)

    # Verify rebuilt records match incremental records field by field
    with UnitOfWork(session_factory) as uow:
        rebuilt_stm = uow.projections.get_short_term_memory(student_id, session_id)
        rebuilt_ltm = uow.projections.get_long_term_memory(student_id)
        rebuilt_concept = uow.projections.get_concept_memory(student_id, skill_id)

    assert incremental_stm.interaction_count == rebuilt_stm.interaction_count
    assert incremental_stm.correct_count == rebuilt_stm.correct_count
    assert incremental_stm.incorrect_count == rebuilt_stm.incorrect_count
    assert incremental_stm.recent_accuracy == rebuilt_stm.recent_accuracy
    assert (
        incremental_stm.attempt_observation_count
        == rebuilt_stm.attempt_observation_count
    )
    assert incremental_stm.attempt_sum == rebuilt_stm.attempt_sum
    assert (
        incremental_stm.response_time_sum_ms
        == rebuilt_stm.response_time_sum_ms
    )

    assert incremental_ltm.total_sessions == rebuilt_ltm.total_sessions
    assert incremental_ltm.concept_count == rebuilt_ltm.concept_count
    assert incremental_ltm.interaction_count == rebuilt_ltm.interaction_count
    assert incremental_ltm.overall_accuracy == rebuilt_ltm.overall_accuracy

    assert incremental_concept.interaction_count == rebuilt_concept.interaction_count
    assert incremental_concept.accuracy == rebuilt_concept.accuracy


def test_student_isolation(live_database):
    """Verify that student records and projections do not leak into one another."""
    _, session_factory = live_database
    s1_id, s1_session, skill_id = _seed_student_session_skill(
        session_factory, "iso1"
    )
    s2_id, s2_session, _ = _seed_student_session_skill(
        session_factory, "iso2"
    )

    service = PostgresMemoryUpdateService(session_factory)

    service.update_interaction(
        RawInteractionCreate(
            student_id=s1_id,
            session_id=s1_session,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"iso_s1_{uuid.uuid4().hex}",
            is_correct=True,
            identified_error="error_s1",
        )
    )

    service.update_interaction(
        RawInteractionCreate(
            student_id=s2_id,
            session_id=s2_session,
            canonical_skill_id=skill_id,
            source="evaluator_v1",
            external_interaction_id=f"iso_s2_{uuid.uuid4().hex}",
            is_correct=False,
            identified_error="error_s2",
        )
    )

    with UnitOfWork(session_factory) as uow:
        # S1 assertions
        s1_stm = uow.projections.get_short_term_memory(s1_id, s1_session)
        assert s1_stm.correct_count == 1
        assert s1_stm.incorrect_count == 0
        s1_history = uow.supporting.get_history(s1_id, skill_id)
        assert len(s1_history) == 1

        # S2 assertions
        s2_stm = uow.projections.get_short_term_memory(s2_id, s2_session)
        assert s2_stm.correct_count == 0
        assert s2_stm.incorrect_count == 1
        s2_history = uow.supporting.get_history(s2_id, skill_id)
        assert len(s2_history) == 1
