"""Deterministic, API-free checks for the Repository B -> frozen-v3 bridge."""

from __future__ import annotations

import copy
import json
import os
import socket
import sqlite3
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


REPOSITORY_A_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_B_ROOT = Path(
    os.environ.get(
        "STUDENT_MODEL_REPO",
        REPOSITORY_A_ROOT.parent / "student-modeling",
    )
).resolve()
if not (REPOSITORY_B_ROOT / "bkt" / "predict.py").is_file():
    raise RuntimeError(
        "Repository B is incomplete. Set STUDENT_MODEL_REPO to a complete "
        "student-modeling checkout."
    )
sys.path.insert(0, str(REPOSITORY_B_ROOT))


from bkt.predict import BKTPredictor, ColdStartPriorCalculator  # noqa: E402
import core.knowledge_graph as knowledge_graph_module  # noqa: E402
from core.cross_session_pipeline import (  # noqa: E402
    CrossSessionStudentModelPipeline,
)
from core.knowledge_graph import KnowledgeGraph  # noqa: E402
from db.database import initialise_database  # noqa: E402
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.experience_logger import (  # noqa: E402
    ExperienceLogger,
)
from src.self_improvement.lints_policy import (  # noqa: E402
    TrueDisjointLinTS,
)
from src.self_improvement.state_io import (  # noqa: E402
    load_policy_state,
)
from src.self_improvement.turn_context_builder import (  # noqa: E402
    TURN_FEATURE_NAMES,
)
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


TARGET_PARAMS = {
    "Target": {
        "prior": 0.20,
        "learns": 0.10,
        "guesses": 0.20,
        "slips": 0.10,
        "forgets": 0.0,
    }
}
TURN_PROBABILITIES = (
    {"generic": 0.10, "probing": 0.34, "focus": 0.40, "telling": 0.16},
    {"generic": 0.36, "probing": 0.10, "focus": 0.34, "telling": 0.20},
    {"generic": 0.10, "probing": 0.18, "focus": 0.34, "telling": 0.38},
)
MRB1_SCORES = {
    "Mistake_Identification": 0.80,
    "Mistake_Location": 0.70,
    "Providing_Guidance": 0.60,
    "Actionability": 0.50,
}


class StubStudentModelPipeline:
    def __init__(self, mastery_before: float) -> None:
        self.mastery_before = mastery_before
        self.start_calls: list[dict[str, object]] = []

    def start_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        attempt_skill: str | None = None,
    ) -> object:
        self.start_calls.append(
            {
                "student_id": student_id,
                "attempt_id": attempt_id,
                "attempt_skill": attempt_skill,
            }
        )
        return SimpleNamespace(
            student_id=student_id,
            attempt_id=attempt_id,
            skill=attempt_skill,
            mastery_before=self.mastery_before,
        )


class NeverCalledMD6:
    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> Mapping[str, float]:
        del problem, conversation_history
        raise AssertionError("The bridge must not run MD6.")


class NeverCalledMRB1:
    def score_response(
        self,
        conversation_history: Sequence[Mapping[str, object]],
        tutor_response: str,
    ) -> Mapping[str, float]:
        del conversation_history, tutor_response
        raise AssertionError("The bridge must not run MRB1.")


class RecordingPolicy(TrueDisjointLinTS):
    def __init__(self, *, seed: int = 20260825) -> None:
        super().__init__(
            context_dim=len(TURN_FEATURE_NAMES),
            seed=seed,
            data_mode="synthetic",
        )
        self.observed_updates: list[dict[str, object]] = []

    def update(
        self,
        arm: str,
        context: object,
        reward: object,
        sample_weight: object = 1.0,
    ) -> None:
        super().update(arm, context, reward, sample_weight)
        self.observed_updates.append(
            {
                "arm": arm,
                "context": tuple(float(value) for value in context),
                "reward": float(reward),
                "sample_weight": float(sample_weight),
            }
        )


class ControlledColdStart:
    def __init__(self, prior: float | None = None) -> None:
        self.prior = prior
        self.calls = 0

    def compute_prior(
        self,
        *,
        skill: str,
        student_masteries: dict[str, float],
        population_prior: float,
    ) -> dict[str, object]:
        del skill
        self.calls += 1
        selected = population_prior if self.prior is None else self.prior
        return {
            "prior": selected,
            "used_transfer": selected > population_prior,
            "related_skills_used": sorted(student_masteries),
            "transfer_evidence": "controlled bridge test prior",
        }


class StaticExtractor:
    def __init__(self, events: Sequence[tuple[int, str]]) -> None:
        self.events = tuple(events)

    def extract(self, transcript: list[dict]) -> dict[str, object]:
        return {
            "events": [
                {
                    "turn_index": turn_index,
                    "skill": skill,
                    "student_text": transcript[turn_index]["text"],
                    "evidence_span": transcript[turn_index]["text"],
                }
                for turn_index, skill in self.events
            ],
            "misconceptions": [],
        }


class StructuredEvaluator:
    def __init__(
        self,
        assessed_skills: Sequence[str],
        verdicts: Mapping[object, str],
    ) -> None:
        self.context = SimpleNamespace(
            assessed_skills=tuple(assessed_skills)
        )
        self.verdicts = dict(verdicts)

    def __call__(self, event: dict, transcript: list[dict]) -> dict[str, object]:
        del transcript
        skill = event["skill"]
        if skill not in self.context.assessed_skills:
            return {
                "correctness": "unknown",
                "confidence": 0.0,
                "source": "skill_not_authorized_for_assessment",
            }
        correctness = self.verdicts.get(
            (skill, event["turn_index"]),
            self.verdicts.get(skill, "unknown"),
        )
        return {
            "correctness": correctness,
            "confidence": 1.0 if correctness != "unknown" else 0.0,
            "source": "phase3_test_evaluator",
        }


class QuietDetectors:
    def reasoning_predictor(self, current: str, previous: str | None = None) -> float:
        del current, previous
        return 0.0

    def uncertainty_predictor(self, text: str) -> float:
        del text
        return 0.0

    def clarification_predictor(self, text: str) -> float:
        del text
        return 0.0


@contextmanager
def sqlite_connection(path: Path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def adaptive_memory(attempt_id: str) -> dict[str, object]:
    return {
        "attempt_id": attempt_id,
        "problem": "Synthetic bridge-only problem.",
        "conversation_history": [],
        "mastery_before": 0.0,
    }


def student_model_result(
    *,
    attempt_id: str,
    skill: str,
    before: float,
    after: float,
    delta: float,
    diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "learning_outcome": {
            "attempt_id": attempt_id,
            "skill": skill,
            "mastery_before": before,
            "mastery_after": after,
            "delta_mastery": delta,
        }
    }
    if diagnostics is not None:
        result["diagnostics"] = dict(diagnostics)
    return result


def build_adaptive_pipeline(
    root: Path,
    *,
    seed: int = 20260825,
) -> tuple[AdaptiveTutorPipeline, RecordingPolicy, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    policy = RecordingPolicy(seed=seed)
    controller = TurnLevelAttemptController(policy)
    log_path = root / "synthetic_bridge_attempts_v3.jsonl"
    state_path = root / "synthetic_bridge_state_v3.json"
    pipeline = AdaptiveTutorPipeline(
        md6=NeverCalledMD6(),
        mrb1=NeverCalledMRB1(),
        controller=controller,
        experience_logger=ExperienceLogger(
            log_path,
            data_mode="synthetic",
            source_policy=policy,
        ),
        policy_state_path=state_path,
    )
    return pipeline, policy, log_path, state_path


def record_stub_turns(
    pipeline: AdaptiveTutorPipeline,
    turn_count: int,
) -> list[object]:
    decisions = []
    for index in range(turn_count):
        probabilities = TURN_PROBABILITIES[index % len(TURN_PROBABILITIES)]
        decision = pipeline.controller.select_turn(probabilities)
        pipeline.controller.record_mrb1_scores(MRB1_SCORES)
        decisions.append(decision)
    return decisions


def build_stub_bridge(
    root: Path,
    *,
    mastery_before: float,
    seed: int = 20260825,
) -> tuple[
    StudentModelFrozenV3Bridge,
    AdaptiveTutorPipeline,
    RecordingPolicy,
    Path,
    Path,
]:
    adaptive, policy, log_path, state_path = build_adaptive_pipeline(
        root,
        seed=seed,
    )
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=StubStudentModelPipeline(mastery_before),
        adaptive_pipeline=adaptive,
    )
    return bridge, adaptive, policy, log_path, state_path


def build_real_student_model(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    target_skill: str,
    events: Sequence[tuple[int, str]],
    verdicts: Mapping[object, str],
    predictor: BKTPredictor,
    cold_start: object,
) -> tuple[
    CrossSessionStudentModelPipeline,
    KnowledgeGraph,
    StructuredEvaluator,
    Path,
]:
    db_path = root / "student_model.sqlite3"

    def get_test_connection():
        return sqlite_connection(db_path)

    def initialise_test_database():
        initialise_database(db_path)

    monkeypatch.setattr(
        knowledge_graph_module,
        "get_connection",
        get_test_connection,
    )
    monkeypatch.setattr(
        knowledge_graph_module,
        "initialise_database",
        initialise_test_database,
    )
    graph = KnowledgeGraph(
        predictor=predictor,
        cold_start=cold_start,
    )
    evaluator = StructuredEvaluator((target_skill,), verdicts)
    pipeline = CrossSessionStudentModelPipeline(
        concept_extractor=StaticExtractor(events),
        evaluator=evaluator,
        detectors=QuietDetectors(),
        knowledge_graph=graph,
    )
    return pipeline, graph, evaluator, db_path


def table_count(db_path: Path, table: str) -> int:
    with sqlite_connection(db_path) as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        )


def test_01_positive_outcome_reaches_lints_raw(tmp_path: Path) -> None:
    bridge, adaptive, policy, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("positive")
    bridge.start_attempt(
        student_id="student",
        attempt_id="positive",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="positive",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert result["reward"] == pytest.approx(0.07)
    assert policy.observed_updates[0]["reward"] == pytest.approx(0.07)
    assert set(result["experience_record"]["learning_outcome"]) == {
        "skill",
        "mastery_before",
        "mastery_after",
        "delta_mastery",
    }


def test_02_negative_outcome_reaches_lints_raw(tmp_path: Path) -> None:
    bridge, adaptive, policy, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.41,
    )
    memory = adaptive_memory("negative")
    bridge.start_attempt(
        student_id="student",
        attempt_id="negative",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="negative",
            skill="Target",
            before=0.41,
            after=0.35,
            delta=-0.06,
        ),
    )
    assert result["reward"] == pytest.approx(-0.06)
    assert policy.observed_updates[0]["reward"] == pytest.approx(-0.06)


def test_03_zero_outcome_remains_zero(tmp_path: Path) -> None:
    bridge, adaptive, policy, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.41,
    )
    memory = adaptive_memory("zero")
    bridge.start_attempt(
        student_id="student",
        attempt_id="zero",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="zero",
            skill="Target",
            before=0.41,
            after=0.41,
            delta=0.0,
        ),
    )
    assert result["reward"] == 0.0
    assert policy.observed_updates[0]["reward"] == 0.0


def test_04_inconsistent_delta_is_rejected_by_frozen_v3(tmp_path: Path) -> None:
    bridge, adaptive, policy, log_path, state_path = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("bad-delta")
    bridge.start_attempt(
        student_id="student",
        attempt_id="bad-delta",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    with pytest.raises(ValueError, match="delta_mastery must equal"):
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=student_model_result(
                attempt_id="bad-delta",
                skill="Target",
                before=0.34,
                after=0.41,
                delta=0.50,
            ),
        )
    assert policy.total_updates == 0
    assert not log_path.exists()
    assert not state_path.exists()


def test_05_attempt_id_match_and_mismatch(tmp_path: Path) -> None:
    bridge, adaptive, _, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("shared-id")
    bridge.start_attempt(
        student_id="student",
        attempt_id="shared-id",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    original_finish = adaptive.finish_attempt
    adaptive.finish_attempt = Mock(wraps=original_finish)
    with pytest.raises(ValueError, match="attempt_id does not match"):
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=student_model_result(
                attempt_id="different-id",
                skill="Target",
                before=0.34,
                after=0.41,
                delta=0.07,
            ),
        )
    assert adaptive.finish_attempt.call_count == 0
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="shared-id",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert adaptive.finish_attempt.call_count == 1
    assert result["attempt_id"] == "shared-id"


def test_06_target_skill_match_and_mismatch(tmp_path: Path) -> None:
    bridge, adaptive, _, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("skill-check")
    bridge.start_attempt(
        student_id="student",
        attempt_id="skill-check",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    original_finish = adaptive.finish_attempt
    adaptive.finish_attempt = Mock(wraps=original_finish)
    with pytest.raises(ValueError, match="authoritative target skill"):
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=student_model_result(
                attempt_id="skill-check",
                skill="Incidental Skill",
                before=0.34,
                after=0.41,
                delta=0.07,
            ),
        )
    assert adaptive.finish_attempt.call_count == 0
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="skill-check",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert adaptive.finish_attempt.call_count == 1
    assert result["skill"] == "Target"


def test_07_real_bkt_cold_start_uses_persisted_population_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictor = BKTPredictor.load()
    target_skill = "Percent Of"
    assert predictor.has_skill(target_skill)
    student_pipeline, graph, _, db_path = build_real_student_model(
        monkeypatch,
        tmp_path / "student",
        target_skill=target_skill,
        events=(),
        verdicts={},
        predictor=predictor,
        cold_start=ColdStartPriorCalculator(),
    )
    adaptive, _, _, _ = build_adaptive_pipeline(tmp_path / "adaptive")
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=student_pipeline,
        adaptive_pipeline=adaptive,
    )
    memory = adaptive_memory("real-cold-start")
    started = bridge.start_attempt(
        student_id="student",
        attempt_id="real-cold-start",
        target_skill=target_skill,
        adaptive_memory=memory,
    )
    population_prior = predictor.params[target_skill]["prior"]
    persisted = graph.get_effective_initial_prior("student", target_skill)
    assert started.mastery_before == pytest.approx(population_prior)
    assert persisted["effective_initial_prior"] == pytest.approx(
        population_prior
    )
    assert persisted["prior_source"] == "population"
    assert table_count(db_path, "attempts") == 0
    assert table_count(db_path, "sessions") == 0
    assert table_count(db_path, "mastery") == 0


def test_08_sequential_retry_starts_at_previous_bkt_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cold_start = ControlledColdStart(0.75)
    evaluator_verdicts = {"Target": "incorrect"}
    student_pipeline, _, _, _ = build_real_student_model(
        monkeypatch,
        tmp_path / "student",
        target_skill="Target",
        events=((0, "Target"),),
        verdicts=evaluator_verdicts,
        predictor=BKTPredictor(TARGET_PARAMS),
        cold_start=cold_start,
    )
    adaptive, _, _, _ = build_adaptive_pipeline(tmp_path / "adaptive")
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=student_pipeline,
        adaptive_pipeline=adaptive,
    )

    first_memory = adaptive_memory("retry-1")
    bridge.start_attempt(
        student_id="student",
        attempt_id="retry-1",
        target_skill="Target",
        adaptive_memory=first_memory,
    )
    record_stub_turns(adaptive, 1)
    first_b = student_pipeline.process_transcript(
        transcript=[{"role": "student", "text": "incorrect"}],
        student_id="student",
        session_id="retry-session-1",
        attempt_context=bridge.student_model_context,
    )
    first_outcome = first_b["learning_outcome"]
    bridge.finish_attempt(
        adaptive_memory=first_memory,
        student_model_result=first_b,
    )

    second_memory = adaptive_memory("retry-2")
    second_start = bridge.start_attempt(
        student_id="student",
        attempt_id="retry-2",
        target_skill="Target",
        adaptive_memory=second_memory,
    )
    assert first_outcome["mastery_after"] == pytest.approx(
        second_start.mastery_before
    )
    assert second_start.mastery_before == pytest.approx(
        0.3454545454545455
    )
    record_stub_turns(adaptive, 1)
    second_b = student_pipeline.process_transcript(
        transcript=[{"role": "student", "text": "incorrect again"}],
        student_id="student",
        session_id="retry-session-2",
        attempt_context=bridge.student_model_context,
    )
    bridge.finish_attempt(
        adaptive_memory=second_memory,
        student_model_result=second_b,
    )
    assert second_b["learning_outcome"]["mastery_after"] == pytest.approx(
        0.15570032573289902
    )


def test_09_multiple_evaluation_observations_finish_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verdicts = {
        ("Target", 0): "correct",
        ("Target", 1): "correct",
        ("Target", 2): "incorrect",
    }
    student_pipeline, graph, _, _ = build_real_student_model(
        monkeypatch,
        tmp_path / "student",
        target_skill="Target",
        events=((0, "Target"), (1, "Target"), (2, "Target")),
        verdicts=verdicts,
        predictor=BKTPredictor(TARGET_PARAMS),
        cold_start=ControlledColdStart(),
    )
    adaptive, policy, _, _ = build_adaptive_pipeline(tmp_path / "adaptive")
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=student_pipeline,
        adaptive_pipeline=adaptive,
    )
    memory = adaptive_memory("multi-observation")
    bridge.start_attempt(
        student_id="student",
        attempt_id="multi-observation",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 2)
    b_result = student_pipeline.process_transcript(
        transcript=[
            {"role": "student", "text": "correct one"},
            {"role": "student", "text": "correct two"},
            {"role": "student", "text": "incorrect three"},
        ],
        student_id="student",
        session_id="multi-observation-session",
        attempt_context=bridge.student_model_context,
    )
    original_finish = adaptive.finish_attempt
    adaptive.finish_attempt = Mock(wraps=original_finish)
    completion = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=b_result,
    )
    assert len(b_result["resolved_events"]) == 3
    assert len(graph.get_attempts("student", "Target")) == 3
    assert list(key for key in b_result if key == "learning_outcome") == [
        "learning_outcome"
    ]
    assert adaptive.finish_attempt.call_count == 1
    assert completion["turn_count"] == 2
    assert policy.total_updates == 2
    with pytest.raises(RuntimeError, match="No cross-repository attempt"):
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=b_result,
        )
    assert adaptive.finish_attempt.call_count == 1
    assert policy.total_updates == 2


def test_10_full_history_matches_bkt_with_persisted_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictor = BKTPredictor(TARGET_PARAMS)
    cold_start = ControlledColdStart(0.60)
    student_pipeline, graph, evaluator, _ = build_real_student_model(
        monkeypatch,
        tmp_path / "student",
        target_skill="Target",
        events=((0, "Target"),),
        verdicts={"Target": "correct"},
        predictor=predictor,
        cold_start=cold_start,
    )
    adaptive, _, _, _ = build_adaptive_pipeline(tmp_path / "adaptive")
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=student_pipeline,
        adaptive_pipeline=adaptive,
    )
    for index, verdict in enumerate(("correct", "incorrect"), start=1):
        evaluator.verdicts["Target"] = verdict
        attempt_id = f"history-{index}"
        memory = adaptive_memory(attempt_id)
        bridge.start_attempt(
            student_id="student",
            attempt_id=attempt_id,
            target_skill="Target",
            adaptive_memory=memory,
        )
        record_stub_turns(adaptive, 1)
        b_result = student_pipeline.process_transcript(
            transcript=[{"role": "student", "text": verdict}],
            student_id="student",
            session_id=f"history-session-{index}",
            attempt_context=bridge.student_model_context,
        )
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=b_result,
        )

    prior = graph.get_effective_initial_prior(
        "student", "Target"
    )["effective_initial_prior"]
    history = graph.get_attempts("student", "Target")
    expected = predictor.predict(
        "Target",
        history,
        initial_prior=prior,
    )
    persisted = graph.get_mastery("student", "Target")
    assert prior == pytest.approx(0.60)
    assert cold_start.calls == 1
    assert persisted["mastery_probability"] == pytest.approx(expected)
    assert expected == pytest.approx(0.5387900355871885)


def test_11_no_policy_update_before_completion(tmp_path: Path) -> None:
    bridge, adaptive, policy, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("delayed-only")
    bridge.start_attempt(
        student_id="student",
        attempt_id="delayed-only",
        target_skill="Target",
        adaptive_memory=memory,
    )
    initial_A = {arm: value.copy() for arm, value in policy.A.items()}
    initial_b = {arm: value.copy() for arm, value in policy.b.items()}
    record_stub_turns(adaptive, 3)
    assert policy.total_updates == 0
    for arm in policy.arms:
        np.testing.assert_array_equal(policy.A[arm], initial_A[arm])
        np.testing.assert_array_equal(policy.b[arm], initial_b[arm])
    bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="delayed-only",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert policy.total_updates == 3


def test_12_equal_delayed_credit_has_unit_total_weight(tmp_path: Path) -> None:
    bridge, adaptive, policy, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.41,
    )
    memory = adaptive_memory("credit")
    bridge.start_attempt(
        student_id="student",
        attempt_id="credit",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 3)
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="credit",
            skill="Target",
            before=0.41,
            after=0.35,
            delta=-0.06,
        ),
    )
    weights = [
        update["sample_weight"] for update in policy.observed_updates
    ]
    assert len(policy.observed_updates) == 3
    assert weights == pytest.approx([1.0 / 3.0] * 3)
    assert sum(weights) == pytest.approx(1.0)
    assert [
        update["reward"] for update in policy.observed_updates
    ] == pytest.approx([-0.06] * 3)
    assert result["sample_weight_per_turn"] == pytest.approx(1.0 / 3.0)
    assert result["total_attempt_weight"] == pytest.approx(1.0)


def test_13_optional_diagnostics_absent_succeeds(tmp_path: Path) -> None:
    bridge, adaptive, _, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("no-diagnostics")
    bridge.start_attempt(
        student_id="student",
        attempt_id="no-diagnostics",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    result = bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="no-diagnostics",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert "diagnostics" not in result
    assert "diagnostics" not in result["experience_record"]


def test_14_present_diagnostics_do_not_change_reward_context_or_action(
    tmp_path: Path,
) -> None:
    plain = build_stub_bridge(
        tmp_path / "plain",
        mastery_before=0.34,
        seed=99,
    )
    diagnostic = build_stub_bridge(
        tmp_path / "diagnostic",
        mastery_before=0.34,
        seed=99,
    )
    plain_bridge, plain_adaptive, plain_policy, _, _ = plain
    diag_bridge, diag_adaptive, diag_policy, _, _ = diagnostic
    plain_memory = adaptive_memory("plain")
    diag_memory = adaptive_memory("diagnostic")
    plain_bridge.start_attempt(
        student_id="student",
        attempt_id="plain",
        target_skill="Target",
        adaptive_memory=plain_memory,
    )
    diag_bridge.start_attempt(
        student_id="student",
        attempt_id="diagnostic",
        target_skill="Target",
        adaptive_memory=diag_memory,
    )
    plain_decision = record_stub_turns(plain_adaptive, 1)[0]
    diag_decision = record_stub_turns(diag_adaptive, 1)[0]
    assert plain_decision.context == diag_decision.context
    assert plain_decision.selected_arm == diag_decision.selected_arm
    assert plain_decision.final_move == diag_decision.final_move

    plain_result = plain_bridge.finish_attempt(
        adaptive_memory=plain_memory,
        student_model_result=student_model_result(
            attempt_id="plain",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    diagnostic_values = {
        "evaluator_result": {"score": 2},
        "reasoning": "already supplied",
        "uncertainty": 0.25,
        "clarification": False,
        "repeated_misunderstanding": True,
    }
    diag_result = diag_bridge.finish_attempt(
        adaptive_memory=diag_memory,
        student_model_result=student_model_result(
            attempt_id="diagnostic",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
            diagnostics=diagnostic_values,
        ),
    )
    assert plain_result["reward"] == diag_result["reward"] == pytest.approx(
        0.07
    )
    assert plain_policy.observed_updates == diag_policy.observed_updates
    assert diag_result["diagnostics"] == diagnostic_values
    assert diag_result["experience_record"]["diagnostics"] == diagnostic_values


def test_15_synthetic_state_cannot_seed_real_v3_lineage(tmp_path: Path) -> None:
    bridge, adaptive, _, _, state_path = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("synthetic-only")
    bridge.start_attempt(
        student_id="student",
        attempt_id="synthetic-only",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="synthetic-only",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    assert envelope["policy_state"]["data_mode"] == "synthetic"
    real_receiver = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        data_mode="real",
    )
    with pytest.raises(ValueError, match="Synthetic/real"):
        load_policy_state(
            real_receiver,
            state_path,
            expected_data_mode="real",
        )


def test_16_bridge_makes_no_external_api_call(tmp_path: Path) -> None:
    bridge, adaptive, _, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("no-api")
    with patch.object(
        socket,
        "create_connection",
        side_effect=AssertionError("network access forbidden"),
    ) as socket_call, patch.object(
        urllib.request,
        "urlopen",
        side_effect=AssertionError("network access forbidden"),
    ) as urlopen_call:
        bridge.start_attempt(
            student_id="student",
            attempt_id="no-api",
            target_skill="Target",
            adaptive_memory=memory,
        )
        record_stub_turns(adaptive, 1)
        bridge.finish_attempt(
            adaptive_memory=memory,
            student_model_result=student_model_result(
                attempt_id="no-api",
                skill="Target",
                before=0.34,
                after=0.41,
                delta=0.07,
            ),
        )
    socket_call.assert_not_called()
    urlopen_call.assert_not_called()


def test_17_bridge_accesses_no_held_out_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protected = {
        (
            REPOSITORY_A_ROOT
            / "data"
            / "raw"
            / "mathdial"
            / "test.jsonl"
        ).resolve(),
        (
            REPOSITORY_A_ROOT
            / "data"
            / "processed"
            / "mathdial"
            / "test.jsonl"
        ).resolve(),
        (
            REPOSITORY_A_ROOT
            / "data"
            / "external"
            / "mrbench"
            / "mrbench_v3_testset.json"
        ).resolve(),
    }
    original_open = Path.open
    accessed: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        resolved = path.resolve()
        if resolved in protected:
            accessed.append(resolved)
            raise AssertionError(f"held-out dataset access forbidden: {resolved}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    bridge, adaptive, _, _, _ = build_stub_bridge(
        tmp_path,
        mastery_before=0.34,
    )
    memory = adaptive_memory("no-held-out")
    bridge.start_attempt(
        student_id="student",
        attempt_id="no-held-out",
        target_skill="Target",
        adaptive_memory=memory,
    )
    record_stub_turns(adaptive, 1)
    bridge.finish_attempt(
        adaptive_memory=memory,
        student_model_result=student_model_result(
            attempt_id="no-held-out",
            skill="Target",
            before=0.34,
            after=0.41,
            delta=0.07,
        ),
    )
    assert accessed == []
