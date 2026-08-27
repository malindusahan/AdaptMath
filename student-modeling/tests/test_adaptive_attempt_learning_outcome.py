from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from core.attempt_learning_outcome import (
    AdaptiveAttemptContext,
    AttemptLearningOutcome,
)
from core.cross_session_pipeline import CrossSessionStudentModelPipeline
from tests.test_effective_initial_prior import build_graph


class StaticExtractor:
    def __init__(self, skills_by_turn):
        self.skills_by_turn = list(skills_by_turn)

    def extract(self, transcript):
        return {
            "events": [
                {
                    "turn_index": turn_index,
                    "skill": skill,
                    "student_text": transcript[turn_index]["text"],
                    "evidence_span": transcript[turn_index]["text"],
                }
                for turn_index, skill in self.skills_by_turn
            ],
            "misconceptions": [],
        }


class StructuredEvaluator:
    def __init__(self, assessed_skills, verdicts):
        self.context = SimpleNamespace(
            assessed_skills=tuple(assessed_skills)
        )
        self.verdicts = dict(verdicts)

    def __call__(self, event, transcript):
        correctness = self.verdicts.get(
            (event["skill"], event.get("turn_index")),
            self.verdicts.get(event["skill"], "unknown"),
        )
        if event["skill"] not in self.context.assessed_skills:
            return {
                "correctness": "unknown",
                "confidence": 0.0,
                "source": "skill_not_authorized_for_assessment",
            }
        return {
            "correctness": correctness,
            "confidence": 1.0 if correctness != "unknown" else 0.0,
            "source": "test_evaluator",
        }


class QuietDetectors:
    def reasoning_predictor(self, current, previous=None):
        return 0.0

    def uncertainty_predictor(self, text):
        return 0.0

    def clarification_predictor(self, text):
        return 0.0


class HighDetectors:
    def reasoning_predictor(self, current, previous=None):
        return 0.95

    def uncertainty_predictor(self, text):
        return 0.95

    def clarification_predictor(self, text):
        return 0.95


def build_pipeline(graph, *, extractor, evaluator, detectors=None):
    return CrossSessionStudentModelPipeline(
        concept_extractor=extractor,
        evaluator=evaluator,
        detectors=detectors or QuietDetectors(),
        knowledge_graph=graph,
    )


def student_transcript(*texts):
    return [{"role": "student", "text": text} for text in texts]


def test_attempt_boundary_models_are_frozen_and_validate_arithmetic():
    context = AdaptiveAttemptContext(
        student_id="student",
        attempt_id="attempt",
        skill="Target",
        mastery_before=0.4,
    )
    outcome = AttemptLearningOutcome(
        attempt_id="attempt",
        skill="Target",
        mastery_before=0.4,
        mastery_after=0.5,
        delta_mastery=0.1,
    )

    with pytest.raises(FrozenInstanceError):
        context.mastery_before = 0.2
    with pytest.raises(FrozenInstanceError):
        outcome.delta_mastery = 0.0
    with pytest.raises(ValueError):
        AttemptLearningOutcome("a", "Target", 0.4, 0.5, 0.2)
    with pytest.raises(ValueError):
        AttemptLearningOutcome("a", "Target", -0.1, 0.5, 0.6)
    with pytest.raises(ValueError):
        AttemptLearningOutcome("a", "Target", 0.4, float("nan"), 0.0)
    with pytest.raises(TypeError):
        AdaptiveAttemptContext("s", "a", "Target", True)

    assert outcome.as_dict() == {
        "attempt_id": "attempt",
        "skill": "Target",
        "mastery_before": 0.4,
        "mastery_after": 0.5,
        "delta_mastery": 0.1,
    }


def test_legacy_pipeline_call_retains_exact_result_keys(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor([(0, "Target")]),
        evaluator=StructuredEvaluator(("Target",), {"Target": "correct"}),
    )

    result = pipeline.process_transcript(
        transcript=student_transcript("correct"),
        student_id="student",
        session_id="legacy-session",
    )

    assert set(result) == {
        "raw_extraction",
        "evaluated_extraction",
        "resolved_events",
        "knowledge_graph_result",
        "learning_path",
    }


def test_multiple_same_skill_observations_produce_one_session_outcome(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor(
            [(0, "Target"), (1, "Target"), (1, "Other")]
        ),
        evaluator=StructuredEvaluator(
            ("Target",),
            {"Target": "correct"},
        ),
        detectors=HighDetectors(),
    )
    context = pipeline.start_attempt(
        student_id="student",
        attempt_id="adaptive-1",
    )

    result = pipeline.process_transcript(
        transcript=student_transcript("first", "second"),
        student_id="student",
        session_id="session-1",
        attempt_context=context,
    )

    outcome = result["learning_outcome"]
    assert outcome["attempt_id"] == "adaptive-1"
    assert outcome["skill"] == "Target"
    assert outcome["mastery_after"] > outcome["mastery_before"]
    assert outcome["delta_mastery"] == pytest.approx(
        outcome["mastery_after"] - outcome["mastery_before"]
    )
    assert len(graph.get_attempts("student", "Target")) == 2
    assert graph.get_attempts("student", "Other") == []
    assert len(result["knowledge_graph_result"]["skills_updated"]) == 1
    assert outcome["mastery_after"] == pytest.approx(
        graph.get_mastery("student", "Target")["mastery_probability"]
    )


def test_three_observation_outcome_uses_final_bkt_mastery_not_raw_percentage(
    monkeypatch,
    tmp_path,
):
    graph, predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor(
            [(0, "Target"), (1, "Target"), (2, "Target")]
        ),
        evaluator=StructuredEvaluator(
            ("Target",),
            {
                ("Target", 0): "correct",
                ("Target", 1): "correct",
                ("Target", 2): "incorrect",
            },
        ),
    )
    context = pipeline.start_attempt(
        student_id="student",
        attempt_id="two-of-three",
    )

    result = pipeline.process_transcript(
        transcript=student_transcript("yes", "yes", "no"),
        student_id="student",
        session_id="two-of-three-session",
        attempt_context=context,
    )
    outcome = result["learning_outcome"]
    prior = graph.get_effective_initial_prior(
        "student", "Target"
    )["effective_initial_prior"]
    expected = predictor.predict(
        "Target",
        graph.get_attempts("student", "Target"),
        initial_prior=prior,
    )

    assert len(graph.get_attempts("student", "Target")) == 3
    assert outcome["mastery_after"] == pytest.approx(expected)
    assert outcome["mastery_after"] == pytest.approx(
        graph.get_mastery("student", "Target")["mastery_probability"]
    )
    assert outcome["mastery_after"] != pytest.approx(2 / 3)
    assert outcome["delta_mastery"] == pytest.approx(
        outcome["mastery_after"] - outcome["mastery_before"]
    )


def test_two_failed_attempts_are_continuous_and_negative(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
        transferred_prior=0.75,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor([(0, "Target")]),
        evaluator=StructuredEvaluator(
            ("Target",),
            {"Target": "incorrect"},
        ),
    )

    first_context = pipeline.start_attempt(
        student_id="student",
        attempt_id="failed-1",
    )
    first = pipeline.process_transcript(
        transcript=student_transcript("wrong"),
        student_id="student",
        session_id="failed-session-1",
        attempt_context=first_context,
    )["learning_outcome"]

    second_context = pipeline.start_attempt(
        student_id="student",
        attempt_id="failed-2",
    )
    second = pipeline.process_transcript(
        transcript=student_transcript("wrong again"),
        student_id="student",
        session_id="failed-session-2",
        attempt_context=second_context,
    )["learning_outcome"]

    assert first["delta_mastery"] < 0
    assert second["delta_mastery"] < 0
    assert first["mastery_after"] == pytest.approx(
        second["mastery_before"]
    )


def test_no_target_observation_returns_zero_target_outcome(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor([(0, "Other")]),
        evaluator=StructuredEvaluator(("Target",), {}),
    )
    context = pipeline.start_attempt(
        student_id="student",
        attempt_id="zero",
    )

    result = pipeline.process_transcript(
        transcript=student_transcript("not an answer"),
        student_id="student",
        session_id="zero-session",
        attempt_context=context,
    )
    outcome = result["learning_outcome"]

    assert outcome["skill"] == "Target"
    assert outcome["mastery_after"] == pytest.approx(
        outcome["mastery_before"]
    )
    assert outcome["delta_mastery"] == pytest.approx(0.0)
    assert graph.get_attempts("student", "Target") == []
    assert graph.get_attempts("student", "Other") == []


def test_explicit_target_remains_authoritative_when_two_skills_update(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor([(0, "Target"), (0, "Other")]),
        evaluator=StructuredEvaluator(
            ("Target", "Other"),
            {"Target": "correct", "Other": "correct"},
        ),
    )
    context = pipeline.start_attempt(
        student_id="student",
        attempt_id="two-skills",
        attempt_skill="Target",
    )

    result = pipeline.process_transcript(
        transcript=student_transcript("two skills"),
        student_id="student",
        session_id="two-skills-session",
        attempt_context=context,
    )

    assert {
        update["skill"]
        for update in result["knowledge_graph_result"]["skills_updated"]
    } == {"Target", "Other"}
    assert result["learning_outcome"]["skill"] == "Target"


def test_ambiguous_or_unauthorized_target_fails_before_knowledge_graph_call():
    class NeverCalledGraph:
        def __init__(self):
            self.called = False

        def start_attempt(self, **kwargs):
            self.called = True
            raise AssertionError("must fail before KnowledgeGraph mutation")

    graph = NeverCalledGraph()
    evaluator = StructuredEvaluator(("Target", "Other"), {})
    pipeline = build_pipeline(
        graph,
        extractor=StaticExtractor([]),
        evaluator=evaluator,
    )

    with pytest.raises(ValueError, match="explicit attempt_skill"):
        pipeline.start_attempt(student_id="student", attempt_id="ambiguous")
    with pytest.raises(ValueError, match="not authorized"):
        pipeline.start_attempt(
            student_id="student",
            attempt_id="unauthorized",
            attempt_skill="Related",
        )
    assert graph.called is False


def test_learning_outcome_is_deterministic_from_identical_persisted_state(
    monkeypatch,
    tmp_path,
):
    def run_once(path, suffix):
        graph, _predictor, _cold_start, _db_path = build_graph(
            monkeypatch,
            path,
            transferred_prior=0.55,
        )
        pipeline = build_pipeline(
            graph,
            extractor=StaticExtractor([(0, "Target")]),
            evaluator=StructuredEvaluator(
                ("Target",),
                {"Target": "correct"},
            ),
        )
        graph.start_attempt(
            student_id="student",
            attempt_id="seed",
            skill="Target",
        )
        graph.process_resolved_events(
            student_id="student",
            resolved_events=[
                SimpleNamespace(
                    event_id="seed-event",
                    skill_id="Target",
                    primary_signal=SimpleNamespace(value="correct_answer"),
                    behaviour=SimpleNamespace(
                        reasoning_probability=None,
                        uncertainty_probability=None,
                        clarification_probability=None,
                    ),
                    history=SimpleNamespace(
                        repeated_misunderstanding=False
                    ),
                    bkt_update=SimpleNamespace(
                        should_update=True,
                        outcome=1,
                        update_confidence=1.0,
                    ),
                )
            ],
            session_id=f"seed-session-{suffix}",
        )
        context = pipeline.start_attempt(
            student_id="student",
            attempt_id="deterministic",
        )
        return pipeline.process_transcript(
            transcript=student_transcript("same evidence"),
            student_id="student",
            session_id=f"session-{suffix}",
            attempt_context=context,
        )["learning_outcome"]

    first = run_once(tmp_path / "first", "first")
    second = run_once(tmp_path / "second", "second")
    assert first == second
