import sqlite3

import pytest

import core.knowledge_graph as kg_module
from bkt.predict import BKTPredictor
from core.knowledge_graph import KnowledgeGraph
from core.signal_resolver import Correctness, ResolverInput, resolve_signal
from db.database import get_connection, initialise_database


PARAMS = {
    skill: {
        "prior": 0.2,
        "learns": 0.1,
        "guesses": 0.2,
        "slips": 0.1,
        "forgets": 0.0,
    }
    for skill in ("algebra", "fractions")
}


class PopulationColdStart:
    def compute_prior(self, *, population_prior, **kwargs):
        del kwargs
        return {
            "prior": population_prior,
            "used_transfer": False,
            "related_skills_used": [],
            "transfer_evidence": "fixture population prior",
        }


@pytest.fixture
def kg(monkeypatch, tmp_path):
    db_path = tmp_path / "student.db"
    initialise_database(db_path)
    monkeypatch.setattr(
        kg_module,
        "get_connection",
        lambda: get_connection(db_path),
    )
    monkeypatch.setattr(
        kg_module,
        "initialise_database",
        lambda: initialise_database(db_path),
    )
    graph = KnowledgeGraph(
        predictor=BKTPredictor(PARAMS),
        cold_start=PopulationColdStart(),
    )
    graph._test_db_path = db_path
    return graph


def _count(graph, table):
    with sqlite3.connect(graph._test_db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _resolved(
    event_id,
    correctness,
    *,
    skill="algebra",
    source_action_event_id=None,
    **kwargs,
):
    return resolve_signal(
        ResolverInput(
            event_id=event_id,
            source_action_event_id=source_action_event_id,
            student_id="student_1",
            skill_id=skill,
            correctness=correctness,
            **kwargs,
        )
    )


def test_multiple_behaviours_create_only_one_bkt_attempt(kg):
    resolved = _resolved(
        "event_1",
        Correctness.INCORRECT,
        uncertainty_probability=0.95,
        clarification_probability=0.90,
    )
    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_1",
    )
    assert _count(kg, "attempts") == 1
    assert result["behaviour_events"][0]["uncertainty_probability"] == 0.95
    assert result["behaviour_events"][0]["clarification_probability"] == 0.90
    assert len(result["skills_updated"]) == 1


def test_unknown_correctness_with_strong_behaviour_creates_one_weak_proxy(kg):
    resolved = _resolved(
        "event_2",
        Correctness.UNKNOWN,
        uncertainty_probability=0.85,
        clarification_probability=0.91,
    )
    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_2",
    )
    assert resolved.bkt_update.should_update is True
    assert resolved.bkt_update.outcome == 0
    assert 0.0 < resolved.bkt_update.update_confidence <= 0.25
    assert _count(kg, "attempts") == 1
    assert result["observation_results"][0]["observation_source"] == (
        "behavioural_proxy"
    )


def test_correct_uncertain_event_stays_positive(kg):
    resolved = _resolved(
        "event_3",
        Correctness.CORRECT,
        uncertainty_probability=0.90,
    )
    kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_3",
    )
    with sqlite3.connect(kg._test_db_path) as conn:
        outcome, confidence = conn.execute(
            "SELECT correct, confidence FROM attempts"
        ).fetchone()
    assert outcome == 1
    assert confidence == pytest.approx(resolved.bkt_update.update_confidence)


def test_partial_correct_uses_resolver_confidence(kg):
    resolved = _resolved(
        "event_4",
        Correctness.PARTIAL,
        evaluator_confidence=0.80,
    )
    kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_4",
    )
    with sqlite3.connect(kg._test_db_path) as conn:
        confidence = conn.execute("SELECT confidence FROM attempts").fetchone()[0]
    assert confidence == pytest.approx(0.32)


def test_multiple_events_same_skill_remain_two_observations(kg):
    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[
            _resolved("event_5", Correctness.INCORRECT),
            _resolved("event_6", Correctness.CORRECT),
        ],
        session_id="session_5",
    )
    assert _count(kg, "attempts") == 2
    assert _count(kg, "resolved_events") == 2
    assert len(result["observation_results"]) == 2
    assert len(result["skills_updated"]) == 1


def test_multiple_skills_are_updated_separately(kg):
    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[
            _resolved("event_7", Correctness.CORRECT),
            _resolved("event_8", Correctness.INCORRECT, skill="fractions"),
        ],
        session_id="session_6",
    )
    assert _count(kg, "attempts") == 2
    assert {item["skill"] for item in result["skills_updated"]} == {
        "algebra",
        "fractions",
    }


def test_repeated_misunderstanding_is_metadata_not_extra_attempt(kg):
    resolved = _resolved(
        "event_9",
        Correctness.INCORRECT,
        recent_same_skill_outcomes=[Correctness.INCORRECT],
    )
    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="session_7",
    )
    assert resolved.history.repeated_misunderstanding is True
    assert _count(kg, "attempts") == 1
    assert result["behaviour_events"][0]["repeated_misunderstanding"] is True


def test_duplicate_resolver_event_is_durably_idempotent(kg):
    action_id = "thread:2:action:1"
    resolved = _resolved(
        f"{action_id}:resolver:algebra",
        Correctness.CORRECT,
        source_action_event_id=action_id,
    )
    first = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="thread:2:dialogue:1",
    )
    second = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="thread:2:dialogue:1",
    )
    assert _count(kg, "attempts") == 1
    assert _count(kg, "resolved_events") == 1
    assert second["observation_results"][0]["persistence_status"] == (
        "already_persisted"
    )
    assert second["observation_results"][0]["delta_mastery"] == pytest.approx(
        first["observation_results"][0]["delta_mastery"]
    )


def test_failure_after_observation_insert_rolls_back_and_retry_converges(
    kg,
    monkeypatch,
):
    resolved = _resolved("retry-event", Correctness.CORRECT)
    calls = 0

    def fail_once(conn, event_id):
        del conn, event_id
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated failure after observation insert")

    monkeypatch.setattr(kg, "_after_resolved_observation_insert", fail_once)
    with pytest.raises(RuntimeError, match="simulated failure"):
        kg.process_resolved_events(
            student_id="student_1",
            resolved_events=[resolved],
            session_id="retry-session",
        )
    assert _count(kg, "resolved_events") == 0
    assert _count(kg, "attempts") == 0
    assert _count(kg, "mastery") == 0

    result = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="retry-session",
    )
    assert _count(kg, "resolved_events") == 1
    assert _count(kg, "attempts") == 1
    assert _count(kg, "mastery") == 1
    assert result["observation_results"][0]["persistence_status"] == "persisted"


def test_no_update_event_is_durably_ledgered_without_bkt_evidence(kg):
    resolved = _resolved("no-update", Correctness.UNKNOWN)
    first = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="no-update-session",
    )
    second = kg.process_resolved_events(
        student_id="student_1",
        resolved_events=[resolved],
        session_id="no-update-session",
    )
    assert first["observation_results"][0]["should_update"] is False
    assert second["observation_results"][0]["persistence_status"] == (
        "already_persisted"
    )
    assert _count(kg, "resolved_events") == 1
    assert _count(kg, "attempts") == 0
