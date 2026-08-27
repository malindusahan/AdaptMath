from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import core.knowledge_graph as knowledge_graph_module
from bkt.predict import BKTPredictor
from core.knowledge_graph import (
    HistoricalEffectivePriorUnavailableError,
    KnowledgeGraph,
)
from db.database import initialise_database


PARAMS = {
    "Target": {
        "prior": 0.20,
        "learns": 0.10,
        "guesses": 0.20,
        "slips": 0.10,
        "forgets": 0.0,
    },
    "Related": {
        "prior": 0.25,
        "learns": 0.10,
        "guesses": 0.20,
        "slips": 0.10,
        "forgets": 0.0,
    },
    "Other": {
        "prior": 0.30,
        "learns": 0.10,
        "guesses": 0.20,
        "slips": 0.10,
        "forgets": 0.0,
    },
}


class ControlledColdStart:
    def __init__(self, transferred_prior: float | None = None):
        self.transferred_prior = transferred_prior
        self.calls = 0

    def compute_prior(
        self,
        *,
        skill: str,
        student_masteries: dict,
        population_prior: float,
    ) -> dict:
        self.calls += 1
        prior = (
            population_prior
            if self.transferred_prior is None
            else self.transferred_prior
        )
        return {
            "prior": prior,
            "used_transfer": prior > population_prior,
            "related_skills_used": sorted(student_masteries),
            "transfer_evidence": "controlled test prior",
        }


def resolved_event(
    *,
    event_id: str,
    skill: str = "Target",
    outcome: int = 1,
    confidence: float = 1.0,
):
    return SimpleNamespace(
        event_id=event_id,
        skill_id=skill,
        primary_signal=SimpleNamespace(
            value="correct_answer" if outcome else "incorrect_answer"
        ),
        behaviour=SimpleNamespace(
            reasoning_probability=None,
            uncertainty_probability=None,
            clarification_probability=None,
        ),
        history=SimpleNamespace(repeated_misunderstanding=False),
        bkt_update=SimpleNamespace(
            should_update=True,
            outcome=outcome,
            update_confidence=confidence,
        ),
    )


@contextmanager
def sqlite_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_graph(monkeypatch, tmp_path, *, transferred_prior=None):
    db_path = tmp_path / "student-model.sqlite3"

    def get_temp_connection():
        return sqlite_connection(db_path)

    def initialise_temp_database():
        initialise_database(db_path)

    monkeypatch.setattr(
        knowledge_graph_module,
        "get_connection",
        get_temp_connection,
    )
    monkeypatch.setattr(
        knowledge_graph_module,
        "initialise_database",
        initialise_temp_database,
    )

    cold_start = ControlledColdStart(transferred_prior)
    predictor = BKTPredictor(PARAMS)
    graph = KnowledgeGraph(
        predictor=predictor,
        cold_start=cold_start,
    )
    return graph, predictor, cold_start, db_path


def table_count(db_path, table):
    with sqlite_connection(db_path) as conn:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]


def test_schema_migration_is_additive_idempotent_and_does_not_guess_legacy_prior(
    tmp_path,
):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE students (
                student_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                profile_json TEXT
            );
            CREATE TABLE mastery (
                student_id TEXT,
                skill_name TEXT,
                mastery_probability REAL NOT NULL,
                mastery_label TEXT NOT NULL,
                previous_mastery_probability REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (student_id, skill_name)
            );
            CREATE TABLE attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                correct INTEGER NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                signal_type TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                concept_count INTEGER
            );
            INSERT INTO students (student_id) VALUES ('legacy');
            INSERT INTO mastery (
                student_id, skill_name, mastery_probability, mastery_label
            ) VALUES ('legacy', 'Target', 0.73, 'strong');
            INSERT INTO students (student_id) VALUES ('attempts-only');
            INSERT INTO attempts (
                student_id, skill_name, correct, confidence
            ) VALUES ('attempts-only', 'Other', 1, 1.0);
            """
        )

    initialise_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE bkt_initial_priors
            SET effective_initial_prior = 0.42,
                prior_source = 'historical_population_rebase'
            WHERE student_id = 'legacy' AND skill_name = 'Target'
            """
        )
        conn.execute("PRAGMA user_version = 7")

    initialise_database(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        legacy_mastery = conn.execute(
            """
            SELECT mastery_probability FROM mastery
            WHERE student_id = 'legacy' AND skill_name = 'Target'
            """
        ).fetchone()[0]
        prior_rows = conn.execute(
            """
            SELECT student_id, skill_name, effective_initial_prior, prior_source
            FROM bkt_initial_priors
            ORDER BY student_id, skill_name
            """
        ).fetchall()
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]

    assert "bkt_initial_priors" in tables
    assert legacy_mastery == pytest.approx(0.73)
    assert prior_rows == [
        ("attempts-only", "Other", None, "legacy_unknown"),
        (
            "legacy",
            "Target",
            pytest.approx(0.42),
            "historical_population_rebase",
        ),
    ]
    assert schema_version == 7


def test_no_transfer_start_snapshots_and_persists_population_prior_read_only(
    monkeypatch,
    tmp_path,
):
    graph, predictor, cold_start, db_path = build_graph(
        monkeypatch,
        tmp_path,
    )

    context = graph.start_attempt("student", "attempt-1", "Target")

    assert context.mastery_before == pytest.approx(
        predictor.params["Target"]["prior"]
    )
    assert cold_start.calls == 1
    assert table_count(db_path, "attempts") == 0
    assert table_count(db_path, "sessions") == 0
    assert table_count(db_path, "mastery") == 0

    prior = graph.get_effective_initial_prior("student", "Target")
    assert prior["effective_initial_prior"] == pytest.approx(
        context.mastery_before
    )
    assert prior["prior_source"] == "population"


def test_transferred_prior_is_selected_once_reused_and_batch_equivalent(
    monkeypatch,
    tmp_path,
):
    graph, predictor, cold_start, db_path = build_graph(
        monkeypatch,
        tmp_path,
        transferred_prior=0.60,
    )

    graph.ensure_student("student")
    with sqlite_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO mastery
                (student_id, skill_name, mastery_probability, mastery_label)
            VALUES ('student', 'Related', 0.80, 'strong')
            """
        )

    first_context = graph.start_attempt(
        "student",
        "attempt-1",
        "Target",
    )
    assert first_context.mastery_before == pytest.approx(0.60)

    first_result = graph.process_resolved_events(
        student_id="student",
        resolved_events=[resolved_event(event_id="first", outcome=1)],
        session_id="session-1",
    )
    first_after = first_result["skills_updated"][0]["probability"]
    assert first_after > first_context.mastery_before

    with sqlite_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE mastery SET mastery_probability = 0.05
            WHERE student_id = 'student' AND skill_name = 'Related'
            """
        )

    second_context = graph.start_attempt(
        "student",
        "attempt-2",
        "Target",
    )
    assert second_context.mastery_before == pytest.approx(first_after)

    second_result = graph.process_resolved_events(
        student_id="student",
        resolved_events=[resolved_event(event_id="second", outcome=0)],
        session_id="session-2",
    )
    second_after = second_result["skills_updated"][0]["probability"]
    assert second_after < second_context.mastery_before
    assert cold_start.calls == 1

    attempts = graph.get_attempts("student", "Target")
    expected = predictor.predict(
        "Target",
        attempts,
        initial_prior=0.60,
    )
    persisted = graph.get_mastery("student", "Target")
    assert persisted["mastery_probability"] == pytest.approx(expected)
    assert second_after == pytest.approx(expected)
    assert graph.get_effective_initial_prior(
        "student", "Target"
    )["effective_initial_prior"] == pytest.approx(0.60)


def test_zero_confidence_observation_preserves_mastery_with_zero_forgetting(
    monkeypatch,
    tmp_path,
):
    graph, _predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    context = graph.start_attempt("student", "attempt-0", "Target")

    graph.process_resolved_events(
        student_id="student",
        resolved_events=[
            resolved_event(
                event_id="zero",
                outcome=0,
                confidence=0.0,
            )
        ],
        session_id="session-0",
    )
    after = graph.get_current_mastery_probability(
        "student",
        "Target",
        require_effective_prior=True,
    )

    assert after == pytest.approx(context.mastery_before)
    assert after - context.mastery_before == pytest.approx(0.0)


def test_process_session_uses_bkt_not_raw_two_of_three_percentage(
    monkeypatch,
    tmp_path,
):
    graph, predictor, _cold_start, _db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    context = graph.start_attempt("student", "two-of-three", "Target")

    result = graph.process_session(
        student_id="student",
        signals=[
            {"skill": "Target", "label": 1, "confidence": 1.0},
            {"skill": "Target", "label": 1, "confidence": 1.0},
            {"skill": "Target", "label": 0, "confidence": 1.0},
        ],
        session_id="two-of-three-session",
    )

    prior = graph.get_effective_initial_prior(
        "student", "Target"
    )["effective_initial_prior"]
    attempts = graph.get_attempts("student", "Target")
    expected = predictor.predict(
        "Target",
        attempts,
        initial_prior=prior,
    )
    after = result["skills_updated"][0]["probability"]

    assert context.mastery_before == pytest.approx(prior)
    assert after == pytest.approx(expected)
    assert after != pytest.approx(2 / 3)


def test_historical_row_fails_adaptive_start_then_rebases_explicitly(
    monkeypatch,
    tmp_path,
):
    graph, predictor, _cold_start, db_path = build_graph(
        monkeypatch,
        tmp_path,
    )
    graph.ensure_student("legacy")
    with sqlite_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO attempts
                (student_id, skill_name, correct, confidence)
            VALUES ('legacy', 'Target', 1, 1.0)
            """
        )
        conn.execute(
            """
            INSERT INTO mastery
                (student_id, skill_name, mastery_probability, mastery_label,
                 previous_mastery_probability)
            VALUES ('legacy', 'Target', 0.73, 'strong', 0.91)
            """
        )
        # This is the exact state produced by the additive migration for a
        # historical student-skill whose original transferred prior is not
        # reconstructable.
        conn.execute(
            """
            INSERT INTO bkt_initial_priors
                (student_id, skill_name, effective_initial_prior, prior_source)
            VALUES ('legacy', 'Target', NULL, 'legacy_unknown')
            """
        )

    with pytest.raises(HistoricalEffectivePriorUnavailableError):
        graph.start_attempt("legacy", "blocked", "Target")
    unknown_prior = graph.get_effective_initial_prior("legacy", "Target")
    assert unknown_prior["effective_initial_prior"] is None
    assert unknown_prior["prior_source"] == "legacy_unknown"

    result = graph.process_resolved_events(
        student_id="legacy",
        resolved_events=[resolved_event(event_id="rebase", outcome=0)],
        session_id="legacy-rebase",
    )
    update = result["skills_updated"][0]
    assert update["compatibility_rebased"] is True
    assert update["effective_initial_prior_source"] == (
        "historical_population_rebase"
    )
    assert update["effective_initial_prior"] == pytest.approx(
        predictor.params["Target"]["prior"]
    )
    expected = predictor.predict(
        "Target",
        graph.get_attempts("legacy", "Target"),
        initial_prior=predictor.params["Target"]["prior"],
    )
    assert update["probability"] == pytest.approx(expected)
    assert update["effective_initial_prior"] != pytest.approx(0.73)
    assert update["effective_initial_prior"] != pytest.approx(0.91)

    context = graph.start_attempt("legacy", "valid-now", "Target")
    assert context.mastery_before == pytest.approx(
        graph.get_mastery("legacy", "Target")["mastery_probability"]
    )
    follow_up = graph.process_resolved_events(
        student_id="legacy",
        resolved_events=[resolved_event(event_id="valid", outcome=1)],
        session_id="legacy-adaptive",
    )
    persisted_follow_up = graph.get_mastery("legacy", "Target")
    assert context.mastery_before == pytest.approx(
        persisted_follow_up["previous_mastery_probability"]
    )
    assert persisted_follow_up["mastery_probability"] == pytest.approx(
        follow_up["skills_updated"][0]["probability"]
    )
