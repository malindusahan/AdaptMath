from __future__ import annotations

import sys
from pathlib import Path

import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STUDENT_ROOT = WORKSPACE_ROOT / "student-modeling"
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
for root in (STUDENT_ROOT, MOVE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import core.knowledge_graph as knowledge_graph_module
from bkt.predict import BKTPredictor
from core.knowledge_graph import KnowledgeGraph
from core.signal_resolver import Correctness, ResolverInput, resolve_signal
from db.database import get_connection, initialise_database
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.lints_policy import TrueDisjointLinTS
from src.self_improvement.turn_context_builder import MRB1_TASKS
from src.self_improvement.turn_level_controller import TurnLevelAttemptController


PROBABILITIES = dict(
    zip(MOVE_ORDER, (0.40, 0.30, 0.20, 0.10), strict=True)
)
QUALITY = {task: 0.5 for task in MRB1_TASKS}
PARAMS = {
    "algebra": {
        "prior": 0.2,
        "learns": 0.1,
        "guesses": 0.2,
        "slips": 0.1,
        "forgets": 0.0,
    }
}


class PopulationColdStart:
    def compute_prior(self, *, population_prior, **kwargs):
        del kwargs
        return {
            "prior": population_prior,
            "used_transfer": False,
            "related_skills_used": [],
            "transfer_evidence": "fixture",
        }


def test_reteaching_action_indices_reset_and_bkt_retains_exact_identity(
    monkeypatch,
    tmp_path,
):
    policy = TrueDisjointLinTS(context_dim=9, seed=7, data_mode="synthetic")
    controller = TurnLevelAttemptController(policy)

    controller.start_attempt(0.2, attempt_id="thread:1")
    attempt_1 = []
    for _ in range(3):
        attempt_1.append(controller.select_turn(PROBABILITIES))
        controller.record_mrb1_scores(QUALITY)
    controller.abort_attempt()

    controller.start_attempt(0.2, attempt_id="thread:2")
    attempt_2 = []
    for _ in range(2):
        attempt_2.append(controller.select_turn(PROBABILITIES))
        controller.record_mrb1_scores(QUALITY)

    assert [item.action_turn_index for item in attempt_1] == [1, 2, 3]
    assert [item.action_turn_index for item in attempt_2] == [1, 2]
    assert [item.action_event_id for item in attempt_2] == [
        "thread:2:action:1",
        "thread:2:action:2",
    ]

    db_path = tmp_path / "identity.db"
    initialise_database(db_path)
    monkeypatch.setattr(
        knowledge_graph_module,
        "get_connection",
        lambda: get_connection(db_path),
    )
    monkeypatch.setattr(
        knowledge_graph_module,
        "initialise_database",
        lambda: initialise_database(db_path),
    )
    graph = KnowledgeGraph(
        predictor=BKTPredictor(PARAMS),
        cold_start=PopulationColdStart(),
    )
    graph.start_attempt("student", "thread:2", "algebra")

    observed_pairs = []
    for decision in attempt_2:
        resolver_event_id = f"{decision.action_event_id}:resolver:algebra"
        resolved = resolve_signal(
            ResolverInput(
                event_id=resolver_event_id,
                source_action_event_id=decision.action_event_id,
                student_id="student",
                skill_id="algebra",
                correctness=Correctness.CORRECT,
                evaluator_confidence=0.5,
            )
        )
        result = graph.process_resolved_events(
            student_id="student",
            resolved_events=[resolved],
            session_id=(
                f"thread:2:dialogue:{decision.action_turn_index}"
            ),
        )
        observation = result["observation_results"][0]
        observed_pairs.append(
            (
                observation["source_action_event_id"],
                observation["event_id"],
            )
        )
        assert observation["mastery_after"] - observation["mastery_before"] == (
            pytest.approx(observation["delta_mastery"])
        )

    assert observed_pairs == [
        (
            "thread:2:action:1",
            "thread:2:action:1:resolver:algebra",
        ),
        (
            "thread:2:action:2",
            "thread:2:action:2:resolver:algebra",
        ),
    ]
