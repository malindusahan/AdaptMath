"""Focused contracts for the inactive MD7-proportional warm-start mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


MOVE_ROOT = Path(__file__).resolve().parents[3] / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.randomized_warmstart import (  # noqa: E402
    BEHAVIOR_POLICY_ID,
    MD7ProportionalAssigner,
)
from src.self_improvement.turn_lints_context import (  # noqa: E402
    BLOCK_ORDER,
    context_schema_id,
    feature_names_for_blocks,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DIRECT_ARMS,
    DirectTurnLinTS,
)
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
    DirectTurnLinTSPipeline,
    TurnActionLedger,
    TurnLinTSMode,
)


SELECTOR_SHA = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
PROBABILITIES = {
    "generic": 0.10,
    "probing": 0.20,
    "focus": 0.30,
    "telling": 0.40,
}
MRB1 = {
    "Mistake_Identification": 0.1,
    "Mistake_Location": 0.2,
    "Providing_Guidance": 0.3,
    "Actionability": 0.4,
}
PRE_ACTION = {
    "mastery_before": 0.4,
    "previous_mastery_delta": None,
    "previous_learner_signals": None,
}


class FixedRNG:
    def __init__(self, draw: float) -> None:
        self.draw = draw

    def random(self) -> float:
        return self.draw


def make_policy() -> DirectTurnLinTS:
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(BLOCK_ORDER),
        enabled_context_blocks=BLOCK_ORDER,
        feature_names=feature_names_for_blocks(BLOCK_ORDER),
        selector_version="MD7-R2-TELL-C1 epoch 2",
        selector_sha256=SELECTOR_SHA,
        reward_mode="headroom_normalized",
    )


def make_controller(tmp_path: Path, *, draw: float = 0.75) -> DirectTurnController:
    return DirectTurnController(
        policy=make_policy(),
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=BLOCK_ORDER,
        state_path=tmp_path / "policy_state.json",
        ledger=TurnActionLedger(tmp_path),
        warmstart_assigner=MD7ProportionalAssigner(rng=FixedRNG(draw)),
    )


@pytest.mark.parametrize(
    ("draw", "expected"),
    [(0.05, "generic"), (0.15, "probing"), (0.45, "focus"), (0.75, "telling")],
)
def test_raw_probabilities_assign_all_arms_without_filtering(
    draw: float, expected: str
) -> None:
    assignment = MD7ProportionalAssigner(rng=FixedRNG(draw)).assign(PROBABILITIES)
    assert assignment.selected_move == expected
    assert assignment.behavior_propensity == PROBABILITIES[expected]
    assert assignment.full_probability_vector == PROBABILITIES
    assert assignment.randomized_assignment is True
    assert assignment.behavior_policy == BEHAVIOR_POLICY_ID


def test_probability_validation_has_no_floor_or_subset_renormalization() -> None:
    probabilities = {
        "generic": 0.0,
        "probing": 0.000001,
        "focus": 0.499999,
        "telling": 0.5,
    }
    assignment = MD7ProportionalAssigner(rng=FixedRNG(0.0000005)).assign(
        probabilities
    )
    assert assignment.selected_move == "probing"
    assert assignment.behavior_propensity == pytest.approx(0.000001)
    assert assignment.full_probability_vector["generic"] == 0.0
    with pytest.raises(ValueError, match="exactly the four"):
        MD7ProportionalAssigner().assign({"telling": 1.0})


def test_seeded_assignment_is_reproducible() -> None:
    first = MD7ProportionalAssigner(seed=20260829)
    second = MD7ProportionalAssigner(seed=20260829)
    first_sequence = [first.assign(PROBABILITIES).selected_move for _ in range(20)]
    second_sequence = [second.assign(PROBABILITIES).selected_move for _ in range(20)]
    assert first_sequence == second_sequence


def test_warmstart_event_logs_exact_propensity_and_never_updates(
    tmp_path: Path,
) -> None:
    controller = make_controller(tmp_path)
    controller.start_attempt(0.4, attempt_id="warm-attempt")
    decision = controller.select_turn(PROBABILITIES, PRE_ACTION)
    assert decision.final_move == "telling"
    assert decision.selected_arm == "telling"
    assert decision.hypothetical_arm is None
    assert decision.behavior_propensity == 0.4
    assert decision.full_behavior_probability_vector == PROBABILITIES
    controller.record_mrb1_scores(MRB1, "Here is the explicit next step.")
    outcome = controller.complete_outcome(
        action_event_id=decision.action_event_id,
        should_update=True,
        mastery_before=0.4,
        mastery_after=0.5,
        resolver_event_id="resolver-warm",
        previous_learner_signals=None,
    )
    assert outcome["raw_delta"] == pytest.approx(0.1)
    assert outcome["headroom_normalized_delta"] == pytest.approx(1 / 6)
    assert outcome["posterior_updated"] is False
    assert controller.policy.total_updates == 0
    assert controller.policy.arm_update_counts == {arm: 0 for arm in DIRECT_ARMS}

    event = json.loads((tmp_path / "turn_events.jsonl").read_text(encoding="utf-8"))
    assert event["schema_version"] == "adaptmath_turn_lints_event_v4"
    assert event["actual_treatment_move"] == "telling"
    assert event["behavior_policy"] == BEHAVIOR_POLICY_ID
    assert event["behavior_propensity"] == 0.4
    assert event["full_behavior_probability_vector"] == PROBABILITIES
    assert event["randomized_assignment"] is True
    assert event["posterior_updated"] is False
    assert event["context"]["enabled_blocks"] == list(BLOCK_ORDER)
    assert event["context"]["context_dimension"] == 22
    assert all(
        not name.startswith("current_")
        for name in event["context"]["feature_names"]
    )


def test_censored_and_externally_forced_actions_are_semantically_distinct(
    tmp_path: Path,
) -> None:
    censored = make_controller(tmp_path / "censored")
    censored.start_attempt(0.4, attempt_id="censored-attempt")
    censored.select_turn(PROBABILITIES, PRE_ACTION)
    censored.abort_attempt()
    event = json.loads(
        (tmp_path / "censored" / "turn_events.jsonl").read_text(encoding="utf-8")
    )
    assert event["outcome_observed"] is False
    assert event["configured_reward_value"] is None
    assert event["posterior_updated"] is False
    assert event["resolution_status"] == "censored_no_response"

    forced = make_controller(tmp_path / "forced")
    forced.start_attempt(0.4, attempt_id="forced-attempt")
    decision = forced.select_turn(
        PROBABILITIES,
        PRE_ACTION,
        externally_forced_move="focus",
        external_assignment_source="authorized_manual_test_override",
    )
    assert decision.final_move == "focus"
    assert decision.randomized_assignment is False
    assert decision.behavior_policy == "external_override"
    assert decision.behavior_propensity is None
    forced.abort_attempt()
    forced_event = json.loads(
        (tmp_path / "forced" / "turn_events.jsonl").read_text(encoding="utf-8")
    )
    assert forced_event["randomized_assignment"] is False
    assert forced_event["behavior_propensity"] is None
    assert forced_event["treatment_assignment_source"] == (
        "authorized_manual_test_override"
    )


def test_sampled_move_reaches_tutor_generation_path(tmp_path: Path) -> None:
    class Selector:
        def predict_probabilities(self, _problem: str, _history: object) -> dict[str, float]:
            return dict(PROBABILITIES)

    class Critic:
        def score_response(self, _history: object, _response: str) -> dict[str, float]:
            return dict(MRB1)

    class Tutor:
        received_move: str | None = None

        def generate(self, *, problem: str, conversation_history: object,
                     pedagogical_move: str) -> str:
            del problem, conversation_history
            self.received_move = pedagogical_move
            return "Use the stated method on the next step."

    class MemoryAdapter:
        @staticmethod
        def get_attempt_id(memory: dict[str, object]) -> object:
            return memory["attempt_id"]

        @staticmethod
        def get_mastery_before(memory: dict[str, object]) -> object:
            return memory["mastery_before"]

        @staticmethod
        def get_problem(memory: dict[str, object]) -> object:
            return memory["problem"]

        @staticmethod
        def get_conversation_history(memory: dict[str, object]) -> object:
            return memory["history"]

        @staticmethod
        def get_turn_lints_pre_action_state(_memory: object) -> dict[str, object]:
            return dict(PRE_ACTION)

        @staticmethod
        def append_tutor_response(memory: dict[str, object], response: str) -> None:
            memory["history"].append({"role": "tutor", "content": response})

    # MD7/SHADOW top-1 is telling, while this raw draw selects probing. This
    # proves warm-start changes the actual treatment rather than logging a
    # shadow hypothetical action.
    controller = make_controller(tmp_path, draw=0.15)
    pipeline = DirectTurnLinTSPipeline(
        selector=Selector(),
        mrb1=Critic(),
        controller=controller,
        memory_adapter=MemoryAdapter(),
    )
    tutor = Tutor()
    memory: dict[str, object] = {
        "attempt_id": "pipeline-attempt",
        "mastery_before": 0.4,
        "problem": "Solve x + 2 = 4.",
        "history": [],
    }
    pipeline.start_attempt(memory)
    result = pipeline.run_tutor_turn(memory, tutor)
    assert result["pedagogical_move"] == "probing"
    assert tutor.received_move == "probing"
    assert result["base_move"] == "telling"
    assert result["selected_arm"] == "probing"
    assert result["adaptive_decision"]["hypothetical_arm"] is None
    assert result["adaptive_decision"]["behavior_propensity"] == 0.2
    pipeline.abort_attempt(memory)


def test_safe_launcher_default_and_explicit_skl_warmstart_contract() -> None:
    assert TurnLinTSMode.parse("SHADOW") is TurnLinTSMode.SHADOW
    launcher = (Path(__file__).resolve().parents[3] / "start-adaptmath-local.ps1").read_text(
        encoding="utf-8"
    )
    assert "else { 'SHADOW' }" in launcher
    assert "[switch]$FinalSKLWarmstart" in launcher
    assert "turn_lints_randomized_warmstart_skl_final_v1" in launcher
    assert "turnLinTSDiagnosticContextBlocks = 'S+K+L+H+Q'" in launcher
