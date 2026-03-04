"""Focused scientific/runtime contract for the frozen Turn-LinTS v1 design."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


MOVE_ROOT = Path(__file__).resolve().parents[3] / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.explicit_learner_agency import (  # noqa: E402
    AGENCY_ASSIGNMENT_SOURCE,
    AGENCY_BEHAVIOR_POLICY,
    decide_explicit_learner_agency,
)
from src.self_improvement.randomized_warmstart import (  # noqa: E402
    BEHAVIOR_POLICY_ID,
    MD7ProportionalAssigner,
)
from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    ANCHOR_GAMMA,
    ANCHOR_MODE,
    ARMS,
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    FORMAL_ASSESSMENT_ROLE,
    HANDWRITTEN_PEDAGOGICAL_STATE_ACTION_RULES,
    ORDERED_CONTEXT_FEATURES,
    P1_ANCHOR_ACTIVE,
    P2_ACTIVE,
    REWARD_MODE,
    SELECTOR_NAME,
    TAU_GATE,
)
from src.self_improvement.turn_lints_context import (  # noqa: E402
    build_turn_lints_context,
    context_schema_id,
    parse_context_blocks,
)
from src.self_improvement.turn_lints_policy import DirectTurnLinTS  # noqa: E402
from src.self_improvement.turn_lints_reward import turn_rewards  # noqa: E402
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
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


class FixedRNG:
    def __init__(self, draw: float) -> None:
        self.draw = draw

    def random(self) -> float:
        return self.draw


def make_policy() -> DirectTurnLinTS:
    blocks = parse_context_blocks(CONTEXT_PRESET)
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(blocks),
        enabled_context_blocks=blocks,
        feature_names=ORDERED_CONTEXT_FEATURES,
        selector_version=SELECTOR_NAME,
        selector_sha256=SELECTOR_SHA,
        reward_mode=REWARD_MODE,
        anchor_mode=ANCHOR_MODE,
        anchor_gamma=ANCHOR_GAMMA,
    )


def test_exact_context_order_dimension_and_no_same_turn_leakage() -> None:
    context = build_turn_lints_context(
        enabled_blocks=CONTEXT_PRESET,
        selector_probabilities=PROBABILITIES,
        mastery_before=0.25,
        previous_mastery_delta=0.2,
        previous_learner_signals={
            "reasoning_probability": 0.7,
            "uncertainty_probability": 0.2,
            "clarification_probability": 0.1,
        },
        previous_move="focus",
        prior_tutor_turn_count=3,
        previous_mrb1_scores={
            "Mistake_Identification": 0.1,
            "Mistake_Location": 0.2,
            "Providing_Guidance": 0.3,
            "Actionability": 0.4,
        },
    )
    assert context.feature_names == ORDERED_CONTEXT_FEATURES
    assert CONTEXT_PRESET == "S+K+L"
    assert context.dimension == CONTEXT_DIMENSION == 11
    assert context.values == (
        0.1, 0.2, 0.3, 0.4,
        0.25, 0.2, 0.0,
        0.7, 0.2, 0.1, 0.0,
    )
    joined = " ".join(context.feature_names).casefold()
    assert "mastery_after" not in joined
    assert "current_mrb1" not in joined
    assert "previous_move" not in joined
    assert "previous_mrb1" not in joined
    assert "prior_tutor_turn_count" not in joined


def test_missing_previous_state_is_distinct_from_real_zero() -> None:
    common = {
        "enabled_blocks": CONTEXT_PRESET,
        "selector_probabilities": PROBABILITIES,
        "mastery_before": 0.4,
        "previous_move": None,
        "prior_tutor_turn_count": 0,
        "previous_mrb1_scores": None,
    }
    missing = build_turn_lints_context(
        **common,
        previous_mastery_delta=None,
        previous_learner_signals=None,
    )
    real_zero = build_turn_lints_context(
        **common,
        previous_mastery_delta=0.0,
        previous_learner_signals={
            "reasoning_probability": 0.0,
            "uncertainty_probability": 0.0,
            "clarification_probability": 0.0,
        },
    )
    missing_values = dict(zip(missing.feature_names, missing.values, strict=True))
    real_values = dict(zip(real_zero.feature_names, real_zero.values, strict=True))
    assert missing_values["previous_mastery_delta_missing"] == 1.0
    assert missing_values["previous_learner_signals_missing"] == 1.0
    assert real_values["previous_mastery_delta_missing"] == 0.0
    assert real_values["previous_learner_signals_missing"] == 0.0


@pytest.mark.parametrize(
    ("before", "after"),
    [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1e-12, 0.9), (0.999999999999, 0.1)],
)
def test_selected_reward_is_finite_bounded_and_sign_preserving(
    before: float, after: float
) -> None:
    reward = turn_rewards(before, after)[REWARD_MODE]
    assert math.isfinite(reward)
    assert -1.0 <= reward <= 1.0
    assert math.copysign(1.0, reward) == math.copysign(1.0, after - before) or reward == 0.0


def test_all_arms_available_with_no_gate_filter_anchor_or_rules() -> None:
    assert ARMS == ("generic", "probing", "focus", "telling")
    assert TAU_GATE is None
    assert ANCHOR_MODE == "none" and ANCHOR_GAMMA == 0.0
    assert P1_ANCHOR_ACTIVE is False and P2_ACTIVE is False
    assert HANDWRITTEN_PEDAGOGICAL_STATE_ACTION_RULES is False
    assignments = [
        MD7ProportionalAssigner(rng=FixedRNG(draw)).assign(PROBABILITIES)
        for draw in (0.05, 0.15, 0.45, 0.75)
    ]
    assert tuple(item.selected_move for item in assignments) == ARMS
    assert all(item.behavior_policy == BEHAVIOR_POLICY_ID for item in assignments)
    assert all(
        item.behavior_propensity == PROBABILITIES[item.selected_move]
        for item in assignments
    )


def test_warmstart_does_not_mutate_posterior_and_agency_is_nonrandomized(
    tmp_path: Path,
) -> None:
    policy = make_policy()
    controller = DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=parse_context_blocks(CONTEXT_PRESET),
        state_path=tmp_path / "policy_state.json",
        ledger=TurnActionLedger(tmp_path),
        warmstart_assigner=MD7ProportionalAssigner(rng=FixedRNG(0.75)),
    )
    controller.start_attempt(0.4, attempt_id="agency-attempt")
    agency = decide_explicit_learner_agency(
        [{"role": "student", "content": "Could you quiz me with questions?"}]
    )
    assert agency.triggered and agency.requested_move == "probing"
    decision = controller.select_turn(
        PROBABILITIES,
        {
            "mastery_before": 0.4,
            "previous_mastery_delta": None,
            "previous_learner_signals": None,
        },
        externally_forced_move=agency.requested_move,
        external_assignment_source=agency.assignment_source,
        external_assignment_category=agency.request_category,
    )
    assert decision.final_move == "probing"
    assert decision.behavior_policy == AGENCY_BEHAVIOR_POLICY
    assert decision.treatment_assignment_source == AGENCY_ASSIGNMENT_SOURCE
    assert decision.randomized_assignment is False
    assert decision.behavior_propensity is None
    assert policy.total_updates == 0
    pending = list((tmp_path / "pending_actions").glob("*.json"))
    assert len(pending) == 1
    event = json.loads(pending[0].read_text(encoding="utf-8"))
    assert event["decision_source"] == "learner_agency"
    assert event["treatment_assignment_source"] == "learner_agency"
    assert event["randomized_assignment"] is False
    assert event["behavior_propensity"] is None


def test_policy_vector_excludes_hq_but_event_logs_full_diagnostics(
    tmp_path: Path,
) -> None:
    policy = make_policy()
    controller = DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=parse_context_blocks(CONTEXT_PRESET),
        state_path=tmp_path / "policy_state.json",
        ledger=TurnActionLedger(tmp_path),
        warmstart_assigner=MD7ProportionalAssigner(rng=FixedRNG(0.45)),
    )
    controller.start_attempt(0.4, attempt_id="diagnostic-attempt")
    decision = controller.select_turn(
        PROBABILITIES,
        {
            "mastery_before": 0.4,
            "previous_mastery_delta": None,
            "previous_learner_signals": None,
        },
    )
    assert len(decision.context) == 11
    assert decision.context_snapshot["enabled_blocks"] == ["S", "K", "L"]
    assert decision.diagnostic_context_snapshot["enabled_blocks"] == [
        "S", "K", "L", "H", "Q"
    ]
    assert decision.diagnostic_context_snapshot["context_dimension"] == 22
    policy_features = set(decision.context_snapshot["feature_names"])
    assert "previous_move_generic" not in policy_features
    assert "previous_mistake_identification" not in policy_features


def test_fresh_skl_state_has_zero_updates_and_11_by_11_matrices() -> None:
    policy = make_policy()
    state = policy.state_dict()
    assert state["context_dimension"] == 11
    assert state["update_count"] == 0
    assert state["arm_update_counts"] == {arm: 0 for arm in ARMS}
    for arm in ARMS:
        matrix = state["A"][arm]
        assert len(matrix) == 11
        assert all(len(row) == 11 for row in matrix)
        assert state["b"][arm] == [0.0] * 11


def test_formal_assessment_contract_is_external_to_turn_reward() -> None:
    assert FORMAL_ASSESSMENT_ROLE == "separate_external_policy_health_signal"
    assert set(turn_rewards(0.4, 0.5)) == {"raw_delta", "headroom_normalized"}
