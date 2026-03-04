"""Focused implementation-contract tests for direct-arm Turn-LinTS v1."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest


MOVE_ROOT = Path(__file__).resolve().parents[3] / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.turn_lints_context import (  # noqa: E402
    BLOCK_ORDER,
    CONTEXT_PRESETS,
    build_turn_lints_context,
    context_schema_id,
    feature_names_for_blocks,
    parse_context_blocks,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    ANCHOR_PROBABILITY_FLOOR,
    DIRECT_ARMS,
    DirectTurnLinTS,
    save_turn_lints_state,
)
from src.self_improvement.turn_lints_reward import turn_rewards  # noqa: E402
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
    DirectTurnLinTSPipeline,
    TurnActionLedger,
    TurnLinTSMode,
)
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)


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
SELECTOR_SHA = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"


def make_policy(
    blocks: str = "S",
    reward_mode: str = "raw_delta",
    *,
    anchor_mode: str = "none",
    anchor_gamma: float = 0.0,
) -> DirectTurnLinTS:
    parsed = parse_context_blocks(blocks)
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(parsed),
        enabled_context_blocks=parsed,
        feature_names=feature_names_for_blocks(parsed),
        selector_version="MD7-R2-TELL-C1 epoch 2",
        selector_sha256=SELECTOR_SHA,
        reward_mode=reward_mode,
        anchor_mode=anchor_mode,
        anchor_gamma=anchor_gamma,
    )


def test_presets_are_deterministic_and_full_context_is_causal() -> None:
    assert CONTEXT_PRESETS == (
        "S", "S+K", "S+L", "S+K+L", "S+K+L+H", "S+K+L+H+Q"
    )
    context = build_turn_lints_context(
        enabled_blocks="S+K+L+H+Q",
        selector_probabilities=PROBABILITIES,
        mastery_before=0.4,
        previous_mastery_delta=None,
        previous_learner_signals=None,
        previous_move=None,
        prior_tutor_turn_count=0,
        previous_mrb1_scores=None,
    )
    assert context.enabled_blocks == BLOCK_ORDER
    assert context.dimension == 22
    assert context.feature_names == feature_names_for_blocks(BLOCK_ORDER)
    joined = " ".join(context.feature_names).lower()
    assert "mastery_after" not in joined
    assert "current" not in joined
    assert "mean" not in joined
    snapshot = context.as_mapping()
    assert snapshot["context_dimension"] == 22
    assert snapshot["feature_values"] == dict(
        zip(context.feature_names, context.values, strict=True)
    )


@pytest.mark.parametrize(
    ("before", "after", "raw", "headroom"),
    [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0),
     (0.25, 1.0, 0.75, 1.0), (0.75, 0.0, -0.75, -1.0)],
)
def test_reward_boundaries(before: float, after: float,
                           raw: float, headroom: float) -> None:
    result = turn_rewards(before, after)
    assert result == {"raw_delta": raw, "headroom_normalized": headroom}


def test_direct_arms_and_immediate_idempotent_update(tmp_path: Path) -> None:
    assert DIRECT_ARMS == ("generic", "probing", "focus", "telling")
    assert len(set(DIRECT_ARMS)) == 4
    policy = make_policy()
    controller = DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.LIVE,
        enabled_context_blocks=("S",),
        state_path=tmp_path / "policy_state.json",
        ledger=TurnActionLedger(tmp_path),
    )
    controller.start_attempt(0.4, attempt_id="attempt-1")
    decision = controller.select_turn(
        PROBABILITIES,
        {"mastery_before": 0.4, "previous_mastery_delta": None,
         "previous_learner_signals": None},
    )
    assert decision.selected_arm in DIRECT_ARMS
    controller.record_mrb1_scores(MRB1)
    first = controller.complete_outcome(
        action_event_id=decision.action_event_id,
        should_update=True,
        mastery_before=0.4,
        mastery_after=0.5,
        resolver_event_id="resolver-1",
        previous_learner_signals={
            "reasoning_probability": 0.6,
            "uncertainty_probability": 0.2,
            "clarification_probability": 0.1,
        },
    )
    second = controller.complete_outcome(
        action_event_id=decision.action_event_id,
        should_update=True,
        mastery_before=0.4,
        mastery_after=0.5,
        resolver_event_id="resolver-1",
        previous_learner_signals={
            "reasoning_probability": 0.6,
            "uncertainty_probability": 0.2,
            "clarification_probability": 0.1,
        },
    )
    assert first == second
    assert policy.total_updates == 1


def test_state_rejects_reward_or_context_lineage_mismatch() -> None:
    raw = make_policy("S", "raw_delta")
    with pytest.raises(ValueError, match="reward_mode"):
        make_policy("S", "headroom_normalized").load_state_dict(raw.state_dict())
    with pytest.raises(ValueError, match="context_schema_id"):
        make_policy("S+K", "raw_delta").load_state_dict(raw.state_dict())


def test_selector_version_labels_ignore_case_but_sha_remains_strict() -> None:
    state = make_policy().state_dict()
    policy = make_policy()
    policy.selector_version = "MD7-R2-TELL-C1 Epoch 2"
    policy.anchor_selector_version = "MD7-R2-TELL-C1 Epoch 2"
    policy.load_state_dict(state)

    different_version = make_policy()
    different_version.selector_version = "MD7-R2-TELL-C1 Epoch 3"
    with pytest.raises(ValueError, match="selector_version"):
        different_version.load_state_dict(state)

    different_sha = make_policy()
    different_sha.selector_sha256 = "0" * 64
    with pytest.raises(ValueError, match="selector_sha256"):
        different_sha.load_state_dict(state)


def test_md7_logprob_anchor_score_is_separate_and_numerically_safe() -> None:
    policy = make_policy(
        "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.5
    )
    context = build_turn_lints_context(
        enabled_blocks="S",
        selector_probabilities=PROBABILITIES,
        mastery_before=0.4,
        previous_mastery_delta=None,
        previous_learner_signals=None,
        previous_move=None,
        prior_tutor_turn_count=0,
        previous_mrb1_scores=None,
    ).vector()
    probabilities_with_zero = {
        "generic": 0.2,
        "probing": 0.3,
        "focus": 0.5,
        "telling": 0.0,
    }
    result = policy.select_arm(
        context, selector_probabilities=probabilities_with_zero
    )
    assert set(result["policy_scores"]) == set(DIRECT_ARMS)
    assert set(result["anchor_scores"]) == set(DIRECT_ARMS)
    assert result["anchor_scores"]["telling"] == pytest.approx(
        0.5 * math.log(ANCHOR_PROBABILITY_FLOOR)
    )
    for arm in DIRECT_ARMS:
        assert math.isfinite(result["anchor_scores"][arm])
        assert result["policy_scores"][arm] == pytest.approx(
            result["contextual_ts_reward_scores"][arm]
            + result["anchor_scores"][arm]
        )


def test_zero_gamma_anchor_reproduces_p0_with_all_arms_available() -> None:
    p0 = make_policy("S+K+L+H+Q")
    p1_zero = make_policy(
        "S+K+L+H+Q", anchor_mode="md7_logprob_anchor", anchor_gamma=0.0
    )
    context = build_turn_lints_context(
        enabled_blocks="S+K+L+H+Q",
        selector_probabilities=PROBABILITIES,
        mastery_before=0.4,
        previous_mastery_delta=None,
        previous_learner_signals=None,
        previous_move=None,
        prior_tutor_turn_count=0,
        previous_mrb1_scores=None,
    ).vector()
    p0_result = p0.select_arm(context)
    p1_result = p1_zero.select_arm(
        context, selector_probabilities=PROBABILITIES
    )
    assert p1_result["selected_arm"] == p0_result["selected_arm"]
    assert p1_result["contextual_ts_reward_scores"] == p0_result[
        "contextual_ts_reward_scores"
    ]
    assert p1_result["anchor_scores"] == {arm: 0.0 for arm in DIRECT_ARMS}
    assert set(p1_result["policy_scores"]) == set(DIRECT_ARMS)


def test_anchor_metadata_mismatch_fails_closed() -> None:
    anchored = make_policy(
        "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
    )
    state = anchored.state_dict()
    with pytest.raises(ValueError, match="anchor_mode"):
        make_policy("S").load_state_dict(state)
    with pytest.raises(ValueError, match="anchor_gamma"):
        make_policy(
            "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.64
        ).load_state_dict(state)
    changed_version = make_policy(
        "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
    )
    changed_version.anchor_selector_version = "different frozen selector"
    with pytest.raises(ValueError, match="anchor_selector_version"):
        changed_version.load_state_dict(state)
    changed_sha = make_policy(
        "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
    )
    changed_sha.anchor_selector_sha256 = "0" * 64
    with pytest.raises(ValueError, match="anchor_selector_sha256"):
        changed_sha.load_state_dict(state)
    changed_floor = json.loads(json.dumps(state))
    changed_floor["anchor_probability_floor"] = 1e-9
    with pytest.raises(ValueError, match="anchor_probability_floor"):
        make_policy(
            "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
        ).load_state_dict(changed_floor)


def test_v1_state_is_compatible_only_with_unanchored_policy() -> None:
    state = make_policy("S").state_dict()
    state["schema_version"] = "adaptmath_turn_lints_state_v1"
    for field in (
        "anchor_mode",
        "anchor_gamma",
        "anchor_selector_version",
        "anchor_selector_sha256",
        "anchor_probability_floor",
    ):
        state.pop(field)
    make_policy("S").load_state_dict(state)
    with pytest.raises(ValueError, match="anchor_mode"):
        make_policy(
            "S", anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
        ).load_state_dict(state)


def test_anchor_does_not_require_context_s() -> None:
    blocks = "K+L+H+Q"
    policy = make_policy(
        blocks, anchor_mode="md7_logprob_anchor", anchor_gamma=0.66
    )
    context = build_turn_lints_context(
        enabled_blocks=blocks,
        selector_probabilities=PROBABILITIES,
        mastery_before=0.4,
        previous_mastery_delta=None,
        previous_learner_signals=None,
        previous_move=None,
        prior_tutor_turn_count=0,
        previous_mrb1_scores=None,
    )
    assert "S" not in context.enabled_blocks
    assert not any(name.startswith("selector_p_") for name in context.feature_names)
    result = policy.select_arm(
        context.vector(), selector_probabilities=PROBABILITIES
    )
    assert result["selected_arm"] in DIRECT_ARMS
    assert set(result["policy_scores"]) == set(DIRECT_ARMS)


def test_off_and_shadow_never_update_or_override(tmp_path: Path) -> None:
    for mode in (TurnLinTSMode.OFF, TurnLinTSMode.SHADOW):
        root = tmp_path / mode.value
        policy = make_policy()
        controller = DirectTurnController(
            policy=policy, mode=mode, enabled_context_blocks=("S",),
            state_path=root / "policy_state.json",
            ledger=TurnActionLedger(root),
        )
        controller.start_attempt(0.4, attempt_id=f"attempt-{mode.value}")
        decision = controller.select_turn(
            PROBABILITIES,
            {"mastery_before": 0.4, "previous_mastery_delta": None,
             "previous_learner_signals": None},
        )
        assert decision.final_move == "telling"
        assert decision.selected_arm is None
        assert (decision.hypothetical_arm is None) is (mode is TurnLinTSMode.OFF)
        controller.record_mrb1_scores(MRB1)
        controller.complete_outcome(
            action_event_id=decision.action_event_id,
            should_update=True,
            mastery_before=0.4,
            mastery_after=0.5,
            resolver_event_id="resolver",
            previous_learner_signals=None,
        )
        assert policy.total_updates == 0


def test_fresh_selection_is_deterministic_for_fixed_seed() -> None:
    first = make_policy("S+K+L+H+Q")
    second = make_policy("S+K+L+H+Q")
    context = build_turn_lints_context(
        enabled_blocks="S+K+L+H+Q",
        selector_probabilities=PROBABILITIES,
        mastery_before=0.4,
        previous_mastery_delta=None,
        previous_learner_signals=None,
        previous_move=None,
        prior_tutor_turn_count=0,
        previous_mrb1_scores=None,
    ).vector()
    assert first.select_arm(context) == second.select_arm(context)


def test_full_headroom_shadow_event_is_causal_complete_and_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "turn_lints_shadow_full_headroom_v1"
    state_path = root / "policy_state.json"
    policy = make_policy(
        "S+K+L+H+Q",
        "headroom_normalized",
        anchor_mode="md7_logprob_anchor",
        anchor_gamma=0.66,
    )
    save_turn_lints_state(policy, state_path)
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    initial_A = {arm: policy.A[arm].copy() for arm in DIRECT_ARMS}
    initial_b = {arm: policy.b[arm].copy() for arm in DIRECT_ARMS}
    controller = DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.SHADOW,
        enabled_context_blocks=BLOCK_ORDER,
        state_path=state_path,
        ledger=TurnActionLedger(root),
    )
    controller.start_attempt(0.4, attempt_id="attempt-shadow")
    first = controller.select_turn(
        PROBABILITIES,
        {
            "mastery_before": 0.4,
            "previous_mastery_delta": None,
            "previous_learner_signals": None,
        },
    )
    assert first.final_move == "telling"
    assert first.selected_arm is None
    assert first.hypothetical_arm in DIRECT_ARMS
    controller.record_mrb1_scores(MRB1, "A natural Tutor response.")
    outcome = controller.complete_outcome(
        action_event_id=first.action_event_id,
        should_update=True,
        mastery_before=0.4,
        mastery_after=0.5,
        resolver_event_id="resolver-shadow-1",
        previous_learner_signals={
            "reasoning_probability": 0.6,
            "uncertainty_probability": 0.2,
            "clarification_probability": 0.1,
        },
    )
    assert outcome["raw_delta"] == pytest.approx(0.1)
    assert outcome["headroom_normalized_delta"] == pytest.approx(1.0 / 6.0)
    assert outcome["configured_reward_mode"] == "headroom_normalized"
    assert outcome["configured_reward_value"] == pytest.approx(1.0 / 6.0)
    assert outcome["reward_attributed_to_move"] == "telling"
    assert outcome["reward_attributed_to_shadow_hypothetical"] is False
    assert policy.total_updates == 0
    assert policy.arm_update_counts == {arm: 0 for arm in DIRECT_ARMS}
    for arm in DIRECT_ARMS:
        assert (policy.A[arm] == initial_A[arm]).all()
        assert (policy.b[arm] == initial_b[arm]).all()
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == state_hash

    event = json.loads((root / "turn_events.jsonl").read_text(encoding="utf-8"))
    assert event["schema_version"] == "adaptmath_turn_lints_event_v4"
    assert event["collection_lineage"] == root.name
    assert event["attempt_id"] == "attempt-shadow"
    assert event["action_event_id"] == first.action_event_id
    assert event["action_turn_index"] == 1
    assert event["actual_selector_move"] == "telling"
    assert event["actual_treatment_move"] == "telling"
    assert event["shadow_hypothetical_move"] == first.hypothetical_arm
    assert event["shadow_p0_move"] == first.shadow_p0_move
    assert event["shadow_p1_move"] == first.shadow_p1_move
    assert event["shadow_moves_are_observed_treatments"] is False
    assert event["anchor_mode"] == "md7_logprob_anchor"
    assert event["anchor_gamma"] == 0.66
    assert event["hypothetical_is_observed_treatment"] is False
    assert event["reward_attribution"] == "actual_treatment_move"
    assert event["tutor_response"] == "A natural Tutor response."
    assert event["current_response_mrb1"] == MRB1
    assert event["context"]["enabled_blocks"] == list(BLOCK_ORDER)
    assert event["context"]["context_dimension"] == 22
    assert len(event["context"]["feature_names"]) == 22
    assert len(event["context"]["vector"]) == 22
    assert set(event["context"]["feature_values"]) == set(
        event["context"]["feature_names"]
    )

    second = controller.select_turn(
        PROBABILITIES,
        {
            "mastery_before": 0.5,
            "previous_mastery_delta": 0.1,
            "previous_learner_signals": {
                "reasoning_probability": 0.6,
                "uncertainty_probability": 0.2,
                "clarification_probability": 0.1,
            },
        },
    )
    values = second.context_snapshot["feature_values"]
    assert values["previous_mastery_delta"] == pytest.approx(0.1)
    assert values["previous_reasoning_probability"] == pytest.approx(0.6)
    assert values["previous_uncertainty_probability"] == pytest.approx(0.2)
    assert values["previous_clarification_probability"] == pytest.approx(0.1)
    assert values["previous_move_telling"] == 1.0
    assert values["prior_tutor_turn_count"] == 1.0
    assert values["previous_mistake_identification"] == MRB1["Mistake_Identification"]
    assert values["previous_actionability"] == MRB1["Actionability"]
    assert all(not name.startswith("current_") for name in values)
    controller.abort_attempt()


def test_cross_repository_bridge_accepts_direct_pipeline_lifecycle() -> None:
    class StudentPipeline:
        def start_attempt(self, **_kwargs: object) -> object:
            return object()

    direct = object.__new__(DirectTurnLinTSPipeline)
    direct._active_attempt_id = None  # type: ignore[attr-defined]
    bridge = StudentModelFrozenV3Bridge(
        student_model_pipeline=StudentPipeline(),
        adaptive_pipeline=direct,
    )
    assert bridge.adaptive_pipeline is direct
