"""End-to-end contract for the audited synthetic-initialized LIVE lineage."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
MOVE_ROOT = ROOT / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.explicit_learner_agency import (  # noqa: E402
    AGENCY_ASSIGNMENT_SOURCE,
)
from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    FORMAL_ASSESSMENT_ROLE,
    ORDERED_CONTEXT_FEATURES,
    REWARD_MODE,
)
from src.self_improvement.turn_lints_context import (  # noqa: E402
    context_schema_id,
    feature_names_for_blocks,
)
from src.self_improvement.turn_lints_live_lineage import (  # noqa: E402
    validate_live_lineage,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DIRECT_ARMS,
    DirectTurnLinTS,
    load_turn_lints_state,
    save_turn_lints_state,
)
from src.self_improvement.turn_lints_reward import turn_rewards  # noqa: E402
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
    TurnActionLedger,
    TurnLinTSMode,
)


RESULTS = MOVE_ROOT / "results" / "turn_lints_live_skl_synthetic_warmstart_v1"
LIVE_ROOT = ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "turn_lints_live_skl_final_v1"
OBSERVATIONS = RESULTS / "synthetic_warmstart_observations.jsonl"
STATE = LIVE_ROOT / "policy_state.json"
MANIFEST = LIVE_ROOT / "lineage_manifest.json"
SELECTOR_SHA = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
MRB1 = {
    "Mistake_Identification": 0.1,
    "Mistake_Location": 0.2,
    "Providing_Guidance": 0.3,
    "Actionability": 0.4,
}
PROBABILITIES = {"generic": .1, "probing": .2, "focus": .3, "telling": .4}


def make_policy() -> DirectTurnLinTS:
    blocks = ("S", "K", "L")
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(blocks),
        enabled_context_blocks=blocks,
        feature_names=feature_names_for_blocks(blocks),
        selector_version="MD7-R2-TELL-C1 Epoch 2",
        selector_sha256=SELECTOR_SHA,
        reward_mode="headroom_normalized",
        anchor_mode="none",
        anchor_gamma=0.0,
        anchor_selector_version="MD7-R2-TELL-C1 Epoch 2",
        anchor_selector_sha256=SELECTOR_SHA,
        ridge_lambda=1.0,
        exploration_scale=.20,
        seed=42,
        data_mode="real",
    )


@pytest.fixture(scope="module")
def records() -> list[dict[str, object]]:
    return [json.loads(line) for line in OBSERVATIONS.read_text(encoding="utf-8").splitlines()]


def test_exact_500_provenance_schema_and_no_leakage(records) -> None:
    assert len(records) == 500
    assert CONTEXT_PRESET == "S+K+L" and CONTEXT_DIMENSION == 11
    for index, row in enumerate(records, start=1):
        assert row["synthetic_observation_id"] == f"synthetic-init:{index:04d}"
        assert row["observation_origin"] == "synthetic_initialization"
        assert row["synthetic_generator_version"] == "adaptmath_skl_synthetic_warmstart_v1"
        assert row["real_live_observation"] is False
        assert row["posterior_initialization_observation"] is True
        assert row["ordered_context_features"] == list(ORDERED_CONTEXT_FEATURES)
        assert len(row["context_vector"]) == 11
        assert row["current_learner_response_in_context"] is False
        assert row["current_mrb1_in_context"] is False
        assert row["mastery_after_in_context"] is False
        assert row["future_information_in_context"] is False


def test_context_ranges_missing_L_and_temporal_sources(records) -> None:
    contexts = np.asarray([row["context_vector"] for row in records], dtype=float)
    assert np.isfinite(contexts).all()
    assert np.allclose(contexts[:, :4].sum(axis=1), 1.0, atol=1e-6)
    assert ((0 <= contexts[:, 4]) & (contexts[:, 4] <= 1)).all()
    assert ((0 <= contexts[:, 7:10]) & (contexts[:, 7:10] <= 1)).all()
    assert set(contexts[:, 10]) == {0.0, 1.0}
    assert (contexts[contexts[:, 10] == 1.0, 7:10] == 0.0).all()
    assert {row["L_source"] for row in records} == {
        "retrospective_preaction_recomputation", "runtime_missing_representation"
    }


def test_md7_proportional_assignments_propensities_and_all_arm_support(records) -> None:
    counts = {arm: 0 for arm in DIRECT_ARMS}
    for row in records:
        vector = row["full_MD7_probability_vector"]
        arm = row["synthetic_action"]
        assert row["action_assignment_source"] == "raw_md7_probability_sample"
        assert math.isclose(sum(vector.values()), 1.0, abs_tol=1e-9)
        assert row["synthetic_behavior_propensity"] == vector[arm]
        counts[arm] += 1
    assert counts == {"generic": 123, "probing": 175, "focus": 174, "telling": 28}


def test_rewards_are_finite_bounded_and_bkt_roundtrip(records) -> None:
    for row in records:
        reward = float(row["synthetic_reward"])
        before = float(row["synthetic_mastery_before"])
        after = float(row["synthetic_mastery_after"])
        assert math.isfinite(reward) and -1 <= reward <= 1
        assert 0 <= before <= 1 and 0 <= after <= 1
        assert turn_rewards(before, after)[REWARD_MODE] == pytest.approx(reward, abs=1e-10)
        assert abs(float(row["reward_roundtrip_error"])) <= 1e-10


def test_weighted_update_math_and_real_default_weight() -> None:
    x = np.linspace(0.05, .55, 11)
    weighted = make_policy(); weighted.update_once(update_id="w", arm="generic", context=x, reward=.4, weight=.02)
    assert np.allclose(weighted.A["generic"], np.eye(11) + .02 * np.outer(x, x))
    assert np.allclose(weighted.b["generic"], .02 * .4 * x)
    real = make_policy(); real.update_once(update_id="r", arm="generic", context=x, reward=.4)
    assert np.allclose(real.A["generic"], np.eye(11) + np.outer(x, x))
    assert np.allclose(real.b["generic"], .4 * x)
    with pytest.raises(ValueError):
        real.update_once(update_id="bad", arm="focus", context=x, reward=.1, weight=0)


def test_initialized_state_is_fresh_11d_finite_and_effective_weight_10(records) -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    assert state["context_dimension"] == 11
    assert state["ordered_feature_names"] == list(ORDERED_CONTEXT_FEATURES)
    assert state["reward_mode"] == "headroom_normalized"
    assert state["anchor_mode"] == "none" and state["anchor_gamma"] == 0.0
    assert state["update_count"] == 500
    assert state["applied_update_ids"] == [f"synthetic-init:{i:04d}" for i in range(1, 501)]
    assert sum(float(row["posterior_update_weight"]) for row in records) == pytest.approx(10.0)
    for arm in DIRECT_ARMS:
        A = np.asarray(state["A"][arm]); b = np.asarray(state["b"][arm])
        assert A.shape == (11, 11) and b.shape == (11,)
        assert np.isfinite(A).all() and np.isfinite(b).all()
        assert np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > 0


def test_heldout_audit_and_real_update_responsiveness_pass() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["activation_decision"] == "A"
    assert all(summary["predeclared_sanity_checks"].values())
    heldout = pd.read_csv(RESULTS / "heldout_context_policy_summary.csv").set_index("policy")
    assert heldout.loc["P_SYNTH", "md7_top1_agreement_rate"] > heldout.loc["P0_uninformed_prior", "md7_top1_agreement_rate"]
    assert heldout.loc["P_SYNTH", "telling_rate_when_md7_top1_not_telling"] < .4
    assert all(heldout.loc["P_SYNTH", f"action_rate_{arm}"] > .01 for arm in DIRECT_ARMS)
    sensitivity = pd.read_csv(RESULTS / "single_real_update_sensitivity.csv")
    assert (sensitivity["posterior_mean_parameter_change_l2"] > 1e-6).all()


def test_live_update_is_immediate_unit_weight_and_persists(tmp_path: Path) -> None:
    policy = make_policy(); state_path = tmp_path / "policy_state.json"
    controller = DirectTurnController(policy=policy, mode=TurnLinTSMode.LIVE,
        enabled_context_blocks=("S", "K", "L"), state_path=state_path,
        ledger=TurnActionLedger(tmp_path))
    controller.start_attempt(.4, attempt_id="real-live")
    decision = controller.select_turn(PROBABILITIES, {"mastery_before": .4,
        "previous_mastery_delta": None, "previous_learner_signals": None})
    arm = decision.selected_arm; before_A = policy.A[arm].copy()
    controller.record_mrb1_scores(MRB1, "deterministic mocked tutor response")
    outcome = controller.complete_outcome(action_event_id=decision.action_event_id,
        should_update=True, mastery_before=.4, mastery_after=.5,
        resolver_event_id="resolver", previous_learner_signals=None)
    assert outcome["posterior_update_origin"] == "real_live"
    assert outcome["posterior_update_weight"] == 1.0
    assert outcome["posterior_updated"] is True
    assert not np.allclose(before_A, policy.A[arm]) and state_path.is_file()
    clone = make_policy(); load_turn_lints_state(clone, state_path)
    assert clone.state_dict() == policy.state_dict()


def test_no_reward_and_agency_override_never_update(tmp_path: Path) -> None:
    policy = make_policy(); controller = DirectTurnController(policy=policy,
        mode=TurnLinTSMode.LIVE, enabled_context_blocks=("S", "K", "L"),
        state_path=tmp_path / "state.json", ledger=TurnActionLedger(tmp_path))
    controller.start_attempt(.4, attempt_id="agency")
    decision = controller.select_turn(PROBABILITIES, {"mastery_before": .4,
        "previous_mastery_delta": None, "previous_learner_signals": None},
        externally_forced_move="probing", external_assignment_source=AGENCY_ASSIGNMENT_SOURCE,
        external_assignment_category="request_questioning")
    assert decision.final_move == "probing" and decision.selected_arm is None
    assert decision.behavior_propensity is None and decision.randomized_assignment is False
    controller.record_mrb1_scores(MRB1)
    outcome = controller.complete_outcome(action_event_id=decision.action_event_id,
        should_update=True, mastery_before=.4, mastery_after=.5,
        resolver_event_id="resolver", previous_learner_signals=None)
    assert outcome["agency_forced_no_policy_update"] is True
    assert outcome["posterior_updated"] is False and policy.total_updates == 0


def test_live_manifest_fail_closed_and_architecture_roles(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")); manifest["live_activated_at"] = "2026-08-29T00:00:00Z"
    state_path = tmp_path / "policy_state.json"; manifest_path = tmp_path / "lineage_manifest.json"
    state_path.write_bytes(STATE.read_bytes()); manifest["initial_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = make_policy(); load_turn_lints_state(policy, state_path)
    counters = validate_live_lineage(policy, manifest_path, state_path=state_path)
    assert counters["synthetic_initialization_updates"] == 500
    assert counters["synthetic_effective_sample_size"] == 10.0
    assert counters["real_live_updates"] == 0
    broken = copy.deepcopy(manifest); broken["context_dimension"] = 22
    manifest_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="context_dimension"):
        validate_live_lineage(policy, manifest_path, state_path=state_path)
    assert FORMAL_ASSESSMENT_ROLE == "separate_external_policy_health_signal"
