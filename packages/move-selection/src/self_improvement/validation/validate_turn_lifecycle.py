"""CPU-only validation of turn-level selection and delayed attempt credit."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from math import inf, isclose, nan

import numpy as np

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.lints_policy import (
    ARMS,
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    TrueDisjointLinTS,
)
from src.self_improvement.turn_context_builder import MRB1_TASKS
from src.self_improvement.turn_level_controller import (
    AttemptCompletion,
    TurnLevelAttemptController,
)


CONTEXT_DIM = 9


def make_policy(seed: int = 42) -> TrueDisjointLinTS:
    return TrueDisjointLinTS(
        context_dim=CONTEXT_DIM,
        seed=seed,
        data_mode="synthetic",
    )


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    try:
        action()
    except (TypeError, ValueError, RuntimeError):
        return
    raise AssertionError(f"Invalid turn-lifecycle case was accepted: {case_name}")


def md6_a() -> dict[str, float]:
    return {
        "generic": 0.10,
        "probing": 0.34,
        "focus": 0.40,
        "telling": 0.16,
    }


def md6_b() -> dict[str, float]:
    return {
        "generic": 0.36,
        "probing": 0.10,
        "focus": 0.34,
        "telling": 0.20,
    }


def md6_c() -> dict[str, float]:
    return {
        "generic": 0.10,
        "probing": 0.18,
        "focus": 0.34,
        "telling": 0.38,
    }


def scores_one() -> dict[str, float]:
    return {
        "Mistake_Identification": 0.90,
        "Mistake_Location": 0.80,
        "Providing_Guidance": 0.70,
        "Actionability": 0.60,
    }


def scores_two() -> dict[str, float]:
    return {
        "Mistake_Identification": 0.70,
        "Mistake_Location": 0.80,
        "Providing_Guidance": 0.90,
        "Actionability": 0.80,
    }


def scores_three() -> dict[str, float]:
    return {
        "Mistake_Identification": 0.80,
        "Mistake_Location": 0.60,
        "Providing_Guidance": 0.80,
        "Actionability": 0.70,
    }


def posterior_snapshot(
    policy: TrueDisjointLinTS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], int]:
    return (
        {arm: policy.A[arm].copy() for arm in ARMS},
        {arm: policy.b[arm].copy() for arm in ARMS},
        dict(policy.arm_update_counts),
        policy.total_updates,
    )


def assert_posterior_equal(
    policy: TrueDisjointLinTS,
    expected: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, int],
        int,
    ],
) -> None:
    expected_A, expected_b, expected_counts, expected_total = expected
    for arm in ARMS:
        np.testing.assert_array_equal(policy.A[arm], expected_A[arm])
        np.testing.assert_array_equal(policy.b[arm], expected_b[arm])
    assert policy.arm_update_counts == expected_counts
    assert policy.total_updates == expected_total


def test_eligible_arm_mechanics() -> None:
    assert eligible_arms(md6_a(), gap_threshold=0.10) == (
        "baseline",
        "probing_bias",
    )

    far = {
        "generic": 0.08,
        "probing": 0.15,
        "focus": 0.67,
        "telling": 0.10,
    }
    assert eligible_arms(far) == ("baseline",)

    boundary = {
        "generic": 0.24,
        "probing": 0.21,
        "focus": 0.31,
        "telling": 0.24,
    }
    assert boundary["focus"] - boundary["probing"] == 0.10
    boundary_eligible = eligible_arms(boundary, gap_threshold=0.10)
    assert boundary_eligible == (
        "baseline",
        "generic_bias",
        "probing_bias",
        "telling_bias",
    )
    assert "focus_bias" not in boundary_eligible
    assert "probing_bias" in boundary_eligible
    assert boundary_eligible[0] == "baseline"


def test_eligible_lints_selection() -> None:
    policy = make_policy(seed=17)
    context = np.linspace(0.05, 0.50, CONTEXT_DIM, dtype=np.float64)
    before = posterior_snapshot(policy)
    allowed = ("probing_bias", "baseline")
    decision = policy.select_arm(context, eligible_arms=allowed)
    canonical_allowed = ("baseline", "probing_bias")
    assert decision["selected_arm"] in canonical_allowed
    assert decision["selected_arm"] == max(
        canonical_allowed,
        key=lambda arm: decision["sampled_scores"][arm],
    )
    assert tuple(decision["sampled_scores"]) == ARMS
    assert_posterior_equal(policy, before)

    only_baseline = policy.select_arm(context, eligible_arms=("baseline",))
    assert only_baseline["selected_arm"] == "baseline"
    assert_posterior_equal(policy, before)

    rng_before_invalid = json.dumps(
        copy.deepcopy(policy.rng.bit_generator.state),
        sort_keys=True,
    )
    for name, invalid in (
        ("empty", ()),
        ("duplicate", ("baseline", "baseline")),
        ("unknown", ("baseline", "unknown")),
        ("string", "baseline"),
    ):
        expect_rejection(
            lambda invalid=invalid: policy.select_arm(
                context,
                eligible_arms=invalid,  # type: ignore[arg-type]
            ),
            name,
        )
        assert json.dumps(policy.rng.bit_generator.state, sort_keys=True) == (
            rng_before_invalid
        )
        assert_posterior_equal(policy, before)


def test_weighted_policy_update() -> None:
    policy = make_policy()
    context = np.linspace(0.05, 0.50, CONTEXT_DIM, dtype=np.float64)
    before = posterior_snapshot(policy)
    reward = 0.75
    weight = 0.25
    policy.update(
        "focus_bias",
        context,
        reward,
        sample_weight=weight,
    )
    expected_A = before[0]["focus_bias"] + weight * np.outer(context, context)
    expected_b = before[1]["focus_bias"] + weight * reward * context
    np.testing.assert_array_equal(policy.A["focus_bias"], expected_A)
    np.testing.assert_array_equal(policy.b["focus_bias"], expected_b)
    assert policy.arm_update_counts["focus_bias"] == 1
    assert policy.total_updates == 1
    for arm in ARMS:
        if arm != "focus_bias":
            np.testing.assert_array_equal(policy.A[arm], before[0][arm])
            np.testing.assert_array_equal(policy.b[arm], before[1][arm])
            assert policy.arm_update_counts[arm] == 0

    default_policy = make_policy()
    explicit_policy = make_policy()
    default_policy.update("baseline", context, reward)
    explicit_policy.update("baseline", context, reward, sample_weight=1.0)
    np.testing.assert_array_equal(
        default_policy.A["baseline"], explicit_policy.A["baseline"]
    )
    np.testing.assert_array_equal(
        default_policy.b["baseline"], explicit_policy.b["baseline"]
    )

    for invalid_weight in (0.0, -0.1, 1.1, nan, inf, -inf, True, False, "0.5"):
        candidate = make_policy()
        candidate_before = posterior_snapshot(candidate)
        expect_rejection(
            lambda invalid_weight=invalid_weight: candidate.update(
                "baseline",
                context,
                reward,
                sample_weight=invalid_weight,
            ),
            f"sample_weight={invalid_weight!r}",
        )
        assert_posterior_equal(candidate, candidate_before)


def test_turn_lifecycle_and_delayed_mathematics() -> None:
    policy = make_policy(seed=2026)
    controller = TurnLevelAttemptController(policy, gap_threshold=0.10)
    before_attempt = posterior_snapshot(policy)

    controller.start_attempt(mastery_before=0.42)
    assert controller.is_active is True
    assert controller.awaiting_mrb1 is False
    assert controller.running_quality is not None
    assert controller.running_quality.count == 0
    assert_posterior_equal(policy, before_attempt)

    rng_before_turn_one = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    turn_one = controller.select_turn(md6_a())
    rng_after_turn_one = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    assert rng_after_turn_one != rng_before_turn_one
    assert turn_one.turn_index == 1
    assert turn_one.context[:4] == (0.10, 0.34, 0.40, 0.16)
    assert turn_one.context[4:] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert turn_one.eligible_arms == ("baseline", "probing_bias")
    assert turn_one.selected_arm in turn_one.eligible_arms
    assert controller.awaiting_mrb1 is True
    assert_posterior_equal(policy, before_attempt)

    controller.record_mrb1_scores(scores_one())
    assert controller.awaiting_mrb1 is False
    assert controller.running_quality is not None
    assert controller.running_quality.count == 1
    assert_posterior_equal(policy, before_attempt)

    rng_before_turn_two = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    turn_two = controller.select_turn(md6_b())
    rng_after_turn_two = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    assert rng_after_turn_two != rng_before_turn_two
    assert turn_two.turn_index == 2
    assert turn_two.context[:4] == (0.36, 0.10, 0.34, 0.20)
    np.testing.assert_allclose(
        turn_two.context[4:8],
        (0.90, 0.80, 0.70, 0.60),
        rtol=0.0,
        atol=1e-15,
    )
    assert turn_two.context[8] == 1.0
    assert turn_two.selected_arm in turn_two.eligible_arms
    assert dict(turn_two.sampled_scores) != dict(turn_one.sampled_scores)
    assert_posterior_equal(policy, before_attempt)

    controller.record_mrb1_scores(scores_two())
    assert_posterior_equal(policy, before_attempt)

    rng_before_turn_three = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    turn_three = controller.select_turn(md6_c())
    rng_after_turn_three = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    assert rng_after_turn_three != rng_before_turn_three
    assert turn_three.turn_index == 3
    assert turn_three.context[:4] == (0.10, 0.18, 0.34, 0.38)
    np.testing.assert_allclose(
        turn_three.context[4:8],
        (0.80, 0.80, 0.80, 0.70),
        rtol=0.0,
        atol=1e-15,
    )
    assert turn_three.context[8] == 1.0
    assert turn_three.selected_arm in turn_three.eligible_arms
    assert_posterior_equal(policy, before_attempt)

    controller.record_mrb1_scores(scores_three())
    assert controller.awaiting_mrb1 is False
    assert_posterior_equal(policy, before_attempt)
    saved_turns = controller.completed_turns
    assert len(saved_turns) == 3

    reward = 0.0
    weight = 1.0 / 3.0
    expected_A = {arm: before_attempt[0][arm].copy() for arm in ARMS}
    expected_b = {arm: before_attempt[1][arm].copy() for arm in ARMS}
    selected_counts: Counter[str] = Counter()
    for completed_turn in saved_turns:
        arm = completed_turn.decision.selected_arm
        context = np.asarray(completed_turn.decision.context, dtype=np.float64)
        expected_A[arm] = expected_A[arm] + weight * np.outer(context, context)
        expected_b[arm] = expected_b[arm] + weight * reward * context
        selected_counts[arm] += 1

    completion = controller.finish_attempt(
        evaluator_score=2,
        mastery_after=0.55,
    )
    assert isinstance(completion, AttemptCompletion)
    assert completion.evaluator_score == 2
    assert completion.reward == reward
    assert completion.mastery_before == 0.42
    assert completion.mastery_after == 0.55
    assert isclose(completion.mastery_delta, 0.13, rel_tol=0.0, abs_tol=1e-15)
    assert completion.turn_count == 3
    assert completion.sample_weight_per_turn == weight
    assert isclose(
        completion.total_attempt_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert completion.turns == saved_turns
    assert completion.final_quality.count == 3
    np.testing.assert_allclose(
        completion.final_quality.feature_values,
        (0.80, 2.20 / 3.0, 0.80, 0.70),
        rtol=0.0,
        atol=1e-15,
    )

    for arm in ARMS:
        np.testing.assert_array_equal(policy.A[arm], expected_A[arm])
        np.testing.assert_array_equal(policy.b[arm], expected_b[arm])
        assert policy.arm_update_counts[arm] == selected_counts[arm]
    assert policy.total_updates == 3
    assert controller.completed_attempts == 1
    assert controller.number_of_weighted_turn_updates == 3
    assert completion.completed_attempts == 1
    assert completion.number_of_weighted_turn_updates == 3
    assert controller.is_active is False
    assert controller.awaiting_mrb1 is False
    assert controller.running_quality is None

    try:
        completion.reward = 0.0  # type: ignore[misc]
    except (FrozenInstanceError, AttributeError):
        pass
    else:
        raise AssertionError("AttemptCompletion is not immutable.")


def test_invalid_lifecycle() -> None:
    policy = make_policy(seed=8)
    controller = TurnLevelAttemptController(policy)
    expect_rejection(lambda: controller.select_turn(md6_a()), "select before start")
    expect_rejection(
        lambda: controller.record_mrb1_scores(scores_one()),
        "MRB1 before start",
    )
    expect_rejection(
        lambda: controller.finish_attempt(2, 0.55),
        "finish before start",
    )
    expect_rejection(controller.abort_attempt, "abort before start")

    controller.start_attempt(0.42)
    before_attempt = posterior_snapshot(policy)
    expect_rejection(
        lambda: controller.start_attempt(0.50),
        "second attempt while active",
    )
    expect_rejection(
        lambda: controller.record_mrb1_scores(scores_one()),
        "MRB1 before turn selection",
    )
    expect_rejection(
        lambda: controller.finish_attempt(2, 0.55),
        "finish with zero turns",
    )

    controller.select_turn(md6_a())
    expect_rejection(
        lambda: controller.select_turn(md6_b()),
        "second turn while MRB1 pending",
    )
    expect_rejection(
        lambda: controller.finish_attempt(2, 0.55),
        "finish while MRB1 pending",
    )
    for name, invalid_scores in (
        ("missing MRB1 task", {key: scores_one()[key] for key in MRB1_TASKS[:-1]}),
        ("boolean MRB1", {**scores_one(), "Actionability": True}),
    ):
        expect_rejection(
            lambda invalid_scores=invalid_scores: controller.record_mrb1_scores(
                invalid_scores
            ),
            name,
        )
        assert controller.awaiting_mrb1 is True
        assert_posterior_equal(policy, before_attempt)

    controller.record_mrb1_scores(scores_one())
    assert_posterior_equal(policy, before_attempt)
    for invalid_mastery in (-0.1, 1.1, nan, inf, True, "0.55"):
        expect_rejection(
            lambda invalid_mastery=invalid_mastery: controller.finish_attempt(
                2,
                invalid_mastery,  # type: ignore[arg-type]
            ),
            f"mastery_after={invalid_mastery!r}",
        )
        assert controller.is_active is True
        assert_posterior_equal(policy, before_attempt)

    expect_rejection(
        lambda: controller.finish_attempt(4, 0.55),
        "invalid evaluator score",
    )
    assert_posterior_equal(policy, before_attempt)

    controller.finish_attempt(2, 0.55)
    expect_rejection(
        lambda: controller.finish_attempt(2, 0.55),
        "double finish",
    )


def test_abort() -> None:
    policy = make_policy(seed=9)
    controller = TurnLevelAttemptController(policy)
    before = posterior_snapshot(policy)
    controller.start_attempt(0.42)
    controller.select_turn(md6_a())
    controller.record_mrb1_scores(scores_one())
    controller.select_turn(md6_b())
    assert controller.awaiting_mrb1 is True

    controller.abort_attempt()
    assert_posterior_equal(policy, before)
    assert controller.is_active is False
    assert controller.awaiting_mrb1 is False
    assert controller.completed_turns == ()
    assert controller.running_quality is None
    assert controller.completed_attempts == 0
    assert controller.number_of_weighted_turn_updates == 0

    controller.start_attempt(0.50)
    assert controller.is_active is True
    controller.abort_attempt()


def test_controller_configuration() -> None:
    defaults = make_policy()
    assert defaults.ridge_lambda == DEFAULT_RIDGE_LAMBDA == 1.0
    assert defaults.exploration_scale == DEFAULT_EXPLORATION_SCALE == 0.20
    expect_rejection(
        lambda: TurnLevelAttemptController(TrueDisjointLinTS(context_dim=12)),
        "12-D policy",
    )
    for threshold in (-0.1, 1.1, nan, inf, True, False, "0.10"):
        expect_rejection(
            lambda threshold=threshold: TurnLevelAttemptController(
                make_policy(),
                gap_threshold=threshold,  # type: ignore[arg-type]
            ),
            f"gap_threshold={threshold!r}",
        )
    for mastery in (-0.1, 1.1, nan, inf, True, "0.42"):
        controller = TurnLevelAttemptController(make_policy())
        expect_rejection(
            lambda mastery=mastery: controller.start_attempt(  # type: ignore[arg-type]
                mastery
            ),
            f"mastery_before={mastery!r}",
        )
        assert controller.is_active is False


def main() -> None:
    checks = (
        ("eligible-arm mechanics", test_eligible_arm_mechanics),
        ("eligible true-LinTS selection", test_eligible_lints_selection),
        ("weighted policy update", test_weighted_policy_update),
        (
            "turn lifecycle and delayed weighted mathematics",
            test_turn_lifecycle_and_delayed_mathematics,
        ),
        ("invalid lifecycle rejection", test_invalid_lifecycle),
        ("abort without posterior update", test_abort),
        ("controller configuration", test_controller_configuration),
    )

    print("CANONICAL TURN-LEVEL DELAYED-FEEDBACK VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
