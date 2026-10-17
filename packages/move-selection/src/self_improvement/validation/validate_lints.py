"""CPU-only deterministic validation of canonical true LinTS mechanics."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable

import numpy as np

from src.self_improvement.lints_policy import (
    ARMS,
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    TrueDisjointLinTS,
)


CONTEXT_DIM = 12
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20


def context_vector() -> np.ndarray:
    """Return a deterministic nonzero 12-D synthetic context."""

    return np.linspace(0.05, 0.60, CONTEXT_DIM, dtype=np.float64)


def make_policy(
    seed: int = 42,
    data_mode: str = "synthetic",
) -> TrueDisjointLinTS:
    return TrueDisjointLinTS(
        context_dim=CONTEXT_DIM,
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=seed,
        data_mode=data_mode,
    )


def expect_rejection(
    action: Callable[[], object],
    case_name: str,
) -> None:
    """Assert that invalid input raises without accepting the case."""

    try:
        action()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"Invalid LinTS case was accepted: {case_name}")


def state_json(policy: TrueDisjointLinTS) -> str:
    return json.dumps(policy.state_dict(), sort_keys=True)


def test_arm_contract() -> None:
    expected = (
        "baseline",
        "generic_bias",
        "probing_bias",
        "focus_bias",
        "telling_bias",
    )
    assert isinstance(ARMS, tuple)
    assert ARMS == expected
    assert make_policy().arms == expected


def test_canonical_defaults() -> None:
    policy = TrueDisjointLinTS(context_dim=9)
    assert DEFAULT_RIDGE_LAMBDA == policy.ridge_lambda == 1.0
    assert DEFAULT_EXPLORATION_SCALE == policy.exploration_scale == 0.20


def test_initial_prior() -> None:
    policy = make_policy()
    identity = np.eye(CONTEXT_DIM, dtype=np.float64)
    zero = np.zeros(CONTEXT_DIM, dtype=np.float64)
    expected_covariance = EXPLORATION_SCALE**2 * identity

    for arm in ARMS:
        np.testing.assert_array_equal(policy.A[arm], identity)
        np.testing.assert_array_equal(policy.b[arm], zero)
        mean, covariance = policy.posterior_for_arm(arm)
        np.testing.assert_array_equal(mean, zero)
        np.testing.assert_allclose(
            covariance,
            expected_covariance,
            rtol=0.0,
            atol=1e-15,
        )


def test_true_posterior_sampling() -> None:
    policy = make_policy(seed=17)
    x = context_vector()
    mean, _ = policy.posterior_for_arm("baseline")
    np.testing.assert_array_equal(mean, np.zeros(CONTEXT_DIM))

    first = policy.select_arm(x)
    second = policy.select_arm(x)
    baseline_theta = np.asarray(first["sampled_thetas"]["baseline"])
    assert not np.allclose(baseline_theta, mean, rtol=0.0, atol=1e-14)

    first_scores = np.asarray([first["sampled_scores"][arm] for arm in ARMS])
    second_scores = np.asarray([second["sampled_scores"][arm] for arm in ARMS])
    assert not np.allclose(first_scores, second_scores, rtol=0.0, atol=1e-14)

    source = inspect.getsource(TrueDisjointLinTS.sample_arm_scores).lower()
    assert "multivariate_normal" in source
    assert "uncertainty" not in source
    assert "exploration_bonus" not in source
    assert "ucb_score" not in source


def test_seed_reproducibility() -> None:
    policy_a = make_policy(seed=2026)
    policy_b = make_policy(seed=2026)
    decision_a = policy_a.select_arm(context_vector())
    decision_b = policy_b.select_arm(context_vector())
    assert decision_a["selected_arm"] == decision_b["selected_arm"]
    for arm in ARMS:
        assert decision_a["sampled_scores"][arm] == decision_b["sampled_scores"][arm]
        np.testing.assert_array_equal(
            decision_a["sampled_thetas"][arm],
            decision_b["sampled_thetas"][arm],
        )


def test_rng_advancement() -> None:
    policy = make_policy(seed=9)
    first = policy.select_arm(context_vector())
    second = policy.select_arm(context_vector())
    assert any(
        first["sampled_scores"][arm] != second["sampled_scores"][arm]
        for arm in ARMS
    )


def test_update_mathematics() -> None:
    policy = make_policy()
    x = context_vector()
    reward = 2.0 / 3.0
    selected_arm = "focus_bias"
    before_A = {arm: policy.A[arm].copy() for arm in ARMS}
    before_b = {arm: policy.b[arm].copy() for arm in ARMS}
    before_counts = dict(policy.arm_update_counts)

    policy.update(selected_arm, x, reward)

    np.testing.assert_allclose(
        policy.A[selected_arm],
        before_A[selected_arm] + np.outer(x, x),
    )
    np.testing.assert_allclose(
        policy.b[selected_arm],
        before_b[selected_arm] + reward * x,
    )
    assert policy.arm_update_counts[selected_arm] == 1
    assert policy.total_updates == 1

    for arm in ARMS:
        if arm == selected_arm:
            continue
        np.testing.assert_array_equal(policy.A[arm], before_A[arm])
        np.testing.assert_array_equal(policy.b[arm], before_b[arm])
        assert policy.arm_update_counts[arm] == before_counts[arm]

    state_before_overflow = state_json(policy)
    expect_rejection(
        lambda: policy.update(
            "baseline",
            np.full(CONTEXT_DIM, 1e308, dtype=np.float64),
            0.5,
        ),
        "overflowing update",
    )
    assert state_json(policy) == state_before_overflow


def test_context_validation() -> None:
    policy = make_policy()
    invalid_contexts = (
        np.ones(CONTEXT_DIM - 1),
        np.ones((1, CONTEXT_DIM)),
        np.asarray([0.0] * (CONTEXT_DIM - 1) + [np.nan]),
        np.asarray([0.0] * (CONTEXT_DIM - 1) + [np.inf]),
        np.asarray([], dtype=np.float64),
        ["0.1"] * CONTEXT_DIM,
    )
    for index, context in enumerate(invalid_contexts):
        expect_rejection(
            lambda context=context: policy.select_arm(context),
            f"invalid context {index}",
        )


def test_reward_validation() -> None:
    x = context_vector()
    for reward in (-1, -0.1, 0, 0.25, 1):
        policy = make_policy()
        policy.update("baseline", x, reward)
        assert policy.total_updates == 1

    for reward in (-1.1, 1.1, np.nan, np.inf, -np.inf, True, False):
        policy = make_policy()
        expect_rejection(
            lambda reward=reward: policy.update("baseline", x, reward),
            f"invalid reward {reward!r}",
        )
        assert policy.total_updates == 0


def test_arm_validation() -> None:
    policy = make_policy()
    expect_rejection(
        lambda: policy.update("unknown", context_vector(), 0.5),
        "unknown arm update",
    )
    expect_rejection(
        lambda: policy.posterior_for_arm("unknown"),
        "unknown arm posterior",
    )


def test_state_json_compatibility() -> None:
    policy = make_policy()
    serialized = json.dumps(policy.state_dict())
    assert isinstance(serialized, str)


def test_state_round_trip_and_rng_restoration() -> None:
    original = make_policy(seed=101)
    x = context_vector()
    for reward in (0.25, 0.75, 1.0):
        decision = original.select_arm(x)
        original.update(decision["selected_arm"], x, reward)

    saved_state = json.loads(json.dumps(original.state_dict()))
    clone = make_policy(seed=999)
    clone.load_state_dict(saved_state, expected_data_mode="synthetic")

    assert clone.arms == original.arms
    assert clone.context_dim == original.context_dim
    assert clone.ridge_lambda == original.ridge_lambda
    assert clone.exploration_scale == original.exploration_scale
    assert clone.data_mode == original.data_mode
    assert clone.arm_update_counts == original.arm_update_counts
    assert clone.total_updates == original.total_updates
    for arm in ARMS:
        np.testing.assert_array_equal(clone.A[arm], original.A[arm])
        np.testing.assert_array_equal(clone.b[arm], original.b[arm])

    original_decision = original.select_arm(x)
    clone_decision = clone.select_arm(x)
    assert original_decision["selected_arm"] == clone_decision["selected_arm"]
    for arm in ARMS:
        assert (
            original_decision["sampled_scores"][arm]
            == clone_decision["sampled_scores"][arm]
        )


def test_synthetic_real_barrier() -> None:
    synthetic_state = make_policy(data_mode="synthetic").state_dict()
    real_policy = make_policy(data_mode="real")
    expect_rejection(
        lambda: real_policy.load_state_dict(
            synthetic_state,
            expected_data_mode="real",
        ),
        "synthetic state into real policy",
    )

    real_state = make_policy(data_mode="real").state_dict()
    synthetic_policy = make_policy(data_mode="synthetic")
    expect_rejection(
        lambda: synthetic_policy.load_state_dict(
            real_state,
            expected_data_mode="synthetic",
        ),
        "real state into synthetic policy",
    )


def test_failed_load_atomicity() -> None:
    policy = make_policy(seed=88)
    x = context_vector()
    policy.update("probing_bias", x, 0.5)
    valid_state = policy.state_dict()
    assert valid_state["schema_version"] == "true_disjoint_lints_v3"

    invalid_states = []

    old_schema = copy.deepcopy(valid_state)
    old_schema["schema_version"] = "true_disjoint_lints_v2"
    invalid_states.append(("old state schema", old_schema))

    wrong_order = copy.deepcopy(valid_state)
    wrong_order["arms"][1], wrong_order["arms"][2] = (
        wrong_order["arms"][2],
        wrong_order["arms"][1],
    )
    invalid_states.append(("arm order", wrong_order))

    wrong_context_dim = copy.deepcopy(valid_state)
    wrong_context_dim["context_dim"] = CONTEXT_DIM + 1
    invalid_states.append(("context dimension", wrong_context_dim))

    wrong_ridge = copy.deepcopy(valid_state)
    wrong_ridge["ridge_lambda"] = 2.0
    invalid_states.append(("ridge lambda", wrong_ridge))

    wrong_exploration = copy.deepcopy(valid_state)
    wrong_exploration["exploration_scale"] = 0.10
    invalid_states.append(("exploration scale", wrong_exploration))

    wrong_matrix_shape = copy.deepcopy(valid_state)
    wrong_matrix_shape["A"]["baseline"] = [[1.0]]
    invalid_states.append(("matrix shape", wrong_matrix_shape))

    nonfinite_matrix = copy.deepcopy(valid_state)
    nonfinite_matrix["A"]["baseline"][0][0] = np.nan
    invalid_states.append(("non-finite matrix", nonfinite_matrix))

    wrong_vector_shape = copy.deepcopy(valid_state)
    wrong_vector_shape["b"]["baseline"] = [0.0]
    invalid_states.append(("vector shape", wrong_vector_shape))

    negative_count = copy.deepcopy(valid_state)
    negative_count["arm_update_counts"]["baseline"] = -1
    invalid_states.append(("negative count", negative_count))

    inconsistent_total = copy.deepcopy(valid_state)
    inconsistent_total["total_updates"] += 1
    invalid_states.append(("inconsistent total", inconsistent_total))

    invalid_rng = copy.deepcopy(valid_state)
    invalid_rng["rng_state"] = {"invalid": True}
    invalid_states.append(("invalid RNG", invalid_rng))

    for name, invalid_state in invalid_states:
        before = state_json(policy)
        expect_rejection(
            lambda invalid_state=invalid_state: policy.load_state_dict(
                invalid_state,
                expected_data_mode="synthetic",
            ),
            name,
        )
        assert state_json(policy) == before


def test_configuration_validation() -> None:
    invalid_constructors = (
        lambda: TrueDisjointLinTS(context_dim=0),
        lambda: TrueDisjointLinTS(context_dim=-1),
        lambda: TrueDisjointLinTS(context_dim=1.5),
        lambda: TrueDisjointLinTS(context_dim=12, ridge_lambda=0.0),
        lambda: TrueDisjointLinTS(context_dim=12, ridge_lambda=-1.0),
        lambda: TrueDisjointLinTS(context_dim=12, ridge_lambda=np.nan),
        lambda: TrueDisjointLinTS(context_dim=12, exploration_scale=0.0),
        lambda: TrueDisjointLinTS(context_dim=12, exploration_scale=-0.1),
        lambda: TrueDisjointLinTS(context_dim=12, exploration_scale=np.inf),
        lambda: TrueDisjointLinTS(context_dim=12, data_mode="training"),
        lambda: TrueDisjointLinTS(context_dim=12, data_mode="Synthetic"),
    )
    for index, constructor in enumerate(invalid_constructors):
        expect_rejection(constructor, f"invalid configuration {index}")


def main() -> None:
    checks = (
        ("arm contract", test_arm_contract),
        ("canonical hyperparameter defaults", test_canonical_defaults),
        ("initial prior", test_initial_prior),
        ("true posterior sampling", test_true_posterior_sampling),
        ("seed reproducibility", test_seed_reproducibility),
        ("RNG advancement", test_rng_advancement),
        ("update mathematics and selected-arm-only update", test_update_mathematics),
        ("context validation", test_context_validation),
        ("reward validation", test_reward_validation),
        ("arm validation", test_arm_validation),
        ("state JSON compatibility", test_state_json_compatibility),
        ("state round-trip and RNG restoration", test_state_round_trip_and_rng_restoration),
        ("synthetic/real barrier", test_synthetic_real_barrier),
        ("failed-load atomicity", test_failed_load_atomicity),
        ("configuration validation", test_configuration_validation),
    )

    print("CANONICAL TRUE DISJOINT LINTS VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
