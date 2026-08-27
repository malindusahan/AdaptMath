import numpy as np

from src.self_improvement.lints_policy import LinTSPolicy


ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias",
]


def main():

    print("=" * 70)
    print("SI6-C TRUE LinTS VALIDATION")
    print("=" * 70)
    print()

    context = np.array(
        [
            0.42,   # mastery_before
            0.10,   # normalized turn number
            0.35,   # normalized student response length
            1.00,   # asked_question
            1.00,   # bias
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------
    # TEST 1:
    # Basic Thompson Sampling arm selection
    # --------------------------------------------------

    policy = LinTSPolicy(
        n_features=5,
        arms=ARMS,
        alpha=1.0,
        seed=42,
    )

    selected_arm = policy.select_arm(
        context
    )

    assert selected_arm in ARMS

    print(
        "Initial selected arm:",
        selected_arm,
    )

    print(
        "Arm selection: PASSED"
    )

    # --------------------------------------------------
    # TEST 2:
    # Same seed must generate same sampling sequence
    # --------------------------------------------------

    policy_a = LinTSPolicy(
        n_features=5,
        arms=ARMS,
        alpha=1.0,
        seed=12345,
    )

    policy_b = LinTSPolicy(
        n_features=5,
        arms=ARMS,
        alpha=1.0,
        seed=12345,
    )

    sequence_a = []

    sequence_b = []

    for _ in range(20):

        sequence_a.append(
            policy_a.select_arm(
                context
            )
        )

        sequence_b.append(
            policy_b.select_arm(
                context
            )
        )

    assert sequence_a == sequence_b

    print(
        "Seed reproducibility: PASSED"
    )

    # --------------------------------------------------
    # TEST 3:
    # Thompson Sampling should explore different arms
    # --------------------------------------------------

    exploration_policy = LinTSPolicy(
        n_features=5,
        arms=ARMS,
        alpha=1.0,
        seed=2026,
    )

    sampled_arms = []

    for _ in range(100):

        sampled_arms.append(
            exploration_policy.select_arm(
                context
            )
        )

    unique_arms = sorted(
        set(sampled_arms)
    )

    print(
        "Unique arms sampled:",
        unique_arms,
    )

    assert len(unique_arms) >= 2

    print(
        "Posterior sampling exploration: PASSED"
    )

    # --------------------------------------------------
    # TEST 4:
    # Verify update mathematics
    # --------------------------------------------------

    update_policy = LinTSPolicy(
        n_features=5,
        arms=ARMS,
        alpha=1.0,
        seed=7,
    )

    update_arm = "focus_bias"

    reward = 0.925

    A_before = {}

    b_before = {}

    for arm in ARMS:

        A_before[arm] = (
            update_policy.A[arm].copy()
        )

        b_before[arm] = (
            update_policy.b[arm].copy()
        )

    update_policy.update(
        arm=update_arm,
        context=context,
        reward=reward,
    )

    expected_A = (
        A_before[update_arm]
        +
        np.outer(
            context,
            context,
        )
    )

    expected_b = (
        b_before[update_arm]
        +
        reward * context
    )

    assert np.allclose(
        update_policy.A[update_arm],
        expected_A,
    )

    assert np.allclose(
        update_policy.b[update_arm],
        expected_b,
    )

    print(
        "Selected-arm update: PASSED"
    )

    # --------------------------------------------------
    # TEST 5:
    # Non-selected arms must remain unchanged
    # --------------------------------------------------

    for arm in ARMS:

        if arm == update_arm:
            continue

        assert np.allclose(
            update_policy.A[arm],
            A_before[arm],
        )

        assert np.allclose(
            update_policy.b[arm],
            b_before[arm],
        )

    print(
        "Non-selected arms unchanged: PASSED"
    )

    # --------------------------------------------------
    # TEST 6:
    # Posterior mean/covariance should change
    # after observing reward
    # --------------------------------------------------

    mean_before = np.zeros(
        5,
        dtype=np.float64,
    )

    mean_after, covariance_after = (
        update_policy.arm_posterior(
            update_arm
        )
    )

    assert not np.allclose(
        mean_after,
        mean_before,
    )

    assert covariance_after.shape == (
        5,
        5,
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_after
    )

    assert np.all(
        eigenvalues > 0
    )

    print(
        "Posterior changed after reward: PASSED"
    )

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("TRUE LinTS VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()