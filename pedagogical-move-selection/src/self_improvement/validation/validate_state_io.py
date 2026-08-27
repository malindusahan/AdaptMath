"""CPU-only temporary-file validation of canonical LinTS state persistence."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.self_improvement.lints_policy import ARMS, TrueDisjointLinTS
from src.self_improvement.state_io import (
    POLICY_CLASS_NAME,
    POLICY_STATE_SCHEMA_VERSION,
    load_policy_state,
    save_policy_state,
    sha256_file,
)


CONTEXT_DIM = 9
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    try:
        action()
    except (TypeError, ValueError, FileNotFoundError):
        return
    raise AssertionError(f"Invalid state-I/O case was accepted: {case_name}")


def make_policy(
    *,
    context_dim: int = CONTEXT_DIM,
    ridge_lambda: float = RIDGE_LAMBDA,
    exploration_scale: float = EXPLORATION_SCALE,
    seed: int = 42,
    data_mode: str = "synthetic",
) -> TrueDisjointLinTS:
    return TrueDisjointLinTS(
        context_dim=context_dim,
        ridge_lambda=ridge_lambda,
        exploration_scale=exploration_scale,
        seed=seed,
        data_mode=data_mode,
    )


def contexts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(0.05, 0.50, CONTEXT_DIM, dtype=np.float64),
        np.linspace(0.50, 0.05, CONTEXT_DIM, dtype=np.float64),
        np.asarray(
            [0.10, 0.34, 0.40, 0.16, 0.90, 0.80, 0.70, 0.60, 1.0],
            dtype=np.float64,
        ),
    )


def make_nonprior_policy(
    *,
    seed: int = 42,
    data_mode: str = "synthetic",
) -> TrueDisjointLinTS:
    policy = make_policy(seed=seed, data_mode=data_mode)
    eligible_sets = (
        ("baseline", "probing_bias"),
        ("baseline", "generic_bias", "focus_bias"),
        ("baseline", "focus_bias", "telling_bias"),
    )
    rewards = (2.0 / 3.0, 0.75, 1.0)
    weights = (1.0 / 3.0, 0.50, 1.0)
    for context, eligible, reward, weight in zip(
        contexts(),
        eligible_sets,
        rewards,
        weights,
        strict=True,
    ):
        decision = policy.select_arm(context, eligible_arms=eligible)
        assert decision["selected_arm"] in eligible
        policy.update(
            decision["selected_arm"],
            context,
            reward,
            sample_weight=weight,
        )
    return policy


def policy_state_json(policy: TrueDisjointLinTS) -> str:
    return json.dumps(
        policy.state_dict(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def assert_policy_equal(
    original: TrueDisjointLinTS,
    clone: TrueDisjointLinTS,
) -> None:
    assert original.arms == clone.arms == ARMS
    assert original.context_dim == clone.context_dim
    assert original.ridge_lambda == clone.ridge_lambda
    assert original.exploration_scale == clone.exploration_scale
    assert original.data_mode == clone.data_mode
    for arm in ARMS:
        np.testing.assert_array_equal(original.A[arm], clone.A[arm])
        np.testing.assert_array_equal(original.b[arm], clone.b[arm])
    assert original.arm_update_counts == clone.arm_update_counts
    assert original.total_updates == clone.total_updates
    assert original.rng.bit_generator.state == clone.rng.bit_generator.state
    assert policy_state_json(original) == policy_state_json(clone)


def read_json_without_nonfinite(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise AssertionError(f"Non-finite JSON constant found: {value}")

    parsed = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert isinstance(parsed, dict)
    return parsed


def write_json_fixture(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=True, sort_keys=True)
        handle.write("\n")


def test_basic_save_json_and_purity() -> None:
    with TemporaryDirectory() as temporary_directory:
        policy = make_nonprior_policy()
        before = policy_state_json(policy)
        destination = Path(temporary_directory) / "nested" / "policy_state.json"
        returned_path = save_policy_state(policy, destination)
        after = policy_state_json(policy)

        assert returned_path == destination
        assert destination.is_file()
        assert before == after
        envelope = read_json_without_nonfinite(destination)
        assert set(envelope) == {
            "schema_version",
            "policy_class",
            "policy_state",
        }
        assert envelope["schema_version"] == POLICY_STATE_SCHEMA_VERSION
        assert envelope["policy_class"] == POLICY_CLASS_NAME
        assert isinstance(envelope["policy_state"], dict)
        assert envelope["policy_state"]["data_mode"] == "synthetic"
        assert envelope["policy_state"]["total_updates"] == 3
        assert "A" not in {"schema_version", "policy_class", "policy_state"}
        assert "b" not in {"schema_version", "policy_class", "policy_state"}
        json.dumps(envelope, allow_nan=False)
        raw = destination.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw


def test_round_trip_weighted_state_and_next_draw() -> None:
    with TemporaryDirectory() as temporary_directory:
        policy = make_nonprior_policy(seed=2026)
        destination = Path(temporary_directory) / "round_trip.json"
        saved_state = copy.deepcopy(policy.state_dict())
        save_policy_state(policy, destination)

        clone = make_policy(seed=999)
        load_policy_state(clone, destination, expected_data_mode="synthetic")
        assert_policy_equal(policy, clone)

        for arm in ARMS:
            np.testing.assert_array_equal(
                clone.A[arm],
                np.asarray(saved_state["A"][arm], dtype=np.float64),
            )
            np.testing.assert_array_equal(
                clone.b[arm],
                np.asarray(saved_state["b"][arm], dtype=np.float64),
            )
        assert any(
            not np.array_equal(policy.A[arm], np.eye(CONTEXT_DIM))
            for arm in ARMS
        )

        next_context = np.asarray(
            [0.12, 0.28, 0.44, 0.16, 0.80, 0.80, 0.80, 0.70, 1.0],
            dtype=np.float64,
        )
        eligible = ("baseline", "probing_bias", "focus_bias")
        original_decision = policy.select_arm(
            next_context,
            eligible_arms=eligible,
        )
        clone_decision = clone.select_arm(
            next_context,
            eligible_arms=eligible,
        )
        assert original_decision["selected_arm"] == clone_decision["selected_arm"]
        assert original_decision["sampled_scores"] == clone_decision["sampled_scores"]
        assert policy.rng.bit_generator.state == clone.rng.bit_generator.state


def test_data_mode_barrier() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        synthetic_path = root / "synthetic.json"
        synthetic_policy = make_nonprior_policy(data_mode="synthetic", seed=1)
        save_policy_state(synthetic_policy, synthetic_path)

        synthetic_clone = make_policy(data_mode="synthetic", seed=2)
        load_policy_state(
            synthetic_clone,
            synthetic_path,
            expected_data_mode="synthetic",
        )
        assert_policy_equal(synthetic_policy, synthetic_clone)

        real_target = make_policy(data_mode="real", seed=3)
        real_before = policy_state_json(real_target)
        expect_rejection(
            lambda: load_policy_state(
                real_target,
                synthetic_path,
                expected_data_mode="real",
            ),
            "synthetic state into real policy",
        )
        assert policy_state_json(real_target) == real_before

        real_path = root / "real.json"
        real_policy = make_nonprior_policy(data_mode="real", seed=4)
        save_policy_state(real_policy, real_path)
        real_clone = make_policy(data_mode="real", seed=5)
        load_policy_state(real_clone, real_path, expected_data_mode="real")
        assert_policy_equal(real_policy, real_clone)

        synthetic_target = make_policy(data_mode="synthetic", seed=6)
        synthetic_before = policy_state_json(synthetic_target)
        expect_rejection(
            lambda: load_policy_state(
                synthetic_target,
                real_path,
                expected_data_mode="synthetic",
            ),
            "real state into synthetic policy",
        )
        assert policy_state_json(synthetic_target) == synthetic_before

        for invalid_mode in ("training", "Synthetic", "", True, None):
            target = make_policy(data_mode="synthetic", seed=7)
            before = policy_state_json(target)
            expect_rejection(
                lambda invalid_mode=invalid_mode: load_policy_state(
                    target,
                    synthetic_path,
                    expected_data_mode=invalid_mode,  # type: ignore[arg-type]
                ),
                f"expected_data_mode={invalid_mode!r}",
            )
            assert policy_state_json(target) == before


def test_configuration_mismatch_atomicity() -> None:
    with TemporaryDirectory() as temporary_directory:
        policy = make_nonprior_policy(seed=12)
        destination = Path(temporary_directory) / "configuration.json"
        save_policy_state(policy, destination)

        incompatible_policies = (
            ("context_dim", make_policy(context_dim=10)),
            ("ridge_lambda", make_policy(ridge_lambda=2.0)),
            ("exploration_scale", make_policy(exploration_scale=0.10)),
        )
        for name, incompatible in incompatible_policies:
            before = policy_state_json(incompatible)
            expect_rejection(
                lambda incompatible=incompatible: load_policy_state(
                    incompatible,
                    destination,
                    expected_data_mode="synthetic",
                ),
                name,
            )
            assert policy_state_json(incompatible) == before


def test_corruption_rejection_and_load_atomicity() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_policy = make_nonprior_policy(seed=21)
        valid_path = root / "valid.json"
        save_policy_state(source_policy, valid_path)
        valid_envelope = read_json_without_nonfinite(valid_path)

        fixtures: list[tuple[str, object]] = []
        fixtures.append(("malformed JSON", "{not valid json"))

        missing_schema = copy.deepcopy(valid_envelope)
        del missing_schema["schema_version"]
        fixtures.append(("missing schema_version", missing_schema))

        unknown_schema = copy.deepcopy(valid_envelope)
        unknown_schema["schema_version"] = "future_state_v999"
        fixtures.append(("unknown schema version", unknown_schema))

        old_envelope_schema = copy.deepcopy(valid_envelope)
        old_envelope_schema["schema_version"] = "turn_lints_state_v1"
        fixtures.append(("old envelope schema", old_envelope_schema))

        old_policy_schema = copy.deepcopy(valid_envelope)
        old_policy_schema["policy_state"]["schema_version"] = (
            "true_disjoint_lints_v1"
        )
        fixtures.append(("old policy state schema", old_policy_schema))

        missing_policy_state = copy.deepcopy(valid_envelope)
        del missing_policy_state["policy_state"]
        fixtures.append(("missing policy_state", missing_policy_state))

        wrong_class = copy.deepcopy(valid_envelope)
        wrong_class["policy_class"] = "LinUCB"
        fixtures.append(("wrong policy_class", wrong_class))

        invalid_mode = copy.deepcopy(valid_envelope)
        invalid_mode["policy_state"]["data_mode"] = "training"
        fixtures.append(("invalid data_mode", invalid_mode))

        missing_mode = copy.deepcopy(valid_envelope)
        del missing_mode["policy_state"]["data_mode"]
        fixtures.append(("missing data_mode", missing_mode))

        incomplete_state = copy.deepcopy(valid_envelope)
        del incomplete_state["policy_state"]["A"]
        fixtures.append(("incomplete policy state", incomplete_state))

        inconsistent_counts = copy.deepcopy(valid_envelope)
        inconsistent_counts["policy_state"]["total_updates"] += 1
        fixtures.append(("corrupt update counts", inconsistent_counts))

        nonfinite_state = copy.deepcopy(valid_envelope)
        nonfinite_state["policy_state"]["A"]["baseline"][0][0] = float("nan")
        fixtures.append(("non-finite policy state", nonfinite_state))

        for index, (name, fixture) in enumerate(fixtures):
            corrupt_path = root / f"corrupt_{index}.json"
            if isinstance(fixture, str):
                with corrupt_path.open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(fixture)
            else:
                write_json_fixture(corrupt_path, fixture)

            target = make_policy(seed=100 + index)
            before = policy_state_json(target)
            expect_rejection(
                lambda corrupt_path=corrupt_path, target=target: load_policy_state(
                    target,
                    corrupt_path,
                    expected_data_mode="synthetic",
                ),
                name,
            )
            assert policy_state_json(target) == before


def test_atomic_write_safety_and_sha256() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        destination = root / "atomic_state.json"
        policy = make_nonprior_policy(seed=31)
        save_policy_state(policy, destination)
        original_bytes = destination.read_bytes()
        original_hash = sha256_file(destination)
        assert original_hash == sha256_file(destination)
        assert len(original_hash) == 64
        assert all(character in "0123456789abcdef" for character in original_hash)

        with patch.object(
            policy,
            "state_dict",
            return_value={"data_mode": "synthetic", "bad": object()},
        ):
            expect_rejection(
                lambda: save_policy_state(policy, destination),
                "serialization failure",
            )
        assert destination.read_bytes() == original_bytes
        assert sha256_file(destination) == original_hash

        update_context = contexts()[0]
        policy.update(
            "baseline",
            update_context,
            0.50,
            sample_weight=0.25,
        )
        try:
            with patch(
                "src.self_improvement.state_io.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                save_policy_state(policy, destination)
        except OSError as exc:
            assert "injected replace failure" in str(exc)
        else:
            raise AssertionError("Injected atomic replace failure was not raised.")

        assert destination.read_bytes() == original_bytes
        assert sha256_file(destination) == original_hash
        assert list(root.glob(f".{destination.name}.*.tmp")) == []

        save_policy_state(policy, destination)
        changed_hash = sha256_file(destination)
        assert changed_hash != original_hash
        changed_bytes = destination.read_bytes()
        save_policy_state(policy, destination)
        assert destination.read_bytes() == changed_bytes
        assert sha256_file(destination) == changed_hash
        assert list(root.glob(f".{destination.name}.*.tmp")) == []


def main() -> None:
    checks = (
        ("basic JSON save and observational purity", test_basic_save_json_and_purity),
        (
            "round trip, weighted state, and next-draw RNG",
            test_round_trip_weighted_state_and_next_draw,
        ),
        ("synthetic/real hard barrier", test_data_mode_barrier),
        ("configuration mismatch atomicity", test_configuration_mismatch_atomicity),
        (
            "corruption rejection and failed-load atomicity",
            test_corruption_rejection_and_load_atomicity,
        ),
        ("atomic write safety and SHA256", test_atomic_write_safety_and_sha256),
    )

    print("CANONICAL VERSIONED LINTS STATE I/O VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
