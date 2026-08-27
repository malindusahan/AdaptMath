"""Deterministic integration validation for the frozen adaptive pipeline.

The repository does not contain a production tutoring host, mastery store/BKT
implementation, tutor generator, evaluator, or frozen-MRB1 inference wrapper.
This validation therefore uses small test-local stubs for those external
components while exercising the canonical controller, v2 experience logger,
and v2 atomic state persistence without changing their interfaces.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.self_improvement.conservative_overlay import DEFAULT_GAP_THRESHOLD
from src.self_improvement.experience_logger import (
    CREDIT_SCHEME,
    SCHEMA_VERSION,
    ExperienceLogger,
)
from src.self_improvement.lints_policy import (
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    TrueDisjointLinTS,
)
from src.self_improvement.reward import primary_reward
from src.self_improvement.state_io import (
    POLICY_STATE_SCHEMA_VERSION,
    load_policy_state,
    save_policy_state,
)
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
)
from src.self_improvement.turn_level_controller import (
    TurnLevelAttemptController,
)


EXPECTED_FEATURES = (
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "running_mistake_identification",
    "running_mistake_location",
    "running_providing_guidance",
    "running_actionability",
    "has_within_attempt_quality",
)


class RecordingPolicy(TrueDisjointLinTS):
    """Canonical policy with test-only observation of update arguments."""

    def __init__(self, *, seed: int, data_mode: str) -> None:
        super().__init__(
            context_dim=len(TURN_FEATURE_NAMES),
            seed=seed,
            data_mode=data_mode,
        )
        self.observed_updates: list[dict[str, object]] = []

    def update(
        self,
        arm: str,
        context: object,
        reward: object,
        sample_weight: object = 1.0,
    ) -> None:
        self.observed_updates.append(
            {
                "arm": arm,
                "context": tuple(float(value) for value in context),
                "reward": float(reward),
                "sample_weight": float(sample_weight),
            }
        )
        super().update(arm, context, reward, sample_weight)


class StubMasteryState:
    """Test-local stand-in for the repository's missing mastery/BKT service."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def current_mastery(self) -> float:
        self.events.append("mastery_before")
        return 0.42

    def update_with_bkt(self) -> float:
        self.events.append("bkt_mastery_after")
        return 0.55


class StubMD6:
    """Return deterministic valid current-turn frozen-selector probabilities."""

    _PROBABILITIES = (
        {"generic": 0.10, "probing": 0.34, "focus": 0.40, "telling": 0.16},
        {"generic": 0.36, "probing": 0.10, "focus": 0.34, "telling": 0.20},
        {"generic": 0.10, "probing": 0.18, "focus": 0.34, "telling": 0.38},
    )

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def predict_probabilities(self, turn_index: int) -> dict[str, float]:
        self.events.append(f"md6:{turn_index}")
        return dict(self._PROBABILITIES[turn_index - 1])


class StubTutorGenerator:
    """Record that only the controller's final move reaches generation."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.received_moves: list[str] = []

    def generate(self, final_move: str) -> str:
        turn_index = len(self.received_moves) + 1
        self.events.append(f"tutor:{turn_index}")
        self.received_moves.append(final_move)
        return f"response-{turn_index}-for-{final_move}"


class StubMRB1:
    """Return post-response quality scores for the completed tutor turn."""

    _SCORES = (
        {
            "Mistake_Identification": 0.90,
            "Mistake_Location": 0.80,
            "Providing_Guidance": 0.70,
            "Actionability": 0.60,
        },
        {
            "Mistake_Identification": 0.70,
            "Mistake_Location": 0.80,
            "Providing_Guidance": 0.90,
            "Actionability": 0.80,
        },
        {
            "Mistake_Identification": 0.80,
            "Mistake_Location": 0.60,
            "Providing_Guidance": 0.80,
            "Actionability": 0.70,
        },
    )

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def score_response(self, response: str) -> dict[str, float]:
        turn_index = int(response.split("-", maxsplit=2)[1])
        self.events.append(f"mrb1:{turn_index}")
        return dict(self._SCORES[turn_index - 1])


class StubEvaluator:
    """Test-local stand-in for the repository's missing evaluator service."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def score_attempt(self) -> int:
        self.events.append("evaluator")
        return 3


def _expect_rejection(action: Callable[[], object], name: str) -> None:
    try:
        action()
    except (TypeError, ValueError, RuntimeError):
        return
    raise AssertionError(f"Invalid integrated-pipeline case was accepted: {name}")


def _posterior_snapshot(
    policy: TrueDisjointLinTS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], int]:
    return (
        {arm: matrix.copy() for arm, matrix in policy.A.items()},
        {arm: vector.copy() for arm, vector in policy.b.items()},
        dict(policy.arm_update_counts),
        policy.total_updates,
    )


def _assert_posterior_equal(
    policy: TrueDisjointLinTS,
    snapshot: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, int],
        int,
    ],
) -> None:
    matrices, vectors, counts, total = snapshot
    for arm in policy.arms:
        np.testing.assert_array_equal(policy.A[arm], matrices[arm])
        np.testing.assert_array_equal(policy.b[arm], vectors[arm])
    assert policy.arm_update_counts == counts
    assert policy.total_updates == total


def _run_complete_attempt(root: Path) -> None:
    events: list[str] = []
    mastery = StubMasteryState(events)
    md6 = StubMD6(events)
    tutor = StubTutorGenerator(events)
    mrb1 = StubMRB1(events)
    evaluator = StubEvaluator(events)

    policy = RecordingPolicy(seed=20260824, data_mode="synthetic")
    controller = TurnLevelAttemptController(policy)
    assert TURN_FEATURE_NAMES == EXPECTED_FEATURES
    assert policy.context_dim == 9
    assert policy.ridge_lambda == DEFAULT_RIDGE_LAMBDA == 1.0
    assert policy.exploration_scale == DEFAULT_EXPLORATION_SCALE == 0.20
    assert controller.gap_threshold == DEFAULT_GAP_THRESHOLD == 0.10

    mastery_before = mastery.current_mastery()
    controller.start_attempt(mastery_before)
    events.append("start_attempt")
    initial_posterior = _posterior_snapshot(policy)

    decisions = []
    for turn_index in range(1, 4):
        probabilities = md6.predict_probabilities(turn_index)
        decision = controller.select_turn(probabilities)
        events.append(f"select:{turn_index}")
        assert decision.selected_arm in decision.eligible_arms
        response = tutor.generate(decision.final_move)
        scores = mrb1.score_response(response)
        completed = controller.record_mrb1_scores(scores)
        events.append(f"record:{turn_index}")
        assert completed.decision == decision
        decisions.append(decision)
        _assert_posterior_equal(policy, initial_posterior)

    assert all(len(decision.context) == 9 for decision in decisions)
    assert "mastery_before" not in TURN_FEATURE_NAMES
    assert decisions[0].context[4:] == (0.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        decisions[1].context[4:8],
        (0.90, 0.80, 0.70, 0.60),
        rtol=0.0,
        atol=1e-15,
    )
    assert decisions[1].context[8] == 1.0
    np.testing.assert_allclose(
        decisions[2].context[4:8],
        (0.80, 0.80, 0.80, 0.70),
        rtol=0.0,
        atol=1e-15,
    )
    assert decisions[2].context[8] == 1.0
    assert tutor.received_moves == [decision.final_move for decision in decisions]

    evaluator_score = evaluator.score_attempt()
    mastery_after = mastery.update_with_bkt()
    _assert_posterior_equal(policy, initial_posterior)
    completion = controller.finish_attempt(evaluator_score, mastery_after)
    events.append("finish_attempt")

    assert policy.total_updates == 3
    assert len(policy.observed_updates) == 3
    assert all(
        update["sample_weight"] == 1.0 / 3.0
        for update in policy.observed_updates
    )
    assert all(update["reward"] == 1.0 for update in policy.observed_updates)
    assert completion.turn_count == 3
    assert completion.sample_weight_per_turn == 1.0 / 3.0
    assert completion.total_attempt_weight == 1.0
    assert completion.reward == 1.0
    assert [primary_reward(score) for score in range(4)] == [0.0, 0.0, 0.0, 1.0]
    assert completion.mastery_before == 0.42
    assert completion.mastery_after == 0.55
    assert abs(completion.mastery_delta - 0.13) <= 1e-15

    log_path = root / "attempts.jsonl"
    state_path = root / "policy_state.json"
    state_before_log = copy.deepcopy(policy.state_dict())
    logger = ExperienceLogger(
        log_path,
        data_mode="synthetic",
        source_policy=policy,
    )
    record = logger.append_attempt("integration-attempt-001", completion)
    events.append("experience_log")
    assert policy.state_dict() == state_before_log

    assert record["schema_version"] == SCHEMA_VERSION == "turn_lints_v2"
    assert record["mastery_before"] == 0.42
    assert record["mastery_after"] == 0.55
    assert record["mastery_delta"] == completion.mastery_delta
    assert record["evaluator_score"] == 3
    assert record["reward"] == 1.0
    assert record["num_turns"] == 3
    assert record["credit_scheme"] == CREDIT_SCHEME
    assert record["credit_weight_per_turn"] == 1.0 / 3.0
    assert record["total_credit_weight"] == 1.0
    assert [turn["context"] for turn in record["turns"]] == [
        list(decision.context) for decision in decisions
    ]
    assert [turn["selected_arm"] for turn in record["turns"]] == [
        decision.selected_arm for decision in decisions
    ]
    assert [turn["final_move"] for turn in record["turns"]] == (
        tutor.received_moves
    )
    assert record["policy_config"]["context"] == {
        "name": "C3",
        "dimension": 9,
        "feature_order": list(EXPECTED_FEATURES),
    }
    assert record["policy_config"]["gap_threshold"] == 0.10
    assert record["policy_config"]["reward"]["name"] == "success_only"

    save_policy_state(policy, state_path)
    events.append("policy_state_save")
    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == POLICY_STATE_SCHEMA_VERSION
    assert envelope["schema_version"] == "turn_lints_state_v2"
    assert envelope["policy_state"]["schema_version"] == (
        "true_disjoint_lints_v2"
    )

    clone = RecordingPolicy(seed=999, data_mode="synthetic")
    load_policy_state(clone, state_path, expected_data_mode="synthetic")
    assert clone.state_dict() == policy.state_dict()
    next_context = np.asarray(decisions[-1].context, dtype=np.float64)
    eligible = decisions[-1].eligible_arms
    original_next = policy.select_arm(next_context, eligible_arms=eligible)
    clone_next = clone.select_arm(next_context, eligible_arms=eligible)
    assert original_next == clone_next

    real_policy = RecordingPolicy(seed=999, data_mode="real")
    real_before = copy.deepcopy(real_policy.state_dict())
    _expect_rejection(
        lambda: load_policy_state(
            real_policy,
            state_path,
            expected_data_mode="real",
        ),
        "synthetic state into real policy",
    )
    assert real_policy.state_dict() == real_before

    expected_events = ["mastery_before", "start_attempt"]
    for turn_index in range(1, 4):
        expected_events.extend(
            [
                f"md6:{turn_index}",
                f"select:{turn_index}",
                f"tutor:{turn_index}",
                f"mrb1:{turn_index}",
                f"record:{turn_index}",
            ]
        )
    expected_events.extend(
        [
            "evaluator",
            "bkt_mastery_after",
            "finish_attempt",
            "experience_log",
            "policy_state_save",
        ]
    )
    assert events == expected_events


def _valid_probabilities() -> dict[str, float]:
    return {"generic": 0.10, "probing": 0.34, "focus": 0.40, "telling": 0.16}


def _valid_scores() -> dict[str, float]:
    return {
        "Mistake_Identification": 0.90,
        "Mistake_Location": 0.80,
        "Providing_Guidance": 0.70,
        "Actionability": 0.60,
    }


def _run_failure_contracts(root: Path) -> None:
    policy = RecordingPolicy(seed=7, data_mode="synthetic")
    controller = TurnLevelAttemptController(policy)
    initial = _posterior_snapshot(policy)

    _expect_rejection(
        lambda: controller.select_turn(_valid_probabilities()),
        "selection before attempt start",
    )
    controller.start_attempt(0.42)
    _expect_rejection(
        lambda: controller.finish_attempt(3, 0.55),
        "finish with zero completed turns",
    )
    _expect_rejection(
        lambda: controller.select_turn(
            {"generic": 0.2, "probing": 0.2, "focus": 0.2}
        ),
        "malformed MD6 probabilities",
    )
    controller.select_turn(_valid_probabilities())
    _expect_rejection(
        lambda: controller.select_turn(_valid_probabilities()),
        "new selection while MRB1 is pending",
    )
    _expect_rejection(
        lambda: controller.finish_attempt(3, 0.55),
        "finish with a pending response",
    )
    _expect_rejection(
        lambda: controller.record_mrb1_scores(
            {key: 0.5 for key in MRB1_TASKS[:-1]}
        ),
        "invalid MRB1 structure",
    )
    controller.record_mrb1_scores(_valid_scores())
    _expect_rejection(
        lambda: controller.finish_attempt(4, 0.55),
        "invalid evaluator score",
    )
    _assert_posterior_equal(policy, initial)
    controller.abort_attempt()
    _assert_posterior_equal(policy, initial)

    controller.start_attempt(0.42)
    controller.select_turn(_valid_probabilities())
    controller.record_mrb1_scores(_valid_scores())
    completion = controller.finish_attempt(3, 0.55)

    log_path = root / "failure-safety.jsonl"
    logger = ExperienceLogger(
        log_path,
        data_mode="synthetic",
        source_policy=policy,
    )
    logger.append_attempt("valid", completion)
    existing_log = log_path.read_bytes()
    policy_before_failed_log = copy.deepcopy(policy.state_dict())
    _expect_rejection(
        lambda: logger.append_attempt(
            "invalid-metadata",
            completion,
            metadata={"policy_state": {}},
        ),
        "invalid experience metadata",
    )
    assert log_path.read_bytes() == existing_log
    assert policy.state_dict() == policy_before_failed_log

    state_path = root / "failure-safety-state.json"
    save_policy_state(policy, state_path)
    existing_state = state_path.read_bytes()
    with patch(
        "src.self_improvement.state_io.os.replace",
        side_effect=OSError("injected atomic-save failure"),
    ):
        try:
            save_policy_state(policy, state_path)
        except OSError as exc:
            assert "injected atomic-save failure" in str(exc)
        else:
            raise AssertionError("State-save failure was silently hidden.")
    assert state_path.read_bytes() == existing_state


CHECK_NAMES = (
    "attempt starts correctly",
    "exact context dimension is 9",
    "mastery is absent from numerical context",
    "Turn 1 MRB1 context is zeros with flag 0",
    "Turn 2 uses only Turn 1 MRB1",
    "Turn 3 uses the mean of Turns 1-2 MRB1",
    "every selected arm is eligible",
    "overlay final move is passed to tutor generation",
    "no posterior update after Turn 1",
    "no posterior update after Turn 2",
    "no posterior update before finish_attempt",
    "exactly three delayed updates occur at completion",
    "each delayed update has weight 1/3",
    "total delayed attempt weight is 1",
    "success-only binary reward mapping is exact",
    "mastery delta is exact",
    "experience record is valid turn_lints_v2",
    "policy state saves and loads with v2 schemas",
    "RNG state survives persistence round trip",
    "synthetic/real state barrier remains enforced",
)


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        _run_complete_attempt(root)
        _run_failure_contracts(root)

    print("FROZEN ADAPTIVE PIPELINE INTEGRATION VALIDATION")
    for index, name in enumerate(CHECK_NAMES, start=1):
        print(f"{index}. {name}: PASS")
    print("FAILURE CONTRACTS AND FILE SAFETY: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
