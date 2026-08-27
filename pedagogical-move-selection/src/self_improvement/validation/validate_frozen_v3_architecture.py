"""Deterministic CPU-light validation of the frozen v3 adaptive runtime.

The checks exercise implementation contracts only. They do not call frozen
models, external services, held-out datasets, or scientific experiments.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from math import inf, isclose, nan
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.self_improvement.adaptive_tutor_pipeline import AdaptiveTutorPipeline
from src.self_improvement.experience_logger import (
    REWARD_NAME,
    SCHEMA_VERSION,
    ExperienceLogger,
)
from src.self_improvement.lints_policy import (
    ARMS,
    LINTS_STATE_SCHEMA_VERSION,
    TrueDisjointLinTS,
)
from src.self_improvement.reward import (
    LearningOutcome,
    mastery_delta_reward,
    validate_learning_outcome,
)
from src.self_improvement.state_io import (
    POLICY_STATE_SCHEMA_VERSION,
    load_policy_state,
    save_policy_state,
    sha256_file,
)
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
    build_turn_context,
)
from src.self_improvement.turn_level_controller import TurnLevelAttemptController


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

PROBABILITIES = (
    {"generic": 0.10, "probing": 0.34, "focus": 0.40, "telling": 0.16},
    {"generic": 0.36, "probing": 0.10, "focus": 0.34, "telling": 0.20},
    {"generic": 0.10, "probing": 0.18, "focus": 0.34, "telling": 0.38},
)

SCORES = (
    {
        "Mistake_Identification": 0.90,
        "Mistake_Location": 0.80,
        "Providing_Guidance": 0.70,
        "Actionability": 0.60,
    },
    {
        "Mistake_Identification": 0.70,
        "Mistake_Location": 0.60,
        "Providing_Guidance": 0.50,
        "Actionability": 0.40,
    },
    {
        "Mistake_Identification": 0.50,
        "Mistake_Location": 0.40,
        "Providing_Guidance": 0.30,
        "Actionability": 0.20,
    },
)


def expect_rejection(action: Callable[[], object], name: str) -> None:
    try:
        action()
    except (TypeError, ValueError, RuntimeError):
        return
    raise AssertionError(f"Invalid v3 case was accepted: {name}.")


def outcome(
    before: object,
    after: object,
    delta: object,
    **diagnostics: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "skill": "Percent Of",
        "mastery_before": before,
        "mastery_after": after,
        "delta_mastery": delta,
    }
    result.update(diagnostics)
    return result


def new_policy(*, seed: int = 42, data_mode: str = "synthetic") -> TrueDisjointLinTS:
    return TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=seed,
        data_mode=data_mode,
    )


def complete_three_turn_controller(
    *,
    before: float = 0.40,
    after: float = 0.30,
    reward: float = -0.10,
) -> tuple[TrueDisjointLinTS, TurnLevelAttemptController, object]:
    policy = new_policy(seed=17)
    controller = TurnLevelAttemptController(policy)
    controller.start_attempt(before)
    for probabilities, scores in zip(PROBABILITIES, SCORES, strict=True):
        controller.select_turn(probabilities)
        controller.record_mrb1_scores(scores)
    completion = controller.finish_attempt(reward=reward, mastery_after=after)
    return policy, controller, completion


class StubMD6:
    def __init__(self) -> None:
        self.calls = 0

    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> Mapping[str, float]:
        del problem, conversation_history
        result = PROBABILITIES[self.calls % len(PROBABILITIES)]
        self.calls += 1
        return dict(result)


class StubMRB1:
    def __init__(self) -> None:
        self.calls = 0

    def score_response(
        self,
        conversation_history: Sequence[Mapping[str, object]],
        tutor_response: str,
    ) -> Mapping[str, float]:
        del conversation_history, tutor_response
        result = SCORES[self.calls % len(SCORES)]
        self.calls += 1
        return dict(result)


class StubTutor:
    def generate(
        self,
        *,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
        pedagogical_move: str,
    ) -> str:
        del problem, conversation_history
        return f"Tutor response for {pedagogical_move}."


def memory(attempt_id: str, mastery_before: float = 0.30) -> dict[str, object]:
    return {
        "attempt_id": attempt_id,
        "problem": "What is 25 percent of 80?",
        "conversation_history": [
            {"user": "student", "text": "I am not sure how to begin."}
        ],
        "mastery_before": mastery_before,
    }


def build_pipeline(
    root: Path,
    *,
    attempt_id: str,
    seed: int = 2026,
    mastery_before: float = 0.30,
) -> tuple[
    AdaptiveTutorPipeline,
    TrueDisjointLinTS,
    dict[str, object],
    Path,
    Path,
]:
    policy = new_policy(seed=seed, data_mode="real")
    controller = TurnLevelAttemptController(policy)
    log_path = root / f"{attempt_id}_attempts_v3.jsonl"
    state_path = root / f"{attempt_id}_policy_state_v3.json"
    logger = ExperienceLogger(log_path, data_mode="real", source_policy=policy)
    pipeline = AdaptiveTutorPipeline(
        md6=StubMD6(),
        mrb1=StubMRB1(),
        controller=controller,
        experience_logger=logger,
        policy_state_path=state_path,
    )
    return (
        pipeline,
        policy,
        memory(attempt_id, mastery_before),
        log_path,
        state_path,
    )


def test_a_context_freeze() -> None:
    assert TURN_FEATURE_NAMES == EXPECTED_FEATURES
    assert len(TURN_FEATURE_NAMES) == 9
    forbidden = {
        "skill",
        "mastery_before",
        "mastery_after",
        "delta_mastery",
        "reasoning",
        "uncertainty",
        "clarification",
        "repeated_misunderstanding",
        "evaluator_result",
    }
    assert forbidden.isdisjoint(TURN_FEATURE_NAMES)
    assert tuple(inspect.signature(build_turn_context).parameters) == (
        "md6_probabilities",
        "running_quality",
    )


def test_b_positive_delta() -> None:
    value = outcome(0.050, 0.116, 0.066)
    assert mastery_delta_reward(value) == 0.066


def test_c_negative_delta() -> None:
    value = outcome(0.116, 0.049, -0.067)
    assert mastery_delta_reward(value) == -0.067


def test_d_zero_delta() -> None:
    assert mastery_delta_reward(outcome(0.30, 0.30, 0.0)) == 0.0


def test_e_inconsistent_delta_rejected() -> None:
    expect_rejection(
        lambda: validate_learning_outcome(outcome(0.10, 0.20, 0.05)),
        "inconsistent delta",
    )


def test_f_range_and_type_validation() -> None:
    invalid = (
        outcome(-0.01, 0.20, 0.21),
        outcome(0.10, 1.01, 0.91),
        outcome(1.0, 0.0, -1.01),
        outcome(0.0, 1.0, 1.01),
        outcome(nan, 0.20, 0.0),
        outcome(0.10, inf, 0.0),
        outcome(0.10, 0.20, nan),
        outcome(True, 0.20, 0.0),
        outcome(0.10, False, 0.0),
        outcome(0.10, 0.20, True),
        {**outcome(0.10, 0.20, 0.10), "skill": "   "},
    )
    for index, value in enumerate(invalid):
        expect_rejection(
            lambda value=value: validate_learning_outcome(value),
            f"range/type case {index}",
        )


def test_g_negative_lints_update() -> None:
    policy = new_policy()
    context = np.linspace(0.1, 0.9, len(TURN_FEATURE_NAMES))
    arm = "focus_bias"
    weight = 0.25
    negative_reward = -0.40
    before_b = policy.b[arm].copy()
    policy.update(arm, context, negative_reward, sample_weight=weight)
    expected = before_b + weight * negative_reward * context
    np.testing.assert_allclose(policy.b[arm], expected, rtol=0.0, atol=1e-15)


def test_h_no_within_attempt_update_and_causal_mrb1() -> None:
    policy = new_policy(seed=31)
    controller = TurnLevelAttemptController(policy)
    controller.start_attempt(0.40)
    update_count = policy.total_updates
    contexts: list[tuple[float, ...]] = []
    for probabilities, scores in zip(PROBABILITIES, SCORES, strict=True):
        decision = controller.select_turn(probabilities)
        contexts.append(decision.context)
        assert policy.total_updates == update_count
        controller.record_mrb1_scores(scores)
        assert policy.total_updates == update_count

    assert contexts[0][4:] == (0.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(contexts[1][4:8], tuple(SCORES[0].values()))
    np.testing.assert_allclose(
        contexts[2][4:8],
        tuple((SCORES[0][task] + SCORES[1][task]) / 2.0 for task in MRB1_TASKS),
    )
    assert contexts[1][8] == contexts[2][8] == 1.0


def test_i_equal_delayed_credit() -> None:
    policy, controller, completion = complete_three_turn_controller()
    assert isclose(completion.sample_weight_per_turn, 1.0 / 3.0)
    assert isclose(completion.total_attempt_weight, 1.0)
    assert completion.turn_count == 3
    assert policy.total_updates == 3
    assert controller.number_of_weighted_turn_updates == 3
    assert completion.number_of_weighted_turn_updates == 3


def test_j_reward_consistency() -> None:
    _, _, completion = complete_three_turn_controller()
    learning_outcome = validate_learning_outcome(outcome(0.40, 0.30, -0.10))
    assert isclose(
        completion.reward,
        completion.mastery_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert isclose(
        completion.reward,
        learning_outcome.delta_mastery,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_k_optional_diagnostics_are_non_scientific() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        plain = build_pipeline(root, attempt_id="plain", seed=91)
        diagnostic = build_pipeline(root, attempt_id="diagnostic", seed=91)

        plain_pipeline, _, plain_memory, _, _ = plain
        diagnostic_pipeline, _, diagnostic_memory, _, _ = diagnostic
        plain_pipeline.start_attempt(plain_memory)
        diagnostic_pipeline.start_attempt(diagnostic_memory)
        plain_turn = plain_pipeline.run_tutor_turn(plain_memory, StubTutor())
        diagnostic_turn = diagnostic_pipeline.run_tutor_turn(
            diagnostic_memory,
            StubTutor(),
        )

        assert plain_turn["context"] == diagnostic_turn["context"]
        assert plain_turn["eligible_arms"] == diagnostic_turn["eligible_arms"]
        plain_result = plain_pipeline.finish_attempt(
            plain_memory,
            outcome(0.30, 0.35, 0.05),
        )
        diagnostic_result = diagnostic_pipeline.finish_attempt(
            diagnostic_memory,
            outcome(
                0.30,
                0.35,
                0.05,
                evaluator_result={"score": 2},
                reasoning="Already supplied diagnostic text.",
                uncertainty=0.25,
                clarification=False,
                repeated_misunderstanding=True,
            ),
        )
        assert plain_result["reward"] == diagnostic_result["reward"] == 0.05
        assert "diagnostics" not in plain_result["experience_record"]
        assert set(diagnostic_result["experience_record"]["diagnostics"]) == {
            "evaluator_result",
            "reasoning",
            "uncertainty",
            "clarification",
            "repeated_misunderstanding",
        }


def test_l_v3_logger_contract() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        pipeline, _, host_memory, log_path, _ = build_pipeline(
            root,
            attempt_id="logger",
        )
        pipeline.start_attempt(host_memory)
        pipeline.run_tutor_turn(host_memory, StubTutor())
        result = pipeline.finish_attempt(
            host_memory,
            outcome(0.30, 0.35, 0.05),
        )
        record = result["experience_record"]
        assert record["schema_version"] == SCHEMA_VERSION == "turn_lints_v3"
        assert record["policy_config"]["context"]["dimension"] == 9
        assert record["policy_config"]["reward"] == {
            "name": "mastery_delta",
            "source": "delta_mastery",
            "formula": "mastery_after - mastery_before",
            "range": [-1.0, 1.0],
        }
        assert record["learning_outcome"] == outcome(0.30, 0.35, 0.05)
        saved = json.loads(log_path.read_text(encoding="utf-8"))
        assert saved == record


def test_m_old_log_rejection() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        old_log = root / "attempts_v2.jsonl"
        old_log.write_text(
            json.dumps({"schema_version": "turn_lints_v2"}) + "\n",
            encoding="utf-8",
        )
        policy = new_policy(data_mode="real")
        expect_rejection(
            lambda: ExperienceLogger(
                old_log,
                data_mode="real",
                source_policy=policy,
            ),
            "v2 log append target",
        )


def test_n_v3_state_round_trip() -> None:
    with TemporaryDirectory() as temporary_directory:
        state_path = Path(temporary_directory) / "policy_state_v3.json"
        source = new_policy(seed=810, data_mode="real")
        context = np.linspace(0.05, 0.45, len(TURN_FEATURE_NAMES))
        source.update("probing_bias", context, -0.20, sample_weight=0.5)
        source.select_arm(context)
        save_policy_state(source, state_path)

        receiving = new_policy(seed=999, data_mode="real")
        load_policy_state(receiving, state_path, expected_data_mode="real")
        assert json.dumps(source.state_dict(), sort_keys=True) == json.dumps(
            receiving.state_dict(),
            sort_keys=True,
        )
        source_decision = source.select_arm(context)
        receiving_decision = receiving.select_arm(context)
        assert source_decision == receiving_decision


def test_o_old_state_rejection_without_migration() -> None:
    policy = new_policy(data_mode="real")
    old_internal = copy.deepcopy(policy.state_dict())
    old_internal["schema_version"] = "true_disjoint_lints_v2"
    expect_rejection(
        lambda: policy.load_state_dict(old_internal, expected_data_mode="real"),
        "v2 internal state",
    )

    with TemporaryDirectory() as temporary_directory:
        old_path = Path(temporary_directory) / "real_policy_state_v2.json"
        old_path.write_text(
            json.dumps(
                {
                    "schema_version": "turn_lints_state_v2",
                    "policy_class": "TrueDisjointLinTS",
                    "policy_state": old_internal,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        original_hash = sha256_file(old_path)
        expect_rejection(
            lambda: load_policy_state(policy, old_path, expected_data_mode="real"),
            "v2 envelope load",
        )
        expect_rejection(
            lambda: save_policy_state(policy, old_path),
            "v2 envelope overwrite",
        )
        assert sha256_file(old_path) == original_hash


def test_p_data_mode_barrier() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        synthetic_path = root / "synthetic_state_v3.json"
        real_path = root / "real_state_v3.json"
        save_policy_state(new_policy(data_mode="synthetic"), synthetic_path)
        save_policy_state(new_policy(data_mode="real"), real_path)
        expect_rejection(
            lambda: load_policy_state(
                new_policy(data_mode="real"),
                synthetic_path,
                expected_data_mode="real",
            ),
            "synthetic into real",
        )
        expect_rejection(
            lambda: load_policy_state(
                new_policy(data_mode="synthetic"),
                real_path,
                expected_data_mode="synthetic",
            ),
            "real into synthetic",
        )


def test_q_retry_safety() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        pipeline, policy, host_memory, log_path, state_path = build_pipeline(
            root,
            attempt_id="retry",
        )
        pipeline.start_attempt(host_memory)
        pipeline.run_tutor_turn(host_memory, StubTutor())
        fixed_outcome = outcome(0.30, 0.35, 0.05)
        fixed_metadata = {"source": "retry-validation"}

        with patch(
            "src.self_improvement.adaptive_tutor_pipeline.save_policy_state",
            side_effect=OSError("forced state-save failure"),
        ):
            try:
                pipeline.finish_attempt(
                    host_memory,
                    fixed_outcome,
                    metadata=fixed_metadata,
                )
            except OSError:
                pass
            else:
                raise AssertionError("Forced state-save failure was not propagated.")

        assert policy.total_updates == 1
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1
        changed_outcome = outcome(0.30, 0.36, 0.06)
        expect_rejection(
            lambda: pipeline.finish_attempt(
                host_memory,
                changed_outcome,
                metadata=fixed_metadata,
            ),
            "changed retry learning outcome",
        )
        assert policy.total_updates == 1
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

        result = pipeline.finish_attempt(
            host_memory,
            fixed_outcome,
            metadata=fixed_metadata,
        )
        assert result["reward"] == 0.05
        assert policy.total_updates == 1
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1
        assert state_path.is_file()


def test_r_binary_rewards_remain_legal() -> None:
    policy = new_policy()
    context = np.ones(len(TURN_FEATURE_NAMES), dtype=np.float64)
    policy.update("baseline", context, 0.0)
    policy.update("baseline", context, 1.0)
    assert policy.total_updates == 2


CHECKS = (
    ("A context freeze", test_a_context_freeze),
    ("B positive delta", test_b_positive_delta),
    ("C negative delta", test_c_negative_delta),
    ("D zero delta", test_d_zero_delta),
    ("E inconsistent delta rejected", test_e_inconsistent_delta_rejected),
    ("F range/type validation", test_f_range_and_type_validation),
    ("G negative LinTS update", test_g_negative_lints_update),
    ("H no within-attempt update", test_h_no_within_attempt_update_and_causal_mrb1),
    ("I equal delayed credit", test_i_equal_delayed_credit),
    ("J reward consistency", test_j_reward_consistency),
    ("K optional diagnostics", test_k_optional_diagnostics_are_non_scientific),
    ("L v3 logger", test_l_v3_logger_contract),
    ("M old log rejection", test_m_old_log_rejection),
    ("N state round trip", test_n_v3_state_round_trip),
    ("O old state rejection", test_o_old_state_rejection_without_migration),
    ("P data-mode barrier", test_p_data_mode_barrier),
    ("Q retry safety", test_q_retry_safety),
    ("R binary reward compatibility", test_r_binary_rewards_remain_legal),
)


def main() -> None:
    assert SCHEMA_VERSION == "turn_lints_v3"
    assert LINTS_STATE_SCHEMA_VERSION == "true_disjoint_lints_v3"
    assert POLICY_STATE_SCHEMA_VERSION == "turn_lints_state_v3"
    assert REWARD_NAME == "mastery_delta"
    print("FROZEN V3 IMPLEMENTATION VALIDATION")
    for name, check in CHECKS:
        check()
        print(f"- {name}: PASS")
    print(f"PASSED CHECKS: {len(CHECKS)}")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
