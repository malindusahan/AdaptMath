"""Temporary-file validation of canonical turn-level trajectory logging."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from math import inf, isclose, nan
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.self_improvement.experience_logger import (
    CONTEXT_NAME,
    CREDIT_SCHEME,
    REWARD_NAME,
    SCHEMA_VERSION,
    SELECTION_MODE,
    UPDATE_TIMING,
    ExperienceLogger,
    serialize_attempt,
)
from src.self_improvement.lints_policy import (
    ARMS,
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    TrueDisjointLinTS,
)
from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
    TutorQualitySnapshot,
)
from src.self_improvement.turn_level_controller import (
    AttemptCompletion,
    TurnLevelAttemptController,
)


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    try:
        action()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"Invalid logger case was accepted: {case_name}")


def make_policy(
    data_mode: str = "synthetic",
    seed: int = 42,
) -> TrueDisjointLinTS:
    return TrueDisjointLinTS(
        context_dim=9,
        seed=seed,
        data_mode=data_mode,
    )


def probabilities_one() -> dict[str, float]:
    return {
        "generic": 0.10,
        "probing": 0.34,
        "focus": 0.40,
        "telling": 0.16,
    }


def probabilities_two() -> dict[str, float]:
    return {
        "generic": 0.36,
        "probing": 0.10,
        "focus": 0.34,
        "telling": 0.20,
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


def complete_two_turn_attempt(
    controller: TurnLevelAttemptController,
    *,
    mastery_before: float = 0.42,
    mastery_after: float = 0.55,
    evaluator_score: int = 2,
) -> AttemptCompletion:
    controller.start_attempt(mastery_before)
    controller.select_turn(probabilities_one())
    controller.record_mrb1_scores(scores_one())
    controller.select_turn(probabilities_two())
    controller.record_mrb1_scores(scores_two())
    return controller.finish_attempt(evaluator_score, mastery_after)


def policy_state_json(policy: TrueDisjointLinTS) -> str:
    return json.dumps(policy.state_dict(), sort_keys=True, separators=(",", ":"))


def recursively_collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(recursively_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursively_collect_keys(item))
    return keys


def test_json_trajectory_and_observational_purity() -> None:
    with TemporaryDirectory() as temporary_directory:
        policy = make_policy(seed=2026)
        controller = TurnLevelAttemptController(policy)
        completion = complete_two_turn_attempt(controller)
        state_before_logging = policy_state_json(policy)

        path = Path(temporary_directory) / "nested" / "attempts.jsonl"
        logger = ExperienceLogger(
            path,
            data_mode="synthetic",
            source_policy=policy,
        )
        returned = logger.append_attempt(
            "attempt-001",
            completion,
            metadata={
                "run_id": "validation-run",
                "seed": np.int64(2026),
                "folds": np.asarray([1, 2], dtype=np.int64),
                "tags": ("synthetic", "smoke"),
            },
        )
        state_after_logging = policy_state_json(policy)
        assert state_after_logging == state_before_logging

        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        lines = raw.splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed == returned
        json.dumps(parsed, allow_nan=False)

        assert parsed["schema_version"] == SCHEMA_VERSION
        assert parsed["attempt_id"] == "attempt-001"
        assert parsed["data_mode"] == "synthetic"
        assert parsed["policy_config"] == {
            "context": {
                "name": CONTEXT_NAME,
                "dimension": 9,
                "feature_order": list(TURN_FEATURE_NAMES),
            },
            "gap_threshold": 0.10,
            "reward": {
                "name": REWARD_NAME,
                "evaluator_score_mapping": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 0.0,
                    "3": 1.0,
                },
            },
            "credit": {"name": CREDIT_SCHEME, "sample_weight": "1/T"},
            "ridge_lambda": DEFAULT_RIDGE_LAMBDA,
            "exploration_scale": DEFAULT_EXPLORATION_SCALE,
            "selection": SELECTION_MODE,
            "update_timing": UPDATE_TIMING,
        }
        assert parsed["mastery_before"] == 0.42
        assert parsed["mastery_after"] == 0.55
        assert isclose(parsed["mastery_delta"], 0.13, abs_tol=1e-15)
        assert parsed["evaluator_score"] == 2
        assert parsed["reward"] == 0.0
        assert parsed["num_turns"] == 2
        assert parsed["credit_scheme"] == CREDIT_SCHEME
        assert parsed["credit_weight_per_turn"] == 0.5
        assert parsed["total_credit_weight"] == 1.0
        assert parsed["controller_counters"] == {
            "completed_attempts": 1,
            "number_of_weighted_turn_updates": 2,
        }
        assert parsed["metadata"] == {
            "run_id": "validation-run",
            "seed": 2026,
            "folds": [1, 2],
            "tags": ["synthetic", "smoke"],
        }

        expected_final = {
            "Mistake_Identification": 0.80,
            "Mistake_Location": 0.80,
            "Providing_Guidance": 0.80,
            "Actionability": 0.70,
        }
        for task in MRB1_TASKS:
            assert isclose(
                parsed["final_average_mrb1"][task],
                expected_final[task],
                abs_tol=1e-15,
            )

        assert [turn["turn_index"] for turn in parsed["turns"]] == [1, 2]
        for logged_turn, completed_turn in zip(
            parsed["turns"],
            completion.turns,
            strict=True,
        ):
            decision = completed_turn.decision
            assert len(logged_turn["context"]) == 9
            assert logged_turn["context"] == list(decision.context)
            assert logged_turn["md6_probabilities"] == dict(
                decision.md6_probabilities
            )
            assert logged_turn["eligible_arms"] == list(decision.eligible_arms)
            assert logged_turn["sampled_scores"] == dict(decision.sampled_scores)
            assert logged_turn["selected_arm"] == decision.selected_arm
            assert logged_turn["base_move"] == decision.base_move
            assert logged_turn["final_move"] == decision.final_move
            assert logged_turn["target_move"] == decision.target_move
            assert logged_turn["overridden"] is decision.overridden
            assert logged_turn["gap"] == decision.gap
            assert logged_turn["gap_threshold"] == decision.gap_threshold
            assert logged_turn["base_probability"] == decision.base_probability
            assert logged_turn["target_probability"] == (
                decision.target_probability
            )
            assert logged_turn["mrb1_scores"] == dict(completed_turn.mrb1_scores)

        first_context = parsed["turns"][0]["context"]
        assert first_context[4:8] == [0.0, 0.0, 0.0, 0.0]
        assert first_context[8] == 0.0
        second_context = parsed["turns"][1]["context"]
        np.testing.assert_allclose(
            second_context[4:8],
            [0.90, 0.80, 0.70, 0.60],
            rtol=0.0,
            atol=1e-15,
        )
        assert second_context[8] == 1.0

        all_keys = recursively_collect_keys(parsed)
        assert "A" not in all_keys
        assert "b" not in all_keys
        assert "model_weights" not in all_keys
        assert "tokenizer_contents" not in all_keys


def test_append_only_behavior() -> None:
    with TemporaryDirectory() as temporary_directory:
        policy = make_policy(seed=81)
        controller = TurnLevelAttemptController(policy)
        path = Path(temporary_directory) / "append" / "attempts.jsonl"
        logger = ExperienceLogger(
            path,
            data_mode="synthetic",
            source_policy=policy,
        )

        first_completion = complete_two_turn_attempt(controller)
        before_first_log = policy_state_json(policy)
        logger.append_attempt("attempt-001", first_completion)
        assert policy_state_json(policy) == before_first_log
        first_line = path.read_text(encoding="utf-8").splitlines()[0]

        second_completion = complete_two_turn_attempt(
            controller,
            mastery_before=0.55,
            mastery_after=0.65,
            evaluator_score=3,
        )
        before_second_log = policy_state_json(policy)
        logger.append_attempt("attempt-002", second_completion)
        assert policy_state_json(policy) == before_second_log

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert lines[0] == first_line
        first_record, second_record = (json.loads(line) for line in lines)
        assert first_record["attempt_id"] == "attempt-001"
        assert second_record["attempt_id"] == "attempt-002"
        assert first_record["schema_version"] == SCHEMA_VERSION
        assert second_record["schema_version"] == SCHEMA_VERSION


def test_data_mode_barrier() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)

        synthetic_policy = make_policy(data_mode="synthetic", seed=10)
        synthetic_controller = TurnLevelAttemptController(synthetic_policy)
        synthetic_completion = complete_two_turn_attempt(synthetic_controller)
        synthetic_path = root / "synthetic.jsonl"
        ExperienceLogger(
            synthetic_path,
            data_mode="synthetic",
            source_policy=synthetic_policy,
        ).append_attempt("synthetic-attempt", synthetic_completion)

        real_policy = make_policy(data_mode="real", seed=11)
        expect_rejection(
            lambda: ExperienceLogger(
                synthetic_path,
                data_mode="real",
                source_policy=real_policy,
            ),
            "synthetic log opened as real",
        )
        expect_rejection(
            lambda: ExperienceLogger(
                root / "mode-mismatch.jsonl",
                data_mode="real",
                source_policy=synthetic_policy,
            ),
            "real logger bound to synthetic policy",
        )
        expect_rejection(
            lambda: serialize_attempt(
                "mode-mismatch",
                synthetic_completion,
                "real",
                source_policy=synthetic_policy,
            ),
            "real serializer bound to synthetic policy",
        )

        real_controller = TurnLevelAttemptController(real_policy)
        real_completion = complete_two_turn_attempt(real_controller)
        real_path = root / "real.jsonl"
        ExperienceLogger(
            real_path,
            data_mode="real",
            source_policy=real_policy,
        ).append_attempt("real-mode-validation-attempt", real_completion)
        expect_rejection(
            lambda: ExperienceLogger(
                real_path,
                data_mode="synthetic",
                source_policy=synthetic_policy,
            ),
            "real log opened as synthetic",
        )


def test_invalid_record_rejection() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        policy = make_policy(seed=5)
        controller = TurnLevelAttemptController(policy)
        completion = complete_two_turn_attempt(controller)

        for attempt_id in ("", "   ", None, [], True):
            expect_rejection(
                lambda attempt_id=attempt_id: serialize_attempt(
                    attempt_id,
                    completion,
                    "synthetic",
                    source_policy=policy,
                ),
                f"attempt_id={attempt_id!r}",
            )
        expect_rejection(
            lambda: serialize_attempt(
                "attempt",
                completion,
                "training",
                source_policy=policy,
            ),
            "unsupported data_mode",
        )
        expect_rejection(
            lambda: serialize_attempt(
                "attempt",
                object(),
                "synthetic",
                source_policy=policy,
            ),
            "non-completion object",
        )

        zero_turn = replace(
            completion,
            turns=(),
            turn_count=0,
            sample_weight_per_turn=1.0,
            final_quality=TutorQualitySnapshot(0, 0.0, 0.0, 0.0, 0.0),
        )
        expect_rejection(
            lambda: serialize_attempt(
                "zero-turn",
                zero_turn,
                "synthetic",
                source_policy=policy,
            ),
            "zero-turn completion",
        )
        expect_rejection(
            lambda: serialize_attempt(
                "nan-reward",
                replace(completion, reward=nan),
                "synthetic",
                source_policy=policy,
            ),
            "NaN reward",
        )
        expect_rejection(
            lambda: serialize_attempt(
                "wrong-reward-mapping",
                replace(completion, reward=1.0),
                "synthetic",
                source_policy=policy,
            ),
            "reward inconsistent with success-only mapping",
        )

        first_completed_turn = completion.turns[0]
        bad_scores = dict(first_completed_turn.decision.sampled_scores)
        bad_scores[ARMS[0]] = inf
        bad_decision = replace(
            first_completed_turn.decision,
            sampled_scores=bad_scores,
        )
        bad_turn = replace(first_completed_turn, decision=bad_decision)
        bad_turn_completion = replace(
            completion,
            turns=(bad_turn, *completion.turns[1:]),
        )
        expect_rejection(
            lambda: serialize_attempt(
                "infinite-score",
                bad_turn_completion,
                "synthetic",
                source_policy=policy,
            ),
            "infinite sampled score",
        )
        expect_rejection(
            lambda: serialize_attempt(
                "nan-metadata",
                completion,
                "synthetic",
                source_policy=policy,
                metadata={"bad": nan},
            ),
            "NaN metadata",
        )
        for forbidden_key in (
            "A",
            "b",
            "model_weights",
            "policy_state",
            "state_dict",
            "tokenizer_contents",
        ):
            expect_rejection(
                lambda forbidden_key=forbidden_key: serialize_attempt(
                    "forbidden-metadata",
                    completion,
                    "synthetic",
                    source_policy=policy,
                    metadata={forbidden_key: [[1.0]]},
                ),
                f"forbidden metadata key {forbidden_key}",
            )

        valid_path = root / "valid.jsonl"
        logger = ExperienceLogger(
            valid_path,
            data_mode="synthetic",
            source_policy=policy,
        )
        logger.append_attempt("valid-attempt", completion)
        valid_contents = valid_path.read_bytes()
        expect_rejection(
            lambda: logger.append_attempt("", completion),
            "invalid append attempt ID",
        )
        assert valid_path.read_bytes() == valid_contents

        malformed_path = root / "malformed.jsonl"
        with malformed_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("{not valid json}\n")
        expect_rejection(
            lambda: ExperienceLogger(
                malformed_path,
                data_mode="synthetic",
                source_policy=policy,
            ),
            "malformed existing JSONL",
        )

        old_schema_path = root / "old-schema.jsonl"
        old_record = serialize_attempt(
            "old-schema",
            completion,
            "synthetic",
            source_policy=policy,
        )
        old_record["schema_version"] = "turn_lints_v1"
        old_schema_path.write_text(
            json.dumps(old_record, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        expect_rejection(
            lambda: ExperienceLogger(
                old_schema_path,
                data_mode="synthetic",
                source_policy=policy,
            ),
            "old experience schema",
        )


def main() -> None:
    checks = (
        (
            "JSON serialization, trajectory preservation, and purity",
            test_json_trajectory_and_observational_purity,
        ),
        ("append-only behavior", test_append_only_behavior),
        ("synthetic/real data-mode barrier", test_data_mode_barrier),
        ("invalid-record rejection", test_invalid_record_rejection),
    )

    print("CANONICAL TURN-LEVEL EXPERIENCE LOGGER VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
