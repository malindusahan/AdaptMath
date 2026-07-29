"""API-free local smoke check for the student-model -> frozen-v3 bridge.

This script intentionally reuses the deterministic helpers from the bridge
test suite. All database, experience-log, and policy-state writes are confined
to a temporary directory and the policy remains in synthetic data mode.
"""

from __future__ import annotations

import gc
import math
import os
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest


REPOSITORY_A_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_B_ROOT = (
    REPOSITORY_A_ROOT.parent / "student-modeling-clean"
).resolve()

os.environ["STUDENT_MODEL_REPO"] = str(REPOSITORY_B_ROOT)
sys.path.insert(0, str(REPOSITORY_A_ROOT))
sys.path.insert(0, str(TESTS_ROOT))


from test_student_model_v3_bridge import (  # noqa: E402
    BKTPredictor,
    ColdStartPriorCalculator,
    StudentModelFrozenV3Bridge,
    adaptive_memory,
    build_adaptive_pipeline,
    build_real_student_model,
    record_stub_turns,
)


STUDENT_ID = "local-integration-student-001"
ATTEMPT_ID = "local-integration-attempt-001"
SECOND_ATTEMPT_ID = "local-integration-attempt-002"
TARGET_SKILL = "Percent Of"
TOLERANCE = 1e-9


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)


def main() -> None:
    if not (REPOSITORY_B_ROOT / "bkt" / "predict.py").is_file():
        raise RuntimeError(f"Incomplete Repository B: {REPOSITORY_B_ROOT}")

    protected_datasets = {
        (
            REPOSITORY_A_ROOT
            / "data"
            / "raw"
            / "mathdial"
            / "test.jsonl"
        ).resolve(),
        (
            REPOSITORY_A_ROOT
            / "data"
            / "processed"
            / "mathdial"
            / "test.jsonl"
        ).resolve(),
        (
            REPOSITORY_A_ROOT
            / "data"
            / "external"
            / "mrbench"
            / "mrbench_v3_testset.json"
        ).resolve(),
    }
    original_path_open = Path.open
    accessed_protected_datasets: list[Path] = []

    def guarded_path_open(path: Path, *args, **kwargs):
        resolved = path.resolve()
        if resolved in protected_datasets:
            accessed_protected_datasets.append(resolved)
            raise AssertionError(f"Held-out dataset access forbidden: {resolved}")
        return original_path_open(path, *args, **kwargs)

    with tempfile.TemporaryDirectory(
        prefix="student-model-v3-local-smoke-"
    ) as temporary_directory, pytest.MonkeyPatch.context() as monkeypatch:
        temporary_root = Path(temporary_directory)
        predictor = BKTPredictor.load()
        if not predictor.has_skill(TARGET_SKILL):
            raise AssertionError(f"Unsupported target skill: {TARGET_SKILL}")

        student_pipeline, _graph, _evaluator, _db_path = (
            build_real_student_model(
                monkeypatch,
                temporary_root / "student",
                target_skill=TARGET_SKILL,
                events=((0, TARGET_SKILL),),
                verdicts={TARGET_SKILL: "correct"},
                predictor=predictor,
                cold_start=ColdStartPriorCalculator(),
            )
        )
        adaptive_pipeline, policy, _log_path, _state_path = (
            build_adaptive_pipeline(temporary_root / "adaptive")
        )
        bridge = StudentModelFrozenV3Bridge(
            student_model_pipeline=student_pipeline,
            adaptive_pipeline=adaptive_pipeline,
        )

        first_memory = adaptive_memory(ATTEMPT_ID)
        with (
            patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("Network access forbidden"),
            ) as socket_call,
            patch.object(
                urllib.request,
                "urlopen",
                side_effect=AssertionError("Network access forbidden"),
            ) as urlopen_call,
            patch.object(Path, "open", guarded_path_open),
        ):
            started = bridge.start_attempt(
                student_id=STUDENT_ID,
                attempt_id=ATTEMPT_ID,
                target_skill=TARGET_SKILL,
                adaptive_memory=first_memory,
            )
            if policy.total_updates != 0:
                raise AssertionError("Policy updated at attempt start")

            decisions = record_stub_turns(adaptive_pipeline, 3)
            if len(decisions) != 3:
                raise AssertionError("The smoke trajectory must contain 3 turns")
            updates_before_finish = policy.total_updates
            if updates_before_finish != 0:
                raise AssertionError("Policy updated before completion")

            student_result = student_pipeline.process_transcript(
                transcript=[
                    {
                        "role": "student",
                        "text": "The answer is 25 percent.",
                    }
                ],
                student_id=STUDENT_ID,
                session_id="local-integration-session-001",
                attempt_context=bridge.student_model_context,
            )
            outcome = student_result["learning_outcome"]
            mastery_before = float(outcome["mastery_before"])
            mastery_after = float(outcome["mastery_after"])
            delta_mastery = float(outcome["delta_mastery"])
            if started.attempt_id != ATTEMPT_ID:
                raise AssertionError("Attempt start changed the attempt ID")
            if started.target_skill != TARGET_SKILL:
                raise AssertionError("Attempt start changed the target skill")
            if not _close(float(started.mastery_before), mastery_before):
                raise AssertionError(
                    "Completion mastery_before differs from the start snapshot"
                )
            if not _close(delta_mastery, mastery_after - mastery_before):
                raise AssertionError("Repository B outcome arithmetic failed")

            completion = bridge.finish_attempt(
                adaptive_memory=first_memory,
                student_model_result=student_result,
            )
            reward = float(completion["reward"])
            updates_after_finish = policy.total_updates
            weights = [
                float(update["sample_weight"])
                for update in policy.observed_updates
            ]
            total_weight = sum(weights)

            if outcome["attempt_id"] != ATTEMPT_ID:
                raise AssertionError("Repository B changed the attempt ID")
            if outcome["skill"] != TARGET_SKILL:
                raise AssertionError("Repository B changed the target skill")
            if completion["attempt_id"] != ATTEMPT_ID:
                raise AssertionError("Repository A changed the attempt ID")
            if completion["skill"] != TARGET_SKILL:
                raise AssertionError("Repository A changed the target skill")
            if not _close(reward, delta_mastery):
                raise AssertionError("Frozen reward is not raw delta mastery")
            if updates_after_finish != 3:
                raise AssertionError("Expected one delayed update per saved turn")
            if len(weights) != 3 or not all(
                _close(weight, 1.0 / 3.0) for weight in weights
            ):
                raise AssertionError("Per-turn delayed credit is not 1/3")
            if not _close(total_weight, 1.0):
                raise AssertionError("Total attempt credit is not 1")

            second_memory = adaptive_memory(SECOND_ATTEMPT_ID)
            second_started = bridge.start_attempt(
                student_id=STUDENT_ID,
                attempt_id=SECOND_ATTEMPT_ID,
                target_skill=TARGET_SKILL,
                adaptive_memory=second_memory,
            )
            continuity_pass = _close(
                mastery_after,
                float(second_started.mastery_before),
            )
            if not continuity_pass:
                raise AssertionError("Sequential BKT continuity failed")

            socket_call.assert_not_called()
            urlopen_call.assert_not_called()

        if accessed_protected_datasets:
            raise AssertionError(
                f"Held-out datasets accessed: {accessed_protected_datasets}"
            )

        print(f"student_id={STUDENT_ID}")
        print(f"attempt_id={outcome['attempt_id']}")
        print(f"skill={outcome['skill']}")
        print(f"mastery_before={mastery_before:.17g}")
        print(f"mastery_after={mastery_after:.17g}")
        print(f"delta_mastery={delta_mastery:.17g}")
        print(f"repo_a_reward={reward:.17g}")
        print(f"T={len(decisions)}")
        print(f"policy_updates_before_finish={updates_before_finish}")
        print(f"policy_updates_after_finish={updates_after_finish}")
        print(f"per_turn_weights={weights}")
        print(f"total_weight={total_weight:.17g}")
        print(f"attempt_1_mastery_after={mastery_after:.17g}")
        print(
            "attempt_2_mastery_before="
            f"{float(second_started.mastery_before):.17g}"
        )
        print(f"sequential_continuity={'PASS' if continuity_pass else 'FAIL'}")
        print("network_calls=0")
        print("held_out_dataset_accesses=0")
        print("data_mode=synthetic")
        print("SMOKE_RESULT=PASS")
        # The recovered database initializer uses sqlite's transaction context
        # manager, whose final handle release may be deferred until collection.
        # Collect here so Windows can remove the temporary SQLite file cleanly.
        gc.collect()


if __name__ == "__main__":
    main()
