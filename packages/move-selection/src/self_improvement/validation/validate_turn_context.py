"""CPU-only deterministic validation of the candidate turn-level context."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from math import inf, nan

import numpy as np

from src.self_improvement.turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
    RunningTutorQuality,
    TutorQualitySnapshot,
    build_turn_context,
)


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    """Assert that invalid context input raises without state mutation."""

    try:
        action()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"Invalid turn-context case was accepted: {case_name}")


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


def test_feature_order() -> None:
    expected = (
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
    assert isinstance(TURN_FEATURE_NAMES, tuple)
    assert TURN_FEATURE_NAMES == expected
    assert len(TURN_FEATURE_NAMES) == 9
    assert "mastery_before" not in TURN_FEATURE_NAMES
    assert MRB1_TASKS == (
        "Mistake_Identification",
        "Mistake_Location",
        "Providing_Guidance",
        "Actionability",
    )


def test_turn_one_cold_state() -> None:
    quality = RunningTutorQuality()
    context = build_turn_context(md6_a(), quality)
    expected = np.asarray(
        [0.10, 0.34, 0.40, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(context, expected)
    assert quality.count == 0
    assert quality.snapshot() == TutorQualitySnapshot(0, 0.0, 0.0, 0.0, 0.0)


def test_one_quality_observation() -> None:
    quality = RunningTutorQuality()
    snapshot = quality.add_scores(scores_one())
    assert snapshot.count == 1
    assert snapshot.as_task_mapping() == scores_one()

    context = build_turn_context(md6_a(), quality)
    np.testing.assert_array_equal(
        context[4:8],
        np.asarray([0.90, 0.80, 0.70, 0.60], dtype=np.float64),
    )
    assert context[8] == 1.0


def test_running_arithmetic_means() -> None:
    quality = RunningTutorQuality()
    quality.add_scores(scores_one())
    snapshot = quality.add_scores(scores_two())
    expected_means = np.asarray([0.80, 0.80, 0.80, 0.70], dtype=np.float64)
    assert snapshot.count == 2
    np.testing.assert_allclose(
        snapshot.feature_values,
        expected_means,
        rtol=0.0,
        atol=1e-15,
    )

    context = build_turn_context(md6_a(), snapshot)
    np.testing.assert_allclose(
        context[4:8],
        expected_means,
        rtol=0.0,
        atol=1e-15,
    )
    assert context[8] == 1.0


def test_current_md6_changes_and_quality_stays_fixed() -> None:
    quality = RunningTutorQuality()
    quality.add_scores(scores_one())
    context_a = build_turn_context(md6_a(), quality)
    context_b = build_turn_context(md6_b(), quality)

    np.testing.assert_array_equal(
        context_a[:4],
        np.asarray([0.10, 0.34, 0.40, 0.16]),
    )
    np.testing.assert_array_equal(
        context_b[:4],
        np.asarray([0.36, 0.10, 0.34, 0.20]),
    )
    assert not np.array_equal(context_a[:4], context_b[:4])
    np.testing.assert_array_equal(context_a[4:], context_b[4:])


def test_bad_md6() -> None:
    invalid_cases: dict[str, object] = {
        "missing": {"generic": 0.20, "probing": 0.30, "focus": 0.50},
        "extra": {**md6_a(), "other": 0.0},
        "negative": {
            "generic": -0.10,
            "probing": 0.30,
            "focus": 0.50,
            "telling": 0.30,
        },
        "non-unit": {
            "generic": 0.10,
            "probing": 0.10,
            "focus": 0.10,
            "telling": 0.10,
        },
        "NaN": {**md6_a(), "generic": nan},
        "infinity": {**md6_a(), "generic": inf},
        "boolean": {
            "generic": True,
            "probing": 0.0,
            "focus": 0.0,
            "telling": 0.0,
        },
        "numeric string": {**md6_a(), "generic": "0.10"},
        "not mapping": [0.10, 0.34, 0.40, 0.16],
    }
    for name, probabilities in invalid_cases.items():
        expect_rejection(
            lambda probabilities=probabilities: build_turn_context(
                probabilities,  # type: ignore[arg-type]
                RunningTutorQuality(),
            ),
            name,
        )


def test_bad_mrb1_tasks_and_scores() -> None:
    invalid_cases: dict[str, object] = {
        "missing task": {
            "Mistake_Identification": 0.9,
            "Mistake_Location": 0.8,
            "Providing_Guidance": 0.7,
        },
        "extra task": {**scores_one(), "Other": 0.5},
        "wrong task case": {
            **scores_one(),
            "mistake_identification": 0.9,
        },
        "negative": {**scores_one(), "Actionability": -0.1},
        "above one": {**scores_one(), "Actionability": 1.1},
        "NaN": {**scores_one(), "Actionability": nan},
        "infinity": {**scores_one(), "Actionability": inf},
        "boolean": {**scores_one(), "Actionability": True},
        "numeric string": {**scores_one(), "Actionability": "0.60"},
        "not mapping": [0.9, 0.8, 0.7, 0.6],
    }
    for name, scores in invalid_cases.items():
        quality = RunningTutorQuality()
        before = quality.snapshot()
        expect_rejection(
            lambda scores=scores: quality.add_scores(  # type: ignore[arg-type]
                scores
            ),
            name,
        )
        assert quality.snapshot() == before


def test_output_contract() -> None:
    quality = RunningTutorQuality()
    quality.add_scores(scores_one())
    context = build_turn_context(md6_a(), quality)
    assert isinstance(context, np.ndarray)
    assert context.shape == (9,)
    assert context.dtype == np.float64
    assert np.isfinite(context).all()


def test_leakage_signature_guard() -> None:
    signature = inspect.signature(build_turn_context)
    assert tuple(signature.parameters) == (
        "md6_probabilities",
        "running_quality",
    )
    for parameter in signature.parameters.values():
        assert parameter.kind is not inspect.Parameter.VAR_KEYWORD

    forbidden = {
        "current_mrb1_scores",
        "evaluator_score",
        "reward",
        "mastery_before",
        "mastery_after",
        "student_response",
        "future_student_response",
    }
    assert set(signature.parameters).isdisjoint(forbidden)

    add_signature = inspect.signature(RunningTutorQuality.add_scores)
    assert tuple(add_signature.parameters) == ("self", "scores")


def main() -> None:
    checks = (
        ("feature order", test_feature_order),
        ("Turn 1 cold state", test_turn_one_cold_state),
        ("one prior quality observation", test_one_quality_observation),
        ("running arithmetic means", test_running_arithmetic_means),
        (
            "changing current MD6 and fixed running quality",
            test_current_md6_changes_and_quality_stays_fixed,
        ),
        ("MD6 validation", test_bad_md6),
        ("MRB1 task/score validation", test_bad_mrb1_tasks_and_scores),
        ("output shape/dtype/finiteness", test_output_contract),
        ("temporal leakage signature guard", test_leakage_signature_guard),
    )

    print("CANONICAL TURN-LEVEL CONTEXT VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
