"""Deterministic, dependency-free validation of the SI7 reward contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from math import inf, nan

from src.self_improvement.reward import primary_reward


def expect_rejection(
    action: Callable[[], object],
    case_name: str,
    expected_exception: type[Exception],
) -> None:
    """Assert that invalid reward input raises a clear contract exception."""

    try:
        action()
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"{case_name} raised {type(exc).__name__}, expected "
            f"{expected_exception.__name__}."
        ) from exc
    raise AssertionError(f"Invalid evaluator score was accepted: {case_name}")


def test_exact_mapping() -> None:
    expected = (
        (0, 0.0),
        (1, 0.0),
        (2, 0.0),
        (3, 1.0),
    )
    for evaluator_score, expected_reward in expected:
        assert primary_reward(evaluator_score) == expected_reward


def test_output_type() -> None:
    for evaluator_score in range(4):
        assert type(primary_reward(evaluator_score)) is float


def test_output_range() -> None:
    for evaluator_score in range(4):
        reward = primary_reward(evaluator_score)
        assert 0.0 <= reward <= 1.0


def test_invalid_integer_values() -> None:
    for value in (-1, 4, 100):
        expect_rejection(
            lambda value=value: primary_reward(value),
            repr(value),
            ValueError,
        )


def test_non_integer_values() -> None:
    for value in (1.5, 2.0, "2", None):
        expect_rejection(
            lambda value=value: primary_reward(value),
            repr(value),
            TypeError,
        )


def test_boolean_values() -> None:
    for value in (True, False):
        expect_rejection(
            lambda value=value: primary_reward(value),
            repr(value),
            TypeError,
        )


def test_non_finite_values() -> None:
    for value in (nan, inf, -inf):
        expect_rejection(
            lambda value=value: primary_reward(value),
            repr(value),
            TypeError,
        )


def test_function_signature() -> None:
    signature = inspect.signature(primary_reward)
    assert tuple(signature.parameters) == ("evaluator_score",)
    parameter = signature.parameters["evaluator_score"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty

    forbidden = {
        "mastery_before",
        "mastery_after",
        "mastery_delta",
        "mrb1_scores",
        "md6_probabilities",
        "context",
        "student_response",
        "future_student_response",
    }
    assert set(signature.parameters).isdisjoint(forbidden)

    expect_rejection(
        lambda: primary_reward(evaluator_score=2, mastery_after=0.8),
        "forbidden mastery_after keyword",
        TypeError,
    )


def main() -> None:
    checks = (
        ("exact mapping", test_exact_mapping),
        ("output type", test_output_type),
        ("output range", test_output_range),
        ("invalid integer values", test_invalid_integer_values),
        ("non-integer values", test_non_integer_values),
        ("boolean rejection", test_boolean_values),
        ("non-finite rejection", test_non_finite_values),
        ("signature guard", test_function_signature),
    )

    print("SI7 PRIMARY REWARD VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
