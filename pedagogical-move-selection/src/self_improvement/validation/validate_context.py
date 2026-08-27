"""Deterministic, dataset-free validation for the SI7-A context contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import MappingProxyType

import numpy as np

from src.self_improvement.context_builder import (
    FEATURE_NAMES,
    PreviousAttemptFeedback,
    build_lints_context,
    mrb1_expected_score,
)


EXPECTED_FEATURE_NAMES = (
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "mastery_before",
    "prev_evaluator_rate",
    "prev_mastery_delta",
    "prev_mistake_identification",
    "prev_mistake_location",
    "prev_providing_guidance",
    "prev_actionability",
    "has_previous_feedback",
)

VALID_MD6 = MappingProxyType(
    {
        "generic": 0.12,
        "probing": 0.25,
        "focus": 0.51,
        "telling": 0.12,
    }
)


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    """Assert that invalid synthetic input raises a validation exception."""

    try:
        action()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"Invalid case was accepted: {case_name}")


def test_feature_order() -> None:
    assert isinstance(FEATURE_NAMES, tuple)
    assert len(FEATURE_NAMES) == 12
    assert FEATURE_NAMES == EXPECTED_FEATURE_NAMES


def test_cold_start() -> None:
    shuffled_md6 = {
        "telling": 0.12,
        "focus": 0.51,
        "generic": 0.12,
        "probing": 0.25,
    }
    context = build_lints_context(shuffled_md6, mastery_before=0.42)
    np.testing.assert_allclose(
        context[:4],
        np.asarray([0.12, 0.25, 0.51, 0.12], dtype=np.float64),
    )
    assert context[4] == 0.42
    np.testing.assert_array_equal(context[5:], np.zeros(7, dtype=np.float64))


def test_previous_feedback() -> None:
    feedback = PreviousAttemptFeedback(
        evaluator_score=2,
        mastery_before=0.42,
        mastery_after=0.55,
        mistake_identification=0.84,
        mistake_location=0.73,
        providing_guidance=0.79,
        actionability=0.68,
    )
    context = build_lints_context(
        VALID_MD6,
        mastery_before=0.55,
        previous_feedback=feedback,
    )

    np.testing.assert_allclose(
        context[5:],
        np.asarray(
            [2.0 / 3.0, 0.13, 0.84, 0.73, 0.79, 0.68, 1.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )

    positive_endpoint = PreviousAttemptFeedback(
        evaluator_score=3,
        mastery_before=0.0,
        mastery_after=1.0,
        mistake_identification=0.0,
        mistake_location=1.0,
        providing_guidance=0.0,
        actionability=1.0,
    )
    negative_endpoint = PreviousAttemptFeedback(
        evaluator_score=0,
        mastery_before=1.0,
        mastery_after=0.0,
        mistake_identification=1.0,
        mistake_location=0.0,
        providing_guidance=1.0,
        actionability=0.0,
    )
    assert build_lints_context(VALID_MD6, 0.0, positive_endpoint)[6] == 1.0
    assert build_lints_context(VALID_MD6, 1.0, negative_endpoint)[6] == -1.0


def test_mrb1_expected_score() -> None:
    assert mrb1_expected_score(
        {"No": 1.0, "To some extent": 0.0, "Yes": 0.0}
    ) == 0.0
    assert mrb1_expected_score(
        {"No": 0.0, "To some extent": 1.0, "Yes": 0.0}
    ) == 0.5
    assert mrb1_expected_score(
        {"No": 0.0, "To some extent": 0.0, "Yes": 1.0}
    ) == 1.0
    assert np.isclose(
        mrb1_expected_score(
            {"Yes": 0.50, "No": 0.20, "To some extent": 0.30}
        ),
        0.65,
    )


def test_bad_md6_input() -> None:
    cases = {
        "missing move": {"generic": 0.2, "probing": 0.3, "focus": 0.5},
        "extra move": {**VALID_MD6, "other": 0.0},
        "negative probability": {
            "generic": -0.1,
            "probing": 0.4,
            "focus": 0.4,
            "telling": 0.3,
        },
        "non-unit sum": {
            "generic": 0.1,
            "probing": 0.1,
            "focus": 0.1,
            "telling": 0.1,
        },
        "probability above one": {
            "generic": 1.000001,
            "probing": 0.0,
            "focus": 0.0,
            "telling": 0.0,
        },
        "NaN": {**VALID_MD6, "generic": np.nan},
        "infinity": {**VALID_MD6, "generic": np.inf},
    }
    for name, probabilities in cases.items():
        expect_rejection(
            lambda probabilities=probabilities: build_lints_context(
                probabilities,
                mastery_before=0.42,
            ),
            name,
        )


def test_bad_mastery() -> None:
    assert build_lints_context(VALID_MD6, mastery_before=0.0)[4] == 0.0
    assert build_lints_context(VALID_MD6, mastery_before=1.0)[4] == 1.0

    for value in (-0.01, 1.01, np.nan, np.inf, -np.inf):
        expect_rejection(
            lambda value=value: build_lints_context(
                VALID_MD6,
                mastery_before=value,
            ),
            f"current mastery {value!r}",
        )

    for field_name in ("mastery_before", "mastery_after"):
        for value in (-0.01, 1.01, np.nan, np.inf):
            values = {
                "evaluator_score": 2,
                "mastery_before": 0.42,
                "mastery_after": 0.55,
                "mistake_identification": 0.8,
                "mistake_location": 0.7,
                "providing_guidance": 0.9,
                "actionability": 0.6,
            }
            values[field_name] = value
            expect_rejection(
                lambda values=values: PreviousAttemptFeedback(**values),
                f"previous {field_name} {value!r}",
            )


def test_bad_previous_evaluator() -> None:
    for value in (-1, 4, 1.5, 2.0, "2", True, False, np.nan, np.inf):
        expect_rejection(
            lambda value=value: PreviousAttemptFeedback(
                evaluator_score=value,
                mastery_before=0.42,
                mastery_after=0.55,
                mistake_identification=0.8,
                mistake_location=0.7,
                providing_guidance=0.9,
                actionability=0.6,
            ),
            f"previous evaluator {value!r}",
        )


def test_bad_previous_mrb1_scores() -> None:
    fields = (
        "mistake_identification",
        "mistake_location",
        "providing_guidance",
        "actionability",
    )
    for field_name in fields:
        for value in (-0.01, 1.01, np.nan, np.inf):
            values = {
                "evaluator_score": 2,
                "mastery_before": 0.42,
                "mastery_after": 0.55,
                "mistake_identification": 0.8,
                "mistake_location": 0.7,
                "providing_guidance": 0.9,
                "actionability": 0.6,
            }
            values[field_name] = value
            expect_rejection(
                lambda values=values: PreviousAttemptFeedback(**values),
                f"previous {field_name} {value!r}",
            )


def test_bad_mrb1_probability_dictionary() -> None:
    valid = {"No": 0.20, "To some extent": 0.30, "Yes": 0.50}
    cases = {
        "missing label": {"No": 0.4, "Yes": 0.6},
        "extra label": {**valid, "Unknown": 0.0},
        "negative probability": {
            "No": -0.1,
            "To some extent": 0.5,
            "Yes": 0.6,
        },
        "non-unit sum": {"No": 0.1, "To some extent": 0.1, "Yes": 0.1},
        "NaN": {**valid, "No": np.nan},
        "infinity": {**valid, "No": np.inf},
    }
    for name, probabilities in cases.items():
        expect_rejection(
            lambda probabilities=probabilities: mrb1_expected_score(
                probabilities
            ),
            name,
        )


def test_temporal_leakage_api_guard() -> None:
    signature = inspect.signature(build_lints_context)
    assert tuple(signature.parameters) == (
        "md6_probabilities",
        "mastery_before",
        "previous_feedback",
    )
    accepted = set(signature.parameters)
    assert all(
        parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in signature.parameters.values()
    )

    forbidden = {
        "evaluator_score",
        "current_evaluator_score",
        "mastery_after",
        "current_mastery_after",
        "current_mrb1_scores",
        "student_response",
        "future_student_response",
    }
    assert accepted.isdisjoint(forbidden)
    expect_rejection(
        lambda: build_lints_context(
            VALID_MD6,
            mastery_before=0.42,
            evaluator_score=3,
        ),
        "forbidden current evaluator keyword",
    )


def test_output_contract() -> None:
    context = build_lints_context(VALID_MD6, mastery_before=0.42)
    assert isinstance(context, np.ndarray)
    assert context.shape == (12,)
    assert context.dtype == np.float64
    assert np.isfinite(context).all()


def main() -> None:
    checks = (
        ("feature order", test_feature_order),
        ("cold start", test_cold_start),
        ("previous feedback", test_previous_feedback),
        ("MRB1 transformation", test_mrb1_expected_score),
        ("MD6 validation", test_bad_md6_input),
        ("mastery validation", test_bad_mastery),
        ("previous evaluator validation", test_bad_previous_evaluator),
        ("previous MRB1 validation", test_bad_previous_mrb1_scores),
        ("MRB1 probability validation", test_bad_mrb1_probability_dictionary),
        ("temporal leakage guard", test_temporal_leakage_api_guard),
        ("output dtype/shape", test_output_contract),
    )

    print("SI7-A CONTEXT VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
