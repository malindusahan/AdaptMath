"""CPU-only deterministic validation of conservative MD6 overlay mechanics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from math import inf, nan, nextafter

from src.self_improvement.conservative_overlay import (
    ARMS,
    BIAS_TARGET,
    DEFAULT_GAP_THRESHOLD,
    MOVE_ORDER,
    OverlayDecision,
    apply_conservative_overlay,
)
from src.self_improvement.context_builder import MOVE_ORDER as CONTEXT_MOVE_ORDER
from src.self_improvement.lints_policy import ARMS as LINTS_ARMS


def expect_rejection(action: Callable[[], object], case_name: str) -> None:
    """Assert that invalid overlay input raises a contract exception."""

    try:
        action()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"Invalid overlay case was accepted: {case_name}")


def baseline_probabilities() -> dict[str, float]:
    return {
        "generic": 0.10,
        "probing": 0.20,
        "focus": 0.50,
        "telling": 0.20,
    }


def test_move_contract() -> None:
    expected = ("generic", "probing", "focus", "telling")
    assert isinstance(MOVE_ORDER, tuple)
    assert MOVE_ORDER == expected
    assert MOVE_ORDER is CONTEXT_MOVE_ORDER


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
    assert ARMS is LINTS_ARMS


def test_baseline() -> None:
    decision = apply_conservative_overlay("baseline", baseline_probabilities())
    assert isinstance(decision, OverlayDecision)
    assert decision.arm == "baseline"
    assert decision.base_move == "focus"
    assert decision.final_move == "focus"
    assert decision.target_move is None
    assert decision.overridden is False
    assert decision.gap == 0.0
    assert decision.gap_threshold == DEFAULT_GAP_THRESHOLD
    assert decision.base_probability == 0.50
    assert decision.target_probability is None
    assert type(decision.gap) is float
    assert type(decision.gap_threshold) is float
    assert type(decision.base_probability) is float


def test_close_override() -> None:
    decision = apply_conservative_overlay(
        "probing_bias",
        {
            "generic": 0.18,
            "probing": 0.30,
            "focus": 0.36,
            "telling": 0.16,
        },
    )
    assert decision.base_move == "focus"
    assert decision.target_move == "probing"
    assert decision.final_move == "probing"
    assert decision.overridden is True
    assert abs(decision.gap - 0.06) < 1e-15
    assert type(decision.target_probability) is float


def test_far_target_block() -> None:
    decision = apply_conservative_overlay(
        "probing_bias",
        {
            "generic": 0.10,
            "probing": 0.15,
            "focus": 0.67,
            "telling": 0.08,
        },
    )
    assert decision.base_move == "focus"
    assert decision.target_move == "probing"
    assert decision.final_move == "focus"
    assert decision.overridden is False
    assert abs(decision.gap - 0.52) < 1e-15


def test_exact_threshold_boundary() -> None:
    probabilities = {
        "generic": 0.24,
        "probing": 0.21,
        "focus": 0.31,
        "telling": 0.24,
    }
    assert probabilities["focus"] - probabilities["probing"] == 0.10

    allowed = apply_conservative_overlay(
        "probing_bias",
        probabilities,
        gap_threshold=0.10,
    )
    assert allowed.gap == 0.10
    assert allowed.final_move == "probing"
    assert allowed.overridden is True

    slightly_lower = nextafter(allowed.gap, 0.0)
    assert slightly_lower < allowed.gap
    blocked = apply_conservative_overlay(
        "probing_bias",
        probabilities,
        gap_threshold=slightly_lower,
    )
    assert blocked.final_move == "focus"
    assert blocked.overridden is False


def test_target_already_base() -> None:
    decision = apply_conservative_overlay(
        "focus_bias",
        baseline_probabilities(),
    )
    assert decision.base_move == "focus"
    assert decision.target_move == "focus"
    assert decision.final_move == "focus"
    assert decision.overridden is False
    assert decision.gap == 0.0
    assert decision.base_probability == decision.target_probability


def test_all_bias_mappings() -> None:
    expected = {
        "baseline": None,
        "generic_bias": "generic",
        "probing_bias": "probing",
        "focus_bias": "focus",
        "telling_bias": "telling",
    }
    assert dict(BIAS_TARGET) == expected

    uniform = {move: 0.25 for move in MOVE_ORDER}
    for arm, target_move in tuple(expected.items())[1:]:
        decision = apply_conservative_overlay(arm, uniform)
        assert decision.target_move == target_move
        assert decision.final_move == target_move
        if target_move == "generic":
            assert decision.overridden is False
        else:
            assert decision.overridden is True


def test_tie_breaking() -> None:
    scrambled = {
        "telling": 0.10,
        "probing": 0.40,
        "generic": 0.40,
        "focus": 0.10,
    }
    decision = apply_conservative_overlay("baseline", scrambled)
    assert decision.base_move == "generic"
    assert decision.final_move == "generic"


def test_bad_arm() -> None:
    for arm in ("probing", "unknown_bias", "Baseline", "", None, True):
        expect_rejection(
            lambda arm=arm: apply_conservative_overlay(
                arm,  # type: ignore[arg-type]
                baseline_probabilities(),
            ),
            repr(arm),
        )


def test_probability_validation() -> None:
    invalid_cases: dict[str, object] = {
        "missing move": {
            "generic": 0.20,
            "probing": 0.30,
            "focus": 0.50,
        },
        "extra move": {
            **baseline_probabilities(),
            "other": 0.0,
        },
        "negative": {
            "generic": -0.10,
            "probing": 0.30,
            "focus": 0.50,
            "telling": 0.30,
        },
        "above one": {
            "generic": 1.10,
            "probing": 0.0,
            "focus": 0.0,
            "telling": -0.10,
        },
        "non-unit sum": {
            "generic": 0.10,
            "probing": 0.10,
            "focus": 0.10,
            "telling": 0.10,
        },
        "NaN": {
            "generic": nan,
            "probing": 0.20,
            "focus": 0.50,
            "telling": 0.30,
        },
        "positive infinity": {
            "generic": inf,
            "probing": 0.0,
            "focus": 0.0,
            "telling": 0.0,
        },
        "negative infinity": {
            "generic": -inf,
            "probing": 0.0,
            "focus": 1.0,
            "telling": 0.0,
        },
        "boolean true": {
            "generic": True,
            "probing": 0.0,
            "focus": 0.0,
            "telling": 0.0,
        },
        "boolean false": {
            "generic": False,
            "probing": 0.20,
            "focus": 0.50,
            "telling": 0.30,
        },
        "numeric string": {
            "generic": "0.10",
            "probing": 0.20,
            "focus": 0.50,
            "telling": 0.20,
        },
        "not a mapping": [0.10, 0.20, 0.50, 0.20],
    }
    for name, probabilities in invalid_cases.items():
        expect_rejection(
            lambda probabilities=probabilities: apply_conservative_overlay(
                "baseline",
                probabilities,  # type: ignore[arg-type]
            ),
            name,
        )


def test_threshold_validation() -> None:
    for threshold in (0, 0.10, 1):
        decision = apply_conservative_overlay(
            "baseline",
            baseline_probabilities(),
            gap_threshold=threshold,
        )
        assert type(decision.gap_threshold) is float
        assert decision.gap_threshold == float(threshold)

    for threshold in (-0.01, 1.01, nan, inf, -inf, True, False, "0.10", None):
        expect_rejection(
            lambda threshold=threshold: apply_conservative_overlay(
                "baseline",
                baseline_probabilities(),
                gap_threshold=threshold,  # type: ignore[arg-type]
            ),
            repr(threshold),
        )


def test_deterministic_purity() -> None:
    probabilities = {
        "generic": 0.18,
        "probing": 0.30,
        "focus": 0.36,
        "telling": 0.16,
    }
    snapshot = probabilities.copy()
    decisions = tuple(
        apply_conservative_overlay("probing_bias", probabilities)
        for _ in range(5)
    )
    assert all(decision == decisions[0] for decision in decisions)
    assert probabilities == snapshot

    try:
        decisions[0].final_move = "telling"  # type: ignore[misc]
    except (FrozenInstanceError, AttributeError):
        pass
    else:
        raise AssertionError("OverlayDecision is not immutable.")


def main() -> None:
    checks = (
        ("move contract", test_move_contract),
        ("arm contract", test_arm_contract),
        ("baseline", test_baseline),
        ("close override", test_close_override),
        ("far-target block", test_far_target_block),
        ("exact threshold boundary", test_exact_threshold_boundary),
        ("target already base", test_target_already_base),
        ("all bias mappings", test_all_bias_mappings),
        ("tie breaking", test_tie_breaking),
        ("bad arm", test_bad_arm),
        ("probability validation", test_probability_validation),
        ("threshold validation", test_threshold_validation),
        ("deterministic purity", test_deterministic_purity),
    )

    print("CANONICAL CONSERVATIVE MD6 OVERLAY VALIDATION")
    for name, check in checks:
        check()
        print(f"- {name}: PASS")
    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
