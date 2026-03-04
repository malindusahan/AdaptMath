"""Focused offline-calibration contracts for the MD7 action anchor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_PATH = (
    WORKSPACE_ROOT
    / "pedagogical-move-selection"
    / "results"
    / "turn_lints_selector_anchor_v1"
    / "analysis.py"
)
SPEC = importlib.util.spec_from_file_location("turn_lints_anchor_analysis", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_grouped_calibration_validation_split_has_no_attempt_leakage() -> None:
    contexts = analysis.reconstruct_pre_action_contexts(
        analysis.load_jsonl(analysis.PASSIVE_TURNS)
    )
    first = analysis.grouped_split(contexts)
    second = analysis.grouped_split(contexts)
    calibration, validation, calibration_attempts, validation_attempts = first
    assert first[2:] == second[2:]
    assert len(contexts) == 98
    assert len(calibration) == 73
    assert len(validation) == 25
    assert len(calibration_attempts) == 10
    assert len(validation_attempts) == 5
    assert set(calibration_attempts).isdisjoint(validation_attempts)
    assert {item["attempt_id"] for item in calibration} == set(calibration_attempts)
    assert {item["attempt_id"] for item in validation} == set(validation_attempts)


def test_seeded_score_sampling_and_gamma_zero_are_deterministic(
    monkeypatch,
) -> None:
    contexts = analysis.reconstruct_pre_action_contexts(
        analysis.load_jsonl(analysis.PASSIVE_TURNS)
    )[:3]
    monkeypatch.setattr(analysis, "SEED_COUNT", 4)
    first = analysis.sample_reward_scores(contexts)
    second = analysis.sample_reward_scores(contexts)
    assert np.array_equal(first, second)
    q_zero_first = analysis.empirical_policy_distributions(contexts, first, 0.0)
    q_zero_second = analysis.empirical_policy_distributions(contexts, second, 0.0)
    assert np.array_equal(q_zero_first, q_zero_second)
    assert np.allclose(q_zero_first.sum(axis=1), 1.0)


def test_js_distribution_matching_uses_full_probability_vector() -> None:
    target = np.asarray([[0.30, 0.28, 0.27, 0.15]], dtype=np.float64)
    exact = analysis.js_divergence_rows(target, target)
    top1_clone = analysis.js_divergence_rows(
        target, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    )
    assert exact[0] == 0.0
    assert top1_clone[0] > exact[0]
