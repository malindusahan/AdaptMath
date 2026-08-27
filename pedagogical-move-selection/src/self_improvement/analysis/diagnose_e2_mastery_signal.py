"""Neutral counterfactual diagnostic of the existing E2_MASTERY signal.

This module imports the already-defined context_ablation_v1 environment,
coefficient generator, world generator, hidden utility, and canonical arm mask.
It does not run LinTS, alter the environment, or modify the base experiment.

Sanity-only execution::

    python -m src.self_improvement.analysis.diagnose_e2_mastery_signal --sanity-only

Full diagnostic::

    python -m src.self_improvement.analysis.diagnose_e2_mastery_signal --run-full
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import io
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.experiments import context_ablation as base
from src.self_improvement.lints_policy import ARMS
from src.self_improvement.turn_context_builder import RunningTutorQuality


DIAGNOSTIC_NAME: Final[str] = "e2_mastery_signal_diagnostic_v1"
BASE_EXPERIMENT: Final[str] = "context_ablation_v1"
ENVIRONMENT: Final[str] = "E2_MASTERY"
ENVIRONMENT_INDEX: Final[int] = base.ENVIRONMENTS.index(ENVIRONMENT)
SEEDS: Final[tuple[int, ...]] = base.SEEDS
STATES_PER_SEED: Final[int] = 200
MASTERY_GRID: Final[np.ndarray] = np.linspace(0.0, 1.0, 21, dtype=np.float64)
GAP_THRESHOLD: Final[float] = base.GAP_THRESHOLD
BOOTSTRAP_SEED: Final[int] = 620241
BOOTSTRAP_RESAMPLES: Final[int] = 5_000
NUMERIC_TOLERANCE: Final[float] = 1e-12
REPRESENTATIVE_SEEDS: Final[tuple[int, ...]] = (0, 10, 20, 30, 40)

UTILITY_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"utility_{arm}" for arm in ARMS
)
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "seed",
    "state_index",
    "mastery",
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "eligible_arm_count",
    "eligible_arms",
    "oracle_arm",
    "mastery_blind_arm",
    "oracle_utility",
    "mastery_blind_utility",
    "mastery_information_advantage",
    "oracle_margin",
    *UTILITY_COLUMNS,
)
STATE_COLUMNS: Final[tuple[str, ...]] = (
    "seed",
    "state_index",
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "eligible_arm_count",
    "eligible_arms",
    "optimal_arm_changes_over_sweep",
    "num_optimal_arm_transitions",
    "num_distinct_optimal_arms",
    "arm_at_mastery_020",
    "arm_at_mastery_080",
    "different_020_vs_080",
    "max_mastery_utility_range",
    "mean_mastery_information_advantage",
    "max_mastery_information_advantage",
)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "section",
    "metric",
    "subgroup",
    "n",
    "n_seeds",
    "mean",
    "sd",
    "median",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "p25",
    "p75",
    "p90",
    "p95",
    "maximum",
    "unit",
    "notes",
)


@dataclass(slots=True)
class DiagnosticCore:
    raw_sweep: pd.DataFrame
    state_level: pd.DataFrame
    arm_ranges: pd.DataFrame


def _md6_mapping(probabilities: np.ndarray) -> dict[str, float]:
    return base._md6_mapping(probabilities)


def _first_200_base_states(seed: int) -> np.ndarray:
    """Return the first 200 E2 MD6 draws from the unchanged world stream."""

    attempts_needed = math.ceil(STATES_PER_SEED / base.TURNS_PER_ATTEMPT)
    world = base.generate_exogenous_world(
        ENVIRONMENT_INDEX,
        seed,
        attempts=attempts_needed,
    )
    flattened = world.md6_probabilities.reshape(-1, len(base.MOVE_ORDER))
    states = flattened[:STATES_PER_SEED].copy()
    if states.shape != (STATES_PER_SEED, len(base.MOVE_ORDER)):
        raise RuntimeError("The unchanged E2 generator did not yield 200 states.")
    return states


def build_diagnostic_core() -> DiagnosticCore:
    """Evaluate every prescribed counterfactual without invoking LinTS."""

    raw_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []
    quality_placeholder = RunningTutorQuality().snapshot()

    for seed in SEEDS:
        hidden = base.generate_hidden_environment(
            ENVIRONMENT,
            ENVIRONMENT_INDEX,
            seed,
        )
        md6_states = _first_200_base_states(seed)
        for state_zero, md6_array in enumerate(md6_states):
            state_index = state_zero + 1
            md6 = _md6_mapping(md6_array)
            candidates = eligible_arms(md6, GAP_THRESHOLD)
            utility_by_arm = {
                arm: np.asarray(
                    [
                        base.hidden_arm_utility(
                            hidden,
                            arm,
                            md6_array,
                            float(mastery),
                            quality_placeholder,
                        )
                        for mastery in MASTERY_GRID
                    ],
                    dtype=np.float64,
                )
                for arm in candidates
            }
            mastery_average = {
                arm: float(np.mean(values))
                for arm, values in utility_by_arm.items()
            }
            mastery_blind_arm = max(
                candidates,
                key=mastery_average.__getitem__,
            )
            oracle_arms: list[str] = []
            advantages: list[float] = []

            for mastery_index, mastery in enumerate(MASTERY_GRID):
                utilities_at_mastery = {
                    arm: float(utility_by_arm[arm][mastery_index])
                    for arm in candidates
                }
                oracle_arm = max(
                    candidates,
                    key=utilities_at_mastery.__getitem__,
                )
                oracle_utility = utilities_at_mastery[oracle_arm]
                blind_utility = utilities_at_mastery[mastery_blind_arm]
                advantage = oracle_utility - blind_utility
                if advantage < -NUMERIC_TOLERANCE:
                    raise AssertionError(
                        "Mastery-aware oracle fell below the mastery-blind oracle."
                    )
                advantage = max(0.0, float(advantage))
                if len(candidates) == 1:
                    margin = float("nan")
                else:
                    ordered_utilities = sorted(
                        utilities_at_mastery.values(), reverse=True
                    )
                    margin = float(ordered_utilities[0] - ordered_utilities[1])

                row: dict[str, object] = {
                    "seed": seed,
                    "state_index": state_index,
                    "mastery": float(mastery),
                    "md6_p_generic": float(md6_array[0]),
                    "md6_p_probing": float(md6_array[1]),
                    "md6_p_focus": float(md6_array[2]),
                    "md6_p_telling": float(md6_array[3]),
                    "eligible_arm_count": len(candidates),
                    "eligible_arms": "|".join(candidates),
                    "oracle_arm": oracle_arm,
                    "mastery_blind_arm": mastery_blind_arm,
                    "oracle_utility": oracle_utility,
                    "mastery_blind_utility": blind_utility,
                    "mastery_information_advantage": advantage,
                    "oracle_margin": margin,
                }
                for arm in ARMS:
                    row[f"utility_{arm}"] = (
                        utilities_at_mastery[arm]
                        if arm in utilities_at_mastery
                        else float("nan")
                    )
                raw_rows.append(row)
                oracle_arms.append(oracle_arm)
                advantages.append(advantage)

            arm_ranges = {
                arm: float(np.max(values) - np.min(values))
                for arm, values in utility_by_arm.items()
            }
            for arm, utility_range in arm_ranges.items():
                range_rows.append(
                    {
                        "seed": seed,
                        "state_index": state_index,
                        "arm": arm,
                        "utility_range_mastery": utility_range,
                    }
                )
            transition_count = sum(
                left != right
                for left, right in zip(
                    oracle_arms[:-1], oracle_arms[1:], strict=True
                )
            )
            state_rows.append(
                {
                    "seed": seed,
                    "state_index": state_index,
                    "md6_p_generic": float(md6_array[0]),
                    "md6_p_probing": float(md6_array[1]),
                    "md6_p_focus": float(md6_array[2]),
                    "md6_p_telling": float(md6_array[3]),
                    "eligible_arm_count": len(candidates),
                    "eligible_arms": "|".join(candidates),
                    # Required field is an at-least-one-change indicator.
                    "optimal_arm_changes_over_sweep": int(transition_count > 0),
                    "num_optimal_arm_transitions": transition_count,
                    "num_distinct_optimal_arms": len(set(oracle_arms)),
                    "arm_at_mastery_020": oracle_arms[4],
                    "arm_at_mastery_080": oracle_arms[16],
                    "different_020_vs_080": int(oracle_arms[4] != oracle_arms[16]),
                    "max_mastery_utility_range": max(arm_ranges.values()),
                    "mean_mastery_information_advantage": float(
                        np.mean(advantages)
                    ),
                    "max_mastery_information_advantage": float(
                        np.max(advantages)
                    ),
                }
            )

    raw = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
    state = pd.DataFrame(state_rows, columns=STATE_COLUMNS)
    arm_ranges = pd.DataFrame(range_rows)
    return DiagnosticCore(raw, state, arm_ranges)


def _logical_dataframe_hash(frame: pd.DataFrame) -> str:
    """Hash dataframe values, columns, dtypes, and index deterministically."""

    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(
        dtype=np.uint64
    )
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _core_hashes(core: DiagnosticCore) -> dict[str, str]:
    return {
        "raw_mastery_sweep_logical_sha256": _logical_dataframe_hash(
            core.raw_sweep
        ),
        "state_level_summary_logical_sha256": _logical_dataframe_hash(
            core.state_level
        ),
        "arm_ranges_internal_logical_sha256": _logical_dataframe_hash(
            core.arm_ranges
        ),
    }


def run_sanity_checks() -> tuple[DiagnosticCore, list[dict[str, str]], dict[str, str]]:
    """Build twice and verify all required invariants before writing files."""

    first = build_diagnostic_core()
    checks: list[dict[str, str]] = []

    def passed(name: str) -> None:
        checks.append({"check": name, "status": "PASS"})

    utility_values = first.raw_sweep[list(UTILITY_COLUMNS)].to_numpy(dtype=float)
    finite_utility_values = utility_values[np.isfinite(utility_values)]
    assert np.all((finite_utility_values >= 0.0) & (finite_utility_values <= 1.0))
    passed("all eligible-arm utilities are finite and in [0,1]")

    for row in first.state_level.itertuples(index=False):
        md6 = {
            "generic": row.md6_p_generic,
            "probing": row.md6_p_probing,
            "focus": row.md6_p_focus,
            "telling": row.md6_p_telling,
        }
        expected = eligible_arms(md6, GAP_THRESHOLD)
        assert row.eligible_arms == "|".join(expected)
        assert row.eligible_arm_count == len(expected)
    passed("eligible arms match the canonical 0.10 mask")

    eligible_sets = first.raw_sweep["eligible_arms"].str.split("|")
    assert all(
        oracle in candidates
        for oracle, candidates in zip(
            first.raw_sweep["oracle_arm"], eligible_sets, strict=True
        )
    )
    passed("mastery-aware oracle is always eligible")
    assert all(
        oracle in candidates
        for oracle, candidates in zip(
            first.raw_sweep["mastery_blind_arm"], eligible_sets, strict=True
        )
    )
    passed("mastery-blind oracle is always eligible")

    assert float(first.raw_sweep["mastery_information_advantage"].min()) >= 0.0
    passed("mastery-information advantage is nonnegative")

    state_keys = ["seed", "state_index"]
    for column in (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
    ):
        unique_counts = first.raw_sweep.groupby(state_keys)[column].nunique()
        assert int(unique_counts.max()) == 1
    passed("changing mastery does not change MD6 probabilities")

    eligible_unique = first.raw_sweep.groupby(state_keys)["eligible_arms"].nunique()
    assert int(eligible_unique.max()) == 1
    passed("changing mastery does not change the eligible-arm set")

    assert len(first.raw_sweep) == len(SEEDS) * STATES_PER_SEED * len(MASTERY_GRID)
    assert len(first.state_level) == len(SEEDS) * STATES_PER_SEED
    passed("prescribed seed/state/mastery-grid row counts")

    first_hashes = _core_hashes(first)
    second = build_diagnostic_core()
    second_hashes = _core_hashes(second)
    assert first_hashes == second_hashes
    passed("deterministic in-memory rerun has identical logical result hashes")
    return first, checks, first_hashes


def _bootstrap_multiplicities() -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(SEEDS),
        size=(BOOTSTRAP_RESAMPLES, len(SEEDS)),
    )
    counts = np.zeros((BOOTSTRAP_RESAMPLES, len(SEEDS)), dtype=np.float64)
    rows = np.repeat(np.arange(BOOTSTRAP_RESAMPLES), len(SEEDS))
    np.add.at(counts, (rows, indices.ravel()), 1.0)
    return counts


def _seed_cluster_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    multiplicities: np.ndarray,
) -> tuple[float, float]:
    grouped = frame.groupby("seed")[value_column].agg(["sum", "count"])
    grouped = grouped.reindex(SEEDS, fill_value=0)
    boot_sum = multiplicities @ grouped["sum"].to_numpy(dtype=float)
    boot_count = multiplicities @ grouped["count"].to_numpy(dtype=float)
    valid = boot_count > 0
    if not np.any(valid):
        return float("nan"), float("nan")
    boot_means = boot_sum[valid] / boot_count[valid]
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def _distribution_row(
    section: str,
    metric: str,
    frame: pd.DataFrame,
    value_column: str,
    multiplicities: np.ndarray,
    *,
    subgroup: str = "all",
    unit: str = "utility",
    notes: str = "",
) -> dict[str, object]:
    clean = frame[["seed", value_column]].dropna()
    values = clean[value_column].to_numpy(dtype=float)
    if values.size == 0:
        return {
            "section": section,
            "metric": metric,
            "subgroup": subgroup,
            "n": 0,
            "n_seeds": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "maximum": float("nan"),
            "unit": unit,
            "notes": notes,
        }
    low, high = _seed_cluster_mean_ci(
        clean, value_column, multiplicities
    )
    return {
        "section": section,
        "metric": metric,
        "subgroup": subgroup,
        "n": int(values.size),
        "n_seeds": int(clean["seed"].nunique()),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "unit": unit,
        "notes": notes,
    }


def _correlation_row(
    raw: pd.DataFrame,
    method: str,
    multiplicities: np.ndarray,
) -> dict[str, object]:
    multi = raw.dropna(subset=["oracle_margin"])
    seed_correlations: list[dict[str, float]] = []
    for seed, group in multi.groupby("seed"):
        correlation = group[
            ["mastery_information_advantage", "oracle_margin"]
        ].corr(method=method).iloc[0, 1]
        seed_correlations.append({"seed": int(seed), "correlation": correlation})
    correlation_frame = pd.DataFrame(seed_correlations).dropna()
    row = _distribution_row(
        "decision_margin",
        f"within_seed_{method}_correlation_advantage_vs_margin",
        correlation_frame,
        "correlation",
        multiplicities,
        unit="correlation",
        notes="Distribution and seed-bootstrap CI of within-seed correlations.",
    )
    row["n"] = len(multi)
    return row


def compute_summary_statistics(core: DiagnosticCore) -> pd.DataFrame:
    """Compute all Section 3--7 metrics with seed-cluster bootstrap CIs."""

    raw = core.raw_sweep.copy()
    state = core.state_level.copy()
    ranges = core.arm_ranges.copy()
    multiplicities = _bootstrap_multiplicities()
    rows: list[dict[str, object]] = []

    rows.append(
        _distribution_row(
            "optimal_arm_change",
            "optimal_arm_change_rate_over_mastery_sweep",
            state,
            "optimal_arm_changes_over_sweep",
            multiplicities,
            unit="fraction",
            notes="1 when at least one adjacent mastery-grid transition changes the oracle arm.",
        )
    )
    rows.append(
        _distribution_row(
            "optimal_arm_change",
            "arm_change_rate_mastery_020_vs_080",
            state,
            "different_020_vs_080",
            multiplicities,
            unit="fraction",
        )
    )
    rows.append(
        _distribution_row(
            "optimal_arm_change",
            "num_distinct_optimal_arms_over_sweep",
            state,
            "num_distinct_optimal_arms",
            multiplicities,
            unit="arms",
        )
    )

    rows.append(
        _distribution_row(
            "mastery_utility_variation",
            "utility_range_mastery_state_arm",
            ranges,
            "utility_range_mastery",
            multiplicities,
            unit="utility",
        )
    )
    rows.append(
        _distribution_row(
            "mastery_utility_variation",
            "max_mastery_utility_range_per_state",
            state,
            "max_mastery_utility_range",
            multiplicities,
            unit="utility",
        )
    )

    rows.append(
        _distribution_row(
            "mastery_information_advantage",
            "mastery_information_advantage",
            raw,
            "mastery_information_advantage",
            multiplicities,
            unit="utility",
        )
    )
    for threshold in (0.0, 0.01, 0.025, 0.05):
        column = f"advantage_gt_{str(threshold).replace('.', '_')}"
        raw[column] = (
            raw["mastery_information_advantage"] > threshold
        ).astype(float)
        rows.append(
            _distribution_row(
                "mastery_information_advantage",
                f"fraction_advantage_gt_{threshold:g}",
                raw,
                column,
                multiplicities,
                unit="fraction",
                notes="Threshold is descriptive only.",
            )
        )

    state["baseline_only"] = (state["eligible_arm_count"] == 1).astype(float)
    state["two_eligible_arms"] = (state["eligible_arm_count"] == 2).astype(float)
    state["three_plus_eligible_arms"] = (
        state["eligible_arm_count"] >= 3
    ).astype(float)
    for metric, column in (
        ("fraction_baseline_only", "baseline_only"),
        ("fraction_two_eligible_arms", "two_eligible_arms"),
        ("fraction_three_plus_eligible_arms", "three_plus_eligible_arms"),
    ):
        rows.append(
            _distribution_row(
                "eligibility_mask",
                metric,
                state,
                column,
                multiplicities,
                unit="fraction",
            )
        )

    eligibility_groups = (
        ("baseline_only", state["eligible_arm_count"] == 1),
        ("two_eligible_arms", state["eligible_arm_count"] == 2),
        ("three_plus_eligible_arms", state["eligible_arm_count"] >= 3),
    )
    for subgroup, mask in eligibility_groups:
        subset = state.loc[mask]
        rows.append(
            _distribution_row(
                "eligibility_mask",
                "optimal_arm_change_rate_over_mastery_sweep",
                subset,
                "optimal_arm_changes_over_sweep",
                multiplicities,
                subgroup=subgroup,
                unit="fraction",
            )
        )
        rows.append(
            _distribution_row(
                "eligibility_mask",
                "mean_mastery_information_advantage_per_state",
                subset,
                "mean_mastery_information_advantage",
                multiplicities,
                subgroup=subgroup,
                unit="utility",
            )
        )

    multi = raw.dropna(subset=["oracle_margin"]).copy()
    rows.append(
        _distribution_row(
            "decision_margin",
            "oracle_margin_multi_arm_states",
            multi,
            "oracle_margin",
            multiplicities,
            subgroup="eligible_arm_count_gte_2",
            unit="utility",
            notes="Margin is unavailable and excluded for baseline-only states.",
        )
    )
    rows.append(_correlation_row(raw, "pearson", multiplicities))
    rows.append(_correlation_row(raw, "spearman", multiplicities))

    multi["margin_quartile"] = pd.qcut(
        multi["oracle_margin"],
        q=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        duplicates="drop",
    )
    for quartile, subset in multi.groupby("margin_quartile", observed=True):
        rows.append(
            _distribution_row(
                "decision_margin",
                "mastery_information_advantage_by_oracle_margin_quartile",
                subset,
                "mastery_information_advantage",
                multiplicities,
                subgroup=str(quartile),
                unit="utility",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as binary_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary_handle,
            mtime=0,
            compresslevel=6,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle,
                encoding="utf-8",
                newline="",
            ) as text_handle:
                frame.to_csv(text_handle, index=False, float_format="%.17g")


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    fig.savefig(
        figures_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "matplotlib", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _figure_information_advantage(raw: pd.DataFrame, figures_dir: Path) -> None:
    values = raw["mastery_information_advantage"].to_numpy(dtype=float)
    maximum = float(np.max(values))
    bins = np.linspace(0.0, maximum if maximum > 0 else 1.0, 61)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].hist(values, bins=bins, color="#4477AA", alpha=0.85, edgecolor="white")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xlabel("Mastery-information advantage (utility)")
    axes[0].set_ylabel("State–mastery observations (symlog count)")
    axes[0].set_title("Histogram")
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    axes[1].plot(ordered, cumulative, color="#228833", linewidth=1.8)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Mastery-information advantage (utility)")
    axes[1].set_ylabel("Empirical cumulative proportion")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Empirical cumulative distribution")
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
    fig.suptitle(
        "Full-oracle utility minus mastery-blind-oracle utility",
        fontsize=13,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir, "mastery_information_advantage")


def _figure_optimal_arm_by_mastery(raw: pd.DataFrame, figures_dir: Path) -> None:
    raw = raw.copy()
    raw["mastery_grid_index"] = np.rint(
        raw["mastery"].to_numpy(dtype=float) * 20.0
    ).astype(int)
    counts = (
        raw.groupby(["mastery_grid_index", "oracle_arm"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(len(MASTERY_GRID)), fill_value=0)
        .reindex(columns=ARMS, fill_value=0)
    )
    if not np.all(counts.sum(axis=1) == len(SEEDS) * STATES_PER_SEED):
        raise RuntimeError("A mastery-grid value is missing fixed MD6 states.")
    proportions = counts / (len(SEEDS) * STATES_PER_SEED)
    colors = ("#4C4C4C", "#4477AA", "#66CCEE", "#228833", "#CC6677")
    fig, axis = plt.subplots(figsize=(9.2, 5.1))
    for arm, color in zip(ARMS, colors, strict=True):
        axis.plot(
            MASTERY_GRID,
            proportions[arm],
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            label=arm,
            color=color,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Counterfactual mastery")
    axis.set_ylabel("Proportion of fixed MD6 states")
    axis.set_title("Hidden eligible-oracle arm by mastery-grid value")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(ncol=2, frameon=False)
    _save_figure(fig, figures_dir, "optimal_arm_by_mastery")


def _figure_utility_range(state: pd.DataFrame, figures_dir: Path) -> None:
    values = state["max_mastery_utility_range"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.hist(values, bins=50, color="#AA3377", alpha=0.85, edgecolor="white")
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    axis.set_xlabel("Maximum mastery-induced utility range within state")
    axis.set_ylabel("Fixed MD6 states")
    axis.set_title("Distribution of maximum eligible-arm utility range over mastery")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    _save_figure(fig, figures_dir, "mastery_utility_range")


def _figure_eligibility_signal(
    state: pd.DataFrame,
    figures_dir: Path,
) -> None:
    groups = (
        ("Baseline only", state["eligible_arm_count"] == 1),
        ("2 eligible arms", state["eligible_arm_count"] == 2),
        ("3+ eligible arms", state["eligible_arm_count"] >= 3),
    )
    multiplicities = _bootstrap_multiplicities()
    change_rates: list[float] = []
    change_low: list[float] = []
    change_high: list[float] = []
    advantages: list[float] = []
    advantage_low: list[float] = []
    advantage_high: list[float] = []
    labels: list[str] = []
    counts: list[int] = []
    for label, mask in groups:
        subset = state.loc[mask]
        labels.append(label)
        counts.append(len(subset))
        change = subset["optimal_arm_changes_over_sweep"].to_numpy(dtype=float)
        change_rates.append(float(np.mean(change)))
        low, high = _seed_cluster_mean_ci(
            subset,
            "optimal_arm_changes_over_sweep",
            multiplicities,
        )
        change_low.append(low)
        change_high.append(high)
        advantages.append(float(np.mean(subset["mean_mastery_information_advantage"])))
        low, high = _seed_cluster_mean_ci(
            subset,
            "mean_mastery_information_advantage",
            multiplicities,
        )
        advantage_low.append(low)
        advantage_high.append(high)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    axes[0].bar(x, change_rates, color="#4477AA", alpha=0.86)
    axes[0].errorbar(
        x,
        change_rates,
        yerr=np.vstack(
            (np.asarray(change_rates) - change_low, change_high - np.asarray(change_rates))
        ),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Optimal-arm change rate over mastery sweep")
    axes[0].set_title("Arm-ranking changes")
    axes[1].bar(x, advantages, color="#228833", alpha=0.86)
    axes[1].errorbar(
        x,
        advantages,
        yerr=np.vstack(
            (np.asarray(advantages) - advantage_low, advantage_high - np.asarray(advantages))
        ),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Mean mastery-information advantage")
    axes[1].set_title("Oracle utility difference")
    tick_labels = [f"{label}\nN={count:,}" for label, count in zip(labels, counts, strict=True)]
    for axis in axes:
        axis.set_xticks(x, tick_labels)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    fig.suptitle(
        "Mastery signal by canonical eligible-arm count\n"
        "Error bars are 95% seed-bootstrap intervals",
        fontsize=13,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir, "mastery_signal_by_eligible_count")


def _figure_representative_sweeps(
    raw: pd.DataFrame,
    state: pd.DataFrame,
    figures_dir: Path,
) -> None:
    selected: list[tuple[int, int]] = []
    for seed in REPRESENTATIVE_SEEDS:
        candidates = state[
            (state["seed"] == seed) & (state["eligible_arm_count"] > 1)
        ].sort_values("state_index")
        if candidates.empty:
            raise RuntimeError(f"Seed {seed} has no multi-arm diagnostic state.")
        selected.append((seed, int(candidates.iloc[0]["state_index"])))

    colors = dict(
        zip(
            ARMS,
            ("#4C4C4C", "#4477AA", "#66CCEE", "#228833", "#CC6677"),
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.0), sharex=True, sharey=False)
    for axis, (seed, state_index) in zip(axes, selected, strict=True):
        subset = raw[
            (raw["seed"] == seed) & (raw["state_index"] == state_index)
        ].sort_values("mastery")
        candidates = str(subset.iloc[0]["eligible_arms"]).split("|")
        for arm in candidates:
            axis.plot(
                subset["mastery"],
                subset[f"utility_{arm}"],
                color=colors[arm],
                linewidth=1.8,
                label=arm,
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(f"Seed {seed}, first multi-arm state\nstate {state_index}")
        axis.set_xlabel("Mastery")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Hidden expected utility")
    handles: dict[str, object] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            handles[label] = handle
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        "Deterministically selected E2 mastery utility sweeps",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.72, wspace=0.25)
    _save_figure(fig, figures_dir, "representative_mastery_sweeps")


def generate_figures_from_saved(output_dir: Path) -> None:
    """Generate all figures by rereading saved diagnostic CSV files."""

    raw = pd.read_csv(output_dir / "raw_mastery_sweep.csv.gz")
    state = pd.read_csv(output_dir / "state_level_summary.csv")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _figure_information_advantage(raw, figures_dir)
    _figure_optimal_arm_by_mastery(raw, figures_dir)
    _figure_utility_range(state, figures_dir)
    _figure_eligibility_signal(state, figures_dir)
    _figure_representative_sweeps(raw, state, figures_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _manifest(
    project_root: Path,
    stage_dir: Path,
    checks: list[dict[str, str]],
    core_hashes: dict[str, str],
) -> dict[str, object]:
    base_manifest_path = (
        project_root
        / "results"
        / "self_improvement"
        / BASE_EXPERIMENT
        / "experiment_manifest.json"
    )
    base_source_path = (
        project_root
        / "src"
        / "self_improvement"
        / "experiments"
        / "context_ablation.py"
    )
    with base_manifest_path.open("r", encoding="utf-8") as handle:
        base_manifest = json.load(handle)
    output_hashes = {
        str(path.relative_to(stage_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path.name != "diagnostic_manifest.json"
    }
    return {
        "diagnostic_name": DIAGNOSTIC_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_timezone": "UTC",
        "base_experiment": BASE_EXPERIMENT,
        "base_experiment_manifest_sha256": _sha256(base_manifest_path),
        "context_ablation_py_sha256": _sha256(base_source_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "environment": ENVIRONMENT,
        "environment_index": ENVIRONMENT_INDEX,
        "number_of_seeds": len(SEEDS),
        "seed_list": list(SEEDS),
        "states_per_seed": STATES_PER_SEED,
        "total_base_states": len(SEEDS) * STATES_PER_SEED,
        "mastery_grid": MASTERY_GRID.tolist(),
        "gap_threshold": GAP_THRESHOLD,
        "base_seed": base.BASE_SEED,
        "bootstrap_method": (
            "Nonparametric seed-cluster bootstrap. Seeds, not state rows or "
            "state-mastery rows, are resampled; percentile 95% intervals."
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "environment_reconstruction_strategy": (
            "Direct imports of context_ablation.generate_hidden_environment, "
            "generate_exogenous_world, hidden_arm_utility, and the canonical "
            "eligible_arms function. For each seed, generate 67 unchanged "
            "three-turn attempts and flatten/take the first 200 MD6 draws. "
            "Use an empty quality placeholder, which is ignored exactly by E2."
        ),
        "reused_function_source_sha256": {
            "generate_hidden_environment": _source_hash(
                base.generate_hidden_environment
            ),
            "generate_exogenous_world": _source_hash(
                base.generate_exogenous_world
            ),
            "hidden_arm_utility": _source_hash(base.hidden_arm_utility),
            "eligible_arms": _source_hash(eligible_arms),
        },
        "relevant_base_configuration": {
            "hidden_utility_generator": base_manifest["hidden_utility_generator"],
            "md6_generator": base_manifest["state_generators"]["md6"],
            "rng_strategy": base_manifest["rng_strategy"],
            "canonical_source_sha256": base_manifest[
                "canonical_source_sha256"
            ],
        },
        "mastery_blind_oracle": (
            "Canonical-order argmax among current eligible arms of utility "
            "averaged uniformly over the same 21-point mastery grid."
        ),
        "representative_sweep_selection": (
            "First state_index with eligible_arm_count > 1, independently for "
            "predeclared seeds [0,10,20,30,40]."
        ),
        "sanity_checks": checks,
        "deterministic_core_result_hashes": core_hashes,
        "output_file_sha256": output_hashes,
    }


def refresh_diagnostic_figures(output_dir: Path) -> None:
    """Regenerate figures from saved CSVs and refresh diagnostic hashes."""

    manifest_path = output_dir / "diagnostic_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing diagnostic manifest: {manifest_path}")
    generate_figures_from_saved(output_dir)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["output_file_sha256"] = {
        str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "diagnostic_manifest.json"
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_full_diagnostic(
    project_root: Path,
    output_dir: Path,
    core: DiagnosticCore,
    checks: list[dict[str, str]],
    core_hashes: dict[str, str],
) -> None:
    if not all(item["status"] == "PASS" for item in checks):
        raise RuntimeError("Sanity check failure; diagnostic output was not created.")
    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_dir}")
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {stage_dir}")
    stage_dir.mkdir(parents=True)

    _write_deterministic_gzip_csv(
        core.raw_sweep,
        stage_dir / "raw_mastery_sweep.csv.gz",
    )
    core.state_level.to_csv(
        stage_dir / "state_level_summary.csv",
        index=False,
        float_format="%.17g",
    )
    summary = compute_summary_statistics(core)
    summary.to_csv(
        stage_dir / "summary_statistics.csv",
        index=False,
        float_format="%.17g",
    )
    generate_figures_from_saved(stage_dir)
    manifest = _manifest(project_root, stage_dir, checks, core_hashes)
    with (stage_dir / "diagnostic_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    stage_dir.replace(output_dir)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity-only", action="store_true")
    mode.add_argument("--run-full", action="store_true")
    mode.add_argument("--refresh-figures", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = _default_project_root()
    output_dir = args.output_dir or (
        project_root
        / "results"
        / "self_improvement"
        / BASE_EXPERIMENT
        / "e2_mastery_diagnostic"
    )
    try:
        if args.refresh_figures:
            refresh_diagnostic_figures(output_dir)
            print(f"Refreshed diagnostic figures: {output_dir / 'figures'}")
            return 0
        core, checks, hashes = run_sanity_checks()
        print("E2 MASTERY SIGNAL DIAGNOSTIC SANITY CHECKS")
        for item in checks:
            print(f"- {item['check']}: {item['status']}")
        print("OVERALL: PASS")
        if args.run_full:
            run_full_diagnostic(
                project_root,
                output_dir,
                core,
                checks,
                hashes,
            )
            print(f"Diagnostic outputs: {output_dir}")
    except Exception as exc:
        print(f"OVERALL: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
