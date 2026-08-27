"""Generate the six final-sensitivity figures from saved CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_NAME: Final[str] = "final_sensitivity_suite_v1"
ENVIRONMENTS: Final[tuple[str, ...]] = ("E3_QUALITY", "E4_MIXED")
ENVIRONMENT_LABELS: Final[dict[str, str]] = {
    "E3_QUALITY": "E3: quality",
    "E4_MIXED": "E4: mixed",
}
CONDITION_ORDERS: Final[dict[str, tuple[str, ...]]] = {
    "reward": ("R_LINEAR", "R_SUCCESS_ONLY"),
    "credit": ("C_EQUAL", "C_RECENCY"),
    "lints": (
        "H_BASE",
        "H_RIDGE_LOW",
        "H_RIDGE_HIGH",
        "H_EXP_LOW",
        "H_EXP_HIGH",
    ),
}
COLORS: Final[dict[str, str]] = {
    "R_LINEAR": "#4c78a8",
    "R_SUCCESS_ONLY": "#e45756",
    "C_EQUAL": "#4c78a8",
    "C_RECENCY": "#f58518",
    "H_BASE": "#4c78a8",
    "H_RIDGE_LOW": "#72b7b2",
    "H_RIDGE_HIGH": "#e45756",
    "H_EXP_LOW": "#f2cf5b",
    "H_EXP_HIGH": "#b279a2",
}
DPI: Final[int] = 240


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _bootstrap_weights(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    weights = np.zeros((resamples, n), dtype=np.float64)
    rows = np.repeat(np.arange(resamples), n)
    np.add.at(weights, (rows, indices.ravel()), 1.0)
    return weights / n


def _curve(
    raw: pd.DataFrame,
    environment: str,
    condition: str,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = raw[
        (raw["environment"] == environment) & (raw["condition"] == condition)
    ]
    pivot = subset.pivot(
        index="seed", columns="attempt_index", values="cumulative_regret"
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.shape != (50, 500):
        raise RuntimeError("Learning-curve group is incomplete.")
    values = pivot.to_numpy(dtype=np.float64)
    bootstrap = weights @ values
    low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return (
        pivot.columns.to_numpy(dtype=np.int64),
        values.mean(axis=0),
        low,
        high,
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_two_condition_comparison(
    summary: pd.DataFrame,
    study: str,
    output_path: Path,
) -> None:
    endpoint = summary[summary["checkpoint"] == 500]
    conditions = CONDITION_ORDERS[study]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    metrics = (
        (
            "cumulative_regret",
            "Cumulative eligible-oracle regret @500",
            None,
        ),
        (
            "mean_evaluator_fraction",
            "Mean evaluator fraction (0–1)",
            (0.0, 1.0),
        ),
    )
    regret_high = float(endpoint["cumulative_regret_ci_high"].max()) * 1.10
    for row_index, environment in enumerate(ENVIRONMENTS):
        rows = endpoint[endpoint["environment"] == environment].set_index("condition")
        for column_index, (metric, ylabel, limits) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for x, condition in enumerate(conditions):
                mean = float(rows.loc[condition, f"{metric}_mean"])
                low = float(rows.loc[condition, f"{metric}_ci_low"])
                high = float(rows.loc[condition, f"{metric}_ci_high"])
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=COLORS[condition],
                    capsize=4,
                    markersize=7,
                    linewidth=1.8,
                )
            ax.set_xticks(range(len(conditions)), conditions, rotation=12)
            ax.set_ylabel(ylabel)
            ax.set_title(ENVIRONMENT_LABELS[environment])
            if limits is None:
                ax.set_ylim(0, regret_high)
            else:
                ax.set_ylim(*limits)
    title = "Reward-definition sensitivity" if study == "reward" else "Delayed-credit sensitivity"
    fig.suptitle(f"{title} at checkpoint 500\nError bars: 95% seed-bootstrap CI")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_path)


def plot_learning_curves(
    raw: pd.DataFrame,
    study: str,
    output_path: Path,
    weights: np.ndarray,
) -> None:
    conditions = CONDITION_ORDERS[study]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
    curve_cache: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    global_high = 0.0
    for environment in ENVIRONMENTS:
        for condition in conditions:
            curve_cache[(environment, condition)] = _curve(
                raw, environment, condition, weights
            )
            global_high = max(
                global_high,
                float(np.max(curve_cache[(environment, condition)][3])),
            )
    for ax, environment in zip(axes, ENVIRONMENTS, strict=True):
        for condition in conditions:
            attempts, mean, low, high = curve_cache[(environment, condition)]
            ax.plot(
                attempts,
                mean,
                color=COLORS[condition],
                linewidth=1.8,
                label=condition,
            )
            ax.fill_between(
                attempts, low, high, color=COLORS[condition], alpha=0.12
            )
        ax.set_title(ENVIRONMENT_LABELS[environment])
        ax.set_xlabel("Completed attempts")
        ax.set_xlim(1, 500)
        ax.set_ylim(0, global_high * 1.04)
    axes[0].set_ylabel("Mean cumulative eligible-oracle regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(conditions),
        bbox_to_anchor=(0.5, 0.98),
    )
    title = {
        "reward": "Reward-definition learning curves",
        "credit": "Delayed-credit learning curves",
        "lints": "Local LinTS learning curves",
    }[study]
    fig.suptitle(f"{title}\nBands: 95% seed-bootstrap CI", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, output_path)


def plot_lints_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    endpoint = summary[summary["checkpoint"] == 500]
    conditions = CONDITION_ORDERS["lints"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    global_high = float(endpoint["cumulative_regret_ci_high"].max()) * 1.08
    for ax, environment in zip(axes, ENVIRONMENTS, strict=True):
        rows = endpoint[endpoint["environment"] == environment].set_index("condition")
        for x, condition in enumerate(conditions):
            mean = float(rows.loc[condition, "cumulative_regret_mean"])
            low = float(rows.loc[condition, "cumulative_regret_ci_low"])
            high = float(rows.loc[condition, "cumulative_regret_ci_high"])
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color=COLORS[condition],
                capsize=4,
                markersize=7,
                linewidth=1.8,
            )
        ax.set_xticks(range(len(conditions)), conditions, rotation=24, ha="right")
        ax.set_title(ENVIRONMENT_LABELS[environment])
        ax.set_xlabel("LinTS configuration")
        ax.set_ylim(0, global_high)
    axes[0].set_ylabel("Mean cumulative eligible-oracle regret @500")
    fig.suptitle("Local LinTS sensitivity\nError bars: 95% seed-bootstrap CI")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, output_path)


def generate_all(results_dir: Path) -> None:
    manifest = json.loads((results_dir / "experiment_manifest.json").read_text())
    weights = _bootstrap_weights(
        50,
        int(manifest["bootstrap_resamples"]),
        int(manifest["bootstrap_seed"]),
    )
    _style()
    for study in ("reward", "credit"):
        summary = pd.read_csv(results_dir / study / "summary.csv")
        raw = pd.read_csv(results_dir / study / "raw_attempt_metrics.csv")
        plot_two_condition_comparison(
            summary,
            study,
            results_dir / study / "figures" / f"{study}_comparison.png",
        )
        plot_learning_curves(
            raw,
            study,
            results_dir / study / "figures" / f"{study}_learning_curve.png",
            weights,
        )
    lints_summary = pd.read_csv(results_dir / "lints" / "summary.csv")
    lints_raw = pd.read_csv(results_dir / "lints" / "raw_attempt_metrics.csv")
    plot_lints_sensitivity(
        lints_summary,
        results_dir / "lints" / "figures" / "lints_sensitivity.png",
    )
    plot_learning_curves(
        lints_raw,
        "lints",
        results_dir / "lints" / "figures" / "lints_learning_curve.png",
        weights,
    )


def _default_results_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "results"
        / "self_improvement"
        / EXPERIMENT_NAME
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_default_results_dir())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generate_all(args.results_dir.resolve())
    except Exception as exc:
        raise SystemExit(
            f"Figure generation failed: {type(exc).__name__}: {exc}"
        ) from exc
    print(f"generated six figures under {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
