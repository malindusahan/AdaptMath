"""Generate Stage B threshold-sensitivity figures from saved CSV results only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_NAME: Final[str] = "threshold_sensitivity_v1"
ENVIRONMENT_LABELS: Final[dict[str, str]] = {
    "E0_NO_CONTEXT": "E0: no context",
    "E3_QUALITY": "E3: quality",
    "E4_MIXED": "E4: mixed",
}
THRESHOLD_COLORS: Final[dict[float, str]] = {
    0.00: "#4c78a8",
    0.05: "#72b7b2",
    0.10: "#f58518",
    0.15: "#eeca3b",
    0.20: "#b279a2",
    0.30: "#e45756",
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


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _threshold_ticks(ax: plt.Axes, values: Sequence[float]) -> None:
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:.2f}" for value in values])


def _endpoint(summary: pd.DataFrame) -> pd.DataFrame:
    checkpoint = int(summary["checkpoint"].max())
    return summary[summary["checkpoint"] == checkpoint].copy()


def _validate_inputs(
    summary: pd.DataFrame,
    checkpoints: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    environments = list(dict.fromkeys(summary["environment"].astype(str)))
    thresholds = sorted(summary["threshold"].astype(float).unique().tolist())
    if len(environments) != 3 or len(thresholds) != 6:
        raise RuntimeError("Expected exactly three environments and six thresholds.")
    expected_summary_rows = len(environments) * len(thresholds) * 5
    if len(summary) != expected_summary_rows:
        raise RuntimeError("Summary CSV has an unexpected number of rows.")
    if len(checkpoints) != len(environments) * len(thresholds) * 50 * 5:
        raise RuntimeError("Checkpoint CSV has an unexpected number of rows.")
    if len(tradeoff) != len(environments) * len(thresholds):
        raise RuntimeError("Tradeoff CSV has an unexpected number of rows.")
    return environments, thresholds


def plot_regret_vs_threshold(
    endpoint: pd.DataFrame,
    environments: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True)
    threshold_values = sorted(endpoint["threshold"].astype(float).unique())
    for ax, environment in zip(axes, environments, strict=True):
        rows = endpoint[endpoint["environment"] == environment].sort_values("threshold")
        x = rows["threshold"].to_numpy(float)
        mean = rows["cumulative_regret_mean"].to_numpy(float)
        low = rows["cumulative_regret_ci_low"].to_numpy(float)
        high = rows["cumulative_regret_ci_high"].to_numpy(float)
        ax.plot(x, mean, color="#3b528b", marker="o", linewidth=2)
        ax.fill_between(x, low, high, color="#3b528b", alpha=0.18)
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Probability-gap threshold")
        _threshold_ticks(ax, threshold_values)
    global_high = float(endpoint["cumulative_regret_ci_high"].max())
    axes[0].set_ylim(0, global_high * 1.06)
    axes[0].set_ylabel("Cumulative eligible-oracle regret @500")
    fig.suptitle("Cumulative regret across gap thresholds")
    fig.text(0.5, -0.01, "Bands: 95% percentile bootstrap CI across 50 seeds", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, figures_dir, "regret_vs_threshold")


def plot_reward_vs_threshold(
    endpoint: pd.DataFrame,
    environments: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    threshold_values = sorted(endpoint["threshold"].astype(float).unique())
    for ax, environment in zip(axes, environments, strict=True):
        rows = endpoint[endpoint["environment"] == environment].sort_values("threshold")
        x = rows["threshold"].to_numpy(float)
        mean = rows["mean_reward_mean"].to_numpy(float)
        low = rows["mean_reward_ci_low"].to_numpy(float)
        high = rows["mean_reward_ci_high"].to_numpy(float)
        ax.plot(x, mean, color="#2a9d8f", marker="o", linewidth=2)
        ax.fill_between(x, low, high, color="#2a9d8f", alpha=0.18)
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Probability-gap threshold")
        ax.set_ylim(0, 1)
        _threshold_ticks(ax, threshold_values)
    axes[0].set_ylabel("Mean evaluator reward (0–1)")
    fig.suptitle("Evaluator reward across gap thresholds")
    fig.text(0.5, -0.01, "Bands: 95% percentile bootstrap CI across 50 seeds", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, figures_dir, "reward_vs_threshold")


def plot_eligibility_vs_threshold(
    endpoint: pd.DataFrame,
    environments: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    threshold_values = sorted(endpoint["threshold"].astype(float).unique())
    handles = []
    labels = []
    for ax, environment in zip(axes, environments, strict=True):
        rows = endpoint[endpoint["environment"] == environment].sort_values("threshold")
        x = rows["threshold"].to_numpy(float)
        first = ax.plot(x, rows["baseline_only_rate_mean"], color="#4c78a8", marker="o", label="Baseline-only rate")[0]
        second = ax.plot(x, rows["at_least_one_alternative_rate_mean"], color="#e45756", marker="s", label="≥1 alternative rate")[0]
        count_axis = ax.twinx()
        third = count_axis.plot(x, rows["mean_eligible_arm_count_mean"], color="#54a24b", marker="^", linestyle="--", label="Mean eligible arms")[0]
        ax.set_ylim(0, 1)
        count_axis.set_ylim(1, 4)
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Probability-gap threshold")
        _threshold_ticks(ax, threshold_values)
        if ax is axes[0]:
            ax.set_ylabel("Turn rate")
        if ax is axes[-1]:
            count_axis.set_ylabel("Mean eligible-arm count")
        handles = [first, second, third]
        labels = [item.get_label() for item in handles]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Eligibility induced by the threshold", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, figures_dir, "eligibility_vs_threshold")


def plot_override_rate_vs_threshold(
    endpoint: pd.DataFrame,
    environments: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True)
    series = (
        ("override_rate_mean", "Actual override rate", "#e45756", "o", "-", "full", 3),
        ("bias_arm_selection_rate_mean", "Bias-arm selection rate", "#d6a800", "s", "--", "none", 4),
        ("baseline_selection_rate_mean", "Baseline selection rate", "#4c78a8", "^", "-", "full", 2),
    )
    threshold_values = sorted(endpoint["threshold"].astype(float).unique())
    for ax, environment in zip(axes, environments, strict=True):
        rows = endpoint[endpoint["environment"] == environment].sort_values("threshold")
        for column, label, color, marker, linestyle, fillstyle, zorder in series:
            ax.plot(
                rows["threshold"],
                rows[column],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                fillstyle=fillstyle,
                markeredgewidth=1.4,
                zorder=zorder,
            )
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Probability-gap threshold")
        ax.set_ylim(0, 1)
        _threshold_ticks(ax, threshold_values)
    axes[0].set_ylabel("Fraction of turns")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Policy intervention behavior", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, figures_dir, "override_rate_vs_threshold")


def plot_safety_performance_tradeoff(
    tradeoff: pd.DataFrame,
    environments: list[str],
    thresholds: list[float],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    for ax, environment in zip(axes, environments, strict=True):
        rows = tradeoff[tradeoff["environment"] == environment].sort_values("threshold")
        for _, row in rows.iterrows():
            threshold = float(row["threshold"])
            ax.scatter(
                row["override_rate"],
                row["mean_cumulative_regret"],
                s=52,
                color=THRESHOLD_COLORS[threshold],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                f"{threshold:.2f}",
                (row["override_rate"], row["mean_cumulative_regret"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Actual override rate @500")
    max_x = float(tradeoff["override_rate"].max())
    max_y = float(tradeoff["mean_cumulative_regret"].max())
    for ax in axes:
        ax.set_xlim(0, max_x * 1.10)
    axes[0].set_ylim(0, max_y * 1.10)
    axes[0].set_ylabel("Mean cumulative eligible-oracle regret @500")
    fig.suptitle("Intervention–performance trade-off (labels are thresholds)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, figures_dir, "safety_performance_tradeoff")


def plot_override_gap_distribution(
    checkpoints: pd.DataFrame,
    environments: list[str],
    figures_dir: Path,
) -> None:
    endpoint = checkpoints[checkpoints["checkpoint"] == checkpoints["checkpoint"].max()]
    grouped = endpoint.groupby(["environment", "threshold"], sort=False)[
        ["mean_override_gap", "median_override_gap", "p90_override_gap"]
    ].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True)
    threshold_values = sorted(grouped["threshold"].astype(float).unique())
    series = (
        ("mean_override_gap", "Mean override gap", "#4c78a8", "o"),
        ("median_override_gap", "Median override gap", "#54a24b", "s"),
        ("p90_override_gap", "P90 override gap", "#e45756", "^"),
    )
    for ax, environment in zip(axes, environments, strict=True):
        rows = grouped[grouped["environment"] == environment].sort_values("threshold")
        for column, label, color, marker in series:
            ax.plot(rows["threshold"], rows[column], label=label, color=color, marker=marker)
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Probability-gap threshold")
        _threshold_ticks(ax, threshold_values)
    global_high = float(grouped[["mean_override_gap", "median_override_gap", "p90_override_gap"]].max().max())
    axes[0].set_ylim(0, global_high * 1.08)
    axes[0].set_ylabel("MD6 probability gap on override turns")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Override-gap summaries", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, figures_dir, "override_gap_distribution")


def plot_sample_efficiency(
    summary: pd.DataFrame,
    environments: list[str],
    thresholds: list[float],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), sharey=True)
    for ax, environment in zip(axes, environments, strict=True):
        env_rows = summary[summary["environment"] == environment]
        for threshold in thresholds:
            rows = env_rows[np.isclose(env_rows["threshold"], threshold)].sort_values("checkpoint")
            ax.plot(
                rows["checkpoint"],
                rows["cumulative_regret_mean"],
                color=THRESHOLD_COLORS[threshold],
                marker="o",
                linewidth=1.6,
                label=f"{threshold:.2f}",
            )
        ax.set_title(ENVIRONMENT_LABELS.get(environment, environment))
        ax.set_xlabel("Completed attempts")
        ax.set_xticks([25, 50, 100, 250, 500])
    global_high = float(summary["cumulative_regret_mean"].max())
    axes[0].set_ylim(0, global_high * 1.05)
    axes[0].set_ylabel("Mean cumulative eligible-oracle regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Threshold", loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Cumulative regret at pre-specified checkpoints", y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, figures_dir, "sample_efficiency_by_threshold")


def generate_all(results_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "summary_statistics.csv")
    checkpoints = pd.read_csv(results_dir / "checkpoint_metrics.csv")
    tradeoff = pd.read_csv(results_dir / "tradeoff_summary.csv")
    environments, thresholds = _validate_inputs(summary, checkpoints, tradeoff)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _style()
    endpoint = _endpoint(summary)
    plot_regret_vs_threshold(endpoint, environments, figures_dir)
    plot_reward_vs_threshold(endpoint, environments, figures_dir)
    plot_eligibility_vs_threshold(endpoint, environments, figures_dir)
    plot_override_rate_vs_threshold(endpoint, environments, figures_dir)
    plot_safety_performance_tradeoff(tradeoff, environments, thresholds, figures_dir)
    plot_override_gap_distribution(checkpoints, environments, figures_dir)
    plot_sample_efficiency(summary, environments, thresholds, figures_dir)


def _default_results_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "results" / "self_improvement" / EXPERIMENT_NAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_default_results_dir())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generate_all(args.results_dir.resolve())
    except Exception as exc:
        raise SystemExit(f"Figure generation failed: {type(exc).__name__}: {exc}") from exc
    print(f"generated threshold-sensitivity figures in {args.results_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
