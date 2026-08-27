"""Generate context_ablation_v1 figures only from saved CSV result files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


CONTEXTS: Final[tuple[str, ...]] = ("C0", "C1", "C2", "C3", "C4")
ENVIRONMENTS: Final[tuple[str, ...]] = (
    "E0_NO_CONTEXT",
    "E1_MD6",
    "E2_MASTERY",
    "E3_QUALITY",
    "E4_MIXED",
)
CHECKPOINTS: Final[tuple[int, ...]] = (25, 50, 100, 250, 500)
BOOTSTRAP_SEED: Final[int] = 814730
NUM_BOOTSTRAP_RESAMPLES: Final[int] = 2_000
COLORS: Final[dict[str, str]] = {
    "C0": "#4C4C4C",
    "C1": "#4477AA",
    "C2": "#66CCEE",
    "C3": "#228833",
    "C4": "#CC6677",
}


def _bootstrap_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(NUM_BOOTSTRAP_RESAMPLES, n))
    weights = np.zeros((NUM_BOOTSTRAP_RESAMPLES, n), dtype=np.float64)
    rows = np.repeat(np.arange(NUM_BOOTSTRAP_RESAMPLES), n)
    np.add.at(weights, (rows, indices.ravel()), 1.0)
    return weights / n


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        figures_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def cumulative_regret_figure(raw: pd.DataFrame, figures_dir: Path) -> None:
    weights = _bootstrap_weights(raw["seed"].nunique())
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1), sharex=True, sharey=False)
    for axis, environment in zip(axes, ENVIRONMENTS, strict=True):
        subset = raw[raw["environment"] == environment]
        max_upper = 0.0
        for context in CONTEXTS:
            values = subset[subset["context_condition"] == context].pivot(
                index="seed", columns="attempt_index", values="cumulative_regret"
            )
            values = values.sort_index().sort_index(axis=1)
            matrix = values.to_numpy(dtype=np.float64)
            mean = np.mean(matrix, axis=0)
            bootstrap_means = weights @ matrix
            low, high = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
            attempts = values.columns.to_numpy(dtype=int)
            axis.plot(
                attempts,
                mean,
                color=COLORS[context],
                linewidth=1.8,
                label=context,
            )
            axis.fill_between(
                attempts, low, high, color=COLORS[context], alpha=0.14, linewidth=0
            )
            max_upper = max(max_upper, float(np.max(high)))
        axis.set_title(environment.replace("_", " "), fontsize=10)
        axis.set_xlabel("Completed attempts")
        axis.set_xlim(1, 500)
        axis.set_ylim(0, max_upper * 1.04 if max_upper > 0 else 1.0)
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Cumulative eligible-oracle regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Turn-level LinTS cumulative regret by synthetic environment\n"
        "Lines are seed means; bands are 95% seed-bootstrap intervals",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.74, wspace=0.27)
    _save(fig, figures_dir, "cumulative_regret")


def paired_context_effects_figure(
    paired: pd.DataFrame, figures_dir: Path
) -> None:
    data = paired[paired["checkpoint"] == 500].copy()
    data["label"] = data["comparison"] + " · " + data["environment"]
    data = data.reset_index(drop=True)
    positions = np.arange(len(data))
    means = data["mean_paired_improvement"].to_numpy(dtype=float)
    low = data["bootstrap_ci_low"].to_numpy(dtype=float)
    high = data["bootstrap_ci_high"].to_numpy(dtype=float)
    colors = [COLORS[str(value).split("-")[0]] for value in data["comparison"]]

    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    for position, mean, lower, upper, color in zip(
        positions, means, low, high, colors, strict=True
    ):
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
    axis.set_yticks(positions, data["label"])
    axis.invert_yaxis()
    axis.set_xlabel(
        "Paired cumulative-regret improvement at attempt 500\n"
        "(simpler − richer; positive indicates lower observed regret for richer)"
    )
    axis.set_title(
        "Pre-specified paired context effects\n"
        "Points are mean paired differences; bars are 95% seed-bootstrap intervals"
    )
    axis.grid(axis="x", alpha=0.25, linewidth=0.7)
    _save(fig, figures_dir, "paired_context_effects")


def heatmap_figure(summary: pd.DataFrame, figures_dir: Path) -> None:
    at_500 = summary[summary["checkpoint"] == 500]
    pivot = at_500.pivot(
        index="context_condition",
        columns="environment",
        values="cumulative_regret_mean",
    ).reindex(index=CONTEXTS, columns=ENVIRONMENTS)
    matrix = pivot.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10.2, 5.2))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(
        np.arange(len(ENVIRONMENTS)),
        [value.replace("_", "\n") for value in ENVIRONMENTS],
        fontsize=9,
    )
    axis.set_yticks(np.arange(len(CONTEXTS)), CONTEXTS)
    axis.set_xlabel("Synthetic environment")
    axis.set_ylabel("Context condition")
    axis.set_title("Mean cumulative eligible-oracle regret at attempt 500 (lower is better)")
    threshold = float(np.nanmedian(matrix))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= threshold else "black",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("Mean cumulative regret")
    _save(fig, figures_dir, "context_environment_heatmap")


def optimal_arm_rate_figure(raw: pd.DataFrame, figures_dir: Path) -> None:
    windows = ((1, 100, "Attempts 1–100"), (401, 500, "Attempts 401–500"))
    weights = _bootstrap_weights(raw["seed"].nunique())
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), sharey=True)
    for axis, environment in zip(axes, ENVIRONMENTS, strict=True):
        env_data = raw[raw["environment"] == environment]
        x = np.arange(len(CONTEXTS), dtype=float)
        for window_index, (start, end, label) in enumerate(windows):
            window = env_data[
                (env_data["attempt_index"] >= start)
                & (env_data["attempt_index"] <= end)
            ]
            seed_rates = (
                window.groupby(["context_condition", "seed"], sort=False)[
                    "optimal_arm_rate_for_attempt"
                ]
                .mean()
                .reset_index()
            )
            means = []
            lows = []
            highs = []
            for context in CONTEXTS:
                values = (
                    seed_rates[seed_rates["context_condition"] == context]
                    .sort_values("seed")["optimal_arm_rate_for_attempt"]
                    .to_numpy(dtype=float)
                )
                boot = weights @ values
                low, high = np.percentile(boot, [2.5, 97.5])
                means.append(float(np.mean(values)))
                lows.append(float(low))
                highs.append(float(high))
            offset = -0.10 if window_index == 0 else 0.10
            marker = "o" if window_index == 0 else "s"
            asymmetric_error = np.vstack(
                (
                    np.asarray(means) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(means),
                )
            )
            axis.errorbar(
                x + offset,
                means,
                yerr=asymmetric_error,
                fmt=marker,
                color="#4477AA" if window_index == 0 else "#CC6677",
                capsize=2.5,
                markersize=4.5,
                linewidth=1.1,
                label=label,
            )
        axis.set_xticks(x, CONTEXTS)
        axis.set_title(environment.replace("_", " "), fontsize=10)
        axis.set_xlabel("Context")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Optimal eligible-arm selection rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Optimal eligible-arm selection rate by early and late attempt windows\n"
        "Points are seed means; bars are 95% seed-bootstrap intervals",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.72, wspace=0.18)
    _save(fig, figures_dir, "optimal_arm_rate")


def sample_efficiency_figure(summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), sharex=True, sharey=False)
    for axis, environment in zip(axes, ENVIRONMENTS, strict=True):
        env_data = summary[summary["environment"] == environment]
        maximum = 0.0
        for context in CONTEXTS:
            data = env_data[env_data["context_condition"] == context].sort_values(
                "checkpoint"
            )
            x = data["checkpoint"].to_numpy(dtype=int)
            mean = data["cumulative_regret_mean"].to_numpy(dtype=float)
            low = data["cumulative_regret_ci_low"].to_numpy(dtype=float)
            high = data["cumulative_regret_ci_high"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                color=COLORS[context],
                label=context,
            )
            axis.fill_between(x, low, high, color=COLORS[context], alpha=0.13)
            maximum = max(maximum, float(np.max(high)))
        axis.set_xscale("log")
        axis.set_xticks(CHECKPOINTS, [str(value) for value in CHECKPOINTS], rotation=30)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.set_xlim(22, 525)
        axis.set_ylim(0, maximum * 1.05 if maximum > 0 else 1.0)
        axis.set_title(environment.replace("_", " "), fontsize=10)
        axis.set_xlabel("Checkpoint (attempts, log scale)")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Cumulative eligible-oracle regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Cumulative regret at pre-specified sample-efficiency checkpoints\n"
        "Lines are seed means; bands are 95% seed-bootstrap intervals",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.74, wspace=0.27)
    _save(fig, figures_dir, "sample_efficiency")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_manifest(results_dir: Path, figures_dir: Path) -> None:
    manifest_path = results_dir / "experiment_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["figure_generation"] = {
        "source_data": [
            "raw_attempt_metrics.csv",
            "summary_statistics.csv",
            "paired_comparisons.csv",
        ],
        "matplotlib_version": matplotlib.__version__,
        "bootstrap_unit": "seed",
        "bootstrap_resamples": NUM_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "figure_sha256": {
            path.name: _sha256(path) for path in sorted(figures_dir.iterdir())
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_figures(results_dir: Path) -> None:
    required = (
        results_dir / "raw_attempt_metrics.csv",
        results_dir / "summary_statistics.csv",
        results_dir / "paired_comparisons.csv",
        results_dir / "experiment_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required saved result files: {missing}")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    raw = pd.read_csv(
        results_dir / "raw_attempt_metrics.csv",
        usecols=(
            "environment",
            "context_condition",
            "seed",
            "attempt_index",
            "cumulative_regret",
            "optimal_arm_rate_for_attempt",
        ),
    )
    summary = pd.read_csv(results_dir / "summary_statistics.csv")
    paired = pd.read_csv(results_dir / "paired_comparisons.csv")

    cumulative_regret_figure(raw, figures_dir)
    paired_context_effects_figure(paired, figures_dir)
    heatmap_figure(summary, figures_dir)
    optimal_arm_rate_figure(raw, figures_dir)
    sample_efficiency_figure(summary, figures_dir)
    update_manifest(results_dir, figures_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing saved context_ablation_v1 CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generate_figures(args.results_dir.resolve())
    print(f"Generated context ablation figures in {args.results_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
