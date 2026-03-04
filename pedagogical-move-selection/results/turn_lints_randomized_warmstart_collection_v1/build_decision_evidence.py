"""Build compact decision-focused figures from the completed offline audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent / "turn_lints_randomized_warmstart_audit_v1"
FIGURE_ROOT = HERE / "figures"
MOVES = ("generic", "probing", "focus", "telling")
COLORS = {
    "generic": "#4C78A8",
    "probing": "#F58518",
    "focus": "#54A24B",
    "telling": "#E45756",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (AUDIT_ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure: plt.Figure, name: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_ROOT / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def action_coverage() -> None:
    rows = read_csv("expected_action_coverage.csv")
    exact = [float(next(row for row in rows if row["move"] == move)["exact_expected_action_rate"]) for move in MOVES]
    simulated = [float(next(row for row in rows if row["move"] == move)["simulated_action_rate"]) for move in MOVES]
    x = np.arange(len(MOVES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(x - width / 2, exact, width, label="Exact mean MD7 probability", color="#4C78A8")
    axis.bar(x + width / 2, simulated, width, label="Deterministic MC verification", color="#9ECAE9")
    axis.set_title("Raw MD7 sampling gives non-zero expected support to all four moves")
    axis.set_ylabel("Expected action rate (proportion)")
    axis.set_xlabel("Pedagogical move")
    axis.set_xticks(x, MOVES)
    axis.set_ylim(0.0, 0.40)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    save(figure, "expected_vs_simulated_action_rates.png")


def confidence_exploration() -> None:
    summary = json.loads((AUDIT_ROOT / "summary.json").read_text(encoding="utf-8"))
    rows = summary["top1"]["confidence_rank_quartiles"]
    x = np.arange(1, 5)
    agreement = [float(row["expected_top1_agreement"]) for row in rows]
    deviation = [float(row["expected_deviation_rate"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(x, agreement, marker="o", linewidth=2, label="Top-1 agreement", color="#4C78A8")
    axis.plot(x, deviation, marker="o", linewidth=2, label="Non-top-1 sampling", color="#E45756")
    axis.set_title("Exploration decreases as MD7 top-1 confidence increases")
    axis.set_ylabel("Expected probability")
    axis.set_xlabel("Empirical top-1-confidence quartile (low to high)")
    axis.set_xticks(x, ("Q1 low", "Q2", "Q3", "Q4 high"))
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save(figure, "confidence_vs_top1_deviation.png")


def sampled_propensity() -> None:
    row = next(
        item for item in read_csv("sampling_propensity_summary.csv")
        if item["metric"] == "sampled_action_propensity"
    )
    labels = ("P1", "P5", "P10", "Q1", "Median", "P90", "P95", "P99")
    keys = ("p1", "p5", "p10", "q1", "median", "p90", "p95", "p99")
    values = [float(row[key]) for key in keys]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(np.arange(len(labels)), values, marker="o", linewidth=2, color="#7A5195")
    axis.axhline(0.05, linestyle="--", linewidth=1, color="#777777", label="Descriptive .05 reference")
    axis.set_title("Selected-action propensities are typically well above the thin lower tail")
    axis.set_ylabel("Selected-action propensity")
    axis.set_xlabel("Empirical quantile across deterministic sampling draws")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save(figure, "sampled_action_propensity_quantiles.png")


def horizon_coverage() -> None:
    rows = read_csv("coverage_horizon_simulation.csv")
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for move in MOVES:
        selected = sorted(
            (row for row in rows if row["move"] == move),
            key=lambda row: int(row["horizon"]),
        )
        horizons = np.asarray([int(row["horizon"]) for row in selected])
        means = np.asarray([float(row["simulation_mean_count"]) for row in selected])
        lower = np.asarray([float(row["simulation_p5_count"]) for row in selected])
        upper = np.asarray([float(row["simulation_p95_count"]) for row in selected])
        axis.plot(horizons, means, marker="o", label=move, color=COLORS[move])
        axis.fill_between(horizons, lower, upper, color=COLORS[move], alpha=0.12)
    axis.set_title("Planning horizons retain expected support across every action")
    axis.set_ylabel("Simulated action count")
    axis.set_xlabel("Hypothetical randomized warm-start horizon (turns)")
    axis.set_xlim(50, 500)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.25)
    save(figure, "action_coverage_by_planning_horizon.png")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    action_coverage()
    confidence_exploration()
    sampled_propensity()
    horizon_coverage()
    print(f"Wrote 4 decision figures to {FIGURE_ROOT}")
