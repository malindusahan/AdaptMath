"""Reproducibly build the derived S+K+L architecture-decision amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
ORIGINAL = HERE.parent / "turn_lints_final_architecture_selection_v1"
FIGURES.mkdir(parents=True, exist_ok=True)

ORIGINAL_HASHES = {
    "FINAL_ARCHITECTURE.json": "b90b8efc87079e59ab88e041c5e8e4c9a6150350d6847a1976367311996834a0",
    "analysis.py": "586ba4ff25ba4462eca712320753a07dc88a884e0de16f9d93bc005e29ab53e5",
    "derived_analysis_dataset.csv": "d8f219b7e0cf351fb0bc2fc460f5183ff057a87a8fdce8d372ff2ae27dd67786",
    "context_ablation_results.csv": "c411e48c6b9983ba36c0944f4157abd9d058f7a462d74d345583588727b1a7b0",
    "context_bootstrap_comparisons.csv": "cd5007259ea88b5d6c10c08ddd4a85debd14230e7da9feabf2383c5737ce0b87",
    "context_feature_coverage.csv": "7f3a51ea0fd239d16dbd14dffd5dcebb26623a20b66e714dea730bd753773e52",
}

FEATURES = [
    "selector_p_generic",
    "selector_p_probing",
    "selector_p_focus",
    "selector_p_telling",
    "mastery_before",
    "previous_mastery_delta",
    "previous_mastery_delta_missing",
    "previous_reasoning_probability",
    "previous_uncertainty_probability",
    "previous_clarification_probability",
    "previous_learner_signals_missing",
]
DISPLAY = {
    "S": "S",
    "S+K": "S+K",
    "S+K+L": "S+K+L",
    "S+K+L+H+Q": "FULL",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify_original() -> None:
    for name, expected in ORIGINAL_HASHES.items():
        actual = sha(ORIGINAL / name)
        if actual != expected:
            raise RuntimeError(f"Historical evidence changed: {name}: {actual}")
    old = json.loads((ORIGINAL / "FINAL_ARCHITECTURE.json").read_text(encoding="utf-8"))
    if old["context_preset"] != "S" or old["context_dimension"] != 4:
        raise RuntimeError("Historical manifest is no longer the S-selection result.")


def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contexts = pd.read_csv(ORIGINAL / "context_ablation_results.csv")
    bootstrap = pd.read_csv(ORIGINAL / "context_bootstrap_comparisons.csv")
    coverage = pd.read_csv(ORIGINAL / "context_feature_coverage.csv")
    rewards = pd.read_csv(ORIGINAL / "reward_experiment_results.csv")
    chosen = contexts[contexts["blocks"].isin(DISPLAY)].copy()
    chosen["label"] = chosen["blocks"].map(DISPLAY)
    chosen = chosen.set_index("blocks").loc[list(DISPLAY)].reset_index()
    expected = {
        "S": (4, 0.16490832093301827, 0.20959047733336078, -0.2517050416959512),
        "S+K": (7, 0.16258889191023523, 0.20523015308213552, -0.2052541314478286),
        "S+K+L": (11, 0.16162730348763021, 0.2046320420681585, -0.1948852942438591),
        "S+K+L+H+Q": (22, 0.16800578218683307, 0.20904457170949403, -0.2570096112554379),
    }
    for name, values in expected.items():
        row = chosen[chosen["blocks"] == name].iloc[0]
        actual = (int(row.context_dimension), row.MAE_mean, row.RMSE_mean, row.R2_mean)
        if not np.allclose(actual, values, atol=1e-12):
            raise RuntimeError(f"Unexpected historical metrics for {name}: {actual}")
    return chosen, bootstrap, coverage, rewards


def metric_figure(data: pd.DataFrame, metric: str, filename: str) -> None:
    means = data[f"{metric}_mean"].to_numpy(float)
    errors = data[f"{metric}_SE"].to_numpy(float)
    labels = data["label"].tolist()
    colors = ["#718096", "#63b3ed", "#2f855a", "#c47f44"]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, alpha=0.9)
    ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#1a202c", capsize=7, lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"Mean outer-fold {metric}")
    ax.set_title(f"Final context comparison: grouped {metric}", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    if metric == "R2":
        ax.axhline(0, color="black", lw=1)
        ax.set_ylim(min(means - errors) - 0.04, 0.02)
        note = "Higher is better; negative R² reflects heterogeneous held-out attempt groups."
    else:
        ax.set_ylim(0, max(means + errors) * 1.18)
        note = "Lower is better. Error bars are SE across five outer attempt-group folds."
    for i, (mean, error) in enumerate(zip(means, errors, strict=True)):
        if metric == "R2":
            ax.text(i, mean + error + 0.008, f"{mean:.4f}", ha="center")
        else:
            ax.text(i, mean * 0.55, f"{mean:.4f}", ha="center", color="white", weight="bold")
    fig.text(0.5, 0.015, f"N=84 turns, 14 attempts. {note}", ha="center", fontsize=10)
    fig.subplots_adjust(bottom=0.15)
    save(fig, filename)


def coverage_figure(coverage: pd.DataFrame) -> None:
    rows = coverage.groupby("block", sort=False).first().loc[["S", "K", "L", "H", "Q"]]
    n = rows["real_substantive_observed_N"].to_numpy(int)
    pct = rows["real_substantive_observed_percent"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(rows.index, pct, color=["#2f855a", "#63b3ed", "#e53e3e", "#a0aec0", "#a0aec0"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Included rows with real substantive block data (%)")
    ax.set_title("Pre-action feature-block coverage", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, count, percent in zip(bars, n, pct, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, percent + 2, f"{count}/84\n({percent:.1f}%)", ha="center")
    fig.text(0.5, 0.015, "L coverage is only 3/84; its incremental usefulness is not empirically established.", ha="center", fontsize=10)
    fig.subplots_adjust(bottom=0.14)
    save(fig, "04_context_feature_coverage.png")


def dimension_figure(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    x = data["context_dimension"].to_numpy(float)
    y = data["MAE_mean"].to_numpy(float)
    e = data["MAE_SE"].to_numpy(float)
    colors = ["#718096", "#63b3ed", "#2f855a", "#c47f44"]
    ax.errorbar(x, y, yerr=e, fmt="none", ecolor="#4a5568", capsize=6, lw=1.8)
    ax.scatter(x, y, s=160, c=colors, zorder=3)
    for dim, mean, label in zip(x, y, data["label"], strict=True):
        ax.annotate(f"{label}\n{int(dim)} dims", (dim, mean), xytext=(8, 8), textcoords="offset points")
    ax.set_xlim(2, 24)
    ax.set_ylim(0.13, 0.20)
    ax.set_xlabel("Policy context dimension")
    ax.set_ylabel("Mean outer-fold MAE")
    ax.set_title("Context size versus grouped prediction error", weight="bold")
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.015, "N=84; bars are SE across five attempt-group folds. S+K+L is moderate-sized, not the largest model.", ha="center", fontsize=10)
    fig.subplots_adjust(bottom=0.14)
    save(fig, "05_context_dimension_vs_performance.png")


def reward_figure(rewards: pd.DataFrame) -> None:
    raw = rewards.set_index("reward").loc["RAW_DELTA"]
    head = rewards.set_index("reward").loc["HEADROOM_NORMALIZED"]
    values = [
        abs(float(raw.spearman_absolute_reward_vs_mastery_before_rho)),
        abs(float(head.spearman_absolute_reward_vs_mastery_before_rho)),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.bar(["RAW_DELTA", "HEADROOM_NORMALIZED"], values, color=["#c47f44", "#2f855a"])
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("|Spearman ρ|: |reward| vs mastery_before")
    ax.set_title("Reward selection: dependence on starting mastery", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", weight="bold")
    fig.text(0.5, 0.045, "N=84. HEADROOM preserves sign, is finite and bounded [-1,1], and uses no tuned α/β constants.", ha="center", fontsize=10)
    fig.text(0.5, 0.015, "Reward denotes a BKT posterior-belief update, not measured learning.", ha="center", fontsize=10)
    fig.subplots_adjust(bottom=0.17)
    save(fig, "06_reward_selection.png")


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 14.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title("AdaptMath Turn-LinTS v1 — S+K+L frozen architecture", fontsize=19, weight="bold", pad=15)

    def box(x: float, y: float, w: float, h: float, text: str, color: str, size: int = 10) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", fc=color, ec="#2d3748", lw=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2d3748"))

    box(3, 16.7, 4, 0.75, "Learner message", "#e2e8f0", 11)
    box(3, 15.55, 4, 0.75, "Explicit learner agency\n(request, not inferred state)", "#fff3cd", 10)
    box(3, 14.4, 4, 0.75, "Frozen MD7-R2-TELL-C1 Epoch 2", "#d9ead3", 10)
    arrow(5, 16.7, 5, 16.3); arrow(5, 15.55, 5, 15.15)

    box(0.25, 12.7, 2.8, 1.0, "S (4)\ncurrent pre-action\nMD7 probabilities", "#cfe2f3", 9)
    box(3.6, 12.7, 2.8, 1.0, "K (3)\nmastery_before +\nprevious Δ + missing", "#d9ead3", 9)
    box(6.95, 12.7, 2.8, 1.0, "L (4)\nlearner response t−1\nsignals + missing", "#fce5cd", 9)
    box(3, 11.35, 4, 0.8, "POLICY CONTEXT S+K+L — 11 dimensions", "#b6d7a8", 11)
    arrow(1.65, 12.7, 4.2, 12.15); arrow(5, 12.7, 5, 12.15); arrow(8.35, 12.7, 5.8, 12.15)
    arrow(5, 14.4, 5, 13.7)

    box(0.25, 10.05, 3.0, 0.85, "Diagnostic logging only\nH + previous Q; full 22 fields", "#eeeeee", 9)
    box(3.6, 9.95, 2.8, 1.05, "Warm-start now\nOR future direct\ndisjoint Turn-LinTS", "#d9d2e9", 10)
    box(6.95, 10.05, 2.8, 0.85, "Warm-start behavior\nraw MD7 categorical; no updates", "#fce5cd", 9)
    arrow(5, 11.35, 5, 11.0); arrow(3.25, 10.48, 3.6, 10.48); arrow(6.95, 10.48, 6.4, 10.48)

    stages = [
        (8.8, "Selected move: generic | probing | focus | telling", "#ead1dc"),
        (7.65, "Existing Tutor realization contract", "#d9ead3"),
        (6.5, "Tutor response", "#cfe2f3"),
        (5.1, "Learner response + semantic evidence category", "#f4cccc"),
        (3.7, "Student Modeling / BKT\nvalid knowledge evidence only", "#d9ead3"),
        (2.25, "HEADROOM_NORMALIZED reward\nBKT posterior-belief update, not measured learning", "#cfe2f3"),
        (0.8, "Future immediate LinTS update — LIVE only\nzero updates during randomized warm-start", "#d9d2e9"),
    ]
    for y, text, color in stages:
        box(3, y, 4, 0.8 if "\n" not in text else 0.95, text, color, 9 if "\n" in text else 10)
    arrow(5, 9.95, 5, 9.6)
    # Draw the post-action chain explicitly so no diagnostic branch rejoins it.
    for top, bottom in [(8.8, 8.45), (7.65, 7.3), (6.5, 6.05), (5.1, 4.65), (3.7, 3.2), (2.25, 1.75)]:
        arrow(5, top, 5, bottom)
    box(0.3, 5.9, 2.3, 0.95, "Current MRB1_t\npost-action auxiliary only\nnot context/reward", "#fff3cd", 8)
    arrow(3, 6.85, 2.6, 6.38)
    ax.text(5, 0.25, "Absent: tau gate, eligibility filtering, P1/P2 anchor, handwritten inferred-state action rules.", ha="center", fontsize=9)
    save(fig, "07_final_architecture_skl.png")


def main() -> None:
    verify_original()
    data, bootstrap, coverage, rewards = load_evidence()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    metric_figure(data, "MAE", "01_context_final_comparison.png")
    metric_figure(data, "RMSE", "02_context_final_rmse.png")
    metric_figure(data, "R2", "03_context_final_r2.png")
    coverage_figure(coverage)
    dimension_figure(data)
    reward_figure(rewards)
    architecture_figure()

    by_name = data.set_index("blocks")
    s = by_name.loc["S"]
    sk = by_name.loc["S+K"]
    skl = by_name.loc["S+K+L"]
    full = by_name.loc["S+K+L+H+Q"]
    boot = bootstrap[(bootstrap.first_context == "S+K+L") & (bootstrap.second_context == "S")].set_index("metric")

    amendment = f"""# Turn-LinTS v1 architecture decision amendment: S+K+L

## Status

This is a transparent architecture-design amendment. The original experiment and its one-standard-error selection of **S** remain unchanged under `{ORIGINAL.as_posix()}`.

## Amendment decision

AdaptMath Turn-LinTS v1 freezes **S+K+L** as its policy context with **qualified / limited support**. S+K+L achieved the best nominal grouped MAE/RMSE among tested contexts and provides explicit selector, knowledge-state, and learner-state representation. Its improvement over S was small and uncertainty intervals crossed zero, while substantive L coverage was only 3/84. Therefore the choice must not be interpreted as proof that L improves reward prediction.

## Empirical evidence

| Context | Dimension | MAE ± SE | RMSE ± SE | R² ± SE |
|---|---:|---:|---:|---:|
| S | 4 | {s.MAE_mean:.4f} ± {s.MAE_SE:.4f} | {s.RMSE_mean:.4f} ± {s.RMSE_SE:.4f} | {s.R2_mean:.4f} ± {s.R2_SE:.4f} |
| S+K | 7 | {sk.MAE_mean:.4f} ± {sk.MAE_SE:.4f} | {sk.RMSE_mean:.4f} ± {sk.RMSE_SE:.4f} | {sk.R2_mean:.4f} ± {sk.R2_SE:.4f} |
| S+K+L | 11 | {skl.MAE_mean:.4f} ± {skl.MAE_SE:.4f} | {skl.RMSE_mean:.4f} ± {skl.RMSE_SE:.4f} | {skl.R2_mean:.4f} ± {skl.R2_SE:.4f} |
| Full | 22 | {full.MAE_mean:.4f} ± {full.MAE_SE:.4f} | {full.RMSE_mean:.4f} ± {full.RMSE_SE:.4f} | {full.R2_mean:.4f} ± {full.R2_SE:.4f} |

Attempt-bootstrap S+K+L minus S: MAE mean {boot.loc['MAE'].mean_difference:.5f}, 95% interval [{boot.loc['MAE'].CI95_low:.5f}, {boot.loc['MAE'].CI95_high:+.5f}]; RMSE mean {boot.loc['RMSE'].mean_difference:.5f}, interval [{boot.loc['RMSE'].CI95_low:.5f}, {boot.loc['RMSE'].CI95_high:+.5f}]. Both intervals cross zero; no statistically established superiority is claimed.

## Architecture rationale

- K is retained because S→S+K improved all nominal grouped metrics and adds current mastery plus previous mastery change. This is predictive, not causal, evidence.
- L is retained because it supplies the intended previous learner-response-state representation and avoids changing the policy vector after prospective collection begins. Its substantive coverage was only **3/84**, so L usefulness remains an unresolved empirical limitation.
- H is excluded: S→S+H changed MAE by approximately +0.00234, RMSE by +0.00233, and R² by −0.02929. H may remain diagnostic; no previous-move rule is introduced.
- Q is excluded: adding Q worsened matched MAE by approximately +0.00452 for S+K+H and +0.00418 for S+K+L+H. MRB1 remains auxiliary and is neither policy context nor scalar reward.
- Full context was nominally worse and doubles dimension from 11 to 22 without matched incremental evidence from H/Q.

## Reward

The original experiment-selected reward remains **HEADROOM_NORMALIZED**. It preserves BKT update sign, is finite and bounded [-1,1], uses no tuned α/β constants, and reduced `|ρ|` between reward magnitude and starting mastery from approximately 0.742 to 0.025. It is a **BKT posterior-belief update, not measured learning**.

## Decision boundary

This amendment freezes vector semantics and starts clean randomized collection. It does not fit or activate a LIVE posterior, establish L usefulness, or make causal action-effect claims.

**B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
"""
    amendment_path = HERE / "DECISION_AMENDMENT.md"
    write(amendment_path, amendment)
    amendment_hash = sha(amendment_path)

    manifest = {
        "architecture_version": "adaptmath_turn_lints_v1_skl_final_amendment_1",
        "support_label": "qualified_limited",
        "algorithm": "direct_disjoint_turn_lints_v1",
        "arms": ["generic", "probing", "focus", "telling"],
        "context_preset": "S+K+L",
        "ordered_context_features": FEATURES,
        "context_dimension": 11,
        "diagnostic_logged_context": "S+K+L+H+Q",
        "reward_mode": "headroom_normalized",
        "selector_name": "MD7-R2-TELL-C1 Epoch 2",
        "selector_sha256": "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5",
        "cold_start_behavior_policy": "md7_r2_probability_proportional_v1",
        "tau_gate": "none",
        "action_eligibility": "none",
        "anchor_mode": "none",
        "P1": "inactive",
        "P2": "not_used",
        "handwritten_inferred_state_action_rules": "none",
        "MRB1_role": "auxiliary diagnostic; not frozen policy context or reward",
        "student_model_role": "previous learner-state L context provider plus diagnostics; valid linked BKT update supplies reward",
        "formal_assessment_role": "separate external policy-health evaluation",
        "learner_agency": "explicit requests may override above policy and are non-randomized with null propensity",
        "update_timing": "immediate after valid linked turn-level BKT evidence in future LIVE mode",
        "live_enabled": False,
        "warmstart_posterior_updates_enabled": False,
        "runtime_lineage": "adaptive-math-tutor/backend/runtime/turn_lints_randomized_warmstart_skl_final_v1",
        "experiment_evidence_paths": [
            "pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1/context_ablation_results.csv",
            "pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1/context_bootstrap_comparisons.csv",
            "pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1/context_feature_coverage.csv",
            "pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1/reward_experiment_results.csv",
        ],
        "original_evidence_hashes": ORIGINAL_HASHES,
        "decision_amendment_sha256": amendment_hash,
        "manifest_hash_note": "Final manifest SHA-256 is stored in FINAL_ARCHITECTURE.sha256 to avoid a self-referential hash.",
        "empirical_limitation": "S+K+L advantage over S is nominal and uncertain; bootstrap intervals cross zero and real L coverage is 3/84.",
        "decision_label": "B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT",
    }
    manifest_path = HERE / "FINAL_ARCHITECTURE.json"
    write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    manifest_hash = sha(manifest_path)
    write(HERE / "FINAL_ARCHITECTURE.sha256", f"{manifest_hash}  FINAL_ARCHITECTURE.json")
    write(HERE / "DECISION_AMENDMENT.sha256", f"{amendment_hash}  DECISION_AMENDMENT.md")

    final_md = f"""# AdaptMath Turn-LinTS v1 final S+K+L architecture

- Architecture decision: **S+K+L**, 11 ordered pre-action features, qualified/limited support
- Algorithm: direct disjoint linear Thompson Sampling
- Arms: generic, probing, focus, telling
- Reward: HEADROOM_NORMALIZED BKT posterior-belief update
- Cold start: raw MD7 proportional randomized warm-start with exact propensity logging
- Diagnostic logging: full S+K+L+H+Q snapshot; H/Q never enter the frozen policy vector
- Explicit agency: above-policy, non-randomized override with null propensity
- No tau gate, filtering, P1/P2 anchor, or handwritten inferred-state action rules
- LIVE: disabled; warm-start posterior update count must remain zero

The original S-selection experiment remains unchanged. S+K+L was the nominal predictive winner, but its advantage over S was small and uncertain and L coverage was only 3/84. L usefulness remains unresolved.

Manifest SHA-256: `{manifest_hash}`

**B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
"""
    write(HERE / "FINAL_ARCHITECTURE.md", final_md)

    notes = """# Implementation notes

- Frozen constants are defined in `src/self_improvement/turn_lints_architecture_v1.py`.
- Policy construction uses exactly S+K+L and 11 ordered values.
- Runtime event records contain separate `policy_context` and `diagnostic_logged_context` objects. The diagnostic object uses the existing causal builder and contains previous H/Q only.
- Missing L is represented as three zeros plus `previous_learner_signals_missing=1`; real all-zero signals use indicator 0.
- Pure acknowledgement/interaction-management evidence is categorized semantically and cannot trigger behavioural-proxy BKT updates.
- Explicit agency uses decision/treatment source `learner_agency`, `randomized_assignment=false`, and `behavior_propensity=null`.
- The safe default remains SHADOW. `start-adaptmath-local.ps1 -FinalSKLWarmstart` explicitly selects the final randomized-warm-start lineage.
- The S-only and prior drifted state files are not migrated. The S+K+L lineage starts from fresh 11×11 identity A matrices, zero b vectors, and zero updates.
- No LIVE posterior was fitted or activated.
"""
    write(HERE / "IMPLEMENTATION_NOTES.md", notes)

    refs = "# Analysis references\n\n" + "\n".join(
        f"- `{name}` — SHA-256 `{digest}`" for name, digest in ORIGINAL_HASHES.items()
    ) + "\n\nThe historical folder was verified before and after amendment generation and was not rewritten.\n"
    write(HERE / "analysis_references.md", refs)

    summary = {
        "decision": "S+K+L",
        "dimension": 11,
        "support": "qualified_limited",
        "ordered_features": FEATURES,
        "metrics": {
            row.blocks: {
                "dimension": int(row.context_dimension),
                "MAE_mean": row.MAE_mean,
                "MAE_SE": row.MAE_SE,
                "RMSE_mean": row.RMSE_mean,
                "RMSE_SE": row.RMSE_SE,
                "R2_mean": row.R2_mean,
                "R2_SE": row.R2_SE,
            }
            for row in data.itertuples()
        },
        "bootstrap_skl_minus_s": {
            metric: {
                "mean_difference": float(boot.loc[metric].mean_difference),
                "CI95_low": float(boot.loc[metric].CI95_low),
                "CI95_high": float(boot.loc[metric].CI95_high),
            }
            for metric in ("MAE", "RMSE")
        },
        "real_coverage": {"S": 84, "K": 70, "L": 3, "H": 70, "Q": 70, "N": 84},
        "reward": "HEADROOM_NORMALIZED",
        "original_manifest_sha256": ORIGINAL_HASHES["FINAL_ARCHITECTURE.json"],
        "decision_amendment_sha256": amendment_hash,
        "manifest_sha256": manifest_hash,
        "decision_label": "B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT",
        "safety": {
            "neural_training": 0,
            "protected_mathdial_test_use": 0,
            "mrbench_v3_official_test_use": 0,
            "authoritative_historical_rewrites": 0,
            "fabricated_L_signals": 0,
            "fabricated_rewards": 0,
            "fabricated_propensities": 0,
            "handwritten_inferred_rules": 0,
            "LIVE_updates": 0,
        },
    }
    write(HERE / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    verify_original()
    print(json.dumps({"manifest_sha256": manifest_hash, "amendment_sha256": amendment_hash}, indent=2))


if __name__ == "__main__":
    main()
