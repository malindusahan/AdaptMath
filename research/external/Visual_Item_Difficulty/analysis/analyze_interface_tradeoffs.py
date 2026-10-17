#!/usr/bin/env python3
"""Analyze efficiency and item-level tradeoffs for Q+D versus I+Q.

Reads the three-seed final-mode predictions and the recorded job timings; no
training is involved.  Outputs are written under
``results/analysis/interface_tradeoffs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/interface_tradeoffs"
N_BOOT = 10_000
BOOT_SEED = 20260730

SYSTEMS = {
    "Q+D": ROOT / "results/main/llm/meta-llama_Llama-3.1-8B_vd_qwen25vl7b",
    "I+Q": ROOT / "results/main/vlm/Qwen_Qwen2.5-VL-7B-Instruct_IQ_ablMlp",
}

JOBS = {
    "Q+D": [
        "715_dB_llama31-8b_QDvl7b_s17.job",
        "716_dB_llama31-8b_QDvl7b_s42.job",
        "717_dB_llama31-8b_QDvl7b_s2026.job",
    ],
    "I+Q": [
        "806_abl_qwen25vl7b_ablMlp_s17.job",
        "807_abl_qwen25vl7b_ablMlp_s42.job",
        "808_abl_qwen25vl7b_ablMlp_s2026.job",
    ],
}

SEEDS = (17, 42, 2026)


def rmse(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gold) ** 2)))


def load_seed(run_dir: Path, seed: int) -> pd.DataFrame:
    path = run_dir / f"seed{seed}" / "test_predictions.csv"
    return pd.read_csv(path).sort_values("item_id").reset_index(drop=True)


def load_ensemble(run_dir: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    runs = {seed: load_seed(run_dir, seed) for seed in SEEDS}
    ref = runs[SEEDS[0]]
    for seed, run in runs.items():
        if not np.array_equal(ref["item_id"].to_numpy(), run["item_id"].to_numpy()):
            raise RuntimeError(f"Item mismatch for {run_dir}, seed {seed}")
        if not np.allclose(ref["gold_beta"].to_numpy(), run["gold_beta"].to_numpy()):
            raise RuntimeError(f"Gold-label mismatch for {run_dir}, seed {seed}")
    ensemble = ref[["item_id", "gold_beta", "has_figure", "q_char_len", "d_char_len"]].copy()
    ensemble["pred_beta"] = np.mean(
        [runs[seed]["pred_beta"].to_numpy() for seed in SEEDS], axis=0
    )
    return ensemble, runs


def bootstrap_rmse_delta(
    gold: np.ndarray,
    qd: np.ndarray,
    iq: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(gold)
    indices = rng.integers(0, n, size=(N_BOOT, n))
    qd_sq = (qd - gold) ** 2
    iq_sq = (iq - gold) ** 2
    delta = np.sqrt(iq_sq[indices].mean(axis=1)) - np.sqrt(
        qd_sq[indices].mean(axis=1)
    )
    low, high = np.percentile(delta, [2.5, 97.5])
    return float(low), float(high)


def summarize_group(
    part: pd.DataFrame,
    grouping: str,
    group: str,
    rng: np.random.Generator,
) -> dict:
    gold = part["gold_beta"].to_numpy()
    qd = part["pred_QD"].to_numpy()
    iq = part["pred_IQ"].to_numpy()
    ci_low, ci_high = bootstrap_rmse_delta(gold, qd, iq, rng)
    decisive_iq = int((part["abs_advantage_IQ"] >= 0.10).sum())
    decisive_qd = int((part["abs_advantage_IQ"] <= -0.10).sum())
    return {
        "grouping": grouping,
        "group": group,
        "n": len(part),
        "IQ_wins": int((part["winner"] == "I+Q").sum()),
        "QD_wins": int((part["winner"] == "Q+D").sum()),
        "IQ_win_rate": float((part["winner"] == "I+Q").mean()),
        "unanimous_IQ_wins_across_seeds": int(
            (part["IQ_seed_wins_out_of_3"] == 3).sum()
        ),
        "unanimous_QD_wins_across_seeds": int(
            (part["IQ_seed_wins_out_of_3"] == 0).sum()
        ),
        "mixed_winner_across_seeds": int(
            part["IQ_seed_wins_out_of_3"].isin([1, 2]).sum()
        ),
        "IQ_wins_by_at_least_0.10_AE": decisive_iq,
        "QD_wins_by_at_least_0.10_AE": decisive_qd,
        "QD_RMSE": rmse(gold, qd),
        "IQ_RMSE": rmse(gold, iq),
        "IQ_minus_QD_RMSE": rmse(gold, iq) - rmse(gold, qd),
        "IQ_minus_QD_RMSE_CI_low": ci_low,
        "IQ_minus_QD_RMSE_CI_high": ci_high,
        "QD_MAE": float(np.mean(np.abs(qd - gold))),
        "IQ_MAE": float(np.mean(np.abs(iq - gold))),
        "mean_abs_advantage_IQ": float(part["abs_advantage_IQ"].mean()),
        "median_abs_advantage_IQ": float(part["abs_advantage_IQ"].median()),
    }


def tensor_count(path: Path) -> int:
    """Count trainable parameters in a saved adapter / head state dict.

    The trained checkpoints are not redistributed with this repository (see
    README); re-run the training scripts to regenerate them, or read the
    already-computed counts from results/analysis/interface_tradeoffs/.
    """
    if not path.exists():
        # Trained checkpoints are not redistributed (~9 GB). Fall back to the
        # counts precomputed from them; rerun src/*_lora_regression.py to
        # regenerate the checkpoints themselves.
        cache = json.loads((ROOT / "results/adapter_param_counts.json").read_text())
        key = f"{path.parent.parent.name}"
        for run_key, entry in cache.items():
            if run_key.endswith(f"{key}/{path.parent.name}") and path.name in entry:
                return entry[path.name]
        raise FileNotFoundError(
            f"{path} not found and no precomputed count in "
            "results/adapter_param_counts.json.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(value.numel() for value in state.values() if hasattr(value, "numel")))


def efficiency_table() -> pd.DataFrame:
    ledger = pd.read_csv(
        ROOT / "results/run_timings.csv")
    rows = []
    for system, run_dir in SYSTEMS.items():
        selected = ledger[ledger["job"].isin(JOBS[system])].copy()
        if len(selected) != 3 or (selected["return_code"] != 0).any():
            raise RuntimeError(f"Incomplete ledger entries for {system}")

        seed42_dir = run_dir / "seed42"
        with (seed42_dir / "results.json").open() as handle:
            result = json.load(handle)
        args = result["provenance"]["args"]
        adapter_params = tensor_count(seed42_dir / "lora_adapter.pt")
        head_params = tensor_count(seed42_dir / "regression_head.pt")
        epochs = int(result["epochs"])
        durations = selected["duration_seconds"].to_numpy(dtype=float)
        # Q+D textualization is cached and shared across seeds.  I+Q processes
        # every image once per epoch and once at held-out evaluation.
        if system == "Q+D":
            image_presentations_per_final_run = 0
            one_time_textualization_images = 401
        else:
            image_presentations_per_final_run = 580 * epochs + 145
            one_time_textualization_images = 0
        rows.append(
            {
                "system": system,
                "model": result["model"],
                "route": result["route"],
                "epochs": epochs,
                "batch_size": int(args["batch_size"]),
                "gradient_accumulation": int(args["grad_accum"]),
                "duration_seed17_seconds": int(
                    selected.loc[selected["job"] == JOBS[system][0], "duration_seconds"].iloc[0]
                ),
                "duration_seed42_seconds": int(
                    selected.loc[selected["job"] == JOBS[system][1], "duration_seconds"].iloc[0]
                ),
                "duration_seed2026_seconds": int(
                    selected.loc[selected["job"] == JOBS[system][2], "duration_seconds"].iloc[0]
                ),
                "mean_job_seconds": float(durations.mean()),
                "sd_job_seconds": float(durations.std(ddof=1)),
                "mean_seconds_per_epoch": float(durations.mean() / epochs),
                "saved_adapter_params": adapter_params,
                "saved_head_params": head_params,
                "saved_trainable_params_total": adapter_params + head_params,
                "image_presentations_per_final_run": image_presentations_per_final_run,
                "one_time_textualization_images": one_time_textualization_images,
                "timing_scope": "model load + final fit + test inference + artifact save",
            }
        )
    out = pd.DataFrame(rows)
    qd = out.loc[out["system"] == "Q+D"].iloc[0]
    iq = out.loc[out["system"] == "I+Q"].iloc[0]
    out["job_time_ratio_vs_QD"] = out["mean_job_seconds"] / qd["mean_job_seconds"]
    out["epoch_normalized_time_ratio_vs_QD"] = (
        out["mean_seconds_per_epoch"] / qd["mean_seconds_per_epoch"]
    )
    out["trainable_param_ratio_vs_QD"] = (
        out["saved_trainable_params_total"] / qd["saved_trainable_params_total"]
    )
    # Keep the explicit lookup so accidental row-order changes cannot alter ratios.
    assert iq["mean_job_seconds"] > qd["mean_job_seconds"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensembles = {}
    seed_runs = {}
    for system, path in SYSTEMS.items():
        ensembles[system], seed_runs[system] = load_ensemble(path)

    merged = ensembles["Q+D"].rename(columns={"pred_beta": "pred_QD"})
    merged = merged.merge(
        ensembles["I+Q"][["item_id", "pred_beta"]].rename(columns={"pred_beta": "pred_IQ"}),
        on="item_id",
        validate="one_to_one",
    )

    taxonomy_path = ROOT / "results/analysis/visual_taxonomy_item_predictions.csv"
    taxonomy = pd.read_csv(taxonomy_path)[["item_id", "category"]]
    merged = merged.merge(taxonomy, on="item_id", validate="one_to_one")
    metadata = pd.read_csv(ROOT / "data/items.csv")[
        ["QuestionId", "SE_beta", "n_answers", "correct_rate", "text", "visual_description"]
    ].rename(columns={"QuestionId": "item_id"})
    merged = merged.merge(metadata, on="item_id", validate="one_to_one")

    merged["QD_abs_error"] = np.abs(merged["pred_QD"] - merged["gold_beta"])
    merged["IQ_abs_error"] = np.abs(merged["pred_IQ"] - merged["gold_beta"])
    merged["abs_advantage_IQ"] = merged["QD_abs_error"] - merged["IQ_abs_error"]
    merged["winner"] = np.where(merged["abs_advantage_IQ"] > 0, "I+Q", "Q+D")
    merged["difficulty_quintile"] = pd.qcut(
        merged["gold_beta"],
        5,
        labels=["easiest", "easy", "middle", "hard", "hardest"],
    )
    merged["figure_group"] = np.where(
        merged["has_figure"], "parser-visual", "parser-nonvisual"
    )
    visual_mask = merged["d_char_len"] > 0
    merged["description_length_tertile"] = "no-description"
    merged.loc[visual_mask, "description_length_tertile"] = pd.qcut(
        merged.loc[visual_mask, "d_char_len"],
        3,
        labels=["short-D", "medium-D", "long-D"],
    ).astype(str)

    # Stability: number of matched seeds on which I+Q has lower absolute error.
    iq_seed_wins = np.zeros(len(merged), dtype=int)
    for seed in SEEDS:
        qd = seed_runs["Q+D"][seed].sort_values("item_id")["pred_beta"].to_numpy()
        iq = seed_runs["I+Q"][seed].sort_values("item_id")["pred_beta"].to_numpy()
        gold = merged.sort_values("item_id")["gold_beta"].to_numpy()
        iq_seed_wins += (np.abs(iq - gold) < np.abs(qd - gold)).astype(int)
    seed_stability = pd.DataFrame(
        {
            "item_id": merged.sort_values("item_id")["item_id"].to_numpy(),
            "IQ_seed_wins_out_of_3": iq_seed_wins,
        }
    )
    merged = merged.merge(seed_stability, on="item_id", validate="one_to_one")

    rng = np.random.default_rng(BOOT_SEED)
    summaries = [summarize_group(merged, "overall", "all", rng)]
    for grouping in (
        "category",
        "figure_group",
        "difficulty_quintile",
        "description_length_tertile",
    ):
        for group, part in merged.groupby(grouping, observed=True, sort=False):
            summaries.append(summarize_group(part, grouping, str(group), rng))
    summary = pd.DataFrame(summaries)

    correlations = []
    for feature in ("gold_beta", "SE_beta", "n_answers", "q_char_len"):
        correlations.append(
            {
                "subset": "all",
                "feature": feature,
                "pearson_with_abs_advantage_IQ": merged[
                    [feature, "abs_advantage_IQ"]
                ].corr(method="pearson").iloc[0, 1],
                "spearman_with_abs_advantage_IQ": merged[
                    [feature, "abs_advantage_IQ"]
                ].corr(method="spearman").iloc[0, 1],
                "n": len(merged),
            }
        )
    visual = merged[visual_mask]
    correlations.append(
        {
            "subset": "parser-visual",
            "feature": "d_char_len",
            "pearson_with_abs_advantage_IQ": visual[
                ["d_char_len", "abs_advantage_IQ"]
            ].corr(method="pearson").iloc[0, 1],
            "spearman_with_abs_advantage_IQ": visual[
                ["d_char_len", "abs_advantage_IQ"]
            ].corr(method="spearman").iloc[0, 1],
            "n": len(visual),
        }
    )

    efficiency = efficiency_table()
    merged.sort_values("abs_advantage_IQ", ascending=False).to_csv(
        OUT / "interface_winloss_items.csv", index=False
    )
    summary.to_csv(OUT / "interface_winloss_summary.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "interface_winloss_correlations.csv", index=False)
    efficiency.to_csv(OUT / "interface_efficiency.csv", index=False)
    merged.nlargest(15, "abs_advantage_IQ").to_csv(OUT / "top_IQ_wins.csv", index=False)
    merged.nsmallest(15, "abs_advantage_IQ").to_csv(OUT / "top_QD_wins.csv", index=False)

    overall = summary.iloc[0]
    qd_eff = efficiency[efficiency["system"] == "Q+D"].iloc[0]
    iq_eff = efficiency[efficiency["system"] == "I+Q"].iloc[0]
    report = f"""# Q+D versus I+Q

## Workflow cost

| System | Three job times (s) | Mean | Per epoch | Saved trainable tensors | Image handling |
|---|---:|---:|---:|---:|---|
| Q+D | {int(qd_eff.duration_seed17_seconds)}, {int(qd_eff.duration_seed42_seconds)}, {int(qd_eff.duration_seed2026_seconds)} | {qd_eff.mean_job_seconds:.1f}s | {qd_eff.mean_seconds_per_epoch:.1f}s | {int(qd_eff.saved_trainable_params_total):,} | 401 images textualized once, then cached |
| I+Q | {int(iq_eff.duration_seed17_seconds)}, {int(iq_eff.duration_seed42_seconds)}, {int(iq_eff.duration_seed2026_seconds)} | {iq_eff.mean_job_seconds:.1f}s | {iq_eff.mean_seconds_per_epoch:.1f}s | {int(iq_eff.saved_trainable_params_total):,} | {int(iq_eff.image_presentations_per_final_run):,} image presentations per final run |

The complete I+Q job is {iq_eff.job_time_ratio_vs_QD:.2f}x the Q+D job; after
normalizing by the different epoch counts, it is {iq_eff.epoch_normalized_time_ratio_vs_QD:.2f}x
per epoch.  Timings cover model loading, final fitting, test inference, and
artifact saving on the same host and adjacent GPU slots.  These timings do not
separate training from inference, do not record peak memory, and exclude the
one-time description-generation run, so they should not be read as a
component-level cost breakdown.

## Overall item-level comparison

| n | I+Q wins | Q+D wins | Q+D RMSE | I+Q RMSE | Delta RMSE (I+Q - Q+D) | 95% paired bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|
| {int(overall.n)} | {int(overall.IQ_wins)} | {int(overall.QD_wins)} | {overall.QD_RMSE:.4f} | {overall.IQ_RMSE:.4f} | {overall.IQ_minus_QD_RMSE:+.4f} | [{overall.IQ_minus_QD_RMSE_CI_low:+.4f}, {overall.IQ_minus_QD_RMSE_CI_high:+.4f}] |

An item-level win means lower absolute error for that item.  The win count
measures frequency, while RMSE also reflects the magnitude of a smaller number
of large wins or losses.  The route preference is unanimous across all three
matched seeds for {int(overall.unanimous_IQ_wins_across_seeds)} I+Q items and
{int(overall.unanimous_QD_wins_across_seeds)} Q+D items; the remaining
{int(overall.mixed_winner_across_seeds)} items change winner across seeds.
See the CSV outputs for taxonomy, figure status, difficulty quintile,
description-length, seed stability, and the largest wins for each interface.
"""
    (OUT / "README.md").write_text(report)

    print(report)
    print("\nGrouped win/loss summary:")
    print(
        summary[
            [
                "grouping",
                "group",
                "n",
                "IQ_wins",
                "QD_wins",
                "IQ_win_rate",
                "IQ_minus_QD_RMSE",
                "mean_abs_advantage_IQ",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
