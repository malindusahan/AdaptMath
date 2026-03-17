#!/usr/bin/env python3
"""Deterministic item-type analysis for the three headline systems.

The taxonomy uses only the verified parser output and is intentionally coarse.
Categories are mutually exclusive and assigned in this order:
visual answer options; plots/tables/number lines; geometry diagrams; other
parser-flagged visuals; parser-nonvisual.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARSES = ROOT / "data/question_parsed.json"
OUT_DIR = ROOT / "results/analysis"
N_BOOT = 10_000
SEED = 20260728

SYSTEM_DIRS = {
    "Q": ROOT / "results/main/llm/meta-llama_Llama-3.1-8B_text_ablMlp",
    "Q+D_VL7B": ROOT / "results/main/llm/meta-llama_Llama-3.1-8B_vd_qwen25vl7b",
    "I+Q": ROOT / "results/main/vlm/Qwen_Qwen2.5-VL-7B-Instruct_IQ_ablMlp",
}

PLOT_TABLE_TERMS = (
    "graph", "chart", "table", "coordinate", "axis", "axes", "number line",
    "bar chart", "line plot", "scatter", "grid",
)
GEOMETRY_TERMS = (
    "triangle", "angle", "circle", "quadrilateral", "rectangle", "square",
    "polygon", "trapez", "parallelogram", "rhombus", "shape", "geometric",
    "line segment", "perpendicular", "parallel", "radius", "diameter",
    "vertex", "vertices", "prism", "cube", "cuboid", "net",
)


def classify(record: dict) -> str:
    if not record.get("has_figure", False):
        return "Parser-nonvisual"

    choices = " ".join(str(v).lower() for v in record.get("choices", {}).values())
    description = str(record.get("figure_description", "")).lower()
    combined = f"{choices} {description}"

    if "[image:" in choices or "answer choice" in description or "answer option" in description:
        return "Visual answer options"
    if any(term in combined for term in PLOT_TABLE_TERMS):
        return "Plots/tables/number lines"
    if any(term in combined for term in GEOMETRY_TERMS):
        return "Geometry diagrams"
    return "Other visual"


def load_ensemble(run_dir: Path) -> pd.DataFrame:
    paths = sorted(run_dir.glob("seed*/test_predictions.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three seeds in {run_dir}, found {len(paths)}")
    runs = [pd.read_csv(path).sort_values("item_id").reset_index(drop=True) for path in paths]
    item_ids = runs[0]["item_id"].to_numpy()
    gold = runs[0]["gold_beta"].to_numpy()
    for run in runs[1:]:
        if not np.array_equal(item_ids, run["item_id"].to_numpy()):
            raise RuntimeError(f"Item order mismatch in {run_dir}")
        if not np.allclose(gold, run["gold_beta"].to_numpy()):
            raise RuntimeError(f"Gold-label mismatch in {run_dir}")
    return pd.DataFrame({
        "item_id": item_ids,
        "gold_beta": gold,
        "pred_beta": np.mean([run["pred_beta"].to_numpy() for run in runs], axis=0),
    })


def rmse(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gold) ** 2)))


def paired_ci(gold: np.ndarray, base: np.ndarray, other: np.ndarray, rng: np.random.Generator):
    n = len(gold)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    base_sq = (base - gold) ** 2
    other_sq = (other - gold) ** 2
    delta = np.sqrt(other_sq[idx].mean(axis=1)) - np.sqrt(base_sq[idx].mean(axis=1))
    return np.percentile(delta, [2.5, 97.5])


def main() -> None:
    with PARSES.open() as handle:
        parsed = json.load(handle)
    taxonomy = {int(record["question_id"]): classify(record) for record in parsed}

    frames = {name: load_ensemble(path) for name, path in SYSTEM_DIRS.items()}
    merged = frames["Q"].rename(columns={"pred_beta": "pred_Q"})
    for name in ("Q+D_VL7B", "I+Q"):
        merged = merged.merge(
            frames[name][["item_id", "pred_beta"]].rename(columns={"pred_beta": f"pred_{name}"}),
            on="item_id",
            validate="one_to_one",
        )
    merged["category"] = merged["item_id"].map(taxonomy)
    if merged["category"].isna().any():
        raise RuntimeError("Missing taxonomy labels for test items")

    category_order = [
        "Visual answer options",
        "Plots/tables/number lines",
        "Geometry diagrams",
        "Other visual",
        "Parser-nonvisual",
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for category in category_order:
        part = merged[merged["category"] == category]
        gold = part["gold_beta"].to_numpy()
        q = part["pred_Q"].to_numpy()
        qd = part["pred_Q+D_VL7B"].to_numpy()
        iq = part["pred_I+Q"].to_numpy()
        q_rmse, qd_rmse, iq_rmse = rmse(gold, q), rmse(gold, qd), rmse(gold, iq)
        qd_ci = paired_ci(gold, q, qd, rng)
        iq_ci = paired_ci(gold, q, iq, rng)
        rows.append({
            "category": category,
            "n": len(part),
            "Q_RMSE": q_rmse,
            "Q+D_VL7B_RMSE": qd_rmse,
            "I+Q_RMSE": iq_rmse,
            "Q+D_minus_Q": qd_rmse - q_rmse,
            "Q+D_minus_Q_CI_low": qd_ci[0],
            "Q+D_minus_Q_CI_high": qd_ci[1],
            "I+Q_minus_Q": iq_rmse - q_rmse,
            "I+Q_minus_Q_CI_low": iq_ci[0],
            "I+Q_minus_Q_CI_high": iq_ci[1],
        })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "visual_taxonomy.csv", index=False)

    abs_q = np.abs(merged["pred_Q"] - merged["gold_beta"])
    abs_qd = np.abs(merged["pred_Q+D_VL7B"] - merged["gold_beta"])
    abs_iq = np.abs(merged["pred_I+Q"] - merged["gold_beta"])
    merged["best_vision_abs_error"] = np.minimum(abs_qd, abs_iq)
    merged["vision_gain_abs_error"] = abs_q - merged["best_vision_abs_error"]
    merged["better_route"] = np.where(abs_qd <= abs_iq, "Q+D_VL7B", "I+Q")
    merged.to_csv(OUT_DIR / "visual_taxonomy_item_predictions.csv", index=False)
    cases = merged.sort_values("vision_gain_abs_error", ascending=False).head(20)
    cases.to_csv(OUT_DIR / "visual_taxonomy_top_cases.csv", index=False)

    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTop cases by reduction in absolute error:")
    print(cases[["item_id", "category", "vision_gain_abs_error", "better_route"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
