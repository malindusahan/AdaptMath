from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_TOP_LEVEL = (
    "skill",
    "mastery_before",
    "mastery_after",
    "delta_mastery",
    "evaluator_result",
    "reasoning",
    "uncertainty",
    "clarification",
    "repeated_misunderstanding",
)

ZERO_TOL = 1e-12


def load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                data = data["records"]
            else:
                raise ValueError(
                    "JSON input must be a list of records or a mapping with a "
                    "'records' list."
                )
        if not isinstance(data, list):
            raise ValueError("JSON input must contain a list of records.")
        return data

    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"JSONL line {line_number} is not an object."
                    )
                records.append(obj)
        return records

    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")

    raise ValueError("Supported input formats: .json, .jsonl/.ndjson, .csv")


def flatten_record(record: dict, index: int) -> dict:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in record]
    if missing:
        raise ValueError(f"Record {index} is missing required fields: {missing}")

    evaluator = record["evaluator_result"]

    if isinstance(evaluator, str):
        try:
            evaluator = json.loads(evaluator)
        except json.JSONDecodeError:
            evaluator = {
                "correctness": evaluator,
                "confidence": np.nan,
                "source": "",
            }

    if not isinstance(evaluator, dict):
        raise ValueError(f"Record {index}: evaluator_result must be an object.")

    correctness = evaluator.get("correctness", "missing")
    confidence = evaluator.get(
        "confidence",
        evaluator.get("evaluator_confidence", np.nan),
    )
    source = evaluator.get(
        "source",
        evaluator.get("evaluator_source", ""),
    )

    row = {
        "record_index": index,
        "skill": str(record["skill"]),
        "mastery_before": float(record["mastery_before"]),
        "mastery_after": float(record["mastery_after"]),
        "delta_mastery": float(record["delta_mastery"]),
        "evaluator_correctness": str(correctness).lower(),
        "evaluator_confidence": (
            float(confidence)
            if confidence is not None and not pd.isna(confidence)
            else np.nan
        ),
        "evaluator_source": str(source),
        "reasoning": float(record["reasoning"]),
        "uncertainty": float(record["uncertainty"]),
        "clarification": float(record["clarification"]),
        "repeated_misunderstanding": bool(record["repeated_misunderstanding"]),
    }

    for name in (
        "mastery_before",
        "mastery_after",
        "reasoning",
        "uncertainty",
        "clarification",
    ):
        value = row[name]
        if not math.isfinite(value):
            raise ValueError(f"Record {index}: {name} must be finite.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Record {index}: {name} must be in [0,1], got {value}."
            )

    if not math.isfinite(row["delta_mastery"]):
        raise ValueError(f"Record {index}: delta_mastery must be finite.")

    return row


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 20260824,
    n_bootstrap: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the scale and behavior of Malindu-provided delta_mastery "
            "before using it as a LinTS reward."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .json, .jsonl/.ndjson, or .csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/self_improvement/delta_mastery_reward_scale_audit"
        ),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise ValueError("Input contains no records.")

    rows = [
        flatten_record(record, i)
        for i, record in enumerate(records, start=1)
    ]
    df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    df["recomputed_delta"] = df["mastery_after"] - df["mastery_before"]
    df["delta_error"] = df["delta_mastery"] - df["recomputed_delta"]

    max_abs_delta_error = float(df["delta_error"].abs().max())
    inconsistent_count = int((df["delta_error"].abs() > 1e-9).sum())

    delta = df["delta_mastery"].to_numpy(dtype=np.float64)
    positive = delta > ZERO_TOL
    negative = delta < -ZERO_TOL
    zero = ~(positive | negative)

    mean_ci = bootstrap_mean_ci(delta)

    quantiles = df["delta_mastery"].quantile(
        [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    )

    evaluator_summary = (
        df.groupby("evaluator_correctness", dropna=False, as_index=False)
        .agg(
            n=("delta_mastery", "size"),
            mean_delta=("delta_mastery", "mean"),
            median_delta=("delta_mastery", "median"),
            mean_abs_delta=("delta_mastery", lambda x: x.abs().mean()),
            positive_fraction=(
                "delta_mastery",
                lambda x: float((x > ZERO_TOL).mean()),
            ),
            negative_fraction=(
                "delta_mastery",
                lambda x: float((x < -ZERO_TOL).mean()),
            ),
        )
        .sort_values("evaluator_correctness")
    )

    repeated_summary = (
        df.groupby("repeated_misunderstanding", as_index=False)
        .agg(
            n=("delta_mastery", "size"),
            mean_delta=("delta_mastery", "mean"),
            median_delta=("delta_mastery", "median"),
            mean_abs_delta=("delta_mastery", lambda x: x.abs().mean()),
        )
    )

    corr_columns = [
        "delta_mastery",
        "mastery_before",
        "reasoning",
        "uncertainty",
        "clarification",
    ]
    correlations = df[corr_columns].corr(method="spearman")

    summary = {
        "n_records": int(len(df)),
        "n_skills": int(df["skill"].nunique()),
        "delta_consistency": {
            "max_abs_error_vs_after_minus_before": max_abs_delta_error,
            "count_abs_error_gt_1e-9": inconsistent_count,
        },
        "delta_mastery": {
            "mean": float(df["delta_mastery"].mean()),
            "mean_bootstrap_ci95": [mean_ci[0], mean_ci[1]],
            "std": float(df["delta_mastery"].std(ddof=1))
            if len(df) > 1
            else 0.0,
            "mean_absolute": float(df["delta_mastery"].abs().mean()),
            "positive_fraction": float(positive.mean()),
            "zero_fraction": float(zero.mean()),
            "negative_fraction": float(negative.mean()),
            "quantiles": {
                str(float(q)): float(v)
                for q, v in quantiles.items()
            },
        },
        "input_file": str(args.input),
        "zero_tolerance": ZERO_TOL,
    }

    df.to_csv(args.output_dir / "validated_records.csv", index=False)
    evaluator_summary.to_csv(
        args.output_dir / "delta_by_evaluator.csv",
        index=False,
    )
    repeated_summary.to_csv(
        args.output_dir / "delta_by_repeated_misunderstanding.csv",
        index=False,
    )
    correlations.to_csv(args.output_dir / "spearman_correlations.csv")

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["delta_mastery"], bins=30)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("delta_mastery")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of delta_mastery")
    fig.tight_layout()
    fig.savefig(figure_dir / "delta_mastery_histogram.png", dpi=180)
    plt.close(fig)

    preferred_order = ["correct", "partial", "incorrect", "unknown"]
    present = set(df["evaluator_correctness"])
    labels = [x for x in preferred_order if x in present] + sorted(
        present - set(preferred_order)
    )

    if labels:
        values = [
            df.loc[
                df["evaluator_correctness"] == label,
                "delta_mastery",
            ].to_numpy()
            for label in labels
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(values, tick_labels=labels, showfliers=True)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Evaluator correctness")
        ax.set_ylabel("delta_mastery")
        ax.set_title("delta_mastery by evaluator result")
        fig.tight_layout()
        fig.savefig(
            figure_dir / "delta_by_evaluator_boxplot.png",
            dpi=180,
        )
        plt.close(fig)

    print("=" * 78)
    print("DELTA MASTERY REWARD-SCALE AUDIT")
    print("=" * 78)
    print(f"Input records: {len(df)}")
    print(f"Distinct skills: {df['skill'].nunique()}")
    print()
    print("Consistency")
    print(
        "  max |provided delta - (after-before)|:",
        f"{max_abs_delta_error:.12g}",
    )
    print("  inconsistent > 1e-9:", inconsistent_count)
    print()
    print("Scale")
    print("  mean:", f"{df['delta_mastery'].mean():.6f}")
    print(
        "  mean 95% bootstrap CI:",
        f"[{mean_ci[0]:.6f}, {mean_ci[1]:.6f}]",
    )
    if len(df) > 1:
        print("  std:", f"{df['delta_mastery'].std(ddof=1):.6f}")
    else:
        print("  std: 0.000000")
    print(
        "  mean absolute:",
        f"{df['delta_mastery'].abs().mean():.6f}",
    )
    print(
        "  positive / zero / negative:",
        f"{positive.mean():.3%} / {zero.mean():.3%} / {negative.mean():.3%}",
    )
    print()
    print("Quantiles")
    for q, value in quantiles.items():
        print(f"  q={q:>4.2f}: {value:+.6f}")

    print()
    print("By evaluator correctness")
    print(evaluator_summary.to_string(index=False))

    print()
    print("By repeated misunderstanding")
    print(repeated_summary.to_string(index=False))

    print()
    print("Saved:")
    for name in (
        "validated_records.csv",
        "delta_by_evaluator.csv",
        "delta_by_repeated_misunderstanding.csv",
        "spearman_correlations.csv",
        "summary.json",
        "figures/delta_mastery_histogram.png",
        "figures/delta_by_evaluator_boxplot.png",
    ):
        print(" ", args.output_dir / name)


if __name__ == "__main__":
    main()
