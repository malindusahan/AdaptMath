"""Stage A: offline frozen-MD6 margin and eligibility audit.

This script reads only the processed MathDial validation split, reproduces the
frozen MD6 validation metrics, and stops before any threshold analysis if that
gate fails.  It performs no threshold-performance simulation and makes no
threshold selection or recommendation.

Smoke-test local loading and paired tokenization::

    python -m src.self_improvement.analysis.audit_md6_gap_thresholds --smoke-test

Run the complete validation audit::

    python -m src.self_improvement.analysis.audit_md6_gap_thresholds --run-full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


AUDIT_NAME: Final[str] = "md6_gap_threshold_audit_stage_a_v1"
VALIDATION_RELATIVE_PATH: Final[Path] = Path(
    "data/processed/mathdial/validation.jsonl"
)
MODEL_RELATIVE_PATH: Final[Path] = Path("models/frozen/md6")
OUTPUT_RELATIVE_PATH: Final[Path] = Path(
    "results/self_improvement/md6_gap_threshold_audit"
)
EXPECTED_VALIDATION_ROWS: Final[int] = 1_850
LABELS: Final[tuple[str, ...]] = (
    "generic",
    "probing",
    "focus",
    "telling",
)
LABEL_TO_ID: Final[dict[str, int]] = {
    label: index for index, label in enumerate(LABELS)
}
THRESHOLDS: Final[tuple[float, ...]] = (
    0.000,
    0.025,
    0.050,
    0.100,
    0.150,
    0.200,
    0.300,
)
MAX_LENGTH: Final[int] = 512
DEFAULT_BATCH_SIZE: Final[int] = 16
PROBABILITY_TOLERANCE: Final[float] = 1e-6
NUMERIC_TOLERANCE: Final[float] = 1e-12
REPRODUCTION_TOLERANCE: Final[float] = 1e-4

EXPECTED_METRICS: Final[dict[str, object]] = {
    "accuracy": 0.5016216216,
    "macro_f1": 0.4876845727,
    "per_class_f1": {
        "generic": 0.587500,
        "probing": 0.417871,
        "focus": 0.571008,
        "telling": 0.374359,
    },
}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_validation(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Validation row {line_number} is not an object.")
            records.append(record)
    return records


def _format_history(history: object) -> str:
    if not history:
        return "[No previous conversation]"
    if not isinstance(history, list):
        raise TypeError("MathDial history must be a list.")
    lines: list[str] = []
    for turn in history:
        if not isinstance(turn, dict):
            raise TypeError("MathDial history turn must be an object.")
        lines.append(f"{turn['user']}: {turn['text']}")
    return "\n".join(lines)


def build_paired_input(example: dict[str, object]) -> tuple[str, str]:
    """Reproduce the canonical frozen-MD6 paired input exactly."""

    sequence_a = f"Problem:\n{example['problem']}"
    sequence_b = (
        "Conversation:\n"
        f"{_format_history(example['history'])}"
        "\n\n"
        "Next teacher pedagogical move:"
    )
    return sequence_a, sequence_b


def _verify_model_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    observed_id2label = {
        int(index): label for index, label in config.get("id2label", {}).items()
    }
    expected_id2label = dict(enumerate(LABELS))
    if observed_id2label != expected_id2label:
        raise AssertionError(
            f"Frozen MD6 id2label mismatch: {observed_id2label!r}"
        )
    observed_label2id = config.get("label2id")
    if observed_label2id != LABEL_TO_ID:
        raise AssertionError(
            f"Frozen MD6 label2id mismatch: {observed_label2id!r}"
        )
    if config.get("architectures") != ["RobertaForSequenceClassification"]:
        raise AssertionError("Unexpected frozen MD6 architecture.")
    return config


def run_inference(
    records: Sequence[dict[str, object]],
    model_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, str]]:
    """Run deterministic CPU inference using only local frozen assets."""

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    if model.config.id2label != dict(enumerate(LABELS)):
        raise AssertionError("Loaded model class order differs from config.json.")

    probabilities: list[np.ndarray] = []
    total_batches = math.ceil(len(records) / batch_size)
    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, len(records), batch_size), start=1
        ):
            batch = records[start : start + batch_size]
            paired = [build_paired_input(example) for example in batch]
            sequence_a = [item[0] for item in paired]
            sequence_b = [item[1] for item in paired]
            encoded = tokenizer(
                sequence_a,
                sequence_b,
                truncation="only_second",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            batch_probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities.append(
                batch_probabilities.detach().cpu().numpy().astype(np.float64)
            )
            if batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"inference batch {batch_index}/{total_batches}",
                    flush=True,
                )
    return np.concatenate(probabilities, axis=0), {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": str(bool(getattr(tokenizer, "is_fast", False))),
    }


def compute_validation_metrics(
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
) -> dict[str, object]:
    true_ids = np.asarray(
        [LABEL_TO_ID[str(record["target_move"])] for record in records],
        dtype=np.int64,
    )
    predicted_ids = np.argmax(probabilities, axis=1)
    per_class = f1_score(
        true_ids,
        predicted_ids,
        labels=np.arange(len(LABELS)),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(
                true_ids,
                predicted_ids,
                labels=np.arange(len(LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": {
            label: float(value)
            for label, value in zip(LABELS, per_class, strict=True)
        },
    }


def verify_reproduction(metrics: dict[str, object]) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    observed_per_class = metrics["per_class_f1"]
    expected_per_class = EXPECTED_METRICS["per_class_f1"]
    if not isinstance(observed_per_class, dict) or not isinstance(
        expected_per_class, dict
    ):
        raise TypeError("Per-class metric structure is invalid.")
    pairs = [
        ("accuracy", metrics["accuracy"], EXPECTED_METRICS["accuracy"]),
        ("macro_f1", metrics["macro_f1"], EXPECTED_METRICS["macro_f1"]),
        *[
            (
                f"{label}_f1",
                observed_per_class[label],
                expected_per_class[label],
            )
            for label in LABELS
        ],
    ]
    for name, observed, expected in pairs:
        difference = abs(float(observed) - float(expected))
        comparisons.append(
            {
                "metric": name,
                "observed": float(observed),
                "expected": float(expected),
                "absolute_difference": difference,
                "tolerance": REPRODUCTION_TOLERANCE,
                "pass": difference <= REPRODUCTION_TOLERANCE,
            }
        )
    failures = [item for item in comparisons if not item["pass"]]
    if failures:
        raise AssertionError(
            "Frozen MD6 reproduction gate failed: "
            + json.dumps(failures, sort_keys=True)
        )
    return comparisons


def build_probability_table(
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, (record, probs) in enumerate(
        zip(records, probabilities, strict=True), start=1
    ):
        canonical_order = np.argsort(-probs, kind="stable")
        base_index = int(canonical_order[0])
        second_index = int(canonical_order[1])
        rows.append(
            {
                "example_id": record.get("example_id", row_index),
                "qid": record.get("qid"),
                "group_id": record.get("group_id"),
                "target_move": record["target_move"],
                "p_generic": float(probs[0]),
                "p_probing": float(probs[1]),
                "p_focus": float(probs[2]),
                "p_telling": float(probs[3]),
                "base_move": LABELS[base_index],
                "base_probability": float(probs[base_index]),
                "second_move": LABELS[second_index],
                "second_probability": float(probs[second_index]),
                "top1_top2_gap": float(probs[base_index] - probs[second_index]),
            }
        )
    return pd.DataFrame(rows)


def build_alternative_gaps(probability_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in probability_table.itertuples(index=False):
        for target_move in LABELS:
            if target_move == record.base_move:
                continue
            target_probability = float(getattr(record, f"p_{target_move}"))
            rows.append(
                {
                    "example_id": record.example_id,
                    "base_move": record.base_move,
                    "target_move": target_move,
                    "base_probability": float(record.base_probability),
                    "target_probability": target_probability,
                    "gap": float(record.base_probability - target_probability),
                }
            )
    return pd.DataFrame(rows)


def _threshold_state_table(
    probability_table: pd.DataFrame,
    alternative_gaps: pd.DataFrame,
) -> pd.DataFrame:
    grouped_gaps = {
        example_id: group.set_index("target_move")["gap"].to_dict()
        for example_id, group in alternative_gaps.groupby("example_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for record in probability_table.itertuples(index=False):
        gaps = grouped_gaps[record.example_id]
        minimum_alternative_gap = float(min(gaps.values()))
        for threshold in THRESHOLDS:
            eligible_targets = tuple(
                move for move in LABELS if move in gaps and gaps[move] <= threshold
            )
            eligible_gaps = [float(gaps[move]) for move in eligible_targets]
            rows.append(
                {
                    "example_id": record.example_id,
                    "base_move": record.base_move,
                    "threshold": threshold,
                    "eligible_alternative_count": len(eligible_targets),
                    "eligible_arm_count": 1 + len(eligible_targets),
                    "eligible_alternatives": "|".join(eligible_targets),
                    "min_alternative_gap": minimum_alternative_gap,
                    "best_eligible_gap": (
                        min(eligible_gaps) if eligible_gaps else float("nan")
                    ),
                    **{
                        f"{move}_bias_eligible": int(move in eligible_targets)
                        for move in LABELS
                    },
                }
            )
    return pd.DataFrame(rows)


def build_threshold_summary(threshold_states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold, group in threshold_states.groupby("threshold", sort=True):
        alternative_count = group["eligible_alternative_count"]
        best = group["best_eligible_gap"].dropna()
        rows.append(
            {
                "threshold": float(threshold),
                "mean_eligible_arm_count": float(group["eligible_arm_count"].mean()),
                "median_eligible_arm_count": float(
                    group["eligible_arm_count"].median()
                ),
                "baseline_only_rate": float((alternative_count == 0).mean()),
                "at_least_one_alternative_rate": float(
                    (alternative_count >= 1).mean()
                ),
                "at_least_two_alternatives_rate": float(
                    (alternative_count >= 2).mean()
                ),
                "three_alternatives_rate": float((alternative_count == 3).mean()),
                **{
                    f"{move}_bias_eligible_rate": float(
                        group[f"{move}_bias_eligible"].mean()
                    )
                    for move in LABELS
                },
                "mean_min_alternative_gap": float(
                    group["min_alternative_gap"].mean()
                ),
                "median_min_alternative_gap": float(
                    group["min_alternative_gap"].median()
                ),
                "mean_best_eligible_gap": (
                    float(best.mean()) if len(best) else float("nan")
                ),
                "median_best_eligible_gap": (
                    float(best.median()) if len(best) else float("nan")
                ),
                "p90_best_eligible_gap": (
                    float(best.quantile(0.90)) if len(best) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_eligible_count_distribution(
    threshold_states: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold, group in threshold_states.groupby("threshold", sort=True):
        for count in range(1, 5):
            n = int((group["eligible_arm_count"] == count).sum())
            rows.append(
                {
                    "threshold": float(threshold),
                    "eligible_arm_count": count,
                    "n_examples": n,
                    "fraction": n / len(group),
                }
            )
    return pd.DataFrame(rows)


def build_margin_summary(probability_table: pd.DataFrame) -> pd.DataFrame:
    values = probability_table["top1_top2_gap"].to_numpy(dtype=float)
    row: dict[str, object] = {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for threshold in THRESHOLDS[1:]:
        key = f"proportion_gap_le_{threshold:.3f}".replace(".", "_")
        row[key] = float(np.mean(values <= threshold))
    return pd.DataFrame([row])


def build_base_move_threshold_summary(
    threshold_states: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (base_move, threshold), group in threshold_states.groupby(
        ["base_move", "threshold"], sort=False
    ):
        alternative_count = group["eligible_alternative_count"]
        rows.append(
            {
                "base_move": base_move,
                "threshold": float(threshold),
                "n": len(group),
                "baseline_only_rate": float((alternative_count == 0).mean()),
                "mean_eligible_arm_count": float(
                    group["eligible_arm_count"].mean()
                ),
                "at_least_one_alternative_rate": float(
                    (alternative_count >= 1).mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["base_move"] = pd.Categorical(
        result["base_move"], categories=LABELS, ordered=True
    )
    return result.sort_values(["base_move", "threshold"]).reset_index(drop=True)


def run_sanity_checks(
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
    probability_table: pd.DataFrame,
    alternative_gaps: pd.DataFrame,
    threshold_states: pd.DataFrame,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def passed(name: str) -> None:
        checks.append({"check": name, "status": "PASS"})

    assert len(records) == EXPECTED_VALIDATION_ROWS
    passed("validation rows equal 1850")
    assert probabilities.shape == (EXPECTED_VALIDATION_ROWS, len(LABELS))
    assert np.isfinite(probabilities).all()
    passed("all probability vectors are finite")
    assert np.all(probabilities >= 0.0)
    passed("all probabilities are nonnegative")
    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE
    )
    passed("probability vectors sum approximately to one")
    observed_base = probability_table["base_move"].to_numpy()
    expected_base = np.asarray([LABELS[index] for index in np.argmax(probabilities, axis=1)])
    assert np.array_equal(observed_base, expected_base)
    passed("base_move equals canonical-order probability argmax")
    assert float(alternative_gaps["gap"].min()) >= -NUMERIC_TOLERANCE
    passed("all alternative gaps are nonnegative within tolerance")
    per_example_gap_rows = alternative_gaps.groupby("example_id").size()
    assert len(per_example_gap_rows) == EXPECTED_VALIDATION_ROWS
    assert int(per_example_gap_rows.min()) == 3
    assert int(per_example_gap_rows.max()) == 3
    passed("exactly three alternative-gap rows per validation example")
    assert np.all(threshold_states["eligible_arm_count"] >= 1)
    passed("baseline is always eligible")
    for row in threshold_states.itertuples(index=False):
        if row.eligible_alternatives:
            assert row.base_move not in row.eligible_alternatives.split("|")
    passed("base-move bias is never counted as an alternative")
    count_matrix = threshold_states.pivot(
        index="example_id", columns="threshold", values="eligible_arm_count"
    ).reindex(columns=THRESHOLDS)
    assert np.all(np.diff(count_matrix.to_numpy(dtype=int), axis=1) >= 0)
    passed("eligible-arm count is monotonic across increasing thresholds")
    at_zero = threshold_states[threshold_states["threshold"] == 0.0]
    zero_eligible_ids = at_zero.loc[
        at_zero["eligible_alternative_count"] > 0, "example_id"
    ]
    if len(zero_eligible_ids):
        zero_gaps = alternative_gaps[
            alternative_gaps["example_id"].isin(zero_eligible_ids)
        ]
        assert np.all(zero_gaps.loc[zero_gaps["gap"] <= 0.0, "gap"] == 0.0)
    passed("threshold zero admits alternatives only at exact numeric ties")
    passed("only the declared validation dataset path was read")
    return checks


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
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
        metadata={"Creator": "matplotlib", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _add_threshold_lines(axis: plt.Axes) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(THRESHOLDS) - 1))
    for threshold, color in zip(THRESHOLDS[1:], colors, strict=True):
        axis.axvline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label=f"{threshold:.3f}",
        )


def generate_figures(output_dir: Path) -> None:
    probabilities = pd.read_csv(output_dir / "md6_validation_probabilities.csv")
    alternative_gaps = pd.read_csv(output_dir / "alternative_gaps.csv")
    summary = pd.read_csv(output_dir / "threshold_summary.csv")
    count_distribution = pd.read_csv(
        output_dir / "eligible_arm_count_distribution.csv"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    top_gaps = probabilities["top1_top2_gap"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].hist(top_gaps, bins=50, color="#4477AA", alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("MD6 top-1 minus top-2 probability")
    axes[0].set_ylabel("Validation examples")
    axes[0].set_title("Histogram")
    _add_threshold_lines(axes[0])
    ordered = np.sort(top_gaps)
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    axes[1].plot(ordered, ecdf, color="#228833", linewidth=1.8)
    axes[1].set_xlabel("MD6 top-1 minus top-2 probability")
    axes[1].set_ylabel("Empirical cumulative proportion")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Empirical cumulative distribution")
    _add_threshold_lines(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Candidate thresholds",
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
    fig.suptitle("Frozen MD6 validation top-1/top-2 margins", y=0.995, fontsize=13)
    fig.subplots_adjust(top=0.72, wspace=0.25)
    _save_figure(fig, figures_dir, "md6_top1_top2_margin_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for column, label, color in (
        ("baseline_only_rate", "Baseline only", "#CC6677"),
        ("at_least_one_alternative_rate", "At least one alternative", "#4477AA"),
        ("at_least_two_alternatives_rate", "At least two alternatives", "#228833"),
    ):
        axes[0].plot(
            summary["threshold"],
            summary[column],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of validation examples")
    axes[0].set_title("Eligibility rates")
    axes[0].legend(frameon=False)
    axes[1].plot(
        summary["threshold"],
        summary["mean_eligible_arm_count"],
        marker="o",
        color="#AA3377",
    )
    axes[1].set_ylim(1, 4)
    axes[1].set_ylabel("Mean eligible-arm count")
    axes[1].set_title("Action-set size")
    for axis in axes:
        axis.set_xlabel("Gap threshold")
        axis.set_xticks(THRESHOLDS, [f"{value:g}" for value in THRESHOLDS], rotation=30)
        axis.grid(alpha=0.22, linewidth=0.6)
    fig.suptitle("Canonical eligibility as the MD6 gap threshold changes", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "eligibility_vs_threshold")

    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    colors = ("#4477AA", "#66CCEE", "#228833", "#CC6677")
    for move, color in zip(LABELS, colors, strict=True):
        axis.plot(
            summary["threshold"],
            summary[f"{move}_bias_eligible_rate"],
            marker="o",
            label=f"{move}_bias",
            color=color,
        )
    axis.set_xlabel("Gap threshold")
    axis.set_ylabel("Eligibility rate across validation examples")
    axis.set_ylim(0, 1)
    axis.set_xticks(THRESHOLDS, [f"{value:g}" for value in THRESHOLDS])
    axis.set_title("Bias-arm eligibility under the canonical MD6 mask")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    _save_figure(fig, figures_dir, "bias_arm_eligibility_by_threshold")

    pivot = count_distribution.pivot(
        index="threshold", columns="eligible_arm_count", values="fraction"
    ).reindex(index=THRESHOLDS, columns=range(1, 5), fill_value=0.0)
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    bottom = np.zeros(len(pivot), dtype=float)
    colors = ("#CC6677", "#DDCC77", "#44AA99", "#4477AA")
    for count, color in zip(range(1, 5), colors, strict=True):
        values = pivot[count].to_numpy(dtype=float)
        axis.bar(
            np.arange(len(pivot)),
            values,
            bottom=bottom,
            label=f"{count} eligible arm{'s' if count != 1 else ''}",
            color=color,
        )
        bottom += values
    axis.set_xticks(
        np.arange(len(pivot)), [f"{value:g}" for value in THRESHOLDS]
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Gap threshold")
    axis.set_ylabel("Fraction of validation examples")
    axis.set_title("Eligible-arm count distribution by threshold")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    _save_figure(fig, figures_dir, "eligible_arm_count_by_threshold")

    gaps = np.sort(alternative_gaps["gap"].to_numpy(dtype=float))
    ecdf = np.arange(1, len(gaps) + 1) / len(gaps)
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.plot(gaps, ecdf, color="#332288", linewidth=1.8)
    _add_threshold_lines(axis)
    axis.set_xlabel("Base-to-alternative MD6 probability gap")
    axis.set_ylabel("Empirical cumulative proportion of alternatives")
    axis.set_ylim(0, 1)
    axis.set_title("All non-base pedagogical-move gaps")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(
        title="Candidate thresholds", frameon=False, ncol=2, loc="lower right"
    )
    _save_figure(fig, figures_dir, "alternative_gap_ecdf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_tokenizer_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = model_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(
    project_root: Path,
    stage_dir: Path,
    model_dir: Path,
    validation_path: Path,
    metrics: dict[str, object],
    reproduction_comparisons: list[dict[str, object]],
    runtime_versions: dict[str, str],
    checks: list[dict[str, str]],
) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    output_hashes = {
        str(path.relative_to(stage_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path.name != "audit_manifest.json"
    }
    return {
        "audit_name": AUDIT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_timezone": "UTC",
        "stage": "A_real_frozen_md6_margin_eligibility_audit",
        "validation_dataset_path": str(
            validation_path.relative_to(project_root)
        ).replace("\\", "/"),
        "validation_row_count": len(pd.read_csv(stage_dir / "md6_validation_probabilities.csv")),
        "model_path": str(model_dir.relative_to(project_root)).replace("\\", "/"),
        "md6_config_sha256": _sha256(model_dir / "config.json"),
        "md6_model_safetensors_sha256": _sha256(
            model_dir / "model.safetensors"
        ),
        "tokenizer_sha256": _combined_tokenizer_hash(model_dir),
        "tokenizer_json_sha256": _sha256(model_dir / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256(
            model_dir / "tokenizer_config.json"
        ),
        "audit_script_sha256": _sha256(script_path),
        "model_class_order": list(LABELS),
        "input_format": {
            "sequence_a": "Problem:\n{problem}",
            "sequence_b": (
                "Conversation:\n{history_or_[No previous conversation]}\n\n"
                "Next teacher pedagogical move:"
            ),
            "history_turn_format": "{turn['user']}: {turn['text']}",
            "paired_tokenization": True,
            "privileged_fields_used": [],
        },
        "max_length": MAX_LENGTH,
        "truncation_side": "left",
        "truncation_strategy": "only_second",
        "padding": "dynamic longest sequence within each inference batch",
        "threshold_grid": list(THRESHOLDS),
        "threshold_metric_definitions": {
            "eligible_arm_count": (
                "1 baseline arm plus non-base target moves with "
                "P(base)-P(target) <= threshold."
            ),
            "baseline_only_rate": "Fraction with zero eligible alternatives.",
            "at_least_one_alternative_rate": (
                "Fraction with one or more non-base eligible targets."
            ),
            "at_least_two_alternatives_rate": (
                "Fraction with two or more non-base eligible targets."
            ),
            "three_alternatives_rate": (
                "Fraction with all three non-base targets eligible."
            ),
            "bias_arm_eligible_rate": (
                "Fraction of all validation examples where the named target is "
                "non-base and within threshold."
            ),
            "min_alternative_gap": (
                "Minimum P(base)-P(target) among all three non-base moves, "
                "regardless of threshold."
            ),
            "best_eligible_gap": (
                "Minimum gap among eligible non-base moves; summarized only "
                "for examples with at least one eligible alternative."
            ),
        },
        "reproduced_validation_metrics": metrics,
        "expected_canonical_validation_metrics": EXPECTED_METRICS,
        "reproduction_tolerance_absolute": REPRODUCTION_TOLERANCE,
        "reproduction_comparisons": reproduction_comparisons,
        "canonical_reproduction": "PASS",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": runtime_versions["torch"],
        "transformers_version": runtime_versions["transformers"],
        "tokenizer_runtime_class": runtime_versions["tokenizer_class"],
        "tokenizer_is_fast": runtime_versions["tokenizer_is_fast"],
        "local_files_only": True,
        "dataset_paths_read": [
            str(validation_path.relative_to(project_root)).replace("\\", "/")
        ],
        "threshold_performance_simulation_run": False,
        "sanity_checks": checks,
        "output_file_sha256": output_hashes,
    }


def run_smoke_test(project_root: Path, batch_size: int) -> None:
    validation_path = project_root / VALIDATION_RELATIVE_PATH
    model_dir = project_root / MODEL_RELATIVE_PATH
    _verify_model_config(model_dir)
    records = _read_validation(validation_path)
    probabilities, versions = run_inference(records[:4], model_dir, batch_size)
    if probabilities.shape != (4, len(LABELS)):
        raise RuntimeError("Smoke-test probability shape is invalid.")
    print(
        json.dumps(
            {
                "rows": 4,
                "probability_shape": list(probabilities.shape),
                "probability_sums": probabilities.sum(axis=1).tolist(),
                "runtime": versions,
                "first_sequence": build_paired_input(records[0]),
            },
            indent=2,
        )
    )


def run_full_audit(
    project_root: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    validation_path = project_root / VALIDATION_RELATIVE_PATH
    model_dir = project_root / MODEL_RELATIVE_PATH
    _verify_model_config(model_dir)
    records = _read_validation(validation_path)
    if len(records) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError(
            f"Validation row mismatch: observed={len(records)}, "
            f"expected={EXPECTED_VALIDATION_ROWS}."
        )
    probabilities, runtime_versions = run_inference(
        records, model_dir, batch_size
    )
    metrics = compute_validation_metrics(records, probabilities)
    print("reproduced metrics:", json.dumps(metrics, sort_keys=True), flush=True)
    reproduction_comparisons = verify_reproduction(metrics)

    probability_table = build_probability_table(records, probabilities)
    alternative_gaps = build_alternative_gaps(probability_table)
    threshold_states = _threshold_state_table(
        probability_table, alternative_gaps
    )
    checks = run_sanity_checks(
        records,
        probabilities,
        probability_table,
        alternative_gaps,
        threshold_states,
    )
    threshold_summary = build_threshold_summary(threshold_states)
    count_distribution = build_eligible_count_distribution(threshold_states)
    margin_summary = build_margin_summary(probability_table)
    base_summary = build_base_move_threshold_summary(threshold_states)

    stage_dir = output_dir.with_name(output_dir.name + ".in_progress")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_dir}")
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging audit: {stage_dir}")
    stage_dir.mkdir(parents=True)
    probability_table.to_csv(
        stage_dir / "md6_validation_probabilities.csv", index=False
    )
    alternative_gaps.to_csv(stage_dir / "alternative_gaps.csv", index=False)
    threshold_summary.to_csv(stage_dir / "threshold_summary.csv", index=False)
    count_distribution.to_csv(
        stage_dir / "eligible_arm_count_distribution.csv", index=False
    )
    margin_summary.to_csv(stage_dir / "margin_summary.csv", index=False)
    base_summary.to_csv(
        stage_dir / "base_move_threshold_summary.csv", index=False
    )
    generate_figures(stage_dir)
    manifest = build_manifest(
        project_root,
        stage_dir,
        model_dir,
        validation_path,
        metrics,
        reproduction_comparisons,
        runtime_versions,
        checks,
    )
    with (stage_dir / "audit_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    stage_dir.replace(output_dir)
    print(f"Audit outputs: {output_dir}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--run-full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = _default_project_root()
    output_dir = args.output_dir or (project_root / OUTPUT_RELATIVE_PATH)
    try:
        if args.smoke_test:
            run_smoke_test(project_root, args.batch_size)
        else:
            run_full_audit(project_root, output_dir, args.batch_size)
    except Exception as exc:
        print(f"OVERALL: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

