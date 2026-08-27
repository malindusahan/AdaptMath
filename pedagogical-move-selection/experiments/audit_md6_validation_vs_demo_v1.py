"""Compare frozen-MD6 prediction behavior on validation and demo inputs.

This is a descriptive distribution-behavior audit. The 24 hand-written cases
do not have authoritative gold pedagogical-move labels, and differences
between source groups are not automatically evidence of domain shift.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.audit_md6_move_behavior_v1 import (
    fixed_cases,
    validate_fixed_cases,
)
from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.md6_inference import FrozenMD6Inference


AUDIT_NAME: Final[str] = "md6_validation_vs_demo_v1"
PURPOSE: Final[str] = (
    "Compare frozen MD6 prediction and conservative-eligibility distributions "
    "between actual MathDial validation inputs and 24 fixed hand-written cases."
)
INTERPRETATION_BOUNDARY: Final[str] = (
    "Descriptive evidence only. Hand-written cases are not gold-labeled, and "
    "distribution differences alone are not claimed to establish domain shift."
)
VALIDATION_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "mathdial" / "validation.jsonl"
)
CANONICAL_VALIDATION_PROBABILITIES: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_gap_threshold_audit"
    / "md6_validation_probabilities.csv"
)
CANONICAL_VALIDATION_AUDIT_MANIFEST: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_gap_threshold_audit"
    / "audit_manifest.json"
)
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "frozen" / "md6"
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "self_improvement" / AUDIT_NAME
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"
EXPECTED_VALIDATION_ROWS: Final[int] = 1_850
EXPECTED_DEMO_ROWS: Final[int] = 24
GAP_THRESHOLD: Final[float] = 0.10
PROBABILITY_TOLERANCE: Final[float] = 1e-5
CPU_THREADS: Final[int] = max(1, min(8, os.cpu_count() or 1))

SOURCE_MATHDIAL: Final[str] = "mathdial_validation"
SOURCE_DEMO: Final[str] = "handwritten_demo"
SOURCE_GROUPS: Final[tuple[str, ...]] = (SOURCE_MATHDIAL, SOURCE_DEMO)
OPENED_DATASET_PATHS: set[Path] = set()

PREDICTION_FIELDS: Final[tuple[str, ...]] = (
    "source_group",
    "row_id",
    "qid",
    "group_id",
    "problem",
    "conversation_history",
    "gold_move",
    "p_generic",
    "p_probing",
    "p_focus",
    "p_telling",
    "argmax_move",
    "top1_probability",
    "second_move",
    "top2_probability",
    "top1_top2_gap",
    "entropy",
    "eligible_arms",
    "eligible_arm_count",
    "alternative_available",
)

GROUP_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "source_group",
    "n",
    "generic_argmax_count",
    "generic_argmax_percentage",
    "probing_argmax_count",
    "probing_argmax_percentage",
    "focus_argmax_count",
    "focus_argmax_percentage",
    "telling_argmax_count",
    "telling_argmax_percentage",
    "generic_probability_mean",
    "generic_probability_median",
    "probing_probability_mean",
    "probing_probability_median",
    "focus_probability_mean",
    "focus_probability_median",
    "telling_probability_mean",
    "telling_probability_median",
    "top1_confidence_mean",
    "top1_confidence_median",
    "top1_top2_gap_mean",
    "top1_top2_gap_median",
    "gap_q0",
    "gap_q10",
    "gap_q25",
    "gap_q50",
    "gap_q75",
    "gap_q90",
    "gap_q100",
    "mean_entropy",
    "baseline_only_percentage",
    "alternative_available_percentage",
    "mean_eligible_arm_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mathdial_validation() -> list[dict[str, object]]:
    """Read exactly the declared validation JSONL and no other dataset path."""

    resolved = VALIDATION_PATH.resolve()
    expected = (
        PROJECT_ROOT / "data" / "processed" / "mathdial" / "validation.jsonl"
    ).resolve()
    if resolved != expected:
        raise RuntimeError("Validation path guard rejected an unexpected path.")

    rows: list[dict[str, object]] = []
    OPENED_DATASET_PATHS.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank validation row at line {line_number}.")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Validation row {line_number} is not an object.")
            rows.append(record)

    assert len(rows) == EXPECTED_VALIDATION_ROWS
    for index, row in enumerate(rows):
        assert isinstance(row.get("problem"), str) and row["problem"].strip()
        history = row.get("history")
        assert isinstance(history, list)
        for turn in history:
            assert isinstance(turn, Mapping)
            assert isinstance(turn.get("user"), str) and turn["user"].strip()
            assert isinstance(turn.get("text"), str)
        assert isinstance(row.get("target_move"), str)
        assert row["target_move"] in MOVE_ORDER
        if "example_id" not in row:
            row["example_id"] = f"validation_row_{index:04d}"
    return rows


def prepare_mathdial_records(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "source_group": SOURCE_MATHDIAL,
            "row_id": str(row["example_id"]),
            "qid": row.get("qid", ""),
            "group_id": row.get("group_id", ""),
            "problem": row["problem"],
            "history": row["history"],
            "gold_move": row.get("target_move", ""),
        }
        for row in rows
    ]


def prepare_demo_records() -> list[dict[str, object]]:
    cases = fixed_cases()
    validate_fixed_cases(cases)
    assert len(cases) == EXPECTED_DEMO_ROWS
    return [
        {
            "source_group": SOURCE_DEMO,
            "row_id": f"demo_{int(case['case_id']):02d}",
            "qid": "",
            "group_id": case["scenario_group"],
            "problem": case["problem"],
            "history": case["history"],
            "gold_move": "",
        }
        for case in cases
    ]


def _rank_moves(probabilities: Mapping[str, float]) -> list[str]:
    canonical_index = {move: index for index, move in enumerate(MOVE_ORDER)}
    return sorted(
        MOVE_ORDER,
        key=lambda move: (-probabilities[move], canonical_index[move]),
    )


def _entropy(probabilities: Mapping[str, float]) -> float:
    return float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )


def prediction_from_probabilities(
    record: Mapping[str, object],
    probabilities: Mapping[str, float],
) -> dict[str, object]:
    if tuple(probabilities) != MOVE_ORDER:
        raise AssertionError("Canonical MD6 move order changed.")
    ranked = _rank_moves(probabilities)
    argmax_move, second_move = ranked[:2]
    top1 = float(probabilities[argmax_move])
    top2 = float(probabilities[second_move])
    candidates = eligible_arms(probabilities, gap_threshold=GAP_THRESHOLD)
    return {
        "source_group": record["source_group"],
        "row_id": record["row_id"],
        "qid": record["qid"],
        "group_id": record["group_id"],
        "problem": record["problem"],
        "conversation_history": json.dumps(
            record["history"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "gold_move": record["gold_move"],
        "p_generic": float(probabilities["generic"]),
        "p_probing": float(probabilities["probing"]),
        "p_focus": float(probabilities["focus"]),
        "p_telling": float(probabilities["telling"]),
        "argmax_move": argmax_move,
        "top1_probability": top1,
        "second_move": second_move,
        "top2_probability": top2,
        "top1_top2_gap": float(top1 - top2),
        "entropy": _entropy(probabilities),
        "eligible_arms": json.dumps(list(candidates), separators=(",", ":")),
        "eligible_arm_count": len(candidates),
        "alternative_available": len(candidates) > 1,
    }


def load_canonical_mathdial_predictions(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    """Load all canonical validation probabilities after provenance checks."""

    manifest = json.loads(
        CANONICAL_VALIDATION_AUDIT_MANIFEST.read_text(encoding="utf-8")
    )
    assert manifest["canonical_reproduction"] == "PASS"
    assert manifest["validation_row_count"] == EXPECTED_VALIDATION_ROWS
    assert manifest["dataset_paths_read"] == [
        "data/processed/mathdial/validation.jsonl"
    ]
    assert tuple(manifest["model_class_order"]) == MOVE_ORDER
    expected_outputs = manifest["output_file_sha256"]
    assert isinstance(expected_outputs, Mapping)
    assert sha256_file(CANONICAL_VALIDATION_PROBABILITIES) == expected_outputs[
        "md6_validation_probabilities.csv"
    ]
    assert sha256_file(MODEL_PATH / "config.json") == manifest["md6_config_sha256"]
    assert sha256_file(MODEL_PATH / "model.safetensors") == manifest[
        "md6_model_safetensors_sha256"
    ]
    assert sha256_file(MODEL_PATH / "tokenizer.json") == manifest[
        "tokenizer_json_sha256"
    ]
    assert sha256_file(MODEL_PATH / "tokenizer_config.json") == manifest[
        "tokenizer_config_sha256"
    ]

    with CANONICAL_VALIDATION_PROBABILITIES.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        cached_rows = list(csv.DictReader(handle))
    assert len(cached_rows) == EXPECTED_VALIDATION_ROWS
    by_id = {row["example_id"]: row for row in cached_rows}
    assert len(by_id) == EXPECTED_VALIDATION_ROWS

    predictions: list[dict[str, object]] = []
    for record in records:
        cached = by_id[str(record["row_id"])]
        assert cached["qid"] == str(record["qid"])
        assert cached["group_id"] == str(record["group_id"])
        assert cached["target_move"] == str(record["gold_move"])
        probabilities = {
            move: float(cached[f"p_{move}"]) for move in MOVE_ORDER
        }
        prediction = prediction_from_probabilities(record, probabilities)
        assert prediction["argmax_move"] == cached["base_move"]
        assert prediction["second_move"] == cached["second_move"]
        assert math.isclose(
            float(prediction["top1_top2_gap"]),
            float(cached["top1_top2_gap"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        predictions.append(prediction)
    return predictions, manifest


def infer_records_public(
    model: FrozenMD6Inference,
    records: Sequence[Mapping[str, object]],
    *,
    progress_label: str,
    progress_every: int,
) -> list[dict[str, object]]:
    """Run each record through the public production inference method."""

    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        problem = record["problem"]
        history = record["history"]
        if not isinstance(problem, str) or not isinstance(history, Sequence):
            raise TypeError("Inference record has invalid problem/history types.")
        probabilities = model.predict_probabilities(problem, history)
        predictions.append(prediction_from_probabilities(record, probabilities))
        if index % progress_every == 0 or index == len(records):
            elapsed = time.perf_counter() - started
            print(
                f"{progress_label}: {index}/{len(records)} "
                f"({elapsed:.1f} seconds elapsed)",
                flush=True,
            )
    return predictions


def validate_public_wrapper_parity(
    model: FrozenMD6Inference,
    records: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> None:
    """Check representative stored results against the public production API."""

    for record, prediction in zip(records[:2], predictions[:2], strict=True):
        problem = record["problem"]
        history = record["history"]
        if not isinstance(problem, str) or not isinstance(history, Sequence):
            raise TypeError("Parity record has invalid problem/history types.")
        public = model.predict_probabilities(problem, history)
        audited = np.asarray(
            [float(prediction[f"p_{move}"]) for move in MOVE_ORDER],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            audited,
            np.asarray([public[move] for move in MOVE_ORDER], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )


def validate_predictions(
    rows: Sequence[Mapping[str, object]],
    expected_n: int,
    expected_source: str,
) -> None:
    assert len(rows) == expected_n
    for row in rows:
        assert row["source_group"] == expected_source
        probabilities = [float(row[f"p_{move}"]) for move in MOVE_ORDER]
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
        assert math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE,
        )
        assert row["argmax_move"] in MOVE_ORDER
        gap = float(row["top1_top2_gap"])
        assert math.isfinite(gap) and 0.0 <= gap <= 1.0
        entropy = float(row["entropy"])
        assert math.isfinite(entropy) and entropy >= 0.0
        candidates = json.loads(str(row["eligible_arms"]))
        assert candidates and candidates[0] == "baseline"
        assert int(row["eligible_arm_count"]) == len(candidates)
        assert bool(row["alternative_available"]) == (len(candidates) > 1)


def _mean_median(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def summarize_group(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    counts = Counter(str(row["argmax_move"]) for row in rows)
    ordered_counts = {move: int(counts.get(move, 0)) for move in MOVE_ORDER}
    percentages = {
        move: float(100.0 * ordered_counts[move] / n) for move in MOVE_ORDER
    }
    probability_summaries = {
        move: _mean_median([float(row[f"p_{move}"]) for row in rows])
        for move in MOVE_ORDER
    }
    top1 = [float(row["top1_probability"]) for row in rows]
    gaps = np.asarray([float(row["top1_top2_gap"]) for row in rows], dtype=np.float64)
    entropy = [float(row["entropy"]) for row in rows]
    eligible_counts = [int(row["eligible_arm_count"]) for row in rows]
    alternative_count = sum(bool(row["alternative_available"]) for row in rows)
    baseline_count = n - alternative_count
    quantile_levels = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    quantile_values = np.quantile(gaps, quantile_levels)

    return {
        "n": n,
        "argmax_counts": ordered_counts,
        "argmax_percentages": percentages,
        "probability_summaries": probability_summaries,
        "top1_confidence": _mean_median(top1),
        "top1_top2_gap": {
            **_mean_median(gaps.tolist()),
            "quantiles": {
                f"{level:.2f}": float(value)
                for level, value in zip(quantile_levels, quantile_values, strict=True)
            },
        },
        "mean_entropy": float(statistics.fmean(entropy)),
        "eligibility_at_0_10": {
            "baseline_only_count": baseline_count,
            "baseline_only_percentage": float(100.0 * baseline_count / n),
            "alternative_available_count": alternative_count,
            "alternative_available_percentage": float(100.0 * alternative_count / n),
            "mean_eligible_arm_count": float(statistics.fmean(eligible_counts)),
        },
    }


def flatten_group_summary(
    source_group: str,
    summary: Mapping[str, object],
) -> dict[str, object]:
    counts = summary["argmax_counts"]
    percentages = summary["argmax_percentages"]
    probabilities = summary["probability_summaries"]
    top1 = summary["top1_confidence"]
    gap = summary["top1_top2_gap"]
    eligibility = summary["eligibility_at_0_10"]
    assert all(
        isinstance(value, Mapping)
        for value in (counts, percentages, probabilities, top1, gap, eligibility)
    )
    quantiles = gap["quantiles"]
    assert isinstance(quantiles, Mapping)

    row: dict[str, object] = {"source_group": source_group, "n": summary["n"]}
    for move in MOVE_ORDER:
        row[f"{move}_argmax_count"] = counts[move]
        row[f"{move}_argmax_percentage"] = percentages[move]
        move_probability = probabilities[move]
        assert isinstance(move_probability, Mapping)
        row[f"{move}_probability_mean"] = move_probability["mean"]
        row[f"{move}_probability_median"] = move_probability["median"]
    row.update(
        {
            "top1_confidence_mean": top1["mean"],
            "top1_confidence_median": top1["median"],
            "top1_top2_gap_mean": gap["mean"],
            "top1_top2_gap_median": gap["median"],
            "gap_q0": quantiles["0.00"],
            "gap_q10": quantiles["0.10"],
            "gap_q25": quantiles["0.25"],
            "gap_q50": quantiles["0.50"],
            "gap_q75": quantiles["0.75"],
            "gap_q90": quantiles["0.90"],
            "gap_q100": quantiles["1.00"],
            "mean_entropy": summary["mean_entropy"],
            "baseline_only_percentage": eligibility["baseline_only_percentage"],
            "alternative_available_percentage": eligibility[
                "alternative_available_percentage"
            ],
            "mean_eligible_arm_count": eligibility["mean_eligible_arm_count"],
        }
    )
    return row


def write_predictions(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    compressed: bool,
) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in PREDICTION_FIELDS})


def write_group_summary(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in GROUP_SUMMARY_FIELDS})


def create_argmax_figure(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
) -> None:
    counts = Counter(str(row["argmax_move"]) for row in rows)
    figure = plt.figure(figsize=(8, 5))
    plt.bar(MOVE_ORDER, [counts.get(move, 0) for move in MOVE_ORDER])
    plt.title(title)
    plt.xlabel("Frozen MD6 argmax move")
    plt.ylabel("Number of examples")
    plt.ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def create_histogram(
    values: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    figure = plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of examples")
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def create_figures(
    mathdial_rows: Sequence[Mapping[str, object]],
    demo_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    paths = {
        "argmax_mathdial": FIGURE_DIR / "argmax_distribution_mathdial.png",
        "argmax_demo": FIGURE_DIR / "argmax_distribution_demo.png",
        "gap_mathdial": FIGURE_DIR / "top1_gap_mathdial.png",
        "gap_demo": FIGURE_DIR / "top1_gap_demo.png",
        "probing_mathdial": FIGURE_DIR / "probing_probability_mathdial.png",
        "probing_demo": FIGURE_DIR / "probing_probability_demo.png",
    }
    create_argmax_figure(
        mathdial_rows,
        paths["argmax_mathdial"],
        "Frozen MD6 argmax distribution: MathDial validation",
    )
    create_argmax_figure(
        demo_rows,
        paths["argmax_demo"],
        "Frozen MD6 argmax distribution: hand-written cases",
    )
    create_histogram(
        [float(row["top1_top2_gap"]) for row in mathdial_rows],
        paths["gap_mathdial"],
        "Top-1 minus top-2 gaps: MathDial validation",
        "Top-1 minus top-2 probability gap",
    )
    create_histogram(
        [float(row["top1_top2_gap"]) for row in demo_rows],
        paths["gap_demo"],
        "Top-1 minus top-2 gaps: hand-written cases",
        "Top-1 minus top-2 probability gap",
    )
    create_histogram(
        [float(row["p_probing"]) for row in mathdial_rows],
        paths["probing_mathdial"],
        "Probing probability: MathDial validation",
        "Frozen MD6 probing probability",
    )
    create_histogram(
        [float(row["p_probing"]) for row in demo_rows],
        paths["probing_demo"],
        "Probing probability: hand-written cases",
        "Frozen MD6 probing probability",
    )
    return list(paths.values())


def build_manifest(output_files: Sequence[Path], runtime_seconds: float) -> dict[str, object]:
    source_files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "experiments" / "audit_md6_move_behavior_v1.py",
        PROJECT_ROOT / "src" / "self_improvement" / "md6_inference.py",
        PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py",
        PROJECT_ROOT / "src" / "self_improvement" / "context_builder.py",
    )
    model_files = (
        MODEL_PATH / "config.json",
        MODEL_PATH / "model.safetensors",
        MODEL_PATH / "tokenizer.json",
        MODEL_PATH / "tokenizer_config.json",
    )
    return {
        "audit_name": AUDIT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": PURPOSE,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "dataset_scope": "MathDial validation only",
        "dataset_paths_opened": [str(VALIDATION_PATH.relative_to(PROJECT_ROOT))],
        "existing_artifacts_read": [
            str(CANONICAL_VALIDATION_PROBABILITIES.relative_to(PROJECT_ROOT)),
            str(CANONICAL_VALIDATION_AUDIT_MANIFEST.relative_to(PROJECT_ROOT)),
        ],
        "held_out_test_untouched": True,
        "mathdial_validation_n": EXPECTED_VALIDATION_ROWS,
        "handwritten_demo_n": EXPECTED_DEMO_ROWS,
        "frozen_md6": True,
        "frozen_model_path": str(MODEL_PATH),
        "frozen_gap_threshold": GAP_THRESHOLD,
        "model_changes": False,
        "mathdial_validation_probability_source": (
            "All 1,850 rows from the existing canonical frozen-MD6 validation "
            "probability artifact. Its PASS reproduction manifest, artifact hash, "
            "model hashes, tokenizer hashes, dataset scope, class order, row IDs, "
            "labels, and live public-wrapper parity were verified."
        ),
        "handwritten_inference_api": "FrozenMD6Inference.predict_probabilities",
        "cpu_only": True,
        "cpu_threads": CPU_THREADS,
        "public_wrapper_parity_cases": 2,
        "runtime_seconds": runtime_seconds,
        "canonical_move_order": list(MOVE_ORDER),
        "source_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_files
        },
        "validation_dataset_sha256": sha256_file(VALIDATION_PATH),
        "frozen_model_file_sha256": {
            path.name: sha256_file(path) for path in model_files
        },
        "output_files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def _compact_example(row: Mapping[str, object]) -> str:
    problem = " ".join(str(row["problem"]).split())
    if len(problem) > 110:
        problem = problem[:109].rstrip() + "…"
    return (
        f"{row['row_id']} | predicted={row['argmax_move']} | "
        f"top1={float(row['top1_probability']):.4f} | "
        f"second={row['second_move']} | gap={float(row['top1_top2_gap']):.4f} | "
        f"{problem}"
    )


def print_high_confidence_by_move(
    mathdial_rows: Sequence[Mapping[str, object]],
) -> None:
    print("\nMATHDIAL VALIDATION: FIVE HIGHEST-CONFIDENCE EXAMPLES BY PREDICTED MOVE")
    for move in MOVE_ORDER:
        predicted = [row for row in mathdial_rows if row["argmax_move"] == move]
        print(f"\n{move} (N={len(predicted)})")
        if len(predicted) < 5:
            print("Fewer than five examples available.")
            continue
        highest = sorted(
            predicted,
            key=lambda row: (-float(row["top1_probability"]), str(row["row_id"])),
        )[:5]
        for row in highest:
            print(_compact_example(row))


def comparison_metrics(
    summaries: Mapping[str, Mapping[str, object]],
) -> list[tuple[str, float, float]]:
    mathdial = summaries[SOURCE_MATHDIAL]
    demo = summaries[SOURCE_DEMO]

    def values(summary: Mapping[str, object]) -> dict[str, float]:
        percentages = summary["argmax_percentages"]
        probabilities = summary["probability_summaries"]
        gap = summary["top1_top2_gap"]
        eligibility = summary["eligibility_at_0_10"]
        assert all(
            isinstance(value, Mapping)
            for value in (percentages, probabilities, gap, eligibility)
        )
        probing_probability = probabilities["probing"]
        assert isinstance(probing_probability, Mapping)
        return {
            "probing argmax %": float(percentages["probing"]),
            "generic argmax %": float(percentages["generic"]),
            "focus argmax %": float(percentages["focus"]),
            "telling argmax %": float(percentages["telling"]),
            "mean probing probability": float(probing_probability["mean"]),
            "mean top1-top2 gap": float(gap["mean"]),
            "baseline-only %": float(eligibility["baseline_only_percentage"]),
            "alternative available %": float(
                eligibility["alternative_available_percentage"]
            ),
            "mean entropy": float(summary["mean_entropy"]),
        }

    mathdial_values = values(mathdial)
    demo_values = values(demo)
    return [
        (metric, mathdial_values[metric], demo_values[metric])
        for metric in mathdial_values
    ]


def print_comparison_table(
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    print("\n" + "=" * 74)
    print("PROMINENT SOURCE-GROUP COMPARISON")
    print("=" * 74)
    print(f"{'metric':<32} {'MathDial val':>18} {'Hand-written':>18}")
    print("-" * 74)
    percentage_metrics = {
        "probing argmax %",
        "generic argmax %",
        "focus argmax %",
        "telling argmax %",
        "baseline-only %",
        "alternative available %",
    }
    for metric, mathdial, demo in comparison_metrics(summaries):
        if metric in percentage_metrics:
            left = f"{mathdial:.2f}%"
            right = f"{demo:.2f}%"
        else:
            left = f"{mathdial:.6f}"
            right = f"{demo:.6f}"
        print(f"{metric:<32} {left:>18} {right:>18}")
    print("=" * 74)
    print(INTERPRETATION_BOUNDARY)


def main() -> None:
    assert MOVE_ORDER == ("generic", "probing", "focus", "telling")
    started = time.perf_counter()
    validation_rows = read_mathdial_validation()
    mathdial_records = prepare_mathdial_records(validation_rows)
    demo_records = prepare_demo_records()
    assert len(mathdial_records) == EXPECTED_VALIDATION_ROWS
    assert len(demo_records) == EXPECTED_DEMO_ROWS
    assert OPENED_DATASET_PATHS == {VALIDATION_PATH.resolve()}
    assert all(path.name != "test.jsonl" for path in OPENED_DATASET_PATHS)

    mathdial_predictions, canonical_manifest = load_canonical_mathdial_predictions(
        mathdial_records
    )
    print(
        "MathDial validation: verified and loaded all 1,850 canonical "
        "frozen-MD6 probability rows.",
        flush=True,
    )
    assert canonical_manifest["canonical_reproduction"] == "PASS"

    model = FrozenMD6Inference()
    model._torch.set_num_threads(CPU_THREADS)
    demo_predictions = infer_records_public(
        model,
        demo_records,
        progress_label="Hand-written demo",
        progress_every=6,
    )
    validate_public_wrapper_parity(
        model,
        mathdial_records,
        mathdial_predictions,
    )
    validate_predictions(
        mathdial_predictions,
        EXPECTED_VALIDATION_ROWS,
        SOURCE_MATHDIAL,
    )
    validate_predictions(demo_predictions, EXPECTED_DEMO_ROWS, SOURCE_DEMO)

    summaries = {
        SOURCE_MATHDIAL: summarize_group(mathdial_predictions),
        SOURCE_DEMO: summarize_group(demo_predictions),
    }
    flat_summaries = [
        flatten_group_summary(source, summaries[source]) for source in SOURCE_GROUPS
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mathdial_path = OUTPUT_DIR / "mathdial_validation_predictions.csv.gz"
    demo_path = OUTPUT_DIR / "handwritten_demo_predictions.csv"
    group_summary_path = OUTPUT_DIR / "group_summary.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    manifest_path = OUTPUT_DIR / "manifest.json"
    figure_paths = create_figures(mathdial_predictions, demo_predictions)

    write_predictions(mathdial_predictions, mathdial_path, compressed=True)
    write_predictions(demo_predictions, demo_path, compressed=False)
    write_group_summary(flat_summaries, group_summary_path)
    summary_payload = {
        "audit_name": AUDIT_NAME,
        "purpose": PURPOSE,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "gap_threshold": GAP_THRESHOLD,
        "move_order": list(MOVE_ORDER),
        "groups": summaries,
        "comparison_table": [
            {
                "metric": metric,
                "mathdial_validation": mathdial,
                "handwritten_demo": demo,
            }
            for metric, mathdial, demo in comparison_metrics(summaries)
        ],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    runtime_seconds = time.perf_counter() - started
    output_files = [
        mathdial_path,
        demo_path,
        group_summary_path,
        summary_path,
        *figure_paths,
        manifest_path,
    ]
    manifest = build_manifest(output_files, runtime_seconds)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nMD6 VALIDATION VS DEMO BEHAVIOR AUDIT")
    print(f"MathDial validation N: {len(mathdial_predictions)}")
    print(f"Hand-written demo N: {len(demo_predictions)}")
    print(f"CPU runtime: {runtime_seconds:.1f} seconds")
    print_high_confidence_by_move(mathdial_predictions)
    print_comparison_table(summaries)
    print(f"\nOUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
