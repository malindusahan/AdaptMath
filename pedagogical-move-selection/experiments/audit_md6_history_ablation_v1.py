"""Paired frozen-MD6 audit under controlled conversation-history removal.

The audit uses MathDial validation only. Its primary paired cohort contains
exactly the examples with at least one prior student turn. Cases where
LATEST_STUDENT_ONLY is structurally undefined are reported separately, and no
history content is fabricated.
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

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.md6_inference import FrozenMD6Inference


AUDIT_NAME: Final[str] = "md6_history_ablation_v1"
VALIDATION_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "mathdial" / "validation.jsonl"
)
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "frozen" / "md6"
CANONICAL_PROBABILITIES_PATH: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_gap_threshold_audit"
    / "md6_validation_probabilities.csv"
)
CANONICAL_MANIFEST_PATH: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_gap_threshold_audit"
    / "audit_manifest.json"
)
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "self_improvement" / AUDIT_NAME
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"

FULL_HISTORY: Final[str] = "FULL_HISTORY"
LATEST_STUDENT_ONLY: Final[str] = "LATEST_STUDENT_ONLY"
STUDENT_TURNS_ONLY: Final[str] = "STUDENT_TURNS_ONLY"
CONDITIONS: Final[tuple[str, ...]] = (
    FULL_HISTORY,
    LATEST_STUDENT_ONLY,
    STUDENT_TURNS_ONLY,
)

ORIGINAL_VALIDATION_N: Final[int] = 1_850
PAIRED_COHORT_N: Final[int] = 1_548
EXCLUDED_N: Final[int] = 302
EXPECTED_EMPTY_EXCLUSIONS: Final[int] = 278
EXPECTED_TEACHER_ONLY_EXCLUSIONS: Final[int] = 24
EXPECTED_TOTAL: Final[int] = 4_644
GAP_THRESHOLD: Final[float] = 0.10
PROBABILITY_TOLERANCE: Final[float] = 1e-5
BOOTSTRAP_SEED: Final[int] = 20_260_825
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
CPU_THREADS: Final[int] = max(1, min(8, os.cpu_count() or 1))
OPENED_DATASET_PATHS: set[Path] = set()

RAW_FIELDS: Final[tuple[str, ...]] = (
    "row_id",
    "qid",
    "group_id",
    "gold_move",
    "condition",
    "original_history_turn_count",
    "condition_history_turn_count",
    "p_generic",
    "p_probing",
    "p_focus",
    "p_telling",
    "argmax_move",
    "full_history_argmax",
    "top1_probability",
    "second_move",
    "top2_probability",
    "top1_top2_gap",
    "entropy",
    "eligible_arms",
    "eligible_arm_count",
    "alternative_available",
)

SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "condition",
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
    "mean_entropy",
    "baseline_only_percentage",
    "alternative_available_percentage",
    "mean_eligible_arm_count",
)

BASELINE_COHORT_FIELDS: Final[tuple[str, ...]] = (
    "cohort",
    "n",
    "generic_argmax_percentage",
    "probing_argmax_percentage",
    "focus_argmax_percentage",
    "telling_argmax_percentage",
    "mean_probing_probability",
    "mean_top1_top2_gap",
    "baseline_only_percentage",
    "alternative_available_percentage",
)

PAIRED_FIELDS: Final[tuple[str, ...]] = (
    "comparison",
    "n_pairs",
    "argmax_changed_percentage",
    "changed_into_probing_percentage",
    "changed_out_of_probing_percentage",
    "mean_paired_change_p_probing",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "full_history_baseline_only_percentage",
    "ablated_baseline_only_percentage",
    "baseline_only_percentage_point_change",
    "baseline_only_status_changed_percentage",
    "changed_into_baseline_only_percentage",
    "changed_out_of_baseline_only_percentage",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation() -> list[dict[str, object]]:
    resolved = VALIDATION_PATH.resolve()
    expected = (
        PROJECT_ROOT / "data" / "processed" / "mathdial" / "validation.jsonl"
    ).resolve()
    if resolved != expected:
        raise RuntimeError("Dataset path guard rejected an unexpected path.")
    OPENED_DATASET_PATHS.add(resolved)
    rows: list[dict[str, object]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank validation row at line {line_number}.")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Validation row {line_number} is not an object.")
            rows.append(row)
    assert len(rows) == ORIGINAL_VALIDATION_N
    assert len({str(row["example_id"]) for row in rows}) == ORIGINAL_VALIDATION_N
    return rows


def is_student_turn(turn: Mapping[str, object]) -> bool:
    return str(turn.get("user", "")).strip().casefold() == "student"


def validate_and_build_histories(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, dict[str, list[Mapping[str, object]]]],
    list[Mapping[str, object]],
]:
    histories: dict[str, dict[str, list[Mapping[str, object]]]] = {}
    excluded: list[Mapping[str, object]] = []

    for row in rows:
        row_id = str(row["example_id"])
        history = row.get("history")
        if not isinstance(history, list):
            raise TypeError(f"{row_id}: history is not a list.")
        validated: list[Mapping[str, object]] = []
        for turn in history:
            if not isinstance(turn, Mapping):
                raise TypeError(f"{row_id}: history turn is not an object.")
            if not isinstance(turn.get("user"), str):
                raise TypeError(f"{row_id}: turn user is not text.")
            if not isinstance(turn.get("text"), str):
                raise TypeError(f"{row_id}: turn text is not text.")
            validated.append(turn)

        students = [turn for turn in validated if is_student_turn(turn)]
        if not students:
            excluded.append(row)
            continue
        histories[row_id] = {
            FULL_HISTORY: list(validated),
            LATEST_STUDENT_ONLY: [students[-1]],
            STUDENT_TURNS_ONLY: list(students),
        }

    all_ids = {str(row["example_id"]) for row in rows}
    paired_ids = set(histories)
    excluded_ids = {str(row["example_id"]) for row in excluded}
    empty_count = sum(len(row["history"]) == 0 for row in excluded)
    teacher_only_count = sum(len(row["history"]) > 0 for row in excluded)
    assert len(all_ids) == ORIGINAL_VALIDATION_N
    assert len(paired_ids) == PAIRED_COHORT_N
    assert len(excluded_ids) == EXCLUDED_N
    assert paired_ids.isdisjoint(excluded_ids)
    assert paired_ids | excluded_ids == all_ids
    assert empty_count == EXPECTED_EMPTY_EXCLUSIONS
    assert teacher_only_count == EXPECTED_TEACHER_ONLY_EXCLUSIONS
    assert all(
        not any(is_student_turn(turn) for turn in row["history"])
        for row in excluded
    )
    print("\nHISTORY-ABLATION COHORT PREFLIGHT: PASS", flush=True)
    print(f"Full validation: {len(all_ids)}", flush=True)
    print(f"Primary paired cohort: {len(paired_ids)}", flush=True)
    print(f"Structural exclusions: {len(excluded_ids)}", flush=True)
    print(f"  Empty history: {empty_count}", flush=True)
    print(f"  Teacher-only history: {teacher_only_count}", flush=True)
    return histories, excluded


def rank_moves(probabilities: Mapping[str, float]) -> list[str]:
    order = {move: index for index, move in enumerate(MOVE_ORDER)}
    return sorted(MOVE_ORDER, key=lambda move: (-probabilities[move], order[move]))


def entropy(probabilities: Mapping[str, float]) -> float:
    return float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )


def make_prediction(
    row: Mapping[str, object],
    condition: str,
    condition_history: Sequence[Mapping[str, object]],
    probabilities: Mapping[str, float],
) -> dict[str, object]:
    assert tuple(probabilities) == MOVE_ORDER
    ranked = rank_moves(probabilities)
    best, second = ranked[:2]
    candidates = eligible_arms(probabilities, gap_threshold=GAP_THRESHOLD)
    original_history = row["history"]
    assert isinstance(original_history, Sequence)
    return {
        "row_id": str(row["example_id"]),
        "qid": row.get("qid", ""),
        "group_id": row.get("group_id", ""),
        "gold_move": row.get("target_move", ""),
        "condition": condition,
        "original_history_turn_count": len(original_history),
        "condition_history_turn_count": len(condition_history),
        "p_generic": float(probabilities["generic"]),
        "p_probing": float(probabilities["probing"]),
        "p_focus": float(probabilities["focus"]),
        "p_telling": float(probabilities["telling"]),
        "argmax_move": best,
        "full_history_argmax": "",
        "top1_probability": float(probabilities[best]),
        "second_move": second,
        "top2_probability": float(probabilities[second]),
        "top1_top2_gap": float(probabilities[best] - probabilities[second]),
        "entropy": entropy(probabilities),
        "eligible_arms": json.dumps(list(candidates), separators=(",", ":")),
        "eligible_arm_count": len(candidates),
        "alternative_available": len(candidates) > 1,
    }


def load_full_history_predictions(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["canonical_reproduction"] == "PASS"
    assert manifest["validation_row_count"] == ORIGINAL_VALIDATION_N
    assert manifest["dataset_paths_read"] == [
        "data/processed/mathdial/validation.jsonl"
    ]
    assert tuple(manifest["model_class_order"]) == MOVE_ORDER
    assert sha256_file(CANONICAL_PROBABILITIES_PATH) == manifest[
        "output_file_sha256"
    ]["md6_validation_probabilities.csv"]
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

    with CANONICAL_PROBABILITIES_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        cached_rows = list(csv.DictReader(handle))
    assert len(cached_rows) == ORIGINAL_VALIDATION_N
    by_id = {cached["example_id"]: cached for cached in cached_rows}
    assert len(by_id) == ORIGINAL_VALIDATION_N

    predictions: list[dict[str, object]] = []
    for row in rows:
        row_id = str(row["example_id"])
        cached = by_id[row_id]
        assert cached["qid"] == str(row.get("qid", ""))
        assert cached["group_id"] == str(row.get("group_id", ""))
        probabilities = {
            move: float(cached[f"p_{move}"]) for move in MOVE_ORDER
        }
        prediction = make_prediction(
            row,
            FULL_HISTORY,
            row["history"],
            probabilities,
        )
        assert prediction["argmax_move"] == cached["base_move"]
        predictions.append(prediction)
    return predictions, manifest


def infer_condition(
    model: FrozenMD6Inference,
    rows: Sequence[Mapping[str, object]],
    histories: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    condition: str,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        row_id = str(row["example_id"])
        history = histories[row_id][condition]
        problem = row.get("problem")
        if not isinstance(problem, str):
            raise TypeError(f"{row_id}: problem is not text.")
        probabilities = model.predict_probabilities(problem, history)
        predictions.append(make_prediction(row, condition, history, probabilities))
        if index % 100 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(
                f"{condition}: {index}/{len(rows)} ({elapsed:.1f} seconds)",
                flush=True,
            )
    return predictions


def validate_probability_values(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        probabilities = [float(row[f"p_{move}"]) for move in MOVE_ORDER]
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
        assert math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE,
        )
        candidates = json.loads(str(row["eligible_arms"]))
        assert candidates and candidates[0] == "baseline"
        assert len(candidates) == int(row["eligible_arm_count"])
        assert bool(row["alternative_available"]) == (len(candidates) > 1)


def validate_predictions(rows: Sequence[Mapping[str, object]]) -> None:
    assert len(rows) == EXPECTED_TOTAL
    grouped = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    assert all(
        len(grouped[condition]) == PAIRED_COHORT_N for condition in CONDITIONS
    )
    paired_ids = [
        {str(row["row_id"]) for row in grouped[condition]}
        for condition in CONDITIONS
    ]
    assert paired_ids[0] == paired_ids[1] == paired_ids[2]
    assert len(paired_ids[0]) == PAIRED_COHORT_N
    assert GAP_THRESHOLD == 0.10
    validate_probability_values(rows)


def summarize_condition(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    counts = Counter(str(row["argmax_move"]) for row in rows)
    result: dict[str, object] = {"condition": rows[0]["condition"], "n": n}
    for move in MOVE_ORDER:
        values = [float(row[f"p_{move}"]) for row in rows]
        result[f"{move}_argmax_count"] = counts.get(move, 0)
        result[f"{move}_argmax_percentage"] = 100.0 * counts.get(move, 0) / n
        result[f"{move}_probability_mean"] = statistics.fmean(values)
        result[f"{move}_probability_median"] = statistics.median(values)
    top1 = [float(row["top1_probability"]) for row in rows]
    gaps = [float(row["top1_top2_gap"]) for row in rows]
    entropies = [float(row["entropy"]) for row in rows]
    baseline = [int(row["eligible_arm_count"]) == 1 for row in rows]
    eligible_counts = [int(row["eligible_arm_count"]) for row in rows]
    result.update(
        {
            "top1_confidence_mean": statistics.fmean(top1),
            "top1_confidence_median": statistics.median(top1),
            "top1_top2_gap_mean": statistics.fmean(gaps),
            "top1_top2_gap_median": statistics.median(gaps),
            "mean_entropy": statistics.fmean(entropies),
            "baseline_only_percentage": 100.0 * sum(baseline) / n,
            "alternative_available_percentage": 100.0 * (n - sum(baseline)) / n,
            "mean_eligible_arm_count": statistics.fmean(eligible_counts),
        }
    )
    return result


def summarize_baseline_cohort(
    cohort: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    n = len(rows)
    counts = Counter(str(row["argmax_move"]) for row in rows)
    baseline_count = sum(int(row["eligible_arm_count"]) == 1 for row in rows)
    return {
        "cohort": cohort,
        "n": n,
        **{
            f"{move}_argmax_percentage": 100.0 * counts.get(move, 0) / n
            for move in MOVE_ORDER
        },
        "mean_probing_probability": statistics.fmean(
            float(row["p_probing"]) for row in rows
        ),
        "mean_top1_top2_gap": statistics.fmean(
            float(row["top1_top2_gap"]) for row in rows
        ),
        "baseline_only_percentage": 100.0 * baseline_count / n,
        "alternative_available_percentage": 100.0 * (n - baseline_count) / n,
    }


def bootstrap_mean_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = differences.size
    bootstrap_means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975]))


def paired_statistics(
    full_rows: Sequence[Mapping[str, object]],
    ablated_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    full = {str(row["row_id"]): row for row in full_rows}
    ablated = {str(row["row_id"]): row for row in ablated_rows}
    assert full.keys() == ablated.keys()
    ordered_ids = sorted(full)
    n = len(ordered_ids)
    full_moves = [str(full[row_id]["argmax_move"]) for row_id in ordered_ids]
    ablated_moves = [str(ablated[row_id]["argmax_move"]) for row_id in ordered_ids]
    differences = np.asarray(
        [
            float(ablated[row_id]["p_probing"])
            - float(full[row_id]["p_probing"])
            for row_id in ordered_ids
        ],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_mean_ci(differences)
    full_baseline = np.asarray(
        [int(full[row_id]["eligible_arm_count"]) == 1 for row_id in ordered_ids]
    )
    ablated_baseline = np.asarray(
        [int(ablated[row_id]["eligible_arm_count"]) == 1 for row_id in ordered_ids]
    )
    return {
        "comparison": f"{ablated_rows[0]['condition']} vs {FULL_HISTORY}",
        "n_pairs": n,
        "argmax_changed_percentage": 100.0
        * sum(before != after for before, after in zip(full_moves, ablated_moves, strict=True))
        / n,
        "changed_into_probing_percentage": 100.0
        * sum(before != "probing" and after == "probing" for before, after in zip(full_moves, ablated_moves, strict=True))
        / n,
        "changed_out_of_probing_percentage": 100.0
        * sum(before == "probing" and after != "probing" for before, after in zip(full_moves, ablated_moves, strict=True))
        / n,
        "mean_paired_change_p_probing": float(differences.mean()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "full_history_baseline_only_percentage": float(100.0 * full_baseline.mean()),
        "ablated_baseline_only_percentage": float(100.0 * ablated_baseline.mean()),
        "baseline_only_percentage_point_change": float(
            100.0 * (ablated_baseline.mean() - full_baseline.mean())
        ),
        "baseline_only_status_changed_percentage": float(
            100.0 * np.mean(full_baseline != ablated_baseline)
        ),
        "changed_into_baseline_only_percentage": float(
            100.0 * np.mean(~full_baseline & ablated_baseline)
        ),
        "changed_out_of_baseline_only_percentage": float(
            100.0 * np.mean(full_baseline & ~ablated_baseline)
        ),
    }


def transition_rows(
    full_rows: Sequence[Mapping[str, object]],
    ablated_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    ablated = {str(row["row_id"]): row for row in ablated_rows}
    table = {move: Counter() for move in MOVE_ORDER}
    for row in full_rows:
        table[str(row["argmax_move"])][
            str(ablated[str(row["row_id"])]["argmax_move"])
        ] += 1
    return [
        {
            "full_history_argmax": source,
            **{target: table[source].get(target, 0) for target in MOVE_ORDER},
        }
        for source in MOVE_ORDER
    ]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_raw(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RAW_FIELDS})


def make_argmax_figure(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    counts = Counter(str(row["argmax_move"]) for row in rows)
    figure = plt.figure(figsize=(8, 5))
    plt.bar(MOVE_ORDER, [counts.get(move, 0) for move in MOVE_ORDER])
    plt.title(f"Frozen MD6 argmax distribution: {rows[0]['condition']}")
    plt.xlabel("Argmax move")
    plt.ylabel("Number of validation examples")
    plt.ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def make_boxplot(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    field: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    figure = plt.figure(figsize=(9, 5))
    plt.boxplot(
        [[float(row[field]) for row in grouped[condition]] for condition in CONDITIONS],
        tick_labels=list(CONDITIONS),
        showfliers=False,
    )
    plt.title(title)
    plt.xlabel("History condition")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def build_manifest(output_files: Sequence[Path], runtime_seconds: float) -> dict[str, object]:
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "experiments" / "audit_md6_validation_vs_demo_v1.py",
        PROJECT_ROOT / "src" / "self_improvement" / "md6_inference.py",
        PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py",
        PROJECT_ROOT / "src" / "self_improvement" / "context_builder.py",
    )
    return {
        "audit_name": AUDIT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Paired descriptive audit of frozen MD6 behavior under controlled "
            "removal of original conversation-history turns."
        ),
        "interpretation_boundary": (
            "No ablated-condition accuracy/F1 is calculated; differences are "
            "descriptive and are not automatically evidence of causal domain shift."
        ),
        "dataset_scope": "MathDial validation only",
        "dataset_paths_opened": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in sorted(OPENED_DATASET_PATHS)
        ],
        "held_out_test_untouched": True,
        "original_validation_n": ORIGINAL_VALIDATION_N,
        "primary_paired_cohort_n": PAIRED_COHORT_N,
        "excluded_from_paired_ablation_n": EXCLUDED_N,
        "exclusion_reason": (
            "No prior student turn; LATEST_STUDENT_ONLY is therefore undefined."
        ),
        "exclusions_determined_before_md6_ablation_inference": True,
        "excluded_history_structure": {
            "empty_history": EXPECTED_EMPTY_EXCLUSIONS,
            "teacher_only_history": EXPECTED_TEACHER_ONLY_EXCLUSIONS,
        },
        "history_fabricated": False,
        "conditions": list(CONDITIONS),
        "condition_definitions": {
            FULL_HISTORY: "Original complete history unchanged.",
            LATEST_STUDENT_ONLY: "Exact most recent original Student turn only.",
            STUDENT_TURNS_ONLY: "All exact original Student turns in chronological order.",
        },
        "frozen_md6": True,
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "gap_threshold": GAP_THRESHOLD,
        "model_changes": False,
        "training_performed": False,
        "cpu_only": True,
        "cpu_threads": CPU_THREADS,
        "bootstrap": {
            "unit": "paired validation example",
            "method": "percentile bootstrap of mean paired p_probing change",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "canonical_full_history_probability_source": CANONICAL_PROBABILITIES_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "canonical_reproduction_manifest": CANONICAL_MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "source_file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in sources
        },
        "validation_dataset_sha256": sha256_file(VALIDATION_PATH),
        "canonical_probability_artifact_sha256": sha256_file(
            CANONICAL_PROBABILITIES_PATH
        ),
        "base_audit_manifest_sha256": sha256_file(CANONICAL_MANIFEST_PATH),
        "frozen_model_sha256": {
            path.name: sha256_file(path)
            for path in (
                MODEL_PATH / "config.json",
                MODEL_PATH / "model.safetensors",
                MODEL_PATH / "tokenizer.json",
                MODEL_PATH / "tokenizer_config.json",
            )
        },
        "output_files": [
            path.relative_to(PROJECT_ROOT).as_posix() for path in output_files
        ],
        "runtime_seconds": runtime_seconds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def print_results(
    baseline_cohorts: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
) -> None:
    print("\n" + "=" * 89)
    print("FULL-HISTORY COHORT COMPARISON")
    print("-" * 89)
    print("cohort                   N   generic  probing    focus  telling  baseline-only")
    for cohort in baseline_cohorts:
        print(
            f"{str(cohort['cohort']):<22}"
            f"{int(cohort['n']):>5}"
            f"{float(cohort['generic_argmax_percentage']):>9.2f}%"
            f"{float(cohort['probing_argmax_percentage']):>9.2f}%"
            f"{float(cohort['focus_argmax_percentage']):>9.2f}%"
            f"{float(cohort['telling_argmax_percentage']):>9.2f}%"
            f"{float(cohort['baseline_only_percentage']):>14.2f}%"
        )
    print("=" * 89)
    print(f"\nHISTORY ABLATION - PAIRED N={PAIRED_COHORT_N}")
    print("\n" + "=" * 89)
    print("CONDITION                GENERIC   PROBING   FOCUS   TELLING   BASELINE_ONLY")
    print("-" * 89)
    for summary in summaries:
        print(
            f"{str(summary['condition']):<25}"
            f"{float(summary['generic_argmax_percentage']):>8.2f}%"
            f"{float(summary['probing_argmax_percentage']):>10.2f}%"
            f"{float(summary['focus_argmax_percentage']):>8.2f}%"
            f"{float(summary['telling_argmax_percentage']):>10.2f}%"
            f"{float(summary['baseline_only_percentage']):>15.2f}%"
        )
    print("=" * 89)
    for result in paired:
        condition = str(result["comparison"]).split(" vs ", maxsplit=1)[0]
        print(f"\n{condition}:")
        print(f"  argmax changed %: {float(result['argmax_changed_percentage']):.4f}")
        print(
            "  changed into probing %: "
            f"{float(result['changed_into_probing_percentage']):.4f}"
        )
        print(
            "  changed out of probing %: "
            f"{float(result['changed_out_of_probing_percentage']):.4f}"
        )
        print(
            f"  mean delta p_probing: {float(result['mean_paired_change_p_probing']):.6f}"
        )
        print(
            "  bootstrap 95% CI: "
            f"[{float(result['bootstrap_ci_low']):.6f}, "
            f"{float(result['bootstrap_ci_high']):.6f}]"
        )
        print(
            "  delta baseline-only percentage points: "
            f"{float(result['baseline_only_percentage_point_change']):.4f}"
        )


def main() -> None:
    assert MOVE_ORDER == ("generic", "probing", "focus", "telling")
    assert GAP_THRESHOLD == 0.10
    started = time.perf_counter()
    validation_rows = read_validation()
    assert OPENED_DATASET_PATHS == {VALIDATION_PATH.resolve()}
    assert all(path.name != "test.jsonl" for path in OPENED_DATASET_PATHS)

    # Cohort membership is fixed before model loading or ablation inference.
    histories, excluded_rows = validate_and_build_histories(validation_rows)
    paired_ids = set(histories)
    excluded_ids = {str(row["example_id"]) for row in excluded_rows}
    all_ids = {str(row["example_id"]) for row in validation_rows}
    assert len(all_ids) == ORIGINAL_VALIDATION_N
    assert len(paired_ids) == PAIRED_COHORT_N
    assert len(excluded_ids) == EXCLUDED_N
    assert paired_ids.isdisjoint(excluded_ids)
    assert paired_ids | excluded_ids == all_ids
    paired_rows = [
        row for row in validation_rows if str(row["example_id"]) in paired_ids
    ]
    assert len(paired_rows) == PAIRED_COHORT_N

    full_validation_predictions, canonical_manifest = load_full_history_predictions(
        validation_rows
    )
    assert canonical_manifest["canonical_reproduction"] == "PASS"
    validate_probability_values(full_validation_predictions)
    full_by_id = {
        str(row["row_id"]): row for row in full_validation_predictions
    }
    assert len(full_by_id) == ORIGINAL_VALIDATION_N
    full_predictions = [full_by_id[str(row["example_id"])] for row in paired_rows]
    excluded_full_predictions = [
        full_by_id[str(row["example_id"])] for row in excluded_rows
    ]
    assert len(full_predictions) == PAIRED_COHORT_N
    assert len(excluded_full_predictions) == EXCLUDED_N

    model = FrozenMD6Inference()
    model._torch.set_num_threads(CPU_THREADS)
    latest_predictions = infer_condition(
        model, paired_rows, histories, LATEST_STUDENT_ONLY
    )
    student_predictions = infer_condition(
        model, paired_rows, histories, STUDENT_TURNS_ONLY
    )

    full_argmax = {
        str(row["row_id"]): str(row["argmax_move"]) for row in full_predictions
    }
    all_predictions = [
        *full_predictions,
        *latest_predictions,
        *student_predictions,
    ]
    for row in all_predictions:
        row["full_history_argmax"] = full_argmax[str(row["row_id"])]
    validate_predictions(all_predictions)

    grouped = {
        condition: [row for row in all_predictions if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summaries = [summarize_condition(grouped[condition]) for condition in CONDITIONS]
    paired = [
        paired_statistics(full_predictions, latest_predictions),
        paired_statistics(full_predictions, student_predictions),
    ]
    latest_transition = transition_rows(full_predictions, latest_predictions)
    student_transition = transition_rows(full_predictions, student_predictions)
    baseline_cohorts = [
        summarize_baseline_cohort(
            "all-validation", full_validation_predictions
        ),
        summarize_baseline_cohort("paired-cohort", full_predictions),
        summarize_baseline_cohort(
            "no-prior-student", excluded_full_predictions
        ),
    ]
    excluded_argmax_counts = Counter(
        str(row["argmax_move"]) for row in excluded_full_predictions
    )
    exclusion_payload = {
        "exclusion_type": "structural design exclusion",
        "reason": (
            "No prior student turn; LATEST_STUDENT_ONLY is undefined."
        ),
        "determined_before_md6_ablation_inference": True,
        "no_history_fabricated": True,
        "no_prior_student_total": len(excluded_rows),
        "empty_history": sum(len(row["history"]) == 0 for row in excluded_rows),
        "teacher_only_history": sum(
            len(row["history"]) > 0 for row in excluded_rows
        ),
        "full_history_argmax_counts": {
            move: excluded_argmax_counts.get(move, 0) for move in MOVE_ORDER
        },
        "full_history_argmax_percentages": {
            move: 100.0 * excluded_argmax_counts.get(move, 0) / EXCLUDED_N
            for move in MOVE_ORDER
        },
        "excluded_row_ids": sorted(excluded_ids),
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "raw_predictions.csv.gz"
    condition_path = OUTPUT_DIR / "condition_summary.csv"
    paired_path = OUTPUT_DIR / "paired_statistics.csv"
    latest_transition_path = OUTPUT_DIR / "transition_latest_student_only.csv"
    student_transition_path = OUTPUT_DIR / "transition_student_turns_only.csv"
    baseline_cohort_path = OUTPUT_DIR / "baseline_cohort_comparison.csv"
    exclusion_path = OUTPUT_DIR / "exclusion_summary.json"
    summary_path = OUTPUT_DIR / "summary.json"
    manifest_path = OUTPUT_DIR / "manifest.json"
    figure_paths = {
        FULL_HISTORY: FIGURE_DIR / "argmax_distribution_full_history.png",
        LATEST_STUDENT_ONLY: FIGURE_DIR
        / "argmax_distribution_latest_student_only.png",
        STUDENT_TURNS_ONLY: FIGURE_DIR
        / "argmax_distribution_student_turns_only.png",
    }
    for condition in CONDITIONS:
        make_argmax_figure(grouped[condition], figure_paths[condition])
    probing_figure = FIGURE_DIR / "probing_probability_by_condition.png"
    gap_figure = FIGURE_DIR / "gap_by_condition.png"
    make_boxplot(
        grouped,
        "p_probing",
        probing_figure,
        "Frozen MD6 probing probability by history condition",
        "Probing probability",
    )
    make_boxplot(
        grouped,
        "top1_top2_gap",
        gap_figure,
        "Frozen MD6 top-1 minus top-2 gap by history condition",
        "Top-1 minus top-2 probability gap",
    )

    write_raw(raw_path, all_predictions)
    write_csv(condition_path, summaries, SUMMARY_FIELDS)
    write_csv(paired_path, paired, PAIRED_FIELDS)
    transition_fields = ("full_history_argmax", *MOVE_ORDER)
    write_csv(latest_transition_path, latest_transition, transition_fields)
    write_csv(student_transition_path, student_transition, transition_fields)
    write_csv(baseline_cohort_path, baseline_cohorts, BASELINE_COHORT_FIELDS)
    exclusion_path.write_text(
        json.dumps(exclusion_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "audit_name": AUDIT_NAME,
        "interpretation_boundary": (
            "Descriptive paired behavior evidence only; no ablated accuracy/F1 "
            "and no automatic causal domain-shift claim."
        ),
        "original_validation_n": ORIGINAL_VALIDATION_N,
        "primary_paired_cohort_n": PAIRED_COHORT_N,
        "structural_exclusion_n": EXCLUDED_N,
        "baseline_cohort_comparison": baseline_cohorts,
        "exclusion_summary": exclusion_payload,
        "condition_summaries": summaries,
        "paired_statistics": paired,
        "transitions": {
            LATEST_STUDENT_ONLY: latest_transition,
            STUDENT_TURNS_ONLY: student_transition,
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    runtime_seconds = time.perf_counter() - started
    outputs = [
        raw_path,
        condition_path,
        paired_path,
        latest_transition_path,
        student_transition_path,
        baseline_cohort_path,
        exclusion_path,
        summary_path,
        *figure_paths.values(),
        probing_figure,
        gap_figure,
        manifest_path,
    ]
    manifest = build_manifest(outputs, runtime_seconds)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_results(baseline_cohorts, summaries, paired)
    print(f"\nOUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
