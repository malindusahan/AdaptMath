"""Describe frozen-MD6 behavior and overlay opportunity by history depth.

The audit reuses the complete canonical MathDial-validation prediction
artifact. It does not rerun MD6, open a dataset split, train a model, load
MRB1 or LinTS, or update policy state.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.self_improvement.conservative_overlay import eligible_arms
from src.self_improvement.context_builder import MOVE_ORDER


AUDIT_NAME: Final[str] = "md6_opportunity_by_history_depth_v1"
PURPOSE: Final[str] = (
    "Measure how frozen MD6 prediction behavior and conservative-overlay "
    "opportunity vary with dialogue-history depth on MathDial validation."
)
EXPECTED_N: Final[int] = 1_850
GAP_THRESHOLD: Final[float] = 0.10
PROBABILITY_TOLERANCE: Final[float] = 1e-5

INPUT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_validation_vs_demo_v1"
    / "mathdial_validation_predictions.csv.gz"
)
INPUT_MANIFEST_PATH: Final[Path] = INPUT_PATH.parent / "manifest.json"
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "self_improvement" / AUDIT_NAME
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"

DEPTH_BINS: Final[tuple[str, ...]] = (
    "0 turns",
    "1-2 turns",
    "3-4 turns",
    "5-8 turns",
    "9+ turns",
)
STUDENT_GROUPS: Final[tuple[str, ...]] = (
    "0 prior student turns",
    "1 prior student turn",
    "2 prior student turns",
    "3+ prior student turns",
)

ROW_FIELDS: Final[tuple[str, ...]] = (
    "row_id",
    "qid",
    "group_id",
    "gold_move",
    "history_turn_count",
    "student_turn_count",
    "teacher_turn_count",
    "latest_turn_role",
    "has_prior_student_turn",
    "history_depth_bin",
    "student_turn_group",
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

METRIC_FIELDS: Final[tuple[str, ...]] = (
    "N",
    "generic_argmax_count",
    "generic_argmax_percentage",
    "probing_argmax_count",
    "probing_argmax_percentage",
    "focus_argmax_count",
    "focus_argmax_percentage",
    "telling_argmax_count",
    "telling_argmax_percentage",
    "mean_p_generic",
    "mean_p_probing",
    "mean_p_focus",
    "mean_p_telling",
    "mean_top1_probability",
    "median_top1_probability",
    "mean_top1_top2_gap",
    "median_top1_top2_gap",
    "mean_entropy",
    "median_entropy",
    "baseline_only_count",
    "baseline_only_percentage",
    "alternative_available_count",
    "alternative_available_percentage",
    "mean_eligible_arm_count",
)

BASE_FIELDS: Final[tuple[str, ...]] = (
    "base_move",
    "N",
    "mean_top1_top2_gap",
    "baseline_only_count",
    "baseline_only_percentage",
    "alternative_available_count",
    "alternative_available_percentage",
    "mean_eligible_arm_count",
)

BASE_DEPTH_FIELDS: Final[tuple[str, ...]] = (
    "base_move",
    "history_depth_bin",
    "N",
    "mean_top1_top2_gap",
    "baseline_only_count",
    "baseline_only_percentage",
    "alternative_available_count",
    "alternative_available_percentage",
    "mean_eligible_arm_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected serialized boolean, got {value!r}.")


def depth_bin(turn_count: int) -> str:
    if turn_count == 0:
        return DEPTH_BINS[0]
    if turn_count <= 2:
        return DEPTH_BINS[1]
    if turn_count <= 4:
        return DEPTH_BINS[2]
    if turn_count <= 8:
        return DEPTH_BINS[3]
    return DEPTH_BINS[4]


def student_group(student_count: int) -> str:
    if student_count == 0:
        return STUDENT_GROUPS[0]
    if student_count == 1:
        return STUDENT_GROUPS[1]
    if student_count == 2:
        return STUDENT_GROUPS[2]
    return STUDENT_GROUPS[3]


def normalized_role(value: object) -> str:
    role = str(value).strip().casefold()
    if role == "student":
        return "Student"
    if role == "teacher":
        return "Teacher"
    raise ValueError(f"Unexpected MathDial history role: {value!r}.")


def ranked_moves(probabilities: Mapping[str, float]) -> list[str]:
    order = {move: index for index, move in enumerate(MOVE_ORDER)}
    return sorted(
        MOVE_ORDER,
        key=lambda move: (-probabilities[move], order[move]),
    )


def calculated_entropy(probabilities: Mapping[str, float]) -> float:
    return float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )


def load_and_validate_input() -> tuple[list[dict[str, object]], list[str]]:
    checks: list[str] = []
    manifest = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["mathdial_validation_n"] == EXPECTED_N
    assert manifest["dataset_scope"] == "MathDial validation only"
    assert manifest["held_out_test_untouched"] is True
    assert manifest["frozen_md6"] is True
    assert math.isclose(
        float(manifest["frozen_gap_threshold"]),
        GAP_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert tuple(manifest["canonical_move_order"]) == MOVE_ORDER
    checks.append("input manifest identifies 1,850 frozen-MD6 validation rows")
    checks.append("input manifest records held-out test untouched")
    checks.append("input manifest threshold and class order match this audit")

    with gzip.open(INPUT_PATH, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Canonical prediction artifact has no header.")
        required = {
            "source_group",
            "row_id",
            "qid",
            "group_id",
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
        }
        assert required <= set(fieldnames)
        probability_columns = tuple(
            fieldnames[fieldnames.index("p_generic") : fieldnames.index("p_telling") + 1]
        )
        assert probability_columns == tuple(f"p_{move}" for move in MOVE_ORDER)
        source_rows = list(reader)
    assert len(source_rows) == EXPECTED_N
    assert len({row["row_id"] for row in source_rows}) == EXPECTED_N
    checks.append("canonical artifact has exactly 1,850 unique rows")
    checks.append("canonical move-probability columns exist in canonical order")

    output: list[dict[str, object]] = []
    for source in source_rows:
        assert source["source_group"] == "mathdial_validation"
        probabilities = {
            move: float(source[f"p_{move}"]) for move in MOVE_ORDER
        }
        assert tuple(probabilities) == MOVE_ORDER
        assert all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in probabilities.values()
        )
        assert math.isclose(
            sum(probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE,
        )
        ranked = ranked_moves(probabilities)
        assert source["argmax_move"] == ranked[0]
        assert source["second_move"] == ranked[1]
        assert math.isclose(
            float(source["top1_probability"]),
            probabilities[ranked[0]],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            float(source["top2_probability"]),
            probabilities[ranked[1]],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        gap = probabilities[ranked[0]] - probabilities[ranked[1]]
        assert math.isclose(
            float(source["top1_top2_gap"]), gap, rel_tol=0.0, abs_tol=1e-12
        )
        entropy = calculated_entropy(probabilities)
        assert math.isclose(
            float(source["entropy"]), entropy, rel_tol=0.0, abs_tol=1e-12
        )

        recomputed_arms = eligible_arms(
            probabilities,
            gap_threshold=GAP_THRESHOLD,
        )
        stored_arms = json.loads(source["eligible_arms"])
        assert stored_arms == list(recomputed_arms)
        assert recomputed_arms[0] == "baseline"
        assert int(source["eligible_arm_count"]) == len(recomputed_arms)
        assert parse_bool(source["alternative_available"]) == (
            len(recomputed_arms) > 1
        )

        history = json.loads(source["conversation_history"])
        if not isinstance(history, list):
            raise TypeError("Serialized conversation_history must decode to a list.")
        roles: list[str] = []
        for turn in history:
            if not isinstance(turn, Mapping):
                raise TypeError("Every parsed history turn must be an object.")
            if not isinstance(turn.get("text"), str):
                raise TypeError("Every parsed history turn requires text.")
            roles.append(normalized_role(turn.get("user")))
        history_count = len(history)
        student_count = roles.count("Student")
        teacher_count = roles.count("Teacher")
        assert student_count + teacher_count == history_count
        latest_role = roles[-1] if roles else "NONE"
        row = {
            "row_id": source["row_id"],
            "qid": source["qid"],
            "group_id": source["group_id"],
            "gold_move": source["gold_move"],
            "history_turn_count": history_count,
            "student_turn_count": student_count,
            "teacher_turn_count": teacher_count,
            "latest_turn_role": latest_role,
            "has_prior_student_turn": student_count > 0,
            "history_depth_bin": depth_bin(history_count),
            "student_turn_group": student_group(student_count),
            **{f"p_{move}": probabilities[move] for move in MOVE_ORDER},
            "argmax_move": ranked[0],
            "top1_probability": probabilities[ranked[0]],
            "second_move": ranked[1],
            "top2_probability": probabilities[ranked[1]],
            "top1_top2_gap": gap,
            "entropy": entropy,
            "eligible_arms": json.dumps(list(recomputed_arms), separators=(",", ":")),
            "eligible_arm_count": len(recomputed_arms),
            "alternative_available": len(recomputed_arms) > 1,
        }
        assert int(row["history_turn_count"]) == len(history)
        output.append(row)

    checks.append("all probabilities are finite, bounded, and sum approximately to 1")
    checks.append("stored argmax, gap, entropy, and eligibility match recomputation")
    checks.append("baseline is always eligible at frozen threshold 0.10")
    checks.append("serialized histories parse safely and counts match parsed length")
    return output, checks


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    counts = Counter(str(row["argmax_move"]) for row in rows)
    baseline_count = sum(int(row["eligible_arm_count"]) == 1 for row in rows)
    result: dict[str, object] = {"N": n}
    for move in MOVE_ORDER:
        result[f"{move}_argmax_count"] = counts.get(move, 0)
        result[f"{move}_argmax_percentage"] = 100.0 * counts.get(move, 0) / n
        result[f"mean_p_{move}"] = statistics.fmean(
            float(row[f"p_{move}"]) for row in rows
        )
    for field in ("top1_probability", "top1_top2_gap", "entropy"):
        values = [float(row[field]) for row in rows]
        result[f"mean_{field}"] = statistics.fmean(values)
        result[f"median_{field}"] = statistics.median(values)
    result.update(
        {
            "baseline_only_count": baseline_count,
            "baseline_only_percentage": 100.0 * baseline_count / n,
            "alternative_available_count": n - baseline_count,
            "alternative_available_percentage": 100.0 * (n - baseline_count) / n,
            "mean_eligible_arm_count": statistics.fmean(
                int(row["eligible_arm_count"]) for row in rows
            ),
        }
    )
    return result


def summarize_groups(
    rows: Sequence[Mapping[str, object]],
    group_field: str,
    ordered_groups: Sequence[str],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for group in ordered_groups:
        members = [row for row in rows if row[group_field] == group]
        if not members:
            raise AssertionError(f"Pre-specified group {group!r} has no rows.")
        summaries.append({group_field: group, **summarize(members)})
    return summaries


def summarize_opportunity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    n = len(rows)
    baseline_count = sum(int(row["eligible_arm_count"]) == 1 for row in rows)
    return {
        "N": n,
        "mean_top1_top2_gap": statistics.fmean(
            float(row["top1_top2_gap"]) for row in rows
        ),
        "baseline_only_count": baseline_count,
        "baseline_only_percentage": 100.0 * baseline_count / n,
        "alternative_available_count": n - baseline_count,
        "alternative_available_percentage": 100.0 * (n - baseline_count) / n,
        "mean_eligible_arm_count": statistics.fmean(
            int(row["eligible_arm_count"]) for row in rows
        ),
    }


def base_move_summaries(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    overall: list[dict[str, object]] = []
    by_depth: list[dict[str, object]] = []
    for move in MOVE_ORDER:
        move_rows = [row for row in rows if row["argmax_move"] == move]
        assert move_rows
        overall.append({"base_move": move, **summarize_opportunity(move_rows)})
        for depth in DEPTH_BINS:
            members = [row for row in move_rows if row["history_depth_bin"] == depth]
            if members:
                by_depth.append(
                    {
                        "base_move": move,
                        "history_depth_bin": depth,
                        **summarize_opportunity(members),
                    }
                )
    return overall, by_depth


def validate_assignments(rows: Sequence[Mapping[str, object]]) -> list[str]:
    checks: list[str] = []
    assert len(rows) == EXPECTED_N
    assert all(row["history_depth_bin"] in DEPTH_BINS for row in rows)
    depth_counts = Counter(str(row["history_depth_bin"]) for row in rows)
    assert sum(depth_counts.values()) == EXPECTED_N
    assert all(row["student_turn_group"] in STUDENT_GROUPS for row in rows)
    student_counts = Counter(str(row["student_turn_group"]) for row in rows)
    assert sum(student_counts.values()) == EXPECTED_N
    assert GAP_THRESHOLD == 0.10
    checks.append("every row is assigned exactly one fixed history-depth bin")
    checks.append("history-depth bin counts sum to 1,850")
    checks.append("every row is assigned exactly one fixed student-turn group")
    checks.append("student-turn group counts sum to 1,850")
    checks.append("threshold remains exactly 0.10")
    return checks


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    *,
    compressed: bool = False,
) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def create_figures(
    depth_summaries: Sequence[Mapping[str, object]],
    student_summaries: Sequence[Mapping[str, object]],
) -> list[Path]:
    alternative_depth_path = (
        FIGURE_DIR / "alternative_availability_by_history_depth.png"
    )
    argmax_depth_path = FIGURE_DIR / "argmax_distribution_by_history_depth.png"
    probability_depth_path = (
        FIGURE_DIR / "mean_move_probabilities_by_history_depth.png"
    )
    alternative_student_path = (
        FIGURE_DIR / "alternative_availability_by_student_turn_count.png"
    )

    x = np.arange(len(DEPTH_BINS), dtype=np.float64)
    figure = plt.figure(figsize=(9, 5))
    plt.bar(
        x,
        [float(row["alternative_available_percentage"]) for row in depth_summaries],
    )
    plt.title("Alternative availability by dialogue-history depth")
    plt.xlabel("History-depth bin")
    plt.ylabel("Examples with an eligible alternative (%)")
    plt.xticks(x, DEPTH_BINS, rotation=15)
    plt.ylim(0.0, 100.0)
    figure.tight_layout()
    figure.savefig(alternative_depth_path, dpi=200)
    plt.close(figure)

    width = 0.18
    figure = plt.figure(figsize=(10, 5))
    for index, move in enumerate(MOVE_ORDER):
        plt.bar(
            x + (index - 1.5) * width,
            [float(row[f"{move}_argmax_percentage"]) for row in depth_summaries],
            width,
            label=move,
        )
    plt.title("Frozen MD6 argmax distribution by history depth")
    plt.xlabel("History-depth bin")
    plt.ylabel("Argmax percentage")
    plt.xticks(x, DEPTH_BINS, rotation=15)
    plt.ylim(0.0, 100.0)
    plt.legend(title="Argmax move")
    figure.tight_layout()
    figure.savefig(argmax_depth_path, dpi=200)
    plt.close(figure)

    figure = plt.figure(figsize=(10, 5))
    for index, move in enumerate(MOVE_ORDER):
        plt.bar(
            x + (index - 1.5) * width,
            [float(row[f"mean_p_{move}"]) for row in depth_summaries],
            width,
            label=move,
        )
    plt.title("Mean frozen MD6 move probabilities by history depth")
    plt.xlabel("History-depth bin")
    plt.ylabel("Mean probability")
    plt.xticks(x, DEPTH_BINS, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.legend(title="Move")
    figure.tight_layout()
    figure.savefig(probability_depth_path, dpi=200)
    plt.close(figure)

    student_x = np.arange(len(STUDENT_GROUPS), dtype=np.float64)
    figure = plt.figure(figsize=(10, 5))
    plt.bar(
        student_x,
        [
            float(row["alternative_available_percentage"])
            for row in student_summaries
        ],
    )
    plt.title("Alternative availability by prior Student-turn count")
    plt.xlabel("Student-turn-count group")
    plt.ylabel("Examples with an eligible alternative (%)")
    plt.xticks(student_x, STUDENT_GROUPS, rotation=10)
    plt.ylim(0.0, 100.0)
    figure.tight_layout()
    figure.savefig(alternative_student_path, dpi=200)
    plt.close(figure)
    return [
        alternative_depth_path,
        argmax_depth_path,
        probability_depth_path,
        alternative_student_path,
    ]


def build_manifest(
    outputs: Sequence[Path],
    checks: Sequence[str],
    runtime_seconds: float,
) -> dict[str, object]:
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py",
        PROJECT_ROOT / "src" / "self_improvement" / "context_builder.py",
    )
    return {
        "audit_name": AUDIT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": PURPOSE,
        "interpretation_boundary": "Descriptive analysis only; no significance tests.",
        "input_artifact": INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "input_artifact_sha256": sha256_file(INPUT_PATH),
        "input_manifest": INPUT_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
        "validation_n": EXPECTED_N,
        "history_depth_bins": list(DEPTH_BINS),
        "student_turn_groups": list(STUDENT_GROUPS),
        "gap_threshold": GAP_THRESHOLD,
        "md6_rerun": False,
        "cpu_only": True,
        "training_performed": False,
        "mrb1_loaded": False,
        "lints_state_loaded": False,
        "policy_updates": False,
        "dataset_files_opened": [],
        "held_out_test_untouched": True,
        "source_file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in sources
        },
        "validation_assertions": [
            {"assertion": check, "status": "PASS"} for check in checks
        ],
        "output_files": [
            path.relative_to(PROJECT_ROOT).as_posix() for path in outputs
        ],
        "runtime_seconds": runtime_seconds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def print_table(
    title: str,
    group_field: str,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    print(f"\n{title}")
    print(
        f"{'Group':<24} {'N':>5} {'Generic':>9} {'Probing':>9} "
        f"{'Focus':>9} {'Telling':>9} {'Baseline':>10} {'Alternative':>12}"
    )
    for row in summaries:
        print(
            f"{str(row[group_field]):<24} {int(row['N']):>5} "
            f"{float(row['generic_argmax_percentage']):>8.2f}% "
            f"{float(row['probing_argmax_percentage']):>8.2f}% "
            f"{float(row['focus_argmax_percentage']):>8.2f}% "
            f"{float(row['telling_argmax_percentage']):>8.2f}% "
            f"{float(row['baseline_only_percentage']):>9.2f}% "
            f"{float(row['alternative_available_percentage']):>11.2f}%"
        )


def main() -> None:
    started = time.perf_counter()
    assert MOVE_ORDER == ("generic", "probing", "focus", "telling")
    assert GAP_THRESHOLD == 0.10
    rows, input_checks = load_and_validate_input()
    assignment_checks = validate_assignments(rows)
    checks = [*input_checks, *assignment_checks]

    depth_summaries = summarize_groups(
        rows,
        "history_depth_bin",
        DEPTH_BINS,
    )
    student_summaries = summarize_groups(
        rows,
        "student_turn_group",
        STUDENT_GROUPS,
    )
    base_summaries, base_depth_summaries = base_move_summaries(rows)
    assert sum(int(row["N"]) for row in depth_summaries) == EXPECTED_N
    assert sum(int(row["N"]) for row in student_summaries) == EXPECTED_N

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    row_path = OUTPUT_DIR / "row_level_history_depth.csv.gz"
    depth_path = OUTPUT_DIR / "history_depth_summary.csv"
    student_path = OUTPUT_DIR / "student_turn_summary.csv"
    base_path = OUTPUT_DIR / "base_move_opportunity_summary.csv"
    base_depth_path = OUTPUT_DIR / "base_move_by_depth_summary.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    manifest_path = OUTPUT_DIR / "manifest.json"
    figure_paths = create_figures(depth_summaries, student_summaries)

    write_csv(row_path, rows, ROW_FIELDS, compressed=True)
    write_csv(
        depth_path,
        depth_summaries,
        ("history_depth_bin", *METRIC_FIELDS),
    )
    write_csv(
        student_path,
        student_summaries,
        ("student_turn_group", *METRIC_FIELDS),
    )
    write_csv(base_path, base_summaries, BASE_FIELDS)
    write_csv(base_depth_path, base_depth_summaries, BASE_DEPTH_FIELDS)
    summary_payload = {
        "audit_name": AUDIT_NAME,
        "purpose": PURPOSE,
        "history_depth_summary": depth_summaries,
        "student_turn_summary": student_summaries,
        "base_move_opportunity_summary": base_summaries,
        "base_move_by_depth_summary": base_depth_summaries,
        "validation_assertions": [
            {"assertion": check, "status": "PASS"} for check in checks
        ],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    runtime_seconds = time.perf_counter() - started
    outputs = [
        row_path,
        depth_path,
        student_path,
        base_path,
        base_depth_path,
        summary_path,
        *figure_paths,
        manifest_path,
    ]
    manifest = build_manifest(outputs, checks, runtime_seconds)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print_table("HISTORY DEPTH TABLE", "history_depth_bin", depth_summaries)
    print_table("STUDENT TURN TABLE", "student_turn_group", student_summaries)
    print("\nBASE-MOVE OPPORTUNITY TABLE")
    print(
        f"{'Base move':<10} {'N':>5} {'Mean gap':>10} "
        f"{'Baseline':>10} {'Alternative':>12}"
    )
    for row in base_summaries:
        print(
            f"{str(row['base_move']):<10} {int(row['N']):>5} "
            f"{float(row['mean_top1_top2_gap']):>10.6f} "
            f"{float(row['baseline_only_percentage']):>9.2f}% "
            f"{float(row['alternative_available_percentage']):>11.2f}%"
        )
    print("\nVALIDATION: ALL ASSERTIONS PASS")
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
