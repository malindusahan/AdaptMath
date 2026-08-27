"""CPU-only fixed-case behavior audit for the frozen MD6 selector.

The manually designed scenario groups are descriptive and are not gold
pedagogical-move labels. This script therefore reports probability and action-
space behavior only; it never calculates accuracy, F1, a confusion matrix, or
any notion of a correct move.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
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


AUDIT_NAME: Final[str] = "md6_move_behavior_v1"
PURPOSE: Final[str] = (
    "Audit whether the frozen MD6 selector exhibits a strong probing tendency "
    "across fixed, diverse tutoring dialogue states."
)
DISCLAIMER: Final[str] = (
    "Diagnostic behavior audit only. Scenario groups are descriptive, not gold "
    "pedagogical-move labels; no accuracy or correctness metric is defined."
)
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "frozen" / "md6"
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "self_improvement" / AUDIT_NAME
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"
GAP_THRESHOLD: Final[float] = 0.10
PROBABILITY_TOLERANCE: Final[float] = 1e-5

GROUP_START: Final[str] = "START_UNCERTAINTY"
GROUP_REASONING: Final[str] = "REASONING_AVAILABLE"
GROUP_ERROR: Final[str] = "CLEAR_ERROR"
GROUP_DIFFICULTY: Final[str] = "REPEATED_DIFFICULTY"
SCENARIO_GROUPS: Final[tuple[str, ...]] = (
    GROUP_START,
    GROUP_REASONING,
    GROUP_ERROR,
    GROUP_DIFFICULTY,
)

RAW_FIELDS: Final[tuple[str, ...]] = (
    "case_id",
    "scenario_group",
    "problem",
    "history",
    "p_generic",
    "p_probing",
    "p_focus",
    "p_telling",
    "argmax_move",
    "top1_probability",
    "second_move",
    "top2_probability",
    "top1_top2_gap",
    "shannon_entropy",
    "eligible_arms",
    "eligible_arm_count",
    "alternative_available",
)


def _turn(user: str, text: str) -> dict[str, str]:
    return {"user": user, "text": text}


def _case(
    case_id: int,
    scenario_group: str,
    problem: str,
    history: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "scenario_group": scenario_group,
        "problem": problem,
        "history": [dict(turn) for turn in history],
    }


def fixed_cases() -> list[dict[str, object]]:
    """Return the 24 pre-specified, non-gold dialogue cases."""

    return [
        _case(
            1,
            GROUP_START,
            "What is 15% of 80?",
            [_turn("student", "I don't know where to start.")],
        ),
        _case(
            2,
            GROUP_START,
            "Solve 5x + 7 = 32.",
            [_turn("student", "I'm confused about what I should do first.")],
        ),
        _case(
            3,
            GROUP_START,
            "A rectangle has length 9 cm and width 4 cm. Find its area.",
            [_turn("student", "I don't remember how to find area.")],
        ),
        _case(
            4,
            GROUP_START,
            "What is 3/4 of 20?",
            [_turn("student", "I'm not sure what \"of\" means here.")],
        ),
        _case(
            5,
            GROUP_START,
            "A bag has 5 red and 7 blue marbles. How many marbles are there altogether?",
            [_turn("student", "I don't know.")],
        ),
        _case(
            6,
            GROUP_START,
            "Find the perimeter of a square with side length 6 cm.",
            [_turn("student", "Can you help me start?")],
        ),
        _case(
            7,
            GROUP_REASONING,
            "Solve 3x + 5 = 20.",
            [_turn("student", "I think x = 3 because 3 times 3 is 9.")],
        ),
        _case(
            8,
            GROUP_REASONING,
            "What is 25% of 60?",
            [
                _turn(
                    "student",
                    "I think I should divide 60 by 4 because 25% is one fourth.",
                )
            ],
        ),
        _case(
            9,
            GROUP_REASONING,
            (
                "A school library has 6 shelves with 24 books on each shelf. "
                "18 books are removed and the rest are shared equally among 9 "
                "students. How many books does each student get?"
            ),
            [
                _turn(
                    "student",
                    "I think I should first find how many books are on all 6 shelves.",
                )
            ],
        ),
        _case(
            10,
            GROUP_REASONING,
            "Find 2/3 of 18.",
            [
                _turn(
                    "student",
                    "I think I should divide 18 by 3 first and then multiply by 2.",
                )
            ],
        ),
        _case(
            11,
            GROUP_REASONING,
            "Solve 4x = 28.",
            [
                _turn(
                    "student",
                    "Since 4 times x is 28, I think I need to undo multiplication by 4.",
                )
            ],
        ),
        _case(
            12,
            GROUP_REASONING,
            "A shirt costs $40 and is discounted by 20%. Find the discount amount.",
            [
                _turn(
                    "student",
                    "I think 20% means I need to find one fifth of 40.",
                )
            ],
        ),
        _case(
            13,
            GROUP_ERROR,
            "A rectangle is 8 cm by 5 cm. Find its area.",
            [
                _turn(
                    "student",
                    "8 + 5 = 13, so the area is 13 square centimeters.",
                )
            ],
        ),
        _case(
            14,
            GROUP_ERROR,
            "Solve 2x + 6 = 18.",
            [_turn("student", "I divided 18 by 2 first, so x = 9.")],
        ),
        _case(
            15,
            GROUP_ERROR,
            "Find 30% of 50.",
            [_turn("student", "30% of 50 is 80 because I added 30 and 50.")],
        ),
        _case(
            16,
            GROUP_ERROR,
            "A triangle has base 10 cm and height 4 cm. Find its area.",
            [
                _turn(
                    "student",
                    "I multiplied 10 by 4 and got 40 square centimeters.",
                )
            ],
        ),
        _case(
            17,
            GROUP_ERROR,
            "Calculate 144 - 18.",
            [_turn("student", "The answer is 125.")],
        ),
        _case(
            18,
            GROUP_ERROR,
            "What is 126 divided by 9?",
            [_turn("student", "I think it is 12.")],
        ),
        _case(
            19,
            GROUP_DIFFICULTY,
            "Solve 4x = 28.",
            [
                _turn("student", "I divided 4 by 28 and got about 0.14."),
                _turn("teacher", "What operation would undo multiplication by 4?"),
                _turn("student", "I'm not sure."),
                _turn("teacher", "Think about how you could isolate x in 4x = 28."),
                _turn("student", "I still don't understand. Can you show me?"),
            ],
        ),
        _case(
            20,
            GROUP_DIFFICULTY,
            "Find 20% of 70.",
            [
                _turn("student", "I added 20 and 70."),
                _turn("teacher", "Think about what percent means."),
                _turn("student", "I still don't get it. Please show me how."),
            ],
        ),
        _case(
            21,
            GROUP_DIFFICULTY,
            "Solve x/5 = 6.",
            [
                _turn("student", "I divided 6 by 5."),
                _turn("teacher", "Which operation reverses division by 5?"),
                _turn("student", "I don't know."),
                _turn("teacher", "What could you do to both sides?"),
                _turn("student", "Can you just show me the next step?"),
            ],
        ),
        _case(
            22,
            GROUP_DIFFICULTY,
            "Find the area of a rectangle with sides 7 cm and 3 cm.",
            [
                _turn("student", "I added them and got 10."),
                _turn(
                    "teacher",
                    "Area uses the two side lengths in a particular operation.",
                ),
                _turn("student", "I don't know which one."),
                _turn("teacher", "Think about rows of 7 repeated 3 times."),
                _turn("student", "I still don't understand."),
            ],
        ),
        _case(
            23,
            GROUP_DIFFICULTY,
            "What is 3/5 of 25?",
            [
                _turn("student", "I did 25 divided by 3."),
                _turn("teacher", "Look again at the denominator."),
                _turn("student", "I'm confused."),
                _turn("teacher", "What does the 5 tell you?"),
                _turn("student", "Please show me."),
            ],
        ),
        _case(
            24,
            GROUP_DIFFICULTY,
            "A car travels 180 km in 3 hours at a constant rate. Find its speed.",
            [
                _turn("student", "I multiplied 180 by 3."),
                _turn("teacher", "Think about distance per one hour."),
                _turn("student", "I don't know."),
                _turn(
                    "teacher",
                    "What operation could split 180 equally across the 3 hours?",
                ),
                _turn("student", "Just tell me what to do."),
            ],
        ),
    ]


def validate_fixed_cases(cases: Sequence[Mapping[str, object]]) -> None:
    """Validate the fixed audit design before the model is loaded."""

    assert len(cases) == 24, f"Expected exactly 24 cases, found {len(cases)}."
    expected_ids = list(range(1, 25))
    assert [case["case_id"] for case in cases] == expected_ids
    group_counts = Counter(case["scenario_group"] for case in cases)
    assert set(group_counts) == set(SCENARIO_GROUPS)
    assert all(group_counts[group] == 6 for group in SCENARIO_GROUPS)

    for case in cases:
        assert isinstance(case["problem"], str) and case["problem"].strip()
        history = case["history"]
        assert isinstance(history, Sequence) and not isinstance(history, (str, bytes))
        assert len(history) > 0
        for turn in history:
            assert isinstance(turn, Mapping)
            assert set(turn) == {"user", "text"}
            assert isinstance(turn["user"], str) and turn["user"].strip()
            assert isinstance(turn["text"], str) and turn["text"].strip()


def _rank_moves(probabilities: Mapping[str, float]) -> list[str]:
    order = {move: index for index, move in enumerate(MOVE_ORDER)}
    return sorted(
        MOVE_ORDER,
        key=lambda move: (-probabilities[move], order[move]),
    )


def _entropy(probabilities: Mapping[str, float]) -> float:
    return float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )


def run_inference(
    cases: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], FrozenMD6Inference]:
    """Call the production wrapper once for every fixed case."""

    model = FrozenMD6Inference()
    rows: list[dict[str, object]] = []
    for case in cases:
        problem = case["problem"]
        history = case["history"]
        if not isinstance(problem, str) or not isinstance(history, Sequence):
            raise TypeError("Validated case unexpectedly changed type.")

        probabilities = model.predict_probabilities(problem, history)
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

        ranked = _rank_moves(probabilities)
        argmax_move, second_move = ranked[:2]
        top1 = float(probabilities[argmax_move])
        top2 = float(probabilities[second_move])
        gap = float(top1 - top2)
        assert argmax_move in MOVE_ORDER
        assert 0.0 <= gap <= 1.0

        candidates = eligible_arms(probabilities, gap_threshold=GAP_THRESHOLD)
        assert candidates and candidates[0] == "baseline"
        row = {
            "case_id": int(case["case_id"]),
            "scenario_group": str(case["scenario_group"]),
            "problem": problem,
            "history": json.dumps(history, ensure_ascii=False, separators=(",", ":")),
            "p_generic": float(probabilities["generic"]),
            "p_probing": float(probabilities["probing"]),
            "p_focus": float(probabilities["focus"]),
            "p_telling": float(probabilities["telling"]),
            "argmax_move": argmax_move,
            "top1_probability": top1,
            "second_move": second_move,
            "top2_probability": top2,
            "top1_top2_gap": gap,
            "shannon_entropy": _entropy(probabilities),
            "eligible_arms": json.dumps(list(candidates), separators=(",", ":")),
            "eligible_arm_count": len(candidates),
            "alternative_available": len(candidates) > 1,
        }
        rows.append(row)

    validate_inference_rows(rows)
    return rows, model


def validate_inference_rows(rows: Sequence[Mapping[str, object]]) -> None:
    assert len(rows) == 24
    for row in rows:
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
        candidates = json.loads(str(row["eligible_arms"]))
        assert candidates and candidates[0] == "baseline"
        assert int(row["eligible_arm_count"]) == len(candidates)
        assert bool(row["alternative_available"]) == (len(candidates) > 1)


def _mean_median(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def build_summary(rows: Sequence[Mapping[str, object]], model_path: Path) -> dict[str, object]:
    n_cases = len(rows)
    counts = Counter(str(row["argmax_move"]) for row in rows)
    ordered_counts = {move: int(counts.get(move, 0)) for move in MOVE_ORDER}
    percentages = {
        move: float(100.0 * ordered_counts[move] / n_cases) for move in MOVE_ORDER
    }

    probability_summaries = {
        move: _mean_median([float(row[f"p_{move}"]) for row in rows])
        for move in MOVE_ORDER
    }
    top1_values = [float(row["top1_probability"]) for row in rows]
    gaps = np.asarray([float(row["top1_top2_gap"]) for row in rows], dtype=np.float64)
    quantile_levels = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    quantile_values = np.quantile(gaps, quantile_levels)
    gap_le_threshold = int(np.count_nonzero(gaps <= GAP_THRESHOLD))

    eligible_counts = [int(row["eligible_arm_count"]) for row in rows]
    alternative_count = sum(bool(row["alternative_available"]) for row in rows)
    baseline_only_count = n_cases - alternative_count

    by_scenario: dict[str, object] = {}
    for group in SCENARIO_GROUPS:
        group_rows = [row for row in rows if row["scenario_group"] == group]
        group_counts = Counter(str(row["argmax_move"]) for row in group_rows)
        by_scenario[group] = {
            "number_of_cases": len(group_rows),
            "counts": {move: int(group_counts.get(move, 0)) for move in MOVE_ORDER},
            "percentages": {
                move: float(100.0 * group_counts.get(move, 0) / len(group_rows))
                for move in MOVE_ORDER
            },
        }

    return {
        "audit_name": AUDIT_NAME,
        "disclaimer": DISCLAIMER,
        "model_path": str(model_path),
        "number_of_cases": n_cases,
        "gap_threshold": GAP_THRESHOLD,
        "move_order": list(MOVE_ORDER),
        "counts": ordered_counts,
        "percentages": percentages,
        "argmax_by_scenario": by_scenario,
        "probability_summaries": probability_summaries,
        "top1_confidence_summary": _mean_median(top1_values),
        "gap_summaries": {
            **_mean_median(gaps.tolist()),
            "quantiles": {
                f"{level:.2f}": float(value)
                for level, value in zip(quantile_levels, quantile_values, strict=True)
            },
            "gap_le_0_10_count": gap_le_threshold,
            "gap_le_0_10_percentage": float(100.0 * gap_le_threshold / n_cases),
        },
        "eligibility_summaries": {
            "baseline_only_count": baseline_only_count,
            "baseline_only_percentage": float(100.0 * baseline_only_count / n_cases),
            "alternative_available_count": alternative_count,
            "alternative_available_percentage": float(
                100.0 * alternative_count / n_cases
            ),
            "mean_eligible_arm_count": float(statistics.fmean(eligible_counts)),
        },
    }


def write_raw_cases(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RAW_FIELDS})


def write_argmax_by_scenario(
    summary: Mapping[str, object],
    path: Path,
) -> None:
    scenario_summary = summary["argmax_by_scenario"]
    if not isinstance(scenario_summary, Mapping):
        raise TypeError("argmax_by_scenario summary must be a mapping.")
    fields = ("scenario_group", "move", "group_n", "count", "percentage")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in SCENARIO_GROUPS:
            group_summary = scenario_summary[group]
            if not isinstance(group_summary, Mapping):
                raise TypeError("Scenario summary must be a mapping.")
            counts = group_summary["counts"]
            percentages = group_summary["percentages"]
            if not isinstance(counts, Mapping) or not isinstance(percentages, Mapping):
                raise TypeError("Scenario counts/percentages must be mappings.")
            for move in MOVE_ORDER:
                writer.writerow(
                    {
                        "scenario_group": group,
                        "move": move,
                        "group_n": group_summary["number_of_cases"],
                        "count": counts[move],
                        "percentage": percentages[move],
                    }
                )


def create_figures(rows: Sequence[Mapping[str, object]]) -> list[Path]:
    counts = Counter(str(row["argmax_move"]) for row in rows)

    argmax_path = FIGURE_DIR / "md6_argmax_distribution.png"
    figure = plt.figure(figsize=(8, 5))
    plt.bar(MOVE_ORDER, [counts.get(move, 0) for move in MOVE_ORDER])
    plt.title("Frozen MD6 argmax distribution across fixed diagnostic cases")
    plt.xlabel("MD6 move")
    plt.ylabel("Number of cases")
    plt.ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(argmax_path, dpi=200)
    plt.close(figure)

    probability_path = FIGURE_DIR / "md6_probability_distributions.png"
    figure = plt.figure(figsize=(8, 5))
    plt.boxplot(
        [[float(row[f"p_{move}"]) for row in rows] for move in MOVE_ORDER],
        labels=list(MOVE_ORDER),
    )
    plt.title("Frozen MD6 probability distributions across fixed cases")
    plt.xlabel("MD6 move")
    plt.ylabel("Predicted probability")
    plt.ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(probability_path, dpi=200)
    plt.close(figure)

    gap_path = FIGURE_DIR / "top1_top2_gap_distribution.png"
    figure = plt.figure(figsize=(8, 5))
    plt.hist([float(row["top1_top2_gap"]) for row in rows], bins=10)
    plt.title("Frozen MD6 top-1 minus top-2 probability gaps")
    plt.xlabel("Top-1 minus top-2 probability gap")
    plt.ylabel("Number of cases")
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(gap_path, dpi=200)
    plt.close(figure)

    return [argmax_path, probability_path, gap_path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(output_files: Sequence[Path], model_path: Path) -> dict[str, object]:
    source_files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "self_improvement" / "md6_inference.py",
        PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py",
        PROJECT_ROOT / "src" / "self_improvement" / "context_builder.py",
    )
    model_files = (
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
    )
    return {
        "audit_name": AUDIT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_purpose": PURPOSE,
        "diagnostic_not_gold_disclaimer": DISCLAIMER,
        "input_case_count": 24,
        "frozen_model_path": str(model_path),
        "source_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_files
        },
        "frozen_model_file_sha256": {
            path.name: sha256_file(path) for path in model_files
        },
        "output_files": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_files
        ],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def _compact_line(row: Mapping[str, object]) -> str:
    candidates = json.loads(str(row["eligible_arms"]))
    return (
        f"{int(row['case_id']):02d} | {row['scenario_group']} | "
        f"generic={float(row['p_generic']):.2f} "
        f"probing={float(row['p_probing']):.2f} "
        f"focus={float(row['p_focus']):.2f} "
        f"telling={float(row['p_telling']):.2f} | "
        f"argmax={row['argmax_move']} | second={row['second_move']} | "
        f"gap={float(row['top1_top2_gap']):.2f} | eligible={candidates}"
    )


def print_summary(rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> None:
    print("MD6 MOVE BEHAVIOR AUDIT V1")
    print(DISCLAIMER)
    print(f"\nN cases: {len(rows)}")

    counts = summary["counts"]
    percentages = summary["percentages"]
    assert isinstance(counts, Mapping) and isinstance(percentages, Mapping)
    print("\nARGMAX COUNTS AND PERCENTAGES")
    for move in MOVE_ORDER:
        print(f"- {move}: {counts[move]} ({float(percentages[move]):.2f}%)")

    print("\nARGMAX BY DESCRIPTIVE SCENARIO GROUP")
    by_scenario = summary["argmax_by_scenario"]
    assert isinstance(by_scenario, Mapping)
    for group in SCENARIO_GROUPS:
        group_summary = by_scenario[group]
        assert isinstance(group_summary, Mapping)
        group_counts = group_summary["counts"]
        group_percentages = group_summary["percentages"]
        assert isinstance(group_counts, Mapping) and isinstance(
            group_percentages, Mapping
        )
        values = ", ".join(
            f"{move}={group_counts[move]} ({float(group_percentages[move]):.1f}%)"
            for move in MOVE_ORDER
        )
        print(f"- {group}: {values}")

    print("\nMOVE PROBABILITY MEAN / MEDIAN")
    probability_summaries = summary["probability_summaries"]
    assert isinstance(probability_summaries, Mapping)
    for move in MOVE_ORDER:
        values = probability_summaries[move]
        assert isinstance(values, Mapping)
        print(
            f"- {move}: mean={float(values['mean']):.6f}, "
            f"median={float(values['median']):.6f}"
        )

    top1 = summary["top1_confidence_summary"]
    gaps = summary["gap_summaries"]
    eligibility = summary["eligibility_summaries"]
    assert isinstance(top1, Mapping)
    assert isinstance(gaps, Mapping)
    assert isinstance(eligibility, Mapping)
    print("\nTOP-1 CONFIDENCE")
    print(f"- mean={float(top1['mean']):.6f}")
    print(f"- median={float(top1['median']):.6f}")
    print("\nTOP1-TOP2 GAP")
    print(f"- mean={float(gaps['mean']):.6f}")
    print(f"- median={float(gaps['median']):.6f}")
    quantiles = gaps["quantiles"]
    assert isinstance(quantiles, Mapping)
    print("- quantiles:")
    for key, value in quantiles.items():
        print(f"  {key}: {float(value):.6f}")
    print(
        "- gap <= .10: "
        f"{gaps['gap_le_0_10_count']} "
        f"({float(gaps['gap_le_0_10_percentage']):.2f}%)"
    )

    print("\nCONSERVATIVE OVERLAY ELIGIBILITY")
    print(
        "- baseline only: "
        f"{eligibility['baseline_only_count']} "
        f"({float(eligibility['baseline_only_percentage']):.2f}%)"
    )
    print(
        "- alternative available: "
        f"{eligibility['alternative_available_count']} "
        f"({float(eligibility['alternative_available_percentage']):.2f}%)"
    )
    print(
        "- mean eligible-arm count: "
        f"{float(eligibility['mean_eligible_arm_count']):.6f}"
    )

    print("\nALL CASES")
    for row in rows:
        print(_compact_line(row))

    highest = sorted(
        rows,
        key=lambda row: (-float(row["p_probing"]), int(row["case_id"])),
    )[:5]
    lowest = sorted(
        rows,
        key=lambda row: (float(row["p_probing"]), int(row["case_id"])),
    )[:5]
    non_probing = [row for row in rows if row["argmax_move"] != "probing"]

    print("\nFIVE HIGHEST PROBING-PROBABILITY CASES")
    for row in highest:
        print(_compact_line(row))
    print("\nFIVE LOWEST PROBING-PROBABILITY CASES")
    for row in lowest:
        print(_compact_line(row))
    print("\nALL CASES WITH NON-PROBING ARGMAX")
    if non_probing:
        for row in non_probing:
            print(_compact_line(row))
    else:
        print("None")


def main() -> None:
    cases = fixed_cases()
    validate_fixed_cases(cases)
    rows, model = run_inference(cases)
    summary = build_summary(rows, model.model_dir)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "raw_cases.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    scenario_path = OUTPUT_DIR / "argmax_by_scenario.csv"
    manifest_path = OUTPUT_DIR / "manifest.json"
    figure_paths = create_figures(rows)

    write_raw_cases(rows, raw_path)
    write_argmax_by_scenario(summary, scenario_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_files = [
        raw_path,
        summary_path,
        scenario_path,
        *figure_paths,
        manifest_path,
    ]
    manifest = build_manifest(output_files, model.model_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print_summary(rows, summary)
    print(f"\nOUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
