"""Frozen-MD6 dialogue-trajectory and matched-teacher-history diagnostic.

This CPU-only script uses ten fixed, hand-written inference cases. It does not
load evaluation datasets, train a model, evaluate accuracy, load MRB1 or LinTS,
or update any policy state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
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


AUDIT_NAME: Final[str] = "md6_dialogue_trajectory_v1"
PURPOSE: Final[str] = (
    "Describe how frozen MD6 probabilities evolve across one cumulative "
    "tutoring dialogue and four matched preceding-teacher-message variants."
)
PROBLEM: Final[str] = "Solve 4x + 8 = 32."
GAP_THRESHOLD: Final[float] = 0.10
PROBABILITY_TOLERANCE: Final[float] = 1e-5
CPU_THREADS: Final[int] = max(1, min(8, os.cpu_count() or 1))
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "frozen" / "md6"
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "self_improvement" / AUDIT_NAME
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"

TRAJECTORY_FIELDS: Final[tuple[str, ...]] = (
    "checkpoint",
    "history_turn_count",
    "latest_student_text",
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

CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "teacher_variant",
    "teacher_text",
    "final_student_text",
    "history_turn_count",
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

TRAJECTORY_EXCHANGES: Final[tuple[tuple[str, str], ...]] = (
    (
        "How would you start solving this problem? Talk me through your thinking.",
        "I think I should divide 32 by 4 first, so x = 8.",
    ),
    (
        "You divided 32 by 4. What part of the equation have you not accounted for yet?",
        "The plus 8, but I still think dividing by 4 gives x = 8.",
    ),
    (
        "Focus on the +8 first. What operation would undo adding 8?",
        "I don't understand why I need to subtract 8 first.",
    ),
    (
        "To isolate 4x, subtract 8 from both sides. What does the equation become?",
        "I'm still confused. I keep getting x = 8.",
    ),
    (
        "Subtracting 8 from both sides gives 4x = 24. What should you do next?",
        "I still don't understand. Can you show me?",
    ),
    (
        "Now divide both sides of 4x = 24 by 4. This gives x = 6.",
        "So I subtract 8 to get 4x = 24, then divide by 4 to get x = 6.",
    ),
)

CONTROL_PREFIX: Final[tuple[tuple[str, str], ...]] = (
    ("Teacher", "How would you start solving this problem?"),
    ("Student", "I divided 32 by 4 and got x = 8."),
)
CONTROL_FINAL_STUDENT: Final[str] = "I still think x = 8."
CONTROL_TEACHER_TEXTS: Final[dict[str, str]] = {
    "GENERIC": "Tell me more about how you are thinking about the problem.",
    "PROBING": "What part of the equation have you not used yet?",
    "FOCUS": "Focus specifically on the +8. What should happen to that term first?",
    "TELLING": "First subtract 8 from both sides so that 4x = 24.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def turn(user: str, text: str) -> dict[str, str]:
    return {"user": user, "text": text}


def build_trajectory() -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    cumulative: list[dict[str, str]] = []
    for checkpoint, (teacher_text, student_text) in enumerate(
        TRAJECTORY_EXCHANGES,
        start=1,
    ):
        cumulative.extend(
            [turn("Teacher", teacher_text), turn("Student", student_text)]
        )
        checkpoints.append(
            {
                "checkpoint": checkpoint,
                "problem": PROBLEM,
                "history": [dict(item) for item in cumulative],
            }
        )
    return checkpoints


def build_controls() -> list[dict[str, object]]:
    prefix = [turn(user, text) for user, text in CONTROL_PREFIX]
    controls: list[dict[str, object]] = []
    for variant, teacher_text in CONTROL_TEACHER_TEXTS.items():
        history = [
            *(dict(item) for item in prefix),
            turn("Teacher", teacher_text),
            turn("Student", CONTROL_FINAL_STUDENT),
        ]
        controls.append(
            {
                "teacher_variant": variant,
                "teacher_text": teacher_text,
                "problem": PROBLEM,
                "history": history,
            }
        )
    return controls


def validate_inputs(
    trajectory: Sequence[Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
) -> list[str]:
    checks: list[str] = []
    assert len(trajectory) == 6
    checks.append("exactly 6 trajectory checkpoints")
    lengths: list[int] = []
    for index, checkpoint in enumerate(trajectory):
        assert checkpoint["problem"] == PROBLEM
        history = checkpoint["history"]
        assert isinstance(history, list)
        lengths.append(len(history))
        assert history[-1] == turn("Student", TRAJECTORY_EXCHANGES[index][1])
        if index:
            previous = trajectory[index - 1]["history"]
            assert isinstance(previous, list)
            assert history[: len(previous)] == previous
    assert all(after > before for before, after in zip(lengths, lengths[1:]))
    checks.append("trajectory histories are cumulative")
    checks.append("trajectory history length strictly increases")
    checks.append("each checkpoint ends with the specified Student turn")

    assert len(controls) == 4
    assert [row["teacher_variant"] for row in controls] == list(
        CONTROL_TEACHER_TEXTS
    )
    checks.append("exactly 4 matched teacher-history variants")
    histories = [row["history"] for row in controls]
    assert all(isinstance(history, list) and len(history) == 4 for history in histories)
    assert all(row["problem"] == PROBLEM for row in controls)
    checks.append("problem is identical in all control variants")
    assert all(history[:2] == histories[0][:2] for history in histories)
    assert histories[0][:2] == [turn(user, text) for user, text in CONTROL_PREFIX]
    checks.append("control common prefix is identical")
    assert all(
        history[-1] == turn("Student", CONTROL_FINAL_STUDENT)
        for history in histories
    )
    checks.append("final student text is exactly identical across controls")
    assert all(history[2]["user"] == "Teacher" for history in histories)
    assert [history[2]["text"] for history in histories] == list(
        CONTROL_TEACHER_TEXTS.values()
    )
    assert len({history[2]["text"] for history in histories}) == 4
    checks.append("only the manipulated Teacher utterance differs after the prefix")
    return checks


def rank_moves(probabilities: Mapping[str, float]) -> list[str]:
    canonical_index = {move: index for index, move in enumerate(MOVE_ORDER)}
    return sorted(
        MOVE_ORDER,
        key=lambda move: (-probabilities[move], canonical_index[move]),
    )


def entropy(probabilities: Mapping[str, float]) -> float:
    return float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )


def prediction_fields(probabilities: Mapping[str, float]) -> dict[str, object]:
    assert tuple(probabilities) == MOVE_ORDER
    ranked = rank_moves(probabilities)
    best, second = ranked[:2]
    candidates = eligible_arms(probabilities, gap_threshold=GAP_THRESHOLD)
    return {
        "p_generic": float(probabilities["generic"]),
        "p_probing": float(probabilities["probing"]),
        "p_focus": float(probabilities["focus"]),
        "p_telling": float(probabilities["telling"]),
        "argmax_move": best,
        "top1_probability": float(probabilities[best]),
        "second_move": second,
        "top2_probability": float(probabilities[second]),
        "top1_top2_gap": float(probabilities[best] - probabilities[second]),
        "entropy": entropy(probabilities),
        "eligible_arms": json.dumps(list(candidates), separators=(",", ":")),
        "eligible_arm_count": len(candidates),
        "alternative_available": len(candidates) > 1,
    }


def infer_trajectory(
    model: FrozenMD6Inference,
    checkpoints: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        history = checkpoint["history"]
        problem = checkpoint["problem"]
        assert isinstance(history, list) and isinstance(problem, str)
        probabilities = model.predict_probabilities(problem, history)
        rows.append(
            {
                "checkpoint": checkpoint["checkpoint"],
                "history_turn_count": len(history),
                "latest_student_text": history[-1]["text"],
                **prediction_fields(probabilities),
            }
        )
    return rows


def infer_controls(
    model: FrozenMD6Inference,
    controls: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control in controls:
        history = control["history"]
        problem = control["problem"]
        assert isinstance(history, list) and isinstance(problem, str)
        probabilities = model.predict_probabilities(problem, history)
        rows.append(
            {
                "teacher_variant": control["teacher_variant"],
                "teacher_text": control["teacher_text"],
                "final_student_text": history[-1]["text"],
                "history_turn_count": len(history),
                **prediction_fields(probabilities),
            }
        )
    return rows


def validate_predictions(
    trajectory_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    checks: list[str] = []
    assert len(trajectory_rows) == 6 and len(control_rows) == 4
    checks.append("6 trajectory and 4 control predictions produced")
    assert MOVE_ORDER == ("generic", "probing", "focus", "telling")
    checks.append("canonical move order unchanged")
    assert GAP_THRESHOLD == 0.10
    checks.append("gap threshold equals 0.10")
    for row in [*trajectory_rows, *control_rows]:
        probabilities = [float(row[f"p_{move}"]) for move in MOVE_ORDER]
        assert all(math.isfinite(value) for value in probabilities)
        assert all(0.0 <= value <= 1.0 for value in probabilities)
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
    checks.append("all probabilities are finite and within [0,1]")
    checks.append("all probability vectors sum approximately to 1")
    checks.append("baseline is always eligible")
    return checks


def analyze_trajectory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    argmax_sequence = [str(row["argmax_move"]) for row in rows]
    first_move = argmax_sequence[0]
    first_change = next(
        (
            int(row["checkpoint"])
            for row in rows[1:]
            if row["argmax_move"] != first_move
        ),
        None,
    )
    return {
        "argmax_sequence": argmax_sequence,
        "argmax_changes_at_any_point": len(set(argmax_sequence)) > 1,
        "first_checkpoint_different_from_checkpoint_1": first_change,
        "checkpoint_1_to_6_change": {
            move: float(rows[-1][f"p_{move}"]) - float(rows[0][f"p_{move}"])
            for move in ("probing", "focus", "telling")
        },
        "alternative_available_checkpoint_count": sum(
            bool(row["alternative_available"]) for row in rows
        ),
    }


def analyze_controls(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_variant = {str(row["teacher_variant"]): row for row in rows}
    generic = by_variant["GENERIC"]
    return [
        {
            "teacher_variant": variant,
            **{
                f"delta_p_{move}": float(by_variant[variant][f"p_{move}"])
                - float(generic[f"p_{move}"])
                for move in MOVE_ORDER
            },
        }
        for variant in ("PROBING", "FOCUS", "TELLING")
    ]


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def create_figures(
    trajectory_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    trajectory_probability_path = FIGURE_DIR / "trajectory_move_probabilities.png"
    trajectory_gap_path = FIGURE_DIR / "trajectory_top1_gap.png"
    control_probability_path = (
        FIGURE_DIR / "teacher_history_control_probabilities.png"
    )

    checkpoints = [int(row["checkpoint"]) for row in trajectory_rows]
    figure = plt.figure(figsize=(9, 5))
    for move in MOVE_ORDER:
        plt.plot(
            checkpoints,
            [float(row[f"p_{move}"]) for row in trajectory_rows],
            label=move,
        )
    plt.title("Frozen MD6 probabilities over the cumulative dialogue")
    plt.xlabel("Checkpoint")
    plt.ylabel("Move probability")
    plt.xticks(checkpoints)
    plt.ylim(0.0, 1.0)
    plt.legend(title="Move")
    figure.tight_layout()
    figure.savefig(trajectory_probability_path, dpi=200)
    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(
        checkpoints,
        [float(row["top1_top2_gap"]) for row in trajectory_rows],
    )
    plt.title("Frozen MD6 top-1 minus top-2 gap over the dialogue")
    plt.xlabel("Checkpoint")
    plt.ylabel("Top-1 minus top-2 probability gap")
    plt.xticks(checkpoints)
    plt.ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(trajectory_gap_path, dpi=200)
    plt.close(figure)

    variants = [str(row["teacher_variant"]) for row in control_rows]
    figure = plt.figure(figsize=(9, 5))
    for move in MOVE_ORDER:
        plt.plot(
            variants,
            [float(row[f"p_{move}"]) for row in control_rows],
            label=move,
        )
    plt.title("Frozen MD6 probabilities by preceding Teacher variant")
    plt.xlabel("Teacher variant")
    plt.ylabel("Move probability")
    plt.ylim(0.0, 1.0)
    plt.legend(title="Move")
    figure.tight_layout()
    figure.savefig(control_probability_path, dpi=200)
    plt.close(figure)
    return [trajectory_probability_path, trajectory_gap_path, control_probability_path]


def build_manifest(
    output_files: Sequence[Path],
    validation_checks: Sequence[str],
    runtime_seconds: float,
) -> dict[str, object]:
    sources = (
        Path(__file__).resolve(),
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
        "interpretation_boundary": (
            "Descriptive model-behavior evidence only; no accuracy evaluation, "
            "causal domain-shift claim, or educational-efficacy claim."
        ),
        "frozen_md6_model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "cpu_only_diagnostic": True,
        "training_performed": False,
        "lints_loaded": False,
        "lints_update_performed": False,
        "mrb1_loaded": False,
        "held_out_test_untouched": True,
        "mrbench_test_untouched": True,
        "dataset_files_opened": [],
        "gap_threshold": GAP_THRESHOLD,
        "problem": PROBLEM,
        "trajectory_checkpoints": len(TRAJECTORY_EXCHANGES),
        "matched_control_variants": list(CONTROL_TEACHER_TEXTS),
        "source_file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in sources
        },
        "frozen_model_file_sha256": {
            path.name: sha256_file(path) for path in model_files
        },
        "validation_assertions": [
            {"assertion": check, "status": "PASS"} for check in validation_checks
        ],
        "output_files": [
            path.relative_to(PROJECT_ROOT).as_posix() for path in output_files
        ],
        "runtime_seconds": runtime_seconds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def eligible_text(row: Mapping[str, object]) -> str:
    return ",".join(json.loads(str(row["eligible_arms"])))


def print_results(
    trajectory_rows: Sequence[Mapping[str, object]],
    trajectory_summary: Mapping[str, object],
    control_rows: Sequence[Mapping[str, object]],
    control_deltas: Sequence[Mapping[str, object]],
) -> None:
    print("\nTRAJECTORY RESULTS")
    print(
        f"{'Checkpoint':>10} {'Generic':>9} {'Probing':>9} {'Focus':>9} "
        f"{'Telling':>9} {'Argmax':>9} {'Gap':>9}  Eligible arms"
    )
    for row in trajectory_rows:
        print(
            f"{int(row['checkpoint']):>10} "
            f"{float(row['p_generic']):>9.6f} "
            f"{float(row['p_probing']):>9.6f} "
            f"{float(row['p_focus']):>9.6f} "
            f"{float(row['p_telling']):>9.6f} "
            f"{str(row['argmax_move']):>9} "
            f"{float(row['top1_top2_gap']):>9.6f}  {eligible_text(row)}"
        )
    print("\nTRAJECTORY SUMMARY")
    print(f"Argmax sequence: {trajectory_summary['argmax_sequence']}")
    print(
        "First checkpoint different from checkpoint 1: "
        f"{trajectory_summary['first_checkpoint_different_from_checkpoint_1']}"
    )
    changes = trajectory_summary["checkpoint_1_to_6_change"]
    assert isinstance(changes, Mapping)
    for move in ("probing", "focus", "telling"):
        print(f"Checkpoint 1 -> 6 delta p_{move}: {float(changes[move]):.6f}")
    print(
        "Alternative-available checkpoints: "
        f"{trajectory_summary['alternative_available_checkpoint_count']}"
    )

    print("\nMATCHED TEACHER-HISTORY CONTROL")
    print(
        f"{'Variant':>9} {'Generic':>9} {'Probing':>9} {'Focus':>9} "
        f"{'Telling':>9} {'Argmax':>9} {'Gap':>9}  Eligible arms"
    )
    for row in control_rows:
        print(
            f"{str(row['teacher_variant']):>9} "
            f"{float(row['p_generic']):>9.6f} "
            f"{float(row['p_probing']):>9.6f} "
            f"{float(row['p_focus']):>9.6f} "
            f"{float(row['p_telling']):>9.6f} "
            f"{str(row['argmax_move']):>9} "
            f"{float(row['top1_top2_gap']):>9.6f}  {eligible_text(row)}"
        )
    print("\nMATCHED DELTAS RELATIVE TO GENERIC")
    for row in control_deltas:
        print(
            f"{row['teacher_variant']}: "
            + ", ".join(
                f"delta p_{move}={float(row[f'delta_p_{move}']):.6f}"
                for move in MOVE_ORDER
            )
        )


def main() -> None:
    started = time.perf_counter()
    assert MOVE_ORDER == ("generic", "probing", "focus", "telling")
    assert GAP_THRESHOLD == 0.10
    trajectory = build_trajectory()
    controls = build_controls()
    input_checks = validate_inputs(trajectory, controls)

    model = FrozenMD6Inference()
    model._torch.set_num_threads(CPU_THREADS)
    trajectory_rows = infer_trajectory(model, trajectory)
    control_rows = infer_controls(model, controls)
    output_checks = validate_predictions(trajectory_rows, control_rows)
    all_checks = [*input_checks, *output_checks]

    trajectory_summary = analyze_trajectory(trajectory_rows)
    control_deltas = analyze_controls(control_rows)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_path = OUTPUT_DIR / "trajectory_predictions.csv"
    control_path = OUTPUT_DIR / "teacher_history_control.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    manifest_path = OUTPUT_DIR / "manifest.json"
    figure_paths = create_figures(trajectory_rows, control_rows)
    write_csv(trajectory_path, trajectory_rows, TRAJECTORY_FIELDS)
    write_csv(control_path, control_rows, CONTROL_FIELDS)
    summary_payload = {
        "audit_name": AUDIT_NAME,
        "purpose": PURPOSE,
        "trajectory_summary": trajectory_summary,
        "teacher_history_control_deltas_relative_to_generic": control_deltas,
        "validation_assertions": [
            {"assertion": check, "status": "PASS"} for check in all_checks
        ],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    runtime_seconds = time.perf_counter() - started
    output_files = [
        trajectory_path,
        control_path,
        summary_path,
        *figure_paths,
        manifest_path,
    ]
    manifest = build_manifest(output_files, all_checks, runtime_seconds)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_results(trajectory_rows, trajectory_summary, control_rows, control_deltas)
    print("\nVALIDATION: ALL ASSERTIONS PASS")
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
