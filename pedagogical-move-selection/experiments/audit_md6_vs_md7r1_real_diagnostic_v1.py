"""Compare frozen MD6 with MD7-R1 epoch 3 on existing local diagnostics.

This audit is deliberately inference-only. It does not import or call LinTS,
the conservative overlay, BKT, MRB1, the Tutor Agent, or any external API. The
live checkpoint database is opened with SQLite ``mode=ro``.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Final[Path] = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from langgraph.checkpoint.sqlite import SqliteSaver

from src.self_improvement.context_builder import MOVE_ORDER
from src.self_improvement.md6_inference import FrozenMD6Inference, MAX_LENGTH


AUDIT_NAME: Final[str] = "md7r1_real_diagnostic_v1"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "results" / AUDIT_NAME
MD6_DIR: Final[Path] = PROJECT_ROOT / "models" / "frozen" / "md6"
MD7_DIR: Final[Path] = (
    PROJECT_ROOT / "models" / "candidates" / "md7r1_epoch3"
)
HANDWRITTEN_DEFINITION_SOURCE: Final[Path] = (
    PROJECT_ROOT / "experiments" / "audit_md6_move_behavior_v1.py"
)
HANDWRITTEN_SOURCE: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "self_improvement"
    / "md6_validation_vs_demo_v1"
    / "handwritten_demo_predictions.csv"
)
LIVE_DB: Final[Path] = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "adaptive_demo"
    / "checkpoints.sqlite3"
)
LIVE_ATTEMPTS: Final[Path] = LIVE_DB.with_name("attempts.jsonl")
LIVE_LOG: Final[Path] = WORKSPACE_ROOT / "_local_runtime_logs" / "tutor.stderr.log"
LIVE_ADAPTER: Final[Path] = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "app"
    / "integrations"
    / "tutor_state_memory_adapter.py"
)
PREFX_INSPECTOR: Final[Path] = (
    WORKSPACE_ROOT
    / "_forensics"
    / "20260826_same_attempt_failure_pre_fix"
    / "inspect_checkpoint_snapshot.py"
)
POSTFIX_INSPECTOR: Final[Path] = PREFX_INSPECTOR.with_name("inspect_live_retest.py")

CONTROLLED_SET: Final[str] = "controlled_20"
HANDWRITTEN_SET: Final[str] = "handwritten_24"
LIVE_SET: Final[str] = "genuine_live_turns"
COMBINED_SET: Final[str] = "combined_available"
CASE_SETS: Final[tuple[str, ...]] = (
    CONTROLLED_SET,
    HANDWRITTEN_SET,
    LIVE_SET,
)
EXPECTED_LABELS: Final[tuple[str, ...]] = (
    "generic",
    "probing",
    "focus",
    "telling",
)
PROBABILITY_TOLERANCE: Final[float] = 1e-6
GAP_THRESHOLD: Final[float] = 0.10

# These are the preserved manual sessions identified by the local investigation.
# The other checkpoint threads are deterministic setup/smoke fixtures and are not
# silently promoted to genuine manual cases.
MANUAL_THREADS: Final[dict[str, str]] = {
    "4e578ab1-225b-4251-9f67-b837f0b3c7a8": "manual equation post-fix retest",
    "9c909170-00a6-4a33-8ad8-9ef003c2102e": "manual push-up pre-fix failure",
    "9ab73ef2-3342-4141-b5ef-1c8c57736883": "manual push-up continued retest",
}

CASE_FIELDS: Final[tuple[str, ...]] = (
    "case_id",
    "case_set",
    "source_path",
    "source_locator",
    "problem",
    "conversation_history",
    "latest_student_text",
    "history_turn_count",
    "historical_md6_available",
    "historical_md6_max_abs_difference",
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "md6_move",
    "md6_top1_probability",
    "md6_top2_probability",
    "md6_top1_top2_gap",
    "md7_p_generic",
    "md7_p_probing",
    "md7_p_focus",
    "md7_p_telling",
    "md7_move",
    "md7_top1_probability",
    "md7_top2_probability",
    "md7_top1_top2_gap",
    "move_changed",
    "transition",
)


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes(model_dir: Path) -> dict[str, str]:
    return {
        "config.json": sha256_file(model_dir / "config.json"),
        "model.safetensors": sha256_file(model_dir / "model.safetensors"),
        "tokenizer.json": sha256_file(model_dir / "tokenizer.json"),
    }


def validate_model_artifacts(model_dir: Path) -> dict[str, object]:
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}: {missing}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {
        int(index): str(label) for index, label in config["id2label"].items()
    }
    label2id = {str(label): int(index) for label, index in config["label2id"].items()}
    expected_id2label = dict(enumerate(EXPECTED_LABELS))
    expected_label2id = {
        label: index for index, label in enumerate(EXPECTED_LABELS)
    }
    if id2label != expected_id2label or label2id != expected_label2id:
        raise ValueError(f"Class order mismatch in {model_dir}")
    if config.get("architectures") != ["RobertaForSequenceClassification"]:
        raise ValueError(f"Unexpected architecture in {model_dir}")
    return {
        "path": str(model_dir.resolve()),
        "required_artifacts_present": True,
        "architecture": config["architectures"][0],
        "id2label": id2label,
        "label2id": label2id,
        "hashes": model_hashes(model_dir),
    }


def latest_student_text(history: Sequence[Mapping[str, object]]) -> str:
    for turn in reversed(history):
        if str(turn.get("user", "")).lower() == "student":
            text = turn.get("text")
            return text if isinstance(text, str) else ""
    return ""


def load_handwritten_cases() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with HANDWRITTEN_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["source_group"] != "handwritten_demo":
                continue
            history = json.loads(raw["conversation_history"])
            probabilities = {
                move: float(raw[f"p_{move}"]) for move in EXPECTED_LABELS
            }
            rows.append(
                {
                    "case_id": raw["row_id"],
                    "case_set": HANDWRITTEN_SET,
                    "source_path": relative_or_absolute(HANDWRITTEN_SOURCE),
                    "source_locator": f"row_id={raw['row_id']}; group_id={raw['group_id']}",
                    "problem": raw["problem"],
                    "conversation_history": history,
                    "latest_student_text": latest_student_text(history),
                    "history_turn_count": len(history),
                    "historical_md6_probabilities": probabilities,
                    "historical_md6_move": raw["argmax_move"],
                    "semantic_group": raw["group_id"],
                }
            )
    if len(rows) != 24:
        raise AssertionError(f"Expected 24 handwritten cases, found {len(rows)}")
    return rows


def project_live_history(
    raw_history: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    projected: list[dict[str, str]] = []
    for index, raw_turn in enumerate(raw_history):
        role = raw_turn.get("role")
        content = raw_turn.get("content")
        if role not in {"student", "teacher"} or not isinstance(content, str):
            raise ValueError(f"Invalid live history turn at index {index}")
        projected.append({"user": role, "text": content})
    return projected


def load_live_cases() -> tuple[list[dict[str, object]], dict[str, object]]:
    db_hash_before = sha256_file(LIVE_DB)
    wal_path = LIVE_DB.with_name(LIVE_DB.name + "-wal")
    wal_hash_before = sha256_file(wal_path) if wal_path.is_file() else None
    connection = sqlite3.connect(
        f"file:{LIVE_DB.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    recovered: dict[tuple[str, int], dict[str, object]] = {}
    latest_values_by_thread: dict[str, Mapping[str, object]] = {}
    try:
        saver = SqliteSaver(connection)
        for thread_id in MANUAL_THREADS:
            tuples = list(saver.list({"configurable": {"thread_id": thread_id}}))
            if not tuples:
                raise AssertionError(f"Manual thread missing from checkpoint DB: {thread_id}")
            latest_values_by_thread[thread_id] = (
                tuples[0].checkpoint.get("channel_values") or {}
            )
            for checkpoint_tuple in tuples:
                values = checkpoint_tuple.checkpoint.get("channel_values") or {}
                diagnostic = values.get("adaptive_turn_diagnostics")
                history = values.get("conversation_history")
                question = values.get("question")
                if not isinstance(diagnostic, Mapping):
                    continue
                if not isinstance(history, list) or not isinstance(question, str):
                    continue
                turn_index = diagnostic.get("turn_index")
                saved = diagnostic.get("md6_probabilities")
                if not isinstance(turn_index, int) or not isinstance(saved, Mapping):
                    continue
                if not history or history[-1].get("role") != "teacher":
                    continue
                pre_history_raw = history[:-1]
                candidate = {
                    "thread_id": thread_id,
                    "turn_index": turn_index,
                    "problem": question,
                    "conversation_history": project_live_history(pre_history_raw),
                    "historical_md6_probabilities": {
                        move: float(saved[move]) for move in EXPECTED_LABELS
                    },
                    "historical_md6_move": str(diagnostic["base_move"]),
                    "checkpoint_id": checkpoint_tuple.checkpoint.get("id"),
                }
                key = (thread_id, turn_index)
                previous = recovered.get(key)
                if previous is None or len(candidate["conversation_history"]) < len(
                    previous["conversation_history"]
                ):
                    recovered[key] = candidate

        # The preserved pre-fix push-up session failed after MD6 had received
        # the next genuine student state. Its state is exact, but its second
        # diagnostic was not committed, so saved MD6 probabilities are absent.
        failed_thread = "9c909170-00a6-4a33-8ad8-9ef003c2102e"
        failed_latest = latest_values_by_thread[failed_thread]
        failed_history = failed_latest.get("conversation_history")
        if not isinstance(failed_history, list) or not failed_history:
            raise AssertionError("Preserved pre-fix live history is unavailable")
        if failed_history[-1].get("role") != "student":
            raise AssertionError("Expected preserved failed live state to end in student")
        failed_diag = failed_latest.get("adaptive_turn_diagnostics")
        if not isinstance(failed_diag, Mapping):
            raise AssertionError("Preserved failed live diagnostic is unavailable")
        pending_turn = int(failed_diag["turn_index"]) + 1
        recovered[(failed_thread, pending_turn)] = {
            "thread_id": failed_thread,
            "turn_index": pending_turn,
            "problem": str(failed_latest["question"]),
            "conversation_history": project_live_history(failed_history),
            "historical_md6_probabilities": None,
            "historical_md6_move": None,
            "checkpoint_id": "latest preserved pre-fix state; diagnostic not committed",
        }
    finally:
        connection.close()

    if sha256_file(LIVE_DB) != db_hash_before:
        raise RuntimeError("Live checkpoint database changed during read-only recovery")
    wal_hash_after = sha256_file(wal_path) if wal_path.is_file() else None
    if wal_hash_after != wal_hash_before:
        raise RuntimeError("Live checkpoint WAL changed during read-only recovery")

    cases: list[dict[str, object]] = []
    for (thread_id, turn_index), item in sorted(recovered.items()):
        history = item["conversation_history"]
        if not isinstance(history, list):
            raise TypeError("Recovered live history has invalid type")
        cases.append(
            {
                "case_id": f"live_{thread_id[:8]}_turn_{turn_index:02d}",
                "case_set": LIVE_SET,
                "source_path": relative_or_absolute(LIVE_DB),
                "source_locator": (
                    f"thread_id={thread_id}; turn_index={turn_index}; "
                    f"checkpoint={item['checkpoint_id']}"
                ),
                "problem": item["problem"],
                "conversation_history": history,
                "latest_student_text": latest_student_text(history),
                "history_turn_count": len(history),
                "historical_md6_probabilities": item[
                    "historical_md6_probabilities"
                ],
                "historical_md6_move": item["historical_md6_move"],
                "semantic_group": MANUAL_THREADS[thread_id],
            }
        )
    return cases, {
        "checkpoint_db_sha256": db_hash_before,
        "checkpoint_wal_sha256": wal_hash_before,
        "manual_thread_ids": list(MANUAL_THREADS),
        "recovered_case_count": len(cases),
        "saved_md6_probability_count": sum(
            case["historical_md6_probabilities"] is not None for case in cases
        ),
    }


def rank(probabilities: Mapping[str, float]) -> list[str]:
    position = {move: index for index, move in enumerate(EXPECTED_LABELS)}
    return sorted(
        EXPECTED_LABELS,
        key=lambda move: (-float(probabilities[move]), position[move]),
    )


def validate_probabilities(probabilities: Mapping[str, float]) -> None:
    if tuple(probabilities) != EXPECTED_LABELS:
        raise AssertionError(f"Probability order changed: {tuple(probabilities)}")
    values = [float(probabilities[move]) for move in EXPECTED_LABELS]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AssertionError("Probability is non-finite or outside [0,1]")
    if not math.isclose(
        sum(values), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise AssertionError(f"Probability sum is not one: {sum(values)}")


def infer_cases(
    model: FrozenMD6Inference,
    cases: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = {}
    model._torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if model.model.device.type != "cpu":
        raise AssertionError("Diagnostic model is not on CPU")
    if model.tokenizer.truncation_side != "left":
        raise AssertionError("Tokenizer truncation_side is not left")
    for case in cases:
        case_id = str(case["case_id"])
        problem = case["problem"]
        history = case["conversation_history"]
        if not isinstance(problem, str) or not isinstance(history, list):
            raise TypeError(f"Invalid inference case {case_id}")
        probabilities = model.predict_probabilities(problem, history)
        validate_probabilities(probabilities)
        predictions[case_id] = probabilities
    return predictions


def reproduce_md6(
    cases: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    handwritten_moves: list[str] = []
    saved_comparisons: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        current = predictions[case_id]
        current_move = rank(current)[0]
        if case["case_set"] == HANDWRITTEN_SET:
            handwritten_moves.append(current_move)
        saved = case.get("historical_md6_probabilities")
        if isinstance(saved, Mapping):
            maximum = max(
                abs(float(current[move]) - float(saved[move]))
                for move in EXPECTED_LABELS
            )
            saved_comparisons.append(
                {
                    "case_id": case_id,
                    "case_set": case["case_set"],
                    "max_abs_difference": maximum,
                    "current_move": current_move,
                    "saved_move": case.get("historical_md6_move"),
                    "probabilities_match_at_1e-6": maximum <= PROBABILITY_TOLERANCE,
                    "argmax_matches": current_move
                    == str(case.get("historical_md6_move")),
                }
            )
    handwritten_counts = Counter(handwritten_moves)
    expected_handwritten = {
        "generic": 0,
        "probing": 24,
        "focus": 0,
        "telling": 0,
    }
    observed_handwritten = {
        move: int(handwritten_counts.get(move, 0)) for move in EXPECTED_LABELS
    }
    saved_ok = all(
        row["probabilities_match_at_1e-6"] and row["argmax_matches"]
        for row in saved_comparisons
    )
    passed = observed_handwritten == expected_handwritten and saved_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "scope": "all exactly recovered cases with historical saved outputs",
        "controlled_20": {
            "status": "UNAVAILABLE",
            "expected_distribution": {
                "generic": 2,
                "probing": 18,
                "focus": 0,
                "telling": 0,
            },
            "observed_distribution": None,
            "reason": "No exact 20-state artifact was found; no replacement was invented.",
        },
        "handwritten_24": {
            "status": "PASS"
            if observed_handwritten == expected_handwritten
            else "FAIL",
            "expected_distribution": expected_handwritten,
            "observed_distribution": observed_handwritten,
        },
        "saved_probability_comparisons": {
            "n": len(saved_comparisons),
            "max_abs_difference": max(
                (float(row["max_abs_difference"]) for row in saved_comparisons),
                default=None,
            ),
            "all_match_at_1e-6": saved_ok,
            "rows": saved_comparisons,
        },
    }


def probability_fields(prefix: str, probabilities: Mapping[str, float]) -> dict[str, object]:
    ordered = rank(probabilities)
    top, second = ordered[:2]
    return {
        **{
            f"{prefix}_p_{move}": float(probabilities[move])
            for move in EXPECTED_LABELS
        },
        f"{prefix}_move": top,
        f"{prefix}_top1_probability": float(probabilities[top]),
        f"{prefix}_top2_probability": float(probabilities[second]),
        f"{prefix}_top1_top2_gap": float(
            probabilities[top] - probabilities[second]
        ),
    }


def build_case_rows(
    cases: Sequence[Mapping[str, object]],
    md6_predictions: Mapping[str, Mapping[str, float]],
    md7_predictions: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        md6 = md6_predictions[case_id]
        md7 = md7_predictions[case_id]
        md6_fields = probability_fields("md6", md6)
        md7_fields = probability_fields("md7", md7)
        saved = case.get("historical_md6_probabilities")
        saved_difference: float | str = ""
        if isinstance(saved, Mapping):
            saved_difference = max(
                abs(float(md6[move]) - float(saved[move]))
                for move in EXPECTED_LABELS
            )
        md6_move = str(md6_fields["md6_move"])
        md7_move = str(md7_fields["md7_move"])
        rows.append(
            {
                "case_id": case_id,
                "case_set": case["case_set"],
                "source_path": case["source_path"],
                "source_locator": case["source_locator"],
                "problem": case["problem"],
                "conversation_history": json.dumps(
                    case["conversation_history"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "latest_student_text": case["latest_student_text"],
                "history_turn_count": case["history_turn_count"],
                "historical_md6_available": isinstance(saved, Mapping),
                "historical_md6_max_abs_difference": saved_difference,
                **md6_fields,
                **md7_fields,
                "move_changed": md6_move != md7_move,
                "transition": f"{md6_move} -> {md7_move}",
            }
        )
    return rows


def transition_matrix(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    return {
        source: {
            target: sum(
                row["md6_move"] == source and row["md7_move"] == target
                for row in rows
            )
            for target in EXPECTED_LABELS
        }
        for source in EXPECTED_LABELS
    }


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    if not n:
        return {"available": False, "n": 0}
    result: dict[str, object] = {"available": True, "n": n}
    for prefix in ("md6", "md7"):
        counts = Counter(str(row[f"{prefix}_move"]) for row in rows)
        ordered_counts = {
            move: int(counts.get(move, 0)) for move in EXPECTED_LABELS
        }
        percentages = {
            move: 100.0 * ordered_counts[move] / n for move in EXPECTED_LABELS
        }
        result[prefix] = {
            "counts": ordered_counts,
            "percentages": percentages,
            "probing_rate": percentages["probing"],
            "focus_rate": percentages["focus"],
            "telling_rate": percentages["telling"],
            "mean_probabilities": {
                move: fmean(float(row[f"{prefix}_p_{move}"]) for row in rows)
                for move in EXPECTED_LABELS
            },
            "maximum_single_class_share": max(percentages.values()),
        }
    changed = sum(bool(row["move_changed"]) for row in rows)
    result["argmax_changed_count"] = changed
    result["argmax_changed_percentage"] = 100.0 * changed / n
    result["probing_rate_change_percentage_points"] = (
        result["md7"]["probing_rate"] - result["md6"]["probing_rate"]
    )
    result["transition_matrix"] = transition_matrix(rows)
    return result


def eligible_moves(row: Mapping[str, object]) -> list[str]:
    base = str(row["md7_move"])
    base_probability = float(row[f"md7_p_{base}"])
    return [
        move
        for move in EXPECTED_LABELS
        if base_probability - float(row[f"md7_p_{move}"]) <= GAP_THRESHOLD
    ]


def overlay_rows(case_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in case_rows:
        eligible = eligible_moves(case)
        rows.append(
            {
                "case_id": case["case_id"],
                "case_set": case["case_set"],
                "md7_base_move": case["md7_move"],
                "md7_base_probability": case[f"md7_p_{case['md7_move']}"],
                "gap_threshold": GAP_THRESHOLD,
                "eligible_moves": json.dumps(eligible, separators=(",", ":")),
                "eligible_move_count": len(eligible),
                "alternative_available": len(eligible) > 1,
            }
        )
    return rows


def summarize_overlay(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    if not n:
        return {"available": False, "n": 0}
    alternative_count = sum(bool(row["alternative_available"]) for row in rows)
    by_base: dict[str, object] = {}
    for move in EXPECTED_LABELS:
        subset = [row for row in rows if row["md7_base_move"] == move]
        by_base[move] = {
            "n": len(subset),
            "alternative_available_count": sum(
                bool(row["alternative_available"]) for row in subset
            ),
            "alternative_available_percentage": (
                100.0
                * sum(bool(row["alternative_available"]) for row in subset)
                / len(subset)
                if subset
                else None
            ),
        }
    return {
        "available": True,
        "n": n,
        "baseline_only_percentage": 100.0 * (n - alternative_count) / n,
        "alternative_available_percentage": 100.0 * alternative_count / n,
        "mean_number_of_eligible_moves": fmean(
            int(row["eligible_move_count"]) for row in rows
        ),
        "by_md7_base_move": by_base,
    }


def critical_case_specs() -> list[dict[str, object]]:
    return [
        {
            "semantic_case": "opening state",
            "case_id": "live_9ab73ef2_turn_01",
            "description": "Opening state of the continued manual push-up retest.",
        },
        {
            "semantic_case": "short unexplained answer",
            "case_id": "live_9ab73ef2_turn_05",
            "description": "The student gives the short answer '15 days'.",
        },
        {
            "semantic_case": "fully correct reasoning",
            "case_id": "demo_10",
            "description": "The student states the correct divide-then-multiply reasoning for two thirds of 18.",
        },
        {
            "semantic_case": "specific arithmetic error",
            "case_id": "demo_17",
            "description": "The student answers 125 for 144 minus 18.",
        },
        {
            "semantic_case": "clear misconception",
            "case_id": "demo_13",
            "description": "The student adds rectangle side lengths and calls the result area.",
        },
        {
            "semantic_case": "repeated failure after hints",
            "case_id": "demo_22",
            "description": "The student still does not understand rectangle area after two hints.",
        },
        {
            "semantic_case": "explicit request for direct help",
            "case_id": "demo_24",
            "description": "After repeated scaffolding, the student says 'Just tell me what to do.'",
        },
        {
            "semantic_case": "I don't know after early prompt",
            "case_id": "demo_05",
            "description": "The first and only student reply is 'I don't know.'",
        },
        {
            "semantic_case": "I don't know after repeated scaffolding",
            "case_id": None,
            "description": "UNAVAILABLE: no recovered inference case ends with that exact semantic state.",
        },
    ]


def qualitative_change(category: str, md6_move: str, md7_move: str) -> str:
    if md6_move == md7_move:
        return "UNCHANGED"
    if category in {"specific arithmetic error", "clear misconception"}:
        return "PLAUSIBLY IMPROVED" if md7_move == "focus" else "QUESTIONABLE"
    if category == "fully correct reasoning":
        if md7_move == "generic":
            return "PLAUSIBLY IMPROVED"
        return "POSSIBLE NEW BIAS" if md7_move == "focus" else "QUESTIONABLE"
    if category == "repeated failure after hints":
        return (
            "PLAUSIBLY IMPROVED"
            if md7_move in {"focus", "telling"}
            else "QUESTIONABLE"
        )
    if category == "explicit request for direct help":
        return "PLAUSIBLY IMPROVED" if md7_move == "telling" else "QUESTIONABLE"
    if category in {
        "opening state",
        "I don't know after early prompt",
    }:
        if md7_move == "focus":
            return "POSSIBLE NEW BIAS"
        return "QUESTIONABLE"
    if category == "short unexplained answer":
        # The available case occurs late in a trajectory where the immediately
        # preceding exchange identifies a concrete missing starting-value issue.
        # Focus is therefore not, by itself, evidence of a new class bias.
        return "QUESTIONABLE"
    return "QUESTIONABLE"


def build_critical_review(
    case_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["case_id"]): row for row in case_rows}
    review: list[dict[str, object]] = []
    for spec in critical_case_specs():
        case_id = spec["case_id"]
        if case_id is None or str(case_id) not in by_id:
            review.append({**spec, "available": False})
            continue
        row = by_id[str(case_id)]
        md6_move = str(row["md6_move"])
        md7_move = str(row["md7_move"])
        review.append(
            {
                **spec,
                "available": True,
                "latest_student_text": row["latest_student_text"],
                "md6_probabilities": {
                    move: float(row[f"md6_p_{move}"]) for move in EXPECTED_LABELS
                },
                "md6_move": md6_move,
                "md7_probabilities": {
                    move: float(row[f"md7_p_{move}"]) for move in EXPECTED_LABELS
                },
                "md7_move": md7_move,
                "transition": row["transition"],
                "interpretation": qualitative_change(
                    str(spec["semantic_case"]), md6_move, md7_move
                ),
                "interpretation_boundary": (
                    "Qualitative diagnostic interpretation only; not a gold label."
                ),
            }
        )
    return review


def collapse_assessment(
    summaries: Mapping[str, Mapping[str, object]],
    critical_review: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    shares: dict[str, float | None] = {}
    focus_shares: dict[str, float | None] = {}
    for name in (*CASE_SETS, COMBINED_SET):
        summary = summaries[name]
        if not summary.get("available"):
            shares[name] = None
            focus_shares[name] = None
            continue
        md7 = summary["md7"]
        shares[name] = float(md7["maximum_single_class_share"])
        focus_shares[name] = float(md7["percentages"]["focus"])
    collapse_trigger_sets = (CONTROLLED_SET, HANDWRITTEN_SET, COMBINED_SET)
    new_collapse = any(
        shares[name] is not None and float(shares[name]) >= 80.0
        for name in collapse_trigger_sets
    )
    semantic_focus_bias_count = sum(
        row.get("available")
        and row.get("md7_move") == "focus"
        and row.get("interpretation") == "POSSIBLE NEW BIAS"
        for row in critical_review
    )
    observed_focus = [float(value) for value in focus_shares.values() if value is not None]
    max_focus = max(observed_focus, default=0.0)
    if max_focus >= 80.0 or semantic_focus_bias_count >= 3:
        risk = "HIGH"
    elif max_focus >= 60.0 or semantic_focus_bias_count >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "maximum_single_class_share_percent": shares,
        "md7_focus_share_percent": focus_shares,
        "new_class_collapse": new_collapse,
        "collapse_threshold_percent": 80.0,
        "collapse_trigger_sets": list(collapse_trigger_sets),
        "focus_collapse_risk": risk,
        "semantic_possible_new_focus_bias_count": semantic_focus_bias_count,
        "risk_basis": (
            "Observed MD7 focus shares plus focus transitions classified as possible new bias "
            "in the existing critical semantic cases."
        ),
    }


def choose_verdict(
    reproduction: Mapping[str, object],
    summaries: Mapping[str, Mapping[str, object]],
    collapse: Mapping[str, object],
) -> str:
    if reproduction["status"] != "PASS":
        return "D. DIAGNOSTIC INVALID — MD6 BASELINE COULD NOT BE REPRODUCED"
    if collapse["focus_collapse_risk"] in {"MEDIUM", "HIGH"}:
        return (
            "B. MD7-R1 REDUCES PROBING BUT SHOWS POSSIBLE NEW FOCUS BIAS — "
            "DO NOT DEPLOY YET"
        )
    combined = summaries[COMBINED_SET]
    probing_change = float(combined["probing_rate_change_percentage_points"])
    md7_nonzero = sum(
        int(combined["md7"]["counts"][move]) > 0 for move in EXPECTED_LABELS
    )
    if probing_change < -10.0 and md7_nonzero >= 3:
        return (
            "A. MD7-R1 SHOWS HEALTHIER ADAPTMATH MOVE DIVERSITY — "
            "READY FOR CONTROLLED LIVE TEST"
        )
    return "C. MD7-R1 DOES NOT MEANINGFULLY IMPROVE THE DIAGNOSTIC BEHAVIOR"


def write_csv_file(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def distribution_text(summary: Mapping[str, object], prefix: str) -> str:
    if not summary.get("available"):
        return "UNAVAILABLE"
    counts = summary[prefix]["counts"]
    return ", ".join(f"{move}={counts[move]}" for move in EXPECTED_LABELS)


def matrix_lines(summary: Mapping[str, object]) -> list[str]:
    if not summary.get("available"):
        return ["UNAVAILABLE"]
    matrix = summary["transition_matrix"]
    return [
        f"{source}->{target}: {matrix[source][target]}"
        for source in EXPECTED_LABELS
        for target in EXPECTED_LABELS
    ]


def build_text_report(summary: Mapping[str, object]) -> str:
    sets = summary["case_set_summaries"]
    collapse = summary["collapse_check"]
    overlay = summary["overlay_opportunity"][COMBINED_SET]
    lines: list[str] = []
    lines.extend(
        [
            "1. MODEL PATHS",
            f"MD6: {summary['models']['md6']['path']}",
            f"MD7-R1 epoch 3: {summary['models']['md7r1_epoch3']['path']}",
            "",
            "2. MODEL HASHES",
        ]
    )
    for model_name in ("md6", "md7r1_epoch3"):
        lines.append(model_name)
        for name, digest in summary["models"][model_name]["hashes"].items():
            lines.append(f"  {name}: {digest}")
    lines.extend(["", "3. CASE SOURCES"])
    for name, source in summary["case_sources"].items():
        lines.append(f"{name}: {json.dumps(source, ensure_ascii=False)}")
    lines.extend(["", "4. CASE AVAILABILITY"])
    for name, availability in summary["case_availability"].items():
        lines.append(
            f"{name}: expected={availability['expected']}, "
            f"recovered={availability['recovered']}, status={availability['status']}"
        )
    reproduction = summary["md6_reproduction"]
    lines.extend(
        [
            "",
            "5. MD6 REPRODUCTION",
            str(reproduction["status"]),
            "Controlled 20: UNAVAILABLE; expected generic=2, probing=18, focus=0, telling=0.",
            "Handwritten 24 observed: "
            + ", ".join(
                f"{move}={reproduction['handwritten_24']['observed_distribution'][move]}"
                for move in EXPECTED_LABELS
            ),
            (
                "Saved-probability parity: n="
                f"{reproduction['saved_probability_comparisons']['n']}, "
                "max_abs_difference="
                f"{reproduction['saved_probability_comparisons']['max_abs_difference']}, "
                "all_match_at_1e-6="
                f"{reproduction['saved_probability_comparisons']['all_match_at_1e-6']}"
            ),
        ]
    )
    section_map = (
        ("6. CONTROLLED 20 COMPARISON", CONTROLLED_SET),
        ("7. HANDWRITTEN 24 COMPARISON", HANDWRITTEN_SET),
        ("8. GENUINE LIVE TURN COMPARISON", LIVE_SET),
    )
    for title, name in section_map:
        set_summary = sets[name]
        lines.extend(["", title, f"N: {set_summary['n']}"])
        lines.append(f"MD6 distribution: {distribution_text(set_summary, 'md6')}")
        lines.append(f"MD7 distribution: {distribution_text(set_summary, 'md7')}")
        if set_summary.get("available"):
            lines.append(
                "Probing-rate change: "
                f"{set_summary['probing_rate_change_percentage_points']:+.6f} percentage points"
            )
            for prefix in ("md6", "md7"):
                metrics = set_summary[prefix]
                means = metrics["mean_probabilities"]
                lines.append(
                    f"{prefix.upper()} rates: probing={metrics['probing_rate']:.6f}%, "
                    f"focus={metrics['focus_rate']:.6f}%, "
                    f"telling={metrics['telling_rate']:.6f}%"
                )
                lines.append(
                    f"{prefix.upper()} mean probabilities: "
                    f"P(probing)={means['probing']:.9f}, "
                    f"P(focus)={means['focus']:.9f}, "
                    f"P(telling)={means['telling']:.9f}"
                )
            lines.append(
                f"Argmax changed: {set_summary['argmax_changed_count']} "
                f"({set_summary['argmax_changed_percentage']:.6f}%)"
            )
        lines.append("Transition matrix:")
        lines.extend(f"  {line}" for line in matrix_lines(set_summary))
    lines.extend(["", "9. CRITICAL CASE REVIEW"])
    for row in summary["critical_case_review"]:
        if not row["available"]:
            lines.append(f"{row['semantic_case']}: UNAVAILABLE")
            continue
        md6_probs = ", ".join(
            f"{move}={row['md6_probabilities'][move]:.6f}"
            for move in EXPECTED_LABELS
        )
        md7_probs = ", ".join(
            f"{move}={row['md7_probabilities'][move]:.6f}"
            for move in EXPECTED_LABELS
        )
        lines.extend(
            [
                f"{row['semantic_case']} | {row['case_id']} | {row['description']}",
                f"  latest student: {row['latest_student_text']}",
                f"  MD6: {md6_probs} | move={row['md6_move']}",
                f"  MD7: {md7_probs} | move={row['md7_move']}",
                f"  {row['transition']} | {row['interpretation']}",
            ]
        )
    shares = collapse["maximum_single_class_share_percent"]
    lines.extend(
        [
            "",
            "10. NEW CLASS COLLAPSE CHECK",
            f"Controlled max-class share: {shares[CONTROLLED_SET]}",
            f"Handwritten max-class share: {shares[HANDWRITTEN_SET]}",
            f"Live max-class share: {shares[LIVE_SET]}",
            f"Combined max-class share: {shares[COMBINED_SET]}",
            f"NEW_CLASS_COLLAPSE: {str(collapse['new_class_collapse']).upper()}",
            f"FOCUS_COLLAPSE_RISK: {collapse['focus_collapse_risk']}",
            "",
            "11. MD7 .10 OVERLAY OPPORTUNITY",
            f"Baseline-only %: {overlay['baseline_only_percentage']:.6f}",
            f"Alternative-available %: {overlay['alternative_available_percentage']:.6f}",
            f"Mean eligible moves: {overlay['mean_number_of_eligible_moves']:.6f}",
            "By MD7 base move:",
        ]
    )
    for move in EXPECTED_LABELS:
        item = overlay["by_md7_base_move"][move]
        lines.append(
            f"  {move}: n={item['n']}, alternative_available_percentage="
            f"{item['alternative_available_percentage']}"
        )
    lines.extend(["", "12. OUTPUT FILES"])
    lines.extend(str(path) for path in summary["output_files"])
    lines.extend(["", "13. EXTERNAL EFFECTS"])
    for name, value in summary["external_effects"].items():
        lines.append(f"{name}: {value}")
    lines.extend(
        [
            "",
            "14. FINAL DIAGNOSTIC VERDICT",
            str(summary["final_diagnostic_verdict"]),
            "",
            "Interpretation boundary: qualitative behavior diagnostic only; no gold-label accuracy claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if MOVE_ORDER != EXPECTED_LABELS:
        raise AssertionError(f"Canonical move order changed: {MOVE_ORDER}")
    if MAX_LENGTH != 512:
        raise AssertionError(f"MD6 max length changed: {MAX_LENGTH}")
    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
        raise FileExistsError(f"Diagnostic output path is not a directory: {OUTPUT_DIR}")

    models = {
        "md6": validate_model_artifacts(MD6_DIR),
        "md7r1_epoch3": validate_model_artifacts(MD7_DIR),
    }
    if models["md6"]["id2label"] != models["md7r1_epoch3"]["id2label"]:
        raise AssertionError("MD6 and MD7 class orders differ")

    handwritten_cases = load_handwritten_cases()
    live_cases, live_recovery = load_live_cases()
    cases = [*handwritten_cases, *live_cases]
    if len({str(case["case_id"]) for case in cases}) != len(cases):
        raise AssertionError("Case ids are not unique")

    md6_model = FrozenMD6Inference(MD6_DIR)
    md6_predictions = infer_cases(md6_model, cases)
    reproduction = reproduce_md6(cases, md6_predictions)
    if reproduction["status"] != "PASS":
        raise RuntimeError(
            "MD6 reproduction failed; stopping before MD7 interpretation. "
            + json.dumps(reproduction, ensure_ascii=False)
        )
    del md6_model
    gc.collect()

    md7_model = FrozenMD6Inference(MD7_DIR)
    md7_predictions = infer_cases(md7_model, cases)
    del md7_model
    gc.collect()

    case_rows = build_case_rows(cases, md6_predictions, md7_predictions)
    summaries: dict[str, Mapping[str, object]] = {}
    for case_set in CASE_SETS:
        summaries[case_set] = summarize_rows(
            [row for row in case_rows if row["case_set"] == case_set]
        )
    summaries[COMBINED_SET] = summarize_rows(case_rows)

    opportunity_rows = overlay_rows(case_rows)
    overlay_summary: dict[str, Mapping[str, object]] = {}
    for case_set in CASE_SETS:
        overlay_summary[case_set] = summarize_overlay(
            [row for row in opportunity_rows if row["case_set"] == case_set]
        )
    overlay_summary[COMBINED_SET] = summarize_overlay(opportunity_rows)

    critical_review = build_critical_review(case_rows)
    collapse = collapse_assessment(summaries, critical_review)
    verdict = choose_verdict(reproduction, summaries, collapse)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_path = OUTPUT_DIR / "md6_vs_md7r1_case_level.csv"
    summary_path = OUTPUT_DIR / "md6_vs_md7r1_summary.json"
    transition_path = OUTPUT_DIR / "md6_vs_md7r1_transition_matrix.csv"
    overlay_path = OUTPUT_DIR / "md7r1_overlay_opportunity.csv"
    critical_path = OUTPUT_DIR / "critical_case_review.json"
    report_path = OUTPUT_DIR / "md6_vs_md7r1_report.txt"

    write_csv_file(case_path, case_rows, CASE_FIELDS)
    transition_rows = [
        {
            "case_set": case_set,
            "n": summaries[case_set]["n"],
            "md6_move": source,
            "md7_move": target,
            "count": (
                summaries[case_set]["transition_matrix"][source][target]
                if summaries[case_set].get("available")
                else 0
            ),
        }
        for case_set in (*CASE_SETS, COMBINED_SET)
        for source in EXPECTED_LABELS
        for target in EXPECTED_LABELS
    ]
    write_csv_file(
        transition_path,
        transition_rows,
        ("case_set", "n", "md6_move", "md7_move", "count"),
    )
    write_csv_file(
        overlay_path,
        opportunity_rows,
        (
            "case_id",
            "case_set",
            "md7_base_move",
            "md7_base_probability",
            "gap_threshold",
            "eligible_moves",
            "eligible_move_count",
            "alternative_available",
        ),
    )
    critical_path.write_text(
        json.dumps(critical_review, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    output_files = [
        str(path.resolve())
        for path in (
            case_path,
            summary_path,
            transition_path,
            overlay_path,
            critical_path,
            report_path,
        )
    ]
    summary_payload: dict[str, object] = {
        "audit_name": AUDIT_NAME,
        "diagnostic_only": True,
        "models": models,
        "model_contract": {
            "input_pair": [
                "Problem:\n<problem>",
                "Conversation:\n<formatted current-session history>\n\nNext teacher pedagogical move:",
            ],
            "shared_inference_implementation": relative_or_absolute(
                PROJECT_ROOT / "src" / "self_improvement" / "md6_inference.py"
            ),
            "truncation_side": "left",
            "truncation": "only_second",
            "max_length": MAX_LENGTH,
            "label_order": list(EXPECTED_LABELS),
            "probability_sum_abs_tolerance": PROBABILITY_TOLERANCE,
            "device": "cpu",
        },
        "case_sources": {
            CONTROLLED_SET: {
                "status": "UNAVAILABLE",
                "exact_source": None,
                "reason": (
                    "No exact artifact for the historical 20-state, 2-generic/18-probing "
                    "audit was found. The repository's 24-case MD6 behavior audit was not substituted."
                ),
                "nonmatching_artifact_checked": relative_or_absolute(
                    PROJECT_ROOT
                    / "results"
                    / "self_improvement"
                    / "md6_move_behavior_v1"
                    / "raw_cases.csv"
                ),
            },
            HANDWRITTEN_SET: {
                "definition_source": relative_or_absolute(
                    HANDWRITTEN_DEFINITION_SOURCE
                ),
                "exact_serialized_case_and_saved_output_source": relative_or_absolute(
                    HANDWRITTEN_SOURCE
                ),
            },
            LIVE_SET: {
                "exact_checkpoint_source_read_only": relative_or_absolute(LIVE_DB),
                "completed_attempt_md6_corroboration": relative_or_absolute(
                    LIVE_ATTEMPTS
                ),
                "runtime_log_corroboration": relative_or_absolute(LIVE_LOG),
                "history_projection_contract": relative_or_absolute(LIVE_ADAPTER),
                "preserved_pre_fix_inspector": relative_or_absolute(PREFX_INSPECTOR),
                "post_fix_inspector": relative_or_absolute(POSTFIX_INSPECTOR),
                "recovery": live_recovery,
            },
        },
        "case_availability": {
            CONTROLLED_SET: {
                "expected": 20,
                "recovered": 0,
                "status": "UNAVAILABLE",
            },
            HANDWRITTEN_SET: {
                "expected": 24,
                "recovered": len(handwritten_cases),
                "status": "AVAILABLE",
            },
            LIVE_SET: {
                "expected": "all exact preserved manual move-selection states",
                "recovered": len(live_cases),
                "status": "AVAILABLE",
            },
            COMBINED_SET: {
                "expected": "all available cases",
                "recovered": len(cases),
                "status": "AVAILABLE",
            },
        },
        "md6_reproduction": reproduction,
        "case_set_summaries": summaries,
        "critical_case_review": critical_review,
        "collapse_check": collapse,
        "overlay_opportunity": overlay_summary,
        "external_effects": {
            "LinTS updates": 0,
            "real policy state writes": 0,
            "real experience log writes": 0,
            "BKT writes": 0,
            "Tutor Agent calls": 0,
            "Gemini/API calls": 0,
            "held-out MathDial test inference/data use": 0,
            "MRBench V3 test inference/data use": 0,
            "training": 0,
            "GPU required": "NO",
            "checkpoint database mode": "read-only",
            "pre-audit discovery caveat": (
                "One broad path-only rg scan touched the held-out MathDial and MRBench V3 "
                "test files during source discovery. No test record content was printed, "
                "extracted, selected, or used in this diagnostic."
            ),
        },
        "output_files": output_files,
        "final_diagnostic_verdict": verdict,
        "interpretation_boundary": (
            "Raw classifier behavior and qualitative semantic review only; no human gold "
            "labels were added and no accuracy claim is made."
        ),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_text_report(summary_payload), encoding="utf-8")

    print(json.dumps(
        {
            "md6_reproduction": reproduction["status"],
            "case_availability": summary_payload["case_availability"],
            "summaries": summaries,
            "collapse_check": collapse,
            "overlay_combined": overlay_summary[COMBINED_SET],
            "verdict": verdict,
            "output_files": output_files,
        },
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
