from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V1 = ROOT / "pedagogical-move-selection" / "results" / "md7_telling_real_state_diagnostic_v1"
ABLATION = ROOT / "pedagogical-move-selection" / "results" / "md7_telling_epoch_ablation_v1"
PREPARATION = ROOT / "pedagogical-move-selection" / "results" / "md7_r2_tell_c1_preparation_v1"
REAL_DATA = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1"
BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
CANDIDATE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7_r2_tell_c1" / "epoch2"
MATHDIAL_VALIDATION = ROOT / "pedagogical-move-selection" / "data" / "processed" / "mathdial" / "validation.jsonl"
SYNTHETIC_CHALLENGE = PREPARATION / "telling_boundary_challenge_val_v1.jsonl"
TURN_PATH = REAL_DATA / "turn_outcomes.jsonl"
ASSESSMENT_PATH = REAL_DATA / "assessment_outcomes.jsonl"
SUMMARY_PATH = REAL_DATA / "attempt_summaries.jsonl"
POLICY_PATH = ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "adaptive_demo_md7r1_real_v1" / "policy_state.json"

MOVES = ("generic", "probing", "focus", "telling")
LABEL_TO_ID = {move: index for index, move in enumerate(MOVES)}
TOLERANCE = 1e-5
EXPECTED_BASELINE_WEIGHT = "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13"
EXPECTED_FROZEN_HASHES = {
    "v1_analysis": "adb98cf8c978479d7aedb437c01bbbbe62e13b6bebd94acf5a932f0397dafd2c",
    "v1_challenge": "c415a0207fb3f4b960606317fc2d32c19cacf73f477d4900649321654bee141e",
    "v1_real_predictions": "ef6616a300770c48d31bf7d3325159f4da180b26d2e7a2b236f84a183d6ac8b7",
    "mathdial_validation": "225e6b8f671bf1e335af175343722aa9ac2a603a96d67bcb5220be1e92c0b2b6",
    "synthetic_challenge": "bc4d947d924fab5f47dc990e384a6715d1bf6bb77cee0b8aa76d2368297bfb4f",
}
EXPECTED_BASELINE_METRICS = {
    "accuracy": 0.5162162162162162,
    "macro_f1": 0.4611286390258053,
    "per_class_f1": {
        "generic": 0.6105263157894737,
        "probing": 0.3333333333333333,
        "focus": 0.6004672897196262,
        "telling": 0.300187617260788,
    },
    "prediction_distribution": {"generic": 413, "probing": 188, "focus": 1034, "telling": 215},
}
EXPECTED_CANDIDATE_MATHDIAL = {
    "accuracy": 0.5070270270270271,
    "macro_f1": 0.46257164850045035,
    "per_class_f1": {
        "generic": 0.6036866359447005,
        "probing": 0.30664395229982966,
        "focus": 0.5921299188007495,
        "telling": 0.34782608695652173,
    },
    "prediction_distribution": {"generic": 426, "probing": 175, "focus": 923, "telling": 326},
}
EXPECTED_CANDIDATE_SYNTHETIC = {
    "telling_precision": 0.9925558312655087,
    "telling_recall": 1.0,
    "telling_false_trigger_rate_on_non_telling": 0.0075,
    "telling_fp": 3,
}
OLD_REFERENCE = {
    "old tell epoch1": {
        "mathdial_macro_f1": 0.425292,
        "telling_plausible": "23/24",
        "hard_negative": "2/24",
        "recoverable": "2/4",
        "real_telling": "0/87",
        "displacement": "15/87",
        "displacement_rate": 0.172414,
        "mean_real_delta": 0.024483,
    },
    "old tell epoch2": {
        "mathdial_macro_f1": 0.441921,
        "telling_plausible": "24/24",
        "hard_negative": "3/24",
        "recoverable": "3/4",
        "real_telling": "5/87",
        "displacement": "20/87",
        "displacement_rate": 0.229885,
        "mean_real_delta": 0.038394,
    },
    "old tell epoch3": {
        "mathdial_macro_f1": 0.449797,
        "telling_plausible": "24/24",
        "hard_negative": "3/24",
        "recoverable": "3/4",
        "real_telling": "4/87",
        "displacement": "24/87",
        "displacement_rate": 0.275862,
        "mean_real_delta": 0.020975,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_v1():
    spec = importlib.util.spec_from_file_location("frozen_md7_telling_v1_for_r2", V1 / "analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen diagnostic implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_model_export(model_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    required = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    assert not missing, (str(model_dir), missing)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert {int(key): value for key, value in config["id2label"].items()} == dict(enumerate(MOVES))
    assert config["label2id"] == LABEL_TO_ID
    assert config["architectures"] == ["RobertaForSequenceClassification"]
    return config, {name: sha256(model_dir / name) for name in required}


def metric_bundle(probabilities: np.ndarray, labels: Sequence[int]) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    per_class = f1_score(truth, predictions, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "per_class_f1": {move: float(per_class[index]) for index, move in enumerate(MOVES)},
        "prediction_distribution": {move: int((predictions == index).sum()) for index, move in enumerate(MOVES)},
    }


def metric_match(observed: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-12) -> tuple[bool, dict[str, Any]]:
    deltas = {
        "accuracy": observed["accuracy"] - expected["accuracy"],
        "macro_f1": observed["macro_f1"] - expected["macro_f1"],
        "per_class_f1": {
            move: observed["per_class_f1"][move] - expected["per_class_f1"][move] for move in MOVES
        },
        "prediction_distribution_match": observed["prediction_distribution"] == expected["prediction_distribution"],
    }
    passed = (
        abs(deltas["accuracy"]) <= tolerance
        and abs(deltas["macro_f1"]) <= tolerance
        and all(abs(value) <= tolerance for value in deltas["per_class_f1"].values())
        and deltas["prediction_distribution_match"]
    )
    return passed, deltas


def predict_preformatted(model_dir: Path, records: Sequence[dict[str, Any]], batch_size: int = 16) -> np.ndarray:
    """Exact Kaggle contract for synthetic rows whose histories are already formatted strings."""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to("cpu")
    model.eval()
    assert model.config.id2label == dict(enumerate(MOVES))
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            first = ["Problem:\n" + str(row["problem"]) for row in batch]
            second = [
                "Conversation:\n"
                + (str(row["history"]) if str(row["history"]).strip() else "[No previous conversation]")
                + "\n\nNext teacher pedagogical move:"
                for row in batch
            ]
            encoded = tokenizer(
                first,
                second,
                truncation="only_second",
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            output.append(torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output, axis=0)


def vector_fields(prefix: str, vector: np.ndarray) -> dict[str, Any]:
    return {
        **{f"{prefix}_p_{move}": float(vector[index]) for index, move in enumerate(MOVES)},
        f"{prefix}_top1": MOVES[int(vector.argmax())],
    }


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    if not len(array):
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def boundary_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    telling = frame[f"{prefix}_p_telling"].astype(float)
    count = int((frame[f"{prefix}_top1"] == "telling").sum())
    return {
        "n": int(len(frame)),
        "top1_telling_count": count,
        "top1_telling_rate": float(count / len(frame)),
        "mean_telling_probability": float(telling.mean()),
        "median_telling_probability": float(telling.median()),
        "minimum_telling_probability": float(telling.min()),
        "maximum_telling_probability": float(telling.max()),
        "top1_distribution": {move: int((frame[f"{prefix}_top1"] == move).sum()) for move in MOVES},
    }


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_None._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in shown.columns]
    rows = [
        [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def compact_vector(row: pd.Series, prefix: str) -> str:
    return "[" + ", ".join(f"{move}={row[f'{prefix}_p_{move}']:.6f}" for move in MOVES) + "]"


def important_blocks(title: str, frame: pd.DataFrame, identity_column: str, history_column: str) -> str:
    text = f"## {title}\n\n"
    if frame.empty:
        return text + "_None._\n\n"
    for _, row in frame.iterrows():
        text += f"### {row[identity_column]}\n\n"
        if "semantic_group" in row:
            text += f"- Semantic group: `{row['semantic_group']}`; expected: `{row['expected_boundary']}`\n"
        else:
            text += (
                f"- Attempt/action: `{row['attempt_id']}` / `{row['action_event_id']}`; "
                f"turn `{row['action_turn_index']}`; skill `{row['skill']}`\n"
            )
        text += (
            f"- Transition: `{row['baseline_top1']} -> {row['candidate_top1']}`\n"
            f"- Baseline: `{compact_vector(row, 'baseline')}`\n"
            f"- Candidate: `{compact_vector(row, 'candidate')}`\n"
            f"- Delta telling/focus: `{row['delta_p_telling']:.6f}` / `{row['delta_p_focus']:.6f}`\n"
            f"- Problem: {row['problem']}\n\n"
            f"```text\n{row[history_column]}\n```\n\n"
        )
    return text


def write_mismatch(stage: str, details: dict[str, Any], side_effects: dict[str, Any]) -> None:
    summary = {
        "diagnostic": "md7_r2_tell_c1_epoch2_diagnostic_v1",
        "decision": "D. IMPLEMENTATION MISMATCH",
        "failed_stage": stage,
        "details": details,
        "side_effects": side_effects,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(
        "# MD7-R2-TELL-C1 Epoch 2 frozen diagnostic\n\n"
        "The required baseline safety gate failed, so candidate scoring stopped.\n\n"
        f"Failed stage: `{stage}`.\n\n"
        f"```json\n{json.dumps(details, indent=2, ensure_ascii=False)}\n```\n\n"
        "DECISION: D\n\n"
        "Recommended next experiment: repair the reproduction mismatch and rerun this same frozen diagnostic without changing cases, preprocessing, or thresholds.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = load_v1()

    frozen_files = {
        "v1_analysis": V1 / "analysis.py",
        "v1_challenge": V1 / "telling_boundary_challenge.csv",
        "v1_real_predictions": V1 / "real_state_predictions.csv",
        "ablation_analysis": ABLATION / "analysis.py",
        "ablation_summary": ABLATION / "summary.json",
        "ablation_epoch_comparison": ABLATION / "epoch_comparison.csv",
        "mathdial_validation": MATHDIAL_VALIDATION,
        "synthetic_challenge": SYNTHETIC_CHALLENGE,
        "turn_outcomes": TURN_PATH,
        "assessment_outcomes": ASSESSMENT_PATH,
        "attempt_summaries": SUMMARY_PATH,
        "policy_state": POLICY_PATH,
    }
    baseline_config, baseline_hashes = verify_model_export(BASELINE)
    candidate_config, candidate_hashes = verify_model_export(CANDIDATE)
    assert baseline_hashes["model.safetensors"] == EXPECTED_BASELINE_WEIGHT
    before_hashes = {name: sha256(path) for name, path in frozen_files.items()}
    before_hashes.update({f"baseline/{name}": value for name, value in baseline_hashes.items()})
    before_hashes.update({f"candidate/{name}": value for name, value in candidate_hashes.items()})
    for key, expected in EXPECTED_FROZEN_HASHES.items():
        assert before_hashes[key] == expected, (key, before_hashes[key], expected)

    # Verify the generated frozen cases are byte-equivalent in content to the frozen CSV artifact.
    challenges = v1.challenge_cases()
    frozen_challenge = pd.read_csv(V1 / "telling_boundary_challenge.csv")
    assert len(challenges) == len(frozen_challenge) == 48
    assert Counter(row["expected_boundary"] for row in challenges) == {"telling_plausible": 24, "non_telling": 24}
    for generated, (_, stored) in zip(challenges, frozen_challenge.iterrows(), strict=True):
        assert generated["case_id"] == stored["case_id"]
        assert generated["semantic_group"] == stored["semantic_group"]
        assert generated["expected_boundary"] == stored["expected_boundary"]
        assert generated["problem"] == stored["problem"]
        assert generated["history"] == json.loads(stored["history"])
        assert generated["rationale"] == stored["rationale"]

    mathdial = read_jsonl(MATHDIAL_VALIDATION)
    assert len(mathdial) == 1850
    mathdial_labels = [LABEL_TO_ID[row["target_move"]] for row in mathdial]

    # Gate 1: run the baseline on MathDial validation before any candidate prediction.
    baseline_mathdial_probabilities = v1.predict(BASELINE, mathdial, batch_size=16)
    baseline_mathdial = metric_bundle(baseline_mathdial_probabilities, mathdial_labels)
    baseline_match, baseline_deltas = metric_match(baseline_mathdial, EXPECTED_BASELINE_METRICS)
    if not baseline_match:
        side_effects = {
            "training": 0, "production_changes": 0, "tutor_api_calls": 0, "bkt_updates": 0,
            "lints_updates": 0, "authoritative_data_writes": 0, "mathdial_test_use": 0,
            "mrbench_v3_test_use": 0,
        }
        write_mismatch("baseline_mathdial_validation", {"observed": baseline_mathdial, "expected": EXPECTED_BASELINE_METRICS, "deltas": baseline_deltas}, side_effects)
        print(json.dumps({"decision": "D", "stage": "baseline_mathdial_validation", "deltas": baseline_deltas}, indent=2))
        return

    # Exact real-state reconstruction from the frozen v1 implementation.
    turns = read_jsonl(TURN_PATH)
    summaries = read_jsonl(SUMMARY_PATH)
    completed_ids = {
        str(nested(row, "identity", "attempt_id"))
        for row in summaries
        if row.get("completion_status") == "completed"
        or (row.get("completion_status") is None and row.get("adaptive_completion"))
    }
    real_source = [
        row
        for row in turns
        if str(nested(row, "identity", "attempt_id")) in completed_ids
        and nested(row, "provenance", "selector_mode") == "ordinary-md7r1-v1"
        and nested(row, "next_learner_observation", "status") in {"observed_update", "resolved_no_update"}
    ]
    assert len(real_source) == 87
    real_records = [
        {
            "problem": row["state_before_action"]["problem"],
            "history": row["state_before_action"].get("history_before_action") or [],
        }
        for row in real_source
    ]
    all_records = real_records + [{"problem": row["problem"], "history": row["history"]} for row in challenges]
    baseline_probabilities = v1.predict(BASELINE, all_records, batch_size=16)
    stored_probabilities = np.asarray([
        [float(nested(row, "selector", "raw", "probabilities", move)) for move in MOVES]
        for row in real_source
    ])
    parity_abs = np.abs(baseline_probabilities[:87] - stored_probabilities)
    parity_max = float(parity_abs.max())
    parity_rows_failed = int((parity_abs.max(axis=1) > TOLERANCE).sum())
    if parity_rows_failed:
        side_effects = {
            "training": 0, "production_changes": 0, "tutor_api_calls": 0, "bkt_updates": 0,
            "lints_updates": 0, "authoritative_data_writes": 0, "mathdial_test_use": 0,
            "mrbench_v3_test_use": 0,
        }
        write_mismatch("historical_real_state_baseline_parity", {"maximum_absolute_difference": parity_max, "rows_failing_1e_5": parity_rows_failed}, side_effects)
        print(json.dumps({"decision": "D", "stage": "historical_real_state_baseline_parity", "parity_max": parity_max, "rows_failed": parity_rows_failed}, indent=2))
        return

    # Only after both baseline gates pass may the candidate be scored.
    candidate_mathdial_probabilities = v1.predict(CANDIDATE, mathdial, batch_size=16)
    candidate_mathdial = metric_bundle(candidate_mathdial_probabilities, mathdial_labels)
    candidate_mathdial_match, candidate_mathdial_deltas = metric_match(
        candidate_mathdial, EXPECTED_CANDIDATE_MATHDIAL, tolerance=1e-12
    )

    synthetic_rows = read_jsonl(SYNTHETIC_CHALLENGE)
    assert len(synthetic_rows) == 800
    synthetic_labels = [LABEL_TO_ID[row["target_move"]] for row in synthetic_rows]
    synthetic_probabilities = predict_preformatted(CANDIDATE, synthetic_rows, batch_size=16)
    synthetic_metrics = metric_bundle(synthetic_probabilities, synthetic_labels)
    synthetic_predictions = synthetic_probabilities.argmax(axis=1)
    synthetic_truth = np.asarray(synthetic_labels)
    tell = LABEL_TO_ID["telling"]
    tell_tp = int(((synthetic_predictions == tell) & (synthetic_truth == tell)).sum())
    tell_fp = int(((synthetic_predictions == tell) & (synthetic_truth != tell)).sum())
    tell_fn = int(((synthetic_predictions != tell) & (synthetic_truth == tell)).sum())
    non_telling_n = int((synthetic_truth != tell).sum())
    synthetic_telling = {
        "telling_precision": tell_tp / (tell_tp + tell_fp) if tell_tp + tell_fp else 0.0,
        "telling_recall": tell_tp / (tell_tp + tell_fn) if tell_tp + tell_fn else 0.0,
        "telling_false_trigger_rate_on_non_telling": tell_fp / non_telling_n if non_telling_n else 0.0,
        "telling_tp": tell_tp,
        "telling_fp": tell_fp,
        "telling_fn": tell_fn,
    }
    candidate_synthetic_match = (
        abs(synthetic_telling["telling_precision"] - EXPECTED_CANDIDATE_SYNTHETIC["telling_precision"]) <= 1e-12
        and abs(synthetic_telling["telling_recall"] - EXPECTED_CANDIDATE_SYNTHETIC["telling_recall"]) <= 1e-12
        and abs(synthetic_telling["telling_false_trigger_rate_on_non_telling"] - EXPECTED_CANDIDATE_SYNTHETIC["telling_false_trigger_rate_on_non_telling"]) <= 1e-12
        and synthetic_telling["telling_fp"] == EXPECTED_CANDIDATE_SYNTHETIC["telling_fp"]
    )

    candidate_probabilities = v1.predict(CANDIDATE, all_records, batch_size=16)
    assert baseline_probabilities.shape == candidate_probabilities.shape == (135, 4)
    assert np.allclose(baseline_probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(candidate_probabilities.sum(axis=1), 1.0, atol=1e-6)

    real_rows: list[dict[str, Any]] = []
    for source, baseline_vector, candidate_vector in zip(
        real_source, baseline_probabilities[:87], candidate_probabilities[:87], strict=True
    ):
        identity = source["identity"]
        state = source["state_before_action"]
        outcome = source["next_learner_observation"]
        evaluator = outcome.get("evaluator") or {}
        update = source["learning_state_update"]
        baseline_top = MOVES[int(baseline_vector.argmax())]
        candidate_top = MOVES[int(candidate_vector.argmax())]
        real_rows.append({
            "learner_pseudonym": identity["student_pseudonymous_id"],
            "attempt_id": identity["attempt_id"],
            "action_event_id": identity["action_event_id"],
            "action_turn_index": identity["action_turn_index"],
            "skill": state["target_skill"],
            "problem": state["problem"],
            "history_before_action": json.dumps(state.get("history_before_action") or [], ensure_ascii=False, separators=(",", ":")),
            "formatted_history": v1.format_history(state.get("history_before_action") or []),
            "historical_raw_md7_move": nested(source, "selector", "raw", "argmax"),
            **vector_fields("baseline", baseline_vector),
            **vector_fields("candidate", candidate_vector),
            **{f"delta_p_{move}": float(candidate_vector[index] - baseline_vector[index]) for index, move in enumerate(MOVES)},
            "top1_changed": baseline_top != candidate_top,
            "changed_to_telling": baseline_top != "telling" and candidate_top == "telling",
            "changed_away_from_telling": baseline_top == "telling" and candidate_top != "telling",
            "top1_transition": f"{baseline_top} -> {candidate_top}",
            "evaluator_category_observational": str(evaluator.get("correctness", "unknown")).casefold(),
            "mastery_before_observational": update.get("mastery_before"),
            "immediate_delta_mastery_observational": update.get("delta_mastery"),
        })
    real_df = pd.DataFrame(real_rows)

    challenge_rows: list[dict[str, Any]] = []
    for source, baseline_vector, candidate_vector in zip(
        challenges, baseline_probabilities[87:], candidate_probabilities[87:], strict=True
    ):
        baseline_top = MOVES[int(baseline_vector.argmax())]
        candidate_top = MOVES[int(candidate_vector.argmax())]
        challenge_rows.append({
            **source,
            "history": json.dumps(source["history"], ensure_ascii=False, separators=(",", ":")),
            "formatted_history": v1.format_history(source["history"]),
            **vector_fields("baseline", baseline_vector),
            **vector_fields("candidate", candidate_vector),
            **{f"delta_p_{move}": float(candidate_vector[index] - baseline_vector[index]) for index, move in enumerate(MOVES)},
            "top1_changed": baseline_top != candidate_top,
            "changed_to_telling": baseline_top != "telling" and candidate_top == "telling",
            "changed_away_from_telling": baseline_top == "telling" and candidate_top != "telling",
            "top1_transition": f"{baseline_top} -> {candidate_top}",
        })
    challenge_df = pd.DataFrame(challenge_rows)
    plausible = challenge_df[challenge_df["expected_boundary"] == "telling_plausible"].copy()
    hard_negative = challenge_df[challenge_df["expected_boundary"] == "non_telling"].copy()
    recoverable = challenge_df[challenge_df["semantic_group"] == "I_recoverable_after_one_scaffold"].copy()
    assert len(plausible) == 24 and len(hard_negative) == 24 and len(recoverable) == 4

    plausible_baseline = boundary_metrics(plausible, "baseline")
    plausible_candidate = boundary_metrics(plausible, "candidate")
    negative_baseline = boundary_metrics(hard_negative, "baseline")
    negative_candidate = boundary_metrics(hard_negative, "candidate")
    recoverable_baseline_count = int((recoverable["baseline_top1"] == "telling").sum())
    recoverable_candidate_count = int((recoverable["candidate_top1"] == "telling").sum())

    real_telling_baseline = int((real_df["baseline_top1"] == "telling").sum())
    real_telling_candidate = int((real_df["candidate_top1"] == "telling").sum())
    displacement = int(real_df["top1_changed"].sum())
    movement = summary_stats(real_df["delta_p_telling"])
    transitions = {
        f"{baseline} -> {candidate}": int(len(frame))
        for (baseline, candidate), frame in real_df.groupby(["baseline_top1", "candidate_top1"], sort=True)
    }
    focus_to_telling = int(((real_df["baseline_top1"] == "focus") & (real_df["candidate_top1"] == "telling")).sum())
    probing_to_telling = int(((real_df["baseline_top1"] == "probing") & (real_df["candidate_top1"] == "telling")).sum())
    generic_to_telling = int(((real_df["baseline_top1"] == "generic") & (real_df["candidate_top1"] == "telling")).sum())
    telling_to_other = int(((real_df["baseline_top1"] == "telling") & (real_df["candidate_top1"] != "telling")).sum())

    sensitivity_gain = plausible_candidate["top1_telling_rate"] - plausible_baseline["top1_telling_rate"]
    false_trigger_change = negative_candidate["top1_telling_rate"] - negative_baseline["top1_telling_rate"]
    recoverable_rate = recoverable_candidate_count / 4
    displacement_rate = displacement / 87

    # Frozen conservative rubric from v1; labels are mapped to this experiment's decision names.
    if recoverable_rate >= 0.50:
        decision_code = "B"
        decision = "B. STILL OVERCORRECTS"
        decision_reason = "At least two of four recoverable-after-one-scaffold cases still trigger telling."
    elif negative_candidate["top1_telling_rate"] >= 0.20 or false_trigger_change >= 0.15:
        decision_code = "B"
        decision = "B. STILL OVERCORRECTS"
        decision_reason = "Hard-negative telling activation exceeds the unchanged frozen diagnostic boundary."
    elif plausible_candidate["top1_telling_rate"] < 0.50 or sensitivity_gain < 0.20:
        decision_code = "C"
        decision = "C. OVER-CORRECTED BACK TOWARD UNDER-TELLING"
        decision_reason = "The candidate no longer provides the required telling sensitivity gain on persistent difficulty."
    elif displacement_rate > 0.40:
        decision_code = "B"
        decision = "B. STILL OVERCORRECTS"
        decision_reason = "Real-state policy displacement exceeds the unchanged frozen diagnostic boundary."
    else:
        decision_code = "A"
        decision = "A. TARGETED RECALIBRATION SUCCESS"
        decision_reason = "The targeted recoverable-state correction passes while persistent-state sensitivity, hard-negative safety, and policy stability are retained."

    if decision_code == "A":
        recommended_experiment = "Run one preregistered blinded expert review on a fresh untouched set of baseline-candidate disagreement states before any live experiment."
    elif decision_code == "B":
        recommended_experiment = "Run a second matched-boundary recalibration focused only on the remaining recoverable false-telling patterns, then repeat this frozen diagnostic."
    else:
        recommended_experiment = "Run a narrowly balanced recalibration that restores multi-scaffold telling positives while preserving the corrected one-scaffold negatives, then repeat this frozen diagnostic."

    real_df.to_csv(OUT / "real_state_predictions.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    challenge_df.to_csv(OUT / "challenge_predictions.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    recoverable.to_csv(OUT / "recoverable_one_scaffold_cases.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    important = (
        "# Important Case Inspection\n\n"
        "All predictions use only pre-action state. Historical outcome fields are observational and are not used as labels.\n\n"
        "## Semantic audit findings\n\n"
        "The five real candidate-telling rows collapse to two attempts. Three rows reflect repeated unresolved "
        "difficulty or sustained uncertainty after multiple scaffolds. Two rows are concerning because the latest "
        "learner response shows correct progress: the warehouse learner has just distinguished per-box quantity from "
        "the total, and the diver has just computed the post-descent depth as -60. Neither is a one-scaffold case, so "
        "they do not negate the frozen 0/4 primary result, but they show residual sensitivity to already-recovering "
        "states. There are no candidate-telling hard-negative rows.\n\n"
        "| Action event | Latest learner signal | Semantic assessment |\n"
        "| --- | --- | --- |\n"
        "| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5` | Still equates division with the total after repeated probing | Persistent unresolved difficulty; telling is plausible |\n"
        "| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6` | Repeats the same division misconception after another scaffold | Persistent unresolved difficulty; telling is plausible |\n"
        "| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7` | Correctly says division gives per-box quantity, not the total | Recent recovery/correct progress; concerning telling trigger |\n"
        "| `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5` | Correct sign reasoning but remains unsure of the arithmetic result | Productively engaging after multiple scaffolds; borderline/possibly early |\n"
        "| `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6` | Correctly computes -18 + (-42) = -60 | Recent recovery/correct progress; concerning telling trigger |\n\n"
        "Across the ten largest telling-probability increases, non-telling top-1 is retained on recent-progress rows "
        "from the rectangle and fraction attempts, while the largest three increases are the persistent warehouse "
        "sequence above. Across the ten largest focus decreases, the candidate often moves mass toward probing or "
        "generic on productive/correct-progress states; only the two diver rows above cross to telling. Full context "
        "and vectors for every inspected row follow.\n\n"
    )
    important += important_blocks(
        "All real deployment states where candidate top-1 is telling",
        real_df[real_df["candidate_top1"] == "telling"].sort_values("candidate_p_telling", ascending=False),
        "action_event_id",
        "formatted_history",
    )
    important += important_blocks(
        "All hard-negative challenge states where candidate top-1 is telling",
        hard_negative[hard_negative["candidate_top1"] == "telling"].sort_values("candidate_p_telling", ascending=False),
        "case_id",
        "formatted_history",
    )
    important += important_blocks(
        "All four recoverable-after-one-scaffold cases",
        recoverable.sort_values("case_id"),
        "case_id",
        "formatted_history",
    )
    important += important_blocks(
        "Largest 10 real-state telling-probability increases",
        real_df.sort_values("delta_p_telling", ascending=False).head(10),
        "action_event_id",
        "formatted_history",
    )
    important += important_blocks(
        "Largest 10 real-state focus-probability decreases",
        real_df.sort_values("delta_p_focus", ascending=True).head(10),
        "action_event_id",
        "formatted_history",
    )
    (OUT / "important_cases.md").write_text(important, encoding="utf-8")

    candidate_telling_real = real_df[real_df["candidate_top1"] == "telling"]
    observational = {
        "n": int(len(candidate_telling_real)),
        "evaluator_category_distribution": dict(Counter(candidate_telling_real["evaluator_category_observational"])),
        "mastery_before": summary_stats(candidate_telling_real["mastery_before_observational"]),
        "immediate_delta_mastery": summary_stats(candidate_telling_real["immediate_delta_mastery_observational"]),
        "causal_interpretation": False,
    }

    after_hashes = {name: sha256(path) for name, path in frozen_files.items()}
    after_hashes.update({f"baseline/{name}": sha256(BASELINE / name) for name in baseline_hashes})
    after_hashes.update({f"candidate/{name}": sha256(CANDIDATE / name) for name in candidate_hashes})
    protected_changes = {
        name: {"before": before_hashes[name], "after": after_hashes[name]}
        for name in before_hashes
        if before_hashes[name] != after_hashes[name]
    }
    assert not protected_changes, protected_changes

    side_effects = {
        "training": 0,
        "production_changes": 0,
        "tutor_api_calls": 0,
        "bkt_updates": 0,
        "lints_updates": 0,
        "authoritative_data_writes": 0,
        "mathdial_test_use": 0,
        "mrbench_v3_test_use": 0,
        "protected_input_hash_changes": protected_changes,
    }
    summary = {
        "diagnostic": "md7_r2_tell_c1_epoch2_diagnostic_v1",
        "decision": decision,
        "decision_code": decision_code,
        "decision_reason": decision_reason,
        "models": {
            "baseline": {"path": str(BASELINE.relative_to(ROOT)).replace("\\", "/"), "hashes": baseline_hashes, "id2label": baseline_config["id2label"]},
            "candidate": {"path": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"), "hashes": candidate_hashes, "id2label": candidate_config["id2label"]},
        },
        "frozen_inputs": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": before_hashes[name]} for name, path in frozen_files.items()},
        "inference_contract": {
            "first_sequence": "Problem:\n<problem>",
            "second_sequence": "Conversation:\n<formatted dialogue>\n\nNext teacher pedagogical move:",
            "dialogue_format": "{user}: {text}",
            "tokenizer_truncation_side": "left",
            "truncation": "only_second",
            "max_length": 512,
            "move_order": list(MOVES),
        },
        "baseline_safety": {
            "status": "PASS",
            "mathdial_validation": baseline_mathdial,
            "expected": EXPECTED_BASELINE_METRICS,
            "deltas": baseline_deltas,
            "real_state_parity_max_abs_error": parity_max,
            "real_state_parity_rows_failed_at_1e_5": parity_rows_failed,
        },
        "candidate_validation_reference": {
            "mathdial": {"observed": candidate_mathdial, "expected": EXPECTED_CANDIDATE_MATHDIAL, "exact_match": candidate_mathdial_match, "deltas": candidate_mathdial_deltas},
            "synthetic_telling_challenge": {"observed": synthetic_telling, "full_metrics": synthetic_metrics, "expected": EXPECTED_CANDIDATE_SYNTHETIC, "exact_match": candidate_synthetic_match},
        },
        "challenge": {
            "n": 48,
            "telling_plausible": {"baseline": plausible_baseline, "candidate": plausible_candidate, "sensitivity_gain": sensitivity_gain},
            "hard_non_telling": {"baseline": negative_baseline, "candidate": negative_candidate, "false_trigger_change": false_trigger_change},
            "recoverable_after_one_scaffold": {
                "n": 4,
                "baseline_false_telling_count": recoverable_baseline_count,
                "candidate_false_telling_count": recoverable_candidate_count,
                "candidate_false_telling_rate": recoverable_rate,
            },
        },
        "real_states": {
            "n": 87,
            "baseline_top1_telling_count": real_telling_baseline,
            "candidate_top1_telling_count": real_telling_candidate,
            "candidate_top1_telling_rate": real_telling_candidate / 87,
            "policy_displacement_count": displacement,
            "policy_displacement_rate": displacement_rate,
            "focus_to_telling_count": focus_to_telling,
            "probing_to_telling_count": probing_to_telling,
            "generic_to_telling_count": generic_to_telling,
            "telling_to_other_count": telling_to_other,
            "top1_transitions": transitions,
            "telling_probability_delta": movement,
        },
        "semantic_inspection": {
            "candidate_telling_real_states_inspected": 5,
            "candidate_telling_hard_negative_states_inspected": 0,
            "largest_telling_probability_increases_inspected": 10,
            "largest_focus_probability_decreases_inspected": 10,
            "candidate_telling_assessment": {
                "persistent_unresolved_or_sustained_uncertainty": 3,
                "latest_response_shows_correct_progress_concerning": 2,
                "one_scaffold_cases": 0
            },
            "concerning_action_event_ids": [
                "a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7",
                "bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6"
            ],
            "interpretation": "Residual telling sensitivity remains on two already-recovering real states, but neither is a recoverable-after-one-scaffold case; no frozen hard negative becomes telling."
        },
        "observational_outcome_join": observational,
        "historical_failed_candidates": OLD_REFERENCE,
        "claims_not_entitled": [
            "candidate telling would have caused better learning",
            "candidate would have improved a historical outcome",
            "negative BKT proves telling was needed",
            "offline or synthetic performance proves production effectiveness",
        ],
        "recommended_next_experiment": recommended_experiment,
        "side_effects": side_effects,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model_table = pd.DataFrame([
        {"model": "MD7-R1", "path": summary["models"]["baseline"]["path"], **baseline_hashes},
        {"model": "MD7-R2-C1 epoch2", "path": summary["models"]["candidate"]["path"], **candidate_hashes},
    ])
    boundary_table = pd.DataFrame([
        {"boundary": "telling-plausible", "model": "MD7-R1", **plausible_baseline},
        {"boundary": "telling-plausible", "model": "MD7-R2-C1 epoch2", **plausible_candidate},
        {"boundary": "hard non-telling", "model": "MD7-R1", **negative_baseline},
        {"boundary": "hard non-telling", "model": "MD7-R2-C1 epoch2", **negative_candidate},
    ])
    recoverable_display = recoverable[[
        "case_id", "problem", "formatted_history", "expected_boundary",
        *[f"baseline_p_{move}" for move in MOVES], "baseline_top1",
        *[f"candidate_p_{move}" for move in MOVES], "candidate_top1",
    ]]
    final_table = pd.DataFrame([
        {
            "metric": "MathDial Macro-F1",
            "MD7-R1": f"{baseline_mathdial['macro_f1']:.6f}",
            "old tell epoch1": "0.425292", "old tell epoch2": "0.441921", "old tell epoch3": "0.449797",
            "MD7-R2-C1 epoch2": f"{candidate_mathdial['macro_f1']:.6f}",
        },
        {
            "metric": "telling-plausible count /24",
            "MD7-R1": f"{plausible_baseline['top1_telling_count']}/24",
            "old tell epoch1": "23/24", "old tell epoch2": "24/24", "old tell epoch3": "24/24",
            "MD7-R2-C1 epoch2": f"{plausible_candidate['top1_telling_count']}/24",
        },
        {
            "metric": "hard-negative false telling /24",
            "MD7-R1": f"{negative_baseline['top1_telling_count']}/24",
            "old tell epoch1": "2/24", "old tell epoch2": "3/24", "old tell epoch3": "3/24",
            "MD7-R2-C1 epoch2": f"{negative_candidate['top1_telling_count']}/24",
        },
        {
            "metric": "recoverable false telling /4",
            "MD7-R1": f"{recoverable_baseline_count}/4",
            "old tell epoch1": "2/4", "old tell epoch2": "3/4", "old tell epoch3": "3/4",
            "MD7-R2-C1 epoch2": f"{recoverable_candidate_count}/4",
        },
        {
            "metric": "real-state telling /87",
            "MD7-R1": f"{real_telling_baseline}/87",
            "old tell epoch1": "0/87", "old tell epoch2": "5/87", "old tell epoch3": "4/87",
            "MD7-R2-C1 epoch2": f"{real_telling_candidate}/87",
        },
        {
            "metric": "real-state displacement /87",
            "MD7-R1": "0/87",
            "old tell epoch1": "15/87", "old tell epoch2": "20/87", "old tell epoch3": "24/87",
            "MD7-R2-C1 epoch2": f"{displacement}/87",
        },
        {
            "metric": "displacement %",
            "MD7-R1": "0.000%",
            "old tell epoch1": "17.241%", "old tell epoch2": "22.989%", "old tell epoch3": "27.586%",
            "MD7-R2-C1 epoch2": f"{100 * displacement_rate:.3f}%",
        },
        {
            "metric": "mean real-state telling-probability delta",
            "MD7-R1": "0.000000",
            "old tell epoch1": "0.024483", "old tell epoch2": "0.038394", "old tell epoch3": "0.020975",
            "MD7-R2-C1 epoch2": f"{movement['mean']:.6f}",
        },
    ])

    report = f"""# MD7-R2-TELL-C1 Epoch 2 frozen telling-boundary diagnostic

This is a derived-only offline evaluation. It trained and promoted nothing, changed no runtime behavior, called no Tutor API, and did not read MathDial test or MRBench V3 test.

## Artifact and contract verification

The candidate is a complete Hugging Face export. Both configs specify the exact move order `generic, probing, focus, telling` and `RobertaForSequenceClassification`.

{markdown_table(model_table)}

The reused contract is exactly: `Problem:\n<problem>` paired with `Conversation:\n<formatted dialogue>\n\nNext teacher pedagogical move:`; dialogue lines are `user: text`; tokenizer truncation side is left; truncation is `only_second`; maximum length is 512.

Frozen artifacts were verified by SHA-256 before scoring: the v1 implementation, frozen 48-case CSV, frozen v1 real-state predictions, MathDial validation, and the C1 synthetic challenge. The generated case objects match all 48 frozen CSV rows field-for-field.

## Baseline safety reproduction

Gate: **PASS**. On all 1,850 MathDial validation rows, MD7-R1 reproduced accuracy **{baseline_mathdial['accuracy']:.16f}**, Macro-F1 **{baseline_mathdial['macro_f1']:.16f}**, and per-class F1 generic/probing/focus/telling **{baseline_mathdial['per_class_f1']['generic']:.16f} / {baseline_mathdial['per_class_f1']['probing']:.16f} / {baseline_mathdial['per_class_f1']['focus']:.16f} / {baseline_mathdial['per_class_f1']['telling']:.16f}**.

The 87-state historical raw-probability parity check also passed: maximum absolute difference **{parity_max:.10g}**; rows above `1e-5`: **{parity_rows_failed}**.

## Candidate validation-reference reproduction

Local candidate MathDial validation: accuracy **{candidate_mathdial['accuracy']:.16f}**, Macro-F1 **{candidate_mathdial['macro_f1']:.16f}**, per-class F1 generic/probing/focus/telling **{candidate_mathdial['per_class_f1']['generic']:.16f} / {candidate_mathdial['per_class_f1']['probing']:.16f} / {candidate_mathdial['per_class_f1']['focus']:.16f} / {candidate_mathdial['per_class_f1']['telling']:.16f}**; prediction distribution `{json.dumps(candidate_mathdial['prediction_distribution'], sort_keys=True)}`. Exact reference match: **{candidate_mathdial_match}**.

Local existing synthetic telling challenge: precision **{synthetic_telling['telling_precision']:.16f}**, recall **{synthetic_telling['telling_recall']:.16f}**, false-trigger rate **{synthetic_telling['telling_false_trigger_rate_on_non_telling']:.16f}**, FP **{synthetic_telling['telling_fp']}**. Exact reference match: **{candidate_synthetic_match}**.

These validation results are retention/calibration evidence only, not evidence of real tutoring effectiveness.

## Frozen 48-case telling-boundary challenge

{markdown_table(boundary_table.drop(columns=['top1_distribution']))}

Top-1 prediction distributions:

- Telling-plausible MD7-R1: `{json.dumps(plausible_baseline['top1_distribution'], sort_keys=True)}`
- Telling-plausible candidate: `{json.dumps(plausible_candidate['top1_distribution'], sort_keys=True)}`
- Hard-negative MD7-R1: `{json.dumps(negative_baseline['top1_distribution'], sort_keys=True)}`
- Hard-negative candidate: `{json.dumps(negative_candidate['top1_distribution'], sort_keys=True)}`

Sensitivity gain relative to baseline: **{sensitivity_gain:.6f}**. Hard-negative false-trigger change: **{false_trigger_change:.6f}**.

## Primary test: recoverable after one scaffold

Baseline false telling: **{recoverable_baseline_count}/4**. Candidate false telling: **{recoverable_candidate_count}/4**.

{markdown_table(recoverable_display, digits=6)}

The full vectors and exact dialogue histories are also preserved in `recoverable_one_scaffold_cases.csv` and `important_cases.md`.

## Exact 87 real deployment states

- Candidate top-1 telling: **{real_telling_candidate}/87 ({100 * real_telling_candidate / 87:.3f}%)**; baseline **{real_telling_baseline}/87**.
- Policy displacement: **{displacement}/87 ({100 * displacement_rate:.3f}%)**.
- Focus/probing/generic to telling: **{focus_to_telling} / {probing_to_telling} / {generic_to_telling}**.
- Telling to another move: **{telling_to_other}**.
- Candidate-minus-baseline telling probability: mean **{movement['mean']:.6f}**, median **{movement['median']:.6f}**, maximum increase **{movement['max']:.6f}**.
- All transitions: `{json.dumps(transitions, sort_keys=True)}`.

Every real candidate-telling state, every hard-negative candidate-telling state, the four recoverable cases, the ten largest telling increases, and the ten largest focus decreases are shown with exact context and both vectors in `important_cases.md`.

## Semantic inspection

All five real candidate-telling rows were inspected. Three reflect repeated unresolved difficulty or sustained uncertainty after multiple meaningful scaffolds. Two are residual concerns because the latest learner response already shows correct progress: the warehouse learner has just recognized that division would give notebooks per box rather than the total, and the diver has just correctly computed the post-descent depth as -60. Neither is a recoverable-after-one-scaffold state. The candidate produces no top-1 telling prediction on any of the 24 hard negatives.

The ten largest telling-probability increases and ten largest focus-probability decreases were also inspected. On recent-progress rows from the rectangle and fraction attempts, the telling probability rises but top-1 remains probing, focus, or generic. Among the largest focus decreases, only the two diver rows cross to telling; other productive-progress rows remain non-telling. Thus the targeted frozen boundary is corrected, but the real-state sample still shows a narrower residual risk around multi-scaffold recovery. Historical outcomes do not establish what telling would have caused.

## Observational historical outcome join

Outcome fields were joined only after independent pre-action prediction. Candidate-telling real-state count: **{observational['n']}**; evaluator distribution `{json.dumps(observational['evaluator_category_distribution'], sort_keys=True)}`. These fields describe historical states only and support no causal claim about an alternative action.

## Comparison and interpretation

The old candidates all failed the recoverable boundary (2/4, 3/4, and 3/4) while displacing 17.241%, 22.989%, and 27.586% of real states. The C1 candidate is judged with the unchanged frozen rubric and the preregistered primary interpretation: 0/4 is strong correction evidence, 1/4 is meaningful improvement requiring inspection, and 2+/4 is inadequate correction.

**{decision}**

{decision_reason} This remains an offline candidate diagnostic; no promotion is authorized.

## Safety audit

- Training: 0
- Production changes: 0
- Tutor API calls: 0
- BKT updates: 0
- LinTS updates: 0
- Authoritative-data writes: 0
- Protected MathDial test use: 0
- MRBench V3 test use: 0
- Protected input hash changes: 0

## Final comparison table

{markdown_table(final_table)}

DECISION: {decision_code}

Recommended next experiment: {recommended_experiment}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "baseline_gate": "PASS",
        "candidate_mathdial_exact_match": candidate_mathdial_match,
        "candidate_synthetic_exact_match": candidate_synthetic_match,
        "real_parity_max_abs": parity_max,
        "real_parity_rows_failed": parity_rows_failed,
        "telling_plausible": f"{plausible_candidate['top1_telling_count']}/24",
        "hard_negative_false_telling": f"{negative_candidate['top1_telling_count']}/24",
        "recoverable_false_telling": f"{recoverable_candidate_count}/4",
        "real_telling": f"{real_telling_candidate}/87",
        "displacement": f"{displacement}/87",
        "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
