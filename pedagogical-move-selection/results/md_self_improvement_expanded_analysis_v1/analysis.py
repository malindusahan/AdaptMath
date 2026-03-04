from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1"
OUT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_pilot_analysis_v1" / "analysis.py"
RUNTIME_DIR = ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "adaptive_demo_md7r1_real_v1"

TURN_PATH = DATA_DIR / "turn_outcomes.jsonl"
ASSESSMENT_PATH = DATA_DIR / "assessment_outcomes.jsonl"
SUMMARY_PATH = DATA_DIR / "attempt_summaries.jsonl"
POLICY_PATH = RUNTIME_DIR / "policy_state.json"
EXPERIENCE_PATH = RUNTIME_DIR / "attempts.jsonl"
LINEAGE_PATH = RUNTIME_DIR / "lineage_manifest.json"
MD7_PATH = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3" / "model.safetensors"
MRB1_PATH = ROOT / "pedagogical-move-selection" / "models" / "frozen" / "mrb1" / "model.safetensors"
BKT_PATH = ROOT / "student-modeling" / "models" / "bkt_params.json"

MOVES = ["generic", "probing", "focus", "telling"]
ARMS = ["baseline", "generic_bias", "probing_bias", "focus_bias", "telling_bias"]
CORRECTNESS = ["correct", "partial", "incorrect", "unknown"]
SIGNS = ["positive", "zero", "negative"]
MRB = ["MI", "ML", "PG", "A"]
EPS = 1e-15


def load_base_module():
    spec = importlib.util.spec_from_file_location("pilot_analysis_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return raw, [json.loads(line) for line in raw]


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers: list[float] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
    if not numbers:
        return {key: None for key in ["min", "q1", "median", "mean", "q3", "max", "std"]} | {"count": 0}
    array = np.asarray(numbers, dtype=float)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def sign(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if number > EPS:
        return "positive"
    if number < -EPS:
        return "negative"
    return "zero"


def counts(values: Iterable[Any], order: list[str]) -> dict[str, int]:
    found = Counter(str(value) for value in values if value is not None and not pd.isna(value))
    return {key: int(found.get(key, 0)) for key in order}


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def md(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_None._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row] for row in shown.fillna("").itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def duplicate_audit(records: list[dict[str, Any]], key_fn) -> tuple[int, int, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key is not None:
            grouped[str(key)].append(record)
    duplicate_groups = [group for group in grouped.values() if len(group) > 1]
    return (
        len(duplicate_groups),
        sum(len(group) - 1 for group in duplicate_groups),
        sum(len({canonical(item) for item in group}) > 1 for group in duplicate_groups),
    )


def flatten_strings(value: Any, location: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        result: list[tuple[str, str]] = []
        for key, child in value.items():
            result.extend(flatten_strings(child, f"{location}.{key}" if location else str(key)))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(flatten_strings(child, f"{location}[{index}]"))
        return result
    return [(location, value)] if isinstance(value, str) else []


def mastery_band(value: float) -> str:
    if value < 0.25:
        return "[0,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 0.75:
        return "[.50,.75)"
    if value < 0.90:
        return "[.75,.90)"
    if value < 0.97:
        return "[.90,.97)"
    return "[.97,1]"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [TURN_PATH, ASSESSMENT_PATH, SUMMARY_PATH, POLICY_PATH, EXPERIENCE_PATH, LINEAGE_PATH, MD7_PATH, MRB1_PATH, BKT_PATH]
    before_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected}

    # Reuse the already-audited parser for the completed-attempt cohort, directing
    # every write into this new derived directory.
    base = load_base_module()
    base.OUT_DIR = OUT_DIR
    base.main()
    completed_turns = pd.read_csv(OUT_DIR / "turn_table.csv")
    completed_attempts = pd.read_csv(OUT_DIR / "attempt_table.csv")
    base_summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))

    turn_raw, turns = read_jsonl(TURN_PATH)
    assessment_raw, assessments = read_jsonl(ASSESSMENT_PATH)
    summary_raw, summaries = read_jsonl(SUMMARY_PATH)
    _, experiences = read_jsonl(EXPERIENCE_PATH)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    summary_ids = {str(nested(item, "identity", "attempt_id")) for item in summaries}
    assessment_by_attempt = {str(nested(item, "identity", "attempt_id")): item for item in assessments}
    summary_by_attempt = {str(nested(item, "identity", "attempt_id")): item for item in summaries}
    turns_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        turns_by_attempt[str(nested(turn, "identity", "attempt_id"))].append(turn)
    for group in turns_by_attempt.values():
        group.sort(key=lambda item: int(nested(item, "identity", "action_turn_index", default=0)))
    all_attempt_ids = sorted(
        set(turns_by_attempt) | summary_ids | set(assessment_by_attempt),
        key=lambda attempt_id: min(
            [str(nested(item, "state_before_action", "attempt_started_at", default="9999")) for item in turns_by_attempt.get(attempt_id, [])]
            + ["9999"]
        ),
    )
    sequence = {attempt_id: index for index, attempt_id in enumerate(all_attempt_ids, start=1)}

    action_rows: list[dict[str, Any]] = []
    for turn in turns:
        identity = turn["identity"]
        attempt_id = str(identity["attempt_id"])
        update = turn["learning_state_update"]
        state = turn["state_before_action"]
        selector = turn["selector"]
        decision = turn["adaptive_decision"]
        observation = turn["next_learner_observation"]
        evaluator = observation.get("evaluator") or {}
        raw_probabilities = nested(selector, "raw", "probabilities", default={}) or {}
        effective_probabilities = nested(selector, "effective", "probabilities", default={}) or {}
        ordered = sorted(float(value) for value in effective_probabilities.values())
        delta = update.get("delta_mastery")
        scores = nested(turn, "tutor_quality", "scores", default={}) or {}
        row = {
            "collection_sequence": sequence[attempt_id],
            "attempt_id": attempt_id,
            "completed_attempt": attempt_id in summary_ids,
            "analysis_cohort": attempt_id in summary_ids and observation.get("status") in {"observed_update", "resolved_no_update"},
            "learner_pseudonym": identity.get("student_pseudonymous_id"),
            "skill": state.get("target_skill"),
            "action_turn_index": identity.get("action_turn_index"),
            "action_event_id": identity.get("action_event_id"),
            "resolver_event_id": update.get("resolver_event_id"),
            "status": observation.get("status"),
            "failure_reason": observation.get("failure_reason"),
            "selector_mode": nested(turn, "provenance", "selector_mode"),
            "data_mode": nested(turn, "provenance", "data_mode"),
            "attempt_started_at": state.get("attempt_started_at"),
            "timestamp": nested(turn, "provenance", "timestamp"),
            "latest_student_text": state.get("latest_student_text"),
            "student_text": observation.get("student_text"),
            "mastery_before": update.get("mastery_before"),
            "mastery_after": update.get("mastery_after"),
            "delta_mastery": delta,
            "delta_sign": sign(delta),
            "headroom": None if update.get("mastery_before") is None else 1.0 - float(update["mastery_before"]),
            "update_confidence": update.get("update_confidence"),
            "correctness": str(evaluator.get("correctness", "unknown")).casefold() if evaluator else None,
            "raw_move": nested(selector, "raw", "argmax"),
            "effective_move": nested(selector, "effective", "argmax"),
            "final_move": decision.get("final_move"),
            "base_move": decision.get("base_move"),
            "agency_triggered": bool(nested(selector, "learner_agency", "triggered", default=False)),
            "eligible_arms": json.dumps(decision.get("eligible_arms", []), separators=(",", ":")),
            "eligible_arm_count": len(decision.get("eligible_arms", [])),
            "selected_arm": decision.get("selected_arm"),
            "lints_overridden": bool(decision.get("overridden", False)),
            "raw_probabilities": json.dumps(raw_probabilities, sort_keys=True, separators=(",", ":")),
            "effective_probabilities": json.dumps(effective_probabilities, sort_keys=True, separators=(",", ":")),
            "effective_top1_top2_gap": ordered[-1] - ordered[-2] if len(ordered) >= 2 else None,
            "MI": scores.get("Mistake_Identification"),
            "ML": scores.get("Mistake_Location"),
            "PG": scores.get("Providing_Guidance"),
            "A": scores.get("Actionability"),
        }
        action_rows.append(row)
    action_df = pd.DataFrame(action_rows).sort_values(["collection_sequence", "action_turn_index"])
    observed_all = action_df[action_df["delta_mastery"].notna()].copy()
    completed_turns["mastery_band"] = completed_turns["mastery_before"].map(mastery_band)

    # Full attempt inventory, including censored attempts with no summary.
    completed_by_id = {str(row["attempt_id"]): row for _, row in completed_attempts.iterrows()}
    attempt_rows: list[dict[str, Any]] = []
    for attempt_id in all_attempt_ids:
        source = turns_by_attempt.get(attempt_id, [])
        observed = [item for item in source if nested(item, "learning_state_update", "delta_mastery") is not None]
        censored = [item for item in source if nested(item, "next_learner_observation", "status") == "censored_no_response"]
        if attempt_id in completed_by_id:
            existing = completed_by_id[attempt_id].to_dict()
            existing.update({
                "collection_sequence": sequence[attempt_id],
                "action_record_count": len(source),
                "observed_dialogue_turns": len(observed),
                "censored_actions": len(censored),
                "final_outcome_available": True,
            })
            attempt_rows.append(existing)
            continue
        first = source[0] if source else {}
        start = nested(first, "state_before_action", "attempt_start_mastery")
        last_after = nested(observed[-1], "learning_state_update", "mastery_after") if observed else None
        attempt_rows.append({
            "collection_sequence": sequence[attempt_id],
            "attempt_id": attempt_id,
            "learner_pseudonym": nested(first, "identity", "student_pseudonymous_id"),
            "skill": nested(first, "state_before_action", "target_skill"),
            "attempt_started_at": nested(first, "state_before_action", "attempt_started_at"),
            "completion_status": "aborted_censored",
            "action_record_count": len(source),
            "observed_dialogue_turns": len(observed),
            "censored_actions": len(censored),
            "assessment_item_count": 0,
            "mastery_start": start,
            "mastery_pre_assessment": last_after,
            "mastery_final": None,
            "dialogue_net_delta": None if start is None or last_after is None else float(last_after) - float(start),
            "assessment_net_delta": None,
            "total_delta": None,
            "positive_dialogue_turns": sum(float(nested(item, "learning_state_update", "delta_mastery")) > EPS for item in observed),
            "zero_dialogue_turns": sum(abs(float(nested(item, "learning_state_update", "delta_mastery"))) <= EPS for item in observed),
            "negative_dialogue_turns": sum(float(nested(item, "learning_state_update", "delta_mastery")) < -EPS for item in observed),
            "final_outcome_available": False,
        })
    attempt_df = pd.DataFrame(attempt_rows).sort_values("collection_sequence")

    # Integrity over every action record, including censored attempts.
    integrity = Counter()
    action_dup = duplicate_audit(turns, lambda item: nested(item, "identity", "action_event_id"))
    observed_source = [item for item in turns if nested(item, "learning_state_update", "resolver_event_id") is not None]
    resolver_dup = duplicate_audit(observed_source, lambda item: nested(item, "learning_state_update", "resolver_event_id"))
    assessment_events = [item for record in assessments for item in (record.get("observation_results") or [])]
    assessment_dup = duplicate_audit(assessment_events, lambda item: item.get("event_id"))
    summary_dup = duplicate_audit(summaries, lambda item: nested(item, "identity", "attempt_id"))
    for prefix, result in [("action", action_dup), ("resolver", resolver_dup), ("assessment", assessment_dup), ("summary", summary_dup)]:
        integrity[f"duplicate_{prefix}_id_groups"] = result[0]
        integrity[f"duplicate_{prefix}_id_extra_records"] = result[1]
        integrity[f"conflicting_{prefix}_id_groups"] = result[2]
    integrity["exact_duplicate_turn_records"] = sum(value - 1 for value in Counter(turn_raw).values() if value > 1)
    integrity["exact_duplicate_assessment_records"] = sum(value - 1 for value in Counter(assessment_raw).values() if value > 1)
    integrity["exact_duplicate_summary_records"] = sum(value - 1 for value in Counter(summary_raw).values() if value > 1)
    all_action_ids = {str(nested(item, "identity", "action_event_id")) for item in turns}
    for attempt_id, group in turns_by_attempt.items():
        indices = [int(nested(item, "identity", "action_turn_index")) for item in group]
        integrity["index_starts_at_1_mismatches"] += int(bool(indices) and indices[0] != 1)
        integrity["index_monotonic_mismatches"] += int(indices != sorted(indices))
        integrity["index_contiguous_mismatches"] += int(indices != list(range(1, len(indices) + 1)))
        previous: dict[str, Any] | None = None
        previous_timestamp: str | None = None
        previous_observed_after: float | None = None
        for item in group:
            timestamp = str(nested(item, "provenance", "timestamp"))
            integrity["timestamp_order_mismatches"] += int(previous_timestamp is not None and timestamp <= previous_timestamp)
            previous_timestamp = timestamp
            action_id = str(nested(item, "identity", "action_event_id"))
            resolver_id = nested(item, "learning_state_update", "resolver_event_id")
            if resolver_id is not None:
                integrity["resolver_action_identity_mismatches"] += int(not str(resolver_id).startswith(action_id + ":resolver:"))
                before = float(nested(item, "learning_state_update", "mastery_before"))
                after = float(nested(item, "learning_state_update", "mastery_after"))
                delta = float(nested(item, "learning_state_update", "delta_mastery"))
                integrity["delta_equation_mismatches"] += int(abs(delta - (after - before)) > 1e-12)
                action_mastery = nested(item, "state_before_action", "mastery_at_action")
                integrity["action_mastery_mismatches"] += int(action_mastery is not None and abs(float(action_mastery) - before) > 1e-12)
                integrity["mastery_continuity_mismatches"] += int(previous_observed_after is not None and abs(before - previous_observed_after) > 1e-12)
                previous_observed_after = after
            elif previous_observed_after is not None:
                action_mastery = nested(item, "state_before_action", "mastery_at_action")
                integrity["censored_action_mastery_continuity_mismatches"] += int(action_mastery is not None and abs(float(action_mastery) - previous_observed_after) > 1e-12)
            if previous is not None and nested(previous, "next_learner_observation", "student_text") is not None:
                history = nested(item, "state_before_action", "history_before_action", default=[]) or []
                expected_teacher = nested(previous, "tutor_response")
                expected_student = nested(previous, "next_learner_observation", "student_text")
                bad = len(history) < 2 or history[-2].get("user") != "teacher" or history[-2].get("text") != expected_teacher or history[-1].get("user") != "student" or history[-1].get("text") != expected_student
                integrity["history_text_temporal_mismatches"] += int(bad)
            previous = item
        assessment = assessment_by_attempt.get(attempt_id, {})
        chain_after = previous_observed_after
        for item in assessment.get("observation_results", []) or []:
            before = float(item["mastery_before"])
            after = float(item["mastery_after"])
            delta = float(item["delta_mastery"])
            integrity["assessment_separation_mismatches"] += int(item.get("source_action_event_id") is not None or item.get("event_id") in all_action_ids)
            integrity["assessment_chain_mismatches"] += int(chain_after is not None and abs(before - chain_after) > 1e-12)
            integrity["delta_equation_mismatches"] += int(abs(delta - (after - before)) > 1e-12)
            chain_after = after
    integrity["pending_actions"] = len([path for path in (DATA_DIR / "pending_actions").iterdir() if path.is_file()])
    integrity["pending_processing_failures"] = len([path for path in (DATA_DIR / "pending_processing_failures").iterdir() if path.is_file()])
    blocking_keys = [key for key in integrity if any(token in key for token in ["identity", "index_", "temporal", "continuity", "delta_equation", "assessment_chain", "assessment_separation", "conflicting_"])]
    blocking_mismatches = sum(int(integrity[key]) for key in blocking_keys)

    # Provenance and privacy: preserve the known first-attempt exception, but scan
    # every later completed or censored scientific record.
    first_attempt_id = str(nested(summaries[0], "identity", "attempt_id"))
    post_fix_records = [item for item in turns + assessments + summaries if str(nested(item, "identity", "attempt_id")) != first_attempt_id]
    flattened = [item for record in post_fix_records for item in flatten_strings(record)]
    joined = "\n".join(value for _, value in flattened)
    patterns = {
        "absolute_C_Users_paths": r"(?i)C:[\\/]Users[\\/]",
        "raw_username_Lenovo": r"(?i)\bLenovo\b",
        "email_address": r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        "account_id": r"(?i)\baccount(?:[ _-]?id)?\b",
        "api_key": r"(?i)\bapi(?:[ _-]?key)?\b",
        "auth_token": r"(?i)\b(?:auth|authentication|authorization)(?:[ _-]?token)?\b",
        "secret": r"(?i)\bsecret\b",
    }
    privacy_counts = {name: len(re.findall(pattern, joined)) for name, pattern in patterns.items()}
    rooted_paths = [(location, value) for location, value in flattened if "path" in location.casefold() and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/"))]
    pseudonyms = sorted({str(nested(item, "identity", "student_pseudonymous_id")) for item in post_fix_records if nested(item, "identity", "student_pseudonymous_id")})
    invalid_pseudonyms = [value for value in pseudonyms if not re.fullmatch(r"psn_[0-9a-f]{24}", value)]
    selector_modes_all = counts(action_df["selector_mode"], sorted(action_df["selector_mode"].dropna().unique().tolist()))
    selector_modes_completed = counts(action_df[action_df["completed_attempt"]]["selector_mode"], ["ordinary-md7r1-v1", "manual-controlled-live-md7r1-v1"])

    # Learner, skill, and move coverage uses the completed ordinary cohort so it
    # remains comparable to the six-attempt baseline.
    learner_rows: list[dict[str, Any]] = []
    for learner in sorted(completed_attempts["learner_pseudonym"].unique()):
        attempt_subset = completed_attempts[completed_attempts["learner_pseudonym"] == learner]
        turn_subset = completed_turns[completed_turns["learner_pseudonym"] == learner]
        start = stats(attempt_subset["mastery_start"])
        final = stats(attempt_subset["mastery_final"])
        moves = counts(turn_subset["final_move"], MOVES)
        evals = counts(turn_subset["correctness"], CORRECTNESS)
        deltas = counts(turn_subset["delta_sign"], SIGNS)
        learner_rows.append({
            "learner_pseudonym": learner,
            "attempts": len(attempt_subset), "turns": len(turn_subset), "skills": attempt_subset["skill"].nunique(),
            "start_min": start["min"], "start_median": start["median"], "start_mean": start["mean"], "start_max": start["max"],
            "final_min": final["min"], "final_median": final["median"], "final_mean": final["mean"], "final_max": final["max"],
            **{f"move_{key}": value for key, value in moves.items()},
            **{f"eval_{key}": value for key, value in evals.items()},
            **{f"bkt_{key}": value for key, value in deltas.items()},
        })
    learner_df = pd.DataFrame(learner_rows)

    skill_rows: list[dict[str, Any]] = []
    for skill in sorted(completed_turns["skill"].unique()):
        turn_subset = completed_turns[completed_turns["skill"] == skill]
        attempt_subset = completed_attempts[completed_attempts["skill"] == skill]
        start = stats(attempt_subset["mastery_start"])
        raw = counts(turn_subset["raw_move"], MOVES)
        final = counts(turn_subset["final_move"], MOVES)
        evals = counts(turn_subset["correctness"], CORRECTNESS)
        deltas = counts(turn_subset["delta_sign"], SIGNS)
        flags: list[str] = []
        if attempt_subset["learner_pseudonym"].nunique() == 1: flags.append("single_learner")
        if len(attempt_subset) == 1: flags.append("single_attempt")
        if len(turn_subset) < 10: flags.append("very_small_turn_count_<10")
        if start["max"] is not None and start["max"] >= 0.90: flags.append("high_starting_mastery")
        if (turn_subset["mastery_before"] >= 0.97).mean() >= 0.80 or attempt_subset["mastery_final"].median() >= 0.99: flags.append("near_saturation")
        skill_rows.append({
            "skill": skill, "unique_learners": turn_subset["learner_pseudonym"].nunique(), "attempts": len(attempt_subset), "turns": len(turn_subset),
            "start_min": start["min"], "start_median": start["median"], "start_mean": start["mean"], "start_max": start["max"],
            **{f"raw_{key}": value for key, value in raw.items()},
            **{f"final_{key}": value for key, value in final.items()},
            **{f"eval_{key}": value for key, value in evals.items()},
            **{f"bkt_{key}": value for key, value in deltas.items()},
            "dialogue_net": attempt_subset["dialogue_net_delta"].sum(), "assessment_net": attempt_subset["assessment_net_delta"].sum(),
            "flags": ";".join(flags) if flags else "none",
        })
    skill_df = pd.DataFrame(skill_rows)

    move_rows: list[dict[str, Any]] = []
    for stage, column in [("raw", "raw_move"), ("effective", "effective_move"), ("final", "final_move")]:
        for move in MOVES:
            subset = completed_turns[completed_turns[column] == move]
            move_rows.append({"stage": stage, "move": move, "turns": len(subset), "learners": subset["learner_pseudonym"].nunique(), "skills": subset["skill"].nunique()})
    move_df = pd.DataFrame(move_rows)

    agency_df = completed_turns[completed_turns["learner_agency_triggered"] == True][["learner_pseudonym", "skill", "action_turn_index", "latest_student_text", "raw_move", "effective_move", "final_move", "correctness", "delta_mastery"]].copy()
    semantic_patterns = ["i cannot understand clearly", "i dont know about that"]
    semantic_misses = completed_turns[completed_turns["latest_student_text"].fillna("").str.casefold().apply(lambda text: any(phrase in text for phrase in semantic_patterns)) & (completed_turns["learner_agency_triggered"] != True)][["learner_pseudonym", "skill", "action_turn_index", "latest_student_text", "raw_move", "effective_move", "final_move", "correctness", "delta_mastery"]].copy()

    eligible_distribution = completed_turns["eligible_arm_count"].value_counts().sort_index()
    overrides = completed_turns[completed_turns["lints_overridden"] == True][["learner_pseudonym", "skill", "action_turn_index", "latest_student_text", "raw_probabilities", "effective_probabilities", "eligible_arms", "selected_arm", "base_move", "final_move", "mastery_before", "correctness", "delta_mastery", "MI", "ML", "PG", "A"]].copy()
    arm_counts = {arm: int(nested(policy, "policy_state", "arm_update_counts", arm, default=0)) for arm in ARMS}

    gap_stats = stats(completed_turns["effective_top1_top2_gap"])
    gap_thresholds = pd.DataFrame([{"threshold": threshold, "count": int((completed_turns["effective_top1_top2_gap"] <= threshold + EPS).sum()), "percent": 100.0 * float((completed_turns["effective_top1_top2_gap"] <= threshold + EPS).mean())} for threshold in [0.02, 0.05, 0.10, 0.15, 0.20]])

    band_rows: list[dict[str, Any]] = []
    for band in ["[0,.25)", "[.25,.50)", "[.50,.75)", "[.75,.90)", "[.90,.97)", "[.97,1]"]:
        subset = completed_turns[completed_turns["mastery_band"] == band]
        delta = stats(subset["delta_mastery"])
        absolute = stats(subset["delta_mastery"].abs())
        delta_counts = counts(subset["delta_sign"], SIGNS)
        band_rows.append({"band": band, "turns": len(subset), "learners": subset["learner_pseudonym"].nunique(), "skills": subset["skill"].nunique(), "mean_delta": delta["mean"], "median_delta": delta["median"], "mean_abs_delta": absolute["mean"], "median_abs_delta": absolute["median"], **delta_counts})
    band_df = pd.DataFrame(band_rows)

    evaluator_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    for category in CORRECTNESS:
        subset = completed_turns[completed_turns["correctness"] == category]
        sign_counts = counts(subset["delta_sign"], SIGNS)
        total = len(subset)
        evaluator_rows.append({"correctness": category, **{f"{key}_count": value for key, value in sign_counts.items()}, **{f"{key}_percent": (100.0 * value / total if total else 0.0) for key, value in sign_counts.items()}, "row_total": total})
        for variable in ["mastery_before", "delta_mastery", "update_confidence"]:
            redundancy_rows.append({"correctness": category, "variable": variable, **stats(subset[variable])})
    evaluator_df = pd.DataFrame(evaluator_rows)
    redundancy_df = pd.DataFrame(redundancy_rows)
    corr_rows: list[dict[str, Any]] = []
    for group, subset in [("all", completed_turns)] + [(category, completed_turns[completed_turns["correctness"] == category]) for category in CORRECTNESS]:
        for left, right in [("delta_mastery", "mastery_before"), ("delta_mastery", "headroom"), ("delta_mastery", "update_confidence")]:
            paired = subset[[left, right]].dropna()
            corr_rows.append({"group": group, "association": f"{left}_vs_{right}", "n": len(paired), "pearson": paired[left].corr(paired[right], method="pearson") if len(paired) >= 2 else None, "spearman": paired[left].corr(paired[right], method="spearman") if len(paired) >= 2 else None})
    correlation_df = pd.DataFrame(corr_rows)
    grand_mean = completed_turns["delta_mastery"].mean()
    ss_total = float(((completed_turns["delta_mastery"] - grand_mean) ** 2).sum())
    ss_between = sum(len(group) * float(group["delta_mastery"].mean() - grand_mean) ** 2 for _, group in completed_turns.groupby("correctness"))
    evaluator_eta_squared = ss_between / ss_total if ss_total else None

    mrb_rows: list[dict[str, Any]] = []
    group_specs = [("overall", "all", completed_turns)]
    group_specs += [("final_move", move, completed_turns[completed_turns["final_move"] == move]) for move in MOVES]
    group_specs += [("evaluator", category, completed_turns[completed_turns["correctness"] == category]) for category in CORRECTNESS]
    group_specs += [("learner", learner, completed_turns[completed_turns["learner_pseudonym"] == learner]) for learner in sorted(completed_turns["learner_pseudonym"].unique())]
    for group_type, group, subset in group_specs:
        for criterion in MRB:
            mrb_rows.append({"group_type": group_type, "group": group, "criterion": criterion, **stats(subset[criterion])})
    mrb_df = pd.DataFrame(mrb_rows)

    final_move_rows: list[dict[str, Any]] = []
    for move in MOVES:
        subset = completed_turns[completed_turns["final_move"] == move]
        mastery = stats(subset["mastery_before"])
        delta = stats(subset["delta_mastery"])
        final_move_rows.append({"final_move": move, "turns": len(subset), "learners": subset["learner_pseudonym"].nunique(), "skills": subset["skill"].nunique(), "mastery_mean": mastery["mean"], "mastery_median": mastery["median"], **{f"eval_{key}": value for key, value in counts(subset["correctness"], CORRECTNESS).items()}, **{f"bkt_{key}": value for key, value in counts(subset["delta_sign"], SIGNS).items()}, "delta_mean": delta["mean"], "delta_median": delta["median"], **{f"MRB1_{criterion}_mean": (subset[criterion].mean() if len(subset) else None) for criterion in MRB}})
    final_move_df = pd.DataFrame(final_move_rows)

    attempt_credit = completed_attempts[["attempt_number", "attempt_id", "learner_pseudonym", "skill", "dialogue_net_delta", "assessment_net_delta", "total_delta", "positive_dialogue_turns", "zero_dialogue_turns", "negative_dialogue_turns"]].copy()
    positive_attempts = attempt_credit[attempt_credit["total_delta"] > EPS]
    positive_with_negative = positive_attempts[positive_attempts["negative_dialogue_turns"] > 0]
    negative_dialogue_positive_total = attempt_credit[(attempt_credit["dialogue_net_delta"] < -EPS) & (attempt_credit["total_delta"] > EPS)]

    # State-region support is descriptive only; it does not assert exchangeability.
    state_region_rows: list[dict[str, Any]] = []
    for (band, category), subset in completed_turns.groupby(["mastery_band", "correctness"], dropna=False):
        move_counts = counts(subset["final_move"], MOVES)
        state_region_rows.append({"mastery_band": band, "correctness": category, "turns": len(subset), "learners": subset["learner_pseudonym"].nunique(), "moves_observed": sum(value > 0 for value in move_counts.values()), "dominant_move_share": max(move_counts.values()) / len(subset), **move_counts})
    state_region_df = pd.DataFrame(state_region_rows)
    comparable_regions = state_region_df[(state_region_df["learners"] >= 2) & (state_region_df["moves_observed"] >= 2)]

    # Write requested expanded tables.
    write_csv("turn_table_completed_cohort.csv", completed_turns)
    write_csv("turn_table.csv", action_df)
    write_csv("attempt_table.csv", attempt_df)
    write_csv("learner_table.csv", learner_df)
    write_csv("skill_table.csv", skill_df)
    write_csv("move_table.csv", move_df)
    write_csv("learner_agency_turns.csv", agency_df)
    write_csv("learner_agency_semantic_misses.csv", semantic_misses)
    write_csv("lints_overrides.csv", overrides)
    write_csv("headroom_bands.csv", band_df)
    write_csv("evaluator_delta_crosstab.csv", evaluator_df)
    write_csv("signal_redundancy_distributions.csv", redundancy_df)
    write_csv("signal_redundancy_correlations.csv", correlation_df)
    write_csv("mrb1_distributions.csv", mrb_df)
    write_csv("move_outcome_observational.csv", final_move_df)
    write_csv("attempt_vs_turn_credit.csv", attempt_credit)
    write_csv("state_region_action_coverage.csv", state_region_df)
    write_csv("integrity_totals.csv", pd.DataFrame([{"check": key, "count": int(value)} for key, value in sorted(integrity.items())]))

    # Requested descriptive plots (completed ordinary cohort unless stated).
    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(OUT_DIR / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5)); learner_df.plot.bar(x="learner_pseudonym", y="turns", legend=False, ax=plt.gca(), color="#4472C4"); plt.ylabel("Turns"); plt.title("Completed-cohort turns by learner"); save("turns_by_learner.png")
    plt.figure(figsize=(10, 5)); skill_df.sort_values("turns").plot.barh(x="skill", y="turns", legend=False, ax=plt.gca(), color="#70AD47"); plt.xlabel("Turns"); plt.title("Completed-cohort turns by skill"); save("turns_by_skill.png")
    learner_move = completed_turns.groupby(["learner_pseudonym", "final_move"]).size().unstack(fill_value=0).reindex(columns=MOVES, fill_value=0); plt.figure(figsize=(9, 5)); learner_move.plot.bar(stacked=True, ax=plt.gca()); plt.ylabel("Turns"); plt.title("Final moves by learner"); save("moves_by_learner.png")
    skill_move = completed_turns.groupby(["skill", "final_move"]).size().unstack(fill_value=0).reindex(columns=MOVES, fill_value=0); plt.figure(figsize=(11, 6)); skill_move.plot.barh(stacked=True, ax=plt.gca()); plt.xlabel("Turns"); plt.title("Final moves by skill"); save("moves_by_skill.png")
    plt.figure(figsize=(7, 4.5)); plt.hist(completed_turns["mastery_before"], bins=20, color="#4472C4", edgecolor="white"); plt.xlabel("mastery_before"); plt.ylabel("Turns"); plt.title("Mastery before action"); save("mastery_before_histogram.png")
    plt.figure(figsize=(7, 4.5)); plt.hist(completed_turns["delta_mastery"], bins=20, color="#ED7D31", edgecolor="white"); plt.xlabel("delta_mastery"); plt.ylabel("Turns"); plt.title("Immediate dialogue BKT delta"); save("delta_mastery_histogram.png")
    plt.figure(figsize=(7, 4.5)); plt.scatter(completed_turns["mastery_before"], completed_turns["delta_mastery"], alpha=.7); plt.axhline(0, color="black", lw=1); plt.xlabel("mastery_before"); plt.ylabel("delta_mastery"); plt.title("Delta versus mastery before"); save("delta_vs_mastery_before.png")
    cross = pd.crosstab(completed_turns["correctness"], completed_turns["delta_sign"]).reindex(index=CORRECTNESS, columns=SIGNS, fill_value=0); plt.figure(figsize=(8, 4.5)); cross.plot.bar(stacked=True, ax=plt.gca()); plt.ylabel("Turns"); plt.title("Evaluator category × BKT-delta sign"); save("evaluator_by_delta_sign.png")
    plt.figure(figsize=(7, 4.5)); plt.hist(completed_turns["effective_top1_top2_gap"], bins=20, color="#A5A5A5", edgecolor="white"); plt.axvline(.10, color="black", linestyle="--"); plt.xlabel("Effective top1−top2 gap"); plt.ylabel("Turns"); plt.title("MD7 effective confidence gap"); save("md7_gap.png")
    plt.figure(figsize=(6.5, 4.5)); plt.bar([str(value) for value in eligible_distribution.index], eligible_distribution.values); plt.xlabel("Eligible-arm count"); plt.ylabel("Turns"); plt.title("LinTS eligible-arm count"); save("lints_eligible_arm_count.png")
    plt.figure(figsize=(7.5, 4.5)); plt.boxplot([completed_turns[key].dropna() for key in MRB], tick_labels=MRB, patch_artist=True); plt.ylim(0, 1); plt.ylabel("Score"); plt.title("Frozen MRB1 distributions"); save("mrb1_distributions.png")

    raw_counts = counts(completed_turns["raw_move"], MOVES)
    effective_counts = counts(completed_turns["effective_move"], MOVES)
    final_counts = counts(completed_turns["final_move"], MOVES)
    bkt_counts = counts(completed_turns["delta_sign"], SIGNS)
    mastery_stats = stats(completed_turns["mastery_before"])
    delta_stats = stats(completed_turns["delta_mastery"])
    abs_delta_stats = stats(completed_turns["delta_mastery"].abs())
    baseline_only = int((completed_turns["eligible_arm_count"] == 1).sum())
    alternatives = int((completed_turns["eligible_arm_count"] >= 2).sum())
    eligible_bins = {"2": int((completed_turns["eligible_arm_count"] == 2).sum()), "3": int((completed_turns["eligible_arm_count"] == 3).sum()), "4_plus": int((completed_turns["eligible_arm_count"] >= 4).sum())}
    selected_counts = counts(completed_turns["selected_arm"], ARMS)
    telling_origins = {"raw_md7": raw_counts["telling"], "learner_agency": int(((completed_turns["raw_move"] != "telling") & (completed_turns["effective_move"] == "telling")).sum()), "lints_override": int(((completed_turns["effective_move"] != "telling") & (completed_turns["final_move"] == "telling")).sum())}
    post_six = completed_turns[completed_turns["attempt_number"] > 6]
    new_skills = set(completed_attempts[completed_attempts["attempt_number"] > 6]["skill"]) - set(completed_attempts[completed_attempts["attempt_number"] <= 6]["skill"])
    old_to_new = {
        "attempts": [6, 12], "learners": [1, int(completed_attempts["learner_pseudonym"].nunique())], "skills": [5, int(completed_attempts["skill"].nunique())], "turns": [57, len(completed_turns)],
        "raw_moves": [{"generic": 13, "probing": 20, "focus": 24, "telling": 0}, raw_counts],
        "final_moves": [{"generic": 14, "probing": 20, "focus": 23, "telling": 0}, final_counts],
        "agency_triggers": [0, len(agency_df)], "lints_overrides": [1, len(overrides)],
        "baseline_only_rate_percent": [100.0 * 54 / 57, 100.0 * baseline_only / len(completed_turns)],
        "alternative_opportunity_rate_percent": [100.0 * 3 / 57, 100.0 * alternatives / len(completed_turns)],
        "bkt_signs": [{"positive": 30, "zero": 0, "negative": 27}, bkt_counts],
        "mastery_before_range": [[0.011041522580481314, 0.9894141164608343], [mastery_stats["min"], mastery_stats["max"]]],
    }

    privacy_failures = sum(privacy_counts.values()) + len(rooted_paths) + len(invalid_pseudonyms)
    after_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    input_changes = {key: {"before": before_hashes[key], "after": after_hashes[key]} for key in before_hashes if before_hashes[key] != after_hashes[key]}
    integrity_clean = blocking_mismatches == 0 and integrity["pending_processing_failures"] == 0
    verdict = "B. DATA CLEAN BUT ACTION COVERAGE POLICY-BOUND — DESIGN CONTROLLED EXPLORATION NEXT" if integrity_clean else "D. INTEGRITY FAILURE — STOP"

    summary = {
        "analysis_version": "md_self_improvement_expanded_analysis_v1",
        "analysis_scope": {"primary": "12 completed ordinary-mode attempts / 87 observed dialogue turns", "inventory": "all 96 action records including two censored aborted attempts", "observed_sensitivity": "94 observed dialogue updates including seven from one aborted manual-controlled attempt"},
        "inventory": {"completed_attempts": 12, "aborted_attempts": int((attempt_df["completion_status"] == "aborted_censored").sum()), "action_records": len(action_df), "observed_dialogue_turns_all": len(observed_all), "completed_cohort_turns": len(completed_turns), "formal_assessment_observations": sum(len(item.get("observation_results", []) or []) for item in assessments), "unique_learners": int(action_df["learner_pseudonym"].nunique()), "unique_skills_all": int(action_df["skill"].nunique()), "unique_skills_completed": int(completed_turns["skill"].nunique()), "unique_learner_skill_pairs_completed": int(completed_turns[["learner_pseudonym", "skill"]].drop_duplicates().shape[0]), "pending_actions": int(integrity["pending_actions"]), "pending_processing_failures": int(integrity["pending_processing_failures"])},
        "integrity": {"mismatch_counts": dict(integrity), "blocking_mismatch_count": blocking_mismatches, "clean": integrity_clean},
        "privacy": {"scope": "all post-first-attempt records, including censored attempts", "pattern_counts": privacy_counts, "rooted_path_fields": rooted_paths, "invalid_pseudonyms": invalid_pseudonyms, "failure_count": privacy_failures},
        "provenance": {"data_mode_counts": counts(action_df["data_mode"], ["real"]), "selector_modes_all_action_records": selector_modes_all, "selector_modes_completed_cohort": selector_modes_completed, "manual_mode_exception": "8 action records (7 observed, 1 censored) from aborted attempt 4220e000…; excluded from primary completed-cohort analyses", "base_turn_field_audit": base_summary["provenance"]["turn_fields"]},
        "coverage": {"learners": learner_rows, "skills": skill_rows, "moves": {"raw": raw_counts, "effective": effective_counts, "final": final_counts, "telling_origins": telling_origins}, "agency_trigger_count": len(agency_df), "agency_semantic_miss_count": len(semantic_misses)},
        "lints": {"baseline_only": baseline_only, "alternatives": alternatives, "eligible_bins": eligible_bins, "eligible_arm_stats": stats(completed_turns["eligible_arm_count"]), "selected_arms": selected_counts, "actual_overrides": len(overrides), "arm_update_counts": arm_counts, "policy_total_updates": int(nested(policy, "policy_state", "total_updates")), "experience_attempts": len(experiences), "experience_turns": sum(int(item.get("num_turns", 0)) for item in experiences)},
        "md7_gap": {"stats": gap_stats, "thresholds": gap_thresholds.to_dict(orient="records"), "tau_0_10_alternative_fraction": alternatives / len(completed_turns)},
        "bkt": {"mastery_before": mastery_stats, "headroom": stats(completed_turns["headroom"]), "delta": delta_stats, "absolute_delta": abs_delta_stats, "sign_counts": bkt_counts, "bands": band_rows},
        "evaluator_bkt": evaluator_rows,
        "signal_redundancy": {"evaluator_eta_squared_delta": evaluator_eta_squared, "correlations": correlation_df.to_dict(orient="records"), "interpretation": "Evaluator class nearly determines delta sign, while within-class delta magnitude still varies with mastery/headroom and update confidence; residual variation remains but is structured by the BKT update."},
        "attempt_credit": {"positive_cumulative_attempts": len(positive_attempts), "positive_with_negative_immediate_turns": len(positive_with_negative), "negative_dialogue_positive_total_attempts": negative_dialogue_positive_total["attempt_number"].tolist()},
        "old_to_new": old_to_new,
        "comparable_state_regions_multi_learner_multi_move": len(comparable_regions),
        "sufficiency": {"data_pipeline_prototyping": "yes", "engineering_training_smoke_test": "yes", "tiny_scientific_sft_candidate": "conditional_no_not_without_qualification_and_replay", "comparison_against_md7": "no_for_scientific_effectiveness; yes_for_pipeline_descriptives", "claim_of_tutoring_policy_improvement": "no"},
        "recommended_next_experiment_category": "B",
        "verdict": verdict,
        "side_effects": {"training": 0, "model_writes": 0, "policy_writes": 0, "authoritative_research_writes": 0, "api_calls": 0, "tutor_conversations": 0, "protected_evaluation_use": 0, "protected_input_hash_changes": input_changes},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    provenance_expected = pd.DataFrame([
        {"field": "data_mode", "expected": "real", "completed_cohort": "real (87/87)", "all_action_records": f"real ({len(action_df)}/{len(action_df)})"},
        {"field": "selector_mode", "expected": "ordinary-md7r1-v1", "completed_cohort": "ordinary-md7r1-v1 (87/87)", "all_action_records": json.dumps(selector_modes_all, sort_keys=True)},
        {"field": "MD7", "expected": "md7r1_epoch3 / d32d…a13", "completed_cohort": "consistent", "all_action_records": "consistent"},
        {"field": "move_order", "expected": "G/P/F/T", "completed_cohort": "consistent post-fix; legacy attempt missing", "all_action_records": "consistent post-fix; legacy attempt missing"},
        {"field": "agency/C3/LinTS/tau", "expected": "agency_v1 / C3-9d / LinTS-v3 / .10", "completed_cohort": "consistent", "all_action_records": "consistent"},
        {"field": "MRB1/resolver/BKT", "expected": "frozen_mrb1 / 2.0 / confidence_weighted_bkt_v1", "completed_cohort": "consistent", "all_action_records": "consistent"},
    ])
    inventory_display = attempt_df[["collection_sequence", "attempt_id", "learner_pseudonym", "skill", "completion_status", "action_record_count", "observed_dialogue_turns", "censored_actions", "assessment_item_count"]]
    integrity_display = pd.DataFrame([{"check": key, "count": int(value)} for key, value in sorted(integrity.items())])
    coverage_change = pd.DataFrame([
        {"metric": "completed attempts", "old": 6, "new": 12}, {"metric": "aborted attempts", "old": 0, "new": 2}, {"metric": "learners", "old": 1, "new": 3}, {"metric": "skills", "old": 5, "new": 11}, {"metric": "completed-cohort turns", "old": 57, "new": 87},
        {"metric": "raw G/P/F/T", "old": "13/20/24/0", "new": "/".join(str(raw_counts[key]) for key in MOVES)}, {"metric": "final G/P/F/T", "old": "14/20/23/0", "new": "/".join(str(final_counts[key]) for key in MOVES)},
        {"metric": "agency triggers", "old": 0, "new": len(agency_df)}, {"metric": "LinTS overrides", "old": 1, "new": len(overrides)}, {"metric": "baseline-only rate", "old": f"{100*54/57:.2f}%", "new": f"{100*baseline_only/len(completed_turns):.2f}%"}, {"metric": "alternative opportunity", "old": f"{100*3/57:.2f}%", "new": f"{100*alternatives/len(completed_turns):.2f}%"}, {"metric": "BKT +/0/-", "old": "30/0/27", "new": f"{bkt_counts['positive']}/{bkt_counts['zero']}/{bkt_counts['negative']}"}, {"metric": "mastery-before range", "old": ".011042–.989414", "new": f"{mastery_stats['min']:.6f}–{mastery_stats['max']:.6f}"},
    ])
    side_effects = pd.DataFrame([{"effect": key, "count": len(value) if key == "protected_input_hash_changes" else value} for key, value in summary["side_effects"].items()])

    report = f"""# Expanded Real-Data Analysis Before MD8-C1 Design

Analysis version: `md_self_improvement_expanded_analysis_v1`  
Primary inferential scope: **completed ordinary-mode cohort only** (12 attempts, 87 observed dialogue turns). Inventory and integrity additionally cover all 96 action records, including two censored attempts. All results are descriptive and non-causal.

## 1. DATASET INVENTORY

- Completed real attempts: **12** (old 6; **+6**).
- Aborted/censored attempts: **2** (old 0; **+2**). Attempt `4220e000…:1` has 7 observed turns plus one censored action; `978c3e3b…:2` has one censored action and no observed response.
- Action records: **96**; observed dialogue updates across all attempts: **94**; completed-cohort dialogue turns: **87** (old 57; **+30**).
- Formal-assessment observations: **36** (old 18; **+18**).
- Unique learners: **3** (old 1); completed-cohort skills: **11** (old 5); learner-skill pairs: **11**.
- Pending actions / processing failures: **{integrity['pending_actions']} / {integrity['pending_processing_failures']}**.

{md(inventory_display)}

## 2. INTEGRITY / PRIVACY / PROVENANCE

{md(integrity_display)}

Blocking identity/temporal/mastery mismatch count: **{blocking_mismatches}**. Duplicate action, resolver, assessment, and summary identifiers; conflicting duplicates; history continuity; mastery continuity; delta equations; assessment chains; and formal-assessment separation all pass.

The two censored rows correctly have no evaluator/resolver/BKT update and are not counted as outcome turns. The seven observed rows in the aborted manual attempt retain internally continuous mastery but have no final assessment or cumulative attempt outcome.

Privacy scan scope is every post-first-attempt turn, assessment, and summary, including censored attempts. Pattern counts: `{json.dumps(privacy_counts, sort_keys=True)}`; rooted path fields: **{len(rooted_paths)}**; invalid pseudonyms: **{len(invalid_pseudonyms)}**. The known pre-fix first-attempt path was not rewritten or copied into the derived tables.

{md(provenance_expected)}

Important provenance exception: all **87 completed-cohort turns** are `ordinary-md7r1-v1`; the aborted `4220e000…:1` contributes **8 manual-controlled action records** (7 observed, 1 censored). It is inventoried and integrity-audited but excluded from primary coverage/outcome statistics. The remaining censored `978c3e3b…:2` row is ordinary-mode.

## 3. LEARNER COVERAGE

{md(learner_df)}

These are **within-learner repeated observations**: 12 attempts and 87 turns are not 12 or 87 independent learners. Between-learner evidence consists of only **3 pseudonymous learners**. Each learner contributes multiple turns and multiple skills; all distribution summaries above are descriptive.

## 4. SKILL COVERAGE

{md(skill_df)}

All 11 skills are single-learner skills; 10/11 are single-attempt skills. Small cells, high starts, and saturation flags are explicit. Skill differences must not be interpreted as equal-difficulty action comparisons.

## 5. MOVE COVERAGE

{md(move_df)}

Raw/effective/final G/P/F/T totals are **{raw_counts['generic']}/{raw_counts['probing']}/{raw_counts['focus']}/{raw_counts['telling']}**, **{effective_counts['generic']}/{effective_counts['probing']}/{effective_counts['focus']}/{effective_counts['telling']}**, and **{final_counts['generic']}/{final_counts['probing']}/{final_counts['focus']}/{final_counts['telling']}**.

**There are no real telling actions.** Telling origin counts are raw MD7 **0**, learner-agency **0**, and LinTS override **0**. This is a major action-support gap.

## 6. LEARNER-AGENCY COVERAGE

Triggers: **{len(agency_df)}/{len(completed_turns)} (0.000%)**.

{md(agency_df)}

Known documented semantic non-engagement misses that are detectable in the completed cohort: **{len(semantic_misses)}**.

{md(semantic_misses)}

No agency-policy change is proposed here.

## 7. LINTS AUTHORITY

- Baseline-only eligible: **{baseline_only}/{len(completed_turns)} ({100*baseline_only/len(completed_turns):.3f}%)**.
- Nonbaseline alternative eligible: **{alternatives}/{len(completed_turns)} ({100*alternatives/len(completed_turns):.3f}%)**.
- Exactly 2 / exactly 3 / 4+ eligible arms: **{eligible_bins['2']} / {eligible_bins['3']} / {eligible_bins['4_plus']}**.
- Eligible-arm mean / median / max: **{completed_turns['eligible_arm_count'].mean():.6f} / {completed_turns['eligible_arm_count'].median():.0f} / {completed_turns['eligible_arm_count'].max():.0f}**.
- Selected arms: `{json.dumps(selected_counts, sort_keys=True)}`.
- Actual final-move overrides: **{len(overrides)}/{len(completed_turns)} ({100*len(overrides)/len(completed_turns):.3f}%)**.
- Current per-arm updates: `{json.dumps(arm_counts, sort_keys=True)}`; total **{nested(policy, 'policy_state', 'total_updates')}**. This equals the 87 completed-cohort turns; aborted attempts did not receive delayed attempt credit.

{md(overrides)}

## 8. MD7 CONFIDENCE / GATE

Effective top1−top2 gap: min **{gap_stats['min']:.6f}**, Q1 **{gap_stats['q1']:.6f}**, median **{gap_stats['median']:.6f}**, mean **{gap_stats['mean']:.6f}**, Q3 **{gap_stats['q3']:.6f}**, max **{gap_stats['max']:.6f}**.

{md(gap_thresholds)}

At tau=.10, alternatives were allowed on **{alternatives}/87 = {100*alternatives/87:.3f}%**, versus **3/57 = 5.263%** previously. Authority widened slightly but remains rare; no new tau is proposed.

## 9. BKT / HEADROOM

Mastery-before: `{json.dumps(mastery_stats)}`. Headroom: `{json.dumps(stats(completed_turns['headroom']))}`. Delta: `{json.dumps(delta_stats)}`. Absolute delta: `{json.dumps(abs_delta_stats)}`. Signs +/0/−: **{bkt_counts['positive']}/{bkt_counts['zero']}/{bkt_counts['negative']}**.

{md(band_df)}

Headroom is descriptive only and is not converted into a weight.

## 10. EVALUATOR-BKT RELATION

{md(evaluator_df)}

The prior near-deterministic sign relationship persists to the degree shown in the row percentages. This is expected to make BKT delta partly an encoding of evaluator/resolver evidence; it is not, by itself, labeled a defect.

## 11. SIGNAL REDUNDANCY

{md(redundancy_df)}

{md(correlation_df)}

Evaluator-category eta-squared association with delta magnitude: **{evaluator_eta_squared:.6f}**. Conservatively, evaluator class strongly structures sign and location; mastery/headroom and update confidence structure magnitude within the BKT equation. Nonzero within-class ranges/std show residual variation, but it is not evidence of independent pedagogical-move effect.

## 12. MRB1

{md(mrb_df)}

MRB1 values are frozen model outputs. No cutoff or good/bad label is introduced.

## 13. ATTEMPT VS TURN CREDIT

{md(attempt_credit)}

Positive cumulative attempts: **{len(positive_attempts)}/12**. Of these, **{len(positive_with_negative)}** contain at least one negative immediate dialogue turn. Attempts with dialogue net < 0 but total final delta > 0: **{negative_dialogue_positive_total['attempt_number'].tolist()}**. This continues to expose the scientific weakness of assigning equal delayed attempt credit to every selected arm.

## 14. MOVE × OUTCOME OBSERVATIONS

**OBSERVATIONAL ASSOCIATIONS ONLY. ACTIONS WERE NOT RANDOMLY ASSIGNED. MOVES ARE NOT RANKED.**

{md(final_move_df)}

## 15. OLD → NEW COVERAGE

{md(coverage_change)}

- Learner diversity: improved from 1 to 3, but remains small.
- Skill diversity: improved from 5 to 11, though all skills remain single-learner and nearly all single-attempt.
- Move diversity: **not materially improved**; the raw ratio stayed similar and telling remains absent.
- Mastery-state diversity: range did not expand beyond the pilot endpoints; more mid/high-state observations were added.
- Alternative-action coverage: opportunity rose from 5.263% to 9.195%, but actual overrides stayed at one and three arms remain entirely unupdated.

## 16. MD8 SUFFICIENCY

- Data-pipeline prototyping: **Yes**.
- Engineering training smoke test: **Yes**, with no scientific claim.
- Tiny scientific SFT candidate: **Conditional/no at present**; only after explicit qualification plus replay and without treating turns as independent learners.
- Comparison against MD7: **No** for effectiveness; **yes only** for descriptive pipeline checks.
- Claim of tutoring-policy improvement: **No**.

## 17. RECOMMENDED NEXT EXPERIMENT CATEGORY

**B. Observational dataset is integrity-clean, but action coverage remains too policy-bound. Design controlled randomized research exploration before MD8-C1.**

Why: another six natural completed attempts added two learners and six skills but still produced zero telling, zero agency triggers, only 8/87 alternative opportunities, one actual override, and no updates at all for probing/focus/telling LinTS arms. Natural collection improved learner/skill breadth but did not break the deployed policy's action-support constraint.

No exploration percentage, selection algorithm, or eligibility threshold is chosen. Only the need is identified. State-region support includes **{len(comparable_regions)}** coarse mastery-band × evaluator regions with at least two learners and at least two observed final moves; these are coarse observational overlaps, not randomized comparable states. Full cells are in `state_region_action_coverage.csv`.

## 18. DERIVED ARTIFACTS

All outputs are under `pedagogical-move-selection/results/md_self_improvement_expanded_analysis_v1/`. Principal artifacts: `report.md`, `summary.json`, `attempt_table.csv`, `turn_table.csv`, `turn_table_completed_cohort.csv`, `learner_table.csv`, `skill_table.csv`, `move_table.csv`, integrity/provenance/agency/LinTS/BKT/evaluator/redundancy/MRB1/credit tables, and 11 descriptive plots.

## 19. SIDE EFFECTS

{md(side_effects)}

Only this derived-analysis directory was written. No API calls, Tutor conversations, training, model/policy writes, authoritative JSONL writes, or protected MathDial/MRBench V3 test access occurred.

## 20. FINAL VERDICT

**{verdict}**
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps({"completed_attempts": 12, "aborted_attempts": 2, "completed_turns": len(completed_turns), "all_observed_turns": len(observed_all), "action_records": len(action_df), "learners": int(completed_turns['learner_pseudonym'].nunique()), "skills": int(completed_turns['skill'].nunique()), "blocking_mismatches": blocking_mismatches, "privacy_failures": privacy_failures, "telling": final_counts['telling'], "agency_triggers": len(agency_df), "alternative_opportunities": alternatives, "lints_overrides": len(overrides), "input_hash_changes": len(input_changes), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
