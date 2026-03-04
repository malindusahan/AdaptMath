from __future__ import annotations

import csv
import hashlib
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
POLICY_DIR = ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "adaptive_demo_md7r1_real_v1"

TURN_PATH = DATA_DIR / "turn_outcomes.jsonl"
ASSESSMENT_PATH = DATA_DIR / "assessment_outcomes.jsonl"
SUMMARY_PATH = DATA_DIR / "attempt_summaries.jsonl"
POLICY_PATH = POLICY_DIR / "policy_state.json"
EXPERIENCE_PATH = POLICY_DIR / "attempts.jsonl"
LINEAGE_PATH = POLICY_DIR / "lineage_manifest.json"

MD7_MODEL_PATH = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3" / "model.safetensors"
MRB1_MODEL_PATH = ROOT / "pedagogical-move-selection" / "models" / "frozen" / "mrb1" / "model.safetensors"
BKT_CONFIG_PATH = ROOT / "student-modeling" / "models" / "bkt_params.json"

MOVES = ["generic", "probing", "focus", "telling"]
ARMS = ["baseline", "generic_bias", "probing_bias", "focus_bias", "telling_bias"]
CORRECTNESS = ["correct", "partial", "incorrect", "unknown", "<missing>"]
MRB_KEYS = [
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
]
MRB_SHORT = {
    "Mistake_Identification": "MI",
    "Mistake_Location": "ML",
    "Providing_Guidance": "PG",
    "Actionability": "A",
}
EPS = 1e-15


def read_jsonl(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return raw, [json.loads(line) for line in raw]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite(values: Iterable[Any]) -> np.ndarray:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def stats(values: Iterable[Any]) -> dict[str, Any]:
    array = finite(values)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "mean": None,
            "q3": None,
            "max": None,
            "std_population": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "max": float(np.max(array)),
        "std_population": float(np.std(array, ddof=0)),
    }


def delta_sign(value: Any) -> str:
    number = float(value)
    if number > EPS:
        return "positive"
    if number < -EPS:
        return "negative"
    return "zero"


def normalized_correctness(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "<missing>"
    return str(value).strip().casefold()


def safe_text(value: Any, limit: int = 100) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def problem_id(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return "problem_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def count_map(values: Iterable[Any], order: list[str]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: int(counts.get(key, 0)) for key in order}


def compact_counts(counts: dict[str, int], order: list[str]) -> str:
    return "/".join(str(counts.get(key, 0)) for key in order)


def bool_label(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "<missing>"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown_table(frame: pd.DataFrame, float_digits: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: (
                ""
                if value is None or (isinstance(value, float) and math.isnan(value))
                else f"{value:.{float_digits}f}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        )
        display[column] = display[column].str.replace("|", "\\|", regex=False).str.replace("\n", " ", regex=False)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_csv(name: str, rows: list[dict[str, Any]] | pd.DataFrame) -> Path:
    path = OUT_DIR / name
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return path


def flatten_strings(value: Any, location: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}" if location else str(key)
            found.extend(flatten_strings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(flatten_strings(item, f"{location}[{index}]"))
    elif isinstance(value, str):
        found.append((location, value))
    return found


def duplicate_audit(records: list[dict[str, Any]], key_fn) -> tuple[int, int, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key is not None:
            grouped[str(key)].append(record)
    duplicate_groups = [group for group in grouped.values() if len(group) > 1]
    duplicate_extras = sum(len(group) - 1 for group in duplicate_groups)
    conflicting_groups = sum(
        1 for group in duplicate_groups if len({canonical(item) for item in group}) > 1
    )
    return len(duplicate_groups), duplicate_extras, conflicting_groups


def distribution_frame(values: Iterable[str], categories: list[str], label: str) -> pd.DataFrame:
    counts = count_map(values, categories)
    total = sum(counts.values())
    return pd.DataFrame(
        [
            {
                label: category,
                "count": counts[category],
                "percent": 100.0 * counts[category] / total if total else 0.0,
            }
            for category in categories
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protected_inputs = [
        TURN_PATH,
        ASSESSMENT_PATH,
        SUMMARY_PATH,
        POLICY_PATH,
        EXPERIENCE_PATH,
        LINEAGE_PATH,
        MD7_MODEL_PATH,
        MRB1_MODEL_PATH,
        BKT_CONFIG_PATH,
    ]
    initial_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected_inputs}

    turn_raw, turns = read_jsonl(TURN_PATH)
    assessment_raw, assessments = read_jsonl(ASSESSMENT_PATH)
    summary_raw, summaries = read_jsonl(SUMMARY_PATH)
    _, experiences = read_jsonl(EXPERIENCE_PATH)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))

    summaries_by_id = {nested(item, "identity", "attempt_id"): item for item in summaries}
    assessments_by_id = {nested(item, "identity", "attempt_id"): item for item in assessments}
    attempt_order = [nested(item, "identity", "attempt_id") for item in summaries]
    turns_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        turns_by_attempt[str(nested(turn, "identity", "attempt_id"))].append(turn)
    for attempt_id in turns_by_attempt:
        turns_by_attempt[attempt_id].sort(key=lambda item: int(nested(item, "identity", "action_turn_index", default=0)))

    turn_rows: list[dict[str, Any]] = []
    for attempt_number, attempt_id in enumerate(attempt_order, start=1):
        for turn in turns_by_attempt[attempt_id]:
            identity = turn["identity"]
            state = turn["state_before_action"]
            selector = turn["selector"]
            decision = turn["adaptive_decision"]
            update = turn["learning_state_update"]
            observation = turn["next_learner_observation"]
            evaluator = observation.get("evaluator") or {}
            resolver = observation.get("resolver") or {}
            raw_probabilities = nested(selector, "raw", "probabilities", default={}) or {}
            effective_probabilities = nested(selector, "effective", "probabilities", default={}) or {}
            ordered_effective = sorted((float(value), str(key)) for key, value in effective_probabilities.items())
            top1_probability = ordered_effective[-1][0] if ordered_effective else None
            top2_probability = ordered_effective[-2][0] if len(ordered_effective) >= 2 else None
            top_gap = (
                top1_probability - top2_probability
                if top1_probability is not None and top2_probability is not None
                else None
            )
            delta = float(update["delta_mastery"])
            scores = nested(turn, "tutor_quality", "scores", default={}) or {}
            eligible = [str(item) for item in decision.get("eligible_arms", [])]
            row: dict[str, Any] = {
                "attempt_number": attempt_number,
                "attempt_id": attempt_id,
                "learner_pseudonym": identity.get("student_pseudonymous_id"),
                "skill": state.get("target_skill"),
                "problem_id": problem_id(str(state.get("problem", ""))),
                "action_turn_index": int(identity.get("action_turn_index")),
                "action_event_id": identity.get("action_event_id"),
                "resolver_event_id": update.get("resolver_event_id"),
                "status": observation.get("status"),
                "attempt_started_at": state.get("attempt_started_at"),
                "outcome_timestamp": nested(turn, "provenance", "timestamp"),
                "latest_student_text": state.get("latest_student_text"),
                "mastery_before": float(update["mastery_before"]),
                "headroom": 1.0 - float(update["mastery_before"]),
                "mastery_after": float(update["mastery_after"]),
                "delta_mastery": delta,
                "delta_sign": delta_sign(delta),
                "should_update": update.get("should_update"),
                "outcome": update.get("outcome"),
                "update_confidence": update.get("update_confidence"),
                "correctness": normalized_correctness(evaluator.get("correctness")),
                "primary_signal": resolver.get("primary_signal", "<missing>"),
                "observation_source": resolver.get(
                    "observation_source", update.get("observation_source", "<missing>")
                ),
                "reasoning_present": bool_label(resolver.get("reasoning_present")),
                "uncertainty_present": bool_label(resolver.get("uncertainty_present")),
                "clarification_present": bool_label(resolver.get("clarification_present")),
                "repeated_misunderstanding": bool_label(resolver.get("repeated_misunderstanding")),
                "raw_move": nested(selector, "raw", "argmax", default="<missing>"),
                "effective_move": nested(selector, "effective", "argmax", default="<missing>"),
                "final_move": decision.get("final_move", "<missing>"),
                "base_move": decision.get("base_move", "<missing>"),
                "learner_agency_triggered": bool(nested(selector, "learner_agency", "triggered", default=False)),
                "learner_agency_reason": nested(selector, "learner_agency", "reason"),
                "eligible_arms": json.dumps(eligible, separators=(",", ":")),
                "eligible_arm_count": len(eligible),
                "selected_arm": decision.get("selected_arm", "<missing>"),
                "lints_overridden": bool(decision.get("overridden", False)),
                "raw_probabilities": json.dumps(raw_probabilities, sort_keys=True, separators=(",", ":")),
                "effective_probabilities": json.dumps(
                    effective_probabilities, sort_keys=True, separators=(",", ":")
                ),
                "effective_top1_probability": top1_probability,
                "effective_top2_probability": top2_probability,
                "effective_top1_top2_gap": top_gap,
                "MI": scores.get("Mistake_Identification"),
                "ML": scores.get("Mistake_Location"),
                "PG": scores.get("Providing_Guidance"),
                "A": scores.get("Actionability"),
            }
            agency_changed = row["raw_move"] != row["effective_move"]
            lints_changed = row["effective_move"] != row["final_move"]
            if not agency_changed and not lints_changed:
                row["decision_group"] = "A_unchanged"
            elif agency_changed and not lints_changed:
                row["decision_group"] = "B_agency_only"
            elif not agency_changed and lints_changed:
                row["decision_group"] = "C_lints_only"
            else:
                row["decision_group"] = "D_both"
            turn_rows.append(row)

    turn_df = pd.DataFrame(turn_rows)

    attempt_rows: list[dict[str, Any]] = []
    assessment_rows: list[dict[str, Any]] = []
    for attempt_number, attempt_id in enumerate(attempt_order, start=1):
        summary = summaries_by_id[attempt_id]
        assessment = assessments_by_id.get(attempt_id, {})
        attempt_turns = turn_df[turn_df["attempt_id"] == attempt_id].sort_values("action_turn_index")
        source_turns = turns_by_attempt[attempt_id]
        start_mastery = summary.get("attempt_start_mastery")
        if start_mastery is None and source_turns:
            start_mastery = nested(source_turns[0], "state_before_action", "attempt_start_mastery")
        if start_mastery is None:
            start_mastery = nested(summary, "adaptive_completion", "mastery_before")
        pre_assessment = float(attempt_turns.iloc[-1]["mastery_after"]) if not attempt_turns.empty else None
        observation_results = assessment.get("observation_results", []) or []
        final_mastery = (
            float(observation_results[-1]["mastery_after"])
            if observation_results
            else nested(summary, "cumulative_attempt_outcome", "mastery_after")
        )
        dialogue_net = pre_assessment - float(start_mastery)
        assessment_net = float(final_mastery) - pre_assessment
        total_delta = float(final_mastery) - float(start_mastery)
        raw_counts = count_map(attempt_turns["raw_move"], MOVES)
        final_counts = count_map(attempt_turns["final_move"], MOVES)
        evaluator_counts = count_map(attempt_turns["correctness"], CORRECTNESS)
        recorded_status = summary.get("completion_status")
        inferred_completed = bool(
            recorded_status == "completed"
            or (
                recorded_status is None
                and summary.get("adaptive_completion")
                and assessment
                and observation_results
            )
        )
        completion_status = (
            recorded_status
            if recorded_status is not None
            else "completed_legacy_inferred"
            if inferred_completed
            else "<missing>"
        )
        problem = str(nested(source_turns[0], "state_before_action", "problem", default=""))
        assessment_path_share = (
            abs(assessment_net) / (abs(dialogue_net) + abs(assessment_net))
            if abs(dialogue_net) + abs(assessment_net) > EPS
            else None
        )
        row = {
            "attempt_number": attempt_number,
            "attempt_id": attempt_id,
            "learner_pseudonym": nested(summary, "identity", "student_pseudonymous_id"),
            "skill": attempt_turns.iloc[0]["skill"] if not attempt_turns.empty else None,
            "problem_id": problem_id(problem),
            "problem_summary": safe_text(problem, 140),
            "attempt_started_at": summary.get("attempt_started_at"),
            "first_outcome_at": attempt_turns.iloc[0]["outcome_timestamp"] if not attempt_turns.empty else None,
            "completion_at": nested(summary, "provenance", "timestamp"),
            "completion_status": completion_status,
            "dialogue_turn_count": int(len(attempt_turns)),
            "assessment_item_count": int(len(assessment.get("evaluated_answers", []) or [])),
            "mastery_start": float(start_mastery),
            "mastery_pre_assessment": pre_assessment,
            "mastery_final": float(final_mastery),
            "dialogue_net_delta": dialogue_net,
            "assessment_net_delta": assessment_net,
            "total_delta": total_delta,
            "raw_G_P_F_T": compact_counts(raw_counts, MOVES),
            "final_G_P_F_T": compact_counts(final_counts, MOVES),
            "learner_agency_triggers": int(attempt_turns["learner_agency_triggered"].sum()),
            "lints_nonbaseline_selections": int((attempt_turns["selected_arm"] != "baseline").sum()),
            "actual_lints_overrides": int(attempt_turns["lints_overridden"].sum()),
            "evaluator_correct": evaluator_counts["correct"],
            "evaluator_partial": evaluator_counts["partial"],
            "evaluator_incorrect": evaluator_counts["incorrect"],
            "evaluator_unknown": evaluator_counts["unknown"],
            "mean_MRB1_MI": float(attempt_turns["MI"].mean()),
            "mean_MRB1_ML": float(attempt_turns["ML"].mean()),
            "mean_MRB1_PG": float(attempt_turns["PG"].mean()),
            "mean_MRB1_A": float(attempt_turns["A"].mean()),
            "negative_dialogue_turns": int((attempt_turns["delta_sign"] == "negative").sum()),
            "positive_dialogue_turns": int((attempt_turns["delta_sign"] == "positive").sum()),
            "zero_dialogue_turns": int((attempt_turns["delta_sign"] == "zero").sum()),
            "assessment_absolute_path_share": assessment_path_share,
            "dialogue_assessment_directions_differ": (
                delta_sign(dialogue_net) != delta_sign(assessment_net)
                and delta_sign(dialogue_net) != "zero"
                and delta_sign(assessment_net) != "zero"
            ),
        }
        attempt_rows.append(row)

        evaluated_answers = assessment.get("evaluated_answers", []) or []
        for index, observation in enumerate(observation_results):
            answer = evaluated_answers[index] if index < len(evaluated_answers) else {}
            assessment_rows.append(
                {
                    "attempt_number": attempt_number,
                    "attempt_id": attempt_id,
                    "skill": row["skill"],
                    "assessment_item_index": index + 1,
                    "question_id": answer.get("question_id"),
                    "is_correct": answer.get("is_correct"),
                    "event_id": observation.get("event_id"),
                    "source_action_event_id": observation.get("source_action_event_id"),
                    "mastery_before": observation.get("mastery_before"),
                    "mastery_after": observation.get("mastery_after"),
                    "delta_mastery": observation.get("delta_mastery"),
                    "persistence_status": observation.get("persistence_status"),
                }
            )

    attempt_df = pd.DataFrame(attempt_rows)
    assessment_df = pd.DataFrame(assessment_rows)

    # Integrity audit.
    action_dup = duplicate_audit(turns, lambda item: nested(item, "identity", "action_event_id"))
    resolver_dup = duplicate_audit(turns, lambda item: nested(item, "learning_state_update", "resolver_event_id"))
    assessment_events = [
        observation
        for assessment in assessments
        for observation in (assessment.get("observation_results", []) or [])
    ]
    assessment_dup = duplicate_audit(assessment_events, lambda item: item.get("event_id"))
    summary_dup = duplicate_audit(summaries, lambda item: nested(item, "identity", "attempt_id"))
    exact_duplicates = {
        "turn_outcomes": sum(count - 1 for count in Counter(turn_raw).values() if count > 1),
        "assessment_outcomes": sum(count - 1 for count in Counter(assessment_raw).values() if count > 1),
        "attempt_summaries": sum(count - 1 for count in Counter(summary_raw).values() if count > 1),
    }
    integrity = Counter()
    per_attempt_integrity: list[dict[str, Any]] = []
    action_ids = {nested(turn, "identity", "action_event_id") for turn in turns}
    for attempt_number, attempt_id in enumerate(attempt_order, start=1):
        ordered_source = turns_by_attempt[attempt_id]
        file_order = [
            turn for turn in turns if nested(turn, "identity", "attempt_id") == attempt_id
        ]
        indices = [int(nested(turn, "identity", "action_turn_index")) for turn in file_order]
        start_mismatch = int(bool(indices) and indices[0] != 1)
        monotonic_mismatch = int(indices != sorted(indices))
        contiguous_mismatch = int(indices != list(range(1, len(indices) + 1)))
        resolver_identity_mismatch = 0
        text_history_mismatch = 0
        mastery_continuity_mismatch = 0
        delta_equation_mismatch = 0
        action_mastery_mismatch = 0
        timestamp_order_mismatch = 0
        previous_timestamp: str | None = None
        for index, turn in enumerate(ordered_source):
            action_id = str(nested(turn, "identity", "action_event_id"))
            resolver_id = str(nested(turn, "learning_state_update", "resolver_event_id"))
            if not resolver_id.startswith(action_id + ":resolver:"):
                resolver_identity_mismatch += 1
            before = float(nested(turn, "learning_state_update", "mastery_before"))
            after = float(nested(turn, "learning_state_update", "mastery_after"))
            delta = float(nested(turn, "learning_state_update", "delta_mastery"))
            if abs(delta - (after - before)) > 1e-12:
                delta_equation_mismatch += 1
            mastery_at_action = nested(turn, "state_before_action", "mastery_at_action")
            if mastery_at_action is not None and abs(float(mastery_at_action) - before) > 1e-12:
                action_mastery_mismatch += 1
            timestamp = nested(turn, "provenance", "timestamp")
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                timestamp_order_mismatch += 1
            previous_timestamp = timestamp
            if index > 0:
                previous = ordered_source[index - 1]
                previous_after = float(nested(previous, "learning_state_update", "mastery_after"))
                if abs(before - previous_after) > 1e-12:
                    mastery_continuity_mismatch += 1
                history = nested(turn, "state_before_action", "history_before_action", default=[]) or []
                prior_tutor = nested(previous, "tutor_response")
                prior_student = nested(previous, "next_learner_observation", "student_text")
                if (
                    len(history) < 2
                    or history[-2].get("user") != "teacher"
                    or history[-2].get("text") != prior_tutor
                    or history[-1].get("user") != "student"
                    or history[-1].get("text") != prior_student
                ):
                    text_history_mismatch += 1
        assessment = assessments_by_id.get(attempt_id, {})
        observations = assessment.get("observation_results", []) or []
        assessment_separation_mismatch = sum(
            1
            for observation in observations
            if observation.get("source_action_event_id") is not None
            or observation.get("event_id") in action_ids
        )
        assessment_chain_mismatch = 0
        previous_after = (
            float(nested(ordered_source[-1], "learning_state_update", "mastery_after"))
            if ordered_source
            else None
        )
        for observation in observations:
            before = float(observation["mastery_before"])
            after = float(observation["mastery_after"])
            delta = float(observation["delta_mastery"])
            if previous_after is not None and abs(before - previous_after) > 1e-12:
                assessment_chain_mismatch += 1
            if abs(delta - (after - before)) > 1e-12:
                delta_equation_mismatch += 1
            previous_after = after
        record = {
            "attempt_number": attempt_number,
            "attempt_id": attempt_id,
            "index_starts_at_1_mismatches": start_mismatch,
            "index_monotonic_mismatches": monotonic_mismatch,
            "index_contiguous_mismatches": contiguous_mismatch,
            "resolver_identity_mismatches": resolver_identity_mismatch,
            "text_history_mismatches": text_history_mismatch,
            "mastery_continuity_mismatches": mastery_continuity_mismatch,
            "action_mastery_mismatches": action_mastery_mismatch,
            "delta_equation_mismatches": delta_equation_mismatch,
            "timestamp_order_mismatches": timestamp_order_mismatch,
            "assessment_chain_mismatches": assessment_chain_mismatch,
            "assessment_separation_mismatches": assessment_separation_mismatch,
        }
        per_attempt_integrity.append(record)
        for key, value in record.items():
            if key not in {"attempt_number", "attempt_id"}:
                integrity[key] += int(value)

    integrity.update(
        {
            "duplicate_action_id_groups": action_dup[0],
            "duplicate_action_id_extra_records": action_dup[1],
            "conflicting_action_id_groups": action_dup[2],
            "duplicate_resolver_id_groups": resolver_dup[0],
            "duplicate_resolver_id_extra_records": resolver_dup[1],
            "conflicting_resolver_id_groups": resolver_dup[2],
            "duplicate_assessment_event_groups": assessment_dup[0],
            "duplicate_assessment_event_extra_records": assessment_dup[1],
            "conflicting_assessment_event_groups": assessment_dup[2],
            "duplicate_attempt_summary_groups": summary_dup[0],
            "duplicate_attempt_summary_extra_records": summary_dup[1],
            "conflicting_attempt_summary_groups": summary_dup[2],
            "exact_duplicate_turn_records": exact_duplicates["turn_outcomes"],
            "exact_duplicate_assessment_records": exact_duplicates["assessment_outcomes"],
            "exact_duplicate_summary_records": exact_duplicates["attempt_summaries"],
        }
    )
    pending_actions = len([path for path in (DATA_DIR / "pending_actions").iterdir() if path.is_file()])
    pending_failures = len(
        [path for path in (DATA_DIR / "pending_processing_failures").iterdir() if path.is_file()]
    )
    integrity["pending_actions"] = pending_actions
    integrity["pending_processing_failures"] = pending_failures

    # Privacy and provenance, scoped to records carrying the post-fix lifecycle schema.
    post_fix_ids = {
        nested(summary, "identity", "attempt_id")
        for summary in summaries
        if summary.get("attempt_started_at") is not None
        and summary.get("completion_status") is not None
        and nested(summary, "provenance", "move_order") is not None
        and nested(summary, "provenance", "policy", "policy_state_relative_path") is not None
    }
    post_fix_records = [
        record
        for record in turns + assessments + summaries
        if nested(record, "identity", "attempt_id") in post_fix_ids
    ]
    post_fix_strings = [item for record in post_fix_records for item in flatten_strings(record)]
    joined_post_fix = "\n".join(value for _, value in post_fix_strings)
    privacy_patterns = {
        "absolute_C_Users_paths": r"(?i)C:[\\/]Users[\\/]",
        "Lenovo": r"(?i)\bLenovo\b",
        "username": r"(?i)\busername\b",
        "email_word": r"(?i)\bemail\b",
        "email_address": r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        "account_id": r"(?i)\baccount(?:[ _-]?id)?\b",
        "api_key": r"(?i)\bapi(?:[ _-]?key)?\b",
        "auth_token": r"(?i)\b(?:auth|authentication|authorization)(?:[ _-]?token)?\b",
        "secret": r"(?i)\bsecret\b",
    }
    privacy_counts = {
        name: len(re.findall(pattern, joined_post_fix)) for name, pattern in privacy_patterns.items()
    }
    path_fields = [
        (location, value)
        for location, value in post_fix_strings
        if "path" in location.rsplit(".", 1)[-1].casefold()
    ]
    rooted_path_fields = [
        (location, value)
        for location, value in path_fields
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")
    ]
    pseudonyms = sorted(
        {
            str(nested(record, "identity", "student_pseudonymous_id"))
            for record in post_fix_records
            if nested(record, "identity", "student_pseudonymous_id") is not None
        }
    )
    invalid_pseudonyms = [value for value in pseudonyms if not re.fullmatch(r"psn_[0-9a-f]{24}", value)]

    provenance_fields = {
        "data_mode": [nested(turn, "provenance", "data_mode") for turn in turns],
        "selector_mode": [nested(turn, "provenance", "selector_mode") for turn in turns],
        "selector_checkpoint": [nested(turn, "selector", "checkpoint") for turn in turns],
        "selector_model_sha256": [nested(turn, "selector", "model_sha256") for turn in turns],
        "move_order": [nested(turn, "provenance", "move_order") for turn in turns],
        "learner_agency_version": [nested(turn, "provenance", "learner_agency_version") for turn in turns],
        "c3_version": [nested(turn, "provenance", "c3_version") for turn in turns],
        "lints_policy_version": [nested(turn, "provenance", "lints_policy_version") for turn in turns],
        "lints_policy_lineage": [nested(turn, "provenance", "lints_policy_lineage") for turn in turns],
        "tau": [nested(turn, "provenance", "tau") for turn in turns],
        "mrb1_version": [nested(turn, "tutor_quality", "model_version") for turn in turns],
        "mrb1_sha256": [nested(turn, "tutor_quality", "model_sha256") for turn in turns],
        "resolver_version": [nested(turn, "provenance", "resolver_version") for turn in turns],
        "bkt_version": [nested(turn, "provenance", "bkt_config_version") for turn in turns],
        "bkt_sha256": [nested(turn, "provenance", "bkt_config_sha256") for turn in turns],
    }
    provenance_unique: dict[str, dict[str, Any]] = {}
    for key, values in provenance_fields.items():
        nonmissing = sorted({canonical(value) if isinstance(value, (dict, list)) else str(value) for value in values if value is not None})
        provenance_unique[key] = {
            "nonmissing_values": nonmissing,
            "missing_count": sum(value is None for value in values),
            "conflicting_nonmissing": len(nonmissing) > 1,
        }
    mixed_runtime_provenance = sum(
        int(details["conflicting_nonmissing"]) for details in provenance_unique.values()
    )

    # Mastery/headroom and BKT distributions.
    mastery_stats = stats(turn_df["mastery_before"])
    headroom_stats = stats(turn_df["headroom"])
    delta_stats = stats(turn_df["delta_mastery"])
    absolute_delta_stats = stats(turn_df["delta_mastery"].abs())
    band_specs = [
        ("[0,.25)", 0.0, 0.25, False),
        ("[.25,.50)", 0.25, 0.50, False),
        ("[.50,.75)", 0.50, 0.75, False),
        ("[.75,.90)", 0.75, 0.90, False),
        ("[.90,.97)", 0.90, 0.97, False),
        ("[.97,1]", 0.97, 1.0, True),
    ]
    band_rows: list[dict[str, Any]] = []
    for label, lower, upper, inclusive_upper in band_specs:
        mask = (turn_df["mastery_before"] >= lower) & (
            turn_df["mastery_before"] <= upper
            if inclusive_upper
            else turn_df["mastery_before"] < upper
        )
        subset = turn_df[mask]
        values = stats(subset["delta_mastery"])
        absolute_values = stats(subset["delta_mastery"].abs())
        signs = count_map(subset["delta_sign"], ["positive", "zero", "negative"])
        band_rows.append(
            {
                "mastery_band": label,
                "turn_count": int(len(subset)),
                "mean_raw_delta": values["mean"],
                "median_raw_delta": values["median"],
                "mean_absolute_delta": absolute_values["mean"],
                "median_absolute_delta": absolute_values["median"],
                "positive": signs["positive"],
                "zero": signs["zero"],
                "negative": signs["negative"],
            }
        )
    band_df = pd.DataFrame(band_rows)

    status_counts = count_map(
        turn_df["status"],
        ["observed_update", "resolved_no_update", "censored_no_response", "processing_failed"],
    )
    delta_sign_counts = count_map(turn_df["delta_sign"], ["positive", "zero", "negative"])

    # Learner-evidence distributions and cross-tabs.
    evaluator_distribution = distribution_frame(turn_df["correctness"], CORRECTNESS, "correctness")
    evaluator_delta_counts = pd.crosstab(turn_df["correctness"], turn_df["delta_sign"]).reindex(
        index=CORRECTNESS, columns=["positive", "zero", "negative"], fill_value=0
    )
    evaluator_delta_percent = evaluator_delta_counts.div(
        evaluator_delta_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0) * 100.0
    evaluator_delta_rows: list[dict[str, Any]] = []
    for correctness in evaluator_delta_counts.index:
        for sign in evaluator_delta_counts.columns:
            evaluator_delta_rows.append(
                {
                    "correctness": correctness,
                    "delta_sign": sign,
                    "count": int(evaluator_delta_counts.loc[correctness, sign]),
                    "row_percent": float(evaluator_delta_percent.loc[correctness, sign]),
                }
            )
    resolver_rows: list[dict[str, Any]] = []
    for field in [
        "primary_signal",
        "should_update",
        "observation_source",
        "reasoning_present",
        "uncertainty_present",
        "clarification_present",
        "repeated_misunderstanding",
    ]:
        values = turn_df[field].map(lambda value: "<missing>" if pd.isna(value) else str(value).casefold() if isinstance(value, bool) else str(value))
        for value, count in values.value_counts(dropna=False).items():
            resolver_rows.append({"field": field, "value": value, "count": int(count)})
    resolver_df = pd.DataFrame(resolver_rows)

    # Selector/move distributions and transitions.
    raw_distribution = distribution_frame(turn_df["raw_move"], MOVES, "move")
    effective_distribution = distribution_frame(turn_df["effective_move"], MOVES, "move")
    final_distribution = distribution_frame(turn_df["final_move"], MOVES, "move")
    transition_rows: list[dict[str, Any]] = []
    for name, source, target in [
        ("raw_to_effective", "raw_move", "effective_move"),
        ("effective_to_final", "effective_move", "final_move"),
        ("raw_to_final", "raw_move", "final_move"),
    ]:
        cross = pd.crosstab(turn_df[source], turn_df[target]).reindex(
            index=MOVES, columns=MOVES, fill_value=0
        )
        for source_move in MOVES:
            for target_move in MOVES:
                transition_rows.append(
                    {
                        "transition": name,
                        "from_move": source_move,
                        "to_move": target_move,
                        "count": int(cross.loc[source_move, target_move]),
                    }
                )
    transition_df = pd.DataFrame(transition_rows)

    agency_turns = turn_df[turn_df["learner_agency_triggered"]].copy()
    agency_detail_columns = [
        "attempt_number",
        "attempt_id",
        "action_turn_index",
        "skill",
        "latest_student_text",
        "raw_move",
        "effective_move",
        "final_move",
        "correctness",
        "delta_mastery",
    ]
    agency_details = agency_turns[agency_detail_columns] if not agency_turns.empty else pd.DataFrame(columns=agency_detail_columns)

    # LinTS authority and override details.
    eligible_counts = turn_df["eligible_arm_count"].astype(int)
    lints_selected = count_map(turn_df["selected_arm"], ARMS)
    override_columns = [
        "attempt_number",
        "attempt_id",
        "action_turn_index",
        "skill",
        "latest_student_text",
        "raw_probabilities",
        "effective_probabilities",
        "eligible_arms",
        "selected_arm",
        "base_move",
        "final_move",
        "mastery_before",
        "correctness",
        "delta_mastery",
        "MI",
        "ML",
        "PG",
        "A",
    ]
    overrides_df = turn_df[turn_df["lints_overridden"]][override_columns].copy()
    policy_counts = {
        arm: int(nested(policy, "policy_state", "arm_update_counts", arm, default=0)) for arm in ARMS
    }
    lints_authority = {
        "total_turns": int(len(turn_df)),
        "baseline_only_eligible_turns": int((eligible_counts == 1).sum()),
        "turns_with_nonbaseline_eligible": int((eligible_counts >= 2).sum()),
        "eligible_arm_count": stats(eligible_counts),
        "selected_arm_counts": lints_selected,
        "actual_override_count": int(turn_df["lints_overridden"].sum()),
        "actual_override_rate_percent": float(100.0 * turn_df["lints_overridden"].mean()),
        "policy_total_updates": int(nested(policy, "policy_state", "total_updates")),
        "policy_arm_update_counts": policy_counts,
        "experience_log_attempts": len(experiences),
        "experience_log_turns": sum(int(item.get("num_turns", 0)) for item in experiences),
    }

    # Effective-confidence/gate analysis.
    gap_stats = stats(turn_df["effective_top1_top2_gap"])
    gap_threshold_rows = [
        {
            "gap_at_or_below": threshold,
            "count": int((turn_df["effective_top1_top2_gap"] <= threshold + EPS).sum()),
            "percent": float(100.0 * (turn_df["effective_top1_top2_gap"] <= threshold + EPS).mean()),
        }
        for threshold in [0.02, 0.05, 0.10, 0.15, 0.20]
    ]
    tau_permits_alternative_count = int((eligible_counts >= 2).sum())
    tau_permits_alternative_fraction = float(tau_permits_alternative_count / len(turn_df))

    # Move x outcome observational associations.
    move_outcome_rows: list[dict[str, Any]] = []
    for move in MOVES:
        subset = turn_df[turn_df["final_move"] == move]
        mastery = stats(subset["mastery_before"])
        delta = stats(subset["delta_mastery"])
        signs = count_map(subset["delta_sign"], ["positive", "zero", "negative"])
        evals = count_map(subset["correctness"], CORRECTNESS)
        move_outcome_rows.append(
            {
                "final_move": move,
                "turn_count": int(len(subset)),
                "mean_mastery_before": mastery["mean"],
                "median_mastery_before": mastery["median"],
                "mean_delta": delta["mean"],
                "median_delta": delta["median"],
                "positive": signs["positive"],
                "zero": signs["zero"],
                "negative": signs["negative"],
                "correct": evals["correct"],
                "partial": evals["partial"],
                "incorrect": evals["incorrect"],
                "unknown": evals["unknown"],
                "mean_MI": float(subset["MI"].mean()) if len(subset) else None,
                "mean_ML": float(subset["ML"].mean()) if len(subset) else None,
                "mean_PG": float(subset["PG"].mean()) if len(subset) else None,
                "mean_A": float(subset["A"].mean()) if len(subset) else None,
            }
        )
    move_outcome_df = pd.DataFrame(move_outcome_rows)
    move_eval_cross = pd.crosstab(turn_df["final_move"], turn_df["correctness"]).reindex(
        index=MOVES, columns=CORRECTNESS, fill_value=0
    )
    move_sign_cross = pd.crosstab(turn_df["final_move"], turn_df["delta_sign"]).reindex(
        index=MOVES, columns=["positive", "zero", "negative"], fill_value=0
    )

    # Decision modification groups.
    decision_group_order = ["A_unchanged", "B_agency_only", "C_lints_only", "D_both"]
    decision_group_rows: list[dict[str, Any]] = []
    for group in decision_group_order:
        subset = turn_df[turn_df["decision_group"] == group]
        mastery = stats(subset["mastery_before"])
        delta = stats(subset["delta_mastery"])
        signs = count_map(subset["delta_sign"], ["positive", "zero", "negative"])
        evals = count_map(subset["correctness"], CORRECTNESS)
        decision_group_rows.append(
            {
                "decision_group": group,
                "turn_count": int(len(subset)),
                "mastery_before_mean": mastery["mean"],
                "mastery_before_median": mastery["median"],
                "delta_mean": delta["mean"],
                "delta_median": delta["median"],
                "positive": signs["positive"],
                "zero": signs["zero"],
                "negative": signs["negative"],
                "correct": evals["correct"],
                "partial": evals["partial"],
                "incorrect": evals["incorrect"],
                "unknown": evals["unknown"],
                "mean_MI": float(subset["MI"].mean()) if len(subset) else None,
                "mean_ML": float(subset["ML"].mean()) if len(subset) else None,
                "mean_PG": float(subset["PG"].mean()) if len(subset) else None,
                "mean_A": float(subset["A"].mean()) if len(subset) else None,
            }
        )
    decision_group_df = pd.DataFrame(decision_group_rows)

    # Skill-level analysis without pooling mastery trajectories across skills.
    skill_rows: list[dict[str, Any]] = []
    for skill in sorted(turn_df["skill"].unique()):
        subset = turn_df[turn_df["skill"] == skill]
        skill_attempts = attempt_df[attempt_df["skill"] == skill]
        start = stats(skill_attempts["mastery_start"])
        turn_mastery = stats(subset["mastery_before"])
        raw_counts = count_map(subset["raw_move"], MOVES)
        final_counts = count_map(subset["final_move"], MOVES)
        evals = count_map(subset["correctness"], CORRECTNESS)
        signs = count_map(subset["delta_sign"], ["positive", "zero", "negative"])
        saturation_fraction = float((subset["mastery_before"] >= 0.97).mean()) if len(subset) else 0.0
        flags: list[str] = []
        if start["max"] is not None and start["max"] >= 0.90:
            flags.append("very_high_starting_mastery_present")
        if len(subset) < 10:
            flags.append("low_turn_count_<10")
        if saturation_fraction >= 0.80 or float(skill_attempts["mastery_final"].median()) >= 0.99:
            flags.append("near_complete_saturation")
        skill_rows.append(
            {
                "skill": skill,
                "attempt_count": int(len(skill_attempts)),
                "turn_count": int(len(subset)),
                "start_mastery_min": start["min"],
                "start_mastery_median": start["median"],
                "start_mastery_max": start["max"],
                "turn_mastery_before_min": turn_mastery["min"],
                "turn_mastery_before_median": turn_mastery["median"],
                "turn_mastery_before_mean": turn_mastery["mean"],
                "turn_mastery_before_max": turn_mastery["max"],
                "raw_G_P_F_T": compact_counts(raw_counts, MOVES),
                "final_G_P_F_T": compact_counts(final_counts, MOVES),
                "correct": evals["correct"],
                "partial": evals["partial"],
                "incorrect": evals["incorrect"],
                "unknown": evals["unknown"],
                "positive": signs["positive"],
                "zero": signs["zero"],
                "negative": signs["negative"],
                "dialogue_net_mastery_change": float(skill_attempts["dialogue_net_delta"].sum()),
                "assessment_net_mastery_change": float(skill_attempts["assessment_net_delta"].sum()),
                "fraction_turns_mastery_ge_0_97": saturation_fraction,
                "descriptive_flags": ";".join(flags) if flags else "none",
            }
        )
    skill_df = pd.DataFrame(skill_rows)

    # Learner dependence.
    unique_learners = sorted(attempt_df["learner_pseudonym"].dropna().unique().tolist())
    learner_skill_counts = (
        attempt_df.groupby(["learner_pseudonym", "skill"], dropna=False)
        .size()
        .reset_index(name="attempt_count")
    )
    repeated_learner_skills = learner_skill_counts[learner_skill_counts["attempt_count"] > 1]

    # MRB1 overall, by final move, and by evaluator category.
    mrb_overall_rows: list[dict[str, Any]] = []
    for key in MRB_KEYS:
        row = {"criterion": key, **stats(turn_df[MRB_SHORT[key]])}
        mrb_overall_rows.append(row)
    mrb_overall_df = pd.DataFrame(mrb_overall_rows)
    mrb_by_move_rows: list[dict[str, Any]] = []
    for move in MOVES:
        subset = turn_df[turn_df["final_move"] == move]
        row = {"final_move": move, "turn_count": int(len(subset))}
        for short in MRB_SHORT.values():
            row[f"mean_{short}"] = float(subset[short].mean()) if len(subset) else None
        mrb_by_move_rows.append(row)
    mrb_by_move_df = pd.DataFrame(mrb_by_move_rows)
    mrb_by_evaluator_rows: list[dict[str, Any]] = []
    for correctness in CORRECTNESS:
        subset = turn_df[turn_df["correctness"] == correctness]
        row = {"correctness": correctness, "turn_count": int(len(subset))}
        for short in MRB_SHORT.values():
            row[f"mean_{short}"] = float(subset[short].mean()) if len(subset) else None
        mrb_by_evaluator_rows.append(row)
    mrb_by_evaluator_df = pd.DataFrame(mrb_by_evaluator_rows)

    # Delayed attempt-level versus immediate turn-level credit.
    credit_rows: list[dict[str, Any]] = []
    for attempt in attempt_rows:
        subset = turn_df[turn_df["attempt_id"] == attempt["attempt_id"]]
        values = [float(value) for value in subset["delta_mastery"]]
        credit_rows.append(
            {
                "attempt_number": attempt["attempt_number"],
                "attempt_id": attempt["attempt_id"],
                "skill": attempt["skill"],
                "old_lints_cumulative_final_mastery_delta": attempt["total_delta"],
                "dialogue_turn_count": len(values),
                "dialogue_delta_min": min(values),
                "dialogue_delta_median": float(np.median(values)),
                "dialogue_delta_mean": float(np.mean(values)),
                "dialogue_delta_max": max(values),
                "positive_turns": int((subset["delta_sign"] == "positive").sum()),
                "zero_turns": int((subset["delta_sign"] == "zero").sum()),
                "negative_turns": int((subset["delta_sign"] == "negative").sum()),
                "turn_deltas": json.dumps(values, separators=(",", ":")),
            }
        )
    credit_df = pd.DataFrame(credit_rows)
    positive_attempts = attempt_df[attempt_df["total_delta"] > EPS]
    positive_attempts_with_negative_turn = positive_attempts[
        positive_attempts["negative_dialogue_turns"] > 0
    ]

    assessment_contribution_df = attempt_df[
        [
            "attempt_number",
            "attempt_id",
            "skill",
            "dialogue_net_delta",
            "assessment_net_delta",
            "total_delta",
            "assessment_absolute_path_share",
            "dialogue_assessment_directions_differ",
        ]
    ].copy()
    negative_dialogue_positive_total = attempt_df[
        (attempt_df["dialogue_net_delta"] < -EPS) & (attempt_df["total_delta"] > EPS)
    ]

    # Consolidated descriptive-statistics table.
    descriptive_rows: list[dict[str, Any]] = []
    for variable, values in [
        ("mastery_before", turn_df["mastery_before"]),
        ("headroom", turn_df["headroom"]),
        ("delta_mastery", turn_df["delta_mastery"]),
        ("absolute_delta_mastery", turn_df["delta_mastery"].abs()),
        ("effective_top1_probability", turn_df["effective_top1_probability"]),
        ("effective_top2_probability", turn_df["effective_top2_probability"]),
        ("effective_top1_top2_gap", turn_df["effective_top1_top2_gap"]),
        ("eligible_arm_count", turn_df["eligible_arm_count"]),
    ]:
        descriptive_rows.append({"variable": variable, **stats(values)})
    descriptive_df = pd.DataFrame(descriptive_rows)

    # Write derived tables before producing the report.
    written_paths: list[Path] = []
    written_paths.append(write_csv("attempt_table.csv", attempt_df))
    written_paths.append(write_csv("turn_table.csv", turn_df))
    written_paths.append(write_csv("assessment_observations.csv", assessment_df))
    written_paths.append(write_csv("descriptive_statistics.csv", descriptive_df))
    written_paths.append(write_csv("headroom_bands.csv", band_df))
    written_paths.append(write_csv("evaluator_delta_crosstab.csv", evaluator_delta_rows))
    written_paths.append(write_csv("resolver_summary.csv", resolver_df))
    written_paths.append(write_csv("move_transitions.csv", transition_df))
    written_paths.append(write_csv("learner_agency_turns.csv", agency_details))
    written_paths.append(write_csv("lints_overrides.csv", overrides_df))
    written_paths.append(write_csv("move_outcome_observational.csv", move_outcome_df))
    written_paths.append(write_csv("decision_group_summary.csv", decision_group_df))
    written_paths.append(write_csv("skill_summary.csv", skill_df))
    written_paths.append(write_csv("learner_skill_counts.csv", learner_skill_counts))
    written_paths.append(write_csv("mrb1_overall.csv", mrb_overall_df))
    written_paths.append(write_csv("mrb1_by_final_move.csv", mrb_by_move_df))
    written_paths.append(write_csv("mrb1_by_evaluator.csv", mrb_by_evaluator_df))
    written_paths.append(write_csv("attempt_vs_turn_credit.csv", credit_df))
    written_paths.append(write_csv("assessment_contribution.csv", assessment_contribution_df))
    written_paths.append(write_csv("integrity_by_attempt.csv", per_attempt_integrity))
    written_paths.append(
        write_csv(
            "integrity_totals.csv",
            [{"check": key, "count": int(value)} for key, value in sorted(integrity.items())],
        )
    )
    provenance_rows = [
        {
            "field": key,
            "nonmissing_values": json.dumps(details["nonmissing_values"], separators=(",", ":")),
            "missing_count": details["missing_count"],
            "conflicting_nonmissing": details["conflicting_nonmissing"],
        }
        for key, details in provenance_unique.items()
    ]
    written_paths.append(write_csv("provenance_audit.csv", provenance_rows))

    # CPU-only plots.
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_paths: list[Path] = []

    def save_plot(name: str) -> None:
        path = OUT_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

    plt.figure(figsize=(7, 4.5))
    plt.hist(turn_df["mastery_before"], bins=np.linspace(0, 1, 21), color="#4472C4", edgecolor="white")
    plt.xlabel("Mastery before dialogue turn")
    plt.ylabel("Turn count")
    plt.title("Starting mastery across real dialogue turns")
    save_plot("mastery_before_histogram.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(turn_df["delta_mastery"], bins=20, color="#70AD47", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Immediate confidence-weighted BKT posterior update")
    plt.ylabel("Turn count")
    plt.title("Immediate dialogue delta distribution")
    save_plot("delta_mastery_histogram.png")

    plt.figure(figsize=(7, 4.5))
    for move, color in zip(MOVES, ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"]):
        subset = turn_df[turn_df["final_move"] == move]
        plt.scatter(subset["mastery_before"], subset["delta_mastery"], label=move, alpha=0.75, s=34, color=color)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Mastery before dialogue turn")
    plt.ylabel("Immediate BKT posterior update")
    plt.title("Immediate delta versus starting mastery")
    plt.legend()
    save_plot("delta_vs_mastery_before_scatter.png")

    plt.figure(figsize=(6.5, 4.5))
    raw_counts_plot = count_map(turn_df["raw_move"], MOVES)
    plt.bar(MOVES, [raw_counts_plot[move] for move in MOVES], color="#4472C4")
    plt.ylabel("Turn count")
    plt.title("Raw MD7 move counts")
    save_plot("raw_md7_move_counts.png")

    plt.figure(figsize=(6.5, 4.5))
    final_counts_plot = count_map(turn_df["final_move"], MOVES)
    plt.bar(MOVES, [final_counts_plot[move] for move in MOVES], color="#70AD47")
    plt.ylabel("Turn count")
    plt.title("Final pedagogical move counts")
    save_plot("final_move_counts.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(turn_df["effective_top1_top2_gap"], bins=20, color="#ED7D31", edgecolor="white")
    plt.axvline(0.10, color="black", linewidth=1, linestyle="--", label="tau=.10")
    plt.xlabel("Effective top-1 minus top-2 probability")
    plt.ylabel("Turn count")
    plt.title("Effective MD7 confidence-gap distribution")
    plt.legend()
    save_plot("effective_gap_distribution.png")

    plt.figure(figsize=(6.5, 4.5))
    eligible_distribution = eligible_counts.value_counts().sort_index()
    plt.bar([str(index) for index in eligible_distribution.index], eligible_distribution.values, color="#A5A5A5")
    plt.xlabel("Eligible-arm count")
    plt.ylabel("Turn count")
    plt.title("LinTS eligible-arm count distribution")
    save_plot("lints_eligible_arm_count_distribution.png")

    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot([turn_df[key].dropna() for key in ["MI", "ML", "PG", "A"]], tick_labels=["MI", "ML", "PG", "A"], patch_artist=True)
    plt.ylim(0, 1)
    plt.ylabel("Frozen MRB1 score")
    plt.title("MRB1 criterion distributions (not ground truth)")
    save_plot("mrb1_criterion_distributions.png")
    written_paths.extend(plot_paths)

    # Side-effect verification after all derived writes.
    final_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected_inputs}
    hash_changes = {
        path: {"before": initial_hashes[path], "after": final_hashes[path]}
        for path in initial_hashes
        if initial_hashes[path] != final_hashes[path]
    }

    completed_attempts = int(attempt_df["completion_status"].isin(["completed", "completed_legacy_inferred"]).sum())
    aborted_attempts = int((attempt_df["completion_status"] == "aborted").sum())
    integrity_failures = sum(
        value
        for key, value in integrity.items()
        if key not in {"pending_actions", "pending_processing_failures"} and int(value) > 0
    ) + pending_actions + pending_failures
    privacy_failures = sum(privacy_counts.values()) + len(rooted_path_fields) + len(invalid_pseudonyms)
    provenance_conflicts = mixed_runtime_provenance
    core_clean = integrity_failures == 0 and privacy_failures == 0 and provenance_conflicts == 0
    verdict = (
        "B. PILOT DATASET CLEAN BUT COVERAGE TOO WEAK — COLLECT MORE BEFORE MD8 DESIGN"
        if core_clean
        else "C. DATA INTEGRITY/PROVENANCE ISSUE FOUND — FIX BEFORE FURTHER ANALYSIS"
    )

    summary_payload = {
        "analysis_version": "md_self_improvement_pilot_analysis_v1",
        "analysis_type": "read_only_descriptive_scientific_analysis",
        "inventory": {
            "completed_attempts": completed_attempts,
            "aborted_attempts": aborted_attempts,
            "dialogue_turns": int(len(turn_df)),
            "formal_assessment_observations": int(len(assessment_df)),
            "pending_actions": pending_actions,
            "pending_processing_failures": pending_failures,
        },
        "integrity": dict(integrity),
        "privacy": {
            "post_fix_attempt_count": len(post_fix_ids),
            "post_fix_record_count": len(post_fix_records),
            "pattern_counts": privacy_counts,
            "path_field_count": len(path_fields),
            "rooted_path_field_count": len(rooted_path_fields),
            "pseudonyms": pseudonyms,
            "invalid_pseudonyms": invalid_pseudonyms,
        },
        "provenance": {
            "turn_fields": provenance_unique,
            "mixed_nonmissing_value_fields": mixed_runtime_provenance,
            "policy_schema": policy.get("schema_version"),
            "policy_class": policy.get("policy_class"),
            "policy_state_schema": nested(policy, "policy_state", "schema_version"),
            "lineage_manifest": lineage,
            "legacy_lineage_manifest_absolute_paths": [
                key
                for key in ["policy_state", "experience_log"]
                if re.match(r"^[A-Za-z]:[\\/]", str(lineage.get(key, "")))
            ],
        },
        "mastery_before": mastery_stats,
        "headroom": headroom_stats,
        "dialogue_delta": {
            "status_counts": status_counts,
            "sign_counts": delta_sign_counts,
            "statistics": delta_stats,
            "absolute_statistics": absolute_delta_stats,
            "interpretation": "confidence-weighted BKT posterior belief update after resolved learner evidence",
        },
        "moves": {
            "raw": raw_distribution.to_dict(orient="records"),
            "effective": effective_distribution.to_dict(orient="records"),
            "final": final_distribution.to_dict(orient="records"),
            "learner_agency_trigger_count": int(len(agency_turns)),
            "learner_agency_trigger_rate_percent": float(100.0 * len(agency_turns) / len(turn_df)),
        },
        "lints_authority": lints_authority,
        "effective_gap": {
            "statistics": gap_stats,
            "thresholds": gap_threshold_rows,
            "tau_0_10_permits_alternative_count": tau_permits_alternative_count,
            "tau_0_10_permits_alternative_fraction": tau_permits_alternative_fraction,
        },
        "learner_dependence": {
            "unique_learner_count": len(unique_learners),
            "learners": unique_learners,
            "repeated_learner_skill_combinations": repeated_learner_skills.to_dict(orient="records"),
        },
        "credit": {
            "positive_cumulative_attempts": int(len(positive_attempts)),
            "positive_cumulative_attempts_with_negative_dialogue_turn": int(len(positive_attempts_with_negative_turn)),
            "negative_dialogue_but_positive_total_attempts": negative_dialogue_positive_total["attempt_number"].tolist(),
        },
        "data_sufficiency": {
            "pipeline_prototyping": "yes",
            "tiny_pilot_retraining_experiment": "not yet supported beyond an engineering smoke test",
            "policy_improvement_claims": "no",
            "reason": "57 turns come from one learner, five skills, sparse intervention coverage, and observational non-randomized decisions.",
        },
        "side_effects": {
            "training": 0,
            "model_writes": 0,
            "policy_writes": 0,
            "authoritative_research_writes": 0,
            "api_calls": 0,
            "tutor_conversations": 0,
            "protected_evaluation_use": 0,
            "input_hash_changes": hash_changes,
        },
        "verdict": verdict,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written_paths.append(summary_path)

    # Markdown report.
    inventory_display = attempt_df[
        [
            "attempt_number",
            "attempt_id",
            "learner_pseudonym",
            "skill",
            "problem_id",
            "problem_summary",
            "attempt_started_at",
            "completion_status",
            "dialogue_turn_count",
            "assessment_item_count",
        ]
    ]
    attempt_display = attempt_df[
        [
            "attempt_number",
            "attempt_id",
            "learner_pseudonym",
            "skill",
            "dialogue_turn_count",
            "mastery_start",
            "mastery_pre_assessment",
            "mastery_final",
            "dialogue_net_delta",
            "assessment_net_delta",
            "total_delta",
            "raw_G_P_F_T",
            "final_G_P_F_T",
            "learner_agency_triggers",
            "lints_nonbaseline_selections",
            "actual_lints_overrides",
            "evaluator_correct",
            "evaluator_partial",
            "evaluator_incorrect",
            "evaluator_unknown",
            "mean_MRB1_MI",
            "mean_MRB1_ML",
            "mean_MRB1_PG",
            "mean_MRB1_A",
        ]
    ]
    integrity_display = pd.DataFrame(
        [{"check": key, "mismatch_count": int(value)} for key, value in sorted(integrity.items())]
    )
    mastery_display = pd.DataFrame([{"variable": "mastery_before", **mastery_stats}, {"variable": "headroom", **headroom_stats}])
    bkt_display = pd.DataFrame([{"variable": "delta_mastery", **delta_stats}, {"variable": "absolute_delta_mastery", **absolute_delta_stats}])
    move_distribution_display = raw_distribution.rename(columns={"count": "raw_count", "percent": "raw_percent"}).merge(
        effective_distribution.rename(columns={"count": "effective_count", "percent": "effective_percent"}), on="move"
    ).merge(final_distribution.rename(columns={"count": "final_count", "percent": "final_percent"}), on="move")
    gap_display = pd.DataFrame([gap_stats])
    side_effect_display = pd.DataFrame(
        [
            {"effect": "training", "count": 0},
            {"effect": "model writes", "count": 0},
            {"effect": "policy writes", "count": 0},
            {"effect": "authoritative research writes", "count": 0},
            {"effect": "API calls", "count": 0},
            {"effect": "Tutor conversations", "count": 0},
            {"effect": "protected evaluation use", "count": 0},
            {"effect": "protected-input hash changes", "count": len(hash_changes)},
        ]
    )

    report = f"""# First Multi-Attempt Analysis of Real Self-Improvement Data

Analysis version: `md_self_improvement_pilot_analysis_v1`  
Scope: read-only descriptive analysis; no training rule, reward, threshold, preference pair, or accepted-label decision is created.

## 1. Dataset inventory

- Real completed attempts: **{completed_attempts}** (Attempt 1 is inferred completed from its pre-fix assessment/summary lifecycle; its missing recorded status is preserved.)
- Aborted attempts: **{aborted_attempts}**
- Dialogue turns: **{len(turn_df)}**
- Formal-assessment observations: **{len(assessment_df)}**
- Pending actions / processing failures: **{pending_actions} / {pending_failures}**

{markdown_table(inventory_display)}

## 2. Integrity / privacy / provenance

{markdown_table(integrity_display)}

All integrity mismatch and duplicate counts are zero. Formal assessment remains separate: every assessment `source_action_event_id` is null and its mastery chain begins after the final dialogue observation.

Post-fix scope: **{len(post_fix_ids)} attempts / {len(post_fix_records)} records**. Privacy pattern counts are `{json.dumps(privacy_counts, sort_keys=True)}`; rooted path fields: **{len(rooted_path_fields)}**; invalid pseudonyms: **{len(invalid_pseudonyms)}**.

Non-null runtime provenance has **{mixed_runtime_provenance} conflicting fields**. The consistent deployment is: real data, ordinary-md7r1-v1, MD7 checkpoint md7r1_epoch3 / hash `{provenance_unique['selector_model_sha256']['nonmissing_values'][0]}`, move order G/P/F/T, learner agency `learner_agency_telling_escalation_v1`, C3 `turn_lints_v3_c3_9d`, LinTS `true_disjoint_lints_v3` / `MD7-R1 fresh LinTS v1`, tau `.10`, MRB1 `frozen_mrb1` / hash `{provenance_unique['mrb1_sha256']['nonmissing_values'][0]}`, resolver `2.0`, and BKT `confidence_weighted_bkt_v1` / hash `{provenance_unique['bkt_sha256']['nonmissing_values'][0]}`.

Documented legacy limitation: Attempt 1 omits post-fix lifecycle/move-order/portable-policy fields, and the older deployment `lineage_manifest.json` still stores absolute local paths plus its creation-time `total_updates=15`. Neither artifact was rewritten. Current authoritative policy state reports `{nested(policy, 'policy_state', 'total_updates')}` updates.

## 3. Attempt table

Move count columns use `G/P/F/T`; evaluator columns are explicit counts. No causal interpretation is attached.

{markdown_table(attempt_display)}

## 4. Starting mastery / headroom

Headroom is descriptive only: `1 - mastery_before`.

{markdown_table(mastery_display)}

{markdown_table(band_df)}

The band table is descriptive. It is not a training threshold or reward transformation.

## 5. Turn-level BKT

Statuses: `{json.dumps(status_counts, sort_keys=True)}`. Delta signs: `{json.dumps(delta_sign_counts, sort_keys=True)}`.

{markdown_table(bkt_display)}

Every delta is interpreted only as a **confidence-weighted BKT posterior belief update after resolved learner evidence**. Standard deviations in this report are population standard deviations (`ddof=0`); quartiles use linear interpolation.

## 6. Evaluator / resolver

{markdown_table(evaluator_distribution)}

Evaluator correctness × delta sign (counts and within-row percentages):

{markdown_table(pd.DataFrame(evaluator_delta_rows))}

Resolver-field frequencies:

{markdown_table(resolver_df)}

The four requested behavioural-presence fields are not stored in the turn-outcome schema; they are reported as missing rather than inferred from contributor names.

## 7. Raw / effective / final move distributions

{markdown_table(move_distribution_display)}

Transition counts are in `move_transitions.csv`. Learner-agency triggers: **{len(agency_turns)} / {len(turn_df)} ({100.0 * len(agency_turns) / len(turn_df):.3f}%)**.

Agency-triggered turn details:

{markdown_table(agency_details)}

## 8. LinTS authority

- Total turns: **{len(turn_df)}**
- Baseline-only eligible: **{lints_authority['baseline_only_eligible_turns']}**
- At least one nonbaseline eligible: **{lints_authority['turns_with_nonbaseline_eligible']}**
- Eligible-arm count mean / median / max: **{lints_authority['eligible_arm_count']['mean']:.6f} / {lints_authority['eligible_arm_count']['median']:.6f} / {lints_authority['eligible_arm_count']['max']:.0f}**
- Selected arms: `{json.dumps(lints_selected, sort_keys=True)}`
- Actual overrides: **{lints_authority['actual_override_count']} ({lints_authority['actual_override_rate_percent']:.3f}%)**
- Current policy updates: `{json.dumps(policy_counts, sort_keys=True)}`; total **{lints_authority['policy_total_updates']}**
- Experience log: **{lints_authority['experience_log_attempts']} attempts / {lints_authority['experience_log_turns']} turns**

Actual override details (outcomes are subsequent observations, not causal effects):

{markdown_table(overrides_df)}

## 9. MD7 confidence / gate

{markdown_table(gap_display)}

{markdown_table(pd.DataFrame(gap_threshold_rows))}

At tau `.10`, an alternative was empirically eligible on **{tau_permits_alternative_count}/{len(turn_df)} turns ({100.0 * tau_permits_alternative_fraction:.3f}%)**. This describes deployed LinTS authority and does not recommend a new tau.

## 10. Move × outcome observational tables

**OBSERVATIONAL ASSOCIATIONS ONLY.** Actions were not randomized; moves are not ranked.

{markdown_table(move_outcome_df)}

Final move × evaluator correctness:

{markdown_table(move_eval_cross.reset_index())}

Final move × delta sign:

{markdown_table(move_sign_cross.reset_index())}

Decision-modification groups:

{markdown_table(decision_group_df)}

## 11. Skill coverage

Skills are kept separate; differences are not interpreted as equal-difficulty comparisons. Diagnostic flags are descriptive only: at least one start ≥.90, fewer than 10 turns, and near saturation based on ≥80% of turns at mastery ≥.97 or median final mastery ≥.99.

{markdown_table(skill_df)}

## 12. Learner dependence

Unique learner pseudonyms: **{len(unique_learners)}** — `{', '.join(unique_learners)}`.

**These attempts are longitudinal observations from one learner and are not six independent learners.**

Repeated learner-skill combinations:

{markdown_table(repeated_learner_skills)}

## 13. MRB1 distributions

MRB1 is a frozen model signal, not ground truth, and no cutoff is selected.

{markdown_table(mrb_overall_df)}

By final move:

{markdown_table(mrb_by_move_df)}

By evaluator correctness:

{markdown_table(mrb_by_evaluator_df)}

## 14. Attempt-level versus turn-level credit

{markdown_table(credit_df.drop(columns=['turn_deltas']))}

Attempts with positive cumulative outcome: **{len(positive_attempts)}**; among them, attempts containing at least one negative dialogue delta: **{len(positive_attempts_with_negative_turn)}**. Full immediate-delta lists are retained in `attempt_vs_turn_credit.csv`. This is descriptive evidence about delayed credit assignment.

## 15. Formal assessment contribution

`assessment_absolute_path_share = |assessment net| / (|dialogue net| + |assessment net|)`; it describes evidence-path magnitude and does not attach assessment evidence to Tutor moves.

{markdown_table(assessment_contribution_df)}

Attempts with negative dialogue net but positive final total: `{negative_dialogue_positive_total['attempt_number'].tolist()}`.

## 16. Data sufficiency

- Total real turns: **{len(turn_df)}**
- Turns per final move: `{json.dumps(count_map(turn_df['final_move'], MOVES), sort_keys=True)}`
- Turns per skill: `{json.dumps({row['skill']: row['turn_count'] for row in skill_rows}, sort_keys=True)}`
- Learner-agency modifications: **{len(agency_turns)}**
- LinTS actual overrides: **{len(overrides_df)}**
- Positive / zero / negative dialogue turns: **{delta_sign_counts['positive']} / {delta_sign_counts['zero']} / {delta_sign_counts['negative']}**
- Distinct learners: **{len(unique_learners)}**
- Starting-mastery range: **{mastery_stats['min']:.6f} to {mastery_stats['max']:.6f}**

This is enough for **pipeline and analysis prototyping**. It is not yet enough for a tiny scientific retraining pilot beyond an engineering smoke test, and it is not enough for any claim of policy improvement. The limiting factors are one learner, only five skills, sparse agency/LinTS intervention coverage, move imbalance, and non-randomized observational decisions.

## 17. Issues / limitations

1. One learner supplies all six attempts; within-learner and repeated-skill dependence is material.
2. Attempt 1 is pre-fix and legitimately lacks the new lifecycle/provenance fields.
3. The legacy deployment lineage manifest contains local absolute paths and a creation-time update count; post-fix research records themselves pass the portability/privacy audit.
4. Resolver behavioural-presence booleans are absent from turn-outcome records and cannot be reconstructed without inventing values.
5. Move and skill coverage are imbalanced, with observational rather than randomized selection.
6. High-mastery turns compress available BKT headroom; immediate deltas must not be treated as causal rewards.
7. MRB1 and evaluator outputs are model signals, not ground truth.

## 18. Derived artifact paths

All artifacts are under `{OUT_DIR.relative_to(ROOT).as_posix()}/`:

{chr(10).join(f'- `{path.name}`' for path in sorted(written_paths, key=lambda item: item.name))}
- `report.md`

## 19. Side effects

{markdown_table(side_effect_display)}

Only the derived-analysis directory was written. No APIs, Tutor conversations, training, model/policy updates, authoritative JSONL writes, or protected MathDial/MRBench V3 test access occurred.

## 20. Final verdict

**{verdict}**
"""
    report_path = OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(
        {
            "attempts": len(attempt_df),
            "turns": len(turn_df),
            "assessment_observations": len(assessment_df),
            "integrity_failures": integrity_failures,
            "privacy_failures": privacy_failures,
            "provenance_conflicts": provenance_conflicts,
            "unique_learners": len(unique_learners),
            "skills": int(turn_df["skill"].nunique()),
            "agency_triggers": len(agency_turns),
            "lints_overrides": len(overrides_df),
            "input_hash_changes": len(hash_changes),
            "verdict": verdict,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
