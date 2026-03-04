"""Print raw completeness evidence for a Turn-LinTS SHADOW event lineage."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MOVE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = MOVE_ROOT.parent
DEFAULT_EVENT_ROOT = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "turn_lints_shadow_full_headroom_v1"
)
DEFAULT_PASSIVE = (
    MOVE_ROOT / "results" / "md_self_improvement_turn_data_v1" / "turn_outcomes.jsonl"
)
MOVES = ("generic", "probing", "focus", "telling")
BLOCK_FEATURES = {
    "S": tuple(f"selector_p_{move}" for move in MOVES),
    "K": ("mastery_before", "previous_mastery_delta", "previous_mastery_delta_missing"),
    "L": (
        "previous_reasoning_probability",
        "previous_uncertainty_probability",
        "previous_clarification_probability",
        "previous_learner_signals_missing",
    ),
    "H": (
        *(f"previous_move_{move}" for move in MOVES),
        "previous_move_missing",
        "prior_tutor_turn_count",
    ),
    "Q": (
        "previous_mistake_identification",
        "previous_mistake_location",
        "previous_providing_guidance",
        "previous_actionability",
        "previous_mrb1_missing",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def describe(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            **{
                key: None
                for key in ("min", "q1", "median", "mean", "q3", "max", "std")
            },
        }

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": quantile(0.25),
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "q3": quantile(0.75),
        "max": ordered[-1],
        "std": statistics.stdev(ordered) if len(ordered) > 1 else None,
    }


def context_values(event: dict[str, Any]) -> dict[str, Any]:
    context = event.get("context") or {}
    direct = context.get("feature_values")
    if isinstance(direct, dict):
        return direct
    names, vector = context.get("feature_names"), context.get("vector")
    if isinstance(names, list) and isinstance(vector, list) and len(names) == len(vector):
        return dict(zip(names, vector, strict=True))
    return {}


def passive_index(path: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for row in load_jsonl(path):
        identity = row.get("identity") or {}
        action_id = identity.get("action_event_id")
        if isinstance(action_id, str):
            index[action_id] = row
    return index


def audit(event_root: Path, passive_path: Path) -> dict[str, Any]:
    completed = load_jsonl(event_root / "turn_events.jsonl")
    pending = []
    pending_root = event_root / "pending_actions"
    if pending_root.exists():
        for path in pending_root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                pending.append(value)
    by_id = {str(row.get("action_event_id")): row for row in pending}
    by_id.update({str(row.get("action_event_id")): row for row in completed})
    events = [row for row in by_id.values() if row.get("policy_mode") == "SHADOW"]
    passive = passive_index(passive_path)
    actual_counts = Counter(
        row.get("actual_treatment_move") or row.get("actual_move") for row in events
    )
    hypothetical_counts = Counter(
        row.get("shadow_hypothetical_move") or row.get("hypothetical_arm") for row in events
    )
    block_availability = Counter()
    schema_complete = 0
    fully_observed = 0
    resolved = []
    for row in events:
        values = context_values(row)
        complete_blocks = {
            block: all(name in values for name in features)
            for block, features in BLOCK_FEATURES.items()
        }
        for block, available in complete_blocks.items():
            block_availability[block] += int(available)
        schema_complete += int(all(complete_blocks.values()))
        fully_observed += int(
            all(complete_blocks.values())
            and values.get("previous_learner_signals_missing") == 0.0
            and values.get("previous_mrb1_missing") == 0.0
        )
        if (
            row.get("outcome_observed") is True
            and row.get("raw_delta") is not None
            and row.get("headroom_normalized_delta") is not None
        ):
            resolved.append(row)

    linked = [passive.get(str(row.get("action_event_id"))) for row in events]
    learners = {
        (item.get("identity") or {}).get("student_pseudonymous_id")
        for item in linked if isinstance(item, dict)
    } - {None}
    skills = {
        (item.get("learning_state_update") or {}).get("skill")
        for item in linked if isinstance(item, dict)
    } - {None}
    return {
        "event_root": str(event_root),
        "collection_lineages": sorted({str(row.get("collection_lineage")) for row in events}),
        "total_shadow_actions": len(events),
        "completed_event_rows": len(completed),
        "pending_action_rows": len(pending),
        "resolved_reward_rows": len(resolved),
        "block_schema_availability": {block: block_availability[block] for block in BLOCK_FEATURES},
        "L_fully_observed_rows": sum(
            context_values(row).get("previous_learner_signals_missing") == 0.0 for row in events
        ),
        "Q_fully_observed_rows": sum(
            context_values(row).get("previous_mrb1_missing") == 0.0 for row in events
        ),
        "schema_complete_S_K_L_H_Q_rows": schema_complete,
        "fully_observed_S_K_L_H_Q_rows": fully_observed,
        "actual_move_counts": {move: actual_counts[move] for move in MOVES},
        "shadow_hypothetical_move_counts": {move: hypothetical_counts[move] for move in MOVES},
        "attempt_count": len({row.get("attempt_id") for row in events}),
        "learner_count_if_linked": len(learners),
        "skill_count_if_linked": len(skills),
        "passive_link_count": sum(item is not None for item in linked),
        "raw_delta": describe([float(row["raw_delta"]) for row in resolved]),
        "headroom_normalized_delta": describe(
            [float(row["headroom_normalized_delta"]) for row in resolved]
        ),
        "automatic_pass_threshold": None,
        "question": "Do we now have enough complete data to rerun context ablation?",
        "answer": "No automatic decision is encoded; inspect the raw completeness, coverage, attempts, and rewards above.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--passive-turn-data", type=Path, default=DEFAULT_PASSIVE)
    args = parser.parse_args()
    print(json.dumps(audit(args.event_root.resolve(), args.passive_turn_data.resolve()), indent=2, allow_nan=False))
