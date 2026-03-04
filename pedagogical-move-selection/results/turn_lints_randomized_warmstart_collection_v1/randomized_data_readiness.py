"""Read-only readiness summary for the real randomized warm-start lineage."""

from __future__ import annotations

import argparse
import json
import math
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
    / "turn_lints_randomized_warmstart_v1"
)
DEFAULT_PASSIVE = (
    MOVE_ROOT / "results" / "md_self_improvement_turn_data_v1" / "turn_outcomes.jsonl"
)
EXPECTED_MODE = "RANDOMIZED_WARMSTART"
EXPECTED_POLICY = "md7_r2_probability_proportional_v1"
MOVES = ("generic", "probing", "focus", "telling")
BLOCK_FEATURES = {
    "S": tuple(f"selector_p_{move}" for move in MOVES),
    "K": (
        "mastery_before",
        "previous_mastery_delta",
        "previous_mastery_delta_missing",
    ),
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def describe(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            **{
                key: None
                for key in (
                    "min", "p1", "p5", "p10", "q1", "median", "mean",
                    "q3", "p90", "p95", "p99", "max", "std",
                )
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
        "p1": quantile(0.01),
        "p5": quantile(0.05),
        "p10": quantile(0.10),
        "q1": quantile(0.25),
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "q3": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "std": statistics.stdev(ordered) if len(ordered) > 1 else None,
    }


def context_values(event: dict[str, Any]) -> dict[str, Any]:
    context = event.get("context") or {}
    direct = context.get("feature_values")
    if isinstance(direct, dict):
        return direct
    names, vector = context.get("feature_names"), context.get("vector")
    if (
        isinstance(names, list)
        and isinstance(vector, list)
        and len(names) == len(vector)
    ):
        return dict(zip(names, vector, strict=True))
    return {}


def passive_index(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        identity = row.get("identity") or {}
        action_id = identity.get("action_event_id")
        if isinstance(action_id, str) and action_id:
            indexed[action_id] = row
    return indexed


def _pending_rows(event_root: Path) -> list[dict[str, Any]]:
    pending_root = event_root / "pending_actions"
    if not pending_root.is_dir():
        return []
    return [load_json(path) for path in sorted(pending_root.glob("*.json"))]


def audit(event_root: Path, passive_path: Path) -> dict[str, Any]:
    manifest = load_json(event_root / "lineage_manifest.json")
    state = load_json(event_root / "policy_state.json")
    completed = load_jsonl(event_root / "turn_events.jsonl")
    pending = _pending_rows(event_root)

    by_id = {
        str(row.get("action_event_id")): row
        for row in pending
        if row.get("action_event_id")
    }
    by_id.update(
        {
            str(row.get("action_event_id")): row
            for row in completed
            if row.get("action_event_id")
        }
    )
    mode_events = [
        row for row in by_id.values()
        if row.get("policy_mode") == EXPECTED_MODE
    ]
    randomized = [
        row for row in mode_events if row.get("randomized_assignment") is True
    ]
    external = [
        row for row in mode_events if row.get("randomized_assignment") is False
    ]
    resolved = [
        row for row in randomized
        if row.get("outcome_observed") is True
        and row.get("headroom_normalized_delta") is not None
    ]

    block_schema = Counter()
    block_fully_observed = Counter()
    full_schema_rows = 0
    full_observed_rows = 0
    for row in randomized:
        values = context_values(row)
        complete = {
            block: all(name in values for name in features)
            for block, features in BLOCK_FEATURES.items()
        }
        for block, available in complete.items():
            block_schema[block] += int(available)
        fully_observed = {
            **complete,
            "L": complete["L"]
            and values.get("previous_learner_signals_missing") == 0.0,
            "Q": complete["Q"]
            and values.get("previous_mrb1_missing") == 0.0,
        }
        for block, available in fully_observed.items():
            block_fully_observed[block] += int(available)
        full_schema_rows += int(all(complete.values()))
        full_observed_rows += int(all(fully_observed.values()))

    passive = passive_index(passive_path)
    linked = [
        passive.get(str(row.get("action_event_id"))) for row in randomized
    ]
    learners = {
        (item.get("identity") or {}).get("student_pseudonymous_id")
        for item in linked if isinstance(item, dict)
    } - {None}
    skills = {
        (item.get("learning_state_update") or {}).get("skill")
        for item in linked if isinstance(item, dict)
    } - {None}
    action_counts = Counter(row.get("actual_treatment_move") for row in randomized)
    resolution_counts = Counter(
        str(row.get("resolution_status") or "pending") for row in randomized
    )
    propensities = [
        float(row["behavior_propensity"])
        for row in randomized
        if row.get("behavior_propensity") is not None
    ]
    reward_values = [float(row["headroom_normalized_delta"]) for row in resolved]
    raw_values = [float(row["raw_delta"]) for row in resolved]
    posterior_event_updates = sum(
        row.get("posterior_updated") is True for row in mode_events
    )
    behavior_policy_mismatches = sum(
        row.get("behavior_policy") != EXPECTED_POLICY for row in randomized
    )
    propensity_contract_violations = 0
    for row in randomized:
        move = row.get("actual_treatment_move")
        propensity = row.get("behavior_propensity")
        vector = row.get("full_behavior_probability_vector")
        valid = (
            move in MOVES
            and isinstance(propensity, (int, float))
            and not isinstance(propensity, bool)
            and isinstance(vector, dict)
            and set(vector) == set(MOVES)
        )
        if valid:
            vector_values = [vector[item] for item in MOVES]
            valid = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in vector_values
            )
        if valid:
            valid = (
                math.isclose(
                    sum(float(value) for value in vector_values),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
                and math.isclose(
                    float(propensity),
                    float(vector[str(move)]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        propensity_contract_violations += int(not valid)

    return {
        "schema_version": "adaptmath_randomized_warmstart_readiness_v1",
        "event_root": str(event_root),
        "event_log": str(event_root / "turn_events.jsonl"),
        "lineage": {
            "manifest_schema": manifest.get("schema_version"),
            "mode": manifest.get("mode"),
            "behavior_policy": manifest.get("behavior_policy"),
            "selector_version": manifest.get("selector_version"),
            "selector_sha256": manifest.get("selector_sha256"),
            "reward_mode": manifest.get("reward_mode"),
            "context_schema_id": manifest.get("context_schema_id"),
            "context_blocks": manifest.get("context_blocks"),
            "context_dimension": manifest.get("context_dimension"),
            "posterior_updates_enabled": manifest.get(
                "posterior_updates_enabled"
            ),
            "data_mode": state.get("data_mode"),
        },
        "counts": {
            "total_randomized_actions": len(randomized),
            "resolved_reward_observations": len(resolved),
            "attempt_count": len(
                {row.get("attempt_id") for row in randomized}
            ),
            "learner_count": len(learners),
            "skill_count": len(skills),
            "completed_event_rows": len(completed),
            "pending_action_rows": len(pending),
            "passive_link_count": sum(item is not None for item in linked),
            "external_nonrandomized_override_count": len(external),
            "behavior_policy_mismatch_count": behavior_policy_mismatches,
            "propensity_contract_violation_count": (
                propensity_contract_violations
            ),
            "posterior_updated_event_count": posterior_event_updates,
            "posterior_state_update_count": int(state.get("update_count", 0)),
        },
        "context_rows_with_complete_schema": {
            **{block: block_schema[block] for block in BLOCK_FEATURES},
            "S+K+L+H+Q": full_schema_rows,
        },
        "context_rows_fully_observed": {
            **{block: block_fully_observed[block] for block in BLOCK_FEATURES},
            "S+K+L+H+Q": full_observed_rows,
        },
        "actual_randomized_action_counts": {
            move: action_counts[move] for move in MOVES
        },
        "selected_action_propensity": describe(propensities),
        "headroom_normalized_reward": describe(reward_values),
        "raw_delta_secondary_robustness": describe(raw_values),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "automatic_stopping_threshold": None,
        "readiness_interpretation": (
            "No automatic sufficiency decision is encoded. Inspect action support, "
            "context completeness, grouped attempts, learners, skills, outcome "
            "status, propensities, and rewards."
        ),
        "causal_warning": (
            "This is descriptive collection readiness, not a causal-effect, "
            "reward-learning, or context-ablation result."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--passive-turn-data", type=Path, default=DEFAULT_PASSIVE)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.event_root.resolve(), args.passive_turn_data.resolve()),
            indent=2,
            allow_nan=False,
        )
    )
