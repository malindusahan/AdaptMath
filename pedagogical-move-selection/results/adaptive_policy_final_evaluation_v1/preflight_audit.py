"""Hard-gate audit for the requested final adaptive-policy evaluation.

This script is intentionally read-only with respect to the authoritative
turn-outcome ledger.  It writes only a preflight report beside this script and
exits non-zero unless the requested final cohort and leakage constraints hold.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = OUTPUT_DIR.parents[2]
CANONICAL_TURN_TABLE = (
    WORKSPACE_ROOT
    / "pedagogical-move-selection"
    / "results"
    / "md_self_improvement_turn_data_v1"
    / "turn_outcomes.jsonl"
)
AUDIT_PATH = OUTPUT_DIR / "preflight_audit.json"

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
EXPECTED_FEATURES = tuple(
    feature for block in ("S", "K", "L", "H", "Q")
    for feature in BLOCK_FEATURES[block]
)
EXPECTED_SELECTOR_SHA256 = (
    "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
)
EXPECTED_LINEAGE = "direct_disjoint_turn_lints_v1"
EXPECTED_SCHEMA = "adaptmath_turn_outcome_v1"


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def load_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw_lines: list[str] = []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw_lines.append(line.rstrip("\r\n"))
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return raw_lines, rows


def duplicate_summary(values: list[Any]) -> dict[str, int]:
    counts = Counter(value for value in values if value not in (None, ""))
    return {
        "duplicate_groups": sum(count > 1 for count in counts.values()),
        "extra_records": sum(count - 1 for count in counts.values() if count > 1),
    }


def inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "turns": len(rows),
        "learners": len({nested(row, "identity", "student_pseudonymous_id") for row in rows}),
        "attempts": len({nested(row, "identity", "attempt_id") for row in rows}),
        "skills": len({nested(row, "state_before_action", "target_skill") for row in rows}),
        "action_counts": dict(sorted(Counter(
            nested(row, "adaptive_decision", "final_move") for row in rows
        ).items(), key=lambda item: str(item[0]))),
    }


def full_context(row: dict[str, Any]) -> dict[str, Any] | None:
    context = nested(row, "adaptive_decision", "diagnostic_logged_context")
    return context if isinstance(context, dict) else None


def valid_reward(row: dict[str, Any]) -> bool:
    learning = row.get("learning_state_update")
    if not isinstance(learning, dict):
        return False
    before = learning.get("mastery_before")
    after = learning.get("mastery_after")
    delta = learning.get("delta_mastery")
    return (
        nested(row, "next_learner_observation", "status") == "observed_update"
        and learning.get("should_update") is True
        and all(finite_number(value) for value in (before, after, delta))
        and abs(float(delta) - (float(after) - float(before))) <= 1e-12
    )


def main() -> int:
    if not CANONICAL_TURN_TABLE.exists():
        raise FileNotFoundError(f"Canonical turn table missing: {CANONICAL_TURN_TABLE}")
    raw_lines, rows = load_rows(CANONICAL_TURN_TABLE)

    source_inventory = inventory(rows)
    contexts = [context for row in rows if (context := full_context(row)) is not None]
    full_context_rows = [
        row for row in rows
        if (context := full_context(row)) is not None
        and tuple(context.get("feature_names", ())) == EXPECTED_FEATURES
        and tuple(context.get("enabled_blocks", ())) == ("S", "K", "L", "H", "Q")
        and context.get("context_dimension") == len(EXPECTED_FEATURES)
    ]
    strict_rows = [
        row for row in full_context_rows
        if row.get("schema_version") == EXPECTED_SCHEMA
        and nested(row, "provenance", "data_mode") == "real"
        and nested(row, "provenance", "lints_policy_lineage") == EXPECTED_LINEAGE
        and nested(row, "selector", "model_sha256") == EXPECTED_SELECTOR_SHA256
        and nested(row, "adaptive_decision", "final_move") in MOVES
        and valid_reward(row)
    ]

    action_ids = [nested(row, "identity", "action_event_id") for row in rows]
    resolver_ids = [nested(row, "learning_state_update", "resolver_event_id") for row in rows]
    attempt_turn_ids = [
        (nested(row, "identity", "attempt_id"), nested(row, "identity", "action_turn_index"))
        for row in rows
    ]

    exact_line_duplicates = duplicate_summary(raw_lines)
    duplicates = {
        "action_event_id": duplicate_summary(action_ids),
        "resolver_event_id": duplicate_summary(resolver_ids),
        "attempt_id_plus_turn_index": duplicate_summary(attempt_turn_ids),
        "exact_jsonl_line": exact_line_duplicates,
    }

    context_name_counts = Counter()
    unexpected_context_features = Counter()
    post_action_context_features = Counter()
    forbidden_tokens = (
        "mastery_after", "current_", "reward", "delta_mastery",
        "learner_response", "evaluator", "outcome", "correctness",
        "tutor_response",
    )
    for context in contexts:
        names = tuple(str(name) for name in context.get("feature_names", ()))
        context_name_counts["+".join(context.get("enabled_blocks", ())) or "unknown"] += 1
        for name in names:
            if name not in EXPECTED_FEATURES:
                unexpected_context_features[name] += 1
            if any(token in name.lower() for token in forbidden_tokens):
                post_action_context_features[name] += 1

    feature_missingness: dict[str, dict[str, int]] = {}
    for feature in EXPECTED_FEATURES:
        absent = 0
        null = 0
        for row in full_context_rows:
            values = full_context(row).get("feature_values", {})  # type: ignore[union-attr]
            if feature not in values:
                absent += 1
            elif values[feature] is None:
                null += 1
        feature_missingness[feature] = {"absent": absent, "null": null}

    semantic_missingness: dict[str, dict[str, int]] = {}
    for indicator in (
        "previous_mastery_delta_missing",
        "previous_learner_signals_missing",
        "previous_move_missing",
        "previous_mrb1_missing",
    ):
        values = [
            full_context(row)["feature_values"][indicator]
            for row in full_context_rows
        ]
        semantic_missingness[indicator] = {
            "missing_1": sum(float(value) == 1.0 for value in values),
            "observed_0": sum(float(value) == 0.0 for value in values),
            "invalid_other": sum(float(value) not in (0.0, 1.0) for value in values),
        }

    l_indicator = [
        float(full_context(row)["feature_values"]["previous_learner_signals_missing"])
        for row in full_context_rows
    ]
    l_observed = sum(value == 0.0 for value in l_indicator)
    l_missing = sum(value == 1.0 for value in l_indicator)

    target_checks = {
        "turn_count_approximately_700": 650 <= source_inventory["turns"] <= 750,
        "attempt_count_exactly_90": source_inventory["attempts"] == 90,
        "learner_count_exactly_30": source_inventory["learners"] == 30,
        "skill_count_exactly_20": source_inventory["skills"] == 20,
        "strict_integrity_rows_preserve_required_counts": (
            inventory(strict_rows)["attempts"] == 90
            and inventory(strict_rows)["learners"] == 30
            and inventory(strict_rows)["skills"] == 20
        ),
        "no_duplicate_ids": all(
            detail["extra_records"] == 0 for detail in duplicates.values()
        ),
        "no_unexpected_full_context_features": not unexpected_context_features,
        "no_current_action_or_post_action_context_features": not post_action_context_features,
    }
    failed_checks = [name for name, passed in target_checks.items() if not passed]

    report = {
        "status": "PASS" if not failed_checks else "FAIL",
        "failure_policy": "No evaluation, CV, bootstrap, figure, or canonical-table rewrite after a failed preflight gate.",
        "canonical_turn_table": str(CANONICAL_TURN_TABLE.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "canonical_sha256": hashlib.sha256(CANONICAL_TURN_TABLE.read_bytes()).hexdigest(),
        "source_inventory": source_inventory,
        "strict_final_lineage_inventory": inventory(strict_rows),
        "strict_final_lineage_criteria": {
            "schema_version": EXPECTED_SCHEMA,
            "data_mode": "real",
            "policy_lineage": EXPECTED_LINEAGE,
            "selector_sha256": EXPECTED_SELECTOR_SHA256,
            "full_context_features": list(EXPECTED_FEATURES),
            "valid_linked_bkt_update": True,
        },
        "lineage_counts": dict(Counter(
            nested(row, "provenance", "lints_policy_lineage") for row in rows
        )),
        "schema_counts": dict(Counter(row.get("schema_version") for row in rows)),
        "observation_status_counts": dict(Counter(
            nested(row, "next_learner_observation", "status") for row in rows
        )),
        "valid_reward_rows": sum(valid_reward(row) for row in rows),
        "context": {
            "recorded_diagnostic_context_rows": len(contexts),
            "exact_full_S_K_L_H_Q_rows": len(full_context_rows),
            "block_counts": dict(context_name_counts),
            "feature_missingness_on_exact_full_rows": feature_missingness,
            "semantic_missingness_on_exact_full_rows": semantic_missingness,
            "L_coverage": {
                "denominator": len(full_context_rows),
                "observed_previous_signal_rows": l_observed,
                "missing_previous_signal_rows": l_missing,
                "observed_fraction": (l_observed / len(full_context_rows)) if full_context_rows else None,
            },
            "unexpected_features": dict(unexpected_context_features),
            "current_action_or_post_action_features": dict(post_action_context_features),
        },
        "duplicates": duplicates,
        "target_checks": target_checks,
        "failed_checks": failed_checks,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("FINAL ADAPTIVE-POLICY DATA PREFLIGHT")
    print(json.dumps({
        "status": report["status"],
        "source_inventory": source_inventory,
        "strict_final_lineage_inventory": report["strict_final_lineage_inventory"],
        "L_coverage": report["context"]["L_coverage"],
        "duplicates": duplicates,
        "unexpected_context_features": report["context"]["unexpected_features"],
        "post_action_context_features": report["context"]["current_action_or_post_action_features"],
        "failed_checks": failed_checks,
    }, indent=2, sort_keys=True))
    if failed_checks:
        print(
            "FATAL: canonical data do not match the required final analysis lineage; "
            "evaluation was not run and the canonical table was not modified.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
