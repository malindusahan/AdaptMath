"""Opt-in JSONL observability for the live pedagogical move selector.

This module only serializes values already produced by the active runtime.  It
must never run inference, sample an arm, calculate a policy update, or persist
scientific state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


DEBUG_ENVIRONMENT_FLAG = "ADAPTMATH_MOVE_SELECTOR_DEBUG"
DEBUG_SCHEMA_VERSION = "adaptmath_move_selector_debug_v1"
_DEBUG_STREAM_PATH = (
    Path(__file__).resolve().parents[4]
    / "_local_runtime_logs"
    / "move_selector_live.jsonl"
)
_WRITE_LOCK = RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def debug_enabled() -> bool:
    """Return whether the explicitly opt-in diagnostic stream is enabled."""

    return os.getenv(DEBUG_ENVIRONMENT_FLAG, "").strip() == "1"


def debug_stream_path() -> Path:
    """Return the repository-local diagnostic stream path."""

    return _DEBUG_STREAM_PATH


def _identity_fields(identity: Mapping[str, object] | None) -> dict[str, object]:
    source = identity if isinstance(identity, Mapping) else {}
    return {
        name: source.get(name)
        for name in (
            "student_id",
            "thread_id",
            "target_skill",
            "supervised_model",
            "checkpoint_identifier",
        )
    }


def _append_event(event: Mapping[str, object]) -> bool:
    """Append one event without allowing diagnostics to affect Tutor behavior."""

    if not debug_enabled():
        return False
    try:
        serialized = json.dumps(
            dict(event),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with _WRITE_LOCK:
            _DEBUG_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _DEBUG_STREAM_PATH.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(serialized + "\n")
                handle.flush()
        return True
    except Exception:
        # Observability is deliberately fail-open.  A disk/serialization error
        # must not change a selector decision, Tutor response, BKT result, or
        # posterior update.
        return False


def emit_pre_action(
    *,
    identity: Mapping[str, object] | None,
    attempt_id: str,
    decision: Any,
    selector_version: str,
    selector_sha256: str,
) -> bool:
    """Serialize the already-selected action and its actual Thompson samples."""

    if not debug_enabled():
        return False
    try:
        context = dict(decision.context_snapshot)
        feature_values = context.get("feature_values")
        features = dict(feature_values) if isinstance(feature_values, Mapping) else {}
        identifiers = _identity_fields(identity)
        event = {
            "schema_version": DEBUG_SCHEMA_VERSION,
            "event_type": "PRE_ACTION",
            "timestamp": _utc_now(),
            **identifiers,
            "attempt_id": attempt_id,
            "action_event_id": decision.action_event_id,
            "turn_index": decision.action_turn_index,
            "supervised_model": (
                identifiers.get("supervised_model") or selector_version
            ),
            "checkpoint_identifier": (
                identifiers.get("checkpoint_identifier") or selector_version
            ),
            "selector_sha256": selector_sha256,
            "md7_probabilities": dict(decision.selector_probabilities),
            "s_features": {
                move: features.get(f"selector_p_{move}")
                for move in ("generic", "probing", "focus", "telling")
            },
            "knowledge_state": {
                "mastery_before": features.get("mastery_before"),
                "previous_mastery_delta": features.get(
                    "previous_mastery_delta"
                ),
                "missing": features.get("previous_mastery_delta_missing"),
            },
            "learner_response": {
                "reasoning": features.get("previous_reasoning_probability"),
                "uncertainty": features.get(
                    "previous_uncertainty_probability"
                ),
                "clarification": features.get(
                    "previous_clarification_probability"
                ),
                "missing": features.get("previous_learner_signals_missing"),
            },
            "context": context,
            # These are copied from the one real select_arm call.  No new
            # sampling occurs for diagnostics.
            "sampled_scores": dict(decision.sampled_scores),
            "selected_move": decision.final_move,
            "selected_arm": decision.selected_arm,
            "policy_mode": decision.policy_mode,
        }
        return _append_event(event)
    except Exception:
        return False


def emit_post_update(
    *,
    identity: Mapping[str, object] | None,
    attempt_id: str,
    decision: Any,
    outcome: Mapping[str, object],
    arm_count_before: int,
    arm_count_after: int,
    mrb1_scores: Mapping[str, object],
    persistence: Mapping[str, object],
) -> bool:
    """Serialize one already-applied and already-persisted LinTS update."""

    if not debug_enabled():
        return False
    try:
        before = float(outcome["mastery_before"])
        after = float(outcome["mastery_after"])
        raw_delta = float(outcome["raw_delta"])
        if raw_delta > 0.0:
            headroom_denominator: float | None = 1.0 - before
        elif raw_delta < 0.0:
            headroom_denominator = before
        else:
            headroom_denominator = None
        identifiers = _identity_fields(identity)
        event = {
            "schema_version": DEBUG_SCHEMA_VERSION,
            "event_type": "POST_UPDATE",
            "timestamp": _utc_now(),
            **identifiers,
            "attempt_id": attempt_id,
            "action_event_id": decision.action_event_id,
            "turn_index": decision.action_turn_index,
            "selected_move": decision.final_move,
            "mastery_before": before,
            "mastery_after": after,
            "raw_mastery_delta": raw_delta,
            "reward_type": "Headroom-normalized mastery delta",
            "headroom_denominator": headroom_denominator,
            "headroom_normalized_reward": outcome.get(
                "headroom_normalized_delta"
            ),
            "final_reward": outcome.get("configured_reward_value"),
            "updated_arm": decision.selected_arm,
            "arm_update_count_before": arm_count_before,
            "arm_update_count_after": arm_count_after,
            "policy_update_count_before": outcome.get(
                "posterior_total_updates_before"
            ),
            "policy_update_count_after": outcome.get(
                "posterior_total_updates_after"
            ),
            "persistence": dict(persistence),
            "mrb1": {
                "scores": dict(mrb1_scores),
                "role": "DIAGNOSTIC ONLY",
            },
        }
        return _append_event(event)
    except Exception:
        return False


__all__ = (
    "DEBUG_ENVIRONMENT_FLAG",
    "DEBUG_SCHEMA_VERSION",
    "debug_enabled",
    "debug_stream_path",
    "emit_post_update",
    "emit_pre_action",
)
