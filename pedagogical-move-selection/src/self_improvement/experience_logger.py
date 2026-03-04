"""Append-only trajectory logging for the canonical turn-level runtime.

This logger preserves sufficient trajectory information for reproducibility
and later causal or temporal audits. It records what the current candidate
policy actually did and observed; it does not choose arms, update LinTS,
compute reward, run MD6 or MRB1, recompute mastery, or change controller state.

The records support later checks that current MD6 probabilities drove each
turn, only earlier-response MRB1 quality appeared in its context, current-turn
MRB1 arrived afterward, eligible arms preceded selection, and terminal
outcomes were separated from turn decisions. The logger preserves evidence;
the controller remains responsible for enforcing causal update timing.

Logging does not establish educational efficacy, an optimal context, reward,
threshold, or delayed-credit scheme. It performs no policy-state persistence
and never writes posterior matrices, reward vectors, model weights, or
tokenizer contents into attempt records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Final

import numpy as np

from .conservative_overlay import DEFAULT_GAP_THRESHOLD
from .context_builder import MOVE_ORDER
from .lints_policy import (
    ARMS,
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    TrueDisjointLinTS,
)
from .reward import (
    LearningOutcome,
    validate_learning_outcome,
    validate_signed_reward,
)
from .turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
    TutorQualitySnapshot,
)
from .turn_level_controller import AttemptCompletion, CompletedTurn, TurnDecision


SCHEMA_VERSION: Final[str] = "turn_lints_v3"
CREDIT_SCHEME: Final[str] = "equal_normalized"
CONTEXT_NAME: Final[str] = "C3"
REWARD_NAME: Final[str] = "mastery_delta"
SELECTION_MODE: Final[str] = "fresh Thompson sample per tutor turn"
UPDATE_TIMING: Final[str] = "after attempt completion only"
DATA_MODES: Final[tuple[str, ...]] = ("synthetic", "real")

_FORBIDDEN_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "A",
        "b",
        "model_weights",
        "policy_state",
        "state_dict",
        "tokenizer_contents",
    }
)

_REQUIRED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "attempt_id",
        "data_mode",
        "policy_config",
        "learning_outcome",
        "mastery_before",
        "mastery_after",
        "mastery_delta",
        "reward",
        "num_turns",
        "credit_scheme",
        "credit_weight_per_turn",
        "total_credit_weight",
        "final_average_mrb1",
        "turns",
        "controller_counters",
    }
)


def _validate_data_mode(value: object) -> str:
    if not isinstance(value, str) or value not in DATA_MODES:
        raise ValueError("data_mode must be exactly 'synthetic' or 'real'.")
    return value


def _validate_source_policy(
    source_policy: object,
    data_mode: str,
) -> TrueDisjointLinTS:
    """Bind a record mode to the public mode of its source policy."""

    if not isinstance(source_policy, TrueDisjointLinTS):
        raise TypeError("source_policy must be a TrueDisjointLinTS instance.")
    if source_policy.data_mode != data_mode:
        raise ValueError(
            "Record data_mode must match source_policy.data_mode; "
            f"record={data_mode!r}, policy={source_policy.data_mode!r}."
        )
    if source_policy.context_dim != len(TURN_FEATURE_NAMES):
        raise ValueError(
            "source_policy must use the canonical turn context dimension "
            f"{len(TURN_FEATURE_NAMES)}."
        )
    if source_policy.ridge_lambda != DEFAULT_RIDGE_LAMBDA:
        raise ValueError("source_policy must use ridge_lambda=1.0.")
    if source_policy.exploration_scale != DEFAULT_EXPLORATION_SCALE:
        raise ValueError("source_policy must use exploration_scale=0.20.")
    return source_policy


def _policy_config(policy: TrueDisjointLinTS) -> dict[str, object]:
    """Return the explicit final-candidate configuration for one record."""

    return {
        "context": {
            "name": CONTEXT_NAME,
            "dimension": len(TURN_FEATURE_NAMES),
            "feature_order": list(TURN_FEATURE_NAMES),
        },
        "gap_threshold": DEFAULT_GAP_THRESHOLD,
        "reward": {
            "name": REWARD_NAME,
            "source": "delta_mastery",
            "formula": "mastery_after - mastery_before",
            "range": [-1.0, 1.0],
        },
        "credit": {
            "name": CREDIT_SCHEME,
            "sample_weight": "1/T",
        },
        "ridge_lambda": policy.ridge_lambda,
        "exploration_scale": policy.exploration_scale,
        "selection": SELECTION_MODE,
        "update_timing": UPDATE_TIMING,
    }


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} could not be represented as a float.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _unit_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}.")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive non-boolean integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative non-boolean integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}.")
    return result


def _json_compatible(value: object, path: str = "value") -> object:
    """Recursively copy supported values into finite JSON-native objects."""

    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), path)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float(path, value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _finite_float(path, value)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings.")
            converted[key] = _json_compatible(item, f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_compatible(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} is not JSON-compatible: {type(value).__name__}.")


def _reject_forbidden_metadata(value: object, path: str = "metadata") -> None:
    """Prevent optional metadata from embedding heavyweight learned state."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(
                    f"{path} cannot contain policy/model payload key {key!r}."
                )
            _reject_forbidden_metadata(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_forbidden_metadata(item, f"{path}[{index}]")


def validate_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return a validated independent JSON-compatible metadata mapping."""

    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a string-keyed mapping or None.")
    _reject_forbidden_metadata(metadata)
    converted = _json_compatible(metadata, "metadata")
    if not isinstance(converted, dict):
        raise RuntimeError("Internal metadata validation did not produce a dict.")
    return converted


def _validate_attempt_id(value: object) -> str | int:
    converted = _json_compatible(value, "attempt_id")
    if isinstance(converted, bool) or not isinstance(converted, (str, int)):
        raise TypeError("attempt_id must be a non-empty string or integer.")
    if isinstance(converted, str) and not converted.strip():
        raise ValueError("attempt_id cannot be empty or whitespace.")
    return converted


def _exact_float_mapping(
    name: str,
    values: Mapping[str, object],
    expected_keys: tuple[str, ...],
    *,
    unit_interval: bool,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if set(values) != set(expected_keys):
        raise ValueError(f"{name} must contain exactly {list(expected_keys)}.")
    validator = _unit_float if unit_interval else _finite_float
    return {
        key: validator(f"{name}.{key}", values[key]) for key in expected_keys
    }


def _serialize_turn(
    completed_turn: CompletedTurn,
    expected_turn_index: int,
) -> dict[str, object]:
    if not isinstance(completed_turn, CompletedTurn):
        raise TypeError("completion.turns must contain CompletedTurn values.")
    decision = completed_turn.decision
    if not isinstance(decision, TurnDecision):
        raise TypeError("CompletedTurn.decision must be a TurnDecision.")

    turn_index = _positive_integer("turn_index", decision.turn_index)
    if turn_index != expected_turn_index:
        raise ValueError(
            "Turn records must be chronological and one-indexed; "
            f"expected {expected_turn_index}, got {turn_index}."
        )

    if not isinstance(decision.context, Sequence) or isinstance(
        decision.context,
        (str, bytes),
    ):
        raise TypeError("Turn context must be a numeric sequence.")
    context = [
        _finite_float(f"turn[{turn_index}].context[{index}]", value)
        for index, value in enumerate(decision.context)
    ]
    if len(context) != len(TURN_FEATURE_NAMES):
        raise ValueError(
            f"Turn context must have length {len(TURN_FEATURE_NAMES)}."
        )

    probabilities = _exact_float_mapping(
        f"turn[{turn_index}].md6_probabilities",
        decision.md6_probabilities,
        MOVE_ORDER,
        unit_interval=True,
    )
    if not isclose(
        sum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError("Turn MD6 probabilities must sum approximately to 1.")

    if not isinstance(decision.eligible_arms, Sequence) or isinstance(
        decision.eligible_arms,
        (str, bytes),
    ):
        raise TypeError("eligible_arms must be a non-empty arm sequence.")
    supplied_eligible = tuple(decision.eligible_arms)
    if not supplied_eligible:
        raise ValueError("eligible_arms cannot be empty.")
    if any(arm not in ARMS for arm in supplied_eligible):
        raise ValueError("eligible_arms contains an unknown canonical arm.")
    if len(set(supplied_eligible)) != len(supplied_eligible):
        raise ValueError("eligible_arms cannot contain duplicates.")
    canonical_eligible = tuple(arm for arm in ARMS if arm in supplied_eligible)
    if supplied_eligible != canonical_eligible:
        raise ValueError("eligible_arms must preserve canonical arm order.")
    if canonical_eligible[0] != "baseline":
        raise ValueError("baseline must be the first eligible arm.")

    sampled_scores = _exact_float_mapping(
        f"turn[{turn_index}].sampled_scores",
        decision.sampled_scores,
        ARMS,
        unit_interval=False,
    )
    if decision.selected_arm not in canonical_eligible:
        raise ValueError("selected_arm must be one of the recorded eligible arms.")
    if decision.base_move not in MOVE_ORDER:
        raise ValueError("base_move is not a canonical pedagogical move.")
    if decision.final_move not in MOVE_ORDER:
        raise ValueError("final_move is not a canonical pedagogical move.")
    if decision.target_move is not None and decision.target_move not in MOVE_ORDER:
        raise ValueError("target_move is not a canonical pedagogical move.")
    if type(decision.overridden) is not bool:
        raise TypeError("overridden must be a Python bool.")

    gap = _unit_float(f"turn[{turn_index}].gap", decision.gap)
    gap_threshold = _unit_float(
        f"turn[{turn_index}].gap_threshold",
        decision.gap_threshold,
    )
    if not isclose(
        gap_threshold,
        DEFAULT_GAP_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Turn gap_threshold must equal the canonical 0.10.")
    base_probability = _unit_float(
        f"turn[{turn_index}].base_probability",
        decision.base_probability,
    )
    target_probability = (
        None
        if decision.target_probability is None
        else _unit_float(
            f"turn[{turn_index}].target_probability",
            decision.target_probability,
        )
    )
    mrb1_scores = _exact_float_mapping(
        f"turn[{turn_index}].mrb1_scores",
        completed_turn.mrb1_scores,
        MRB1_TASKS,
        unit_interval=True,
    )

    return {
        "turn_index": turn_index,
        "context": context,
        "md6_probabilities": probabilities,
        "eligible_arms": list(canonical_eligible),
        "sampled_scores": sampled_scores,
        "selected_arm": decision.selected_arm,
        "base_move": decision.base_move,
        "final_move": decision.final_move,
        "target_move": decision.target_move,
        "overridden": decision.overridden,
        "gap": gap,
        "gap_threshold": gap_threshold,
        "base_probability": base_probability,
        "target_probability": target_probability,
        "mrb1_scores": mrb1_scores,
    }


def serialize_attempt(
    attempt_id: object,
    completion: AttemptCompletion,
    learning_outcome: LearningOutcome | Mapping[str, object],
    data_mode: str,
    *,
    source_policy: TrueDisjointLinTS,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize one actual immutable controller completion without mutation.

    Every value is copied from the canonical completion and its nested turn
    decisions. The function validates consistency but does not recompute an
    arm, overlay, reward, mastery outcome, or MRB1 score. The required
    learning outcome is independently validated and checked against the
    controller completion before it is preserved.
    """

    identifier = _validate_attempt_id(attempt_id)
    mode = _validate_data_mode(data_mode)
    validated_policy = _validate_source_policy(source_policy, mode)
    if not isinstance(completion, AttemptCompletion):
        raise TypeError("completion must be an AttemptCompletion.")
    validated_outcome = validate_learning_outcome(learning_outcome)

    turn_count = _positive_integer("completion.turn_count", completion.turn_count)
    if not isinstance(completion.turns, tuple):
        raise TypeError("completion.turns must be the immutable controller tuple.")
    if len(completion.turns) != turn_count:
        raise ValueError("completion.turn_count must equal len(completion.turns).")
    turns = [
        _serialize_turn(turn, expected_turn_index=index)
        for index, turn in enumerate(completion.turns, start=1)
    ]

    mastery_before = _unit_float("mastery_before", completion.mastery_before)
    mastery_after = _unit_float("mastery_after", completion.mastery_after)
    mastery_delta = validate_signed_reward(completion.mastery_delta)
    reward = validate_signed_reward(completion.reward)
    consistency_checks = (
        (
            "learning_outcome.mastery_before",
            validated_outcome.mastery_before,
            mastery_before,
        ),
        (
            "learning_outcome.mastery_after",
            validated_outcome.mastery_after,
            mastery_after,
        ),
        (
            "learning_outcome.delta_mastery",
            validated_outcome.delta_mastery,
            mastery_delta,
        ),
        ("completion.reward", reward, mastery_delta),
        ("learning_outcome reward", validated_outcome.delta_mastery, reward),
    )
    for name, actual, expected in consistency_checks:
        if not isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} is inconsistent with the v3 completion.")

    credit_weight = _unit_float(
        "sample_weight_per_turn",
        completion.sample_weight_per_turn,
    )
    if credit_weight <= 0.0:
        raise ValueError("sample_weight_per_turn must be greater than zero.")
    expected_weight = 1.0 / turn_count
    if not isclose(
        credit_weight,
        expected_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("sample_weight_per_turn must equal 1 / turn_count.")
    total_credit_weight = _finite_float(
        "total_attempt_weight",
        completion.total_attempt_weight,
    )
    if not isclose(
        total_credit_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("total_attempt_weight must be approximately 1.")

    if not isinstance(completion.final_quality, TutorQualitySnapshot):
        raise TypeError("completion.final_quality must be a TutorQualitySnapshot.")
    if completion.final_quality.count != turn_count:
        raise ValueError("Final MRB1 quality count must equal turn_count.")
    final_average_mrb1 = _exact_float_mapping(
        "final_average_mrb1",
        completion.final_quality.as_task_mapping(),
        MRB1_TASKS,
        unit_interval=True,
    )

    controller_completed_attempts = _positive_integer(
        "completion.completed_attempts",
        completion.completed_attempts,
    )
    weighted_turn_updates = _positive_integer(
        "completion.number_of_weighted_turn_updates",
        completion.number_of_weighted_turn_updates,
    )
    if weighted_turn_updates < turn_count:
        raise ValueError(
            "number_of_weighted_turn_updates cannot be smaller than turn_count."
        )

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": identifier,
        "data_mode": mode,
        "policy_config": _policy_config(validated_policy),
        "learning_outcome": validated_outcome.as_mapping(),
        "mastery_before": mastery_before,
        "mastery_after": mastery_after,
        "mastery_delta": mastery_delta,
        "reward": reward,
        "num_turns": turn_count,
        "credit_scheme": CREDIT_SCHEME,
        "credit_weight_per_turn": credit_weight,
        "total_credit_weight": total_credit_weight,
        "final_average_mrb1": final_average_mrb1,
        "turns": turns,
        "controller_counters": {
            "completed_attempts": controller_completed_attempts,
            "number_of_weighted_turn_updates": weighted_turn_updates,
        },
    }

    diagnostics = validated_outcome.diagnostics_mapping()
    if diagnostics:
        record["diagnostics"] = diagnostics

    validated_metadata = validate_metadata(metadata)
    if validated_metadata is not None:
        record["metadata"] = validated_metadata

    compatible = _json_compatible(record, "attempt_record")
    if not isinstance(compatible, dict):
        raise RuntimeError("Internal attempt serialization did not produce a dict.")
    json.dumps(compatible, ensure_ascii=False, allow_nan=False)
    return compatible


class ExperienceLogger:
    """Append one canonical completed attempt per UTF-8 JSONL line.

    The source policy is required solely to bind the explicit logger
    ``data_mode`` to the public policy mode. The logger never reads or writes
    policy posterior state and never changes its RNG or counters.
    """

    def __init__(
        self,
        path: str | Path,
        data_mode: str,
        *,
        source_policy: TrueDisjointLinTS,
    ) -> None:
        self.path = Path(path)
        self.data_mode = _validate_data_mode(data_mode)
        self._source_policy = _validate_source_policy(
            source_policy,
            self.data_mode,
        )
        self._validate_existing_log()

    def _validate_source_mode(self) -> None:
        if self._source_policy.data_mode != self.data_mode:
            raise ValueError("Source policy data_mode changed after logger creation.")

    def _validate_existing_log(self) -> None:
        from .postgres_repository import postgres_enabled
        if postgres_enabled():
            return
        if not self.path.exists():
            return
        if not self.path.is_file():
            raise ValueError(f"JSONL path is not a regular file: {self.path}.")

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"Existing JSONL contains a blank line at {line_number}."
                        )
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Existing JSONL line {line_number} is malformed."
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Existing JSONL line {line_number} is not an object."
                        )
                    _json_compatible(record, f"existing line {line_number}")
                    if record.get("schema_version") != SCHEMA_VERSION:
                        raise ValueError(
                            f"Existing JSONL line {line_number} has an "
                            "incompatible schema version."
                        )
                    missing = _REQUIRED_RECORD_FIELDS - set(record)
                    if missing:
                        raise ValueError(
                            f"Existing JSONL line {line_number} is missing fields: "
                            f"{sorted(missing)}."
                        )
                    if record["policy_config"] != _policy_config(
                        self._source_policy
                    ):
                        raise ValueError(
                            f"Existing JSONL line {line_number} has an "
                            "incompatible final-candidate policy config."
                        )
                    existing_mode = _validate_data_mode(record["data_mode"])
                    if existing_mode != self.data_mode:
                        raise ValueError(
                            "Cannot mix synthetic and real attempt records: "
                            f"logger={self.data_mode!r}, existing={existing_mode!r}."
                        )
        except UnicodeError as exc:
            raise ValueError("Existing JSONL is not valid UTF-8.") from exc

    def append_attempt(
        self,
        attempt_id: object,
        completion: AttemptCompletion,
        learning_outcome: LearningOutcome | Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Append exactly one validated attempt and return its serialized copy."""

        self._validate_source_mode()
        self._validate_existing_log()
        record = serialize_attempt(
            attempt_id=attempt_id,
            completion=completion,
            learning_outcome=learning_outcome,
            data_mode=self.data_mode,
            source_policy=self._source_policy,
            metadata=metadata,
        )
        line = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        from .postgres_repository import append_experience, postgres_enabled
        if postgres_enabled():
            append_experience(record)
            return record

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
        return record


__all__ = (
    "CONTEXT_NAME",
    "CREDIT_SCHEME",
    "DATA_MODES",
    "ExperienceLogger",
    "REWARD_NAME",
    "SCHEMA_VERSION",
    "SELECTION_MODE",
    "UPDATE_TIMING",
    "serialize_attempt",
    "validate_metadata",
)
