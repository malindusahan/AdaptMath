"""Causally ordered, block-configurable context for direct-arm Turn-LinTS."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Final

import numpy as np

from .context_builder import MOVE_ORDER
from .turn_context_builder import MRB1_TASKS


CONTEXT_SCHEMA_VERSION: Final[str] = "turn_lints_context_v1"
BLOCK_ORDER: Final[tuple[str, ...]] = ("S", "K", "L", "H", "Q")
CONTEXT_PRESETS: Final[tuple[str, ...]] = (
    "S",
    "S+K",
    "S+L",
    "S+K+L",
    "S+K+L+H",
    "S+K+L+H+Q",
)

BLOCK_FEATURE_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "S": tuple(f"selector_p_{move}" for move in MOVE_ORDER),
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
        *(f"previous_move_{move}" for move in MOVE_ORDER),
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

LEARNER_SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "reasoning_probability",
    "uncertainty_probability",
    "clarification_probability",
)


def _unit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number.")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return result


def _signed_unit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number.")
    result = float(value)
    if not isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [-1, 1].")
    return result


def parse_context_blocks(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return a non-empty valid block subset in canonical block order."""

    if isinstance(value, str):
        supplied = tuple(part.strip().upper() for part in value.split("+"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        supplied = tuple(str(part).strip().upper() for part in value)
    else:
        raise TypeError("context blocks must be a '+' preset or sequence.")
    if not supplied or any(not part for part in supplied):
        raise ValueError("At least one context block is required.")
    if len(set(supplied)) != len(supplied):
        raise ValueError("Context blocks cannot contain duplicates.")
    unknown = set(supplied) - set(BLOCK_ORDER)
    if unknown:
        raise ValueError(f"Unknown context blocks: {sorted(unknown)}.")
    return tuple(block for block in BLOCK_ORDER if block in supplied)


def feature_names_for_blocks(blocks: str | Sequence[str]) -> tuple[str, ...]:
    enabled = parse_context_blocks(blocks)
    return tuple(name for block in enabled for name in BLOCK_FEATURE_NAMES[block])


def context_schema_id(blocks: str | Sequence[str]) -> str:
    enabled = parse_context_blocks(blocks)
    payload = json.dumps(
        {"version": CONTEXT_SCHEMA_VERSION, "blocks": enabled,
         "features": feature_names_for_blocks(enabled)},
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{CONTEXT_SCHEMA_VERSION}:{'+'.join(enabled)}:{suffix}"


@dataclass(frozen=True, slots=True)
class TurnContext:
    schema_version: str
    schema_id: str
    enabled_blocks: tuple[str, ...]
    available_blocks: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    missingness: Mapping[str, str]

    @property
    def dimension(self) -> int:
        return len(self.values)

    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_id": self.schema_id,
            "enabled_blocks": list(self.enabled_blocks),
            "available_blocks": list(self.available_blocks),
            "feature_names": list(self.feature_names),
            "context_dimension": self.dimension,
            "vector": list(self.values),
            "feature_values": dict(zip(self.feature_names, self.values, strict=True)),
            "missingness": dict(self.missingness),
        }


def build_turn_lints_context(
    *,
    enabled_blocks: str | Sequence[str],
    selector_probabilities: Mapping[str, object],
    mastery_before: object,
    previous_mastery_delta: object | None,
    previous_learner_signals: Mapping[str, object] | None,
    previous_move: str | None,
    prior_tutor_turn_count: object,
    previous_mrb1_scores: Mapping[str, object] | None,
) -> TurnContext:
    """Build one context exclusively from values available before selection."""

    blocks = parse_context_blocks(enabled_blocks)
    if set(selector_probabilities) != set(MOVE_ORDER):
        raise ValueError("Selector probabilities must contain exactly four moves.")
    selector = [_unit(f"selector probability {move}", selector_probabilities[move])
                for move in MOVE_ORDER]
    if abs(sum(selector) - 1.0) > 1e-5:
        raise ValueError("Selector probabilities must sum to one.")
    mastery = _unit("mastery_before", mastery_before)
    if isinstance(prior_tutor_turn_count, bool) or not isinstance(
        prior_tutor_turn_count, Integral
    ):
        raise TypeError("prior_tutor_turn_count must be a nonnegative integer.")
    prior_count = int(prior_tutor_turn_count)
    if prior_count < 0:
        raise ValueError("prior_tutor_turn_count cannot be negative.")

    values_by_block: dict[str, list[float]] = {"S": selector}
    missingness: dict[str, str] = {
        "previous_mastery_delta": "zero_with_missing_indicator",
        "previous_learner_signals": "zeros_with_shared_missing_indicator",
        "previous_move": "zero_one_hot_with_missing_indicator",
        "previous_mrb1_scores": "zeros_with_shared_missing_indicator",
    }

    if previous_mastery_delta is None:
        values_by_block["K"] = [mastery, 0.0, 1.0]
    else:
        values_by_block["K"] = [
            mastery,
            _signed_unit("previous_mastery_delta", previous_mastery_delta),
            0.0,
        ]

    if previous_learner_signals is None:
        values_by_block["L"] = [0.0, 0.0, 0.0, 1.0]
    else:
        if set(previous_learner_signals) != set(LEARNER_SIGNAL_NAMES):
            raise ValueError("Previous learner signals have incompatible fields.")
        values_by_block["L"] = [
            *(_unit(name, previous_learner_signals[name])
              for name in LEARNER_SIGNAL_NAMES),
            0.0,
        ]

    if previous_move is None:
        move_values = [0.0] * len(MOVE_ORDER) + [1.0]
    else:
        if previous_move not in MOVE_ORDER:
            raise ValueError(f"Unknown previous move: {previous_move!r}.")
        move_values = [float(move == previous_move) for move in MOVE_ORDER] + [0.0]
    values_by_block["H"] = [*move_values, float(prior_count)]

    if previous_mrb1_scores is None:
        values_by_block["Q"] = [0.0, 0.0, 0.0, 0.0, 1.0]
    else:
        if set(previous_mrb1_scores) != set(MRB1_TASKS):
            raise ValueError("Previous MRB1 scores have incompatible fields.")
        values_by_block["Q"] = [
            *(_unit(task, previous_mrb1_scores[task]) for task in MRB1_TASKS),
            0.0,
        ]

    values = tuple(value for block in blocks for value in values_by_block[block])
    names = feature_names_for_blocks(blocks)
    if len(values) != len(names) or not np.isfinite(values).all():
        raise RuntimeError("Turn-LinTS context construction is inconsistent.")
    return TurnContext(
        schema_version=CONTEXT_SCHEMA_VERSION,
        schema_id=context_schema_id(blocks),
        enabled_blocks=blocks,
        available_blocks=BLOCK_ORDER,
        feature_names=names,
        values=values,
        missingness=missingness,
    )


__all__ = (
    "BLOCK_FEATURE_NAMES", "BLOCK_ORDER", "CONTEXT_PRESETS",
    "CONTEXT_SCHEMA_VERSION", "LEARNER_SIGNAL_NAMES", "TurnContext",
    "build_turn_lints_context", "context_schema_id", "feature_names_for_blocks",
    "parse_context_blocks",
)
