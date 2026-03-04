"""Deterministic learner-agency escalation for pedagogical move selection.

The trained selector remains the source of the raw move probabilities.  This
small, auditable layer intervenes only when the latest learner turn explicitly
asks for the answer or when the two latest learner turns both communicate
non-engagement.  An intervention returns a one-hot ``telling`` distribution so
the downstream C3/LinTS controller, Tutor, and logs all observe the move that is
actually delivered.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Any

from .context_builder import MOVE_ORDER


TELLING_PROBABILITIES: dict[str, float] = {
    move: 1.0 if move == "telling" else 0.0 for move in MOVE_ORDER
}

_DIRECT_ANSWER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:please )?(?:just )?(?:give|tell|show) me (?:the )?(?:answer|solution)\b",
        r"\b(?:can|could|will|would) you (?:please )?(?:just )?"
        r"(?:give|tell|show) me (?:the )?(?:answer|solution)\b",
        r"\bwhat(?: is| s) (?:the )?answer\b",
        r"\bi (?:just|only) want (?:the )?(?:answer|solution)\b",
        r"\b(?:please )?(?:answer|solve) (?:it|this|the problem|the question)"
        r"(?: for me)?\b",
    )
)

_NON_ENGAGEMENT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?:i )?(?:really )?(?:dont|do not) know(?: anything| this| that| at all)?",
        r"(?:i )?(?:have )?no idea(?: at all)?",
        r"idk",
        r"(?:no )?i (?:dont|do not|cannot|cant) understand"
        r"(?: anything(?: about)?(?: this| the question)?| this| the question| at all)?",
        r"(?:i )?(?:cannot|cant) think(?: about (?:it|that|this))?"
        r"(?: (?:right )?now)?",
        r"(?:i am|im) stuck",
    )
)


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.casefold().replace("’", "'")
    normalized = normalized.replace("don't", "dont").replace("can't", "cant")
    normalized = normalized.replace("i'm", "im").replace("what's", "what s")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", normalized)).strip()


def is_direct_answer_request(value: object) -> bool:
    """Return whether a learner explicitly requests an answer or solution."""

    text = _normalized_text(value)
    return bool(text) and any(pattern.search(text) for pattern in _DIRECT_ANSWER_PATTERNS)


def is_non_engagement(value: object) -> bool:
    """Recognize short, content-free statements of inability or disengagement."""

    text = _normalized_text(value)
    return bool(text) and any(pattern.fullmatch(text) for pattern in _NON_ENGAGEMENT_PATTERNS)


def _student_texts(
    conversation_history: Sequence[Mapping[str, object]],
) -> list[str]:
    texts: list[str] = []
    for turn in conversation_history:
        if not isinstance(turn, Mapping):
            raise TypeError("conversation_history turns must be mappings.")
        if str(turn.get("user", "")).casefold() != "student":
            continue
        text = turn.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


@dataclass(frozen=True, slots=True)
class LearnerAgencyDecision:
    """One deterministic post-selector escalation decision."""

    triggered: bool
    reason: str | None
    latest_student_text: str | None
    consecutive_non_engagement_turns: int

    def as_mapping(self) -> dict[str, object]:
        return asdict(self)


def decide_learner_agency_escalation(
    conversation_history: Sequence[Mapping[str, object]],
) -> LearnerAgencyDecision:
    """Force telling for a direct request or two latest non-engaged turns."""

    if isinstance(conversation_history, (str, bytes)) or not isinstance(
        conversation_history,
        Sequence,
    ):
        raise TypeError("conversation_history must be a turn sequence.")

    texts = _student_texts(conversation_history)
    latest = texts[-1] if texts else None
    consecutive = 0
    for text in reversed(texts):
        if not is_non_engagement(text):
            break
        consecutive += 1

    if latest is not None and is_direct_answer_request(latest):
        return LearnerAgencyDecision(
            triggered=True,
            reason="explicit_answer_request",
            latest_student_text=latest,
            consecutive_non_engagement_turns=consecutive,
        )
    if consecutive >= 2:
        return LearnerAgencyDecision(
            triggered=True,
            reason="repeated_non_engagement",
            latest_student_text=latest,
            consecutive_non_engagement_turns=consecutive,
        )
    return LearnerAgencyDecision(
        triggered=False,
        reason=None,
        latest_student_text=latest,
        consecutive_non_engagement_turns=consecutive,
    )


def _validated_probabilities(values: Mapping[str, object]) -> dict[str, float]:
    if tuple(values) != MOVE_ORDER:
        raise ValueError("Base selector returned an incompatible label order.")
    probabilities = {move: float(values[move]) for move in MOVE_ORDER}
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("Base selector probabilities must be finite values in [0, 1].")
    if not isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Base selector probabilities must sum approximately to 1.")
    return probabilities


class LearnerAgencyEscalationSelector:
    """Wrap a probability selector with a deterministic telling intervention."""

    def __init__(self, base_selector: Any, *, retain_records: bool = False) -> None:
        if not callable(getattr(base_selector, "predict_probabilities", None)):
            raise TypeError("base_selector must expose predict_probabilities().")
        self.base_selector = base_selector
        self.retain_records = bool(retain_records)
        self.records: list[dict[str, object]] = []
        self.last_decision: LearnerAgencyDecision | None = None
        self.last_raw_probabilities: dict[str, float] | None = None
        self.last_effective_probabilities: dict[str, float] | None = None

    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> dict[str, float]:
        canonical_history = copy.deepcopy(list(conversation_history))
        raw = _validated_probabilities(
            self.base_selector.predict_probabilities(
                problem,
                copy.deepcopy(canonical_history),
            )
        )
        decision = decide_learner_agency_escalation(canonical_history)
        effective = dict(TELLING_PROBABILITIES if decision.triggered else raw)

        self.last_decision = decision
        self.last_raw_probabilities = dict(raw)
        self.last_effective_probabilities = dict(effective)

        if self.retain_records:
            record: dict[str, object] = {
                "problem": problem,
                "history": canonical_history,
            }
            inner_records = getattr(self.base_selector, "records", None)
            if isinstance(inner_records, list) and inner_records:
                inner_record = inner_records[-1]
                if isinstance(inner_record, Mapping):
                    record.update(copy.deepcopy(dict(inner_record)))
            record.update(
                {
                    "raw_active_probabilities": dict(raw),
                    "effective_probabilities": dict(effective),
                    "learner_agency_escalation": decision.as_mapping(),
                }
            )
            self.records.append(record)

        return effective

    @property
    def last_record(self) -> Mapping[str, object]:
        if not self.records:
            raise RuntimeError("No learner-agency selector record is available.")
        return self.records[-1]


__all__ = (
    "LearnerAgencyDecision",
    "LearnerAgencyEscalationSelector",
    "TELLING_PROBABILITIES",
    "decide_learner_agency_escalation",
    "is_direct_answer_request",
    "is_non_engagement",
)
