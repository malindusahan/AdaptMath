"""Explicit learner interaction requests handled above Turn-LinTS.

This module recognizes only direct requests for an interaction style.  It does
not infer a pedagogical action from confusion, uncertainty, detector scores,
mastery, or other learner-state estimates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


AGENCY_BEHAVIOR_POLICY: Final[str] = "explicit_learner_agency_v1"
AGENCY_ASSIGNMENT_SOURCE: Final[str] = "learner_agency"


def _normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.casefold().replace("â€™", "'").replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


_DIRECT_EXPLANATION_REQUESTS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:please )?(?:just )?(?:tell|show|give|explain) (?:me )?(?:the )?(?:answer|solution)\b",
        r"\b(?:can|could|would|will) you (?:please )?(?:explain|show|tell) (?:me )?(?:how|the answer|the solution)\b",
        r"\b(?:please )?(?:solve|explain) (?:it|this|the problem|the question)(?: for me)?\b",
    )
)
_DIRECT_QUESTIONING_REQUESTS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:please )?(?:ask|give) me (?:some |more )?(?:questions|problems)\b",
        r"\b(?:can|could|would|will) you (?:please )?(?:ask|quiz) me\b",
        r"\b(?:quiz|test) me(?: with questions)?\b",
    )
)


@dataclass(frozen=True, slots=True)
class ExplicitLearnerAgencyDecision:
    requested_move: str | None
    request_category: str | None
    latest_student_text: str | None

    @property
    def triggered(self) -> bool:
        return self.requested_move is not None

    @property
    def assignment_source(self) -> str | None:
        if self.request_category is None:
            return None
        return AGENCY_ASSIGNMENT_SOURCE


def decide_explicit_learner_agency(
    conversation_history: Sequence[Mapping[str, object]],
) -> ExplicitLearnerAgencyDecision:
    if isinstance(conversation_history, (str, bytes)) or not isinstance(
        conversation_history, Sequence
    ):
        raise TypeError("conversation_history must be a turn sequence")
    latest: str | None = None
    for turn in conversation_history:
        if not isinstance(turn, Mapping):
            raise TypeError("conversation_history turns must be mappings")
        role = str(turn.get("user", turn.get("role", ""))).casefold()
        if role not in {"student", "learner"}:
            continue
        value = turn.get("text", turn.get("content"))
        if isinstance(value, str) and value.strip():
            latest = value.strip()
    normalized = _normalize(latest)
    if normalized and any(pattern.search(normalized) for pattern in _DIRECT_QUESTIONING_REQUESTS):
        return ExplicitLearnerAgencyDecision("probing", "explicit_questioning_request", latest)
    if normalized and any(pattern.search(normalized) for pattern in _DIRECT_EXPLANATION_REQUESTS):
        return ExplicitLearnerAgencyDecision("telling", "explicit_explanation_request", latest)
    return ExplicitLearnerAgencyDecision(None, None, latest)


__all__ = (
    "AGENCY_BEHAVIOR_POLICY",
    "AGENCY_ASSIGNMENT_SOURCE",
    "ExplicitLearnerAgencyDecision",
    "decide_explicit_learner_agency",
)
