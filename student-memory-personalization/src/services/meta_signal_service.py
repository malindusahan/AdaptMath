"""Meta-Agent Evidence Signals Service emitting transparent evidence markers."""

from __future__ import annotations

import re
from typing import Any

from src.database.postgres_session import SessionFactory, get_session_factory
from src.schemas.meta_signals import MetaSignalItem, MetaSignalsResponse
from src.services.student_context_service import (
    StudentContextService,
    get_student_context_service,
)

# Deterministic regex patterns for conversational cues
CONFUSION_PATTERN = re.compile(
    r"\b(i\s*don'?t\s*understand|i'?m\s*confused|not\s*sure|makes\s*no\s*sense|i'?m\s*lost|i'?m\s*stuck|confusing)\b",
    re.IGNORECASE,
)

CLARIFICATION_PATTERN = re.compile(
    r"\b(can\s*you\s*explain|could\s*you\s*explain|please\s*explain|what\s*does\s*that\s*mean|why\s*is|why\s*does|why\s*not|how\s*do\s*i|how\s*does|help\s*me\s*understand|clarify|what\s*is)\b",
    re.IGNORECASE,
)


class MetaSignalService:
    """Emits auditable learning evidence signals for Meta-Agent analysis without making mastery decisions."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        student_context_service: StudentContextService | None = None,
    ):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        self.student_context_service = (
            student_context_service
            if student_context_service is not None
            else get_student_context_service(session_factory=self.session_factory)
        )

    def get_meta_signals(
        self,
        student_id: str,
        session_id: str | None = None,
        skill_id: str | None = None,
        limit: int = 20,
    ) -> MetaSignalsResponse:
        """
        Generate chronological evidence signals:
        - correct_answer: verified successful interaction
        - incorrect_answer: verified failure event
        - confusion: explicit student expression of confusion/lost
        - clarification_request: explicit request for elaboration or explanation
        - repeated_misunderstanding: persistent misconception occurrence >= 2
        """
        full_context = self.student_context_service.get_student_context(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            limit=limit,
        )

        signals: list[MetaSignalItem] = []

        # 1. Signals derived from Interactions
        for inter in full_context.recent_interactions:
            # Correct answer signal
            if inter.is_correct:
                signals.append(
                    MetaSignalItem(
                        signal_type="correct_answer",
                        student_id=student_id,
                        session_id=inter.session_id,
                        skill_id=inter.canonical_skill_id or skill_id,
                        interaction_id=inter.interaction_id,
                        timestamp=inter.created_at,
                        confidence=1.0,
                        evidence={
                            "attempt_count": inter.attempt_count,
                            "hint_count": inter.hint_count,
                            "response_time_ms": inter.response_time_ms,
                        },
                    )
                )
            else:
                # Incorrect answer signal
                signals.append(
                    MetaSignalItem(
                        signal_type="incorrect_answer",
                        student_id=student_id,
                        session_id=inter.session_id,
                        skill_id=inter.canonical_skill_id or skill_id,
                        interaction_id=inter.interaction_id,
                        timestamp=inter.created_at,
                        confidence=1.0,
                        evidence={
                            "identified_error": inter.identified_error,
                            "attempt_count": inter.attempt_count,
                            "hint_count": inter.hint_count,
                            "response_time_ms": inter.response_time_ms,
                        },
                    )
                )

            # Linguistic evidence from student utterance
            if inter.student_utterance and inter.student_utterance.strip():
                utterance = inter.student_utterance.strip()

                # Confusion signal
                conf_match = CONFUSION_PATTERN.search(utterance)
                if conf_match:
                    signals.append(
                        MetaSignalItem(
                            signal_type="confusion",
                            student_id=student_id,
                            session_id=inter.session_id,
                            skill_id=inter.canonical_skill_id or skill_id,
                            interaction_id=inter.interaction_id,
                            timestamp=inter.created_at,
                            confidence=0.95,
                            evidence={
                                "matched_pattern": conf_match.group(0),
                                "utterance": utterance,
                            },
                        )
                    )

                # Clarification request signal
                clar_match = CLARIFICATION_PATTERN.search(utterance)
                if clar_match:
                    signals.append(
                        MetaSignalItem(
                            signal_type="clarification_request",
                            student_id=student_id,
                            session_id=inter.session_id,
                            skill_id=inter.canonical_skill_id or skill_id,
                            interaction_id=inter.interaction_id,
                            timestamp=inter.created_at,
                            confidence=0.90,
                            evidence={
                                "matched_pattern": clar_match.group(0),
                                "utterance": utterance,
                            },
                        )
                    )

        # 2. Signals derived from Persistent Misconceptions
        for misc in full_context.misconceptions:
            occ_count = misc.get("occurrence_count", 0)
            if occ_count >= 2:
                conf = min(1.0, 0.5 + 0.25 * occ_count)
                signals.append(
                    MetaSignalItem(
                        signal_type="repeated_misunderstanding",
                        student_id=student_id,
                        session_id=session_id,
                        skill_id=skill_id,
                        interaction_id=None,
                        timestamp=misc.get("last_seen_at", ""),
                        confidence=conf,
                        evidence={
                            "misconception_id": misc.get("misconception_id"),
                            "normalized_error": misc.get("normalized_error"),
                            "display_error": misc.get("display_error"),
                            "occurrence_count": occ_count,
                        },
                    )
                )

        # Sort chronologically by timestamp
        signals.sort(key=lambda s: s.timestamp)

        # Bounded by limit
        if len(signals) > limit:
            signals = signals[-limit:]

        return MetaSignalsResponse(
            student_id=student_id,
            session_id=session_id,
            skill_id=skill_id,
            total_signals=len(signals),
            signals=signals,
        )


# Singleton factory cache
_META_SIGNAL_SERVICE: MetaSignalService | None = None


def get_meta_signal_service(
    session_factory: SessionFactory | None = None,
) -> MetaSignalService:
    global _META_SIGNAL_SERVICE
    if _META_SIGNAL_SERVICE is None or session_factory is not None:
        _META_SIGNAL_SERVICE = MetaSignalService(session_factory=session_factory)
    return _META_SIGNAL_SERVICE
