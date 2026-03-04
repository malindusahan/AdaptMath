"""Active-candidate and frozen-shadow selector adapter for diagnostics only.

The adapter deliberately exposes the same ``predict_probabilities`` interface
as the frozen base selector.  It returns the active MD7-R1 probabilities while
retaining frozen-MD6 output for diagnostics.  Both models receive independent
deep copies of one canonical problem/history snapshot, and neither model may
mutate that snapshot.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .context_builder import MOVE_ORDER


class ActiveMD7WithMD6Shadow:
    """Return MD7 probabilities and retain same-input MD6 diagnostics."""

    def __init__(self, *, active_md7: Any, shadow_md6: Any) -> None:
        for name, selector in (
            ("active_md7", active_md7),
            ("shadow_md6", shadow_md6),
        ):
            if not callable(getattr(selector, "predict_probabilities", None)):
                raise TypeError(f"{name} must expose predict_probabilities().")
        self.active_md7 = active_md7
        self.shadow_md6 = shadow_md6
        self.records: list[dict[str, object]] = []

    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> dict[str, float]:
        canonical_history = copy.deepcopy(list(conversation_history))
        md7_history = copy.deepcopy(canonical_history)
        md6_history = copy.deepcopy(canonical_history)
        md7 = dict(self.active_md7.predict_probabilities(problem, md7_history))
        md6 = dict(self.shadow_md6.predict_probabilities(problem, md6_history))
        if md7_history != md6_history or md7_history != canonical_history:
            raise RuntimeError(
                "An active or shadow selector mutated its history input."
            )
        if tuple(md7) != MOVE_ORDER:
            raise ValueError("MD7 returned an incompatible label order.")
        if tuple(md6) != MOVE_ORDER:
            raise ValueError("MD6 returned an incompatible label order.")
        self.records.append(
            {
                "problem": problem,
                "history": canonical_history,
                "md6_probabilities": md6,
                "md7_probabilities": md7,
            }
        )
        return md7

    @property
    def last_record(self) -> Mapping[str, object]:
        """Return the most recent paired inference without removing it."""

        if not self.records:
            raise RuntimeError("No paired selector inference is available.")
        return self.records[-1]


__all__ = ("ActiveMD7WithMD6Shadow",)
