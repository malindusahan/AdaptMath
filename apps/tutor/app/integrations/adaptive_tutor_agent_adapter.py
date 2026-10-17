"""Adapt frozen-v3's generator protocol to the real TutorAgent."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from app.schemas.dialogue import DialogueTurn
from app.schemas.tutor import TutorInput

from app.integrations.tutor_state_memory_adapter import (
    CANONICAL_MOVES,
    TutorStateMemoryAdapter,
)


class AdaptiveTutorAgentAdapter:
    """Supply one frozen-selected move to ``TutorAgent.teach_turn``."""

    def __init__(
        self,
        *,
        tutor_agent: Any,
        tutor_state: Mapping[str, Any],
        memory_adapter: TutorStateMemoryAdapter,
    ) -> None:
        teach_turn = getattr(tutor_agent, "teach_turn", None)
        if not callable(teach_turn):
            raise TypeError("tutor_agent must expose teach_turn().")
        if not isinstance(tutor_state, Mapping):
            raise TypeError("tutor_state must be a mapping.")

        self._tutor_agent = tutor_agent
        self._state = tutor_state
        self._memory_adapter = memory_adapter
        self.last_generation_fallback_used = False

    @staticmethod
    def _dialogue_history(
        history: Sequence[Mapping[str, object]],
    ) -> list[DialogueTurn]:
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise TypeError("Adaptive conversation history must be a sequence.")

        dialogue: list[DialogueTurn] = []
        for index, raw_turn in enumerate(history):
            if not isinstance(raw_turn, Mapping):
                raise TypeError(f"Adaptive history turn {index} must be a mapping.")
            role = raw_turn.get("user")
            text = raw_turn.get("text")
            if role not in {"student", "teacher"}:
                raise ValueError(
                    f"Adaptive history turn {index} has invalid role {role!r}."
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Adaptive history turn {index} requires non-empty text."
                )
            dialogue.append(DialogueTurn(role=role, content=text))
        return dialogue

    def generate(
        self,
        *,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
        pedagogical_move: str,
    ) -> str:
        if pedagogical_move not in CANONICAL_MOVES:
            raise ValueError(
                f"Unknown canonical pedagogical move: {pedagogical_move!r}."
            )

        state_problem = self._state.get("question")
        if problem != state_problem:
            raise ValueError("Adaptive problem does not match TutorState.question.")

        authoritative_projection = list(
            self._memory_adapter.get_conversation_history(self._state)
        )
        if list(conversation_history) != authoritative_projection:
            raise ValueError(
                "Adaptive history snapshot does not match TutorState history."
            )

        evidence = self._state.get("verified_math_evidence")
        if not isinstance(evidence, list):
            raise ValueError("Verified Tutor math evidence is not available.")

        tutor_input = TutorInput(
            question=problem,
            topic=self._state["topic"],
            subtopic=self._state.get("subtopic"),
            student_age=self._state["age"],
            complexity_score=self._state["complexity_score"],
            planner_output=self._state.get("planner_output") or None,
            previous_errors=list(self._state.get("previous_errors", [])),
            reteaching=self._state.get("teaching_phase", "initial")
            == "reteaching",
            pedagogical_move=pedagogical_move,
            conversation_history=self._dialogue_history(conversation_history),
        )

        output = self._tutor_agent.teach_turn(
            tutor_input,
            copy.deepcopy(evidence),
        )
        response = getattr(output, "teaching_response", None)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("TutorAgent returned an empty teaching response.")
        self.last_generation_fallback_used = bool(
            getattr(output, "fallback_used", False)
        )

        self._memory_adapter.set_pending_pedagogical_move(pedagogical_move)
        return response.strip()


__all__ = ("AdaptiveTutorAgentAdapter",)
