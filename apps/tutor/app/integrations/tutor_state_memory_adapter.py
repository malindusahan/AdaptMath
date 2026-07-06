"""Project the persisted TutorState into frozen-v3 tutor memory."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from app.schemas.dialogue import DialogueTurn


CANONICAL_MOVES = frozenset({"generic", "probing", "focus", "telling"})


class TutorStateMemoryAdapter:
    """Expose TutorState without creating a second conversation history.

    ``get_conversation_history`` returns a projection for MD6/MRB1. The only
    mutation performed by this adapter is the frozen pipeline's completed
    teacher-response append to ``TutorState.conversation_history``.
    """

    def __init__(self) -> None:
        self._pending_pedagogical_move: str | None = None
        self._turn_lints_pre_action_state: dict[str, object] | None = None

    @staticmethod
    def _mapping(memory: object) -> Mapping[str, Any]:
        if not isinstance(memory, Mapping):
            raise TypeError("Tutor adaptive memory must be a mapping.")
        return memory

    @staticmethod
    def _mutable_mapping(memory: object) -> MutableMapping[str, Any]:
        if not isinstance(memory, MutableMapping):
            raise TypeError("Tutor adaptive memory must be mutable.")
        return memory

    def get_attempt_id(self, memory: object) -> str:
        value = self._mapping(memory).get("attempt_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("TutorState.attempt_id must be a non-empty string.")
        return value.strip()

    def get_problem(self, memory: object) -> str:
        value = self._mapping(memory).get("question")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("TutorState.question must be a non-empty string.")
        return value

    def get_conversation_history(
        self,
        memory: object,
    ) -> Sequence[Mapping[str, object]]:
        raw_history = self._mapping(memory).get("conversation_history", [])
        if isinstance(raw_history, (str, bytes)) or not isinstance(
            raw_history,
            Sequence,
        ):
            raise TypeError("TutorState.conversation_history must be a sequence.")

        projected: list[dict[str, object]] = []
        for raw_turn in raw_history:
            turn = DialogueTurn.model_validate(raw_turn)
            projected.append({"user": turn.role, "text": turn.content})
        return projected

    def get_mastery_before(self, memory: object) -> object:
        mapping = self._mapping(memory)
        if "mastery_before" not in mapping:
            raise ValueError("TutorState.mastery_before is not available.")
        return mapping["mastery_before"]

    def get_turn_lints_pre_action_state(
        self,
        memory: object,
    ) -> Mapping[str, object]:
        """Return the coordinator's causal snapshot for the next action."""

        self._mapping(memory)
        if self._turn_lints_pre_action_state is None:
            raise ValueError("Turn-LinTS pre-action state is unavailable.")
        return dict(self._turn_lints_pre_action_state)

    def set_turn_lints_pre_action_state(
        self,
        *,
        mastery_before: float,
        previous_mastery_delta: float | None,
        previous_learner_signals: Mapping[str, float] | None,
    ) -> None:
        """Install values already known before the upcoming Tutor action."""

        self._turn_lints_pre_action_state = {
            "mastery_before": mastery_before,
            "previous_mastery_delta": previous_mastery_delta,
            "previous_learner_signals": (
                None
                if previous_learner_signals is None
                else dict(previous_learner_signals)
            ),
        }

    def set_pending_pedagogical_move(self, move: str) -> None:
        """Bind the externally selected move to the next adapter-owned append."""

        if move not in CANONICAL_MOVES:
            raise ValueError(f"Unknown canonical pedagogical move: {move!r}.")
        if self._pending_pedagogical_move is not None:
            raise RuntimeError("A tutor response is already awaiting its append.")
        self._pending_pedagogical_move = move

    def append_tutor_response(self, memory: object, response: str) -> None:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Tutor response must be a non-empty string.")
        if self._pending_pedagogical_move is None:
            raise RuntimeError(
                "Tutor response append is missing its selected pedagogical move."
            )

        mapping = self._mutable_mapping(memory)
        history = mapping.get("conversation_history")
        if not isinstance(history, list):
            raise TypeError("TutorState.conversation_history must be a mutable list.")

        history.append(
            DialogueTurn(
                role="teacher",
                content=response.strip(),
                pedagogical_move=self._pending_pedagogical_move,
            ).model_dump()
        )
        self._pending_pedagogical_move = None


__all__ = ("CANONICAL_MOVES", "TutorStateMemoryAdapter")
