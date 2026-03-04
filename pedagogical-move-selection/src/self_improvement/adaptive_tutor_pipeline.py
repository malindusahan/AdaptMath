"""Working orchestration layer for the frozen adaptive tutor policy."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableSequence, Sequence
from math import isclose
from pathlib import Path
from typing import Protocol, runtime_checkable

from .conservative_overlay import DEFAULT_GAP_THRESHOLD
from .context_builder import MOVE_ORDER
from .experience_logger import ExperienceLogger, SCHEMA_VERSION, validate_metadata
from .lints_policy import DEFAULT_EXPLORATION_SCALE, DEFAULT_RIDGE_LAMBDA
from .md6_inference import FrozenMD6Inference
from .mrb1_inference import FrozenMRB1Inference
from .reward import LearningOutcome, validate_learning_outcome
from .state_io import load_policy_state, save_policy_state
from .turn_context_builder import TURN_FEATURE_NAMES, validate_mastery_probability
from .turn_level_controller import (
    AttemptCompletion,
    TurnDecision,
    TurnLevelAttemptController,
)


@runtime_checkable
class TutorMemoryAdapter(Protocol):
    """Minimal bridge to one authoritative host tutoring memory object."""

    def get_attempt_id(self, memory: object) -> str | int: ...

    def get_problem(self, memory: object) -> str: ...

    def get_conversation_history(
        self,
        memory: object,
    ) -> Sequence[Mapping[str, object]]: ...

    def get_mastery_before(self, memory: object) -> object: ...

    def append_tutor_response(self, memory: object, response: str) -> None: ...


@runtime_checkable
class TutorAgent(Protocol):
    """Injected external tutor-agent contract."""

    def generate(
        self,
        *,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
        pedagogical_move: str,
    ) -> str: ...


@runtime_checkable
class MD6Predictor(Protocol):
    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> Mapping[str, float]: ...


@runtime_checkable
class MRB1Scorer(Protocol):
    def score_response(
        self,
        conversation_history: Sequence[Mapping[str, object]],
        tutor_response: str,
    ) -> Mapping[str, float]: ...


class MappingTutorMemoryAdapter:
    """Default adapter for one mutable mapping-backed host memory.

    Required keys are ``attempt_id``, ``problem``, ``conversation_history``,
    and ``mastery_before``. Conversation history is the single authoritative
    mutable list, whose turns use the already validated MD6 ``user``/``text``
    fields. This adapter appends the completed tutor response as a ``teacher``
    turn; the host remains responsible for appending subsequent student turns.
    """

    @staticmethod
    def _mapping(memory: object) -> Mapping[str, object]:
        if not isinstance(memory, Mapping):
            raise TypeError("memory must be a mapping for MappingTutorMemoryAdapter.")
        return memory

    def get_attempt_id(self, memory: object) -> str | int:
        value = self._mapping(memory).get("attempt_id")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("memory.attempt_id must be a string or integer.")
        if isinstance(value, str) and not value.strip():
            raise ValueError("memory.attempt_id cannot be empty or whitespace.")
        return value

    def get_problem(self, memory: object) -> str:
        value = self._mapping(memory).get("problem")
        if not isinstance(value, str):
            raise TypeError("memory.problem must be a string.")
        if not value.strip():
            raise ValueError("memory.problem cannot be empty or whitespace.")
        return value

    def get_conversation_history(
        self,
        memory: object,
    ) -> Sequence[Mapping[str, object]]:
        value = self._mapping(memory).get("conversation_history")
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("memory.conversation_history must be a turn sequence.")
        for index, turn in enumerate(value):
            if not isinstance(turn, Mapping):
                raise TypeError(
                    f"memory.conversation_history[{index}] must be a mapping."
                )
        return value

    def get_mastery_before(self, memory: object) -> object:
        mapping = self._mapping(memory)
        if "mastery_before" not in mapping:
            raise ValueError("memory.mastery_before is required.")
        return mapping["mastery_before"]

    def append_tutor_response(self, memory: object, response: str) -> None:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("tutor response must be a non-empty string.")
        mapping = self._mapping(memory)
        history = mapping.get("conversation_history")
        if not isinstance(history, MutableSequence):
            raise TypeError(
                "memory.conversation_history must be mutable so the tutor "
                "response can become authoritative history."
            )
        history.append({"user": "teacher", "text": response})


class AdaptiveTutorPipeline:
    """Run real tutor turns through MD6, LinTS/overlay, tutor, and MRB1."""

    def __init__(
        self,
        *,
        md6: MD6Predictor | FrozenMD6Inference,
        mrb1: MRB1Scorer | FrozenMRB1Inference,
        controller: TurnLevelAttemptController,
        experience_logger: ExperienceLogger,
        policy_state_path: str | Path,
        memory_adapter: TutorMemoryAdapter | None = None,
    ) -> None:
        if not isinstance(controller, TurnLevelAttemptController):
            raise TypeError("controller must be a TurnLevelAttemptController.")
        if not isinstance(experience_logger, ExperienceLogger):
            raise TypeError("experience_logger must be an ExperienceLogger.")
        if not isinstance(md6, MD6Predictor):
            raise TypeError("md6 must expose predict_probabilities().")
        if not isinstance(mrb1, MRB1Scorer):
            raise TypeError("mrb1 must expose score_response().")
        adapter = MappingTutorMemoryAdapter() if memory_adapter is None else memory_adapter
        if not isinstance(adapter, TutorMemoryAdapter):
            raise TypeError("memory_adapter does not satisfy TutorMemoryAdapter.")

        policy = controller.policy
        if policy.context_dim != len(TURN_FEATURE_NAMES) or len(TURN_FEATURE_NAMES) != 9:
            raise ValueError("AdaptiveTutorPipeline requires the frozen 9-D C3 context.")
        if policy.ridge_lambda != DEFAULT_RIDGE_LAMBDA:
            raise ValueError("AdaptiveTutorPipeline requires ridge_lambda=1.0.")
        if policy.exploration_scale != DEFAULT_EXPLORATION_SCALE:
            raise ValueError("AdaptiveTutorPipeline requires exploration_scale=0.20.")
        if controller.gap_threshold != DEFAULT_GAP_THRESHOLD:
            raise ValueError("AdaptiveTutorPipeline requires gap_threshold=0.10.")
        if experience_logger.data_mode != policy.data_mode:
            raise ValueError("Logger and policy data modes must match.")
        if SCHEMA_VERSION != "turn_lints_v3":
            raise RuntimeError("AdaptiveTutorPipeline requires turn_lints_v3 logging.")

        self.md6 = md6
        self.mrb1 = mrb1
        self.controller = controller
        self.experience_logger = experience_logger
        self.policy_state_path = Path(policy_state_path)
        self.memory_adapter = adapter

        self._active_attempt_id: str | int | None = None
        self._active_mastery_before: float | None = None
        self._pending_completion: AttemptCompletion | None = None
        self._pending_learning_outcome: LearningOutcome | None = None
        self._pending_metadata: dict[str, object] | None = None
        self._pending_record: dict[str, object] | None = None
        self._pending_turn_problem: str | None = None
        self._pending_turn_history: list[Mapping[str, object]] | None = None
        self._pending_turn_decision: TurnDecision | None = None
        self._pending_tutor_response: str | None = None
        self._pending_mrb1_scores: dict[str, float] | None = None

    @property
    def active_attempt_id(self) -> str | int | None:
        return self._active_attempt_id

    def _require_matching_attempt(self, memory: object) -> str | int:
        if self._active_attempt_id is None:
            raise RuntimeError("No adaptive tutoring attempt is active.")
        attempt_id = self.memory_adapter.get_attempt_id(memory)
        if attempt_id != self._active_attempt_id:
            raise ValueError(
                f"Memory attempt_id {attempt_id!r} does not match active "
                f"attempt {self._active_attempt_id!r}."
            )
        return attempt_id

    def load_persisted_policy_state(self) -> None:
        """Load the configured signed-reward v3 state before an attempt starts."""

        if self._active_attempt_id is not None or self.controller.is_active:
            raise RuntimeError("Cannot load policy state during an active attempt.")
        load_policy_state(
            self.controller.policy,
            self.policy_state_path,
            expected_data_mode=self.controller.policy.data_mode,
        )

    def _clear_pending_turn_retry(self) -> None:
        self._pending_turn_problem = None
        self._pending_turn_history = None
        self._pending_turn_decision = None
        self._pending_tutor_response = None
        self._pending_mrb1_scores = None

    def start_attempt(self, memory: object) -> dict[str, object]:
        """Read attempt identity/mastery from host memory and start cleanly."""

        if self._active_attempt_id is not None:
            raise RuntimeError("An adaptive tutoring attempt is already active.")
        attempt_id = self.memory_adapter.get_attempt_id(memory)
        mastery_before = self.memory_adapter.get_mastery_before(memory)
        validated_mastery = validate_mastery_probability(mastery_before)
        self.controller.start_attempt(validated_mastery, attempt_id=attempt_id)
        self._active_attempt_id = attempt_id
        self._active_mastery_before = validated_mastery
        self._pending_completion = None
        self._pending_learning_outcome = None
        self._pending_metadata = None
        self._pending_record = None
        self._clear_pending_turn_retry()
        return {
            "attempt_id": attempt_id,
            "mastery_before": validated_mastery,
        }

    def restore_attempt(
        self,
        memory: object,
        completed_turns: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Start from durable, already-completed turns without new decisions."""

        started = self.start_attempt(memory)
        try:
            self.controller.restore_completed_turns(completed_turns)
        except Exception:
            self.abort_attempt(memory)
            raise
        return started

    def run_tutor_turn(
        self,
        memory: object,
        tutor_agent: TutorAgent,
    ) -> dict[str, object]:
        """Run one complete MD6 -> action -> tutor -> MRB1 turn."""

        attempt_id = self._require_matching_attempt(memory)
        if self._pending_completion is not None or not self.controller.is_active:
            raise RuntimeError("The active attempt has already entered completion.")
        if not isinstance(tutor_agent, TutorAgent):
            raise TypeError("tutor_agent must expose generate().")

        problem = self.memory_adapter.get_problem(memory)
        authoritative_history = self.memory_adapter.get_conversation_history(memory)
        history_snapshot = copy.deepcopy(list(authoritative_history))

        decision = self._pending_turn_decision
        if decision is None:
            md6_probabilities = dict(
                self.md6.predict_probabilities(problem, history_snapshot)
            )
            decision = self.controller.select_turn(md6_probabilities)
            if decision.final_move not in MOVE_ORDER:
                raise RuntimeError("Controller returned a non-canonical final move.")
            self._pending_turn_problem = problem
            self._pending_turn_history = copy.deepcopy(history_snapshot)
            self._pending_turn_decision = decision
        elif (
            problem != self._pending_turn_problem
            or history_snapshot != self._pending_turn_history
        ):
            raise RuntimeError(
                "A failed tutor turn can be retried only from its identical "
                "problem and conversation-history checkpoint."
            )

        tutor_response = self._pending_tutor_response
        if tutor_response is None:
            tutor_response = tutor_agent.generate(
                problem=problem,
                conversation_history=copy.deepcopy(history_snapshot),
                pedagogical_move=decision.final_move,
            )
            if not isinstance(tutor_response, str) or not tutor_response.strip():
                raise ValueError("Tutor agent must return a non-empty string response.")
            tutor_response = tutor_response.strip()
            self._pending_tutor_response = tutor_response

        mrb1_scores = self._pending_mrb1_scores
        if mrb1_scores is None:
            mrb1_scores = dict(
                self.mrb1.score_response(history_snapshot, tutor_response)
            )
            self.controller.record_mrb1_scores(mrb1_scores)
            self._pending_mrb1_scores = dict(mrb1_scores)

        result = {
            "attempt_id": attempt_id,
            "action_event_id": decision.action_event_id,
            "action_turn_index": decision.action_turn_index,
            "turn_index": decision.turn_index,
            "tutor_response": tutor_response,
            "pedagogical_move": decision.final_move,
            "selected_arm": decision.selected_arm,
            "base_move": decision.base_move,
            "overridden": decision.overridden,
            "eligible_arms": list(decision.eligible_arms),
            "context": list(decision.context),
            "md6_probabilities": dict(decision.md6_probabilities),
            "mrb1_scores": mrb1_scores,
            "tutor_generation_fallback_used": bool(
                getattr(tutor_agent, "last_generation_fallback_used", False)
            ),
        }
        raw_probabilities = getattr(self.md6, "last_raw_probabilities", None)
        if not isinstance(raw_probabilities, Mapping):
            raw_probabilities = decision.md6_probabilities
        agency_decision = getattr(self.md6, "last_decision", None)
        agency_mapping = (
            agency_decision.as_mapping()
            if callable(getattr(agency_decision, "as_mapping", None))
            else {"triggered": False, "reason": None}
        )

        def _argmax(probabilities: Mapping[str, object]) -> str:
            return max(
                MOVE_ORDER,
                key=lambda move: float(probabilities[move]),
            )

        result["selector"] = {
            "raw": {
                "probabilities": dict(raw_probabilities),
                "argmax": _argmax(raw_probabilities),
            },
            "learner_agency": {
                "triggered": bool(agency_mapping.get("triggered", False)),
                "reason": agency_mapping.get("reason"),
            },
            "effective": {
                "probabilities": dict(decision.md6_probabilities),
                "argmax": _argmax(decision.md6_probabilities),
            },
        }
        result["adaptive_decision"] = {
            "context_name": "C3",
            "context_feature_names": list(TURN_FEATURE_NAMES),
            "context": list(decision.context),
            "eligible_arms": list(decision.eligible_arms),
            "sampled_scores": dict(decision.sampled_scores),
            "selected_arm": decision.selected_arm,
            "base_move": decision.base_move,
            "final_move": decision.final_move,
            "overridden": decision.overridden,
            "gap": decision.gap,
            "gap_threshold": decision.gap_threshold,
        }
        self.memory_adapter.append_tutor_response(memory, tutor_response)
        self._clear_pending_turn_retry()
        return result

    def finish_attempt(
        self,
        memory: object,
        learning_outcome: LearningOutcome | Mapping[str, object],
        *,
        diagnostics: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Finish from an already-produced v3 learning outcome.

        If logging or state persistence raises, the exception is propagated.
        A completed controller result and successful log are retained so a
        caller may retry the unfinished finalization stage without applying a
        second posterior update or appending a second experience record. Every
        retry must repeat the exact validated outcome, diagnostics, and
        metadata supplied on the first successful controller completion.
        """

        attempt_id = self._require_matching_attempt(memory)
        validated_outcome = validate_learning_outcome(
            learning_outcome,
            diagnostics=diagnostics,
        )
        validated_metadata = validate_metadata(metadata)
        if self._active_mastery_before is None:
            raise RuntimeError("Active attempt is missing its start mastery.")
        if not isclose(
            validated_outcome.mastery_before,
            self._active_mastery_before,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "learning_outcome.mastery_before does not match the value used "
                "to start the active attempt."
            )

        if self._pending_completion is None:
            if not self.controller.is_active:
                raise RuntimeError("Controller is not active for attempt completion.")
            completion = self.controller.finish_attempt(
                reward=validated_outcome.delta_mastery,
                mastery_after=validated_outcome.mastery_after,
            )
            self._pending_completion = completion
            self._pending_learning_outcome = validated_outcome
            self._pending_metadata = validated_metadata
        else:
            pending_outcome = self._pending_learning_outcome
            if pending_outcome is None:
                raise RuntimeError("Pending completion is missing its outcome.")
            if (
                validated_outcome.as_mapping() != pending_outcome.as_mapping()
                or validated_outcome.diagnostics_mapping()
                != pending_outcome.diagnostics_mapping()
                or validated_metadata != self._pending_metadata
            ):
                raise ValueError(
                    "Retry inputs must exactly match the already validated "
                    "learning outcome, diagnostics, and metadata."
                )

        completion = self._pending_completion
        pending_outcome = self._pending_learning_outcome
        if completion is None or pending_outcome is None:
            raise RuntimeError("Internal completion state is unavailable.")
        for name, actual in (
            ("completion.reward", completion.reward),
            ("completion.mastery_delta", completion.mastery_delta),
        ):
            if not isclose(
                actual,
                pending_outcome.delta_mastery,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"{name} differs from learning_outcome.delta_mastery."
                )

        if self._pending_record is None:
            self._pending_record = self.experience_logger.append_attempt(
                attempt_id,
                completion,
                pending_outcome,
                metadata=self._pending_metadata,
            )

        save_policy_state(self.controller.policy, self.policy_state_path)
        record = self._pending_record
        result = {
            "attempt_id": attempt_id,
            "skill": pending_outcome.skill,
            "reward": completion.reward,
            "mastery_before": completion.mastery_before,
            "mastery_after": completion.mastery_after,
            "mastery_delta": completion.mastery_delta,
            "turn_count": completion.turn_count,
            "sample_weight_per_turn": completion.sample_weight_per_turn,
            "total_attempt_weight": completion.total_attempt_weight,
            "experience_record": copy.deepcopy(record),
            "policy_state_path": str(self.policy_state_path),
        }
        supplied_diagnostics = pending_outcome.diagnostics_mapping()
        if supplied_diagnostics:
            result["diagnostics"] = supplied_diagnostics
        self._active_attempt_id = None
        self._active_mastery_before = None
        self._pending_completion = None
        self._pending_learning_outcome = None
        self._pending_metadata = None
        self._pending_record = None
        self._clear_pending_turn_retry()
        return result

    def abort_attempt(self, memory: object | None = None) -> None:
        """Abort the active pre-completion attempt without a posterior update."""

        if self._active_attempt_id is None:
            raise RuntimeError("No adaptive tutoring attempt is active.")
        if memory is not None:
            self._require_matching_attempt(memory)
        if self._pending_completion is not None or not self.controller.is_active:
            raise RuntimeError("A completed attempt cannot be aborted.")
        self.controller.abort_attempt()
        self._active_attempt_id = None
        self._active_mastery_before = None
        self._pending_completion = None
        self._pending_learning_outcome = None
        self._pending_metadata = None
        self._pending_record = None
        self._clear_pending_turn_retry()


__all__ = (
    "AdaptiveTutorPipeline",
    "MD6Predictor",
    "MRB1Scorer",
    "MappingTutorMemoryAdapter",
    "TutorAgent",
    "TutorMemoryAdapter",
)
