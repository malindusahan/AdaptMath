"""Lifecycle boundary for Tutor, student modeling, and frozen-v3 policy."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Literal

from app.integrations.adaptive_tutor_agent_adapter import (
    AdaptiveTutorAgentAdapter,
)
from app.integrations.adaptive_selector_mode import (
    AdaptiveSelectorMode,
    resolve_selector_mode,
)
from app.integrations.manual_controlled_live import (
    ManualControlledLiveDiagnostics,
)
from app.integrations.live_bkt_activity import LiveBKTActivityLog
from app.integrations.tutor_state_memory_adapter import (
    TutorStateMemoryAdapter,
)


AdaptiveLifecycleStatus = Literal[
    "not_started",
    "active",
    "completed",
    "aborted",
]

# Informal, scaffolded dialogue is useful mastery evidence but is less
# authoritative than the formal assessment. Keep its evaluator contribution
# conservatively bounded even when the progress judge reports high confidence.
DIALOGUE_EVALUATOR_CONFIDENCE_CAP = 0.5
NO_KNOWLEDGE_EVIDENCE_CATEGORIES = frozenset(
    {"interaction_management", "acknowledgement", "unclear_no_evidence"}
)
EVIDENCE_CATEGORIES = frozenset(
    {
        "mathematical_evidence",
        "knowledge_state_evidence",
        *NO_KNOWLEDGE_EVIDENCE_CATEGORIES,
    }
)
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _research_resolved_event(event: object) -> dict[str, object]:
    """Copy resolver diagnostics while excluding the raw student identifier."""

    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            raw = dump(mode="json")
        except TypeError:
            raw = dump()
    elif isinstance(event, Mapping):
        raw = dict(event)
    else:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        key: copy.deepcopy(raw.get(key))
        for key in (
            "event_id",
            "source_action_event_id",
            "skill_id",
            "primary_signal",
            "bkt_update",
            "behaviour",
            "history",
            "resolver_version",
        )
        if key in raw
    }


def _turn_lints_outcome_projection(
    student_model_result: Mapping[str, object],
    action_event_id: str,
) -> dict[str, object]:
    """Project the unique linked resolver/BKT event for immediate policy credit."""

    raw_events = student_model_result.get("resolved_events", [])
    if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
        raise AdaptiveIntegrationError("Resolved events are unavailable.")
    projected = [_research_resolved_event(event) for event in raw_events]
    matches = [event for event in projected
               if event.get("source_action_event_id") == action_event_id]
    if len(matches) != 1:
        raise AdaptiveIntegrationError(
            "Turn-LinTS requires one uniquely linked action/resolver event."
        )
    event = matches[0]
    bkt = event.get("bkt_update")
    behaviour = event.get("behaviour")
    if not isinstance(bkt, Mapping) or not isinstance(behaviour, Mapping):
        raise AdaptiveIntegrationError("Resolved event lacks BKT/behaviour data.")
    signals = {
        name: behaviour.get(name)
        for name in (
            "reasoning_probability",
            "uncertainty_probability",
            "clarification_probability",
        )
    }
    learner_signals = (
        None if any(value is None for value in signals.values()) else signals
    )
    incremental = student_model_result.get("incremental_learning_outcome")
    if not isinstance(incremental, Mapping):
        raise AdaptiveIntegrationError("Incremental turn mastery is unavailable.")
    should_update = bkt.get("should_update")
    if not isinstance(should_update, bool):
        raise AdaptiveIntegrationError("BKT update lacks should_update.")
    return {
        "resolver_event_id": event.get("event_id"),
        "should_update": should_update,
        "mastery_before": incremental.get("mastery_before") if should_update else None,
        "mastery_after": incremental.get("mastery_after") if should_update else None,
        "previous_learner_signals": learner_signals,
    }


class AdaptiveIntegrationError(RuntimeError):
    """Base error for an integration-boundary failure."""


class AdaptiveConcurrencyError(AdaptiveIntegrationError):
    """Raised when a second overlapping attempt would corrupt shared policy."""


class AdaptiveRestartRequiredError(AdaptiveIntegrationError):
    """Raised when persisted state has no safe in-process controller context."""


@dataclass(slots=True)
class AdaptiveAttemptComponents:
    """Attempt-scoped public objects built around shared scientific resources."""

    bridge: Any
    student_model_pipeline: Any
    memory_adapter: TutorStateMemoryAdapter


@dataclass(slots=True)
class ActiveAdaptiveContext:
    student_id: str
    thread_id: str
    attempt_id: str
    target_skill: str
    adaptive_attempt_index: int
    attempt_started_at: str
    bridge: Any = field(repr=False)
    student_model_pipeline: Any = field(repr=False)
    memory_adapter: TutorStateMemoryAdapter = field(repr=False)
    mastery_before: float
    status: AdaptiveLifecycleStatus = "active"
    last_turn_pre_history: list[dict[str, object]] | None = field(
        default=None,
        repr=False,
    )
    last_turn_result: dict[str, object] | None = field(default=None, repr=False)
    student_model_result: Mapping[str, object] | None = field(
        default=None,
        repr=False,
    )
    completion_result: dict[str, object] | None = field(
        default=None,
        repr=False,
    )
    bkt_activity_recorded: bool = field(default=False, repr=False)
    dialogue_results: dict[str, dict[str, object]] = field(
        default_factory=dict,
        repr=False,
    )
    dialogue_activity_recorded: set[str] = field(
        default_factory=set,
        repr=False,
    )
    research_staged_actions: set[str] = field(default_factory=set, repr=False)
    research_completed_actions: set[str] = field(default_factory=set, repr=False)
    research_assessment_recorded: bool = field(default=False, repr=False)
    research_attempt_summary_recorded: bool = field(default=False, repr=False)
    previous_mastery_delta: float | None = field(default=None, repr=False)
    previous_learner_signals: dict[str, float] | None = field(
        default=None,
        repr=False,
    )


class _AssessmentEvidenceSource:
    """Project formal assessments and live dialogue into student-model evidence."""

    def __init__(self, target_skill: str) -> None:
        self.target_skill = target_skill
        self.context = SimpleNamespace(assessed_skills=(target_skill,))
        self._verdicts: dict[int, dict[str, object]] = {}

    def build_transcript(
        self,
        evaluated_answers: Sequence[Mapping[str, object]],
    ) -> list[dict[str, str]]:
        transcript: list[dict[str, str]] = []
        verdicts: dict[int, dict[str, object]] = {}

        for index, answer in enumerate(evaluated_answers):
            question = answer.get("question")
            student_answer = answer.get("student_answer")
            is_correct = answer.get("is_correct")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Evaluated assessment answer {index} has no question."
                )
            if not isinstance(student_answer, str) or not student_answer.strip():
                raise ValueError(
                    f"Evaluated assessment answer {index} has no student answer."
                )
            if not isinstance(is_correct, bool):
                raise ValueError(
                    f"Evaluated assessment answer {index} has no boolean verdict."
                )

            transcript.append({"role": "tutor", "text": question.strip()})
            student_turn_index = len(transcript)
            transcript.append(
                {"role": "student", "text": student_answer.strip()}
            )
            verdicts[student_turn_index] = {
                "correctness": "correct" if is_correct else "incorrect",
                "confidence": 1.0,
                "source": "adaptmath_structured_assessment_evaluator",
            }

        if not transcript:
            raise ValueError("Assessment completion requires evaluated answers.")
        self._verdicts = verdicts
        return transcript

    def build_dialogue_turn(
        self,
        *,
        teacher_text: str,
        student_text: str,
        correctness: str,
        confidence: float,
        evidence_category: str = "mathematical_evidence",
    ) -> list[dict[str, str]]:
        """Project exactly one live dialogue exchange into student-model input."""

        teacher = str(teacher_text).strip()
        student = str(student_text).strip()
        if not teacher or not student:
            raise ValueError("Dialogue BKT evidence requires teacher and student text.")
        if correctness not in {"correct", "partial", "incorrect", "unknown"}:
            raise ValueError(f"Invalid dialogue correctness: {correctness!r}.")
        if evidence_category not in EVIDENCE_CATEGORIES:
            raise ValueError(f"Invalid dialogue evidence category: {evidence_category!r}.")
        if (
            evidence_category in NO_KNOWLEDGE_EVIDENCE_CATEGORIES
            and correctness != "unknown"
        ):
            raise ValueError(
                "No-knowledge-evidence responses must have unknown correctness."
            )
        numeric_confidence = float(confidence)
        if not 0.0 <= numeric_confidence <= 1.0:
            raise ValueError("Dialogue evaluator confidence must be in [0, 1].")

        self._verdicts = {
            1: {
                "correctness": correctness,
                "confidence": numeric_confidence,
                "source": (
                    "adaptmath_no_knowledge_evidence"
                    if evidence_category in NO_KNOWLEDGE_EVIDENCE_CATEGORIES
                    else "adaptmath_teaching_progress_judge"
                ),
                "evidence_category": evidence_category,
            }
        }
        return [
            {"role": "tutor", "text": teacher},
            {"role": "student", "text": student},
        ]

    def extract(self, transcript: list[dict[str, object]]) -> dict[str, object]:
        events = [
            {
                "turn_index": turn_index,
                "skill": self.target_skill,
                "evidence_span": str(transcript[turn_index]["text"]),
            }
            for turn_index in sorted(self._verdicts)
        ]
        return {"events": events, "misconceptions": []}

    def __call__(
        self,
        event: Mapping[str, object],
        transcript: list[dict[str, object]],
    ) -> Mapping[str, object]:
        del transcript
        turn_index = event.get("turn_index")
        if not isinstance(turn_index, int) or turn_index not in self._verdicts:
            raise ValueError("Assessment event has no authoritative Tutor verdict.")
        return dict(self._verdicts[turn_index])


class TutorAssessmentStudentModelPipeline:
    """Bind one target skill and Tutor dialogue/assessment evidence to Repository B."""

    def __init__(
        self,
        *,
        target_skill: str,
        cross_session_pipeline: Any,
        evidence_source: _AssessmentEvidenceSource,
    ) -> None:
        self.target_skill = target_skill
        self._pipeline = cross_session_pipeline
        self._evidence_source = evidence_source
        self._attempt_context: Any = None
        self._mastery_checkpoint: float | None = None
        self._dialogue_batches_processed = 0
        self._dialogue_observations_applied = 0

    def start_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        attempt_skill: str | None = None,
    ) -> object:
        if attempt_skill != self.target_skill:
            raise ValueError("Student-model attempt skill differs from target_skill.")
        context = self._pipeline.start_attempt(
            student_id=student_id,
            attempt_id=attempt_id,
            attempt_skill=attempt_skill,
        )
        self._attempt_context = context
        self._mastery_checkpoint = float(context.mastery_before)
        self._dialogue_batches_processed = 0
        self._dialogue_observations_applied = 0
        return context

    def restore_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        mastery_before: float,
        dialogue_turns_processed: int,
        dialogue_observations_applied: int,
    ) -> object:
        """Recreate attempt bookkeeping from durable controlled-live state."""

        if self._attempt_context is not None:
            raise RuntimeError("Student-model adaptive attempt is already active.")
        start_mastery = float(mastery_before)
        if not 0.0 <= start_mastery <= 1.0:
            raise ValueError("mastery_before must be in [0, 1].")
        if dialogue_turns_processed < 0 or dialogue_observations_applied < 0:
            raise ValueError("Restored dialogue counts cannot be negative.")
        context = SimpleNamespace(
            student_id=student_id,
            attempt_id=attempt_id,
            skill=self.target_skill,
            mastery_before=start_mastery,
        )
        self._attempt_context = context
        self._mastery_checkpoint = self._current_mastery(student_id)
        self._dialogue_batches_processed = dialogue_turns_processed
        self._dialogue_observations_applied = dialogue_observations_applied
        return context

    def _validate_identity(self, *, student_id: str, attempt_id: str) -> None:
        if self._attempt_context is None:
            raise RuntimeError("Student-model adaptive attempt has not started.")
        if student_id != self._attempt_context.student_id:
            raise ValueError("student_id differs from the active student-model attempt.")
        if attempt_id != self._attempt_context.attempt_id:
            raise ValueError("attempt_id differs from the active student-model attempt.")

    def _current_mastery(self, student_id: str) -> float:
        graph = getattr(self._pipeline, "knowledge_graph", None)
        getter = getattr(graph, "get_current_mastery_probability", None)
        if not callable(getter):
            raise TypeError(
                "Incremental dialogue BKT requires current-mastery access."
            )
        current = getter(
            student_id,
            self.target_skill,
            require_effective_prior=True,
        )
        if current is None:
            raise RuntimeError("Current BKT mastery is unavailable.")
        return float(current)

    def current_mastery(self, *, student_id: str, attempt_id: str) -> float:
        """Expose the authoritative live BKT state for action-time logging."""

        self._validate_identity(student_id=student_id, attempt_id=attempt_id)
        return self._current_mastery(student_id)

    def _process_incremental_transcript(
        self,
        *,
        transcript: list[dict[str, str]],
        student_id: str,
        session_id: str,
        source_action_event_id: str | None = None,
    ) -> Mapping[str, object]:
        if self._attempt_context is None or self._mastery_checkpoint is None:
            raise RuntimeError("Student-model adaptive attempt has not started.")

        before_event = self._current_mastery(student_id)
        if not isclose(
            before_event,
            self._mastery_checkpoint,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Incremental dialogue BKT mastery checkpoint is stale; no new "
                "conversation evidence was persisted."
            )

        result = dict(
            self._pipeline.process_transcript(
                transcript=transcript,
                student_id=student_id,
                session_id=session_id,
                behavioural_skill=self.target_skill,
                attempt_context=None,
                source_action_event_id=source_action_event_id,
            )
        )
        after_event = self._current_mastery(student_id)
        self._mastery_checkpoint = after_event
        result["incremental_learning_outcome"] = {
            "mastery_before": before_event,
            "mastery_after": after_event,
            "delta_mastery": after_event - before_event,
        }
        result["learning_outcome"] = {
            "attempt_id": self._attempt_context.attempt_id,
            "skill": self._attempt_context.skill,
            "mastery_before": float(self._attempt_context.mastery_before),
            "mastery_after": after_event,
            "delta_mastery": after_event
            - float(self._attempt_context.mastery_before),
        }
        return result

    def process_dialogue_turn(
        self,
        *,
        student_id: str,
        attempt_id: str,
        turn_index: int,
        action_event_id: str | None = None,
        teacher_text: str,
        student_text: str,
        correctness: str,
        evaluator_confidence: float,
        evidence_category: str = "mathematical_evidence",
    ) -> Mapping[str, object]:
        self._validate_identity(student_id=student_id, attempt_id=attempt_id)
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
            raise ValueError("turn_index must be a positive integer.")
        if action_event_id is None:
            action_event_id = f"{attempt_id}:action:{turn_index}"
        if not isinstance(action_event_id, str) or not action_event_id.strip():
            raise ValueError("action_event_id must be a non-empty string.")
        transcript = self._evidence_source.build_dialogue_turn(
            teacher_text=teacher_text,
            student_text=student_text,
            correctness=correctness,
            confidence=evaluator_confidence,
            evidence_category=evidence_category,
        )
        result = self._process_incremental_transcript(
            transcript=transcript,
            student_id=student_id,
            session_id=f"{attempt_id}:dialogue:{turn_index}",
            source_action_event_id=action_event_id.strip(),
        )
        self._dialogue_batches_processed += 1
        raw_events = result.get("resolved_events", [])
        if isinstance(raw_events, Sequence) and not isinstance(
            raw_events,
            (str, bytes),
        ):
            self._dialogue_observations_applied += sum(
                getattr(getattr(event, "bkt_update", None), "should_update", False)
                is True
                for event in raw_events
            )
        return result

    def process_assessment_cycle(
        self,
        *,
        student_id: str,
        attempt_id: str,
        evaluated_answers: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        self._validate_identity(student_id=student_id, attempt_id=attempt_id)
        transcript = self._evidence_source.build_transcript(evaluated_answers)
        if self._dialogue_batches_processed:
            result = dict(self._process_incremental_transcript(
                transcript=transcript,
                student_id=student_id,
                session_id=attempt_id,
            ))
            result["dialogue_evidence_summary"] = {
                "turns_processed": self._dialogue_batches_processed,
                "observations_applied": self._dialogue_observations_applied,
            }
            return result
        return self._pipeline.process_transcript(
            transcript=transcript,
            student_id=student_id,
            session_id=attempt_id,
            behavioural_skill=self.target_skill,
            attempt_context=self._attempt_context,
        )


ComponentFactory = Callable[..., AdaptiveAttemptComponents]
SkillValidator = Callable[[str], bool | None]
TurnObserver = Callable[..., None]
CompletionObserver = Callable[..., None]
BKTActivityObserver = Callable[..., None]
DialogueBKTActivityObserver = Callable[..., None]


class AdaptiveComponentCoordinator:
    """Coordinate attempt-scoped bridges over one shared frozen policy.

    Frozen ``TurnLevelAttemptController`` instances snapshot
    ``policy.total_updates`` at attempt start. Consequently two overlapping
    controllers cannot safely share one policy: completion of either attempt
    invalidates the other's guard. This coordinator therefore grants one
    process-wide attempt lease and rejects overlap explicitly. It never creates
    per-student policy copies or attempts to merge learned posteriors.
    """

    def __init__(
        self,
        *,
        component_factory: ComponentFactory,
        skill_validator: SkillValidator,
        suppress_policy_completion: bool = False,
        turn_observer: TurnObserver | None = None,
        completion_observer: CompletionObserver | None = None,
        bkt_activity_observer: BKTActivityObserver | None = None,
        dialogue_bkt_activity_observer: DialogueBKTActivityObserver | None = None,
        research_collector: Any = None,
    ) -> None:
        if not callable(component_factory):
            raise TypeError("component_factory must be callable.")
        if not callable(skill_validator):
            raise TypeError("skill_validator must be callable.")
        if not isinstance(suppress_policy_completion, bool):
            raise TypeError("suppress_policy_completion must be boolean.")
        for name, observer in (
            ("turn_observer", turn_observer),
            ("completion_observer", completion_observer),
            ("bkt_activity_observer", bkt_activity_observer),
            ("dialogue_bkt_activity_observer", dialogue_bkt_activity_observer),
        ):
            if observer is not None and not callable(observer):
                raise TypeError(f"{name} must be callable when supplied.")
        self._component_factory = component_factory
        self._skill_validator = skill_validator
        self._suppress_policy_completion = suppress_policy_completion
        self._turn_observer = turn_observer
        self._completion_observer = completion_observer
        self._bkt_activity_observer = bkt_activity_observer
        self._dialogue_bkt_activity_observer = dialogue_bkt_activity_observer
        self._research_collector = research_collector
        self._lock = RLock()
        self._contexts: dict[str, ActiveAdaptiveContext] = {}
        self._completed_results: dict[tuple[str, str], dict[str, object]] = {}
        self._active_policy_thread_id: str | None = None

    @staticmethod
    def _identifier(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _attempt_index(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("adaptive_attempt_index must be a positive integer.")
        return value

    @staticmethod
    def derive_attempt_id(thread_id: str, adaptive_attempt_index: int) -> str:
        thread = AdaptiveComponentCoordinator._identifier(thread_id, "thread_id")
        index = AdaptiveComponentCoordinator._attempt_index(
            adaptive_attempt_index
        )
        return f"{thread}:{index}"

    def validate_target_skill(self, target_skill: object) -> str:
        skill = self._identifier(target_skill, "target_skill")
        result = self._skill_validator(skill)
        if result is False:
            raise KeyError(
                f"target_skill {skill!r} is not in the trained BKT vocabulary."
            )
        return skill

    @staticmethod
    def _state_mapping(state: object) -> MutableMapping[str, Any]:
        if not isinstance(state, MutableMapping):
            raise TypeError("TutorState must be a mutable mapping.")
        return state

    def _identity(
        self,
        state: Mapping[str, Any],
    ) -> tuple[str, str, str, str, int]:
        thread_id = self._identifier(state.get("thread_id"), "thread_id")
        student_id = self._identifier(state.get("student_id"), "student_id")
        attempt_index = self._attempt_index(state.get("adaptive_attempt_index"))
        attempt_id = self._identifier(state.get("attempt_id"), "attempt_id")
        expected_attempt_id = self.derive_attempt_id(thread_id, attempt_index)
        if attempt_id != expected_attempt_id:
            raise ValueError(
                f"attempt_id must equal {expected_attempt_id!r} for this cycle."
            )
        target_skill = self.validate_target_skill(state.get("target_skill"))
        return thread_id, student_id, attempt_id, target_skill, attempt_index

    def _start_locked(
        self,
        memory: MutableMapping[str, Any],
    ) -> dict[str, object]:
        (
            thread_id,
            student_id,
            attempt_id,
            target_skill,
            attempt_index,
        ) = self._identity(memory)

        existing = self._contexts.get(thread_id)
        if (
            existing is not None
            and existing.status == "active"
            and existing.attempt_id == attempt_id
        ):
            if self._active_policy_thread_id != thread_id:
                raise AdaptiveRestartRequiredError(
                    "Active Tutor attempt has no matching shared-policy lease."
                )
            memory["mastery_before"] = existing.mastery_before
            return {
                "attempt_id": attempt_id,
                "adaptive_attempt_index": attempt_index,
                "adaptive_lifecycle_status": "active",
                "mastery_before": existing.mastery_before,
                "attempt_started_at": existing.attempt_started_at,
            }

        if existing is not None and existing.status == "active":
            raise AdaptiveIntegrationError(
                "A different adaptive attempt is already active for this thread."
            )
        if existing is not None and existing.attempt_id == attempt_id:
            raise AdaptiveIntegrationError(
                "A completed or aborted adaptive attempt ID cannot be restarted."
            )

        if self._active_policy_thread_id not in {None, thread_id}:
            raise AdaptiveConcurrencyError(
                "Frozen v3 permits only one overlapping adaptive attempt per "
                "process because each controller guards shared policy updates."
            )

        components = self._component_factory(
            student_id=student_id,
            thread_id=thread_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            adaptive_attempt_index=attempt_index,
        )
        if not isinstance(components, AdaptiveAttemptComponents):
            raise TypeError(
                "component_factory must return AdaptiveAttemptComponents."
            )

        attempt_started_at = _utc_now()
        active = components.bridge.start_attempt(
            student_id=student_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            adaptive_memory=memory,
        )
        mastery_before = float(getattr(active, "mastery_before"))
        context = ActiveAdaptiveContext(
            student_id=student_id,
            thread_id=thread_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            adaptive_attempt_index=attempt_index,
            attempt_started_at=attempt_started_at,
            bridge=components.bridge,
            student_model_pipeline=components.student_model_pipeline,
            memory_adapter=components.memory_adapter,
            mastery_before=mastery_before,
        )
        self._contexts[thread_id] = context
        self._active_policy_thread_id = thread_id
        return {
            "attempt_id": attempt_id,
            "adaptive_attempt_index": attempt_index,
            "adaptive_lifecycle_status": "active",
            "mastery_before": mastery_before,
            "attempt_started_at": attempt_started_at,
        }

    @staticmethod
    def _controller_restore_record(
        record: Mapping[str, object],
    ) -> dict[str, object]:
        identity = record.get("identity")
        decision = record.get("adaptive_decision")
        selector = record.get("selector")
        quality = record.get("tutor_quality")
        if not all(
            isinstance(value, Mapping)
            for value in (identity, decision, selector, quality)
        ):
            raise AdaptiveRestartRequiredError(
                "A durable action is missing restart reconstruction fields."
            )
        effective = selector.get("effective")
        scores = quality.get("scores")
        if not isinstance(effective, Mapping) or not isinstance(scores, Mapping):
            raise AdaptiveRestartRequiredError(
                "A durable action is missing selector or MRB1 values."
            )
        probabilities = effective.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise AdaptiveRestartRequiredError(
                "A durable action is missing effective move probabilities."
            )
        selected_arm = str(decision.get("selected_arm"))
        target_move = (
            None if selected_arm == "baseline" else selected_arm.removesuffix("_bias")
        )
        base_move = str(decision.get("base_move"))
        return {
            "decision": {
                "action_event_id": identity.get("action_event_id"),
                "action_turn_index": identity.get("action_turn_index"),
                "context": decision.get("context"),
                "md6_probabilities": probabilities,
                "eligible_arms": decision.get("eligible_arms"),
                "sampled_scores": decision.get("sampled_scores"),
                "selected_arm": selected_arm,
                "base_move": base_move,
                "final_move": decision.get("final_move"),
                "target_move": target_move,
                "overridden": decision.get("overridden"),
                "gap": decision.get("gap"),
                "gap_threshold": decision.get("gap_threshold"),
                "base_probability": probabilities.get(base_move),
                "target_probability": (
                    None if target_move is None else probabilities.get(target_move)
                ),
            },
            "mrb1_scores": scores,
        }

    def _restore_controlled_live_locked(
        self,
        memory: MutableMapping[str, Any],
    ) -> ActiveAdaptiveContext:
        if not self._suppress_policy_completion:
            raise AdaptiveRestartRequiredError(
                "Persisted adaptive attempt cannot be reconstructed after a "
                "process restart while policy completion is enabled."
            )
        loader = getattr(self._research_collector, "finalized_actions_for_attempt", None)
        if not callable(loader):
            raise AdaptiveRestartRequiredError(
                "Controlled-live restart requires durable finalized turn records."
            )
        (
            thread_id,
            student_id,
            attempt_id,
            target_skill,
            attempt_index,
        ) = self._identity(memory)
        diagnostics = memory.get("adaptive_turn_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise AdaptiveRestartRequiredError(
                "Controlled-live restart requires persisted turn diagnostics."
            )
        expected_turn_count = diagnostics.get("action_turn_index")
        if (
            isinstance(expected_turn_count, bool)
            or not isinstance(expected_turn_count, int)
            or expected_turn_count < 1
        ):
            raise AdaptiveRestartRequiredError(
                "Persisted action_turn_index is unavailable for restart."
            )
        finalized = loader(attempt_id)
        if len(finalized) != expected_turn_count:
            raise AdaptiveRestartRequiredError(
                "Durable finalized turns do not match the persisted action index."
            )
        restored_turns = [
            self._controller_restore_record(record) for record in finalized
        ]
        mastery_before = float(memory.get("mastery_before"))
        attempt_started_at = self._identifier(
            memory.get("attempt_started_at"),
            "attempt_started_at",
        )
        components = self._component_factory(
            student_id=student_id,
            thread_id=thread_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            adaptive_attempt_index=attempt_index,
        )
        observations_applied = sum(
            record.get("learning_state_update", {}).get("should_update") is True
            for record in finalized
            if isinstance(record, Mapping)
            and isinstance(record.get("learning_state_update"), Mapping)
        )
        dialogue_turns_processed = sum(
            isinstance(record.get("next_learner_observation"), Mapping)
            and isinstance(
                record["next_learner_observation"].get("student_text"),
                str,
            )
            for record in finalized
        )
        restore_student = getattr(
            components.student_model_pipeline,
            "restore_attempt",
            None,
        )
        restore_bridge = getattr(components.bridge, "restore_attempt", None)
        if not callable(restore_student) or not callable(restore_bridge):
            raise AdaptiveRestartRequiredError(
                "Controlled-live components do not support safe restart restoration."
            )
        student_context = restore_student(
            student_id=student_id,
            attempt_id=attempt_id,
            mastery_before=mastery_before,
            dialogue_turns_processed=dialogue_turns_processed,
            dialogue_observations_applied=observations_applied,
        )
        restore_bridge(
            student_id=student_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            mastery_before=mastery_before,
            student_model_context=student_context,
            completed_turns=restored_turns,
            adaptive_memory=memory,
        )
        action_ids = {
            str(record["identity"]["action_event_id"])
            for record in finalized
        }
        context = ActiveAdaptiveContext(
            student_id=student_id,
            thread_id=thread_id,
            attempt_id=attempt_id,
            target_skill=target_skill,
            adaptive_attempt_index=attempt_index,
            attempt_started_at=attempt_started_at,
            bridge=components.bridge,
            student_model_pipeline=components.student_model_pipeline,
            memory_adapter=components.memory_adapter,
            mastery_before=mastery_before,
            last_turn_result=copy.deepcopy(dict(diagnostics)),
            research_staged_actions=set(action_ids),
            research_completed_actions=set(action_ids),
            dialogue_activity_recorded=set(action_ids),
        )
        self._contexts[thread_id] = context
        self._active_policy_thread_id = thread_id
        logger.warning(
            "Restored controlled-live attempt %s from %d finalized turns.",
            attempt_id,
            expected_turn_count,
        )
        return context

    def start_attempt(self, state: MutableMapping[str, Any]) -> dict[str, object]:
        memory = self._state_mapping(state)
        with self._lock:
            persisted_status = memory.get(
                "adaptive_lifecycle_status",
                "not_started",
            )
            if persisted_status == "active" and memory.get("thread_id") not in (
                self._contexts
            ):
                context = self._restore_controlled_live_locked(memory)
                return {
                    "attempt_id": context.attempt_id,
                    "adaptive_attempt_index": context.adaptive_attempt_index,
                    "adaptive_lifecycle_status": "active",
                    "mastery_before": context.mastery_before,
                    "attempt_started_at": context.attempt_started_at,
                }
            return self._start_locked(memory)

    def has_active_attempt(self, thread_id: str) -> bool:
        with self._lock:
            context = self._contexts.get(thread_id)
            return context is not None and context.status == "active"

    def _require_active_context(
        self,
        state: Mapping[str, Any],
    ) -> ActiveAdaptiveContext:
        thread_id, student_id, attempt_id, target_skill, attempt_index = (
            self._identity(state)
        )
        context = self._contexts.get(thread_id)
        if context is None:
            context = self._restore_controlled_live_locked(
                self._state_mapping(state)
            )
        expected = (
            context.student_id,
            context.attempt_id,
            context.target_skill,
            context.adaptive_attempt_index,
        )
        actual = (student_id, attempt_id, target_skill, attempt_index)
        if actual != expected:
            raise AdaptiveIntegrationError(
                "Tutor and adaptive attempt identities do not match."
            )
        if context.status != "active":
            raise AdaptiveIntegrationError("Adaptive attempt is not active.")
        if self._active_policy_thread_id != thread_id:
            raise AdaptiveRestartRequiredError(
                "Adaptive attempt has no matching shared-policy lease."
            )
        return context

    @staticmethod
    def _latest_dialogue_exchange(
        memory: Mapping[str, Any],
    ) -> tuple[str, str]:
        history = memory.get("conversation_history", [])
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise TypeError("conversation_history must be a sequence.")

        student_index: int | None = None
        student_text: str | None = None
        for index in range(len(history) - 1, -1, -1):
            turn = history[index]
            if not isinstance(turn, Mapping):
                continue
            if turn.get("role") == "student":
                raw_text = turn.get("content")
                if isinstance(raw_text, str) and raw_text.strip():
                    student_index = index
                    student_text = raw_text.strip()
                    break
        if student_index is None or student_text is None:
            raise ValueError("Dialogue BKT requires a latest student response.")

        for index in range(student_index - 1, -1, -1):
            turn = history[index]
            if not isinstance(turn, Mapping):
                continue
            if turn.get("role") == "teacher":
                raw_text = turn.get("content")
                if isinstance(raw_text, str) and raw_text.strip():
                    return raw_text.strip(), student_text
        raise ValueError("Dialogue BKT requires a preceding teacher turn.")

    def process_dialogue_turn(
        self,
        state: MutableMapping[str, Any],
        *,
        correctness: str,
        evaluator_confidence: float,
        evaluator_reason: str,
        evidence_category: str = "mathematical_evidence",
    ) -> dict[str, object]:
        """Persist at most one BKT observation for the latest live student turn."""

        memory = self._state_mapping(state)
        with self._lock:
            context = self._require_active_context(memory)
            # The in-process controller result is authoritative. TutorState may
            # retain diagnostics from a completed attempt until the graph node
            # persists the next action result.
            turn_diagnostics = context.last_turn_result
            if not isinstance(turn_diagnostics, Mapping):
                turn_diagnostics = memory.get("adaptive_turn_diagnostics")
            if not isinstance(turn_diagnostics, Mapping):
                raise TypeError("adaptive_turn_diagnostics must be a mapping.")
            action_event_id = self._identifier(
                turn_diagnostics.get("action_event_id"),
                "action_event_id",
            )
            turn_index = turn_diagnostics.get("action_turn_index")
            if (
                isinstance(turn_index, bool)
                or not isinstance(turn_index, int)
                or turn_index < 1
            ):
                raise ValueError("Dialogue BKT action_turn_index must be positive.")
            expected_action_id = f"{context.attempt_id}:action:{turn_index}"
            if action_event_id != expected_action_id:
                raise AdaptiveIntegrationError(
                    "Dialogue evidence does not match the preceding Tutor action."
                )
            if correctness not in {"correct", "partial", "incorrect", "unknown"}:
                raise ValueError(f"Invalid dialogue correctness: {correctness!r}.")
            if evidence_category not in EVIDENCE_CATEGORIES:
                raise ValueError(
                    f"Invalid dialogue evidence category: {evidence_category!r}."
                )
            if (
                evidence_category in NO_KNOWLEDGE_EVIDENCE_CATEGORIES
                and correctness != "unknown"
            ):
                raise ValueError(
                    "No-knowledge-evidence responses must have unknown correctness."
                )
            reported_confidence = float(evaluator_confidence)
            if not 0.0 <= reported_confidence <= 1.0:
                raise ValueError("Dialogue evaluator confidence must be in [0, 1].")
            reason = str(evaluator_reason).strip()
            if not reason:
                raise ValueError("Dialogue evaluator reason must be non-empty.")

            teacher_text, student_text = self._latest_dialogue_exchange(memory)
            applied_confidence = min(
                reported_confidence,
                DIALOGUE_EVALUATOR_CONFIDENCE_CAP,
            )
            evidence = {
                "action_event_id": action_event_id,
                "action_turn_index": turn_index,
                # Compatibility alias; this is attempt-local, not thread-global.
                "turn_index": turn_index,
                "thread_turn_count": memory.get("turn_count"),
                "teacher_text": teacher_text,
                "student_text": student_text,
                "correctness": correctness,
                "reported_evaluator_confidence": reported_confidence,
                "applied_evaluator_confidence": applied_confidence,
                "evaluator_confidence_cap": DIALOGUE_EVALUATOR_CONFIDENCE_CAP,
                "evaluator_reason": reason,
                "evidence_category": evidence_category,
                "evaluator_source": (
                    "adaptmath_no_knowledge_evidence"
                    if evidence_category in NO_KNOWLEDGE_EVIDENCE_CATEGORIES
                    else "adaptmath_teaching_progress_judge"
                ),
            }

            cached = context.dialogue_results.get(action_event_id)
            if cached is not None:
                if cached.get("evidence") != evidence:
                    raise AdaptiveIntegrationError(
                        "A processed dialogue turn was replayed with different evidence."
                    )
            else:
                process = getattr(
                    context.student_model_pipeline,
                    "process_dialogue_turn",
                    None,
                )
                if not callable(process):
                    raise TypeError(
                        "Student-model pipeline does not support dialogue evidence."
                    )
                try:
                    student_model_result = process(
                        student_id=context.student_id,
                        attempt_id=context.attempt_id,
                        turn_index=turn_index,
                        action_event_id=action_event_id,
                        teacher_text=teacher_text,
                        student_text=student_text,
                        correctness=correctness,
                        evaluator_confidence=applied_confidence,
                        evidence_category=evidence_category,
                    )
                except Exception as exc:
                    note_turn_failure = getattr(
                        context.bridge.adaptive_pipeline,
                        "note_turn_processing_failure",
                        None,
                    )
                    if callable(note_turn_failure):
                        note_turn_failure(action_event_id)
                    note_failure = getattr(
                        self._research_collector,
                        "note_processing_failure",
                        None,
                    )
                    if callable(note_failure):
                        try:
                            note_failure(
                                action_event_id,
                                dialogue_evidence=copy.deepcopy(evidence),
                                failure_reason=(
                                    "student_model_processing_failed:"
                                    f"{type(exc).__name__}"
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "Passive processing-failure marker failed for %s.",
                                action_event_id,
                            )
                    raise
                if not isinstance(student_model_result, Mapping):
                    raise TypeError("Dialogue student-model result must be a mapping.")
                cached = {
                    "evidence": evidence,
                    "student_model_result": copy.deepcopy(dict(student_model_result)),
                }
                context.dialogue_results[action_event_id] = cached

            complete_turn = getattr(
                context.bridge.adaptive_pipeline,
                "complete_turn_outcome",
                None,
            )
            if callable(complete_turn):
                projection = _turn_lints_outcome_projection(
                    cached["student_model_result"],
                    action_event_id,
                )
                complete_turn(
                    action_event_id=action_event_id,
                    **projection,
                )
                signals = projection["previous_learner_signals"]
                context.previous_learner_signals = (
                    dict(signals) if isinstance(signals, Mapping) else None
                )
                if projection["should_update"] is True:
                    context.previous_mastery_delta = float(
                        projection["mastery_after"]
                    ) - float(projection["mastery_before"])
                else:
                    context.previous_mastery_delta = None

            if (
                self._dialogue_bkt_activity_observer is not None
                and action_event_id not in context.dialogue_activity_recorded
            ):
                self._dialogue_bkt_activity_observer(
                    student_id=context.student_id,
                    attempt_id=context.attempt_id,
                    target_skill=context.target_skill,
                    dialogue_evidence=copy.deepcopy(cached["evidence"]),
                    student_model_result=copy.deepcopy(
                        cached["student_model_result"]
                    ),
                )
                context.dialogue_activity_recorded.add(action_event_id)

            if (
                self._research_collector is not None
                and action_event_id not in context.research_completed_actions
            ):
                try:
                    self._research_collector.complete_dialogue_action(
                        action_event_id=action_event_id,
                        dialogue_evidence=copy.deepcopy(cached["evidence"]),
                        student_model_result=copy.deepcopy(
                            cached["student_model_result"]
                        ),
                    )
                    context.research_completed_actions.add(action_event_id)
                except Exception:
                    logger.exception(
                        "Passive dialogue outcome collection failed for %s.",
                        action_event_id,
                    )

            return copy.deepcopy(cached)

    def run_tutor_turn(
        self,
        state: MutableMapping[str, Any],
        *,
        tutor_agent: Any,
    ) -> dict[str, object]:
        memory = self._state_mapping(state)
        with self._lock:
            context = self._require_active_context(memory)
            pre_history = copy.deepcopy(
                list(context.memory_adapter.get_conversation_history(memory))
            )

            # If a graph checkpoint write failed after a successful turn, the
            # persisted pre-history re-enters unchanged. Replay the cached
            # response append without running MD6/LinTS/Tutor/MRB1 twice.
            if (
                context.last_turn_pre_history == pre_history
                and context.last_turn_result is not None
            ):
                cached = copy.deepcopy(context.last_turn_result)
                context.memory_adapter.set_pending_pedagogical_move(
                    str(cached["pedagogical_move"])
                )
                context.memory_adapter.append_tutor_response(
                    memory,
                    str(cached["tutor_response"]),
                )
                return cached

            adapter = AdaptiveTutorAgentAdapter(
                tutor_agent=tutor_agent,
                tutor_state=memory,
                memory_adapter=context.memory_adapter,
            )
            adaptive_pipeline = context.bridge.adaptive_pipeline
            controller = adaptive_pipeline.controller
            completed_before = len(controller.completed_turns)
            current_mastery = getattr(
                context.student_model_pipeline,
                "current_mastery",
                None,
            )
            if callable(current_mastery):
                mastery_at_action = float(
                    current_mastery(
                        student_id=context.student_id,
                        attempt_id=context.attempt_id,
                    )
                )
            elif context.dialogue_results:
                latest_dialogue = next(
                    reversed(context.dialogue_results.values())
                )
                incremental = latest_dialogue.get(
                    "student_model_result",
                    {},
                ).get("incremental_learning_outcome", {})
                if not isinstance(incremental, Mapping) or (
                    "mastery_after" not in incremental
                ):
                    raise RuntimeError(
                        "Authoritative mastery is unavailable for this action."
                    )
                mastery_at_action = float(incremental["mastery_after"])
            else:
                # At the first action the authoritative start snapshot and the
                # live KnowledgeGraph state are identical by contract.
                mastery_at_action = context.mastery_before
            set_pre_action = getattr(
                context.memory_adapter,
                "set_turn_lints_pre_action_state",
                None,
            )
            if callable(set_pre_action):
                set_pre_action(
                    mastery_before=mastery_at_action,
                    previous_mastery_delta=context.previous_mastery_delta,
                    previous_learner_signals=context.previous_learner_signals,
                )
            try:
                result = adaptive_pipeline.run_tutor_turn(
                    memory,
                    adapter,
                )
            except Exception:
                # AdaptiveTutorPipeline retains the sampled decision and any
                # successfully completed downstream stage. A graph retry can
                # therefore resume the failed stage without a second Thompson
                # draw, tutor generation, MRB1 record, or history append.
                raise
            copied_result = copy.deepcopy(dict(result))
            copied_result.update(
                {
                    "mastery_at_action": mastery_at_action,
                    "attempt_start_mastery": context.mastery_before,
                    "attempt_started_at": context.attempt_started_at,
                    "thread_id": context.thread_id,
                    "adaptive_attempt_index": context.adaptive_attempt_index,
                    "thread_turn_count": int(memory.get("turn_count", 0)) + 1,
                }
            )
            action_event_id = str(copied_result["action_event_id"])
            if (
                self._research_collector is not None
                and action_event_id not in context.research_staged_actions
            ):
                try:
                    self._research_collector.stage_runtime_action(
                        state=memory,
                        history_before_action=pre_history,
                        turn_result=copied_result,
                    )
                    context.research_staged_actions.add(action_event_id)
                except Exception:
                    # Passive logging is never allowed to change the selected
                    # move, Tutor response, MRB1 score, or LinTS state.
                    logger.exception(
                        "Passive turn staging failed for %s.",
                        action_event_id,
                    )
            if self._turn_observer is not None:
                try:
                    completed_turns = controller.completed_turns
                    if len(completed_turns) != completed_before + 1:
                        raise RuntimeError(
                            "A completed tutor turn has no unique controller decision."
                        )
                    self._turn_observer(
                        attempt_id=context.attempt_id,
                        turn_result=copied_result,
                        decision=completed_turns[-1].decision,
                    )
                except Exception:
                    if adaptive_pipeline.active_attempt_id is not None:
                        context.bridge.abort_attempt(adaptive_memory=memory)
                    context.status = "aborted"
                    self._active_policy_thread_id = None
                    raise
            context.last_turn_pre_history = pre_history
            context.last_turn_result = copied_result
            return copy.deepcopy(copied_result)

    @staticmethod
    def _abort_only_completion(
        context: ActiveAdaptiveContext,
        memory: MutableMapping[str, Any],
    ) -> dict[str, object]:
        """Expose BKT outcome while aborting policy credit and persistence."""

        student_model_result = context.student_model_result
        if not isinstance(student_model_result, Mapping):
            raise TypeError("Controlled-live completion requires student-model output.")
        learning_outcome = student_model_result.get("learning_outcome")
        if not isinstance(learning_outcome, Mapping):
            raise ValueError("Student-model learning_outcome is unavailable.")
        if learning_outcome.get("attempt_id") != context.attempt_id:
            raise ValueError("Student-model attempt_id differs from controlled-live attempt.")
        if learning_outcome.get("skill") != context.target_skill:
            raise ValueError("Student-model skill differs from controlled-live target.")
        required = ("mastery_before", "mastery_after", "delta_mastery")
        missing = [name for name in required if name not in learning_outcome]
        if missing:
            raise ValueError(f"Student-model learning_outcome is missing {missing}.")

        adaptive_pipeline = context.bridge.adaptive_pipeline
        controller = adaptive_pipeline.controller
        turn_count = len(controller.completed_turns)
        mastery_before = float(learning_outcome["mastery_before"])
        mastery_after = float(learning_outcome["mastery_after"])
        delta_mastery = float(learning_outcome["delta_mastery"])
        if abs(mastery_before - context.mastery_before) > 1e-9:
            raise ValueError(
                "Student-model mastery_before differs from attempt start."
            )
        if abs((mastery_after - mastery_before) - delta_mastery) > 1e-9:
            raise ValueError("Student-model delta_mastery is inconsistent.")

        context.bridge.abort_attempt(adaptive_memory=memory)
        return {
            "attempt_id": context.attempt_id,
            "skill": context.target_skill,
            "mastery_before": mastery_before,
            "mastery_after": mastery_after,
            "mastery_delta": delta_mastery,
            "turn_count": turn_count,
            "controlled_live": True,
            "policy_completion": "aborted_without_update",
            "policy_update_suppressed": True,
            "experience_log_write_suppressed": True,
            "policy_state_write_suppressed": True,
        }

    @staticmethod
    def _ordered_evaluated_answers(
        state: Mapping[str, Any],
    ) -> list[dict[str, object]]:
        questions = state.get("assessment_questions", [])
        evaluation = state.get("evaluation_result", {})
        if not isinstance(questions, Sequence) or isinstance(
            questions,
            (str, bytes),
        ):
            raise TypeError("assessment_questions must be a sequence.")
        if not isinstance(evaluation, Mapping):
            raise TypeError("evaluation_result must be a mapping.")

        evaluated_by_id: dict[str, Mapping[str, object]] = {}
        for bucket in ("correct_answers", "wrong_answers"):
            raw_answers = evaluation.get(bucket, [])
            if not isinstance(raw_answers, Sequence) or isinstance(
                raw_answers,
                (str, bytes),
            ):
                raise TypeError(f"evaluation_result.{bucket} must be a sequence.")
            for raw_answer in raw_answers:
                if not isinstance(raw_answer, Mapping):
                    raise TypeError("Each evaluated answer must be a mapping.")
                question_id = raw_answer.get("question_id")
                if not isinstance(question_id, str) or not question_id:
                    raise ValueError("Evaluated answer requires question_id.")
                if question_id in evaluated_by_id:
                    raise ValueError("Assessment answer was evaluated more than once.")
                evaluated_by_id[question_id] = raw_answer

        ordered: list[dict[str, object]] = []
        for raw_question in questions:
            if not isinstance(raw_question, Mapping):
                raise TypeError("Each assessment question must be a mapping.")
            question_id = raw_question.get("question_id")
            if not isinstance(question_id, str) or question_id not in evaluated_by_id:
                raise ValueError("Every assessment question must be evaluated once.")
            evaluated = evaluated_by_id[question_id]
            ordered.append(
                {
                    "question_id": question_id,
                    "question": raw_question.get("question"),
                    "student_answer": evaluated.get("student_answer"),
                    "is_correct": evaluated.get("is_correct"),
                }
            )
        if set(evaluated_by_id) != {
            str(question.get("question_id"))
            for question in questions
            if isinstance(question, Mapping)
        }:
            raise ValueError("Evaluator returned an unknown assessment question.")
        return ordered

    def finish_attempt(
        self,
        state: MutableMapping[str, Any],
    ) -> dict[str, object]:
        memory = self._state_mapping(state)
        with self._lock:
            thread_id = self._identifier(memory.get("thread_id"), "thread_id")
            attempt_id = self._identifier(memory.get("attempt_id"), "attempt_id")
            cache_key = (thread_id, attempt_id)
            cached = self._completed_results.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)

            context = self._contexts.get(thread_id)
            if context is None or context.attempt_id != attempt_id:
                raise AdaptiveRestartRequiredError(
                    "Assessment completion has no matching in-process attempt."
                )
            self._require_active_context(memory)

            if context.student_model_result is None:
                evaluated_answers = self._ordered_evaluated_answers(memory)
                context.student_model_result = (
                    context.student_model_pipeline.process_assessment_cycle(
                        student_id=context.student_id,
                        attempt_id=context.attempt_id,
                        evaluated_answers=evaluated_answers,
                    )
                )

            if (
                self._research_collector is not None
                and not context.research_assessment_recorded
            ):
                try:
                    model_result = context.student_model_result
                    if not isinstance(model_result, Mapping):
                        raise TypeError("Assessment student-model result is invalid.")
                    knowledge_result = model_result.get(
                        "knowledge_graph_result",
                        {},
                    )
                    observation_results = (
                        copy.deepcopy(
                            knowledge_result.get("observation_results", [])
                        )
                        if isinstance(knowledge_result, Mapping)
                        else []
                    )
                    self._research_collector.record_assessment_outcome(
                        {
                            "provenance": {
                                "timestamp": datetime.now(timezone.utc)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "data_mode": self._research_collector.data_mode,
                            },
                            "identity": {
                                "student_pseudonymous_id": (
                                    self._research_collector.pseudonymize_student_id(
                                        context.student_id
                                    )
                                ),
                                "thread_id": context.thread_id,
                                "attempt_id": context.attempt_id,
                                "adaptive_attempt_index": (
                                    context.adaptive_attempt_index
                                ),
                            },
                            "attempt_started_at": context.attempt_started_at,
                            "evidence_type": "formal_three_question_assessment",
                            "evaluated_answers": self._ordered_evaluated_answers(
                                memory
                            ),
                            "resolved_events": [
                                _research_resolved_event(event)
                                for event in model_result.get(
                                    "resolved_events",
                                    [],
                                )
                            ],
                            "observation_results": observation_results,
                            "cumulative_learning_outcome": copy.deepcopy(
                                model_result.get("learning_outcome")
                            ),
                            "dialogue_evidence_summary": copy.deepcopy(
                                model_result.get("dialogue_evidence_summary")
                            ),
                        }
                    )
                    context.research_assessment_recorded = True
                except Exception:
                    logger.exception(
                        "Passive formal-assessment collection failed for %s.",
                        context.attempt_id,
                    )

            if (
                self._bkt_activity_observer is not None
                and not context.bkt_activity_recorded
            ):
                self._bkt_activity_observer(
                    student_id=context.student_id,
                    attempt_id=context.attempt_id,
                    target_skill=context.target_skill,
                    evaluated_answers=self._ordered_evaluated_answers(memory),
                    student_model_result=context.student_model_result,
                )
                context.bkt_activity_recorded = True

            if context.completion_result is None:
                if self._suppress_policy_completion:
                    context.completion_result = self._abort_only_completion(
                        context,
                        memory,
                    )
                    context.status = "aborted"
                    self._active_policy_thread_id = None
                    if self._completion_observer is not None:
                        self._completion_observer(
                            attempt_id=context.attempt_id,
                            turn_count=context.completion_result["turn_count"],
                            mastery_before=context.completion_result[
                                "mastery_before"
                            ],
                            mastery_after=context.completion_result[
                                "mastery_after"
                            ],
                        )
                else:
                    context.completion_result = context.bridge.finish_attempt(
                        adaptive_memory=memory,
                        student_model_result=context.student_model_result,
                    )
                    context.status = "completed"
            self._active_policy_thread_id = None

            if self._research_collector is not None:
                for action_event_id in sorted(
                    context.research_staged_actions
                    - context.research_completed_actions
                ):
                    try:
                        dialogue = context.dialogue_results.get(action_event_id)
                        if isinstance(dialogue, Mapping):
                            self._research_collector.complete_dialogue_action(
                                action_event_id=action_event_id,
                                dialogue_evidence=copy.deepcopy(
                                    dialogue["evidence"]
                                ),
                                student_model_result=copy.deepcopy(
                                    dialogue["student_model_result"]
                                ),
                            )
                        else:
                            finalize_pending = getattr(
                                self._research_collector,
                                "finalize_pending_action",
                                None,
                            )
                            if callable(finalize_pending):
                                finalize_pending(action_event_id)
                            else:
                                self._research_collector.finalize_without_response(
                                    action_event_id,
                                    status="censored_no_response",
                                )
                        context.research_completed_actions.add(action_event_id)
                    except Exception:
                        logger.exception(
                            "Passive terminal action collection failed for %s.",
                            action_event_id,
                        )

            if (
                self._research_collector is not None
                and not context.research_attempt_summary_recorded
            ):
                try:
                    model_result = context.student_model_result
                    self._research_collector.record_attempt_summary(
                        {
                            "provenance": {
                                "timestamp": datetime.now(timezone.utc)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "data_mode": self._research_collector.data_mode,
                            },
                            "identity": {
                                "student_pseudonymous_id": (
                                    self._research_collector.pseudonymize_student_id(
                                        context.student_id
                                    )
                                ),
                                "thread_id": context.thread_id,
                                "attempt_id": context.attempt_id,
                                "adaptive_attempt_index": (
                                    context.adaptive_attempt_index
                                ),
                            },
                            "attempt_started_at": context.attempt_started_at,
                            "completion_status": context.status,
                            "attempt_start_mastery": context.mastery_before,
                            "cumulative_attempt_outcome": copy.deepcopy(
                                model_result.get("learning_outcome")
                                if isinstance(model_result, Mapping)
                                else None
                            ),
                            "adaptive_completion": copy.deepcopy(
                                context.completion_result
                            ),
                        }
                    )
                    context.research_attempt_summary_recorded = True
                except Exception:
                    logger.exception(
                        "Passive attempt-summary collection failed for %s.",
                        context.attempt_id,
                    )

            completed_attempt_id = context.attempt_id
            result: dict[str, object] = {
                "completed_attempt_id": completed_attempt_id,
                "adaptive_completion_result": copy.deepcopy(
                    context.completion_result
                ),
                "adaptive_lifecycle_status": "completed",
                "attempt_started_at": context.attempt_started_at,
            }

            if bool(memory.get("needs_reteaching", False)):
                next_index = context.adaptive_attempt_index + 1
                next_attempt_id = self.derive_attempt_id(thread_id, next_index)
                next_memory = dict(memory)
                next_memory["adaptive_attempt_index"] = next_index
                next_memory["attempt_id"] = next_attempt_id
                next_memory["adaptive_lifecycle_status"] = "not_started"
                next_memory.pop("mastery_before", None)
                started = self._start_locked(next_memory)
                result.update(started)
            else:
                result.update(
                    {
                        "attempt_id": completed_attempt_id,
                        "adaptive_attempt_index": context.adaptive_attempt_index,
                        "mastery_before": context.mastery_before,
                    }
                )

            self._completed_results[cache_key] = copy.deepcopy(result)
            return copy.deepcopy(result)

    def abort_attempt(
        self,
        state: MutableMapping[str, Any],
    ) -> dict[str, object]:
        memory = self._state_mapping(state)
        with self._lock:
            context = self._require_active_context(memory)
            if self._research_collector is not None:
                for action_event_id in sorted(
                    context.research_staged_actions
                    - context.research_completed_actions
                ):
                    try:
                        finalize_pending = getattr(
                            self._research_collector,
                            "finalize_pending_action",
                            None,
                        )
                        if callable(finalize_pending):
                            finalize_pending(action_event_id)
                        else:
                            self._research_collector.finalize_without_response(
                                action_event_id,
                                status="censored_no_response",
                            )
                        context.research_completed_actions.add(action_event_id)
                    except Exception:
                        logger.exception(
                            "Passive censoring failed for %s.",
                            action_event_id,
                        )
            context.bridge.abort_attempt(adaptive_memory=memory)
            context.status = "aborted"
            self._active_policy_thread_id = None
            if (
                self._research_collector is not None
                and not context.research_attempt_summary_recorded
            ):
                try:
                    model_result = context.student_model_result
                    self._research_collector.record_attempt_summary(
                        {
                            "provenance": {
                                "timestamp": _utc_now(),
                                "data_mode": self._research_collector.data_mode,
                            },
                            "identity": {
                                "student_pseudonymous_id": (
                                    self._research_collector.pseudonymize_student_id(
                                        context.student_id
                                    )
                                ),
                                "thread_id": context.thread_id,
                                "attempt_id": context.attempt_id,
                                "adaptive_attempt_index": (
                                    context.adaptive_attempt_index
                                ),
                            },
                            "attempt_started_at": context.attempt_started_at,
                            "completion_status": context.status,
                            "attempt_start_mastery": context.mastery_before,
                            "cumulative_attempt_outcome": copy.deepcopy(
                                model_result.get("learning_outcome")
                                if isinstance(model_result, Mapping)
                                else None
                            ),
                            "adaptive_completion": copy.deepcopy(
                                context.completion_result
                            ),
                        }
                    )
                    context.research_attempt_summary_recorded = True
                except Exception:
                    logger.exception(
                        "Passive aborted-attempt summary collection failed for %s.",
                        context.attempt_id,
                    )
            return {
                "adaptive_lifecycle_status": "aborted",
                "attempt_started_at": context.attempt_started_at,
            }


class _ProductionAdaptiveResources:
    """Lazily construct shared real scientific resources from sibling repos."""

    def __init__(self) -> None:
        workspace_root = Path(__file__).resolve().parents[4]
        self.workspace_root = workspace_root.resolve()
        self.student_model_root = Path(
            os.getenv(
                "STUDENT_MODEL_REPO",
                workspace_root / "student-modeling",
            )
        ).resolve()
        self.move_selector_root = Path(
            os.getenv(
                "PEDAGOGICAL_MOVE_REPO",
                workspace_root / "pedagogical-move-selection",
            )
        ).resolve()
        backend_root = Path(__file__).resolve().parents[2]
        runtime_root = backend_root / "runtime"
        self.selector_mode = resolve_selector_mode()
        self.manual_controlled_live = (
            self.selector_mode is AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE
        )
        self.manual_output_dir = (
            self.move_selector_root
            / "results"
            / "md7r1_manual_controlled_live_v1"
        ).resolve()
        self.research_output_dir = Path(
            os.getenv(
                "ADAPTIVE_TURN_DATA_DIR",
                self.move_selector_root
                / "results"
                / "md_self_improvement_turn_data_v1",
            )
        ).resolve()
        self.md6_model_dir = (
            self.move_selector_root / "models" / "frozen" / "md6"
        ).resolve()
        self.md7_model_dir = (
            self.move_selector_root
            / "models"
            / "frozen"
            / "md7_r2_tell_c1_epoch2"
        ).resolve()
        self.md7r1_rollback_model_dir = (
            self.move_selector_root / "models" / "candidates" / "md7r1_epoch3"
        ).resolve()
        legacy_policy_path = os.getenv("ADAPTIVE_POLICY_STATE_PATH")
        legacy_experience_path = os.getenv("ADAPTIVE_EXPERIENCE_LOG_PATH")
        self.md6_policy_state_path = Path(
            os.getenv(
                "ADAPTIVE_MD6_POLICY_STATE_PATH",
                legacy_policy_path or runtime_root / "frozen_v3_policy_state.json",
            )
        ).resolve()
        self.md6_experience_log_path = Path(
            os.getenv(
                "ADAPTIVE_MD6_EXPERIENCE_LOG_PATH",
                legacy_experience_path
                or runtime_root / "frozen_v3_experience.jsonl",
            )
        ).resolve()
        md7_runtime_root = runtime_root / "adaptive_demo_md7r1_v1"
        self.md7_policy_state_path = Path(
            os.getenv(
                "ADAPTIVE_MD7_POLICY_STATE_PATH",
                md7_runtime_root / "policy_state.json",
            )
        ).resolve()
        self.md7_experience_log_path = Path(
            os.getenv(
                "ADAPTIVE_MD7_EXPERIENCE_LOG_PATH",
                md7_runtime_root / "attempts.jsonl",
            )
        ).resolve()
        if self.selector_mode.uses_md7_policy_lineage and not self.selector_mode.uses_turn_lints:
            missing_md7_names = [
                name
                for name in (
                    "ADAPTIVE_MD7_POLICY_STATE_PATH",
                    "ADAPTIVE_MD7_EXPERIENCE_LOG_PATH",
                )
                if not os.getenv(name)
            ]
            legacy_names = [
                name
                for name in (
                    "ADAPTIVE_POLICY_STATE_PATH",
                    "ADAPTIVE_EXPERIENCE_LOG_PATH",
                )
                if os.getenv(name)
            ]
            if missing_md7_names and legacy_names:
                raise ValueError(
                    "MD7 mode will not infer its fresh policy lineage from "
                    f"legacy settings {legacy_names}; configure "
                    f"{missing_md7_names} explicitly or remove the legacy settings."
                )
        if self.selector_mode.uses_turn_lints:
            self.turn_lints_mode = os.getenv(
                "ADAPTIVE_TURN_LINTS_MODE", "SHADOW"
            ).strip().upper()
            if self.turn_lints_mode not in {
                "OFF", "SHADOW", "RANDOMIZED_WARMSTART", "LIVE"
            }:
                raise ValueError(
                    "ADAPTIVE_TURN_LINTS_MODE must be OFF, SHADOW, "
                    "RANDOMIZED_WARMSTART, or LIVE."
                )
            self.turn_lints_blocks = os.getenv(
                "ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS", "S+K+L"
            ).strip()
            self.turn_lints_reward_mode = os.getenv(
                "ADAPTIVE_TURN_LINTS_REWARD_MODE", "headroom_normalized"
            ).strip()
            self.turn_lints_anchor_mode = os.getenv(
                "ADAPTIVE_TURN_LINTS_ANCHOR_MODE", "none"
            ).strip()
            self.turn_lints_anchor_gamma = float(
                os.getenv("ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA", "0.0")
            )
            turn_root_name = {
                "RANDOMIZED_WARMSTART": "turn_lints_randomized_warmstart_skl_final_v1",
                "LIVE": "turn_lints_live_skl_final_v1",
            }.get(self.turn_lints_mode, "turn_lints_shadow_skl_final_v1")
            turn_root = runtime_root / turn_root_name
            self.turn_lints_state_path = Path(
                os.getenv(
                    "ADAPTIVE_TURN_LINTS_STATE_PATH",
                    turn_root / "policy_state.json",
                )
            ).resolve()
            self.turn_lints_event_root = Path(
                os.getenv("ADAPTIVE_TURN_LINTS_EVENT_ROOT", turn_root)
            ).resolve()
            self.turn_lints_live_manifest_path = Path(
                os.getenv(
                    "ADAPTIVE_TURN_LINTS_LIVE_MANIFEST_PATH",
                    turn_root / "lineage_manifest.json",
                )
            ).resolve()
            self.policy_state_path = self.turn_lints_state_path
            self.experience_log_path = self.turn_lints_event_root / "turn_events.jsonl"
            self.policy_lineage = "direct_disjoint_turn_lints_v1"
        elif self.selector_mode.uses_md7_policy_lineage:
            self.policy_state_path = self.md7_policy_state_path
            self.experience_log_path = self.md7_experience_log_path
            self.policy_lineage = "MD7-R1 fresh LinTS v1"
        else:
            self.policy_state_path = self.md6_policy_state_path
            self.experience_log_path = self.md6_experience_log_path
            self.policy_lineage = "preserved MD6"
        self.pipeline_policy_state_path = (
            self.manual_output_dir / "_policy_persistence_disabled.json"
            if self.manual_controlled_live
            else self.policy_state_path
        )
        self.pipeline_experience_log_path = (
            self.manual_output_dir / "_experience_persistence_disabled.jsonl"
            if self.manual_controlled_live
            else self.experience_log_path
        )
        self.data_mode = os.getenv("ADAPTIVE_DATA_MODE", "real").strip()
        if self.data_mode not in {"synthetic", "real"}:
            raise ValueError(
                "ADAPTIVE_DATA_MODE must be exactly 'synthetic' or 'real'."
            )
        self._lock = RLock()
        self._predictor: Any = None
        self._shared_ready = False
        self.manual_diagnostics: ManualControlledLiveDiagnostics | None = None
        self.live_bkt_activity: LiveBKTActivityLog | None = None
        self.turn_collector: Any = None
        if self.selector_mode is AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE:
            print(
                "\n".join(
                    (
                        "=" * 60,
                        "MANUAL CONTROLLED-LIVE SELECTOR TEST",
                        "ACTIVE: MD7-R1 epoch 3",
                        "SHADOW: frozen MD6",
                        "Learner-agency telling escalation: ENABLED",
                        "LinTS posterior updates: DISABLED",
                        "Real adaptive persistence: DISABLED",
                        "=" * 60,
                    )
                ),
                flush=True,
            )
        elif self.selector_mode is AdaptiveSelectorMode.MD6_ROLLBACK:
            print(
                "\n".join(
                    (
                        "=" * 60,
                        "ROLLBACK MODE",
                        "ACTIVE SELECTOR: frozen MD6",
                        "POLICY LINEAGE: preserved MD6",
                        "Learner-agency telling escalation: ENABLED",
                        "LinTS learning: ENABLED",
                        "=" * 60,
                    )
                ),
                flush=True,
            )
        elif self.selector_mode is AdaptiveSelectorMode.MD7_R1_ROLLBACK:
            print(
                "\n".join(
                    (
                        "=" * 60,
                        "ROLLBACK MODE",
                        "ACTIVE SELECTOR: MD7-R1 epoch 3",
                        "POLICY LINEAGE: MD7-R1 fresh LinTS v1",
                        "LEGACY TAU/BIAS/DELAYED-CREDIT SEMANTICS: ENABLED",
                        "=" * 60,
                    )
                ),
                flush=True,
            )
        else:
            print(
                "\n".join(
                    (
                        "=" * 60,
                        "ACTIVE SELECTOR: MD7-R2-TELL-C1 Epoch 2",
                        "SELECTOR SHA256: ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5",
                        f"TURN-LINTS MODE: {self.turn_lints_mode}",
                        "POLICY LINEAGE: direct_disjoint_turn_lints_v1",
                        f"CONTEXT BLOCKS: {self.turn_lints_blocks}",
                        f"REWARD MODE: {self.turn_lints_reward_mode}",
                        f"ACTION ANCHOR: {self.turn_lints_anchor_mode}",
                        f"ANCHOR GAMMA: {self.turn_lints_anchor_gamma}",
                        f"POSTERIOR PATH: {self.turn_lints_state_path}",
                        "=" * 60,
                    )
                ),
                flush=True,
            )

    def _install_import_roots(self) -> None:
        for root, label in (
            (self.student_model_root, "student-modeling"),
            (self.move_selector_root, "pedagogical-move-selection"),
        ):
            if not root.is_dir():
                raise FileNotFoundError(f"{label} repository not found: {root}")
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_predictor(self) -> Any:
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            self._install_import_roots()
            from bkt.predict import BKTPredictor

            self._predictor = BKTPredictor.load(
                self.student_model_root / "models" / "bkt_params.json"
            )
            return self._predictor

    def validate_target_skill(self, target_skill: str) -> bool:
        return bool(self._ensure_predictor().has_skill(target_skill))

    def _build_base_selector(self, model_loader: Any) -> Any:
        """Build the selector required by the resolved selector/policy mode."""

        from src.self_improvement.learner_agency_escalation import (
            LearnerAgencyEscalationSelector,
        )

        if self.selector_mode is AdaptiveSelectorMode.MD6_ROLLBACK:
            self.active_model_path = self.md6_model_dir
            self.raw_active_selector = model_loader()
            self.active_selector = LearnerAgencyEscalationSelector(
                self.raw_active_selector,
            )
            return self.active_selector

        if self.selector_mode is AdaptiveSelectorMode.NORMAL_MD7:
            self.active_model_path = self.md7_model_dir
            self.raw_active_selector = model_loader(self.md7_model_dir)
            self.active_selector = self.raw_active_selector
            return self.active_selector

        self.active_model_path = self.md7r1_rollback_model_dir
        active_md7 = model_loader(self.md7r1_rollback_model_dir)
        self.active_md7 = active_md7
        if self.selector_mode is AdaptiveSelectorMode.MD7_R1_ROLLBACK:
            self.active_selector = LearnerAgencyEscalationSelector(active_md7)
            return self.active_selector

        from src.self_improvement.controlled_live_selector import (
            ActiveMD7WithMD6Shadow,
        )

        frozen_md6 = model_loader()
        self.shadow_md6 = frozen_md6
        paired_selector = ActiveMD7WithMD6Shadow(
            active_md7=active_md7,
            shadow_md6=frozen_md6,
        )
        self.active_selector = LearnerAgencyEscalationSelector(
            paired_selector,
            retain_records=True,
        )
        return self.active_selector

    def _ensure_shared(self) -> None:
        with self._lock:
            if self._shared_ready:
                return
            self._install_import_roots()

            from core.detector_service import DetectorService
            from core.knowledge_graph import KnowledgeGraph
            from src.self_improvement.md6_inference import FrozenMD6Inference
            from src.self_improvement.mrb1_inference import FrozenMRB1Inference
            from src.self_improvement.context_builder import MOVE_ORDER

            self.knowledge_graph = KnowledgeGraph(
                predictor=self._ensure_predictor()
            )
            self.detectors = DetectorService.from_project_defaults(
                self.student_model_root
            )
            self.md6 = self._build_base_selector(FrozenMD6Inference)
            self.mrb1 = FrozenMRB1Inference()
            active_weights = self.active_model_path / "model.safetensors"
            selector_sha = self._sha256_file(active_weights)
            if self.selector_mode.uses_turn_lints:
                expected_sha = (
                    "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
                )
                if selector_sha != expected_sha:
                    raise RuntimeError("Promoted MD7-R2-TELL-C1 weight hash mismatch.")
                from src.self_improvement.turn_lints_context import (
                    context_schema_id,
                    feature_names_for_blocks,
                    parse_context_blocks,
                )
                from src.self_improvement.turn_lints_policy import (
                    ALGORITHM_VERSION,
                    DirectTurnLinTS,
                    load_turn_lints_state,
                    save_turn_lints_state,
                )
                from src.self_improvement.turn_lints_runtime import (
                    TurnActionLedger,
                    TurnLinTSMode,
                )
                from src.self_improvement.postgres_repository import (
                    postgres_enabled as research_postgres_enabled,
                )

                blocks = parse_context_blocks(self.turn_lints_blocks)
                ridge = float(os.getenv("ADAPTIVE_TURN_LINTS_RIDGE_LAMBDA", "1.0"))
                exploration = float(
                    os.getenv("ADAPTIVE_TURN_LINTS_EXPLORATION_SCALE", "0.20")
                )
                self.turn_lints_mode_value = TurnLinTSMode.parse(
                    self.turn_lints_mode
                )
                self.policy = DirectTurnLinTS(
                    context_schema_id=context_schema_id(blocks),
                    enabled_context_blocks=blocks,
                    feature_names=feature_names_for_blocks(blocks),
                    selector_version="MD7-R2-TELL-C1 Epoch 2",
                    selector_sha256=selector_sha,
                    reward_mode=self.turn_lints_reward_mode,
                    anchor_mode=self.turn_lints_anchor_mode,
                    anchor_gamma=self.turn_lints_anchor_gamma,
                    anchor_selector_version="MD7-R2-TELL-C1 Epoch 2",
                    anchor_selector_sha256=selector_sha,
                    ridge_lambda=ridge,
                    exploration_scale=exploration,
                    seed=int(os.getenv("ADAPTIVE_TURN_LINTS_SEED", "42")),
                    data_mode=self.data_mode,
                )
                if research_postgres_enabled() or self.policy_state_path.exists():
                    load_turn_lints_state(self.policy, self.policy_state_path)
                else:
                    if self.turn_lints_mode_value is TurnLinTSMode.LIVE:
                        raise RuntimeError(
                            "LIVE Turn-LinTS refuses to create a blank posterior; "
                            "the audited state is required."
                        )
                    save_turn_lints_state(self.policy, self.policy_state_path)
                if self.turn_lints_mode_value is TurnLinTSMode.LIVE:
                    from src.self_improvement.turn_lints_live_lineage import (
                        validate_live_lineage,
                    )

                    self.turn_lints_live_counters = validate_live_lineage(
                        self.policy,
                        self.turn_lints_live_manifest_path,
                        state_path=self.policy_state_path,
                    )
                self.turn_lints_ledger = TurnActionLedger(self.turn_lints_event_root)
                print(
                    "TURN-LINTS INITIALIZED: "
                    f"mode={self.turn_lints_mode}; "
                    f"schema={self.policy.context_schema_id}; "
                    f"dimension={self.policy.context_dim}; "
                    f"blocks={'+'.join(self.policy.enabled_context_blocks)}; "
                    f"reward={self.policy.reward_mode}; "
                    f"anchor={self.policy.anchor_mode}; "
                    f"gamma={self.policy.anchor_gamma}; "
                    f"state={self.policy_state_path}",
                    flush=True,
                )
                self.experience_logger = None
                lints_version = ALGORITHM_VERSION
                context_version = self.policy.context_schema_id
                tau_value = None
                learner_agency_version = "disabled_for_direct_turn_lints"
            else:
                from src.self_improvement.experience_logger import ExperienceLogger
                from src.self_improvement.lints_policy import (
                    LINTS_STATE_SCHEMA_VERSION,
                    TrueDisjointLinTS,
                )
                from src.self_improvement.state_io import load_policy_state
                from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES
                from src.self_improvement.postgres_repository import (
                    postgres_enabled as research_postgres_enabled,
                )

                self.policy = TrueDisjointLinTS(
                    context_dim=len(TURN_FEATURE_NAMES), seed=42,
                    data_mode=self.data_mode,
                )
                if research_postgres_enabled() or self.policy_state_path.exists():
                    load_policy_state(
                        self.policy, self.policy_state_path,
                        expected_data_mode=self.data_mode,
                    )
                self.experience_logger = ExperienceLogger(
                    self.pipeline_experience_log_path,
                    data_mode=self.data_mode,
                    source_policy=self.policy,
                )
                lints_version = LINTS_STATE_SCHEMA_VERSION
                context_version = "turn_lints_v3_c3_9d"
                tau_value = 0.10
                learner_agency_version = "learner_agency_telling_escalation_v1"
            from app.integrations.self_improvement_turn_collector import (
                PassiveTurnCollector,
            )

            mrb1_weights = self.move_selector_root / "models" / "frozen" / "mrb1" / "model.safetensors"
            bkt_config = self.student_model_root / "models" / "bkt_params.json"
            try:
                relative_policy_path: str | None = self.pipeline_policy_state_path.relative_to(
                    self.workspace_root
                ).as_posix()
            except ValueError:
                relative_policy_path = None
            self.turn_collector = PassiveTurnCollector(
                self.research_output_dir,
                data_mode=self.data_mode,
                preserve_pending_for_restart=self.manual_controlled_live,
                version_provenance={
                    "runtime": {
                        "selector_mode": self.selector_mode.value,
                        "selector_deployment_status": "active_runtime",
                        "learner_agency_version": learner_agency_version,
                        "lints_policy_lineage": self.policy_lineage,
                        "lints_policy_version": lints_version,
                        "context_schema": context_version,
                        "tau": tau_value,
                        "turn_lints_mode": (
                            self.turn_lints_mode if self.selector_mode.uses_turn_lints else None
                        ),
                        "move_order": list(MOVE_ORDER),
                        "resolver_version": "2.0",
                        "bkt_config_version": "confidence_weighted_bkt_v1",
                        "bkt_config_sha256": self._sha256_file(bkt_config),
                    },
                    "selector": {
                        "checkpoint": self.active_model_path.name,
                        "model_sha256": self._sha256_file(active_weights),
                        "selector_mode": self.selector_mode.value,
                        "deployment_status": "active_runtime",
                        "learner_agency_version": learner_agency_version,
                        "lints_policy_lineage": self.policy_lineage,
                        "lints_policy_version": lints_version,
                        "context_schema": context_version,
                        "tau": tau_value,
                        "move_order": list(MOVE_ORDER),
                    },
                    "tutor_quality": {
                        "model_version": "frozen_mrb1",
                        "model_sha256": self._sha256_file(mrb1_weights),
                    },
                    "policy": {
                        "policy_lineage": self.policy_lineage,
                        "policy_state_identifier": (
                            f"{self.selector_mode.value}:{self.policy_lineage}"
                        ),
                        "policy_state_relative_path": relative_policy_path,
                        "turn_lints_mode": (
                            self.turn_lints_mode if self.selector_mode.uses_turn_lints else None
                        ),
                        "context_blocks": (
                            list(self.policy.enabled_context_blocks)
                            if self.selector_mode.uses_turn_lints else None
                        ),
                        "context_dimension": (
                            self.policy.context_dim
                            if self.selector_mode.uses_turn_lints else None
                        ),
                        "ordered_feature_names": (
                            list(self.policy.feature_names)
                            if self.selector_mode.uses_turn_lints else None
                        ),
                        "reward_mode": (
                            self.policy.reward_mode
                            if self.selector_mode.uses_turn_lints else None
                        ),
                    },
                },
            )
            if self.manual_controlled_live:
                self.manual_diagnostics = ManualControlledLiveDiagnostics(
                    selector=self.md6,
                    policy=self.policy,
                    output_dir=self.manual_output_dir,
                    production_policy_path=self.policy_state_path,
                    production_experience_path=self.experience_log_path,
                )
            else:
                try:
                    self.live_bkt_activity = LiveBKTActivityLog(
                        self.research_output_dir,
                    )
                except Exception:
                    logger.exception(
                        "Passive LIVE BKT activity log initialization failed."
                    )
            self._shared_ready = True

    def _collector(self) -> Any:
        self._ensure_shared()
        if self.turn_collector is None:
            raise RuntimeError("Passive turn collector is unavailable.")
        return self.turn_collector

    def stage_runtime_action(self, **kwargs: object) -> object:
        return self._collector().stage_runtime_action(**kwargs)

    def finalized_actions_for_attempt(
        self,
        attempt_id: str,
    ) -> list[dict[str, object]]:
        return self._collector().finalized_actions_for_attempt(attempt_id)

    def complete_dialogue_action(self, **kwargs: object) -> object:
        return self._collector().complete_dialogue_action(**kwargs)

    def note_processing_failure(self, *args: object, **kwargs: object) -> object:
        return self._collector().note_processing_failure(*args, **kwargs)

    def finalize_pending_action(self, action_event_id: str) -> object:
        return self._collector().finalize_pending_action(action_event_id)

    def finalize_without_response(self, *args: object, **kwargs: object) -> object:
        return self._collector().finalize_without_response(*args, **kwargs)

    def record_assessment_outcome(self, record: Mapping[str, object]) -> object:
        return self._collector().record_assessment_outcome(record)

    def record_attempt_summary(self, record: Mapping[str, object]) -> object:
        return self._collector().record_attempt_summary(record)

    def pseudonymize_student_id(self, student_id: object) -> str:
        return self._collector().pseudonymize_student_id(student_id)

    def record_manual_turn(self, **turn: object) -> None:
        if not self.manual_controlled_live:
            raise RuntimeError("Manual turn observer used in ordinary mode.")
        if self.manual_diagnostics is None:
            raise RuntimeError("Manual controlled-live diagnostics are unavailable.")
        self.manual_diagnostics.record_turn(**turn)

    def record_manual_completion(self, **completion: object) -> None:
        if not self.manual_controlled_live:
            raise RuntimeError("Manual completion observer used in ordinary mode.")
        if self.manual_diagnostics is None:
            raise RuntimeError("Manual controlled-live diagnostics are unavailable.")
        self.manual_diagnostics.record_attempt_completion(**completion)

    def record_bkt_activity(self, **activity: object) -> None:
        if self.manual_controlled_live:
            if self.manual_diagnostics is None:
                raise RuntimeError(
                    "Manual controlled-live diagnostics are unavailable."
                )
            self.manual_diagnostics.record_bkt_activity(
                **activity,
                knowledge_graph=self.knowledge_graph,
            )
            return
        if self.live_bkt_activity is None:
            logger.error("LIVE assessment BKT activity logging is unavailable.")
            return
        try:
            self.live_bkt_activity.record_bkt_activity(
                **activity,
                knowledge_graph=self.knowledge_graph,
            )
        except Exception:
            logger.exception("Passive assessment BKT activity logging failed.")

    def record_dialogue_bkt_activity(self, **activity: object) -> None:
        if self.manual_controlled_live:
            if self.manual_diagnostics is None:
                raise RuntimeError(
                    "Manual controlled-live diagnostics are unavailable."
                )
            self.manual_diagnostics.record_dialogue_bkt_activity(
                **activity,
                knowledge_graph=self.knowledge_graph,
            )
            return
        if self.live_bkt_activity is None:
            logger.error("LIVE dialogue BKT activity logging is unavailable.")
            return
        try:
            self.live_bkt_activity.record_dialogue_bkt_activity(
                **activity,
                knowledge_graph=self.knowledge_graph,
            )
        except Exception:
            logger.exception("Passive dialogue BKT activity logging failed.")

    def build_components(
        self,
        *,
        student_id: str,
        thread_id: str,
        attempt_id: str,
        target_skill: str,
        adaptive_attempt_index: int,
    ) -> AdaptiveAttemptComponents:
        del student_id, thread_id, attempt_id, adaptive_attempt_index
        self._ensure_shared()

        from core.cross_session_pipeline import CrossSessionStudentModelPipeline
        from src.integration.student_model_v3_bridge import (
            StudentModelFrozenV3Bridge,
        )
        evidence_source = _AssessmentEvidenceSource(target_skill)
        cross_session = CrossSessionStudentModelPipeline(
            concept_extractor=evidence_source,
            evaluator=evidence_source,
            detectors=self.detectors,
            knowledge_graph=self.knowledge_graph,
            curriculum=None,
        )
        student_pipeline = TutorAssessmentStudentModelPipeline(
            target_skill=target_skill,
            cross_session_pipeline=cross_session,
            evidence_source=evidence_source,
        )
        memory_adapter = TutorStateMemoryAdapter()
        if self.selector_mode.uses_turn_lints:
            from src.self_improvement.turn_lints_runtime import (
                DirectTurnController,
                DirectTurnLinTSPipeline,
            )

            adaptive_pipeline = DirectTurnLinTSPipeline(
                selector=self.md6,
                mrb1=self.mrb1,
                controller=DirectTurnController(
                    policy=self.policy,
                    mode=self.turn_lints_mode_value,
                    enabled_context_blocks=self.policy.enabled_context_blocks,
                    state_path=self.policy_state_path,
                    ledger=self.turn_lints_ledger,
                ),
                memory_adapter=memory_adapter,
            )
        else:
            from src.self_improvement.adaptive_tutor_pipeline import (
                AdaptiveTutorPipeline,
            )
            from src.self_improvement.turn_level_controller import (
                TurnLevelAttemptController,
            )

            adaptive_pipeline = AdaptiveTutorPipeline(
                md6=self.md6,
                mrb1=self.mrb1,
                controller=TurnLevelAttemptController(self.policy),
                experience_logger=self.experience_logger,
                policy_state_path=self.pipeline_policy_state_path,
                memory_adapter=memory_adapter,
            )
        bridge = StudentModelFrozenV3Bridge(
            student_model_pipeline=student_pipeline,
            adaptive_pipeline=adaptive_pipeline,
        )
        return AdaptiveAttemptComponents(
            bridge=bridge,
            student_model_pipeline=student_pipeline,
            memory_adapter=memory_adapter,
        )


def build_default_adaptive_component_coordinator() -> AdaptiveComponentCoordinator:
    resources = _ProductionAdaptiveResources()
    return AdaptiveComponentCoordinator(
        component_factory=resources.build_components,
        skill_validator=resources.validate_target_skill,
        suppress_policy_completion=resources.manual_controlled_live,
        turn_observer=(
            resources.record_manual_turn
            if resources.manual_controlled_live
            else None
        ),
        completion_observer=(
            resources.record_manual_completion
            if resources.manual_controlled_live
            else None
        ),
        bkt_activity_observer=(
            resources.record_bkt_activity
        ),
        dialogue_bkt_activity_observer=(
            resources.record_dialogue_bkt_activity
        ),
        research_collector=resources,
    )


__all__ = (
    "ActiveAdaptiveContext",
    "AdaptiveAttemptComponents",
    "AdaptiveComponentCoordinator",
    "AdaptiveConcurrencyError",
    "AdaptiveIntegrationError",
    "AdaptiveRestartRequiredError",
    "DIALOGUE_EVALUATOR_CONFIDENCE_CAP",
    "TutorAssessmentStudentModelPipeline",
    "build_default_adaptive_component_coordinator",
)
