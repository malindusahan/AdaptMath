"""Lifecycle boundary for Tutor, student modeling, and frozen-v3 policy."""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Literal

from app.integrations.adaptive_tutor_agent_adapter import (
    AdaptiveTutorAgentAdapter,
)
from app.integrations.tutor_state_memory_adapter import (
    TutorStateMemoryAdapter,
)


AdaptiveLifecycleStatus = Literal[
    "not_started",
    "active",
    "completed",
    "aborted",
]


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


class _AssessmentEvidenceSource:
    """Project Tutor evaluator judgments into student-model event evidence."""

    def __init__(self, target_skill: str) -> None:
        self.target_skill = target_skill
        self.context = SimpleNamespace(assessed_skills=(target_skill,))
        self._verdicts: dict[int, bool] = {}

    def build_transcript(
        self,
        evaluated_answers: Sequence[Mapping[str, object]],
    ) -> list[dict[str, str]]:
        transcript: list[dict[str, str]] = []
        verdicts: dict[int, bool] = {}

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
            verdicts[student_turn_index] = is_correct

        if not transcript:
            raise ValueError("Assessment completion requires evaluated answers.")
        self._verdicts = verdicts
        return transcript

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
        is_correct = self._verdicts[turn_index]
        return {
            "correctness": "correct" if is_correct else "incorrect",
            "confidence": 1.0,
            "source": "adaptmath_structured_assessment_evaluator",
        }


class TutorAssessmentStudentModelPipeline:
    """Bind one target skill and Tutor assessment evidence to Repository B."""

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
        return context

    def process_assessment_cycle(
        self,
        *,
        student_id: str,
        attempt_id: str,
        evaluated_answers: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        if self._attempt_context is None:
            raise RuntimeError("Student-model adaptive attempt has not started.")
        transcript = self._evidence_source.build_transcript(evaluated_answers)
        return self._pipeline.process_transcript(
            transcript=transcript,
            student_id=student_id,
            session_id=attempt_id,
            behavioural_skill=self.target_skill,
            attempt_context=self._attempt_context,
        )


ComponentFactory = Callable[..., AdaptiveAttemptComponents]
SkillValidator = Callable[[str], bool | None]


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
    ) -> None:
        if not callable(component_factory):
            raise TypeError("component_factory must be callable.")
        if not callable(skill_validator):
            raise TypeError("skill_validator must be callable.")
        self._component_factory = component_factory
        self._skill_validator = skill_validator
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
        }

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
                raise AdaptiveRestartRequiredError(
                    "Persisted adaptive attempt cannot be reconstructed after a "
                    "process restart without changing frozen policy semantics."
                )
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
            raise AdaptiveRestartRequiredError(
                "Persisted TutorState has no matching in-process adaptive context."
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
            try:
                result = adaptive_pipeline.run_tutor_turn(
                    memory,
                    adapter,
                )
            except Exception:
                # A selected/recorded turn cannot be regenerated without
                # changing Thompson-sampling causality or duplicating quality
                # evidence. Abandon only that non-retryable attempt; failures
                # before selection leave the context active for a safe retry.
                turn_advanced = (
                    controller.awaiting_mrb1
                    or len(controller.completed_turns) > completed_before
                )
                if turn_advanced and adaptive_pipeline.active_attempt_id is not None:
                    adaptive_pipeline.abort_attempt(memory)
                    context.status = "aborted"
                    self._active_policy_thread_id = None
                raise
            copied_result = copy.deepcopy(dict(result))
            context.last_turn_pre_history = pre_history
            context.last_turn_result = copied_result
            return copy.deepcopy(copied_result)

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
                context.student_model_result = (
                    context.student_model_pipeline.process_assessment_cycle(
                        student_id=context.student_id,
                        attempt_id=context.attempt_id,
                        evaluated_answers=self._ordered_evaluated_answers(memory),
                    )
                )

            if context.completion_result is None:
                context.completion_result = context.bridge.finish_attempt(
                    adaptive_memory=memory,
                    student_model_result=context.student_model_result,
                )
            context.status = "completed"
            self._active_policy_thread_id = None

            completed_attempt_id = context.attempt_id
            result: dict[str, object] = {
                "completed_attempt_id": completed_attempt_id,
                "adaptive_completion_result": copy.deepcopy(
                    context.completion_result
                ),
                "adaptive_lifecycle_status": "completed",
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
            context.bridge.adaptive_pipeline.abort_attempt(memory)
            context.status = "aborted"
            self._active_policy_thread_id = None
            return {"adaptive_lifecycle_status": "aborted"}


class _ProductionAdaptiveResources:
    """Lazily construct shared real scientific resources from sibling repos."""

    def __init__(self) -> None:
        workspace_root = Path(__file__).resolve().parents[4]
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
        self.policy_state_path = Path(
            os.getenv(
                "ADAPTIVE_POLICY_STATE_PATH",
                runtime_root / "frozen_v3_policy_state.json",
            )
        ).resolve()
        self.experience_log_path = Path(
            os.getenv(
                "ADAPTIVE_EXPERIENCE_LOG_PATH",
                runtime_root / "frozen_v3_experience.jsonl",
            )
        ).resolve()
        self.data_mode = os.getenv("ADAPTIVE_DATA_MODE", "real").strip()
        if self.data_mode not in {"synthetic", "real"}:
            raise ValueError(
                "ADAPTIVE_DATA_MODE must be exactly 'synthetic' or 'real'."
            )
        self._lock = RLock()
        self._predictor: Any = None
        self._shared_ready = False

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

    def _ensure_shared(self) -> None:
        with self._lock:
            if self._shared_ready:
                return
            self._install_import_roots()

            from core.detector_service import DetectorService
            from core.knowledge_graph import KnowledgeGraph
            from src.self_improvement.experience_logger import ExperienceLogger
            from src.self_improvement.lints_policy import TrueDisjointLinTS
            from src.self_improvement.md6_inference import FrozenMD6Inference
            from src.self_improvement.mrb1_inference import FrozenMRB1Inference
            from src.self_improvement.state_io import load_policy_state
            from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES

            self.knowledge_graph = KnowledgeGraph(
                predictor=self._ensure_predictor()
            )
            self.detectors = DetectorService.from_project_defaults(
                self.student_model_root
            )
            self.policy = TrueDisjointLinTS(
                context_dim=len(TURN_FEATURE_NAMES),
                seed=42,
                data_mode=self.data_mode,
            )
            if self.policy_state_path.exists():
                load_policy_state(
                    self.policy,
                    self.policy_state_path,
                    expected_data_mode=self.data_mode,
                )
            self.experience_logger = ExperienceLogger(
                self.experience_log_path,
                data_mode=self.data_mode,
                source_policy=self.policy,
            )
            self.md6 = FrozenMD6Inference()
            self.mrb1 = FrozenMRB1Inference()
            self._shared_ready = True

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
        from src.self_improvement.adaptive_tutor_pipeline import (
            AdaptiveTutorPipeline,
        )
        from src.self_improvement.turn_level_controller import (
            TurnLevelAttemptController,
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
        adaptive_pipeline = AdaptiveTutorPipeline(
            md6=self.md6,
            mrb1=self.mrb1,
            controller=TurnLevelAttemptController(self.policy),
            experience_logger=self.experience_logger,
            policy_state_path=self.policy_state_path,
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
    )


__all__ = (
    "ActiveAdaptiveContext",
    "AdaptiveAttemptComponents",
    "AdaptiveComponentCoordinator",
    "AdaptiveConcurrencyError",
    "AdaptiveIntegrationError",
    "AdaptiveRestartRequiredError",
    "TutorAssessmentStudentModelPipeline",
    "build_default_adaptive_component_coordinator",
)
