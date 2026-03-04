"""Bridge an already-produced student-model outcome into frozen v3.

This module coordinates identity and target-skill ownership only. Repository B
remains responsible for evaluation, signal resolution, BKT, and mastery. The
frozen-v3 pipeline remains responsible for validating the four scientific
outcome fields, applying raw signed reward, delayed credit, logging, and state
persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from math import isclose
from typing import Protocol, runtime_checkable

_OUTCOME_FIELDS = (
    "skill",
    "mastery_before",
    "mastery_after",
    "delta_mastery",
)
_MISSING = object()


@runtime_checkable
class StudentModelAdaptivePipeline(Protocol):
    """The one Repository B operation required at attempt start."""

    def start_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        attempt_skill: str | None = None,
    ) -> object: ...


@runtime_checkable
class AdaptivePolicyPipeline(Protocol):
    """Lifecycle implemented by both legacy and direct Turn-LinTS pipelines."""

    @property
    def active_attempt_id(self) -> str | int | None: ...

    def start_attempt(self, memory: object) -> Mapping[str, object]: ...

    def restore_attempt(
        self,
        memory: object,
        completed_turns: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]: ...

    def finish_attempt(
        self,
        memory: object,
        learning_outcome: Mapping[str, object],
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]: ...

    def abort_attempt(self, memory: object | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class ActiveCrossRepositoryAttempt:
    """Identity and B-canonical target retained across one attempt."""

    student_id: str
    attempt_id: str
    target_skill: str
    mastery_before: float
    student_model_context: object = field(repr=False)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a non-empty string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string.")
    return result


def _context_value(context: object, name: str) -> object:
    try:
        return getattr(context, name)
    except AttributeError as exc:
        raise TypeError(
            f"Repository B start_attempt() result is missing {name!r}."
        ) from exc


class StudentModelFrozenV3Bridge:
    """Coordinate one shared attempt from Repository B into frozen v3."""

    def __init__(
        self,
        *,
        student_model_pipeline: StudentModelAdaptivePipeline,
        adaptive_pipeline: AdaptivePolicyPipeline,
    ) -> None:
        if not isinstance(student_model_pipeline, StudentModelAdaptivePipeline):
            raise TypeError(
                "student_model_pipeline must expose start_attempt()."
            )
        if not isinstance(adaptive_pipeline, AdaptivePolicyPipeline):
            raise TypeError(
                "adaptive_pipeline must implement the adaptive policy lifecycle."
            )
        self.student_model_pipeline = student_model_pipeline
        self.adaptive_pipeline = adaptive_pipeline
        self._active: ActiveCrossRepositoryAttempt | None = None

    @property
    def active_attempt(self) -> ActiveCrossRepositoryAttempt | None:
        return self._active

    @property
    def student_model_context(self) -> object:
        if self._active is None:
            raise RuntimeError("No cross-repository attempt is active.")
        return self._active.student_model_context

    def start_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        target_skill: str,
        adaptive_memory: MutableMapping[str, object],
    ) -> ActiveCrossRepositoryAttempt:
        """Snapshot B mastery, then start A with the same ID and mastery."""

        if self._active is not None:
            raise RuntimeError("A cross-repository attempt is already active.")
        student = _identifier(student_id, "student_id")
        shared_id = _identifier(attempt_id, "attempt_id")
        requested_skill = _identifier(target_skill, "target_skill")
        if not isinstance(adaptive_memory, MutableMapping):
            raise TypeError("adaptive_memory must be a mutable mapping.")
        if adaptive_memory.get("attempt_id", _MISSING) != shared_id:
            raise ValueError(
                "adaptive_memory.attempt_id must equal the shared attempt_id."
            )

        context = self.student_model_pipeline.start_attempt(
            student_id=student,
            attempt_id=shared_id,
            attempt_skill=requested_skill,
        )
        context_student = _identifier(
            _context_value(context, "student_id"),
            "Repository B context.student_id",
        )
        context_attempt = _identifier(
            _context_value(context, "attempt_id"),
            "Repository B context.attempt_id",
        )
        canonical_skill = _identifier(
            _context_value(context, "skill"),
            "Repository B context.skill",
        )
        if context_student != student:
            raise ValueError("Repository B returned a different student_id.")
        if context_attempt != shared_id:
            raise ValueError("Repository B returned a different attempt_id.")
        if canonical_skill != requested_skill:
            raise ValueError(
                "Repository B canonical target skill differs from the "
                "requested target skill."
            )

        mastery_before = _context_value(context, "mastery_before")
        previous_mastery = adaptive_memory.get("mastery_before", _MISSING)
        adaptive_memory["mastery_before"] = mastery_before
        adaptive_started = False
        try:
            started = self.adaptive_pipeline.start_attempt(adaptive_memory)
            adaptive_started = True
            if started.get("attempt_id") != shared_id:
                raise RuntimeError(
                    "Frozen v3 started with a different attempt_id."
                )
            frozen_mastery = float(started["mastery_before"])
            repository_b_mastery = float(mastery_before)
            if not isclose(
                frozen_mastery,
                repository_b_mastery,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Frozen v3 did not start from Repository B mastery_before."
                )
        except Exception:
            if adaptive_started:
                self.adaptive_pipeline.abort_attempt(adaptive_memory)
            if previous_mastery is _MISSING:
                adaptive_memory.pop("mastery_before", None)
            else:
                adaptive_memory["mastery_before"] = previous_mastery
            raise

        self._active = ActiveCrossRepositoryAttempt(
            student_id=student,
            attempt_id=shared_id,
            target_skill=canonical_skill,
            mastery_before=frozen_mastery,
            student_model_context=context,
        )
        return self._active

    def restore_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        target_skill: str,
        mastery_before: float,
        student_model_context: object,
        completed_turns: Sequence[Mapping[str, object]],
        adaptive_memory: MutableMapping[str, object],
    ) -> ActiveCrossRepositoryAttempt:
        """Rehydrate a controlled attempt from validated durable turn records."""

        if self._active is not None:
            raise RuntimeError("A cross-repository attempt is already active.")
        student = _identifier(student_id, "student_id")
        shared_id = _identifier(attempt_id, "attempt_id")
        skill = _identifier(target_skill, "target_skill")
        if adaptive_memory.get("attempt_id", _MISSING) != shared_id:
            raise ValueError("adaptive_memory.attempt_id must equal the shared attempt_id.")
        adaptive_memory["mastery_before"] = mastery_before
        started = self.adaptive_pipeline.restore_attempt(
            adaptive_memory,
            completed_turns,
        )
        restored_mastery = float(started["mastery_before"])
        self._active = ActiveCrossRepositoryAttempt(
            student_id=student,
            attempt_id=shared_id,
            target_skill=skill,
            mastery_before=restored_mastery,
            student_model_context=student_model_context,
        )
        return self._active

    def finish_attempt(
        self,
        *,
        adaptive_memory: MutableMapping[str, object],
        student_model_result: Mapping[str, object],
    ) -> dict[str, object]:
        """Validate identity/skill, then pass exactly four fields into A."""

        active = self._active
        if active is None:
            raise RuntimeError("No cross-repository attempt is active.")
        if not isinstance(adaptive_memory, MutableMapping):
            raise TypeError("adaptive_memory must be a mutable mapping.")
        if adaptive_memory.get("attempt_id", _MISSING) != active.attempt_id:
            raise ValueError(
                "adaptive_memory.attempt_id does not match the active attempt."
            )
        if self.adaptive_pipeline.active_attempt_id != active.attempt_id:
            raise RuntimeError(
                "Frozen v3 active attempt differs from the bridge attempt."
            )
        if not isinstance(student_model_result, Mapping):
            raise TypeError("student_model_result must be a mapping.")

        learning_outcome = student_model_result.get("learning_outcome")
        if not isinstance(learning_outcome, Mapping):
            raise ValueError(
                "student_model_result.learning_outcome must be available."
            )
        if learning_outcome.get("attempt_id", _MISSING) != active.attempt_id:
            raise ValueError(
                "Repository B learning_outcome.attempt_id does not match the "
                "active attempt."
            )
        if learning_outcome.get("skill", _MISSING) != active.target_skill:
            raise ValueError(
                "Repository B learning_outcome.skill does not match the "
                "authoritative target skill."
            )
        missing = [
            name for name in _OUTCOME_FIELDS if name not in learning_outcome
        ]
        if missing:
            raise ValueError(
                "Repository B learning_outcome is missing fields: "
                f"{missing}."
            )

        frozen_v3_outcome = {
            name: learning_outcome[name] for name in _OUTCOME_FIELDS
        }
        diagnostics = student_model_result.get("diagnostics")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError(
                "student_model_result.diagnostics must be a mapping when "
                "supplied."
            )

        result = self.adaptive_pipeline.finish_attempt(
            adaptive_memory,
            frozen_v3_outcome,
            diagnostics=diagnostics,
        )
        self._active = None
        return result

    def abort_attempt(
        self,
        *,
        adaptive_memory: MutableMapping[str, object],
    ) -> None:
        """Abort only the adaptive-policy side and clear the shared identity."""

        active = self._active
        if active is None:
            raise RuntimeError("No cross-repository attempt is active.")
        if not isinstance(adaptive_memory, MutableMapping):
            raise TypeError("adaptive_memory must be a mutable mapping.")
        if adaptive_memory.get("attempt_id", _MISSING) != active.attempt_id:
            raise ValueError(
                "adaptive_memory.attempt_id does not match the active attempt."
            )
        if self.adaptive_pipeline.active_attempt_id != active.attempt_id:
            raise RuntimeError(
                "Frozen v3 active attempt differs from the bridge attempt."
            )
        self.adaptive_pipeline.abort_attempt(adaptive_memory)
        self._active = None


__all__ = (
    "AdaptivePolicyPipeline",
    "ActiveCrossRepositoryAttempt",
    "StudentModelAdaptivePipeline",
    "StudentModelFrozenV3Bridge",
)
