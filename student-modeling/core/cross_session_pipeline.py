from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.attempt_learning_outcome import (
    AdaptiveAttemptContext,
    AttemptLearningOutcome,
)
from core.curriculum import Curriculum
from core.detector_service import DetectorService
from core.evaluation_contract import (
    EventEvaluator,
    apply_evaluator_to_extraction,
)
from core.learning_path import generate_learning_path
from core.pipeline_adapter import resolve_extracted_events


def _authorised_assessed_skills(
    evaluator: EventEvaluator,
) -> tuple[str, ...]:
    context = getattr(evaluator, "context", None)
    assessed_skills = getattr(context, "assessed_skills", ())

    if not isinstance(assessed_skills, (tuple, list)):
        return ()

    normalised: list[str] = []
    for raw_skill in assessed_skills:
        if not isinstance(raw_skill, str) or not raw_skill.strip():
            continue
        skill = raw_skill.strip()
        if skill not in normalised:
            normalised.append(skill)
    return tuple(normalised)


def _resolve_attempt_skill(
    evaluator: EventEvaluator,
    attempt_skill: Optional[str],
) -> str:
    """Resolve one authoritative reward skill before any evaluation writes."""
    authorised = _authorised_assessed_skills(evaluator)

    if attempt_skill is not None:
        if not isinstance(attempt_skill, str) or not attempt_skill.strip():
            raise ValueError("attempt_skill must be a non-empty string")
        selected = attempt_skill.strip()
        if authorised and selected not in authorised:
            raise ValueError(
                f"attempt_skill {selected!r} is not authorized by the "
                f"evaluator's assessed_skills {authorised!r}"
            )
        return selected

    if len(authorised) != 1:
        raise ValueError(
            "An adaptive attempt requires an explicit attempt_skill or "
            "exactly one authorized assessed skill"
        )
    return authorised[0]


def _infer_single_assessed_skill(
    evaluator: EventEvaluator,
) -> Optional[str]:
    """
    Best-effort compatibility bridge for StudentAnswerEvaluator.

    A behavioural proxy must have an explicit skill context. We never infer a
    skill from an utterance like "I don't understand." If the evaluator exposes
    exactly one assessed skill, that is safe to reuse as the conversational
    skill context.
    """
    context = getattr(evaluator, "context", None)
    assessed_skills = getattr(context, "assessed_skills", ())

    if (
        isinstance(assessed_skills, (tuple, list))
        and len(assessed_skills) == 1
        and isinstance(assessed_skills[0], str)
        and assessed_skills[0].strip()
    ):
        return assessed_skills[0].strip()

    return None


def _add_behaviour_only_student_events(
    *,
    evaluated_extraction: dict,
    transcript: list[dict],
    behavioural_skill: Optional[str],
) -> dict:
    """
    Ensure conversational student turns can reach the detector/resolver path.

    ConceptExtractor remains responsible for semantic skill events. However, a
    conversational turn such as "I don't understand" may contain no explicit
    mathematical proposition and may therefore produce no extraction event.

    When one explicit active/assessed skill is available, add an UNKNOWN
    correctness event for an otherwise-unrepresented student turn. The resolver
    may then use uncertainty/clarification as weak proxy evidence.

    Existing (turn_index, skill) events always win, so this never duplicates an
    evaluator-backed event for the same turn and skill.
    """
    if behavioural_skill is None:
        return evaluated_extraction

    skill = str(behavioural_skill).strip()

    if not skill:
        raise ValueError(
            "behavioural_skill must be non-empty when supplied"
        )

    existing_events = list(
        evaluated_extraction.get("events", [])
    )

    existing_keys = {
        (
            event.get("turn_index"),
            str(event.get("skill", "")).strip(),
        )
        for event in existing_events
        if isinstance(event, dict)
    }

    augmented = list(existing_events)

    for turn_index, turn in enumerate(transcript):
        if turn.get("role") != "student":
            continue

        student_text = str(
            turn.get("text", "")
        ).strip()

        if not student_text:
            continue

        key = (turn_index, skill)

        if key in existing_keys:
            continue

        augmented.append(
            {
                "turn_index": turn_index,
                "skill": skill,
                "evidence_span": student_text,
                "student_text": student_text,
                "correctness": "unknown",
                "evaluator_confidence": 0.0,
                "evaluator_source": "behaviour_only_turn",
            }
        )

        existing_keys.add(key)

    augmented.sort(
        key=lambda event: (
            int(event.get("turn_index", 0)),
            str(event.get("skill", "")),
        )
    )

    return {
        **evaluated_extraction,
        "events": augmented,
    }


@dataclass
class CrossSessionStudentModelPipeline:
    """
    Thin orchestration layer for the student-modelling path.

    KnowledgeGraph remains the canonical BKT/database owner.

    Conversational extension:
    - evaluator-backed correctness remains highest-priority evidence;
    - otherwise-unrepresented student turns may be mapped to ONE explicit
      active skill and sent through the behaviour detectors;
    - the resolver decides whether those UNKNOWN-correctness turns create a
      weak behavioural BKT proxy.
    """

    concept_extractor: Any
    evaluator: EventEvaluator
    detectors: DetectorService
    knowledge_graph: Any
    curriculum: Optional[Curriculum] = None

    def start_attempt(
        self,
        *,
        student_id: str,
        attempt_id: str,
        attempt_skill: Optional[str] = None,
    ) -> AdaptiveAttemptContext:
        """Capture the authoritative adaptive start snapshot."""
        skill = _resolve_attempt_skill(
            self.evaluator,
            attempt_skill,
        )
        return self.knowledge_graph.start_attempt(
            student_id=student_id,
            attempt_id=attempt_id,
            skill=skill,
        )

    def process_transcript(
        self,
        *,
        transcript: list[dict],
        student_id: str,
        session_id: str,
        behavioural_skill: Optional[str] = None,
        attempt_context: Optional[AdaptiveAttemptContext] = None,
    ) -> dict:
        if attempt_context is not None:
            if not isinstance(attempt_context, AdaptiveAttemptContext):
                raise TypeError(
                    "attempt_context must be an AdaptiveAttemptContext"
                )
            if attempt_context.student_id != student_id:
                raise ValueError(
                    "attempt_context.student_id must match student_id"
                )

            # Explicitly bind the immutable context to an authorized target and
            # verify that no concurrent update made the start snapshot stale.
            _resolve_attempt_skill(
                self.evaluator,
                attempt_context.skill,
            )
            self.knowledge_graph.validate_attempt_context(
                attempt_context
            )

        raw_extraction = self.concept_extractor.extract(
            transcript
        )

        evaluated_extraction = (
            apply_evaluator_to_extraction(
                extraction=raw_extraction,
                transcript=transcript,
                evaluator=self.evaluator,
            )
        )

        # If the caller did not explicitly supply a conversational skill,
        # safely reuse exactly one assessed skill from StudentAnswerEvaluator.
        if behavioural_skill is None and attempt_context is not None:
            behavioural_skill = attempt_context.skill
        elif behavioural_skill is None:
            behavioural_skill = _infer_single_assessed_skill(
                self.evaluator
            )

        evaluated_extraction = (
            _add_behaviour_only_student_events(
                evaluated_extraction=evaluated_extraction,
                transcript=transcript,
                behavioural_skill=behavioural_skill,
            )
        )

        resolved_events = resolve_extracted_events(
            extraction=evaluated_extraction,
            transcript=transcript,
            student_id=student_id,
            session_id=session_id,
            reasoning_predictor=(
                self.detectors.reasoning_predictor
            ),
            uncertainty_predictor=(
                self.detectors.uncertainty_predictor
            ),
            clarification_predictor=(
                self.detectors.clarification_predictor
            ),
            history_getter=(
                self.knowledge_graph.get_attempts
            ),
        )

        knowledge_graph_result = (
            self.knowledge_graph.process_resolved_events(
                student_id=student_id,
                resolved_events=resolved_events,
                session_id=session_id,
            )
        )

        # FR17: derive the current path only after resolved evidence has been
        # persisted and mastery has been recalculated. The path is intentionally
        # not stored; it is a deterministic view of the returned mastery graph
        # plus the optional curriculum supplied when this pipeline was built.
        learning_path = generate_learning_path(
            knowledge_graph_result["graph"],
            curriculum=self.curriculum,
        )

        result = {
            "raw_extraction": raw_extraction,
            "evaluated_extraction": evaluated_extraction,
            "resolved_events": resolved_events,
            "knowledge_graph_result": knowledge_graph_result,
            "learning_path": learning_path,
        }

        if attempt_context is not None:
            mastery_after = (
                self.knowledge_graph.get_current_mastery_probability(
                    student_id,
                    attempt_context.skill,
                    require_effective_prior=True,
                )
            )
            if mastery_after is None:
                # The real KnowledgeGraph guarantees an effective-prior state
                # once start_attempt succeeds. Avoid selecting another skill or
                # raising a retry-looking post-persistence exception if a custom
                # adapter violates that contract.
                result["learning_outcome"] = None
                result["learning_outcome_status"] = {
                    "status": "unavailable",
                    "reason": "authoritative_target_mastery_unavailable",
                    "skill": attempt_context.skill,
                }
            else:
                outcome = AttemptLearningOutcome(
                    attempt_id=attempt_context.attempt_id,
                    skill=attempt_context.skill,
                    mastery_before=attempt_context.mastery_before,
                    mastery_after=mastery_after,
                    delta_mastery=(
                        mastery_after - attempt_context.mastery_before
                    ),
                )
                result["learning_outcome"] = outcome.as_dict()

        return result
