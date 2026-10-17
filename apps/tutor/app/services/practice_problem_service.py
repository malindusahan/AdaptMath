"""Generate and validate one problem before an adaptive attempt is started."""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agents.tutor.tutor_agent import TutorAgent
from app.clients.gemini import GeminiChatCompatibilityClient
from app.clients.memory.memory_client import MemoryTopicClassification
from app.core.config import get_settings


logger = logging.getLogger(__name__)


class PracticeProblemGenerationError(RuntimeError):
    """Raised when no candidate passes every pre-start validation gate."""


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_expression: str = Field(min_length=1, max_length=500)
    right_expression: str = Field(min_length=1, max_length=500)


class GeneratedPracticeProblem(BaseModel):
    """Private validated problem; the answer and checks never leave the backend."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str = Field(min_length=1, max_length=500)


class PracticeProblemAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    matches_target_skill: bool
    self_contained: bool
    unambiguous: bool
    solvable: bool
    answer_matches_question: bool
    verification_checks: list[VerificationCheck] = Field(min_length=1, max_length=8)
    rejection_reason: str = Field(max_length=1000)


class TopicClassifier(Protocol):
    def classify_question(self, *, question: str) -> MemoryTopicClassification | None: ...


_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "left_expression": {"type": "string"},
        "right_expression": {"type": "string"},
    },
    "required": ["left_expression", "right_expression"],
    "additionalProperties": False,
}

_GENERATED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "expected_answer": {"type": "string"},
    },
    "required": ["question", "expected_answer"],
    "additionalProperties": False,
}

_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "matches_target_skill": {"type": "boolean"},
        "self_contained": {"type": "boolean"},
        "unambiguous": {"type": "boolean"},
        "solvable": {"type": "boolean"},
        "answer_matches_question": {"type": "boolean"},
        "verification_checks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": _CHECK_SCHEMA,
        },
        "rejection_reason": {"type": "string"},
    },
    "required": [
        "valid",
        "matches_target_skill",
        "self_contained",
        "unambiguous",
        "solvable",
        "answer_matches_question",
        "verification_checks",
        "rejection_reason",
    ],
    "additionalProperties": False,
}


class PracticeProblemGenerator:
    """LLM generation guarded by a separate audit, SymPy, and skill classification."""

    def __init__(
        self,
        *,
        client: GeminiChatCompatibilityClient,
        model_name: str,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_attempts = max_attempts

    def generate(
        self,
        *,
        target_skill: str,
        student_age: int,
        classifier: TopicClassifier,
    ) -> GeneratedPracticeProblem:
        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            gate = "generation"
            try:
                candidate = self._generate_candidate(
                    target_skill=target_skill,
                    student_age=student_age,
                    prior_failures=failures,
                )
                heading = f"{target_skill} practice:"
                if not candidate.question.casefold().startswith(heading.casefold()):
                    candidate = candidate.model_copy(
                        update={"question": f"{heading} {candidate.question}"}
                    )
                gate = "independent_audit"
                audit = self._audit_candidate(
                    target_skill=target_skill,
                    student_age=student_age,
                    candidate=candidate,
                )
                gate = "audit_decision"
                self._require_valid_audit(audit)
                gate = "audit_sympy"
                self._require_verified_checks(audit.verification_checks)
                gate = "canonical_skill_classifier"
                self._require_skill_match(
                    target_skill=target_skill,
                    question=candidate.question,
                    classifier=classifier,
                )
                logger.info(
                    "practice_problem_validated target_skill=%s attempt=%d",
                    target_skill,
                    attempt,
                )
                return candidate
            except Exception as exc:
                failures.append(gate)
                logger.warning(
                    "practice_problem_candidate_rejected target_skill=%s attempt=%d gate=%s reason_type=%s",
                    target_skill,
                    attempt,
                    gate,
                    type(exc).__name__,
                )

        raise PracticeProblemGenerationError(
            "A verified practice problem could not be prepared for this skill."
        )

    def _completion(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=2200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The model returned an empty response.")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("The model response was not an object.")
        return payload

    def _generate_candidate(
        self,
        *,
        target_skill: str,
        student_age: int,
        prior_failures: list[str],
    ) -> GeneratedPracticeProblem:
        correction = ""
        if prior_failures:
            correction = (
                "\nEarlier candidates failed validation. Correct these categories: "
                + "; ".join(prior_failures[-2:])
            )
        payload = self._completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create exactly one concise, self-contained mathematics practice "
                        "problem for the supplied canonical skill. It must be appropriate "
                        "for the learner's age, unambiguous, solvable, and require that "
                        "exact skill. Do not reveal the answer in the question. Avoid "
                        "figures or unstated visual information. A multiple-choice format "
                        "is allowed when the skill is not naturally answered by a number. "
                        "Return the private mathematical expected answer, not only a "
                        "multiple-choice letter. The answer will be checked "
                        "by an independent reviewer before the problem can be used."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Canonical target skill: {target_skill}\n"
                        f"Learner age: {student_age}{correction}"
                    ),
                },
            ],
            schema_name="adaptmath_practice_problem",
            schema=_GENERATED_SCHEMA,
        )
        return GeneratedPracticeProblem.model_validate(payload)

    def _audit_candidate(
        self,
        *,
        target_skill: str,
        student_age: int,
        candidate: GeneratedPracticeProblem,
    ) -> PracticeProblemAudit:
        payload = self._completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Act as a strict independent mathematics item reviewer. Reject "
                        "the item unless it tests the exact canonical skill, is fully "
                        "self-contained, has one unambiguous answer, is solvable as "
                        "written, and the private expected answer is correct. Supply "
                        "fresh equality checks in safe Python/SymPy syntax that establish "
                        "the answer. Put one expression in left_expression and its exactly "
                        "equivalent expression in right_expression. For fractions, prefer "
                        "exact rational forms such as left_expression='3/5' and "
                        "right_expression='6/10'. Never include an equals sign, prose, "
                        "units, booleans, LaTeX, or rounded decimal approximations in an "
                        "expression. Set valid true only when every boolean is true."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_skill": target_skill,
                            "student_age": student_age,
                            "question": candidate.question,
                            "expected_answer": candidate.expected_answer,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema_name="adaptmath_practice_problem_audit",
            schema=_AUDIT_SCHEMA,
        )
        return PracticeProblemAudit.model_validate(payload)

    @staticmethod
    def _require_valid_audit(audit: PracticeProblemAudit) -> None:
        decisions = (
            audit.valid,
            audit.matches_target_skill,
            audit.self_contained,
            audit.unambiguous,
            audit.solvable,
            audit.answer_matches_question,
        )
        if not all(decisions):
            raise ValueError(audit.rejection_reason or "Independent audit rejected the item.")

    @staticmethod
    def _require_verified_checks(checks: list[VerificationCheck]) -> None:
        _, errors = TutorAgent._verify_preparation_checks(
            [check.model_dump() for check in checks]
        )
        if errors:
            raise ValueError("Deterministic mathematical verification failed.")

    @staticmethod
    def _require_skill_match(
        *,
        target_skill: str,
        question: str,
        classifier: TopicClassifier,
    ) -> None:
        classification = classifier.classify_question(question=question)
        if classification is None:
            raise RuntimeError("The canonical skill classifier is unavailable.")
        classified = (classification.topic or "").strip()
        if not classification.is_math or classified.casefold() != target_skill.casefold():
            raise ValueError("The canonical skill classifier rejected the item.")


@lru_cache(maxsize=1)
def get_practice_problem_generator() -> PracticeProblemGenerator:
    settings = get_settings()
    return PracticeProblemGenerator(
        client=GeminiChatCompatibilityClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            max_retries=settings.gemini_max_retries,
            timeout_seconds=settings.gemini_timeout_seconds,
        ),
        model_name=settings.gemini_model,
    )
