"""Project a completed Tutor assessment into the stable Memory write contract."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class CompletedQuestionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_interaction_id: str
    question_id: str
    problem_id: str
    student_answer: str | None = Field(default=None, max_length=4000)
    expected_answer: str | None = Field(default=None, max_length=4000)
    is_correct: bool
    identified_error: str | None = Field(default=None, max_length=1000)


class CompletedAttemptEvidence(BaseModel):
    """No mastery, policy, C3, MD6, MRB1, reward, or dialogue fields exist."""

    model_config = ConfigDict(extra="forbid")

    external_student_id: str
    external_session_id: str
    external_attempt_id: str
    source: str = "adaptmath"
    target_skill: str
    session_status: str
    questions: list[CompletedQuestionEvidence] = Field(min_length=1)


def build_completed_attempt_evidence(
    state: Mapping[str, Any],
) -> CompletedAttemptEvidence | None:
    """Return None for aborted/incomplete state; always use completed_attempt_id."""
    completed_attempt_id = state.get("completed_attempt_id")
    thread_id = state.get("thread_id")
    student_id = state.get("student_id")
    target_skill = state.get("target_skill")
    identifiers = (completed_attempt_id, thread_id, student_id, target_skill)
    if not all(isinstance(value, str) and value.strip() for value in identifiers):
        return None

    evaluated = list(state.get("correct_answers", [])) + list(
        state.get("wrong_answers", [])
    )
    if not evaluated:
        return None

    questions: list[CompletedQuestionEvidence] = []
    for item in evaluated:
        question_id = item.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            return None
        question_id = question_id.strip()
        questions.append(
            CompletedQuestionEvidence(
                external_interaction_id=f"{completed_attempt_id}:{question_id}",
                question_id=question_id,
                problem_id=thread_id,
                student_answer=item.get("student_answer"),
                expected_answer=item.get("expected_answer"),
                is_correct=bool(item.get("is_correct")),
                identified_error=item.get("identified_error"),
            )
        )

    return CompletedAttemptEvidence(
        external_student_id=student_id,
        external_session_id=thread_id,
        external_attempt_id=completed_attempt_id,
        target_skill=target_skill,
        session_status=(
            "ACTIVE" if state.get("needs_reteaching", False) else "COMPLETED"
        ),
        questions=questions,
    )
