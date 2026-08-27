from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class LearningStatePredictionResponse(BaseModel):
    """
    Structured response returned by the learning-state prediction engine.
    """

    learning_state: Literal[
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
        "UNAVAILABLE",
    ]

    evidence_level: Literal[
        "COLD_START",
        "OVERALL_ONLY",
        "PARTIAL_SKILL",
        "FULL_SKILL",
    ]

    evidence_strength: Literal[
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    model_used: bool


class AssessmentQuestionResult(BaseModel):
    """One evaluated question from an assessment."""

    question_id: str
    question: str | None = None
    student_answer: str | None = None
    expected_answer: str | None = None
    is_correct: bool
    identified_error: str | None = None

    # Optional production measurements remain None when unavailable.
    attempt_count: int | None = Field(default=None, ge=0)
    hint_count: int | None = Field(default=None, ge=0)
    hint_total: int | None = Field(default=None, ge=0)
    response_time_ms: float | None = Field(default=None, ge=0)

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question_id must not be blank.")
        return value

    @field_validator("identified_error")
    @classmethod
    def validate_identified_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError(
                "identified_error must not be blank when supplied."
            )
        return value

    @model_validator(mode="after")
    def validate_hint_relationship(self):
        if (
            self.hint_count is not None
            and self.hint_total is not None
            and self.hint_count > self.hint_total
        ):
            raise ValueError("hint_count cannot exceed hint_total.")
        return self


class AssessmentMemoryUpdateRequest(BaseModel):
    """Normalized assessment payload accepted by the memory layer."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "student_id": "student_101",
                "topic": "Algebra",
                "subtopic": "Linear equations",
                "assessment_questions": [
                    {
                        "question_id": "q_math_001",
                        "question": "Solve for x: 2x + 4 = 10",
                        "student_answer": "x = 3",
                        "expected_answer": "x = 3",
                        "is_correct": True,
                        "attempt_count": 1,
                        "hint_count": 0,
                        "hint_total": 2,
                        "response_time_ms": 4200.0,
                    }
                ],
                "identified_errors": [],
                "overall_feedback": "Great job solving single-variable linear equations.",
            }
        },
    )

    student_id: str
    topic: str
    subtopic: str | None = None
    assessment_questions: list[AssessmentQuestionResult] = Field(min_length=1)
    identified_errors: list[str] = Field(default_factory=list)
    overall_feedback: str | None = None

    @field_validator("student_id", "topic")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank.")
        return value

    @field_validator("subtopic")
    @classmethod
    def normalize_subtopic(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("identified_errors")
    @classmethod
    def validate_identified_errors(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError(
                    "identified_errors must not contain blank values."
                )
            normalized.append(value)
        return normalized


class StoredAssessmentResult(BaseModel):
    """Result returned after an assessment is stored successfully."""

    assessment_id: int
    student_id: str
    topic: str
    subtopic: str | None = None
    total_questions: int
    correct_count: int
    wrong_count: int
    stored: Literal[True] = True


class StudentHistoryPredictionResponse(BaseModel):
    """Learning-state prediction produced from persisted student history."""

    student_id: str
    topic: str
    subtopic: str | None = None
    learning_state: Literal[
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE"
    ]
    evidence_level: Literal[
        "COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL"
    ]
    evidence_strength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    behavioural_coverage: Literal[
        "FULL_BEHAVIOURAL_COVERAGE",
        "PARTIAL_BEHAVIOURAL_COVERAGE",
        "CORRECTNESS_ONLY_COVERAGE",
    ]
    model_used: bool
    previous_interaction_count: int
    previous_skill_interaction_count: int
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int


class StoredLearningStateSnapshot(BaseModel):
    """Result of persisting one immutable learning-state snapshot."""

    snapshot_id: int
    assessment_id: int | None
    student_id: str
    topic: str
    subtopic: str | None = None
    learning_state: Literal[
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE"
    ]
    evidence_level: Literal[
        "COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL"
    ]
    evidence_strength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    behavioural_coverage: Literal[
        "FULL_BEHAVIOURAL_COVERAGE",
        "PARTIAL_BEHAVIOURAL_COVERAGE",
        "CORRECTNESS_ONLY_COVERAGE",
    ]
    model_used: bool
    previous_interaction_count: int
    previous_skill_interaction_count: int
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    persisted: Literal[True] = True


class MemoryUpdateResponse(BaseModel):
    """Complete result of processing one finished assessment."""

    student_id: str
    topic: str
    subtopic: str | None = None
    assessment_id: int
    snapshot_id: int
    learning_state: Literal[
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE"
    ]
    evidence_level: Literal[
        "COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL"
    ]
    evidence_strength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    behavioural_coverage: Literal[
        "FULL_BEHAVIOURAL_COVERAGE",
        "PARTIAL_BEHAVIOURAL_COVERAGE",
        "CORRECTNESS_ONLY_COVERAGE",
    ]
    model_used: bool
    previous_interaction_count: int
    previous_skill_interaction_count: int
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    misconception_count: int
    misconceptions: list[str] = Field(default_factory=list)
    memory_updated: Literal[True] = True


class CurrentStudentMemory(BaseModel):
    """Latest persisted state for one student learning-area context."""

    student_id: str
    topic: str
    subtopic: str | None = None
    learning_state: Literal[
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE"
    ]
    evidence_level: Literal[
        "COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL"
    ]
    evidence_strength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    behavioural_coverage: Literal[
        "FULL_BEHAVIOURAL_COVERAGE",
        "PARTIAL_BEHAVIOURAL_COVERAGE",
        "CORRECTNESS_ONLY_COVERAGE",
    ]
    model_used: bool
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    last_assessment_id: int | None = None
    last_snapshot_id: int | None = None
    updated_at: str


class LearningStateHistoryItem(BaseModel):
    """One immutable historical learning-state snapshot."""

    snapshot_id: int
    assessment_id: int | None = None
    student_id: str
    topic: str
    subtopic: str | None = None
    learning_state: Literal[
        "NEEDS_SUPPORT", "DEVELOPING", "STRONG", "UNAVAILABLE"
    ]
    evidence_level: Literal[
        "COLD_START", "OVERALL_ONLY", "PARTIAL_SKILL", "FULL_SKILL"
    ]
    evidence_strength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    behavioural_coverage: Literal[
        "FULL_BEHAVIOURAL_COVERAGE",
        "PARTIAL_BEHAVIOURAL_COVERAGE",
        "CORRECTNESS_ONLY_COVERAGE",
    ]
    model_used: bool
    previous_interaction_count: int
    previous_skill_interaction_count: int
    recent_interaction_count: int
    attempt_observation_count: int
    hint_observation_count: int
    response_time_observation_count: int
    created_at: str
