import pytest

from src.database.assessment_repository import store_assessment
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
    StudentHistoryPredictionResponse,
)
from src.services.student_state_service import (
    StudentStateService,
    StudentStateServiceError,
    predict_student_state_from_history,
)


@pytest.fixture
def service():
    return StudentStateService()


def store_history(
    database_path,
    *,
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
    prefix="q",
    correctness,
    attempts=None,
    hints=None,
    hint_totals=None,
    response_times=None,
):
    questions = []
    for index, correct in enumerate(correctness, start=1):
        position = index - 1
        questions.append(
            AssessmentQuestionResult(
                question_id=f"{prefix}{index}",
                is_correct=correct,
                attempt_count=None if attempts is None else attempts[position],
                hint_count=None if hints is None else hints[position],
                hint_total=None if hint_totals is None else hint_totals[position],
                response_time_ms=(
                    None if response_times is None else response_times[position]
                ),
            )
        )
    return store_assessment(
        AssessmentMemoryUpdateRequest(
            student_id=student_id,
            topic=topic,
            subtopic=subtopic,
            assessment_questions=questions,
        ),
        database_path,
    )


def test_no_history_returns_unavailable(service, tmp_path):
    result = service.predict_from_stored_history(
        "new-student", "Algebra", "Linear equations", tmp_path / "memory.db"
    )
    assert isinstance(result, StudentHistoryPredictionResponse)
    assert result.learning_state == "UNAVAILABLE"
    assert result.evidence_level == "COLD_START"
    assert result.evidence_strength == "NONE"
    assert result.model_used is False
    assert result.behavioural_coverage == "CORRECTNESS_ONLY_COVERAGE"
    assert result.previous_interaction_count == 0
    assert result.previous_skill_interaction_count == 0


def test_correctness_only_history_predicts_with_coverage(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(database_path, correctness=[True, False, True, True, False])
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.learning_state in {"NEEDS_SUPPORT", "DEVELOPING", "STRONG"}
    assert result.model_used is True
    assert result.evidence_level == "FULL_SKILL"
    assert result.evidence_strength == "HIGH"
    assert result.behavioural_coverage == "CORRECTNESS_ONLY_COVERAGE"
    assert result.previous_interaction_count == 5
    assert result.previous_skill_interaction_count == 5
    assert result.attempt_observation_count == 0
    assert result.hint_observation_count == 0
    assert result.response_time_observation_count == 0


def test_partial_skill_history_preserves_medium_evidence(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(database_path, correctness=[True, False, True])
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.model_used is True
    assert result.evidence_level == "PARTIAL_SKILL"
    assert result.evidence_strength == "MEDIUM"
    assert result.previous_skill_interaction_count == 3


def test_overall_only_history_preserves_low_evidence(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(
        database_path,
        topic="Geometry",
        subtopic="Angles",
        correctness=[True, False, True, True],
    )
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.model_used is True
    assert result.evidence_level == "OVERALL_ONLY"
    assert result.evidence_strength == "LOW"
    assert result.previous_interaction_count == 4
    assert result.previous_skill_interaction_count == 0


def test_partial_behavioural_coverage(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(
        database_path,
        correctness=[True, False, True, True, False],
        attempts=[1, 2, 1, 3, 1],
    )
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.behavioural_coverage == "PARTIAL_BEHAVIOURAL_COVERAGE"
    assert result.attempt_observation_count == 5
    assert result.hint_observation_count == 0
    assert result.response_time_observation_count == 0


def test_full_behavioural_coverage(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(
        database_path,
        correctness=[True, False, True, True, False],
        attempts=[1, 2, 1, 3, 1],
        hints=[0, 1, 0, 1, 0],
        hint_totals=[2, 2, 2, 2, 2],
        response_times=[1000, 2000, 1500, 2500, 1800],
    )
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.behavioural_coverage == "FULL_BEHAVIOURAL_COVERAGE"
    assert result.attempt_observation_count == 5
    assert result.hint_observation_count == 5
    assert result.response_time_observation_count == 5


def test_before_assessment_cutoff_prevents_current_assessment_leakage(
    service, tmp_path
):
    database_path = tmp_path / "memory.db"
    store_history(database_path, prefix="old", correctness=[True, False, True])
    current = store_history(
        database_path, prefix="current", correctness=[False, False, False]
    )
    result = service.predict_from_stored_history(
        "s1",
        "Algebra",
        "Linear equations",
        database_path,
        before_assessment_id=current.assessment_id,
    )
    assert result.previous_interaction_count == 3
    assert result.previous_skill_interaction_count == 3
    assert result.evidence_level == "PARTIAL_SKILL"


def test_without_cutoff_currently_persisted_history_is_used(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(database_path, prefix="first", correctness=[True, False, True])
    store_history(database_path, prefix="second", correctness=[False, False])
    result = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert result.previous_interaction_count == 5
    assert result.previous_skill_interaction_count == 5
    assert result.evidence_level == "FULL_SKILL"


def test_convenience_history_interface(tmp_path):
    result = predict_student_state_from_history(
        "s-new", "Algebra", "Linear equations", tmp_path / "memory.db"
    )
    assert isinstance(result, StudentHistoryPredictionResponse)
    assert result.learning_state == "UNAVAILABLE"


def test_response_serializes_cleanly(service, tmp_path):
    database_path = tmp_path / "memory.db"
    store_history(database_path, correctness=[True, False, True])
    payload = service.predict_from_stored_history(
        "s1", "Algebra", "Linear equations", database_path
    ).model_dump()
    assert payload["student_id"] == "s1"
    assert payload["topic"] == "Algebra"
    for key in (
        "learning_state",
        "behavioural_coverage",
        "evidence_level",
        "evidence_strength",
        "attempt_observation_count",
        "hint_observation_count",
        "response_time_observation_count",
    ):
        assert key in payload


@pytest.mark.parametrize("student_id", ["", "   "])
def test_invalid_student_id_rejected(service, tmp_path, student_id):
    with pytest.raises(StudentStateServiceError):
        service.predict_from_stored_history(
            student_id, "Algebra", database_path=tmp_path / "memory.db"
        )


@pytest.mark.parametrize("topic", ["", "   "])
def test_invalid_topic_rejected(service, tmp_path, topic):
    with pytest.raises(StudentStateServiceError):
        service.predict_from_stored_history(
            "s1", topic, database_path=tmp_path / "memory.db"
        )


@pytest.mark.parametrize("cutoff", [0, -1, 1.5, True])
def test_invalid_assessment_cutoff_rejected(service, tmp_path, cutoff):
    with pytest.raises(StudentStateServiceError):
        service.predict_from_stored_history(
            "s1",
            "Algebra",
            database_path=tmp_path / "memory.db",
            before_assessment_id=cutoff,
        )
