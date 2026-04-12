from src.database.assessment_repository import store_assessment
from src.database.history_repository import get_student_history
from src.features.behavioural_coverage import classify_behavioural_coverage
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)


def store_history(
    database_path,
    *,
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
                question_id=f"q{index}",
                is_correct=correct,
                attempt_count=None if attempts is None else attempts[position],
                hint_count=None if hints is None else hints[position],
                hint_total=None if hint_totals is None else hint_totals[position],
                response_time_ms=(
                    None if response_times is None else response_times[position]
                ),
            )
        )

    store_assessment(
        AssessmentMemoryUpdateRequest(
            student_id="s1",
            topic="Algebra",
            subtopic="Linear equations",
            assessment_questions=questions,
        ),
        database_path,
    )
    return get_student_history("s1", database_path)


def test_empty_history_is_correctness_only():
    coverage = classify_behavioural_coverage([])
    assert coverage.level == "CORRECTNESS_ONLY_COVERAGE"
    assert coverage.recent_interaction_count == 0
    assert coverage.has_attempt_evidence is False
    assert coverage.has_hint_evidence is False
    assert coverage.has_response_time_evidence is False


def test_correctness_only_history(tmp_path):
    history = store_history(
        tmp_path / "memory.db", correctness=[True, False, True]
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "CORRECTNESS_ONLY_COVERAGE"
    assert coverage.attempt_observation_count == 0
    assert coverage.hint_observation_count == 0
    assert coverage.response_time_observation_count == 0


def test_attempt_only_is_partial(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, False, True],
        attempts=[1, 2, 1],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "PARTIAL_BEHAVIOURAL_COVERAGE"
    assert coverage.has_attempt_evidence is True
    assert coverage.has_hint_evidence is False
    assert coverage.has_response_time_evidence is False


def test_hint_only_is_partial(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, False],
        hints=[0, 1],
        hint_totals=[2, 2],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "PARTIAL_BEHAVIOURAL_COVERAGE"
    assert coverage.has_hint_evidence is True


def test_response_only_is_partial(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, True],
        response_times=[1000, 2000],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "PARTIAL_BEHAVIOURAL_COVERAGE"
    assert coverage.has_response_time_evidence is True


def test_two_categories_are_partial(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, False, True],
        attempts=[1, 2, 1],
        response_times=[1000, 2000, 1500],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "PARTIAL_BEHAVIOURAL_COVERAGE"
    assert coverage.has_attempt_evidence is True
    assert coverage.has_hint_evidence is False
    assert coverage.has_response_time_evidence is True


def test_all_three_categories_are_full(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, False, True],
        attempts=[1, 2, 1],
        hints=[0, 1, 0],
        hint_totals=[2, 2, 2],
        response_times=[1000, 2000, 1500],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "FULL_BEHAVIOURAL_COVERAGE"
    assert coverage.has_attempt_evidence is True
    assert coverage.has_hint_evidence is True
    assert coverage.has_response_time_evidence is True
    assert coverage.attempt_observation_count == 3
    assert coverage.hint_observation_count == 3
    assert coverage.response_time_observation_count == 3


def test_partial_observations_still_count_as_evidence(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True, False, True],
        attempts=[None, 2, None],
        hints=[None, None, 1],
        hint_totals=[None, None, 3],
        response_times=[1000, None, None],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "FULL_BEHAVIOURAL_COVERAGE"
    assert coverage.attempt_observation_count == 1
    assert coverage.hint_observation_count == 1
    assert coverage.response_time_observation_count == 1


def test_only_latest_ten_interactions_are_considered(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True] * 12,
        attempts=[1, 2] + [None] * 10,
        hints=[1, 1] + [None] * 10,
        hint_totals=[2, 2] + [None] * 10,
        response_times=[1000, 2000] + [None] * 10,
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.recent_interaction_count == 10
    assert coverage.level == "CORRECTNESS_ONLY_COVERAGE"
    assert coverage.attempt_observation_count == 0
    assert coverage.hint_observation_count == 0
    assert coverage.response_time_observation_count == 0


def test_zero_measurements_are_real_observations(tmp_path):
    history = store_history(
        tmp_path / "memory.db",
        correctness=[True],
        attempts=[0],
        hints=[0],
        hint_totals=[0],
        response_times=[0],
    )
    coverage = classify_behavioural_coverage(history)
    assert coverage.level == "FULL_BEHAVIOURAL_COVERAGE"
    assert coverage.attempt_observation_count == 1
    assert coverage.hint_observation_count == 1
    assert coverage.response_time_observation_count == 1
