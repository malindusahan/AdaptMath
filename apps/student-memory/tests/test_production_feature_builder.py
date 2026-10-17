import math

import pytest

from src.database.assessment_repository import store_assessment
from src.database.history_repository import (
    get_student_history,
    get_student_topic_history,
)
from src.features.production_feature_builder import (
    MODEL_FEATURES,
    build_production_features,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)


def store_questions(
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


def get_features(database_path):
    overall = get_student_history("s1", database_path)
    skill = get_student_topic_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    return build_production_features(overall, skill)


def test_exact_model_feature_contract(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(database_path, correctness=[True])
    features = get_features(database_path)
    assert list(features.keys()) == MODEL_FEATURES
    assert len(features) == 12


def test_counts_are_derived_from_real_history(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(database_path, correctness=[True, False, True])
    features = get_features(database_path)
    assert features["previous_interaction_count"] == 3
    assert features["previous_skill_interaction_count"] == 3


def test_missing_behaviour_is_not_fabricated(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(database_path, correctness=[True, False, True])
    features = get_features(database_path)
    assert features["recent_attempt_log_mean_10"] is None
    assert features["recent_multi_attempt_rate_10"] is None
    assert features["recent_hint_usage_rate_10"] is None
    assert features["recent_response_log_mean_10"] is None
    assert features["recent_response_median_ms_10"] is None
    assert features["recent_hint_available_rate_10"] == 0.0
    assert features["has_recent_hint_usage_evidence"] == 0


def test_attempt_features_use_only_observed_values(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(
        database_path,
        correctness=[True, True, False],
        attempts=[1, None, 3],
    )
    features = get_features(database_path)
    expected_log_mean = (math.log1p(1) + math.log1p(3)) / 2
    assert features["recent_attempt_log_mean_10"] == pytest.approx(
        expected_log_mean
    )
    assert features["recent_multi_attempt_rate_10"] == pytest.approx(0.5)


def test_hint_availability_and_usage_are_separate(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(
        database_path,
        correctness=[True, False, True, True],
        hints=[1, None, 0, None],
        hint_totals=[4, None, 2, None],
    )
    features = get_features(database_path)
    assert features["recent_hint_available_rate_10"] == pytest.approx(0.5)
    assert features["recent_hint_usage_rate_10"] == pytest.approx(
        (0.25 + 0.0) / 2
    )
    assert features["has_recent_hint_usage_evidence"] == 1


def test_response_features_ignore_missing_measurements(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(
        database_path,
        correctness=[True, False, True],
        response_times=[1000, None, 3000],
    )
    features = get_features(database_path)
    expected_log_mean = (math.log1p(1000) + math.log1p(3000)) / 2
    assert features["recent_response_log_mean_10"] == pytest.approx(
        expected_log_mean
    )
    assert features["recent_response_median_ms_10"] == pytest.approx(2000)


def test_full_skill_window_requires_five_interactions(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(database_path, correctness=[True, False, True, False])
    features = get_features(database_path)
    assert features["has_full_skill_window_5"] == 0
    assert features["recent_skill_accuracy_change_3_vs_previous_2"] is None


def test_skill_trend_after_five_interactions(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(
        database_path,
        correctness=[True, False, True, True, True],
    )
    features = get_features(database_path)
    assert features["has_full_skill_window_5"] == 1
    assert features[
        "recent_skill_accuracy_change_3_vs_previous_2"
    ] == pytest.approx(0.5)


def test_overall_and_skill_counts_are_distinct(tmp_path):
    database_path = tmp_path / "memory.db"
    store_questions(
        database_path,
        correctness=[True, False, True],
        prefix="alg",
    )
    store_questions(
        database_path,
        topic="Geometry",
        subtopic="Angles",
        correctness=[True, True],
        prefix="geo",
    )
    features = get_features(database_path)
    assert features["previous_interaction_count"] == 5
    assert features["previous_skill_interaction_count"] == 3


def test_only_latest_ten_used_for_recent_behaviour(tmp_path):
    database_path = tmp_path / "memory.db"
    attempts = [100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    store_questions(
        database_path,
        correctness=[True] * 12,
        attempts=attempts,
    )
    features = get_features(database_path)
    assert features["recent_multi_attempt_rate_10"] == 0.0
    assert features["recent_attempt_log_mean_10"] == pytest.approx(math.log1p(1))


def test_empty_history_builds_cold_start_compatible_features():
    features = build_production_features([], [])
    assert features["previous_interaction_count"] == 0
    assert features["previous_skill_interaction_count"] == 0
    assert features["has_full_skill_window_5"] == 0
    assert features["has_recent_hint_usage_evidence"] == 0
