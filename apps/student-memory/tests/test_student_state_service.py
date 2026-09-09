import pytest

from src.models.learning_state_model import FeatureValidationError
from src.schemas.interaction import LearningStatePredictionResponse
from src.services.student_state_service import (
    StudentStateService,
    StudentStateServiceError,
    get_student_state_service,
    predict_student_state,
)


@pytest.fixture
def service():
    return StudentStateService()


def make_full_feature_input():
    return {
        "previous_interaction_count": 20,
        "previous_skill_interaction_count": 8,
        "recent_accuracy_change_5_vs_10": 0.10,
        "recent_attempt_log_mean_10": 0.80,
        "recent_multi_attempt_rate_10": 0.20,
        "recent_hint_usage_rate_10": 0.10,
        "recent_hint_available_rate_10": 0.80,
        "recent_response_log_mean_10": 9.80,
        "recent_response_median_ms_10": 19000,
        "recent_skill_accuracy_change_3_vs_previous_2": 0.20,
        "has_full_skill_window_5": 1,
        "has_recent_hint_usage_evidence": 1,
    }


def test_service_returns_prediction_schema(service):
    result = service.predict_state(
        make_full_feature_input()
    )

    assert isinstance(
        result,
        LearningStatePredictionResponse,
    )

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.model_used is True


def test_service_preserves_full_skill_evidence(service):
    result = service.predict_state(
        make_full_feature_input()
    )

    assert result.evidence_level == "FULL_SKILL"
    assert result.evidence_strength == "HIGH"


def test_service_preserves_partial_skill_evidence(service):
    features = make_full_feature_input()

    features["previous_skill_interaction_count"] = 3
    features["has_full_skill_window_5"] = 0
    features["recent_skill_accuracy_change_3_vs_previous_2"] = None

    result = service.predict_state(features)

    assert result.evidence_level == "PARTIAL_SKILL"
    assert result.evidence_strength == "MEDIUM"
    assert result.model_used is True


def test_service_preserves_overall_only_evidence(service):
    features = make_full_feature_input()

    features["previous_skill_interaction_count"] = 0
    features["has_full_skill_window_5"] = 0
    features["recent_skill_accuracy_change_3_vs_previous_2"] = None

    result = service.predict_state(features)

    assert result.evidence_level == "OVERALL_ONLY"
    assert result.evidence_strength == "LOW"
    assert result.model_used is True


def test_service_preserves_cold_start(service):
    result = service.predict_state(
        {
            "previous_interaction_count": 0
        }
    )

    assert result.model_dump() == {
        "learning_state": "UNAVAILABLE",
        "evidence_level": "COLD_START",
        "evidence_strength": "NONE",
        "model_used": False,
    }


def test_service_rejects_non_mapping_input(service):
    with pytest.raises(StudentStateServiceError):
        service.predict_state(
            ["not", "a", "mapping"]
        )


def test_service_preserves_feature_validation_error(service):
    features = make_full_feature_input()

    del features[
        "recent_multi_attempt_rate_10"
    ]

    with pytest.raises(FeatureValidationError):
        service.predict_state(features)


def test_service_rejects_negative_history_count(service):
    features = make_full_feature_input()

    features["previous_interaction_count"] = -1

    with pytest.raises(FeatureValidationError):
        service.predict_state(features)


def test_convenience_function_returns_schema():
    result = predict_student_state(
        {
            "previous_interaction_count": 0
        }
    )

    assert isinstance(
        result,
        LearningStatePredictionResponse,
    )

    assert result.learning_state == "UNAVAILABLE"


def test_service_singleton_reused():
    first = get_student_state_service()
    second = get_student_state_service()

    assert first is second
