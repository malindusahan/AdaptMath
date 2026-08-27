import pytest

from src.models.learning_state_model import (
    LearningStateModel,
    FeatureValidationError,
    EXPECTED_FEATURES,
)
from src.schemas.interaction import LearningStatePredictionResponse


@pytest.fixture(scope="module")
def engine():
    return LearningStateModel()


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


def test_artifacts_load(engine):
    assert engine.model is not None
    assert engine.imputer is not None
    assert engine.metadata is not None


def test_exact_feature_order(engine):
    assert engine.feature_order == EXPECTED_FEATURES
    assert len(engine.feature_order) == 12


def test_full_skill_prediction(engine):
    features = make_full_feature_input()

    result = engine.predict(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "FULL_SKILL"
    assert result.evidence_strength == "HIGH"
    assert result.model_used is True


def test_partial_skill_prediction(engine):
    features = make_full_feature_input()

    features["previous_skill_interaction_count"] = 3
    features["has_full_skill_window_5"] = 0
    features["recent_skill_accuracy_change_3_vs_previous_2"] = None

    result = engine.predict(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "PARTIAL_SKILL"
    assert result.evidence_strength == "MEDIUM"
    assert result.model_used is True


def test_overall_only_prediction(engine):
    features = make_full_feature_input()

    features["previous_skill_interaction_count"] = 0
    features["has_full_skill_window_5"] = 0
    features["recent_skill_accuracy_change_3_vs_previous_2"] = None

    result = engine.predict(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "OVERALL_ONLY"
    assert result.evidence_strength == "LOW"
    assert result.model_used is True


def test_cold_start_does_not_require_all_model_features(engine):
    features = {
        "previous_interaction_count": 0
    }

    result = engine.predict(features)

    assert isinstance(
        result,
        LearningStatePredictionResponse,
    )

    assert result.model_dump() == {
        "learning_state": "UNAVAILABLE",
        "evidence_level": "COLD_START",
        "evidence_strength": "NONE",
        "model_used": False,
    }


def test_negative_previous_interaction_count_rejected(engine):
    features = {
        "previous_interaction_count": -1
    }

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_negative_previous_skill_interaction_count_rejected(engine):
    features = make_full_feature_input()
    features["previous_skill_interaction_count"] = -1

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_skill_history_cannot_exceed_overall_history(engine):
    features = make_full_feature_input()

    features["previous_interaction_count"] = 3
    features["previous_skill_interaction_count"] = 4

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_missing_required_model_feature_rejected(engine):
    features = make_full_feature_input()

    del features["recent_multi_attempt_rate_10"]

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_non_numeric_feature_rejected(engine):
    features = make_full_feature_input()

    features["recent_hint_usage_rate_10"] = "invalid"

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_non_integer_history_count_rejected(engine):
    features = make_full_feature_input()

    features["previous_interaction_count"] = 10.5

    with pytest.raises(FeatureValidationError):
        engine.predict(features)


def test_missing_structural_measurement_can_be_imputed(engine):
    features = make_full_feature_input()

    features["recent_hint_usage_rate_10"] = None
    features["has_recent_hint_usage_evidence"] = 0

    result = engine.predict(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.model_used is True


def test_output_domain(engine):
    features = make_full_feature_input()

    result = engine.predict(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
        "UNAVAILABLE",
    }


def test_prediction_returns_response_schema(engine):
    features = make_full_feature_input()

    result = engine.predict(features)

    assert isinstance(
        result,
        LearningStatePredictionResponse,
    )

    dumped = result.model_dump()

    assert set(dumped.keys()) == {
        "learning_state",
        "evidence_level",
        "evidence_strength",
        "model_used",
    }


def test_model_classes(engine):
    assert set(engine.model.classes_) == {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }
