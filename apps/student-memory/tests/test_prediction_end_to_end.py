import pytest

from src.schemas.interaction import LearningStatePredictionResponse
from src.services.student_state_service import (
    StudentStateService,
    predict_student_state,
)


@pytest.fixture
def service():
    return StudentStateService()


def make_base_features():
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


# -------------------------------------------------------------------------
# 1. Cold start
# -------------------------------------------------------------------------

def test_end_to_end_cold_start(service):
    result = service.predict_state(
        {
            "previous_interaction_count": 0
        }
    )

    assert isinstance(
        result,
        LearningStatePredictionResponse,
    )

    assert result.learning_state == "UNAVAILABLE"
    assert result.evidence_level == "COLD_START"
    assert result.evidence_strength == "NONE"
    assert result.model_used is False


# -------------------------------------------------------------------------
# 2. Overall-history-only case
# -------------------------------------------------------------------------

def test_end_to_end_overall_only(service):
    features = make_base_features()

    features["previous_skill_interaction_count"] = 0
    features["has_full_skill_window_5"] = 0
    features[
        "recent_skill_accuracy_change_3_vs_previous_2"
    ] = None

    result = service.predict_state(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "OVERALL_ONLY"
    assert result.evidence_strength == "LOW"
    assert result.model_used is True


# -------------------------------------------------------------------------
# 3. Partial-skill case
# -------------------------------------------------------------------------

def test_end_to_end_partial_skill(service):
    features = make_base_features()

    features["previous_skill_interaction_count"] = 3
    features["has_full_skill_window_5"] = 0
    features[
        "recent_skill_accuracy_change_3_vs_previous_2"
    ] = None

    result = service.predict_state(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "PARTIAL_SKILL"
    assert result.evidence_strength == "MEDIUM"
    assert result.model_used is True


# -------------------------------------------------------------------------
# 4. Full-skill case
# -------------------------------------------------------------------------

def test_end_to_end_full_skill(service):
    features = make_base_features()

    result = service.predict_state(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }

    assert result.evidence_level == "FULL_SKILL"
    assert result.evidence_strength == "HIGH"
    assert result.model_used is True


# -------------------------------------------------------------------------
# 5. Response serialization
# -------------------------------------------------------------------------

def test_end_to_end_response_serialization(service):
    result = service.predict_state(
        make_base_features()
    )

    serialized = result.model_dump()

    assert set(serialized.keys()) == {
        "learning_state",
        "evidence_level",
        "evidence_strength",
        "model_used",
    }

    assert serialized["learning_state"] in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }


# -------------------------------------------------------------------------
# 6. Repeated predictions are deterministic
# -------------------------------------------------------------------------

def test_end_to_end_repeated_predictions_are_identical(service):
    features = make_base_features()

    first = service.predict_state(features)
    second = service.predict_state(features)

    assert first == second


# -------------------------------------------------------------------------
# 7. Convenience service interface
# -------------------------------------------------------------------------

def test_end_to_end_convenience_interface():
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
    assert result.model_used is False


# -------------------------------------------------------------------------
# 8. All allowed evidence transitions
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "previous_interaction_count",
        "previous_skill_interaction_count",
        "expected_level",
        "expected_strength",
    ),
    [
        (20, 0, "OVERALL_ONLY", "LOW"),
        (20, 1, "PARTIAL_SKILL", "MEDIUM"),
        (20, 4, "PARTIAL_SKILL", "MEDIUM"),
        (20, 5, "FULL_SKILL", "HIGH"),
        (20, 10, "FULL_SKILL", "HIGH"),
    ],
)
def test_end_to_end_evidence_boundaries(
    service,
    previous_interaction_count,
    previous_skill_interaction_count,
    expected_level,
    expected_strength,
):
    features = make_base_features()

    features[
        "previous_interaction_count"
    ] = previous_interaction_count

    features[
        "previous_skill_interaction_count"
    ] = previous_skill_interaction_count

    if previous_skill_interaction_count < 5:
        features["has_full_skill_window_5"] = 0
        features[
            "recent_skill_accuracy_change_3_vs_previous_2"
        ] = None

    result = service.predict_state(features)

    assert result.evidence_level == expected_level
    assert result.evidence_strength == expected_strength


# -------------------------------------------------------------------------
# 9. Prediction state domain
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "skill_history",
    [0, 1, 3, 5, 10],
)
def test_end_to_end_prediction_domain(
    service,
    skill_history,
):
    features = make_base_features()

    features[
        "previous_skill_interaction_count"
    ] = skill_history

    if skill_history < 5:
        features["has_full_skill_window_5"] = 0
        features[
            "recent_skill_accuracy_change_3_vs_previous_2"
        ] = None

    result = service.predict_state(features)

    assert result.learning_state in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }


# -------------------------------------------------------------------------
# 10. No model use during cold start
# -------------------------------------------------------------------------

def test_end_to_end_cold_start_model_bypass(service):
    result = service.predict_state(
        {
            "previous_interaction_count": 0
        }
    )

    assert result.model_used is False
