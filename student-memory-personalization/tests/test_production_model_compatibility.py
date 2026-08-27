import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.production_feature_builder import MODEL_FEATURES


MODEL_PATH = "artifacts/learning_state_model.joblib"
IMPUTER_PATH = "artifacts/learning_state_imputer.joblib"


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer


def make_binuri_style_features():
    """Correctness history exists, but optional behaviour was not supplied."""

    return {
        "previous_interaction_count": 8,
        "previous_skill_interaction_count": 5,
        "recent_accuracy_change_5_vs_10": 0.1,
        "recent_attempt_log_mean_10": None,
        "recent_multi_attempt_rate_10": None,
        "recent_hint_usage_rate_10": None,
        "recent_hint_available_rate_10": 0.0,
        "recent_response_log_mean_10": None,
        "recent_response_median_ms_10": None,
        "recent_skill_accuracy_change_3_vs_previous_2": 0.5,
        "has_full_skill_window_5": 1,
        "has_recent_hint_usage_evidence": 0,
    }


def test_production_contract_matches_saved_model_feature_count(artifacts):
    model, _ = artifacts
    assert len(MODEL_FEATURES) == 12
    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == 12


def test_binuri_style_input_has_exact_feature_order():
    features = make_binuri_style_features()
    assert list(features.keys()) == MODEL_FEATURES


def test_binuri_style_input_contains_expected_missing_values():
    features = make_binuri_style_features()
    expected_missing = {
        "recent_attempt_log_mean_10",
        "recent_multi_attempt_rate_10",
        "recent_hint_usage_rate_10",
        "recent_response_log_mean_10",
        "recent_response_median_ms_10",
    }
    actual_missing = {
        name for name, value in features.items() if value is None
    }
    assert actual_missing == expected_missing


def test_saved_imputer_accepts_binuri_style_missingness(artifacts):
    _, imputer = artifacts
    frame = pd.DataFrame(
        [make_binuri_style_features()],
        columns=MODEL_FEATURES,
    )
    transformed = imputer.transform(frame)
    assert transformed.shape == (1, 12)


def test_saved_imputer_produces_no_missing_or_infinite_values(artifacts):
    _, imputer = artifacts
    frame = pd.DataFrame(
        [make_binuri_style_features()],
        columns=MODEL_FEATURES,
    )
    transformed = np.asarray(imputer.transform(frame), dtype=float)
    assert not np.isnan(transformed).any()
    assert np.isfinite(transformed).all()


def test_saved_model_can_predict_transformed_production_input(artifacts):
    model, imputer = artifacts
    frame = pd.DataFrame(
        [make_binuri_style_features()],
        columns=MODEL_FEATURES,
    )
    transformed = imputer.transform(frame)
    prediction = model.predict(transformed)
    assert len(prediction) == 1
    assert prediction[0] in {
        "NEEDS_SUPPORT",
        "DEVELOPING",
        "STRONG",
    }


def test_imputer_statistics_exist_for_all_model_features(artifacts):
    _, imputer = artifacts
    assert hasattr(imputer, "statistics_")
    assert len(imputer.statistics_) == 12


def test_imputer_statistics_are_finite(artifacts):
    _, imputer = artifacts
    statistics = np.asarray(imputer.statistics_, dtype=float)
    assert not np.isnan(statistics).any()
    assert np.isfinite(statistics).all()
