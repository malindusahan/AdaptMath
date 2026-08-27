"""
Learning-State Prediction Engine.

Responsibilities
----------------
- Load the frozen learning-state model artifacts.
- Validate the frozen behavioural feature set and feature order.
- Handle true cold-start interactions.
- Derive evidence level and evidence strength.
- Apply the saved median imputer.
- Predict NEEDS_SUPPORT / DEVELOPING / STRONG.
- Return a structured prediction result.

This module does NOT:
- retrieve student history from the database,
- engineer historical features,
- save predictions,
- expose API routes.

Those responsibilities belong to the service/API layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.schemas.interaction import LearningStatePredictionResponse


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "learning_state_model.joblib"
IMPUTER_PATH = ARTIFACT_DIR / "learning_state_imputer.joblib"
METADATA_PATH = ARTIFACT_DIR / "learning_state_model_metadata.json"


# =============================================================================
# Frozen modelling contract
# =============================================================================

EXPECTED_FEATURES = [
    "previous_interaction_count",
    "previous_skill_interaction_count",
    "recent_accuracy_change_5_vs_10",
    "recent_attempt_log_mean_10",
    "recent_multi_attempt_rate_10",
    "recent_hint_usage_rate_10",
    "recent_hint_available_rate_10",
    "recent_response_log_mean_10",
    "recent_response_median_ms_10",
    "recent_skill_accuracy_change_3_vs_previous_2",
    "has_full_skill_window_5",
    "has_recent_hint_usage_evidence",
]

ALLOWED_MODEL_STATES = {
    "NEEDS_SUPPORT",
    "DEVELOPING",
    "STRONG",
}

COLD_START_STATE = "UNAVAILABLE"


# =============================================================================
# Exceptions
# =============================================================================

class LearningStateModelError(Exception):
    """Base exception for learning-state prediction errors."""


class ArtifactLoadError(LearningStateModelError):
    """Raised when a required model artifact cannot be loaded."""


class FeatureValidationError(LearningStateModelError):
    """Raised when prediction input is missing or invalid."""


class ArtifactValidationError(LearningStateModelError):
    """Raised when saved artifacts do not match the frozen model contract."""


# =============================================================================
# Prediction engine
# =============================================================================

class LearningStateModel:
    """
    Frozen behavioural learning-state prediction engine.

    The model predicts:

    - NEEDS_SUPPORT
    - DEVELOPING
    - STRONG

    True cold-start interactions are routed to:

    - UNAVAILABLE

    without calling the ML model.
    """

    def __init__(self) -> None:
        self.model = None
        self.imputer = None
        self.metadata: dict[str, Any] = {}
        self.feature_order: list[str] = []

        self._load_artifacts()
        self._validate_artifacts()

    # -------------------------------------------------------------------------
    # Artifact loading
    # -------------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load the final model, imputer, and metadata artifacts."""

        required_paths = {
            "model": MODEL_PATH,
            "imputer": IMPUTER_PATH,
            "metadata": METADATA_PATH,
        }

        missing_files = [
            str(path)
            for path in required_paths.values()
            if not path.exists()
        ]

        if missing_files:
            raise ArtifactLoadError(
                "Required learning-state artifact(s) are missing: "
                + ", ".join(missing_files)
            )

        try:
            self.model = joblib.load(MODEL_PATH)
            self.imputer = joblib.load(IMPUTER_PATH)

            with METADATA_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:
                self.metadata = json.load(file)

        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load learning-state artifacts: {exc}"
            ) from exc

        metadata_features = self.metadata.get("features")

        if not isinstance(metadata_features, list):
            raise ArtifactValidationError(
                "Metadata does not contain a valid 'features' list."
            )

        self.feature_order = metadata_features

    # -------------------------------------------------------------------------
    # Artifact validation
    # -------------------------------------------------------------------------

    def _validate_artifacts(self) -> None:
        """
        Ensure the saved artifacts still match the frozen Phase 6 contract.
        """

        if self.feature_order != EXPECTED_FEATURES:
            raise ArtifactValidationError(
                "Saved metadata feature order does not match the frozen "
                "12-feature behavioural model contract."
            )

        if len(self.feature_order) != 12:
            raise ArtifactValidationError(
                f"Expected 12 model features, found {len(self.feature_order)}."
            )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count != 12:
            raise ArtifactValidationError(
                "Saved model does not expect exactly 12 features."
            )

        imputer_statistics = getattr(
            self.imputer,
            "statistics_",
            None,
        )

        if imputer_statistics is None:
            raise ArtifactValidationError(
                "Saved imputer does not contain fitted statistics."
            )

        if len(imputer_statistics) != 12:
            raise ArtifactValidationError(
                "Saved imputer does not contain statistics for 12 features."
            )

        model_classes = set(
            getattr(
                self.model,
                "classes_",
                [],
            )
        )

        if model_classes != ALLOWED_MODEL_STATES:
            raise ArtifactValidationError(
                "Saved model classes do not match "
                "NEEDS_SUPPORT / DEVELOPING / STRONG."
            )

        cold_start_output = self.metadata.get(
            "cold_start_output"
        )

        if cold_start_output != COLD_START_STATE:
            raise ArtifactValidationError(
                "Saved metadata cold-start output is not UNAVAILABLE."
            )

    # -------------------------------------------------------------------------
    # Evidence calculation
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_evidence_context(
        previous_interaction_count: int,
        previous_skill_interaction_count: int,
    ) -> tuple[str, str]:
        """
        Convert history depth into the Phase 5 evidence framework.

        Returns:
            (evidence_level, evidence_strength)
        """

        if previous_interaction_count == 0:
            return "COLD_START", "NONE"

        if previous_skill_interaction_count == 0:
            return "OVERALL_ONLY", "LOW"

        if previous_skill_interaction_count < 5:
            return "PARTIAL_SKILL", "MEDIUM"

        return "FULL_SKILL", "HIGH"

    # -------------------------------------------------------------------------
    # Input helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _require_non_negative_integer(
        features: Mapping[str, Any],
        field_name: str,
    ) -> int:
        """Validate a required non-negative history-count field."""

        if field_name not in features:
            raise FeatureValidationError(
                f"Missing required field: '{field_name}'."
            )

        value = features[field_name]

        if value is None or pd.isna(value):
            raise FeatureValidationError(
                f"'{field_name}' cannot be missing."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise FeatureValidationError(
                f"'{field_name}' must be numeric."
            ) from exc

        if not np.isfinite(numeric_value):
            raise FeatureValidationError(
                f"'{field_name}' must be finite."
            )

        if numeric_value < 0:
            raise FeatureValidationError(
                f"'{field_name}' cannot be negative."
            )

        if not numeric_value.is_integer():
            raise FeatureValidationError(
                f"'{field_name}' must be an integer count."
            )

        return int(numeric_value)

    def _build_feature_frame(
        self,
        features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Build a one-row feature DataFrame in the exact frozen feature order.

        Structurally missing historical measurements may be None/NaN because
        the saved imputer is responsible for filling them.
        """

        missing_fields = [
            feature
            for feature in self.feature_order
            if feature not in features
        ]

        if missing_fields:
            raise FeatureValidationError(
                "Missing required model feature(s): "
                + ", ".join(missing_fields)
            )

        ordered_values: dict[str, Any] = {}

        for feature in self.feature_order:
            value = features[feature]

            if value is None:
                value = np.nan

            if not pd.isna(value):
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise FeatureValidationError(
                        f"Feature '{feature}' must be numeric or missing."
                    ) from exc

                if not np.isfinite(value):
                    raise FeatureValidationError(
                        f"Feature '{feature}' contains a non-finite value."
                    )

            ordered_values[feature] = value

        feature_frame = pd.DataFrame(
            [ordered_values],
            columns=self.feature_order,
        )

        return feature_frame

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> LearningStatePredictionResponse:
        """
        Predict a student's current learning state.

        Cold-start behaviour
        --------------------
        If previous_interaction_count == 0:

            learning_state = UNAVAILABLE
            evidence_level = COLD_START
            evidence_strength = NONE
            model_used = False

        Otherwise the frozen behavioural model is used.

        Parameters
        ----------
        features:
            Mapping containing historical learning-state features.

        Returns
        -------
        dict
            Structured learning-state result.
        """

        if not isinstance(features, Mapping):
            raise FeatureValidationError(
                "Prediction input must be a mapping/dictionary."
            )

        # ---------------------------------------------------------------------
        # Validate history counters first.
        #
        # This allows genuine cold-start inputs to return UNAVAILABLE without
        # requiring meaningless model features.
        # ---------------------------------------------------------------------

        previous_interaction_count = (
            self._require_non_negative_integer(
                features,
                "previous_interaction_count",
            )
        )

        # True cold start: no model inference.
        if previous_interaction_count == 0:
            return LearningStatePredictionResponse(
                learning_state=COLD_START_STATE,
                evidence_level="COLD_START",
                evidence_strength="NONE",
                model_used=False,
            )

        previous_skill_interaction_count = (
            self._require_non_negative_integer(
                features,
                "previous_skill_interaction_count",
            )
        )

        if (
            previous_skill_interaction_count
            > previous_interaction_count
        ):
            raise FeatureValidationError(
                "'previous_skill_interaction_count' cannot exceed "
                "'previous_interaction_count'."
            )

        evidence_level, evidence_strength = (
            self._get_evidence_context(
                previous_interaction_count,
                previous_skill_interaction_count,
            )
        )

        # ---------------------------------------------------------------------
        # Build model input in exact frozen feature order.
        # ---------------------------------------------------------------------

        feature_frame = self._build_feature_frame(
            features
        )

        try:
            processed_features = self.imputer.transform(
                feature_frame
            )

            prediction = self.model.predict(
                processed_features
            )[0]

        except Exception as exc:
            raise LearningStateModelError(
                f"Learning-state prediction failed: {exc}"
            ) from exc

        prediction = str(prediction)

        if prediction not in ALLOWED_MODEL_STATES:
            raise LearningStateModelError(
                f"Model returned unexpected state: '{prediction}'."
            )

        return LearningStatePredictionResponse(
            learning_state=prediction,
            evidence_level=evidence_level,
            evidence_strength=evidence_strength,
            model_used=True,
        )


# =============================================================================
# Optional module-level singleton
# =============================================================================

_default_engine: LearningStateModel | None = None


def get_learning_state_model() -> LearningStateModel:
    """
    Return a reusable process-level LearningStateModel instance.

    Artifacts are loaded once on first use rather than once per prediction.
    """

    global _default_engine

    if _default_engine is None:
        _default_engine = LearningStateModel()

    return _default_engine


def predict_learning_state(
    features: Mapping[str, Any],
) -> LearningStatePredictionResponse:
    """
    Convenience function for making a learning-state prediction.

    Later service-layer code can call this without managing the model object
    directly.
    """

    engine = get_learning_state_model()

    return engine.predict(features)
