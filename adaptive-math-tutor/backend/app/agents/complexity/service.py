from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "complexity_model.joblib"
)


class ComplexityService:
    """
    Loads the trained mathematical complexity model
    and predicts a continuous difficulty score.

    Output:
        0 < complexity_score < 1

    The service does NOT create:
    - Easy / Medium / Hard labels
    - difficulty thresholds
    - topic-based rules
    - learner-specific decisions
    """

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Complexity model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, question: str) -> float:
        """
        Predict the mathematical complexity of one question.

        Parameters
        ----------
        question:
            Raw mathematics question entered by the learner.

        Returns
        -------
        float
            Continuous learned complexity score in (0, 1).
        """

        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        prediction = self.model.predict(
            [question]
        )

        if len(prediction) != 1:
            raise RuntimeError(
                "Complexity model returned an unexpected result."
            )

        score = float(
            prediction[0]
        )

        if not np.isfinite(score):
            raise RuntimeError(
                "Complexity model produced a non-finite score."
            )

        # This is validation only.
        # We do NOT clip or alter the model's prediction.
        if not 0.0 < score < 1.0:
            raise RuntimeError(
                "Complexity model produced a score "
                f"outside its expected range: {score}"
            )

        return score


@lru_cache(maxsize=1)
def get_complexity_service() -> ComplexityService:
    """
    Keep one loaded model instance in memory instead of
    reloading the .joblib file for every request.
    """

    return ComplexityService()