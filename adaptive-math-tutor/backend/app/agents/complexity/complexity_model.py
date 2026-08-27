from pathlib import Path

import joblib


CURRENT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    CURRENT_DIR
    / "models"
    / "eedi_complexity_v1.joblib"
)


class ComplexityModel:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Complexity model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(
            MODEL_PATH
        )

    @staticmethod
    def _build_input(
        question: str,
        visual_description: str | None = None,
    ) -> str:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        visual_description = (
            visual_description.strip()
            if visual_description
            else ""
        )

        if visual_description:
            return (
                f"{question}\n\n"
                f"Visual information:\n"
                f"{visual_description}"
            )

        return question

    def predict(
        self,
        question: str,
        visual_description: str | None = None,
    ) -> float:
        model_input = self._build_input(
            question=question,
            visual_description=visual_description,
        )

        prediction = self.model.predict(
            [model_input]
        )

        return float(
            prediction[0]
        )