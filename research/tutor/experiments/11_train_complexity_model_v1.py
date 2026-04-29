from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "external"
    / "Visual_Item_Difficulty"
    / "data"
    / "items.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "agents"
    / "complexity"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "eedi_complexity_v1.joblib"
)


def build_input(
    question: str,
    visual_description: str,
) -> str:
    question = str(question).strip()

    visual_description = (
        str(visual_description).strip()
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


def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "regressor",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )


def main() -> None:
    print()
    print("=" * 80)
    print("TRAINING ADAPTMATH COMPLEXITY MODEL V1")
    print("=" * 80)

    items = pd.read_csv(
        DATA_PATH
    )

    train = (
        items.loc[
            items["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train) != 580:
        raise RuntimeError(
            f"Expected 580 training items, "
            f"found {len(train)}."
        )

    print()
    print(
        f"Training items: {len(train)}"
    )

    print(
        "Held-out test items used: 0"
    )

    inputs = []

    for _, row in train.iterrows():
        visual = row[
            "visual_description"
        ]

        if pd.isna(visual):
            visual = ""

        inputs.append(
            build_input(
                question=row["text"],
                visual_description=visual,
            )
        )

    targets = (
        train["difficulty"]
        .astype(float)
        .to_numpy()
    )

    print()
    print(
        "Target: Rasch item difficulty beta"
    )

    print(
        "Model: TF-IDF + Ridge regression"
    )

    print(
        "Input: question content + "
        "available visual description"
    )

    print()
    print("Training...")

    model = create_model()

    model.fit(
        inputs,
        targets,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(
        f"Saved model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print("=" * 80)
    print("MODEL SANITY CHECK")
    print("=" * 80)

    sample = train.iloc[0]

    sample_visual = sample[
        "visual_description"
    ]

    if pd.isna(sample_visual):
        sample_visual = ""

    sample_input = build_input(
        question=sample["text"],
        visual_description=sample_visual,
    )

    prediction = float(
        model.predict(
            [sample_input]
        )[0]
    )

    print()
    print(
        f"QuestionId: "
        f"{int(sample['QuestionId'])}"
    )

    print(
        f"Predicted beta: "
        f"{prediction:.4f}"
    )

    print()
    print("=" * 80)
    print("COMPLEXITY MODEL V1 COMPLETE")
    print("=" * 80)

    print()
    print(
        "This model predicts continuous "
        "mathematical item difficulty."
    )

    print(
        "No easy/medium/hard thresholds "
        "are encoded."
    )

    print(
        "No hand-written difficulty rules "
        "are encoded."
    )

    print(
        "The held-out 145-item Eedi test "
        "set remains untouched."
    )


if __name__ == "__main__":
    main()