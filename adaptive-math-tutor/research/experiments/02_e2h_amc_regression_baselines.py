from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "e2h_amc"
    / "e2h_amc_train.parquet"
)

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "e2h_amc"
    / "e2h_amc_eval.parquet"
)


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def evaluate_regression(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | str]:
    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    if np.std(y_pred) == 0:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(
            pearsonr(
                y_true,
                y_pred,
            ).statistic
        )

        spearman = float(
            spearmanr(
                y_true,
                y_pred,
            ).statistic
        )

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def print_error_examples(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    count: int = 10,
) -> None:
    analysis = dataframe[
        [
            "problem",
            "rating",
            "contest",
            "tag",
        ]
    ].copy()

    analysis["prediction"] = predictions

    analysis["absolute_error"] = np.abs(
        analysis["rating"]
        - analysis["prediction"]
    )

    worst = analysis.sort_values(
        "absolute_error",
        ascending=False,
    ).head(count)

    print_section(
        f"TOP {count} LARGEST TF-IDF + RIDGE ERRORS"
    )

    for position, (_, row) in enumerate(
        worst.iterrows(),
        start=1,
    ):
        print()
        print(f"Example {position}")
        print("-" * 80)

        print(
            f"Contest: {row['contest']} | "
            f"Tag: {row['tag']}"
        )

        print(
            f"True rating: "
            f"{row['rating']:.4f}"
        )

        print(
            f"Predicted rating: "
            f"{row['prediction']:.4f}"
        )

        print(
            f"Absolute error: "
            f"{row['absolute_error']:.4f}"
        )

        problem = str(row["problem"])

        if len(problem) > 500:
            problem = problem[:500] + "..."

        print(f"\nProblem:\n{problem}")


def main() -> None:
    print_section(
        "E2H-AMC COMPLEXITY REGRESSION BASELINES"
    )

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    eval_df = pd.read_parquet(
        EVAL_PATH
    )

    print(
        f"Training examples: "
        f"{len(train_df):,}"
    )

    print(
        f"Evaluation examples: "
        f"{len(eval_df):,}"
    )

    required_columns = {
        "problem",
        "rating",
    }

    for split_name, dataframe in [
        ("train", train_df),
        ("eval", eval_df),
    ]:
        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe[
            ["problem", "rating"]
        ].isna().any().any():
            raise ValueError(
                f"{split_name} contains missing "
                f"problem/rating values."
            )

    x_train = (
        train_df["problem"]
        .astype(str)
        .to_numpy()
    )

    y_train = (
        train_df["rating"]
        .astype(float)
        .to_numpy()
    )

    x_eval = (
        eval_df["problem"]
        .astype(str)
        .to_numpy()
    )

    y_eval = (
        eval_df["rating"]
        .astype(float)
        .to_numpy()
    )

    results = []

    # -------------------------------------------------
    # BASELINE 1: MEAN PREDICTOR
    # -------------------------------------------------

    print_section(
        "BASELINE 1: MEAN PREDICTOR"
    )

    mean_model = DummyRegressor(
        strategy="mean"
    )

    mean_model.fit(
        x_train.reshape(-1, 1),
        y_train,
    )

    mean_predictions = mean_model.predict(
        x_eval.reshape(-1, 1)
    )

    mean_metrics = evaluate_regression(
        model_name="Mean Predictor",
        y_true=y_eval,
        y_pred=mean_predictions,
    )

    results.append(mean_metrics)

    print(
        pd.DataFrame(
            [mean_metrics]
        ).to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # BASELINE 2: TF-IDF + RIDGE
    # -------------------------------------------------

    print_section(
        "BASELINE 2: TF-IDF + RIDGE"
    )

    text_model = Pipeline(
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

    text_model.fit(
        x_train,
        y_train,
    )

    text_predictions = text_model.predict(
        x_eval
    )

    text_metrics = evaluate_regression(
        model_name="TF-IDF + Ridge",
        y_true=y_eval,
        y_pred=text_predictions,
    )

    results.append(text_metrics)

    print(
        pd.DataFrame(
            [text_metrics]
        ).to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # COMPARISON
    # -------------------------------------------------

    print_section(
        "FINAL BASELINE COMPARISON"
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # -------------------------------------------------
    # SIMPLE ERROR ANALYSIS
    # -------------------------------------------------

    print_error_examples(
        dataframe=eval_df,
        predictions=text_predictions,
        count=10,
    )

    print_section(
        "EXPERIMENT COMPLETE"
    )

    print(
        "No E2H-AMC data was modified."
    )

    print(
        "The official training split was used "
        "for fitting."
    )

    print(
        "The official evaluation split was used "
        "only for evaluation."
    )


if __name__ == "__main__":
    main()