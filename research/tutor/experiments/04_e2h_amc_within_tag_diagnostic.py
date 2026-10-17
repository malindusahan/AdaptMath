from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "e2h_amc"
    / "e2h_amc_train.parquet"
)

N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_conflicting_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["normalized_problem"] = (
        dataframe["problem"]
        .astype(str)
        .map(normalize_problem)
    )

    keep_indices = []
    excluded_rows = 0

    for _, group in dataframe.groupby(
        "normalized_problem",
        sort=False,
    ):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue

        unique_ratings = (
            group["rating"]
            .astype(float)
            .unique()
        )

        if len(unique_ratings) == 1:
            keep_indices.append(group.index[0])
        else:
            excluded_rows += len(group)

    cleaned = (
        dataframe
        .loc[keep_indices]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Original rows: {len(dataframe):,}"
    )

    print(
        f"Conflicting duplicate rows excluded: "
        f"{excluded_rows}"
    )

    print(
        f"Final rows: {len(cleaned):,}"
    )

    return cleaned


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
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
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def create_tfidf_model() -> Pipeline:
    return Pipeline(
        [
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
                "ridge",
                Ridge(alpha=1.0),
            ),
        ]
    )


def get_training_tag_means(
    tags: np.ndarray,
    ratings: np.ndarray,
) -> dict[str, float]:
    temp = pd.DataFrame(
        {
            "tag": tags,
            "rating": ratings,
        }
    )

    return (
        temp
        .groupby("tag")["rating"]
        .mean()
        .to_dict()
    )


def map_tag_means(
    tags: np.ndarray,
    tag_means: dict[str, float],
    fallback: float,
) -> np.ndarray:
    return np.array(
        [
            tag_means.get(
                str(tag),
                fallback,
            )
            for tag in tags
        ],
        dtype=float,
    )


def main() -> None:
    print_section(
        "E2H-AMC WITHIN-TAG DIFFICULTY DIAGNOSTIC"
    )

    dataframe = pd.read_parquet(
        TRAIN_PATH
    )

    dataframe = remove_conflicting_duplicates(
        dataframe
    )

    problems = (
        dataframe["problem"]
        .astype(str)
        .to_numpy()
    )

    ratings = (
        dataframe["rating"]
        .astype(float)
        .to_numpy()
    )

    tags = (
        dataframe["tag"]
        .astype(str)
        .to_numpy()
    )

    print_section(
        "TAG DISTRIBUTION"
    )

    print(
        dataframe["tag"]
        .value_counts()
        .to_string()
    )

    cross_validator = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    model_names = [
        "Global Mean",
        "Tag Mean",
        "TF-IDF Raw",
        "Tag Mean + TF-IDF Residual",
    ]

    fold_results = []

    total_runs = (
        N_SPLITS
        * N_REPEATS
    )

    print_section(
        "RUNNING REPEATED CROSS-VALIDATION"
    )

    for run_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(problems),
        start=1,
    ):
        print(
            f"Run {run_number}/{total_runs}"
        )

        x_train = problems[
            train_indices
        ]

        x_validation = problems[
            validation_indices
        ]

        y_train = ratings[
            train_indices
        ]

        y_validation = ratings[
            validation_indices
        ]

        train_tags = tags[
            train_indices
        ]

        validation_tags = tags[
            validation_indices
        ]

        global_mean = float(
            np.mean(y_train)
        )

        # -------------------------------------------------
        # 1. GLOBAL MEAN
        # -------------------------------------------------

        global_predictions = np.full(
            len(validation_indices),
            global_mean,
        )

        fold_results.append(
            {
                "run": run_number,
                "model": "Global Mean",
                **calculate_metrics(
                    y_validation,
                    global_predictions,
                ),
            }
        )

        # -------------------------------------------------
        # 2. TAG-MEAN BASELINE
        # -------------------------------------------------

        tag_means = get_training_tag_means(
            train_tags,
            y_train,
        )

        tag_mean_train = map_tag_means(
            train_tags,
            tag_means,
            global_mean,
        )

        tag_mean_validation = map_tag_means(
            validation_tags,
            tag_means,
            global_mean,
        )

        fold_results.append(
            {
                "run": run_number,
                "model": "Tag Mean",
                **calculate_metrics(
                    y_validation,
                    tag_mean_validation,
                ),
            }
        )

        # -------------------------------------------------
        # 3. RAW TF-IDF MODEL
        # -------------------------------------------------

        raw_model = create_tfidf_model()

        raw_model.fit(
            x_train,
            y_train,
        )

        raw_predictions = raw_model.predict(
            x_validation
        )

        fold_results.append(
            {
                "run": run_number,
                "model": "TF-IDF Raw",
                **calculate_metrics(
                    y_validation,
                    raw_predictions,
                ),
            }
        )

        # -------------------------------------------------
        # 4. WITHIN-TAG RESIDUAL MODEL
        # -------------------------------------------------

        residual_train = (
            y_train
            - tag_mean_train
        )

        residual_model = (
            create_tfidf_model()
        )

        residual_model.fit(
            x_train,
            residual_train,
        )

        predicted_residuals = (
            residual_model.predict(
                x_validation
            )
        )

        residual_predictions = (
            tag_mean_validation
            + predicted_residuals
        )

        fold_results.append(
            {
                "run": run_number,
                "model": (
                    "Tag Mean + TF-IDF Residual"
                ),
                **calculate_metrics(
                    y_validation,
                    residual_predictions,
                ),
            }
        )

    results_df = pd.DataFrame(
        fold_results
    )

    print_section(
        "REPEATED-CV RESULTS"
    )

    metrics = [
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    ]

    summary_rows = []

    for model_name in model_names:
        subset = results_df[
            results_df["model"]
            == model_name
        ]

        row = {
            "model": model_name
        }

        for metric in metrics:
            row[
                f"{metric}_mean"
            ] = subset[
                metric
            ].mean()

            row[
                f"{metric}_std"
            ] = subset[
                metric
            ].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print_section(
        "INTERPRETATION TARGET"
    )

    print(
        "Compare these three questions:"
    )

    print(
        "1. How strong is Tag Mean alone?"
    )

    print(
        "2. Does raw TF-IDF beat Tag Mean?"
    )

    print(
        "3. Does TF-IDF still add useful "
        "signal after tag-level difficulty "
        "has been removed?"
    )

    print_section(
        "EXPERIMENT COMPLETE"
    )

    print(
        "Only the E2H-AMC training split "
        "was used."
    )

    print(
        "Official evaluation data was not "
        "loaded."
    )


if __name__ == "__main__":
    main()