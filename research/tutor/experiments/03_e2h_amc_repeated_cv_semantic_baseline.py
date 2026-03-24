from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyRegressor
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

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
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


def prepare_development_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove conflicting exact-normalized duplicate groups.

    Policy:
    - singleton: keep
    - exact duplicate with one unique rating: keep one
    - exact duplicate with conflicting ratings: exclude whole group

    No target values are averaged or invented.
    """

    dataframe = dataframe.copy()

    dataframe["normalized_problem"] = (
        dataframe["problem"]
        .astype(str)
        .map(normalize_problem)
    )

    groups = dataframe.groupby(
        "normalized_problem",
        sort=False,
    )

    rows_to_keep: list[int] = []
    conflicting_groups = []
    agreement_groups = []

    for normalized_problem, group in groups:
        if len(group) == 1:
            rows_to_keep.append(
                int(group.index[0])
            )
            continue

        unique_ratings = (
            group["rating"]
            .astype(float)
            .unique()
        )

        if len(unique_ratings) == 1:
            rows_to_keep.append(
                int(group.index[0])
            )

            agreement_groups.append(
                group.copy()
            )

        else:
            conflicting_groups.append(
                group.copy()
            )

    print_section(
        "EXACT-DUPLICATE AUDIT"
    )

    print(
        f"Original rows: {len(dataframe):,}"
    )

    print(
        "Duplicate agreement groups: "
        f"{len(agreement_groups)}"
    )

    print(
        "Duplicate conflicting groups: "
        f"{len(conflicting_groups)}"
    )

    excluded_conflict_rows = sum(
        len(group)
        for group in conflicting_groups
    )

    print(
        "Rows excluded because of conflicting "
        f"duplicate targets: {excluded_conflict_rows}"
    )

    if conflicting_groups:
        print()
        print(
            "CONFLICTING DUPLICATE GROUPS"
        )

        for number, group in enumerate(
            conflicting_groups,
            start=1,
        ):
            print()
            print(
                f"Conflict group {number}"
            )
            print("-" * 80)

            print(
                group[
                    [
                        "problem",
                        "rating",
                        "rating_std",
                        "contest",
                        "tag",
                    ]
                ].to_string(
                    index=False
                )
            )

    cleaned = dataframe.loc[
        rows_to_keep
    ].copy()

    cleaned = cleaned.reset_index(
        drop=True
    )

    remaining_duplicates = (
        cleaned[
            "normalized_problem"
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    if remaining_duplicates != 0:
        raise RuntimeError(
            "Duplicate cleanup failed."
        )

    print()
    print(
        f"Final development rows: "
        f"{len(cleaned):,}"
    )

    print(
        "PASS: each normalized question now "
        "has exactly one target."
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
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )


def evaluate_subgroups(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
) -> None:
    print_section(
        f"SUBGROUP ANALYSIS: {model_name}"
    )

    analysis = dataframe.copy()

    analysis["prediction"] = (
        predictions
    )

    rows = []

    for tag, group in analysis.groupby(
        "tag"
    ):
        if len(group) < 2:
            continue

        metrics = calculate_metrics(
            group["rating"]
            .to_numpy(
                dtype=float
            ),
            group["prediction"]
            .to_numpy(
                dtype=float
            ),
        )

        rows.append(
            {
                "tag": tag,
                "n": len(group),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "rmse"
    )

    print(
        results.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )


def main() -> None:
    print_section(
        "E2H-AMC REPEATED-CV "
        "SEMANTIC BASELINE"
    )

    dataframe = pd.read_parquet(
        TRAIN_PATH
    )

    required_columns = {
        "problem",
        "rating",
        "rating_std",
        "contest",
        "tag",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = prepare_development_data(
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

    print_section(
        "GENERATING TRANSFORMER EMBEDDINGS"
    )

    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    embeddings = (
        embedding_model.encode(
            problems.tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    )

    print(
        "Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    print_section(
        "REPEATED CROSS-VALIDATION"
    )

    cross_validator = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    total_runs = (
        N_SPLITS
        * N_REPEATS
    )

    print(
        f"Development rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Folds: {N_SPLITS}"
    )

    print(
        f"Repeats: {N_REPEATS}"
    )

    print(
        f"Total runs: {total_runs}"
    )

    model_names = [
        "Mean Predictor",
        "TF-IDF + Ridge",
        "MiniLM Embeddings + Ridge",
    ]

    prediction_sums = {
        name: np.zeros(
            len(dataframe)
        )
        for name in model_names
    }

    prediction_counts = {
        name: np.zeros(
            len(dataframe),
            dtype=int,
        )
        for name in model_names
    }

    fold_results = []

    for run_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(
            problems
        ),
        start=1,
    ):
        print(
            f"Run {run_number}/"
            f"{total_runs}"
        )

        y_train = ratings[
            train_indices
        ]

        y_validation = ratings[
            validation_indices
        ]

        # -----------------------------
        # Mean predictor
        # -----------------------------

        mean_model = DummyRegressor(
            strategy="mean"
        )

        mean_model.fit(
            np.zeros(
                (
                    len(train_indices),
                    1,
                )
            ),
            y_train,
        )

        mean_predictions = (
            mean_model.predict(
                np.zeros(
                    (
                        len(
                            validation_indices
                        ),
                        1,
                    )
                )
            )
        )

        mean_metrics = (
            calculate_metrics(
                y_validation,
                mean_predictions,
            )
        )

        fold_results.append(
            {
                "run": run_number,
                "model": (
                    "Mean Predictor"
                ),
                **mean_metrics,
            }
        )

        prediction_sums[
            "Mean Predictor"
        ][validation_indices] += (
            mean_predictions
        )

        prediction_counts[
            "Mean Predictor"
        ][validation_indices] += 1

        # -----------------------------
        # TF-IDF
        # -----------------------------

        tfidf_model = (
            create_tfidf_model()
        )

        tfidf_model.fit(
            problems[train_indices],
            y_train,
        )

        tfidf_predictions = (
            tfidf_model.predict(
                problems[
                    validation_indices
                ]
            )
        )

        tfidf_metrics = (
            calculate_metrics(
                y_validation,
                tfidf_predictions,
            )
        )

        fold_results.append(
            {
                "run": run_number,
                "model": (
                    "TF-IDF + Ridge"
                ),
                **tfidf_metrics,
            }
        )

        prediction_sums[
            "TF-IDF + Ridge"
        ][validation_indices] += (
            tfidf_predictions
        )

        prediction_counts[
            "TF-IDF + Ridge"
        ][validation_indices] += 1

        # -----------------------------
        # MiniLM embeddings
        # -----------------------------

        embedding_regressor = Ridge(
            alpha=1.0
        )

        embedding_regressor.fit(
            embeddings[
                train_indices
            ],
            y_train,
        )

        embedding_predictions = (
            embedding_regressor.predict(
                embeddings[
                    validation_indices
                ]
            )
        )

        embedding_metrics = (
            calculate_metrics(
                y_validation,
                embedding_predictions,
            )
        )

        fold_results.append(
            {
                "run": run_number,
                "model": (
                    "MiniLM Embeddings "
                    "+ Ridge"
                ),
                **embedding_metrics,
            }
        )

        prediction_sums[
            "MiniLM Embeddings + Ridge"
        ][validation_indices] += (
            embedding_predictions
        )

        prediction_counts[
            "MiniLM Embeddings + Ridge"
        ][validation_indices] += 1

    print_section(
        "OOF COVERAGE"
    )

    for model_name in model_names:
        unique_counts = np.unique(
            prediction_counts[
                model_name
            ]
        )

        print(
            f"{model_name}: "
            f"{unique_counts.tolist()}"
        )

        if not np.all(
            prediction_counts[
                model_name
            ] == N_REPEATS
        ):
            raise RuntimeError(
                "Invalid OOF coverage."
            )

    fold_results_df = pd.DataFrame(
        fold_results
    )

    print_section(
        "REPEATED-CV SUMMARY"
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
        subset = fold_results_df[
            fold_results_df["model"]
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
        "AVERAGED OUT-OF-FOLD PERFORMANCE"
    )

    averaged_predictions = {}
    oof_rows = []

    for model_name in model_names:
        predictions = (
            prediction_sums[
                model_name
            ]
            / prediction_counts[
                model_name
            ]
        )

        averaged_predictions[
            model_name
        ] = predictions

        result = calculate_metrics(
            ratings,
            predictions,
        )

        oof_rows.append(
            {
                "model": model_name,
                **result,
            }
        )

    oof_df = pd.DataFrame(
        oof_rows
    )

    print(
        oof_df.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    evaluate_subgroups(
        dataframe,
        averaged_predictions[
            "TF-IDF + Ridge"
        ],
        "TF-IDF + Ridge",
    )

    evaluate_subgroups(
        dataframe,
        averaged_predictions[
            "MiniLM Embeddings + Ridge"
        ],
        "MiniLM Embeddings + Ridge",
    )

    print_section(
        "EXPERIMENT COMPLETE"
    )

    print(
        "Official E2H-AMC eval split "
        "was not loaded."
    )

    print(
        "Conflicting exact duplicate "
        "groups were excluded rather "
        "than relabelled."
    )

    print(
        "No hyperparameter tuning "
        "was performed."
    )


if __name__ == "__main__":
    main()