from pathlib import Path
import json

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

DATA_DIR = (
    PROJECT_ROOT
    / "external"
    / "Visual_Item_Difficulty"
    / "data"
)

ITEMS_PATH = DATA_DIR / "items.csv"
CV_PATH = DATA_DIR / "cv_folds.json"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


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

    if np.std(y_pred) < 1e-12:
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
                "ridge",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )


def build_question_plus_visual(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    question = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    visual = (
        dataframe["visual_description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined = []

    for question_text, visual_text in zip(
        question,
        visual,
    ):
        if visual_text:
            combined.append(
                (
                    f"{question_text}\n\n"
                    f"Visual information:\n"
                    f"{visual_text}"
                )
            )
        else:
            combined.append(
                question_text
            )

    return np.array(
        combined,
        dtype=object,
    )


def main() -> None:
    print_section(
        "EEDI FROZEN 5-FOLD BASELINE EXPERIMENT"
    )

    items = pd.read_csv(
        ITEMS_PATH
    )

    with open(
        CV_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        cv_data = json.load(file)

    # -------------------------------------------------
    # DEVELOPMENT DATA ONLY
    # -------------------------------------------------

    train_items = (
        items.loc[
            items["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train_items) != 580:
        raise RuntimeError(
            "Expected exactly 580 training items."
        )

    print(
        f"Development items: {len(train_items):,}"
    )

    print(
        "Held-out test items loaded for modelling: 0"
    )

    required_columns = {
        "QuestionId",
        "difficulty",
        "text",
        "visual_description",
    }

    missing_columns = (
        required_columns
        - set(train_items.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if train_items[
        [
            "QuestionId",
            "difficulty",
            "text",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing required training values."
        )

    # -------------------------------------------------
    # INPUT REPRESENTATIONS
    # -------------------------------------------------

    question_text = (
        train_items["text"]
        .astype(str)
        .to_numpy()
    )

    question_plus_visual = (
        build_question_plus_visual(
            train_items
        )
    )

    ratings = (
        train_items["difficulty"]
        .astype(float)
        .to_numpy()
    )

    question_ids = (
        train_items["QuestionId"]
        .astype(int)
        .to_numpy()
    )

    id_to_index = {
        int(question_id): index
        for index, question_id in enumerate(
            question_ids
        )
    }

    # -------------------------------------------------
    # FROZEN CV VALIDATION
    # -------------------------------------------------

    print_section(
        "FROZEN CV PROTOCOL"
    )

    folds = cv_data["folds"]

    print(
        f"Number of folds: {len(folds)}"
    )

    print(
        f"Fold seed: {cv_data.get('fold_seed')}"
    )

    print(
        "Stratification: "
        f"{cv_data.get('stratify')}"
    )

    if len(folds) != 5:
        raise RuntimeError(
            "Expected exactly five folds."
        )

    # -------------------------------------------------
    # MODELS
    # -------------------------------------------------

    model_names = [
        "Mean Predictor",
        "TF-IDF Q",
        "TF-IDF Q+D",
    ]

    fold_results = []

    oof_predictions = {
        name: np.full(
            len(train_items),
            np.nan,
            dtype=float,
        )
        for name in model_names
    }

    # -------------------------------------------------
    # RUN FROZEN FOLDS
    # -------------------------------------------------

    print_section(
        "RUNNING FROZEN 5-FOLD CV"
    )

    all_train_ids = set(
        question_ids.tolist()
    )

    for fold_info in folds:
        fold_number = int(
            fold_info["fold"]
        )

        validation_ids = {
            int(question_id)
            for question_id in fold_info[
                "val_qids"
            ]
        }

        training_ids = (
            all_train_ids
            - validation_ids
        )

        validation_indices = np.array(
            [
                id_to_index[question_id]
                for question_id in validation_ids
            ],
            dtype=int,
        )

        training_indices = np.array(
            [
                id_to_index[question_id]
                for question_id in training_ids
            ],
            dtype=int,
        )

        if len(validation_indices) != 116:
            raise RuntimeError(
                f"Fold {fold_number} does not "
                "contain 116 validation items."
            )

        if len(training_indices) != 464:
            raise RuntimeError(
                f"Fold {fold_number} does not "
                "contain 464 training items."
            )

        print(
            f"Fold {fold_number}: "
            f"train={len(training_indices)}, "
            f"validation={len(validation_indices)}"
        )

        y_train = ratings[
            training_indices
        ]

        y_validation = ratings[
            validation_indices
        ]

        # -------------------------------------------------
        # MODEL 1: MEAN PREDICTOR
        # -------------------------------------------------

        mean_model = DummyRegressor(
            strategy="mean"
        )

        mean_model.fit(
            np.zeros(
                (
                    len(training_indices),
                    1,
                )
            ),
            y_train,
        )

        mean_predictions = mean_model.predict(
            np.zeros(
                (
                    len(validation_indices),
                    1,
                )
            )
        )

        mean_metrics = calculate_metrics(
            y_validation,
            mean_predictions,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "model": "Mean Predictor",
                **mean_metrics,
            }
        )

        oof_predictions[
            "Mean Predictor"
        ][validation_indices] = (
            mean_predictions
        )

        # -------------------------------------------------
        # MODEL 2: TF-IDF QUESTION ONLY
        # -------------------------------------------------

        q_model = create_tfidf_model()

        q_model.fit(
            question_text[
                training_indices
            ],
            y_train,
        )

        q_predictions = q_model.predict(
            question_text[
                validation_indices
            ]
        )

        q_metrics = calculate_metrics(
            y_validation,
            q_predictions,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "model": "TF-IDF Q",
                **q_metrics,
            }
        )

        oof_predictions[
            "TF-IDF Q"
        ][validation_indices] = (
            q_predictions
        )

        # -------------------------------------------------
        # MODEL 3: TF-IDF QUESTION + VISUAL DESCRIPTION
        # -------------------------------------------------

        qd_model = create_tfidf_model()

        qd_model.fit(
            question_plus_visual[
                training_indices
            ],
            y_train,
        )

        qd_predictions = qd_model.predict(
            question_plus_visual[
                validation_indices
            ]
        )

        qd_metrics = calculate_metrics(
            y_validation,
            qd_predictions,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "model": "TF-IDF Q+D",
                **qd_metrics,
            }
        )

        oof_predictions[
            "TF-IDF Q+D"
        ][validation_indices] = (
            qd_predictions
        )

    # -------------------------------------------------
    # OOF COVERAGE
    # -------------------------------------------------

    print_section(
        "OUT-OF-FOLD COVERAGE"
    )

    for model_name in model_names:
        missing_predictions = int(
            np.isnan(
                oof_predictions[
                    model_name
                ]
            ).sum()
        )

        print(
            f"{model_name}: "
            f"missing OOF predictions = "
            f"{missing_predictions}"
        )

        if missing_predictions != 0:
            raise RuntimeError(
                "OOF coverage is incomplete."
            )

    # -------------------------------------------------
    # FOLD SUMMARY
    # -------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )

    print_section(
        "FOLD-BY-FOLD RESULTS"
    )

    print(
        fold_results_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # -------------------------------------------------
    # MEAN +/- STD
    # -------------------------------------------------

    print_section(
        "5-FOLD SUMMARY"
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
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # -------------------------------------------------
    # FULL OOF PERFORMANCE
    # -------------------------------------------------

    print_section(
        "FULL OUT-OF-FOLD PERFORMANCE"
    )

    oof_rows = []

    for model_name in model_names:
        metrics_result = calculate_metrics(
            ratings,
            oof_predictions[
                model_name
            ],
        )

        oof_rows.append(
            {
                "model": model_name,
                **metrics_result,
            }
        )

    oof_df = pd.DataFrame(
        oof_rows
    )

    print(
        oof_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # -------------------------------------------------
    # Q VS Q+D PAIRED FOLD DIFFERENCE
    # -------------------------------------------------

    print_section(
        "Q VS Q+D RMSE DIFFERENCE"
    )

    q_results = (
        fold_results_df[
            fold_results_df["model"]
            == "TF-IDF Q"
        ]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    qd_results = (
        fold_results_df[
            fold_results_df["model"]
            == "TF-IDF Q+D"
        ]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    comparison = pd.DataFrame(
        {
            "fold": q_results["fold"],
            "q_rmse": q_results["rmse"],
            "qd_rmse": qd_results["rmse"],
        }
    )

    comparison[
        "rmse_change_qd_minus_q"
    ] = (
        comparison["qd_rmse"]
        - comparison["q_rmse"]
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()

    mean_change = float(
        comparison[
            "rmse_change_qd_minus_q"
        ].mean()
    )

    print(
        "Mean Q+D minus Q RMSE change: "
        f"{mean_change:.4f}"
    )

    if mean_change < 0:
        print(
            "Q+D improved RMSE on average."
        )
    elif mean_change > 0:
        print(
            "Q+D worsened RMSE on average."
        )
    else:
        print(
            "Q and Q+D had identical "
            "mean RMSE."
        )

    # -------------------------------------------------
    # FINAL STATEMENT
    # -------------------------------------------------

    print_section(
        "EXPERIMENT COMPLETE"
    )

    print(
        "Only the 580 published training "
        "items were used."
    )

    print(
        "The authors' frozen 5-fold CV "
        "assignments were used unchanged."
    )

    print(
        "The held-out 145-item test split "
        "was not evaluated."
    )

    print(
        "No hyperparameter tuning was "
        "performed."
    )

    print(
        "No dataset labels were modified."
    )


if __name__ == "__main__":
    main()