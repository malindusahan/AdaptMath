from pathlib import Path
import gc
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import BertModel, BertTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "e2h_amc_train_clean.parquet"
)

RAW_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "e2h_amc"
    / "e2h_amc_train.parquet"
)

MATHBERT_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "e2h_amc_mathbert_train_embeddings.npz"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MATHBERT_NAME = "tbs17/MathBERT"

MAX_LENGTH = 512

# Conservative CPU batch size for the user's laptop.
BATCH_SIZE = 4

ALPHA_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# ============================================================
# PRINTING
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# DATA UTILITIES
# ============================================================

def normalize_problem(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def dataset_fingerprint(
    development: pd.DataFrame,
) -> str:
    """
    Creates a reproducible fingerprint so cached MathBERT
    embeddings are never silently reused with different data.
    """

    digest = hashlib.sha256()

    for row in development.itertuples(
        index=False
    ):
        digest.update(
            str(row.source_row_id).encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

        digest.update(
            str(row.problem).encode(
                "utf-8"
            )
        )

        digest.update(b"\n")

    return digest.hexdigest()


def load_development_data():
    print_section(
        "LOADING DEVELOPMENT DATA"
    )

    clean = pd.read_parquet(
        CLEAN_TRAIN_PATH
    )

    raw = pd.read_parquet(
        RAW_TRAIN_PATH
    )

    if len(raw) != 1000:
        raise RuntimeError(
            f"Expected 1000 raw rows, "
            f"found {len(raw)}."
        )

    if len(clean) != 998:
        raise RuntimeError(
            f"Expected 998 cleaned rows, "
            f"found {len(clean)}."
        )

    required_clean = {
        "source_row_id",
        "problem",
        "rating",
    }

    if not required_clean.issubset(
        clean.columns
    ):
        raise RuntimeError(
            "Clean dataset is missing "
            "required columns."
        )

    if not clean[
        "source_row_id"
    ].is_unique:
        raise RuntimeError(
            "source_row_id is not unique."
        )

    required_raw = {
        "tag",
        "contest",
    }

    if not required_raw.issubset(
        raw.columns
    ):
        raise RuntimeError(
            "Raw dataset is missing "
            "tag/contest metadata."
        )

    # source_row_id is the original raw row position.
    # It is NOT E2H-AMC's own non-unique "index" column.
    raw_metadata = raw.reset_index(
        drop=True
    ).copy()

    raw_metadata[
        "source_row_id"
    ] = np.arange(
        len(raw_metadata)
    )

    metadata = raw_metadata[
        [
            "source_row_id",
            "tag",
            "contest",
        ]
    ].copy()

    development = clean.merge(
        metadata,
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )

    if len(development) != 998:
        raise RuntimeError(
            "Metadata merge changed "
            "the row count."
        )

    if development[
        ["tag", "contest"]
    ].isna().any().any():
        raise RuntimeError(
            "Some metadata could not "
            "be reconstructed."
        )

    X_text = (
        development["problem"]
        .astype(str)
        .to_numpy()
    )

    y = (
        development["rating"]
        .astype(float)
        .to_numpy()
    )

    if not np.all(
        (y > 0.0)
        & (y < 1.0)
    ):
        raise RuntimeError(
            "Ratings must be strictly "
            "inside (0,1)."
        )

    groups = (
        development["tag"]
        .astype(str)
        .to_numpy()
    )

    print(
        f"Raw rows: {len(raw)}"
    )

    print(
        f"Clean development rows: "
        f"{len(development)}"
    )

    print(
        "Prediction input: problem text only"
    )

    print(
        "Target: published E2H-AMC rating"
    )

    print()
    print(
        "Tag/contest are NOT predictor features."
    )

    print(
        "They are used only for "
        "generalization evaluation."
    )

    return (
        development,
        X_text,
        y,
        groups,
    )


# ============================================================
# METRICS
# ============================================================

def safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    if len(y_true) < 2:
        return float("nan")

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return float("nan")

    return float(
        pearsonr(
            y_true,
            y_pred,
        ).statistic
    )


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    if len(y_true) < 2:
        return float("nan")

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return float("nan")

    return float(
        spearmanr(
            y_true,
            y_pred,
        ).statistic
    )


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

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": safe_pearson(
            y_true,
            y_pred,
        ),
        "spearman": safe_spearman(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# BOUNDED REGRESSION
# ============================================================

def make_bounded_regressor():
    """
    Difficulty is continuous and bounded in (0,1).

    Ridge learns the logit-transformed target.
    Predictions are mapped back using sigmoid/expit.

    There are no manually defined difficulty thresholds.
    """

    return TransformedTargetRegressor(
        regressor=Ridge(
            alpha=1.0,
            solver="lsqr",
        ),
        func=logit,
        inverse_func=expit,
        check_inverse=True,
    )


# ============================================================
# WORD + CHARACTER BASELINE
# ============================================================

def build_word_char_model():
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "regressor",
                make_bounded_regressor(),
            ),
        ]
    )


# ============================================================
# MATHBERT DOWNSTREAM REGRESSION
# ============================================================

def build_mathbert_regression_model():
    """
    MathBERT is frozen.

    Embeddings are fixed features.

    StandardScaler is fitted inside each CV training fold,
    followed by bounded Ridge regression.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                make_bounded_regressor(),
            ),
        ]
    )


def parameter_grid() -> dict:
    return {
        "regressor__regressor__alpha":
            ALPHA_GRID
    }


# ============================================================
# MATHBERT TOKEN / LENGTH ANALYSIS
# ============================================================

def inspect_mathbert_lengths(
    tokenizer,
    texts: np.ndarray,
) -> None:

    print_section(
        "MATHBERT TOKEN LENGTH AUDIT"
    )

    lengths = []

    for text in texts:
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=False,
        )

        lengths.append(
            len(
                encoded["input_ids"]
            )
        )

    lengths = np.asarray(
        lengths,
        dtype=int,
    )

    truncated = int(
        np.sum(
            lengths > MAX_LENGTH
        )
    )

    print(
        f"Problems: {len(lengths)}"
    )

    print(
        f"Median tokens: "
        f"{np.median(lengths):.0f}"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(lengths, 95):.0f}"
    )

    print(
        f"Maximum tokens: "
        f"{lengths.max()}"
    )

    print(
        f"Problems exceeding "
        f"{MAX_LENGTH} tokens: "
        f"{truncated}"
    )

    print(
        f"Percentage truncated: "
        f"{100 * truncated / len(lengths):.2f}%"
    )


# ============================================================
# MATHBERT EMBEDDINGS
# ============================================================

def create_mathbert_embeddings(
    texts: np.ndarray,
):
    print_section(
        "CREATING MATHBERT EMBEDDINGS"
    )

    print(
        f"Loading locally cached: "
        f"{MATHBERT_NAME}"
    )

    tokenizer = (
        BertTokenizer.from_pretrained(
            MATHBERT_NAME,
            local_files_only=True,
        )
    )

    model = (
        BertModel.from_pretrained(
            MATHBERT_NAME,
            local_files_only=True,
        )
    )

    model.eval()

    model.to("cpu")

    inspect_mathbert_lengths(
        tokenizer,
        texts,
    )

    cls_batches = []
    mean_batches = []

    total = len(texts)

    total_batches = int(
        np.ceil(
            total / BATCH_SIZE
        )
    )

    print_section(
        "MATHBERT CPU FEATURE EXTRACTION"
    )

    print(
        f"Items: {total}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Batches: {total_batches}"
    )

    print(
        "This is feature extraction only."
    )

    print(
        "MathBERT weights are NOT "
        "fine-tuned here."
    )

    for batch_number, start in enumerate(
        range(
            0,
            total,
            BATCH_SIZE,
        ),
        start=1,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_texts = [
            str(value)
            for value
            in texts[start:end]
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        special_tokens_mask = (
            encoded.pop(
                "special_tokens_mask"
            )
        )

        with torch.inference_mode():
            outputs = model(
                **encoded
            )

            hidden = (
                outputs.last_hidden_state
            )

            # --------------------------------------------
            # Representation A:
            # final hidden state of [CLS]
            # --------------------------------------------

            cls_embedding = (
                hidden[:, 0, :]
            )

            # --------------------------------------------
            # Representation B:
            # mean of CONTENT token representations.
            #
            # Padding and special tokens are excluded.
            # --------------------------------------------

            attention_mask = (
                encoded[
                    "attention_mask"
                ]
            )

            content_mask = (
                attention_mask
                * (
                    1
                    - special_tokens_mask
                )
            )

            content_mask = (
                content_mask
                .unsqueeze(-1)
                .to(
                    hidden.dtype
                )
            )

            denominator = (
                content_mask.sum(
                    dim=1
                )
                .clamp(
                    min=1.0
                )
            )

            mean_embedding = (
                (
                    hidden
                    * content_mask
                )
                .sum(
                    dim=1
                )
                / denominator
            )

        cls_batches.append(
            cls_embedding
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        mean_batches.append(
            mean_embedding
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            batch_number == 1
            or batch_number
            % 10 == 0
            or batch_number
            == total_batches
        ):
            print(
                f"Batch "
                f"{batch_number}/"
                f"{total_batches} "
                f"| items "
                f"{end}/{total}"
            )

    cls_embeddings = np.vstack(
        cls_batches
    )

    mean_embeddings = np.vstack(
        mean_batches
    )

    print()
    print(
        "CLS embedding matrix: "
        f"{cls_embeddings.shape}"
    )

    print(
        "Mean embedding matrix: "
        f"{mean_embeddings.shape}"
    )

    del model
    del tokenizer

    gc.collect()

    return (
        cls_embeddings,
        mean_embeddings,
    )


def load_or_create_mathbert_embeddings(
    development: pd.DataFrame,
    texts: np.ndarray,
):
    fingerprint = dataset_fingerprint(
        development
    )

    source_ids = (
        development[
            "source_row_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if MATHBERT_CACHE_PATH.exists():

        print_section(
            "CHECKING MATHBERT EMBEDDING CACHE"
        )

        try:
            cache = np.load(
                MATHBERT_CACHE_PATH,
                allow_pickle=False,
            )

            cached_fingerprint = str(
                cache[
                    "fingerprint"
                ][0]
            )

            cached_ids = cache[
                "source_row_id"
            ]

            cached_cls = cache[
                "cls_embeddings"
            ]

            cached_mean = cache[
                "mean_embeddings"
            ]

            valid = (
                cached_fingerprint
                == fingerprint
                and np.array_equal(
                    cached_ids,
                    source_ids,
                )
                and cached_cls.shape
                == (
                    len(texts),
                    768,
                )
                and cached_mean.shape
                == (
                    len(texts),
                    768,
                )
            )

            if valid:
                print(
                    "Valid cached embeddings found."
                )

                print(
                    f"Cache: "
                    f"{MATHBERT_CACHE_PATH}"
                )

                return (
                    cached_cls,
                    cached_mean,
                )

            print(
                "Cache does not match "
                "the current dataset."
            )

            print(
                "Embeddings will be "
                "recomputed."
            )

        except Exception as exc:
            print(
                "Could not use embedding cache:"
            )

            print(
                str(exc)
            )

            print(
                "Embeddings will be recomputed."
            )

    cls_embeddings, mean_embeddings = (
        create_mathbert_embeddings(
            texts
        )
    )

    MATHBERT_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        MATHBERT_CACHE_PATH,
        fingerprint=np.asarray(
            [fingerprint]
        ),
        source_row_id=source_ids,
        cls_embeddings=cls_embeddings,
        mean_embeddings=mean_embeddings,
    )

    print()
    print(
        "Saved MathBERT embedding cache:"
    )

    print(
        MATHBERT_CACHE_PATH
    )

    return (
        cls_embeddings,
        mean_embeddings,
    )


# ============================================================
# NESTED RANDOM CROSS-VALIDATION
# ============================================================

def run_nested_cv(
    X_text: np.ndarray,
    X_cls: np.ndarray,
    X_mean: np.ndarray,
    y: np.ndarray,
):
    print_section(
        "NESTED REPEATED CROSS-VALIDATION"
    )

    print(
        "Outer evaluation: "
        "5 folds x 3 repeats"
    )

    print(
        "Inner model selection: "
        "3-fold CV"
    )

    print(
        f"Ridge alpha candidates: "
        f"{ALPHA_GRID}"
    )

    candidates = {
        "Word+Char TF-IDF": (
            X_text,
            build_word_char_model,
        ),
        "MathBERT CLS": (
            X_cls,
            build_mathbert_regression_model,
        ),
        "MathBERT Mean": (
            X_mean,
            build_mathbert_regression_model,
        ),
    }

    outer_cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=RANDOM_STATE,
    )

    indices = np.arange(
        len(y)
    )

    splits = list(
        outer_cv.split(
            indices
        )
    )

    rows = []

    for (
        model_name,
        (
            X,
            builder,
        ),
    ) in candidates.items():

        print()
        print("-" * 80)
        print(model_name)
        print("-" * 80)

        for fold_number, (
            train_index,
            test_index,
        ) in enumerate(
            splits,
            start=1,
        ):

            inner_cv = KFold(
                n_splits=3,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            search = GridSearchCV(
                estimator=builder(),
                param_grid=parameter_grid(),
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                cv=inner_cv,
                refit=True,
                n_jobs=1,
            )

            search.fit(
                X[train_index],
                y[train_index],
            )

            predictions = (
                search.predict(
                    X[test_index]
                )
            )

            metrics = calculate_metrics(
                y[test_index],
                predictions,
            )

            alpha = float(
                search.best_params_[
                    "regressor__regressor__alpha"
                ]
            )

            out_of_range = int(
                np.sum(
                    (predictions < 0.0)
                    | (predictions > 1.0)
                )
            )

            rows.append(
                {
                    "model":
                        model_name,
                    "fold":
                        fold_number,
                    "alpha":
                        alpha,
                    **metrics,
                    "out_of_range":
                        out_of_range,
                }
            )

            print(
                f"Fold "
                f"{fold_number:02d}/15 "
                f"| alpha={alpha:g} "
                f"| RMSE="
                f"{metrics['rmse']:.4f} "
                f"| MAE="
                f"{metrics['mae']:.4f} "
                f"| R2="
                f"{metrics['r2']:.4f} "
                f"| Spearman="
                f"{metrics['spearman']:.4f}"
            )

    results = pd.DataFrame(
        rows
    )

    summary_rows = []

    for model_name in (
        candidates.keys()
    ):

        subset = results.loc[
            results["model"]
            == model_name
        ]

        summary_rows.append(
            {
                "model":
                    model_name,
                "rmse_mean":
                    subset[
                        "rmse"
                    ].mean(),
                "rmse_std":
                    subset[
                        "rmse"
                    ].std(),
                "mae_mean":
                    subset[
                        "mae"
                    ].mean(),
                "mae_std":
                    subset[
                        "mae"
                    ].std(),
                "r2_mean":
                    subset[
                        "r2"
                    ].mean(),
                "pearson_mean":
                    subset[
                        "pearson"
                    ].mean(),
                "spearman_mean":
                    subset[
                        "spearman"
                    ].mean(),
                "out_of_range":
                    int(
                        subset[
                            "out_of_range"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print_section(
        "NESTED CV SUMMARY"
    )

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
            ),
        )
    )

    winner = str(
        summary.sort_values(
            "rmse_mean",
            ascending=True,
        )
        .iloc[0]["model"]
    )

    print()
    print(
        "Best nested-CV RMSE:"
    )

    print(
        winner
    )

    return summary


# ============================================================
# LEAVE-ONE-TAG-OUT GENERALIZATION
# ============================================================

def run_cross_tag_evaluation(
    X_text: np.ndarray,
    X_cls: np.ndarray,
    X_mean: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
):
    print_section(
        "LEAVE-ONE-TAG-OUT GENERALIZATION"
    )

    print(
        "Outer test group = one completely "
        "held-out competition tag."
    )

    print(
        "Inner alpha tuning is also "
        "group-aware."
    )

    print()
    print(
        "The tag is NEVER supplied "
        "to a predictor."
    )

    candidates = {
        "Word+Char TF-IDF": (
            X_text,
            build_word_char_model,
        ),
        "MathBERT CLS": (
            X_cls,
            build_mathbert_regression_model,
        ),
        "MathBERT Mean": (
            X_mean,
            build_mathbert_regression_model,
        ),
    }

    logo = LeaveOneGroupOut()

    summary_rows = []

    for (
        model_name,
        (
            X,
            builder,
        ),
    ) in candidates.items():

        print()
        print("-" * 80)
        print(model_name)
        print("-" * 80)

        all_predictions = np.full(
            len(y),
            np.nan,
            dtype=float,
        )

        for (
            train_index,
            test_index,
        ) in logo.split(
            X,
            y,
            groups=groups,
        ):

            train_groups = (
                groups[
                    train_index
                ]
            )

            unique_train_groups = (
                np.unique(
                    train_groups
                )
            )

            inner_splits = min(
                3,
                len(
                    unique_train_groups
                ),
            )

            if inner_splits < 2:
                raise RuntimeError(
                    "Not enough groups "
                    "for inner CV."
                )

            inner_cv = GroupKFold(
                n_splits=inner_splits
            )

            search = GridSearchCV(
                estimator=builder(),
                param_grid=parameter_grid(),
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                cv=inner_cv,
                refit=True,
                n_jobs=1,
            )

            search.fit(
                X[train_index],
                y[train_index],
                groups=train_groups,
            )

            predictions = (
                search.predict(
                    X[test_index]
                )
            )

            all_predictions[
                test_index
            ] = predictions

            held_out_tag = str(
                np.unique(
                    groups[
                        test_index
                    ]
                )[0]
            )

            metrics = calculate_metrics(
                y[test_index],
                predictions,
            )

            alpha = float(
                search.best_params_[
                    "regressor__regressor__alpha"
                ]
            )

            print(
                f"{held_out_tag:20s} "
                f"| n="
                f"{len(test_index):4d} "
                f"| alpha="
                f"{alpha:g} "
                f"| RMSE="
                f"{metrics['rmse']:.4f} "
                f"| MAE="
                f"{metrics['mae']:.4f} "
                f"| R2="
                f"{metrics['r2']:.4f} "
                f"| Spearman="
                f"{metrics['spearman']:.4f}"
            )

        if np.isnan(
            all_predictions
        ).any():
            raise RuntimeError(
                "Cross-tag predictions "
                "are incomplete."
            )

        overall = calculate_metrics(
            y,
            all_predictions,
        )

        out_of_range = int(
            np.sum(
                (all_predictions < 0.0)
                | (all_predictions > 1.0)
            )
        )

        summary_rows.append(
            {
                "model":
                    model_name,
                **overall,
                "out_of_range":
                    out_of_range,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print_section(
        "CROSS-TAG OVERALL RESULT"
    )

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
            ),
        )
    )

    winner = str(
        summary.sort_values(
            "rmse",
            ascending=True,
        )
        .iloc[0]["model"]
    )

    print()
    print(
        "Best cross-tag RMSE:"
    )

    print(
        winner
    )

    return summary


# ============================================================
# DECISION REPORT
# ============================================================

def print_decision_report(
    nested_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
) -> None:

    nested_winner = str(
        nested_summary.sort_values(
            "rmse_mean",
            ascending=True,
        )
        .iloc[0]["model"]
    )

    cross_winner = str(
        cross_summary.sort_values(
            "rmse",
            ascending=True,
        )
        .iloc[0]["model"]
    )

    print_section(
        "ARCHITECTURE COMPARISON RESULT"
    )

    print(
        f"Nested-CV RMSE winner: "
        f"{nested_winner}"
    )

    print(
        f"Cross-tag RMSE winner: "
        f"{cross_winner}"
    )

    print()

    if nested_winner == cross_winner:

        print(
            "RESULT: CLEAR DEVELOPMENT WINNER"
        )

        print(
            f"Architecture: "
            f"{nested_winner}"
        )

        print()
        print(
            "The same architecture achieved "
            "the lowest RMSE under both "
            "development protocols."
        )

    else:

        print(
            "RESULT: NO UNAMBIGUOUS WINNER"
        )

        print()
        print(
            "The standard nested-CV and "
            "cross-tag robustness tests "
            "prefer different architectures."
        )

        print(
            "Do NOT automatically choose "
            "one using the official "
            "evaluation benchmark."
        )

    print()
    print(
        "Official E2H-AMC evaluation "
        "used in this run: NO"
    )

    print(
        "complexity_model.joblib "
        "overwritten in this run: NO"
    )

    print()
    print(
        "No Easy/Medium/Hard thresholds."
    )

    print(
        "No manually assigned "
        "difficulty rules."
    )

    print(
        "No contest/tag predictor feature."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_section(
        "ADAPTMATH FINAL ARCHITECTURE COMPARISON"
    )

    print(
        "Candidate 1: "
        "Word+Char TF-IDF + "
        "bounded Ridge"
    )

    print(
        "Candidate 2: "
        "MathBERT [CLS] + "
        "bounded Ridge"
    )

    print(
        "Candidate 3: "
        "MathBERT mean pooling + "
        "bounded Ridge"
    )

    (
        development,
        X_text,
        y,
        groups,
    ) = load_development_data()

    # --------------------------------------------------------
    # Frozen pretrained MathBERT features
    # --------------------------------------------------------

    (
        X_cls,
        X_mean,
    ) = load_or_create_mathbert_embeddings(
        development,
        X_text,
    )

    if X_cls.shape != (
        len(y),
        768,
    ):
        raise RuntimeError(
            "Unexpected CLS embedding shape."
        )

    if X_mean.shape != (
        len(y),
        768,
    ):
        raise RuntimeError(
            "Unexpected mean embedding shape."
        )

    # --------------------------------------------------------
    # Same-distribution nested evaluation
    # --------------------------------------------------------

    nested_summary = run_nested_cv(
        X_text,
        X_cls,
        X_mean,
        y,
    )

    # --------------------------------------------------------
    # Cross-competition robustness
    # --------------------------------------------------------

    cross_summary = (
        run_cross_tag_evaluation(
            X_text,
            X_cls,
            X_mean,
            y,
            groups,
        )
    )

    # --------------------------------------------------------
    # Evidence-based architecture decision
    # --------------------------------------------------------

    print_decision_report(
        nested_summary,
        cross_summary,
    )


if __name__ == "__main__":
    main()