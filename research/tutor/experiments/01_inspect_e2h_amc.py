from pathlib import Path

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "e2h_amc"

TRAIN_PATH = RAW_DIR / "e2h_amc_train.parquet"
EVAL_PATH = RAW_DIR / "e2h_amc_eval.parquet"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inspect_split(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    print_section(f"{name.upper()} SPLIT")

    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nColumn names:")
    for column in dataframe.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    missing = dataframe.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values.")
    else:
        print(missing.to_string())

    difficulty_columns = [
        "rating",
        "rating_std",
        "rating_quantile",
        "item_difficulty",
        "unnorm_rating",
    ]

    available_difficulty_columns = [
        column
        for column in difficulty_columns
        if column in dataframe.columns
    ]

    print("\nDifficulty statistics:")

    if available_difficulty_columns:
        print(
            dataframe[
                available_difficulty_columns
            ].describe().to_string()
        )
    else:
        print("  No expected difficulty columns found.")

    if "contest" in dataframe.columns:
        print("\nContest distribution:")
        print(
            dataframe["contest"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "tag" in dataframe.columns:
        print("\nTag distribution:")
        print(
            dataframe["tag"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "subtest" in dataframe.columns:
        print("\nTop 20 subtests:")
        print(
            dataframe["subtest"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

    if "year" in dataframe.columns:
        print("\nYear range:")
        print(
            f"  {dataframe['year'].min()} "
            f"to {dataframe['year'].max()}"
        )

    if "rating" in dataframe.columns:
        print("\nRating quantiles:")
        quantiles = dataframe["rating"].quantile(
            [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
        )
        print(quantiles.to_string())

    print("\nFirst problem:")
    if len(dataframe) > 0:
        row = dataframe.iloc[0]

        print(f"\nProblem:\n{row.get('problem', 'N/A')}")
        print(f"\nAnswer:\n{row.get('answer', 'N/A')}")
        print(f"\nRating:\n{row.get('rating', 'N/A')}")
        print(f"\nRating std:\n{row.get('rating_std', 'N/A')}")
        print(
            f"\nRating quantile:\n"
            f"{row.get('rating_quantile', 'N/A')}"
        )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print_section("DOWNLOADING E2H-AMC")

    dataset = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
        "E2H-AMC",
    )

    print(dataset)

    train_df = dataset["train"].to_pandas()
    eval_df = dataset["eval"].to_pandas()

    train_df.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    eval_df.to_parquet(
        EVAL_PATH,
        index=False,
    )

    print_section("RAW DATA SAVED")

    print(f"Train: {TRAIN_PATH}")
    print(f"Eval:  {EVAL_PATH}")

    inspect_split(
        "train",
        train_df,
    )

    inspect_split(
        "eval",
        eval_df,
    )

    print_section("TRAIN VS EVAL RATING COMPARISON")

    comparison = pd.DataFrame(
        {
            "train": train_df["rating"].describe(),
            "eval": eval_df["rating"].describe(),
        }
    )

    print(comparison.to_string())

    print_section("FINISHED")

    print(
        "Dataset downloaded and inspected successfully. "
        "No labels were modified."
    )


if __name__ == "__main__":
    main()