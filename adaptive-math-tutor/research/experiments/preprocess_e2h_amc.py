from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "e2h_amc"
    / "e2h_amc_train.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "e2h_amc_train_clean.parquet"
)


def normalize_problem(text: str) -> str:
    """
    Normalize question text only for duplicate detection.

    The original problem text is preserved for modelling.
    """
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


def main() -> None:
    print()
    print("=" * 80)
    print("E2H-AMC PREPROCESSING")
    print("=" * 80)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    df = pd.read_parquet(
        RAW_PATH
    )

    print()
    print(f"Raw rows: {len(df):,}")

    required_columns = {
        "problem",
        "rating",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------
    # KEEP ONLY WHAT THE COMPLEXITY MODEL NEEDS
    # --------------------------------------------------

    clean = df[
        [
            "problem",
            "rating",
        ]
    ].copy()

    clean.insert(
        0,
        "source_row_id",
        df.index.astype(int),
    )

    # --------------------------------------------------
    # REMOVE MISSING / EMPTY REQUIRED VALUES
    # --------------------------------------------------

    before_missing = len(clean)

    clean = clean.dropna(
        subset=[
            "problem",
            "rating",
        ]
    ).copy()

    clean["problem"] = (
        clean["problem"]
        .astype(str)
        .str.strip()
    )

    clean = clean.loc[
        clean["problem"] != ""
    ].copy()

    removed_missing = (
        before_missing
        - len(clean)
    )

    print(
        "Removed missing/empty rows: "
        f"{removed_missing}"
    )

    # --------------------------------------------------
    # VALIDATE TARGET
    # --------------------------------------------------

    clean["rating"] = pd.to_numeric(
        clean["rating"],
        errors="raise",
    )

    if not clean["rating"].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise RuntimeError(
            "Found rating values outside [0, 1]."
        )

    # --------------------------------------------------
    # EXACT NORMALIZED DUPLICATE DETECTION
    # --------------------------------------------------

    clean["_normalized_problem"] = (
        clean["problem"]
        .map(normalize_problem)
    )

    grouped = clean.groupby(
        "_normalized_problem",
        sort=False,
    )

    rows_to_keep = []
    same_target_duplicate_groups = 0
    conflicting_duplicate_groups = 0
    conflicting_rows_removed = 0
    same_target_rows_removed = 0

    for _, group in grouped:
        if len(group) == 1:
            rows_to_keep.append(
                group.index[0]
            )
            continue

        unique_ratings = (
            group["rating"]
            .nunique()
        )

        if unique_ratings == 1:
            # Same question and same target:
            # keep one representative row.
            rows_to_keep.append(
                group.index[0]
            )

            same_target_duplicate_groups += 1

            same_target_rows_removed += (
                len(group) - 1
            )

        else:
            # Same question but conflicting targets:
            # exclude the entire group rather than
            # inventing or averaging a target.
            conflicting_duplicate_groups += 1

            conflicting_rows_removed += (
                len(group)
            )

    clean = clean.loc[
        rows_to_keep
    ].copy()

    clean = clean.drop(
        columns=[
            "_normalized_problem",
        ]
    )

    clean = clean.sort_values(
        "source_row_id"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # FINAL INTEGRITY CHECK
    # --------------------------------------------------

    if clean["problem"].isna().any():
        raise RuntimeError(
            "Missing problem values remain."
        )

    if clean["rating"].isna().any():
        raise RuntimeError(
            "Missing rating values remain."
        )

    normalized_final = (
        clean["problem"]
        .map(normalize_problem)
    )

    remaining_duplicates = int(
        normalized_final
        .duplicated()
        .sum()
    )

    if remaining_duplicates != 0:
        raise RuntimeError(
            "Exact normalized duplicates remain."
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("DUPLICATE HANDLING")
    print("=" * 80)

    print(
        "Same-target duplicate groups: "
        f"{same_target_duplicate_groups}"
    )

    print(
        "Rows removed from same-target groups: "
        f"{same_target_rows_removed}"
    )

    print(
        "Conflicting-target duplicate groups: "
        f"{conflicting_duplicate_groups}"
    )

    print(
        "Rows removed from conflicting groups: "
        f"{conflicting_rows_removed}"
    )

    print()
    print("=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print(
        f"Final rows: {len(clean):,}"
    )

    print()
    print(
        clean["rating"]
        .describe()
        .to_string()
    )

    print()
    print(
        "Columns used for modelling:"
    )

    for column in clean.columns:
        print(
            f"- {column}"
        )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "No contest/tag/difficulty metadata "
        "is included as model input."
    )

    print(
        "The official E2H-AMC evaluation "
        "split was not loaded or modified."
    )

    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()