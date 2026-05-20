from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "skill_builder_data.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

BASE_INTERACTIONS_PATH = (
    PROCESSED_DATA_DIR
    / "base_interactions.csv"
)

CONCEPT_INTERACTIONS_PATH = (
    PROCESSED_DATA_DIR
    / "concept_interactions.csv"
)


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """
    Load the original ASSISTments Skill Builder dataset.

    The raw dataset must never be modified or overwritten.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin1",
        low_memory=False
    )

    return df


# ---------------------------------------------------------
# Base Interaction Construction
# ---------------------------------------------------------

def build_base_interactions(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct one logical row per order_id.

    This dataset will support general student history.
    """

    base_columns = [
        "order_id",
        "user_id",
        "assignment_id",
        "assistment_id",
        "problem_id",
        "correct",
        "attempt_count",
        "ms_first_response",
        "hint_count",
        "hint_total",
        "overlap_time",
        "first_action",
        "bottom_hint",
        "tutor_mode",
        "answer_type",
        "sequence_id",
        "base_sequence_id",
        "position",
        "original",
        "student_class_id",
        "teacher_id",
        "school_id"
    ]

    missing_columns = [
        column
        for column in base_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required base-interaction columns: "
            f"{missing_columns}"
        )

    base_df = (
        df[base_columns]
        .drop_duplicates(
            subset=["order_id"],
            keep="first"
        )
        .copy()
    )

    base_df = base_df.reset_index(
        drop=True
    )

    return base_df


# ---------------------------------------------------------
# Concept Interaction Construction
# ---------------------------------------------------------

def build_concept_interactions(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct one logical row per order_id + skill_id.

    This dataset will support concept-specific history.
    """

    required_columns = [
        "order_id",
        "user_id",
        "assignment_id",
        "assistment_id",
        "problem_id",
        "skill_id",
        "skill_name",
        "correct",
        "attempt_count",
        "ms_first_response",
        "hint_count",
        "hint_total",
        "overlap_time",
        "first_action",
        "bottom_hint",
        "tutor_mode",
        "answer_type",
        "sequence_id",
        "base_sequence_id",
        "position",
        "original",
        "student_class_id",
        "teacher_id",
        "school_id",
        "opportunity"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required concept-interaction columns: "
            f"{missing_columns}"
        )

    known_skill_df = (
        df[df["skill_id"].notna()]
        .copy()
    )

    group_columns = [
        "order_id",
        "skill_id"
    ]

    concept_df = (
        known_skill_df
        .groupby(
            group_columns,
            sort=False,
            as_index=False
        )
        .agg(
            user_id=("user_id", "first"),
            assignment_id=("assignment_id", "first"),
            assistment_id=("assistment_id", "first"),
            problem_id=("problem_id", "first"),

            skill_name=("skill_name", "first"),

            correct=("correct", "first"),
            attempt_count=("attempt_count", "first"),
            ms_first_response=("ms_first_response", "first"),
            hint_count=("hint_count", "first"),
            hint_total=("hint_total", "first"),
            overlap_time=("overlap_time", "first"),
            first_action=("first_action", "first"),
            bottom_hint=("bottom_hint", "first"),

            tutor_mode=("tutor_mode", "first"),
            answer_type=("answer_type", "first"),

            sequence_id=("sequence_id", "first"),
            base_sequence_id=("base_sequence_id", "first"),
            position=("position", "first"),
            original=("original", "first"),

            student_class_id=("student_class_id", "first"),
            teacher_id=("teacher_id", "first"),
            school_id=("school_id", "first"),

            opportunity_start=("opportunity", "min"),
            opportunity_end=("opportunity", "max"),
            raw_row_count=("opportunity", "size")
        )
    )

    concept_df = concept_df.reset_index(
        drop=True
    )

    concept_df["skill_id"] = (
        concept_df["skill_id"]
        .astype("int64")
    )

    return concept_df


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_base_interactions(
    raw_df: pd.DataFrame,
    base_df: pd.DataFrame
) -> None:
    """
    Validate the processed base-interaction dataset.
    """

    expected_rows = 346860
    expected_students = 4217

    required_columns = [
        "order_id",
        "user_id",
        "assignment_id",
        "problem_id",
        "correct",
        "attempt_count"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in base_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Base interaction validation failed. "
            f"Missing columns: {missing_columns}"
        )

    if len(base_df) != expected_rows:
        raise ValueError(
            "Base interaction validation failed. "
            f"Expected {expected_rows} rows, "
            f"found {len(base_df)}."
        )

    duplicate_order_ids = (
        base_df["order_id"]
        .duplicated()
        .sum()
    )

    if duplicate_order_ids != 0:
        raise ValueError(
            "Base interaction validation failed. "
            f"Found {duplicate_order_ids} duplicate order_id values."
        )

    missing_order_ids = (
        base_df["order_id"]
        .isna()
        .sum()
    )

    if missing_order_ids != 0:
        raise ValueError(
            "Base interaction validation failed. "
            f"Found {missing_order_ids} missing order_id values."
        )

    student_count = (
        base_df["user_id"]
        .nunique()
    )

    if student_count != expected_students:
        raise ValueError(
            "Base interaction validation failed. "
            f"Expected {expected_students} students, "
            f"found {student_count}."
        )

    invalid_correct = (
        ~base_df["correct"].isin([0, 1])
    ).sum()

    if invalid_correct != 0:
        raise ValueError(
            "Base interaction validation failed. "
            f"Found {invalid_correct} invalid correctness values."
        )

    comparison_columns = [
        "order_id",
        "user_id",
        "assignment_id",
        "problem_id",
        "correct",
        "attempt_count",
        "ms_first_response",
        "hint_count",
        "hint_total"
    ]

    raw_reference = (
        raw_df[comparison_columns]
        .drop_duplicates(
            subset=["order_id"],
            keep="first"
        )
        .sort_values("order_id")
        .reset_index(drop=True)
    )

    processed_reference = (
        base_df[comparison_columns]
        .sort_values("order_id")
        .reset_index(drop=True)
    )

    if not raw_reference.equals(
        processed_reference
    ):
        raise ValueError(
            "Base interaction validation failed. "
            "Processed interaction values do not "
            "match the raw dataset."
        )

    raw_missing_skill_status = (
        raw_df
        .groupby("order_id")["skill_id"]
        .apply(
            lambda values: values.isna().all()
        )
    )

    raw_missing_skill_order_ids = set(
        raw_missing_skill_status[
            raw_missing_skill_status
        ].index
    )

    processed_base_order_ids = set(
        base_df["order_id"]
    )

    lost_missing_skill_interactions = (
        raw_missing_skill_order_ids
        - processed_base_order_ids
    )

    if len(lost_missing_skill_interactions) != 0:
        raise ValueError(
            "Base interaction validation failed. "
            f"{len(lost_missing_skill_interactions)} "
            "no-known-skill interactions were lost."
        )

    preserved_missing_skill_count = (
        len(
            raw_missing_skill_order_ids
            & processed_base_order_ids
        )
    )

    if preserved_missing_skill_count != 63755:
        raise ValueError(
            "Base interaction validation failed. "
            "Expected 63755 no-known-skill base interactions, "
            f"found {preserved_missing_skill_count}."
        )

    print("Base interaction validation passed.")
    print(f"Rows: {len(base_df)}")
    print(
        "Unique order_id:",
        base_df["order_id"].nunique()
    )
    print(
        "Students:",
        student_count
    )
    print(
        "Duplicate order_id:",
        duplicate_order_ids
    )
    print(
        "No-known-skill interactions preserved:",
        preserved_missing_skill_count
    )


def validate_concept_interactions(
    raw_df: pd.DataFrame,
    concept_df: pd.DataFrame
) -> None:
    """
    Validate the processed concept-interaction dataset.
    """

    expected_rows = 338001
    expected_students = 4163
    expected_multi_row_interactions = 3227
    expected_max_raw_row_count = 215

    required_columns = [
        "order_id",
        "user_id",
        "skill_id",
        "correct",
        "attempt_count",
        "opportunity_start",
        "opportunity_end",
        "raw_row_count"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in concept_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Missing columns: {missing_columns}"
        )

    if len(concept_df) != expected_rows:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Expected {expected_rows} rows, "
            f"found {len(concept_df)}."
        )

    duplicate_concept_interactions = (
        concept_df
        .duplicated(
            subset=["order_id", "skill_id"]
        )
        .sum()
    )

    if duplicate_concept_interactions != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            "Found "
            f"{duplicate_concept_interactions} duplicate "
            "order_id + skill_id combinations."
        )

    missing_order_ids = (
        concept_df["order_id"]
        .isna()
        .sum()
    )

    if missing_order_ids != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {missing_order_ids} missing order_id values."
        )

    missing_skill_ids = (
        concept_df["skill_id"]
        .isna()
        .sum()
    )

    if missing_skill_ids != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {missing_skill_ids} missing skill_id values."
        )

    student_count = (
        concept_df["user_id"]
        .nunique()
    )

    if student_count != expected_students:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Expected {expected_students} students, "
            f"found {student_count}."
        )

    invalid_ranges = (
        concept_df["opportunity_end"]
        < concept_df["opportunity_start"]
    ).sum()

    if invalid_ranges != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {invalid_ranges} invalid opportunity ranges."
        )

    invalid_raw_row_counts = (
        concept_df["raw_row_count"] < 1
    ).sum()

    if invalid_raw_row_counts != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {invalid_raw_row_counts} invalid raw_row_count values."
        )

    multi_row_interactions = (
        concept_df["raw_row_count"] > 1
    ).sum()

    if multi_row_interactions != expected_multi_row_interactions:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Expected {expected_multi_row_interactions} "
            "multi-row concept interactions, "
            f"found {multi_row_interactions}."
        )

    max_raw_row_count = (
        concept_df["raw_row_count"].max()
    )

    if max_raw_row_count != expected_max_raw_row_count:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Expected maximum raw_row_count "
            f"{expected_max_raw_row_count}, "
            f"found {max_raw_row_count}."
        )

    range_width = (
        concept_df["opportunity_end"]
        - concept_df["opportunity_start"]
        + 1
    )

    inconsistent_ranges = (
        range_width
        != concept_df["raw_row_count"]
    ).sum()

    if inconsistent_ranges != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {inconsistent_ranges} concept interactions "
            "where opportunity-range width does not match "
            "raw_row_count."
        )

    raw_known_skill = (
        raw_df[
            raw_df["skill_id"].notna()
        ]
        .copy()
    )

    raw_reference = (
        raw_known_skill
        .groupby(
            ["order_id", "skill_id"],
            sort=False,
            as_index=False
        )
        .agg(
            user_id=("user_id", "first"),
            assignment_id=("assignment_id", "first"),
            problem_id=("problem_id", "first"),
            correct=("correct", "first"),
            attempt_count=("attempt_count", "first"),
            hint_count=("hint_count", "first"),
            hint_total=("hint_total", "first"),
            ms_first_response=(
                "ms_first_response",
                "first"
            ),
            opportunity_start=(
                "opportunity",
                "min"
            ),
            opportunity_end=(
                "opportunity",
                "max"
            ),
            raw_row_count=(
                "opportunity",
                "size"
            )
        )
    )

    comparison_columns = [
        "order_id",
        "skill_id",
        "user_id",
        "assignment_id",
        "problem_id",
        "correct",
        "attempt_count",
        "hint_count",
        "hint_total",
        "ms_first_response",
        "opportunity_start",
        "opportunity_end",
        "raw_row_count"
    ]

    raw_reference = (
        raw_reference[
            comparison_columns
        ]
        .sort_values(
            ["order_id", "skill_id"]
        )
        .reset_index(drop=True)
    )

    raw_reference["skill_id"] = (
        raw_reference["skill_id"]
        .astype("int64")
    )

    processed_reference = (
        concept_df[
            comparison_columns
        ]
        .sort_values(
            ["order_id", "skill_id"]
        )
        .reset_index(drop=True)
    )

    if not raw_reference.equals(
        processed_reference
    ):
        raise ValueError(
            "Concept interaction validation failed. "
            "Processed concept values do not "
            "match the raw dataset."
        )

    raw_skill_map = (
        raw_df
        .dropna(subset=["skill_id"])
        .groupby("order_id")["skill_id"]
        .apply(
            lambda values: tuple(
                sorted(set(values))
            )
        )
    )

    processed_skill_map = (
        concept_df
        .groupby("order_id")["skill_id"]
        .apply(
            lambda values: tuple(
                sorted(set(values))
            )
        )
    )

    raw_only_order_ids = (
        raw_skill_map.index.difference(
            processed_skill_map.index
        )
    )

    processed_only_order_ids = (
        processed_skill_map.index.difference(
            raw_skill_map.index
        )
    )

    if len(raw_only_order_ids) != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"{len(raw_only_order_ids)} known-skill "
            "order_id values from raw data are missing "
            "from the processed concept dataset."
        )

    if len(processed_only_order_ids) != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"{len(processed_only_order_ids)} processed "
            "order_id values do not exist as known-skill "
            "interactions in raw data."
        )

    skill_set_mismatches = (
        raw_skill_map
        != processed_skill_map
    ).sum()

    if skill_set_mismatches != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"Found {skill_set_mismatches} order_id values "
            "with mismatched skill sets."
        )

    multi_skill_order_ids = (
        concept_df
        .groupby("order_id")["skill_id"]
        .nunique()
    )

    multi_skill_count = (
        multi_skill_order_ids > 1
    ).sum()

    max_skills_per_order = (
        multi_skill_order_ids.max()
    )

    if multi_skill_count != 47037:
        raise ValueError(
            "Concept interaction validation failed. "
            "Expected 47037 multi-skill order_id values, "
            f"found {multi_skill_count}."
        )

    if max_skills_per_order != 4:
        raise ValueError(
            "Concept interaction validation failed. "
            "Expected maximum 4 skills per order_id, "
            f"found {max_skills_per_order}."
        )

    raw_missing_skill_status = (
        raw_df
        .groupby("order_id")["skill_id"]
        .apply(
            lambda values: values.isna().all()
        )
    )

    raw_missing_skill_order_ids = set(
        raw_missing_skill_status[
            raw_missing_skill_status
        ].index
    )

    processed_concept_order_ids = set(
        concept_df["order_id"]
    )

    missing_skill_in_concept = (
        raw_missing_skill_order_ids
        & processed_concept_order_ids
    )

    if len(missing_skill_in_concept) != 0:
        raise ValueError(
            "Concept interaction validation failed. "
            f"{len(missing_skill_in_concept)} interactions "
            "with no known raw skill were incorrectly "
            "included in the concept dataset."
        )

    print("Concept interaction validation passed.")
    print(f"Rows: {len(concept_df)}")
    print(
        "Unique order_id + skill_id:",
        concept_df[
            ["order_id", "skill_id"]
        ].drop_duplicates().shape[0]
    )
    print(
        "Students:",
        student_count
    )
    print(
        "Duplicate order_id + skill_id:",
        duplicate_concept_interactions
    )
    print(
        "Multi-row concept interactions:",
        multi_row_interactions
    )
    print(
        "Maximum raw_row_count:",
        max_raw_row_count
    )
    print(
        "Invalid opportunity ranges:",
        invalid_ranges
    )
    print(
        "Range-width mismatches:",
        inconsistent_ranges
    )
    print(
        "Skill-set mismatches:",
        skill_set_mismatches
    )
    print(
        "Multi-skill order_ids:",
        multi_skill_count
    )
    print(
        "Maximum skills per order_id:",
        max_skills_per_order
    )
    print(
        "No-skill interactions incorrectly in concept:",
        len(missing_skill_in_concept)
    )


# ---------------------------------------------------------
# Saving
# ---------------------------------------------------------

def save_processed_data(
    base_df: pd.DataFrame,
    concept_df: pd.DataFrame
) -> None:
    """
    Save processed datasets under data/processed/.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    base_df.to_csv(
        BASE_INTERACTIONS_PATH,
        index=False
    )

    concept_df.to_csv(
        CONCEPT_INTERACTIONS_PATH,
        index=False
    )

    print(
        "Saved base interactions to:",
        BASE_INTERACTIONS_PATH
    )

    print(
        "Saved concept interactions to:",
        CONCEPT_INTERACTIONS_PATH
    )


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def run_preprocessing() -> None:
    """
    Run the complete preprocessing pipeline.
    """

    print("Loading raw dataset...")
    raw_df = load_raw_data()

    print(
        f"Raw dataset loaded: "
        f"{len(raw_df)} rows, "
        f"{len(raw_df.columns)} columns"
    )

    print("\nBuilding base interactions...")
    base_df = build_base_interactions(
        raw_df
    )

    print(
        f"Base interactions built: "
        f"{len(base_df)} rows"
    )

    print("\nBuilding concept interactions...")
    concept_df = build_concept_interactions(
        raw_df
    )

    print(
        f"Concept interactions built: "
        f"{len(concept_df)} rows"
    )

    print("\nValidating base interactions...")
    validate_base_interactions(
        raw_df,
        base_df
    )

    print("\nValidating concept interactions...")
    validate_concept_interactions(
        raw_df,
        concept_df
    )

    print("\nAll validations passed.")

    print("\nSaving processed datasets...")
    save_processed_data(
        base_df,
        concept_df
    )

    print(
        "\nPreprocessing completed successfully."
    )


if __name__ == "__main__":
    run_preprocessing()
