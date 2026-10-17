from pathlib import Path
import json
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "external"
    / "Visual_Item_Difficulty"
    / "data"
)

ITEMS_PATH = DATA_DIR / "items.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
CV_PATH = DATA_DIR / "cv_folds.json"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def shorten(
    value: object,
    max_length: int = 180,
) -> str:
    text = str(value).replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def main() -> None:
    print_section(
        "EEDI ITEM-DIFFICULTY DATASET AUDIT"
    )

    # -------------------------------------------------
    # FILE CHECK
    # -------------------------------------------------

    required_files = [
        ITEMS_PATH,
        TRAIN_PATH,
        TEST_PATH,
        CV_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    items = pd.read_csv(
        ITEMS_PATH
    )

    train = pd.read_csv(
        TRAIN_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    with open(
        CV_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        cv_data = json.load(file)

    # -------------------------------------------------
    # BASIC STRUCTURE
    # -------------------------------------------------

    print_section(
        "BASIC STRUCTURE"
    )

    print(
        f"items.csv rows: {len(items):,}"
    )

    print(
        f"train.csv rows: {len(train):,}"
    )

    print(
        f"test.csv rows:  {len(test):,}"
    )

    print()
    print("items.csv columns:")

    for column in items.columns:
        print(f"- {column}")

    if len(items) != 725:
        raise RuntimeError(
            "Expected 725 retained items."
        )

    if len(train) != 580:
        raise RuntimeError(
            "Expected 580 training items."
        )

    if len(test) != 145:
        raise RuntimeError(
            "Expected 145 test items."
        )

    print()
    print(
        "PASS: expected dataset sizes found."
    )

    # -------------------------------------------------
    # REQUIRED COLUMNS
    # -------------------------------------------------

    print_section(
        "REQUIRED COLUMN CHECK"
    )

    required_columns = {
        "QuestionId",
        "image_path",
        "difficulty",
        "SE_beta",
        "n_answers",
        "correct_rate",
        "error_rate",
        "split",
        "text",
        "visual_description",
    }

    missing_columns = (
        required_columns
        - set(items.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        "PASS: all required columns present."
    )

    # -------------------------------------------------
    # QUESTION-ID INTEGRITY
    # -------------------------------------------------

    print_section(
        "QUESTION-ID INTEGRITY"
    )

    duplicate_ids = int(
        items["QuestionId"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate QuestionId values: "
        f"{duplicate_ids}"
    )

    if duplicate_ids != 0:
        raise RuntimeError(
            "QuestionId values are not unique."
        )

    item_ids = set(
        items["QuestionId"]
        .astype(int)
    )

    train_ids = set(
        train["QuestionId"]
        .astype(int)
    )

    test_ids = set(
        test["QuestionId"]
        .astype(int)
    )

    overlap = (
        train_ids
        & test_ids
    )

    print(
        "Train/test ID overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train/test QuestionId leakage found."
        )

    if (
        train_ids
        | test_ids
    ) != item_ids:
        raise RuntimeError(
            "Train + test IDs do not exactly "
            "cover items.csv."
        )

    print(
        "PASS: train and test are "
        "disjoint and exhaustive."
    )

    # -------------------------------------------------
    # SPLIT CHECK
    # -------------------------------------------------

    print_section(
        "SPLIT COLUMN"
    )

    print(
        items["split"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    item_train_ids = set(
        items.loc[
            items["split"] == "train",
            "QuestionId",
        ].astype(int)
    )

    item_test_ids = set(
        items.loc[
            items["split"] == "test",
            "QuestionId",
        ].astype(int)
    )

    if item_train_ids != train_ids:
        raise RuntimeError(
            "items.csv train split does not "
            "match train.csv."
        )

    if item_test_ids != test_ids:
        raise RuntimeError(
            "items.csv test split does not "
            "match test.csv."
        )

    print()
    print(
        "PASS: split labels match "
        "train.csv and test.csv."
    )

    # -------------------------------------------------
    # TRAINING DATA QUALITY
    # -------------------------------------------------

    print_section(
        "TRAINING DATA QUALITY"
    )

    train_items = items.loc[
        items["split"] == "train"
    ].copy()

    required_train_columns = [
        "QuestionId",
        "difficulty",
        "SE_beta",
        "n_answers",
        "correct_rate",
        "text",
    ]

    missing_values = (
        train_items[
            required_train_columns
        ]
        .isna()
        .sum()
    )

    print(
        "Missing values in required "
        "training fields:"
    )

    print(
        missing_values.to_string()
    )

    if int(
        missing_values.sum()
    ) != 0:
        raise RuntimeError(
            "Missing required training data."
        )

    below_200 = int(
        (
            train_items["n_answers"]
            < 200
        ).sum()
    )

    print()
    print(
        "Training items with fewer than "
        f"200 responses: {below_200}"
    )

    if below_200 != 0:
        raise RuntimeError(
            "Found training items with fewer "
            "than 200 responses."
        )

    print()
    print(
        "PASS: required training values "
        "are present and response threshold "
        "is satisfied."
    )

    # -------------------------------------------------
    # TRAINING DIFFICULTY STATISTICS
    # -------------------------------------------------

    print_section(
        "TRAINING DIFFICULTY STATISTICS"
    )

    print(
        train_items[
            [
                "difficulty",
                "SE_beta",
                "n_answers",
                "correct_rate",
            ]
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Difficulty quantiles:"
    )

    print(
        train_items["difficulty"]
        .quantile(
            [
                0.00,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                1.00,
            ]
        )
        .to_string()
    )

    # -------------------------------------------------
    # PREPARE NORMALIZED REPRESENTATIONS
    # -------------------------------------------------

    items = items.copy()

    items["normalized_text"] = (
        items["text"]
        .fillna("")
        .map(normalize_text)
    )

    items[
        "normalized_visual_description"
    ] = (
        items[
            "visual_description"
        ]
        .fillna("")
        .map(normalize_text)
    )

    items["has_visual"] = (
        items[
            "visual_description"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    # -------------------------------------------------
    # EXACT TEXT DUPLICATE AUDIT
    # -------------------------------------------------

    print_section(
        "EXACT TEXT DUPLICATE AUDIT"
    )

    duplicate_mask = (
        items["normalized_text"]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        items.loc[
            duplicate_mask
        ]
        .copy()
    )

    print(
        "Rows participating in exact "
        "text duplicate groups: "
        f"{len(duplicate_rows)}"
    )

    if duplicate_rows.empty:
        print(
            "No exact text duplicate "
            "groups found."
        )

    else:
        duplicate_group_count = int(
            duplicate_rows[
                "normalized_text"
            ]
            .nunique()
        )

        print(
            "Exact text duplicate groups: "
            f"{duplicate_group_count}"
        )

        same_visual_groups = 0
        different_visual_groups = 0
        cross_split_groups = 0
        conflicting_difficulty_groups = 0

        for group_number, (
            normalized_text,
            group,
        ) in enumerate(
            duplicate_rows.groupby(
                "normalized_text",
                sort=False,
            ),
            start=1,
        ):
            visual_variants = int(
                group[
                    "normalized_visual_description"
                ]
                .nunique()
            )

            split_variants = int(
                group["split"]
                .nunique()
            )

            difficulty_variants = int(
                group["difficulty"]
                .nunique()
            )

            if visual_variants == 1:
                same_visual_groups += 1
            else:
                different_visual_groups += 1

            if split_variants > 1:
                cross_split_groups += 1

            if difficulty_variants > 1:
                conflicting_difficulty_groups += 1

            print()
            print(
                f"Duplicate group "
                f"{group_number}"
            )
            print("-" * 80)

            print(
                "Question text:"
            )

            print(
                shorten(
                    group.iloc[0]["text"],
                    max_length=350,
                )
            )

            print()

            display_rows = []

            for _, row in group.iterrows():
                display_rows.append(
                    {
                        "QuestionId":
                            int(
                                row[
                                    "QuestionId"
                                ]
                            ),
                        "split":
                            row["split"],
                        "difficulty":
                            round(
                                float(
                                    row[
                                        "difficulty"
                                    ]
                                ),
                                4,
                            ),
                        "has_visual":
                            bool(
                                row[
                                    "has_visual"
                                ]
                            ),
                        "image_path":
                            shorten(
                                row[
                                    "image_path"
                                ],
                                70,
                            ),
                        "visual_description":
                            shorten(
                                row[
                                    "visual_description"
                                ],
                                140,
                            ),
                    }
                )

            print(
                pd.DataFrame(
                    display_rows
                ).to_string(
                    index=False
                )
            )

            print(
                "Visual-description variants: "
                f"{visual_variants}"
            )

            print(
                "Split variants: "
                f"{split_variants}"
            )

            print(
                "Difficulty variants: "
                f"{difficulty_variants}"
            )

        print()
        print(
            "Duplicate group summary:"
        )

        print(
            "Same text + same visual "
            "description groups: "
            f"{same_visual_groups}"
        )

        print(
            "Same text + different visual "
            "description groups: "
            f"{different_visual_groups}"
        )

        print(
            "Duplicate-text groups spanning "
            "train and test: "
            f"{cross_split_groups}"
        )

        print(
            "Duplicate-text groups with "
            "different difficulty targets: "
            f"{conflicting_difficulty_groups}"
        )

    # -------------------------------------------------
    # TRAIN / TEST TEXT OVERLAP
    # -------------------------------------------------

    print_section(
        "TEXT-LEVEL TRAIN/TEST OVERLAP"
    )

    train_texts = set(
        items.loc[
            items["split"] == "train",
            "normalized_text",
        ]
    )

    test_texts = set(
        items.loc[
            items["split"] == "test",
            "normalized_text",
        ]
    )

    cross_split_text_overlap = (
        train_texts
        & test_texts
    )

    print(
        "Exact normalized texts appearing "
        "in both train and test: "
        f"{len(cross_split_text_overlap)}"
    )

    if cross_split_text_overlap:
        print(
            "WARNING: text-level overlap "
            "exists."
        )

        print(
            "Do not remove or relabel these "
            "items yet."
        )

        print(
            "We first need to determine "
            "whether they are distinct "
            "visual items."
        )

    else:
        print(
            "PASS: no exact text overlap "
            "between train and test."
        )

    # -------------------------------------------------
    # FULL TEXT + VISUAL REPRESENTATION OVERLAP
    # -------------------------------------------------

    print_section(
        "TEXT + VISUAL REPRESENTATION OVERLAP"
    )

    items[
        "combined_representation"
    ] = (
        items[
            "normalized_text"
        ]
        + " || "
        + items[
            "normalized_visual_description"
        ]
    )

    train_combined = set(
        items.loc[
            items["split"] == "train",
            "combined_representation",
        ]
    )

    test_combined = set(
        items.loc[
            items["split"] == "test",
            "combined_representation",
        ]
    )

    combined_overlap = (
        train_combined
        & test_combined
    )

    print(
        "Exact text+visual representations "
        "appearing in both train and test: "
        f"{len(combined_overlap)}"
    )

    if combined_overlap:
        print(
            "WARNING: possible full-item "
            "representation leakage exists."
        )
    else:
        print(
            "PASS: no exact text+visual "
            "representation overlap."
        )

    # -------------------------------------------------
    # VISUAL COMPONENT
    # -------------------------------------------------

    print_section(
        "VISUAL COMPONENT"
    )

    visual_count = int(
        items["has_visual"]
        .sum()
    )

    non_visual_count = (
        len(items)
        - visual_count
    )

    train_visual_count = int(
        items.loc[
            items["split"] == "train",
            "has_visual",
        ].sum()
    )

    test_visual_count = int(
        items.loc[
            items["split"] == "test",
            "has_visual",
        ].sum()
    )

    print(
        "Items with visual description: "
        f"{visual_count}"
    )

    print(
        "Items without visual description: "
        f"{non_visual_count}"
    )

    print(
        "Overall visual proportion: "
        f"{visual_count / len(items):.4f}"
    )

    print(
        "Training items with visual "
        f"description: "
        f"{train_visual_count}"
    )

    print(
        "Test items with visual "
        f"description: "
        f"{test_visual_count}"
    )

    # -------------------------------------------------
    # FROZEN CV FOLDS
    # -------------------------------------------------

    print_section(
        "FROZEN CV FOLD INTEGRITY"
    )

    print(
        "Declared n_train: "
        f"{cv_data.get('n_train')}"
    )

    print(
        "Declared n_test: "
        f"{cv_data.get('n_test')}"
    )

    print(
        "Fold seed: "
        f"{cv_data.get('fold_seed')}"
    )

    print(
        "Stratification: "
        f"{cv_data.get('stratify')}"
    )

    folds = cv_data[
        "folds"
    ]

    if len(folds) != 5:
        raise RuntimeError(
            "Expected exactly five CV folds."
        )

    all_validation_ids = []

    for fold in folds:
        fold_number = fold[
            "fold"
        ]

        validation_ids = [
            int(qid)
            for qid in fold[
                "val_qids"
            ]
        ]

        all_validation_ids.extend(
            validation_ids
        )

        print(
            f"Fold {fold_number}: "
            f"{len(validation_ids)} "
            "validation items"
        )

        if (
            set(validation_ids)
            & test_ids
        ):
            raise RuntimeError(
                f"Fold {fold_number} "
                "contains held-out "
                "test items."
            )

    validation_id_set = set(
        all_validation_ids
    )

    duplicate_cv_assignments = (
        len(all_validation_ids)
        - len(validation_id_set)
    )

    print()
    print(
        "Duplicate CV validation "
        "assignments: "
        f"{duplicate_cv_assignments}"
    )

    missing_from_cv = (
        train_ids
        - validation_id_set
    )

    unexpected_in_cv = (
        validation_id_set
        - train_ids
    )

    print(
        "Training IDs missing "
        "from CV: "
        f"{len(missing_from_cv)}"
    )

    print(
        "Unexpected IDs in CV: "
        f"{len(unexpected_in_cv)}"
    )

    if duplicate_cv_assignments != 0:
        raise RuntimeError(
            "A training item appears "
            "in multiple validation folds."
        )

    if (
        missing_from_cv
        or unexpected_in_cv
    ):
        raise RuntimeError(
            "Frozen CV folds do not "
            "exactly cover training data."
        )

    print()
    print(
        "PASS: frozen CV folds "
        "exactly cover training data "
        "and exclude held-out test data."
    )

    # -------------------------------------------------
    # TEST POLICY
    # -------------------------------------------------

    print_section(
        "HELD-OUT TEST POLICY"
    )

    print(
        "The 145 test items were used "
        "only for structural and leakage "
        "integrity checks."
    )

    print(
        "No test difficulty summary, "
        "model prediction, or model "
        "performance metric was inspected."
    )

    print(
        "All model development must use "
        "only the 580 training items and "
        "the provided frozen CV folds."
    )

    print_section(
        "AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()