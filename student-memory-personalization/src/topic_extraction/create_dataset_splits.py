"""Create leakage-safe Train, Validation, and Test dataset splits for Topic Extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import normalize_token


INPUT_DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "topic_extraction"
    / "processed"
    / "topic_extraction_dataset.csv"
)
SPLITS_DIR = PROJECT_ROOT / "data" / "topic_extraction" / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"


def create_splits():
    df = pd.read_csv(INPUT_DATASET_CSV)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    # Partition template groups per canonical skill
    grouped_by_skill = df.groupby("canonical_name")

    for skill_name, skill_df in grouped_by_skill:
        unique_groups = sorted(skill_df["template_group"].unique())
        n_groups = len(unique_groups)

        # Deterministic sort using sha256 of skill_name + group
        def group_hash(grp: str) -> int:
            h = hashlib.sha256(f"{skill_name}::{grp}".encode("utf-8")).hexdigest()
            return int(h[:8], 16)

        sorted_groups = sorted(unique_groups, key=group_hash)

        # Allocate ~15% to test, ~15% to val, ~70% to train (minimum 1 group each)
        n_test = max(1, round(n_groups * 0.15))
        n_val = max(1, round(n_groups * 0.15))
        
        test_groups = set(sorted_groups[:n_test])
        val_groups = set(sorted_groups[n_test : n_test + n_val])
        train_groups = set(sorted_groups[n_test + n_val :])

        for _, row in skill_df.iterrows():
            grp = row["template_group"]
            rec = row.to_dict()
            if grp in test_groups:
                test_rows.append(rec)
            elif grp in val_groups:
                val_rows.append(rec)
            else:
                train_rows.append(rec)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # Save to disk
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    val_df.to_csv(VAL_CSV, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

    # Overlap and integrity assertions
    train_texts = set(train_df["text"].map(normalize_token))
    val_texts = set(val_df["text"].map(normalize_token))
    test_texts = set(test_df["text"].map(normalize_token))

    text_overlap_train_val = len(train_texts.intersection(val_texts))
    text_overlap_train_test = len(train_texts.intersection(test_texts))
    text_overlap_val_test = len(val_texts.intersection(test_texts))
    total_text_overlap = text_overlap_train_val + text_overlap_train_test + text_overlap_val_test

    # Template group overlap per skill
    def get_skill_template_pairs(df_split: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(df_split["canonical_name"], df_split["template_group"]))

    train_st = get_skill_template_pairs(train_df)
    val_st = get_skill_template_pairs(val_df)
    test_st = get_skill_template_pairs(test_df)

    group_overlap = (
        len(train_st.intersection(val_st))
        + len(train_st.intersection(test_st))
        + len(val_st.intersection(test_st))
    )

    train_skills = train_df["canonical_name"].nunique()
    val_skills = val_df["canonical_name"].nunique()
    test_skills = test_df["canonical_name"].nunique()

    min_train_per_skill = int(train_df.groupby("canonical_name").size().min())
    min_val_per_skill = int(val_df.groupby("canonical_name").size().min())
    min_test_per_skill = int(test_df.groupby("canonical_name").size().min())

    print("=" * 60)
    print("Dataset Split Summary (Leakage-Safe Partitioning)")
    print("=" * 60)
    print(f"Train rows:                    {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation rows:               {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test rows:                     {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Total rows:                    {len(df)}")
    print()
    print(f"Skills in Train:               {train_skills} / 111")
    print(f"Skills in Validation:          {val_skills} / 111")
    print(f"Skills in Test:                {test_skills} / 111")
    print()
    print(f"Min examples/skill (Train):    {min_train_per_skill}")
    print(f"Min examples/skill (Val):      {min_val_per_skill}")
    print(f"Min examples/skill (Test):     {min_test_per_skill}")
    print()
    print(f"Normalized text overlap:       {total_text_overlap}")
    print(f"Template-group overlap/skill:  {group_overlap}")
    print("=" * 60)
    print(f"Saved Train to:      {TRAIN_CSV}")
    print(f"Saved Validation to: {VAL_CSV}")
    print(f"Saved Test to:       {TEST_CSV}")


if __name__ == "__main__":
    create_splits()
