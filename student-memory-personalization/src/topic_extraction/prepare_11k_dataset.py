"""Prepare, clean, and stratify the 11,000-question dataset with math symbol preservation and OOD negative samples."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_CSV_PATH = PROJECT_ROOT / "student_skill_questions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "topic_extraction" / "splits_11k"
ONTOLOGY_JSON = PROJECT_ROOT / "data" / "processed" / "canonical_skill_ontology.json"

# Curated Out-of-Domain (OOD) / "I Don't Know" queries across non-math topics & greetings
OOD_QUESTIONS = [
    # Greetings & chit-chat
    "Hello there, how are you doing today?",
    "Hi, who are you?",
    "Good morning tutor",
    "What is your name and what do you do?",
    "Tell me a funny joke or story",
    "How was your day?",
    "Can you write a poem about the autumn leaves?",
    "What is the meaning of life?",
    "Are you a robot or a human?",
    "I am feeling bored right now",
    
    # Science & Biology
    "What is the process of photosynthesis in green plants?",
    "How do cells divide in mitosis and meiosis?",
    "Explain Newton's third law of motion in physics",
    "What is the chemical formula for water and table salt?",
    "How does the human digestive system work?",
    "What is the difference between DNA and RNA?",
    
    # History & Geography
    "What were the main causes of World War 1?",
    "Who was the first president of the United States?",
    "What is the capital city of Australia?",
    "Which country has the longest coastline in the world?",
    "Tell me about the ancient Egyptian pyramids",
    "When did the French Revolution start?",
    
    # Literature & Language
    "What is a metaphor versus a simile?",
    "Can you check my essay for grammar mistakes?",
    "Who wrote Romeo and Juliet and Hamlet?",
    "What is the theme of The Great Gatsby?",
    "How do I write a persuasive introduction paragraph?",
    
    # Computer Science & Coding
    "How do I create a React web application with Vite?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain binary search tree traversal in Python",
    "How does TCP/IP network protocol work?",
    
    # Vague / Non-specific inputs
    "I don't understand anything",
    "Help me please",
    "Can you tell me the answer right now?",
    "What should I do next?",
    "I have no idea what is going on",
    "Why did I get problem 3 wrong?",
    "Give me some hints",
    "I am confused about this question",
    "Is this right or wrong?",
    "What is the next step?",
]


def clean_math_text(text: str) -> str:
    """
    Clean text while strictly preserving math tokens and operators:
    - Lowercases text
    - Replaces corrupted / unicode characters (e.g. ) cleanly
    - Normalizes spacing around operators (+, -, *, /, =, ^, %, |, <, >)
    - Preserves variables and algebraic terms
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Replace corrupted character glyphs from raw CSV encoding
    text = text.replace("", "-")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("²", "^2").replace("³", "^3")
    text = text.replace("√", "sqrt")

    # Lowercase
    text = text.lower().strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def prepare_and_split_dataset(
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Load, clean, and split 11,000 dataset into stratified train, val, test sets."""
    print(f"Loading raw dataset from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"Raw shape: {df.shape} ({df['skill_name'].nunique()} skills, {len(df)} total questions)")

    # Clean text column
    df["clean_question"] = df["question"].apply(clean_math_text)

    # Map skill_name to ontology canonical details if available
    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    # Build lookup map from display_name to canonical metadata
    name_to_canon = {}
    for item in ontology:
        name_to_canon[item["display_name"].lower().strip()] = item
        name_to_canon[item["canonical_name"].lower().strip()] = item

    # Add canonical metadata columns
    def get_canonical_name(skill_name: str) -> str:
        s_clean = skill_name.lower().strip()
        if s_clean in name_to_canon:
            return name_to_canon[s_clean]["canonical_name"]
        # Handle slight wording variations
        for k, v in name_to_canon.items():
            if s_clean.replace("polynomial", "polyomial") == k:
                return v["canonical_name"]
            if s_clean.replace("proportionally", "prportionally") == k:
                return v["canonical_name"]
        return f"math :: general :: {s_clean}"

    df["canonical_name"] = df["skill_name"].apply(get_canonical_name)

    # Perform stratified split (70% train, 15% validation, 15% test per topic)
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for skill, group in df.groupby("skill_name"):
        # group has 100 rows per skill
        train_group, temp_group = train_test_split(
            group,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True,
        )
        val_group, test_group = train_test_split(
            temp_group,
            test_size=0.50,
            random_state=random_state,
            shuffle=True,
        )
        train_dfs.append(train_group)
        val_dfs.append(val_group)
        test_dfs.append(test_group)

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Add OOD set to a separate validation and test file
    ood_df = pd.DataFrame({
        "skill_name": ["__OUT_OF_DOMAIN__"] * len(OOD_QUESTIONS),
        "question": OOD_QUESTIONS,
        "clean_question": [clean_math_text(q) for q in OOD_QUESTIONS],
        "canonical_name": ["__OUT_OF_DOMAIN__"] * len(OOD_QUESTIONS),
    })

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"
    ood_path = OUTPUT_DIR / "ood_negative.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    ood_df.to_csv(ood_path, index=False)

    print("\n" + "=" * 60)
    print("11K DATASET STRATIFIED SPLITS COMPLETE")
    print("=" * 60)
    print(f"Train Set:      {train_df.shape[0]} questions ({train_df['skill_name'].nunique()} skills, {len(train_df)//110} per skill)")
    print(f"Validation Set: {val_df.shape[0]} questions ({val_df['skill_name'].nunique()} skills, {len(val_df)//110} per skill)")
    print(f"Test Set:       {test_df.shape[0]} questions ({test_df['skill_name'].nunique()} skills, {len(test_df)//110} per skill)")
    print(f"OOD Negatives:  {ood_df.shape[0]} questions (for 'I don't know' calibration)")
    print("=" * 60)

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "ood": ood_df,
    }


if __name__ == "__main__":
    prepare_and_split_dataset()
