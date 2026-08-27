"""Audit GSM8K dataset against the 111 Canonical Skills Ontology."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "data" / "topic_extraction" / "raw" / "gsm8k" / "train.csv"
TEST_CSV = PROJECT_ROOT / "data" / "topic_extraction" / "raw" / "gsm8k" / "test.csv"
ONTOLOGY_JSON = PROJECT_ROOT / "data" / "processed" / "canonical_skill_ontology.json"
OUTPUT_AUDIT_CSV = (
    PROJECT_ROOT / "data" / "topic_extraction" / "processed" / "gsm8k_audit.csv"
)


def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, whitespace collapse, remove special chars)."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def build_alias_patterns(
    ontology_data: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[re.Pattern]]]:
    """Compile regex word-boundary patterns for each skill's display name and aliases."""
    skill_patterns = []

    # Stop words / generic terms that shouldn't match standalone
    ignored_aliases = {
        "table",
        "rate",
        "range",
        "mode",
        "mean",
        "median",
        "fraction of",
        "percent of",
    }

    for skill in ontology_data:
        patterns = []
        candidates = [skill["display_name"]] + skill.get("aliases", [])

        for candidate in candidates:
            # Skip short generic numbers or identifiers like "193"
            if candidate.isdigit() or len(candidate.strip()) < 3:
                continue

            norm = normalize_text(candidate)
            if not norm or norm in ignored_aliases:
                continue

            # Build regex with word boundaries
            escaped = re.escape(norm)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            patterns.append(pattern)

        skill_patterns.append((skill, patterns))

    return skill_patterns


def audit_dataset():
    # 1. Load ontology
    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    skill_patterns = build_alias_patterns(ontology)

    # 2. Load GSM8K splits
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    total_questions = len(combined_df)
    unique_questions = combined_df["question"].nunique()

    audit_rows = []
    skill_match_counts: dict[str, int] = {
        s["canonical_name"]: 0 for s in ontology
    }

    confident_count = 0
    ambiguous_count = 0
    unmatched_count = 0

    for idx, row in combined_df.iterrows():
        q_text = str(row["question"]).strip()
        norm_q = normalize_text(q_text)

        matched_skills = []
        for skill, patterns in skill_patterns:
            for pat in patterns:
                if pat.search(norm_q):
                    matched_skills.append(skill)
                    break  # One match per skill is enough

        match_count = len(matched_skills)

        if match_count == 1:
            match_type = "SINGLE_MATCH"
            matched_skill = matched_skills[0]
            matched_code = matched_skill["skill_code"]
            matched_name = matched_skill["display_name"]
            skill_match_counts[matched_skill["canonical_name"]] += 1
            confident_count += 1
        elif match_count > 1:
            match_type = "MULTI_MATCH"
            matched_code = ";".join(s["skill_code"] for s in matched_skills)
            matched_name = ";".join(s["display_name"] for s in matched_skills)
            ambiguous_count += 1
        else:
            match_type = "NO_MATCH"
            matched_code = None
            matched_name = None
            unmatched_count += 1

        audit_rows.append(
            {
                "text": q_text,
                "matched_skill_code": matched_code,
                "matched_skill_name": matched_name,
                "match_type": match_type,
                "match_count": match_count,
            }
        )

    # 3. Save audit output
    OUTPUT_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_AUDIT_CSV, index=False, encoding="utf-8")

    # 4. Compute coverage statistics
    skills_covered = sum(1 for count in skill_match_counts.values() if count > 0)
    skills_zero = sum(1 for count in skill_match_counts.values() if count == 0)

    top_matched = sorted(
        [(k, v) for k, v in skill_match_counts.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    print("=" * 60)
    print("GSM8K Ontology Audit Results")
    print("=" * 60)
    print(f"Total GSM8K questions:         {total_questions}")
    print(f"Unique questions:              {unique_questions}")
    print()
    print(f"Confident single-skill matches:{confident_count}")
    print(f"Ambiguous multi-skill matches: {ambiguous_count}")
    print(f"Unmatched:                     {unmatched_count}")
    print()
    print(f"Canonical skills covered:      {skills_covered} / {len(ontology)}")
    print(f"Skills with 0 matches:         {skills_zero}")
    print("Top matched skills:")
    for name, cnt in top_matched:
        print(f"  - {name}: {cnt} questions")
    print("=" * 60)
    print(f"Saved audit log to: {OUTPUT_AUDIT_CSV}")


if __name__ == "__main__":
    audit_dataset()
