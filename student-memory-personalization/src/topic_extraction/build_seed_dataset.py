"""Build a high-confidence, controlled seed dataset from the 111 Canonical Skills Ontology."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    generate_skill_id,
    normalize_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "topic_extraction"
    / "processed"
    / "ontology_seed_dataset.csv"
)


TEMPLATES: list[tuple[str, str]] = [
    ("I need help with {term}", "help_request"),
    ("Can you explain {term}?", "explanation_request"),
    ("How do I solve problems about {term}?", "problem_solving"),
    ("I don't understand {term}", "misunderstanding_statement"),
    ("Show me how {term} works", "instruction_request"),
    ("I need to practice {term}", "practice_request"),
    ("Can you help me learn {term}?", "learning_request"),
    ("What does {term} mean?", "definition_request"),
    ("Give me an example of {term}", "example_request"),
    ("How does {term} work in math?", "concept_question"),
    ("I am having trouble with {term}", "struggle_statement"),
    ("Can we review {term} together?", "review_request"),
    ("Explain the concept of {term}", "explanation_request"),
    ("What is the rule for {term}?", "rule_question"),
    ("Help me understand {term}", "help_request"),
    ("Teach me about {term}", "learning_request"),
    ("How do you calculate {term}?", "calculation_question"),
]


def build_seed_dataset():
    # 1. Load canonical ontology
    with open(DEFAULT_ONTOLOGY_JSON_PATH, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    raw_records: list[dict[str, Any]] = []

    for skill in ontology:
        canonical_name = skill["canonical_name"]
        skill_id = str(generate_skill_id(canonical_name))
        skill_code = skill["skill_code"]
        display_name = skill["display_name"]
        category = skill.get("category", "Mathematics")

        # Collect distinct surface terms for this skill
        surface_terms = set()
        surface_terms.add(display_name)

        for alias in skill.get("aliases", []):
            # Skip code identifiers (e.g. "skill_193", "assistments_193") in natural phrasing templates
            if (
                alias.lower().startswith("skill_")
                or alias.lower().startswith("assistments")
                or alias.isdigit()
            ):
                continue
            # Skip delimiter syntax like "Algebra / Linear Equations" in simple templates
            if "::" in alias or "/" in alias:
                continue
            surface_terms.add(alias)

        # Generate examples from templates
        for term in sorted(surface_terms):
            for tpl, group in TEMPLATES:
                text = tpl.format(term=term)
                raw_records.append(
                    {
                        "text": text,
                        "skill_id": skill_id,
                        "skill_code": skill_code,
                        "skill_name": display_name,
                        "canonical_name": canonical_name,
                        "category": category,
                        "source": "ontology_template",
                        "template_group": group,
                    }
                )

            # Also add direct mentions (e.g. "Linear Equations", "PEMDAS")
            raw_records.append(
                {
                    "text": term,
                    "skill_id": skill_id,
                    "skill_code": skill_code,
                    "skill_name": display_name,
                    "canonical_name": canonical_name,
                    "category": category,
                    "source": "ontology_template",
                    "template_group": "direct_mention",
                }
            )

    # 2. De-duplicate and validate single-label consistency
    seen_texts: dict[str, dict[str, Any]] = {}
    conflicting_texts: set[str] = set()
    duplicate_count = 0

    for rec in raw_records:
        norm = normalize_token(rec["text"])
        if norm in seen_texts:
            existing = seen_texts[norm]
            if existing["skill_id"] != rec["skill_id"]:
                conflicting_texts.add(norm)
            else:
                duplicate_count += 1
        else:
            seen_texts[norm] = rec

    # 3. Filter out any conflicting texts
    clean_records = [
        rec
        for norm, rec in seen_texts.items()
        if norm not in conflicting_texts
    ]

    df = pd.DataFrame(clean_records)

    # 4. Save dataset
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    # 5. Compute statistics
    skills_represented = df["canonical_name"].nunique()
    examples_per_skill = df.groupby("canonical_name").size()
    min_examples = int(examples_per_skill.min())
    max_examples = int(examples_per_skill.max())
    total_examples = len(df)

    print("=" * 60)
    print("Ontology Seed Dataset Generation Summary")
    print("=" * 60)
    print(f"Total examples:                {total_examples}")
    print(f"Skills represented:            {skills_represented} / {len(ontology)}")
    print(f"Minimum examples per skill:    {min_examples}")
    print(f"Maximum examples per skill:    {max_examples}")
    print(f"Duplicate normalized texts:    {duplicate_count}")
    print(f"Conflicting labels:            {len(conflicting_texts)}")
    print("=" * 60)
    print(f"Saved dataset to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    build_seed_dataset()
