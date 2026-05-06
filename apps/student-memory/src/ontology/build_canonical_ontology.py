"""Builder script to construct the canonical skill ontology dataset from ASSISTments raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "skill_builder_data.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "canonical_skill_ontology.json"


SKILL_CATEGORY_MAP: dict[int, str] = {
    # Statistics & Probability
    1: "Statistics & Probability",      # Box and Whisker
    2: "Statistics & Probability",      # Circle Graph
    4: "Statistics & Probability",      # Histogram as Table or Graph
    5: "Number Sense & Operations",     # Number Line
    8: "Statistics & Probability",      # Scatter Plot
    9: "Statistics & Probability",      # Stem and Leaf Plot
    10: "Statistics & Probability",     # Table
    11: "Statistics & Probability",     # Venn Diagram
    12: "Statistics & Probability",     # Mean
    13: "Statistics & Probability",     # Median
    14: "Statistics & Probability",     # Mode
    15: "Statistics & Probability",     # Range
    16: "Statistics & Probability",     # Counting Methods
    17: "Statistics & Probability",     # Probability of Two Distinct Events
    18: "Statistics & Probability",     # Probability of a Single Event
    110: "Statistics & Probability",    # D.4.8-understanding-concept-of-probabilities
    365: "Statistics & Probability",    # Interpreting Coordinate Graphs

    # Geometry & Measurement
    21: "Geometry & Measurement",       # Interior Angles Figures with More than 3 Sides
    22: "Geometry & Measurement",       # Interior Angles Triangle
    24: "Geometry & Measurement",       # Congruence
    25: "Geometry & Measurement",       # Complementary and Supplementary Angles
    26: "Geometry & Measurement",       # Angles on Parallel Lines Cut by a Transversal
    27: "Geometry & Measurement",       # Pythagorean Theorem
    32: "Geometry & Measurement",       # Nets of 3D Figures
    34: "Geometry & Measurement",       # Unit Conversion Within a System
    35: "Geometry & Measurement",       # Effect of Changing Dimensions of a Shape Prportionally
    39: "Geometry & Measurement",       # Area Circle
    40: "Geometry & Measurement",       # Circumference
    42: "Geometry & Measurement",       # Perimeter of a Polygon
    43: "Geometry & Measurement",       # Reading a Ruler or Scale
    46: "Geometry & Measurement",       # Calculations with Similar Figures
    290: "Geometry & Measurement",      # Reflection
    292: "Geometry & Measurement",      # Rotations
    293: "Geometry & Measurement",      # Translations
    294: "Geometry & Measurement",      # Area Irregular Figure
    295: "Geometry & Measurement",      # Area Parallelogram
    296: "Geometry & Measurement",      # Area Rectangle
    297: "Geometry & Measurement",      # Area Trapezoid
    298: "Geometry & Measurement",      # Area Triangle
    299: "Geometry & Measurement",      # Surface Area Cylinder
    301: "Geometry & Measurement",      # Surface Area Rectangular Prism
    303: "Geometry & Measurement",      # Volume Cylinder
    307: "Geometry & Measurement",      # Volume Rectangular Prism
    308: "Geometry & Measurement",      # Volume Sphere
    314: "Geometry & Measurement",      # Angles - Obtuse, Acute, and Right
    343: "Geometry & Measurement",      # Midpoint

    # Number Sense & Arithmetic Operations
    47: "Proportional Reasoning & Percents", # Conversion of Fraction Decimals Percents
    48: "Number Sense & Operations",     # Equivalent Fractions
    49: "Number Sense & Operations",     # Ordering Positive Decimals
    50: "Number Sense & Operations",     # Ordering Fractions
    51: "Number Sense & Operations",     # Ordering Integers
    53: "Number Sense & Operations",     # Ordering Real Numbers
    54: "Number Sense & Operations",     # Rounding
    58: "Number Sense & Operations",     # Addition Whole Numbers
    61: "Number Sense & Operations",     # Division Fractions
    63: "Number Sense & Operations",     # Estimation
    64: "Number Sense & Operations",     # Fraction Of
    65: "Number Sense & Operations",     # Least Common Multiple
    67: "Number Sense & Operations",     # Multiplication Fractions
    69: "Number Sense & Operations",     # Multiplication Whole Numbers
    74: "Number Sense & Operations",     # Subtraction Whole Numbers
    75: "Number Sense & Operations",     # Square Root
    82: "Number Sense & Operations",     # Scientific Notation
    83: "Number Sense & Operations",     # Divisibility Rules
    84: "Number Sense & Operations",     # Prime Number
    85: "Number Sense & Operations",     # Absolute Value (Standard / Introductory)
    86: "Number Sense & Operations",     # Exponents
    163: "Number Sense & Operations",    # Absolute Value (Advanced / High School)
    276: "Number Sense & Operations",    # Multiplication and Division Positive Decimals
    277: "Number Sense & Operations",    # Addition and Subtraction Integers
    278: "Number Sense & Operations",    # Addition and Subtraction Positive Decimals
    279: "Number Sense & Operations",    # Multiplication and Division Integers
    280: "Number Sense & Operations",    # Addition and Subtraction Fractions
    309: "Number Sense & Operations",    # Order of Operations +,-,/,* () positive reals
    310: "Number Sense & Operations",    # Order of Operations All
    317: "Number Sense & Operations",    # Greatest Common Factor
    321: "Number Sense & Operations",    # Computation with Real Numbers

    # Proportional Reasoning & Percents
    70: "Proportional Reasoning & Percents", # Percent Of
    77: "Proportional Reasoning & Percents", # Finding Percents
    79: "Proportional Reasoning & Percents", # Proportion
    80: "Proportional Reasoning & Percents", # Scale Factor
    81: "Proportional Reasoning & Percents", # Unit Rate
    203: "Proportional Reasoning & Percents", # Percent Discount
    204: "Proportional Reasoning & Percents", # Percents
    217: "Proportional Reasoning & Percents", # Rate

    # Algebra & Functions
    92: "Algebra & Functions",           # Pattern Finding
    165: "Algebra & Functions",          # Algebraic Simplification
    166: "Algebra & Functions",          # Algebraic Solving
    173: "Algebra & Functions",          # Choose an Equation from Given Information
    190: "Algebra & Functions",          # Intercept
    193: "Algebra & Functions",          # Linear Equations
    221: "Algebra & Functions",          # Slope
    311: "Algebra & Functions",          # Equation Solving Two or Fewer Steps
    312: "Algebra & Functions",          # Equation Solving More Than Two Steps
    322: "Algebra & Functions",          # Write Linear Equation from Ordered Pairs
    323: "Algebra & Functions",          # Write Linear Equation from Situation
    324: "Algebra & Functions",          # Recognize Linear Pattern
    325: "Algebra & Functions",          # Write Linear Equation from Graph
    331: "Algebra & Functions",          # Finding Slope From Situation
    333: "Algebra & Functions",          # Finding Slope From Equation
    334: "Algebra & Functions",          # Finding Slope from Ordered Pairs
    340: "Algebra & Functions",          # Distributive Property
    346: "Algebra & Functions",          # Polynomial Factors
    348: "Algebra & Functions",          # Recognize Quadratic Pattern
    350: "Algebra & Functions",          # Solving Systems of Linear Equations
    356: "Algebra & Functions",          # Quadratic Formula to Solve Quadratic Equation
    362: "Algebra & Functions",          # Parts of a Polyomial, Terms, Coefficient, Monomial, Exponent, Variable
    368: "Algebra & Functions",          # Solving for a variable
    371: "Algebra & Functions",          # Simplifying Expressions positive exponents
    375: "Algebra & Functions",          # Solving Inequalities
    378: "Algebra & Functions",          # Solving Systems of Linear Equations by Graphing
}


def normalize_token(text: str) -> str:
    """Normalize text to lowercase whitespace-collapsed string."""
    return " ".join(text.strip().split()).lower()


def build_ontology() -> list[dict[str, Any]]:
    """Load raw ASSISTments data and generate the canonical ontology dataset."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="latin1", low_memory=False)
    skill_df = df[["skill_id", "skill_name"]].dropna().copy()
    skill_df["skill_id"] = skill_df["skill_id"].astype(int)
    skill_df["skill_name"] = skill_df["skill_name"].str.strip()

    # Drop exact duplicates and sort
    unique_skills = skill_df.drop_duplicates().sort_values("skill_id")

    ontology_items: list[dict[str, Any]] = []
    global_seen_aliases: dict[str, int] = {}

    for _, row in unique_skills.iterrows():
        s_id = int(row["skill_id"])
        s_name = str(row["skill_name"])
        category = SKILL_CATEGORY_MAP.get(s_id, "Mathematics")

        cat_norm = normalize_token(category)
        name_norm = normalize_token(s_name)

        if s_id == 163 and s_name == "Absolute Value":
            canonical_name = f"math :: {cat_norm} :: absolute value (advanced)"
            display_name = "Absolute Value (Advanced)"
            raw_alias_base = "Absolute Value (Advanced)"
        else:
            canonical_name = f"math :: {cat_norm} :: {name_norm}"
            display_name = s_name
            raw_alias_base = s_name

        skill_code = f"SKILL_{s_id}"

        # Candidate aliases for this specific skill
        candidate_aliases: list[str] = [
            raw_alias_base,
            f"skill_{s_id}",
            f"assistments_{s_id}",
            f"assistments:{s_id}",
            f"{category} :: {display_name}",
            f"{category} / {display_name}",
        ]

        # Add distinct synonyms
        if s_id == 193:
            candidate_aliases.extend(["Linear Equations", "Linear Equation Solving", "Solving Linear Equations"])
        elif s_id == 350:
            candidate_aliases.extend(["Systems of Linear Equations", "Solving Linear Systems"])
        elif s_id == 378:
            candidate_aliases.extend(["Systems of Linear Equations by Graphing", "Graphing Linear Systems"])
        elif s_id == 221:
            candidate_aliases.extend(["Slope", "Line Slope", "Finding Slope"])
        elif s_id == 331:
            candidate_aliases.extend(["Slope From Situation", "Word Problem Slope"])
        elif s_id == 333:
            candidate_aliases.extend(["Slope From Equation", "Finding Slope From Formula"])
        elif s_id == 334:
            candidate_aliases.extend(["Slope From Ordered Pairs", "Slope Between Two Points"])
        elif s_id == 309:
            candidate_aliases.extend(["Order of Operations Positive Reals", "Basic Order of Operations"])
        elif s_id == 310:
            candidate_aliases.extend(["Order of Operations All", "PEMDAS", "BODMAS", "BIDMAS", "General Order of Operations"])
        elif s_id == 27:
            candidate_aliases.extend(["Pythagorean Theorem", "Pythagoras Theorem", "Pythagorean Equation"])
        elif s_id == 317:
            candidate_aliases.extend(["Greatest Common Factor", "GCF", "Greatest Common Divisor", "GCD"])
        elif s_id == 65:
            candidate_aliases.extend(["Least Common Multiple", "LCM"])

        # Filter and ensure global uniqueness across all skills
        valid_aliases: list[str] = []
        for alias_str in candidate_aliases:
            clean = " ".join(alias_str.strip().split())
            norm = normalize_token(clean)
            if not norm:
                continue
            if norm in global_seen_aliases:
                continue  # Avoid collision across skills
            global_seen_aliases[norm] = s_id
            valid_aliases.append(clean)

        item = {
            "skill_code": skill_code,
            "assistments_skill_id": s_id,
            "canonical_name": canonical_name,
            "display_name": display_name,
            "category": category,
            "description": f"ASSISTments skill {s_id}: {display_name} in {category}.",
            "ontology_version": "1.0",
            "is_active": True,
            "aliases": valid_aliases,
        }
        ontology_items.append(item)

    return ontology_items


def main():
    items = build_ontology()
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated canonical ontology with {len(items)} skills.")
    print(f"Saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
