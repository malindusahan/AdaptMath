from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "external"
    / "Visual_Item_Difficulty"
    / "data"
    / "items.csv"
)

ENV_PATH = (
    PROJECT_ROOT
    / "backend"
    / ".env"
)

MODEL_NAME = "openai/gpt-oss-20b"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # -------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------

    load_dotenv(ENV_PATH)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in backend/.env"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # -------------------------------------------------
    # LOAD ONE TRAINING ITEM
    # -------------------------------------------------

    items = pd.read_csv(DATA_PATH)

    train_items = items.loc[
        items["split"] == "train"
    ].copy()

    visual_mask = (
        train_items["visual_description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    visual_train_items = train_items.loc[
        visual_mask
    ].copy()

    if visual_train_items.empty:
        raise RuntimeError(
            "No visual training items found."
        )

    item = visual_train_items.iloc[0]

    question_id = int(
        item["QuestionId"]
    )

    question_text = str(
        item["text"]
    ).strip()

    visual_description = str(
        item["visual_description"]
    ).strip()

    # IMPORTANT:
    # We deliberately do NOT provide:
    #
    # - difficulty
    # - correct_rate
    # - error_rate
    # - n_answers
    # - SE_beta
    #
    # The LLM receives question content only.

    print_section(
        "SELECTED EEDI TRAINING ITEM"
    )

    print(
        f"QuestionId: {question_id}"
    )

    print()
    print("Question:")
    print(question_text)

    print()
    print("Visual description:")
    print(visual_description)

    # -------------------------------------------------
    # STRICT OUTPUT SCHEMA
    # -------------------------------------------------

    feature_schema = {
        "type": "object",
        "properties": {
            "solution_step_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
            "bloom_level": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
            },
            "misconception_count": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
            },
            "information_gap": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
            },
            "knowledge_dimension": {
                "type": "string",
                "enum": [
                    "Factual",
                    "Conceptual",
                    "Procedural",
                ],
            },
        },
        "required": [
            "solution_step_count",
            "bloom_level",
            "misconception_count",
            "information_gap",
            "knowledge_dimension",
        ],
        "additionalProperties": False,
    }

    # -------------------------------------------------
    # PROMPT
    # -------------------------------------------------

    system_prompt = """
You are an educational measurement feature extractor
specialized in mathematics assessment.

Analyze only the mathematical item content supplied by
the user.

Do NOT estimate Rasch difficulty.
Do NOT predict a difficulty score.
Do NOT infer or use student response statistics.

Extract these pedagogical characteristics:

1. solution_step_count
   Number of discrete, pedagogically atomic steps a
   competent student would normally need to solve the item.

2. bloom_level
   Primary cognitive level:
   1 = Remember
   2 = Understand
   3 = Apply
   4 = Analyze
   5 = Evaluate
   6 = Create

3. misconception_count
   Number of distinct plausible conceptual or procedural
   misconceptions that could lead a student to an
   incorrect response.

4. information_gap
   Amount of mathematical knowledge or inference needed
   beyond information explicitly provided in the item:
   1 = very small
   2 = small
   3 = moderate
   4 = large

5. knowledge_dimension
   Choose exactly one:
   Factual
   Conceptual
   Procedural

Use the visual description when it is necessary to
understand the mathematics.

Return only the required structured fields.
""".strip()

    user_prompt = f"""
MATHEMATICS ITEM

Question:
{question_text}

Visual information:
{visual_description}
""".strip()

    # -------------------------------------------------
    # GROQ REQUEST
    # -------------------------------------------------

    print_section(
        "CALLING GROQ"
    )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": (
                    "pedagogical_features"
                ),
                "strict": True,
                "schema": feature_schema,
            },
        },
    )

    raw_content = (
        response
        .choices[0]
        .message
        .content
    )

    if not raw_content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    features = json.loads(
        raw_content
    )

    # -------------------------------------------------
    # SIMPLE LOCAL VALIDATION
    # -------------------------------------------------

    expected_keys = {
        "solution_step_count",
        "bloom_level",
        "misconception_count",
        "information_gap",
        "knowledge_dimension",
    }

    returned_keys = set(
        features.keys()
    )

    if returned_keys != expected_keys:
        raise RuntimeError(
            "Returned feature keys do not "
            "match the expected schema."
        )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    print_section(
        "STRUCTURED PEDAGOGICAL FEATURES"
    )

    print(
        json.dumps(
            features,
            indent=2,
        )
    )

    print_section(
        "LEAKAGE CHECK"
    )

    print(
        "Empirical Rasch difficulty "
        "was NOT supplied to Groq."
    )

    print(
        "Student correct/error rates "
        "were NOT supplied to Groq."
    )

    print(
        "Only question content and "
        "visual description were supplied."
    )

    print_section(
        "SMOKE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()