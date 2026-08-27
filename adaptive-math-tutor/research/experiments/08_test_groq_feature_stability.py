from pathlib import Path
from collections import Counter
import json
import os
import statistics

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
N_RUNS = 3


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def has_visual_description(row: pd.Series) -> bool:
    value = row.get(
        "visual_description",
        "",
    )

    if pd.isna(value):
        return False

    return bool(
        str(value).strip()
    )


def get_visual_description(
    row: pd.Series,
) -> str:
    value = row.get(
        "visual_description",
        "",
    )

    if pd.isna(value):
        return ""

    return str(value).strip()


def mode_and_agreement(
    values: list,
) -> tuple[object, float]:
    counts = Counter(values)

    mode_value, mode_count = (
        counts.most_common(1)[0]
    )

    agreement = (
        mode_count
        / len(values)
    )

    return (
        mode_value,
        agreement,
    )


def feature_schema() -> dict:
    return {
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


SYSTEM_PROMPT = """
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


def extract_features(
    client: Groq,
    question_text: str,
    visual_description: str,
) -> dict:
    if visual_description:
        visual_part = (
            visual_description
        )
    else:
        visual_part = (
            "No additional visual "
            "information is present."
        )

    user_prompt = f"""
MATHEMATICS ITEM

Question:
{question_text}

Visual information:
{visual_part}
""".strip()

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
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
                    "schema": (
                        feature_schema()
                    ),
                },
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "Groq returned an empty "
            "response."
        )

    return json.loads(
        content
    )


def analyse_runs(
    results: list[dict],
) -> None:
    numeric_features = [
        "solution_step_count",
        "misconception_count",
        "information_gap",
    ]

    ordinal_features = [
        "bloom_level",
    ]

    categorical_features = [
        "knowledge_dimension",
    ]

    print()
    print("Numeric feature stability:")

    for feature in numeric_features:
        values = [
            result[feature]
            for result in results
        ]

        mean_value = statistics.mean(
            values
        )

        if len(values) > 1:
            std_value = statistics.stdev(
                values
            )
        else:
            std_value = 0.0

        value_range = (
            max(values)
            - min(values)
        )

        print(
            f"{feature}: "
            f"values={values}, "
            f"mean={mean_value:.3f}, "
            f"std={std_value:.3f}, "
            f"range={value_range}"
        )

    print()
    print("Ordinal feature stability:")

    for feature in ordinal_features:
        values = [
            result[feature]
            for result in results
        ]

        mode_value, agreement = (
            mode_and_agreement(
                values
            )
        )

        print(
            f"{feature}: "
            f"values={values}, "
            f"mode={mode_value}, "
            f"agreement={agreement:.3f}"
        )

    print()
    print("Categorical feature stability:")

    for feature in categorical_features:
        values = [
            result[feature]
            for result in results
        ]

        mode_value, agreement = (
            mode_and_agreement(
                values
            )
        )

        print(
            f"{feature}: "
            f"values={values}, "
            f"mode={mode_value}, "
            f"agreement={agreement:.3f}"
        )

    serialized = [
        json.dumps(
            result,
            sort_keys=True,
        )
        for result in results
    ]

    unique_outputs = len(
        set(serialized)
    )

    print()
    print(
        "Unique complete outputs "
        f"across {len(results)} runs: "
        f"{unique_outputs}"
    )


def main() -> None:
    load_dotenv(
        ENV_PATH
    )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found."
        )

    items = pd.read_csv(
        DATA_PATH
    )

    train_items = (
        items.loc[
            items["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ---------------------------------------------
    # Select one visual and one non-visual item.
    #
    # Selection does NOT use difficulty or any
    # student-response field.
    # ---------------------------------------------

    visual_mask = (
        train_items.apply(
            has_visual_description,
            axis=1,
        )
    )

    visual_items = (
        train_items.loc[
            visual_mask
        ]
    )

    non_visual_items = (
        train_items.loc[
            ~visual_mask
        ]
    )

    if visual_items.empty:
        raise RuntimeError(
            "No visual training item found."
        )

    if non_visual_items.empty:
        raise RuntimeError(
            "No non-visual training item found."
        )

    selected_items = [
        (
            "VISUAL",
            visual_items.iloc[0],
        ),
        (
            "NON-VISUAL",
            non_visual_items.iloc[0],
        ),
    ]

    client = Groq(
        api_key=api_key
    )

    print_section(
        "GROQ PEDAGOGICAL FEATURE "
        "STABILITY TEST"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Repeated calls per item: "
        f"{N_RUNS}"
    )

    print(
        "Item selection used no "
        "difficulty labels."
    )

    for item_type, row in selected_items:
        question_id = int(
            row["QuestionId"]
        )

        question_text = str(
            row["text"]
        ).strip()

        visual_description = (
            get_visual_description(
                row
            )
        )

        print_section(
            f"{item_type} ITEM "
            f"- QuestionId {question_id}"
        )

        print("Question:")
        print(question_text)

        print()

        if visual_description:
            print(
                "Visual description:"
            )

            print(
                visual_description
            )
        else:
            print(
                "Visual description: "
                "NONE"
            )

        results = []

        for run_number in range(
            1,
            N_RUNS + 1,
        ):
            print()
            print(
                f"Calling Groq: "
                f"run {run_number}/"
                f"{N_RUNS}"
            )

            result = extract_features(
                client=client,
                question_text=(
                    question_text
                ),
                visual_description=(
                    visual_description
                ),
            )

            results.append(
                result
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

        print_section(
            f"STABILITY SUMMARY "
            f"- QuestionId {question_id}"
        )

        analyse_runs(
            results
        )

    print_section(
        "LEAKAGE CONTROL"
    )

    print(
        "Groq received only question "
        "content and visual descriptions."
    )

    print(
        "Rasch difficulty, correct_rate, "
        "error_rate, n_answers, and SE_beta "
        "were not supplied."
    )

    print_section(
        "EXPERIMENT COMPLETE"
    )

    print(
        "No model was trained."
    )

    print(
        "No held-out test item was used."
    )

    print(
        "This experiment evaluates only "
        "LLM feature repeatability."
    )


if __name__ == "__main__":
    main()