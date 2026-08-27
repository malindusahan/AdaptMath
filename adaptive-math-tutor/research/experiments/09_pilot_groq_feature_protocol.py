from pathlib import Path
from collections import Counter
import json
import os
import time

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "research"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "eedi_groq_feature_pilot_runs.csv"
)

MODEL_NAME = "openai/gpt-oss-20b"

N_VISUAL = 10
N_NON_VISUAL = 10
N_RUNS = 3

RANDOM_SEED = 42

# Intentionally conservative because free-tier
# limits can also be constrained by tokens/minute.
SECONDS_BETWEEN_CALLS = 10


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


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
            "max_expression_nesting_depth": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
            },
            "has_abstract_symbols": {
                "type": "boolean",
            },
            "units_check": {
                "type": "boolean",
            },
            "real_world_context": {
                "type": "boolean",
            },
        },
        "required": [
            "solution_step_count",
            "bloom_level",
            "misconception_count",
            "information_gap",
            "knowledge_dimension",
            "max_expression_nesting_depth",
            "has_abstract_symbols",
            "units_check",
            "real_world_context",
        ],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """
You are an educational measurement feature extractor
specialized in mathematics assessment.

Analyze only the mathematical content supplied to you.

Do NOT estimate question difficulty.
Do NOT estimate a Rasch parameter.
Do NOT predict student success rates.
Do NOT infer student response statistics.

Extract the following features.

1. solution_step_count
Number of discrete, pedagogically atomic mathematical
steps that a competent student would normally require
to solve the item.

2. bloom_level
Primary cognitive demand:
1 = Remember
2 = Understand
3 = Apply
4 = Analyze
5 = Evaluate
6 = Create

3. misconception_count
Number of distinct plausible conceptual or procedural
misconceptions that could reasonably lead to an
incorrect answer. Count misconceptions, not careless
typing mistakes.

4. information_gap
Amount of mathematical knowledge or inference required
beyond information explicitly supplied:
1 = very small
2 = small
3 = moderate
4 = large

5. knowledge_dimension
Choose exactly one:
Factual
Conceptual
Procedural

6. max_expression_nesting_depth
Maximum nesting depth of mathematical expressions
required in interpreting or solving the item.
A simple un-nested expression has depth 0 or 1;
nested brackets, fractions, powers, functions, or
expressions inside expressions increase the depth.

7. has_abstract_symbols
True when variables, algebraic symbols, or other
abstract symbolic quantities play a meaningful role.
Otherwise false.

8. units_check
True when understanding, comparing, converting, or
operating with measurement units is materially required.
Otherwise false.

9. real_world_context
True when the mathematics is framed using a real-world
situation or objects rather than only abstract
mathematical entities.

Use supplied visual information when it is necessary
to understand the mathematical item.

Return only the required structured fields.
""".strip()


def has_visual(row: pd.Series) -> bool:
    value = row.get(
        "visual_description",
        "",
    )

    if pd.isna(value):
        return False

    return bool(
        str(value).strip()
    )


def visual_text(row: pd.Series) -> str:
    value = row.get(
        "visual_description",
        "",
    )

    if pd.isna(value):
        return ""

    return str(value).strip()


def call_groq(
    client: Groq,
    question: str,
    visual_description: str,
) -> dict:
    if visual_description:
        visual_part = visual_description
    else:
        visual_part = (
            "No additional visual information "
            "is present."
        )

    user_prompt = f"""
MATHEMATICS ITEM

Question:
{question}

Visual information:
{visual_part}
""".strip()

    response = client.chat.completions.create(
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
                "name": "pedagogical_features",
                "strict": True,
                "schema": feature_schema(),
            },
        },
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return json.loads(content)


def agreement(values: list) -> float:
    counts = Counter(values)

    most_common_count = (
        counts.most_common(1)[0][1]
    )

    return (
        most_common_count
        / len(values)
    )


def main() -> None:
    load_dotenv(ENV_PATH)

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
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

    if len(train_items) != 580:
        raise RuntimeError(
            "Expected 580 training items."
        )

    train_items[
        "has_visual"
    ] = train_items.apply(
        has_visual,
        axis=1,
    )

    visual_items = (
        train_items.loc[
            train_items["has_visual"]
        ]
        .sample(
            n=N_VISUAL,
            random_state=RANDOM_SEED,
        )
    )

    non_visual_items = (
        train_items.loc[
            ~train_items["has_visual"]
        ]
        .sample(
            n=N_NON_VISUAL,
            random_state=RANDOM_SEED,
        )
    )

    selected = pd.concat(
        [
            visual_items,
            non_visual_items,
        ],
        ignore_index=True,
    )

    print_section(
        "EXPERIMENT 09 - GROQ FEATURE "
        "PROTOCOL PILOT"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Visual items: {N_VISUAL}"
    )

    print(
        f"Non-visual items: {N_NON_VISUAL}"
    )

    print(
        f"Runs per item: {N_RUNS}"
    )

    print(
        "Total planned API calls: "
        f"{len(selected) * N_RUNS}"
    )

    print(
        f"Selection seed: {RANDOM_SEED}"
    )

    print()
    print(
        "Difficulty labels were NOT used "
        "for item selection."
    )

    client = Groq(
        api_key=api_key
    )

    all_records = []

    total_calls = (
        len(selected)
        * N_RUNS
    )

    call_number = 0

    for item_number, row in selected.iterrows():
        question_id = int(
            row["QuestionId"]
        )

        question = str(
            row["text"]
        ).strip()

        description = visual_text(
            row
        )

        item_type = (
            "visual"
            if row["has_visual"]
            else "non_visual"
        )

        print()
        print(
            f"Item {item_number + 1}/"
            f"{len(selected)} "
            f"- QuestionId {question_id} "
            f"({item_type})"
        )

        for run_number in range(
            1,
            N_RUNS + 1,
        ):
            call_number += 1

            print(
                f"  API call "
                f"{call_number}/{total_calls} "
                f"- run {run_number}/{N_RUNS}"
            )

            features = call_groq(
                client=client,
                question=question,
                visual_description=(
                    description
                ),
            )

            record = {
                "QuestionId":
                    question_id,
                "item_type":
                    item_type,
                "run":
                    run_number,
            }

            record.update(
                features
            )

            all_records.append(
                record
            )

            # Do not sleep after the final call.
            if call_number < total_calls:
                time.sleep(
                    SECONDS_BETWEEN_CALLS
                )

    results = pd.DataFrame(
        all_records
    )

    # -------------------------------------------------
    # STABILITY SUMMARY
    # -------------------------------------------------

    print_section(
        "STABILITY SUMMARY"
    )

    feature_columns = [
        "solution_step_count",
        "bloom_level",
        "misconception_count",
        "information_gap",
        "knowledge_dimension",
        "max_expression_nesting_depth",
        "has_abstract_symbols",
        "units_check",
        "real_world_context",
    ]

    perfect_complete_outputs = 0

    per_feature_agreements = {
        feature: []
        for feature in feature_columns
    }

    for question_id, group in (
        results.groupby(
            "QuestionId"
        )
    ):
        serialized = []

        for _, row in group.iterrows():
            feature_dict = {
                feature: row[feature]
                for feature
                in feature_columns
            }

            serialized.append(
                json.dumps(
                    feature_dict,
                    sort_keys=True,
                    default=str,
                )
            )

        if len(set(serialized)) == 1:
            perfect_complete_outputs += 1

        for feature in feature_columns:
            values = (
                group[feature]
                .tolist()
            )

            per_feature_agreements[
                feature
            ].append(
                agreement(values)
            )

    print(
        "Items with identical complete "
        "output across all 3 runs: "
        f"{perfect_complete_outputs}/"
        f"{len(selected)}"
    )

    print()
    print(
        "Mean within-item agreement "
        "by feature:"
    )

    for feature in feature_columns:
        mean_agreement = (
            sum(
                per_feature_agreements[
                    feature
                ]
            )
            / len(
                per_feature_agreements[
                    feature
                ]
            )
        )

        print(
            f"{feature}: "
            f"{mean_agreement:.3f}"
        )

    # -------------------------------------------------
    # AGGREGATED FEATURE DIVERSITY
    # -------------------------------------------------

    print_section(
        "FEATURE DIVERSITY"
    )

    aggregated_rows = []

    numeric_mean_features = [
        "solution_step_count",
        "misconception_count",
        "information_gap",
    ]

    mode_features = [
        "bloom_level",
        "knowledge_dimension",
        "max_expression_nesting_depth",
        "has_abstract_symbols",
        "units_check",
        "real_world_context",
    ]

    for question_id, group in (
        results.groupby(
            "QuestionId"
        )
    ):
        row = {
            "QuestionId":
                int(question_id),
            "item_type":
                group[
                    "item_type"
                ].iloc[0],
        }

        for feature in numeric_mean_features:
            row[feature] = (
                group[feature]
                .astype(float)
                .mean()
            )

        for feature in mode_features:
            values = (
                group[feature]
                .tolist()
            )

            row[feature] = (
                Counter(values)
                .most_common(1)[0][0]
            )

        aggregated_rows.append(
            row
        )

    aggregated = pd.DataFrame(
        aggregated_rows
    )

    for feature in feature_columns:
        print()
        print(feature)

        if pd.api.types.is_numeric_dtype(
            aggregated[feature]
        ):
            print(
                aggregated[feature]
                .describe()
                .to_string()
            )
        else:
            print(
                aggregated[feature]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )

    # -------------------------------------------------
    # SAVE ONE AUDITABLE FILE
    # -------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_section(
        "LEAKAGE CONTROL"
    )

    print(
        "Groq received question text "
        "and visual description only."
    )

    print(
        "The following were NOT supplied:"
    )

    print(
        "difficulty, SE_beta, n_answers, "
        "correct_rate, error_rate."
    )

    print_section(
        "SAVED OUTPUT"
    )

    print(
        f"Raw pilot runs saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print_section(
        "PILOT COMPLETE"
    )

    print(
        "No predictive model was trained."
    )

    print(
        "No held-out test item was used."
    )

    print(
        "Do not run the 580-item "
        "extraction yet."
    )


if __name__ == "__main__":
    main()