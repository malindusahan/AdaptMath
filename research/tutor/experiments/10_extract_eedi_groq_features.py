from pathlib import Path
import json
import os
import re
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

RAW_OUTPUT_PATH = (
    RESULTS_DIR
    / "eedi_groq_features_raw.csv"
)

FINAL_OUTPUT_PATH = (
    RESULTS_DIR
    / "eedi_groq_features_run1.csv"
)

MODEL_NAME = "openai/gpt-oss-20b"

# From now on we collect exactly one extraction
# per training question for model development.
TARGET_RUN = 1

MAX_RETRIES = 10


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

Extract the following mathematical and pedagogical
characteristics.

1. solution_step_count

Estimate the number of discrete, pedagogically atomic
mathematical steps that a competent student would
normally require to solve the problem.

2. bloom_level

Identify the primary cognitive demand:

1 = Remember
2 = Understand
3 = Apply
4 = Analyze
5 = Evaluate
6 = Create

3. misconception_count

Estimate the number of distinct plausible conceptual
or procedural misconceptions that could reasonably
lead a student to an incorrect answer.

Count misconceptions rather than accidental typing
or copying mistakes.

4. information_gap

Estimate how much mathematical knowledge or inference
is needed beyond information explicitly supplied:

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

Estimate the maximum structural nesting depth of the
mathematical expressions that must be interpreted or
used when solving the problem.

Use 0 when there is effectively no nested mathematical
expression. Increase the value for nested brackets,
fractions, powers, functions, or expressions embedded
inside other expressions.

7. has_abstract_symbols

True when variables, algebraic symbols, or abstract
symbolic quantities play a meaningful mathematical
role.

Otherwise false.

8. units_check

True when understanding, comparing, converting, or
operating with measurement units is materially
required to solve the problem.

Otherwise false.

9. real_world_context

True when the mathematics is framed using real-world
objects, situations, measurements, events, or applied
contexts.

Otherwise false.

Use the supplied visual description when it contains
mathematical information required to understand the
item.

Return only the required structured fields.
""".strip()


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


def extract_features(
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


def append_record(record: dict) -> None:
    dataframe = pd.DataFrame(
        [record]
    )

    file_exists = (
        RAW_OUTPUT_PATH.exists()
    )

    dataframe.to_csv(
        RAW_OUTPUT_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_raw_results() -> pd.DataFrame:
    if not RAW_OUTPUT_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        RAW_OUTPUT_PATH
    )

    if dataframe.empty:
        return dataframe

    duplicate_count = int(
        dataframe[
            [
                "QuestionId",
                "run",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate QuestionId/run pairs "
            "already exist in the raw file."
        )

    return dataframe


def get_completed_run1_ids(
    raw_results: pd.DataFrame,
) -> set[int]:
    if raw_results.empty:
        return set()

    run1 = raw_results.loc[
        raw_results["run"] == TARGET_RUN
    ]

    return set(
        run1["QuestionId"]
        .astype(int)
        .tolist()
    )


def parse_wait_seconds(
    error_message: str,
) -> int:
    """
    If Groq tells us something like:

    'Please try again in 2m12.624s'

    extract that value so the script waits
    approximately as long as requested.
    """

    minutes_match = re.search(
        r"try again in\s+"
        r"(?:(\d+)m)?"
        r"([\d.]+)s",
        error_message,
        flags=re.IGNORECASE,
    )

    if minutes_match:
        minutes_text = (
            minutes_match.group(1)
        )

        seconds_text = (
            minutes_match.group(2)
        )

        minutes = (
            int(minutes_text)
            if minutes_text
            else 0
        )

        seconds = float(
            seconds_text
        )

        total = (
            minutes * 60
            + seconds
        )

        # Small safety margin.
        return int(total) + 5

    # Generic fallback.
    return 60


def build_final_run1_file(
    train_items: pd.DataFrame,
) -> pd.DataFrame:
    raw = pd.read_csv(
        RAW_OUTPUT_PATH
    )

    run1 = (
        raw.loc[
            raw["run"] == TARGET_RUN
        ]
        .copy()
    )

    duplicate_run1 = int(
        run1["QuestionId"]
        .duplicated()
        .sum()
    )

    if duplicate_run1 != 0:
        raise RuntimeError(
            "Duplicate run-1 QuestionIds found."
        )

    if len(run1) != 580:
        raise RuntimeError(
            "Expected exactly 580 run-1 "
            f"feature rows, found {len(run1)}."
        )

    train_ids = set(
        train_items["QuestionId"]
        .astype(int)
    )

    run1_ids = set(
        run1["QuestionId"]
        .astype(int)
    )

    missing = (
        train_ids
        - run1_ids
    )

    unexpected = (
        run1_ids
        - train_ids
    )

    if missing:
        raise RuntimeError(
            "Training questions missing from "
            f"features: {len(missing)}"
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected QuestionIds in "
            f"features: {len(unexpected)}"
        )

    run1 = (
        run1.sort_values(
            "QuestionId"
        )
        .reset_index(
            drop=True
        )
    )

    run1.to_csv(
        FINAL_OUTPUT_PATH,
        index=False,
    )

    return run1


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

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
            "Expected exactly 580 "
            "training items."
        )

    raw_results = (
        load_raw_results()
    )

    completed_ids = (
        get_completed_run1_ids(
            raw_results
        )
    )

    print_section(
        "EEDI SINGLE-RUN GROQ "
        "FEATURE EXTRACTION"
    )

    print(
        f"Training questions: "
        f"{len(train_items)}"
    )

    print(
        "Required extraction per "
        "question: 1"
    )

    print(
        "Run-1 questions already "
        f"completed: {len(completed_ids)}"
    )

    print(
        "Remaining questions: "
        f"{580 - len(completed_ids)}"
    )

    print()
    print(
        "Existing run-2/run-3 rows "
        "are preserved for the earlier "
        "stability analysis."
    )

    print()
    print(
        "Held-out test questions used: 0"
    )

    print(
        "Difficulty labels sent to Groq: NO"
    )

    client = Groq(
        api_key=api_key
    )

    for position, row in (
        train_items.iterrows()
    ):
        question_id = int(
            row["QuestionId"]
        )

        if question_id in completed_ids:
            print(
                f"Item {position + 1}/580 "
                f"- QuestionId {question_id}: "
                "already saved"
            )

            continue

        question = str(
            row["text"]
        ).strip()

        visual_description = (
            get_visual_description(
                row
            )
        )

        print()
        print(
            f"Item {position + 1}/580 "
            f"- QuestionId {question_id}"
        )

        success = False

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                print(
                    "  Run 1 "
                    f"- API attempt "
                    f"{attempt}/{MAX_RETRIES}"
                )

                features = (
                    extract_features(
                        client=client,
                        question=question,
                        visual_description=(
                            visual_description
                        ),
                    )
                )

                record = {
                    "QuestionId":
                        question_id,
                    "run":
                        TARGET_RUN,
                    "model":
                        MODEL_NAME,
                    "has_visual":
                        bool(
                            visual_description
                        ),
                }

                record.update(
                    features
                )

                append_record(
                    record
                )

                completed_ids.add(
                    question_id
                )

                print(
                    "    saved"
                )

                print(
                    "    Run-1 progress: "
                    f"{len(completed_ids)}/580"
                )

                print(
                    "    Remaining: "
                    f"{580 - len(completed_ids)}"
                )

                success = True

                break

            except Exception as error:
                error_message = str(
                    error
                )

                print(
                    "    API error:"
                )

                print(
                    f"    {type(error).__name__}: "
                    f"{error_message}"
                )

                if attempt == MAX_RETRIES:
                    print()
                    print(
                        "Maximum retries reached."
                    )

                    print(
                        "All successful progress "
                        "has already been saved."
                    )

                    print(
                        "You can safely rerun this "
                        "same script later."
                    )

                    raise

                wait_seconds = (
                    parse_wait_seconds(
                        error_message
                    )
                )

                print(
                    "    Waiting approximately "
                    f"{wait_seconds} seconds "
                    "before retry..."
                )

                time.sleep(
                    wait_seconds
                )

        if not success:
            raise RuntimeError(
                "Feature extraction failed."
            )

        # Small pause between successful calls.
        if len(completed_ids) < 580:
            time.sleep(2)

    print_section(
        "FINAL INTEGRITY CHECK"
    )

    raw = pd.read_csv(
        RAW_OUTPUT_PATH
    )

    run_counts = (
        raw["run"]
        .value_counts()
        .sort_index()
    )

    print(
        "All raw saved rows: "
        f"{len(raw)}"
    )

    print()
    print(
        "Rows by run:"
    )

    print(
        run_counts.to_string()
    )

    run1_count = int(
        (
            raw["run"]
            == TARGET_RUN
        ).sum()
    )

    print()
    print(
        "Run-1 feature rows: "
        f"{run1_count}"
    )

    if run1_count != 580:
        raise RuntimeError(
            "Run-1 extraction is incomplete."
        )

    final_features = (
        build_final_run1_file(
            train_items
        )
    )

    print(
        "Unique run-1 questions: "
        f"{final_features['QuestionId'].nunique()}"
    )

    print()
    print(
        "PASS: exactly one primary GenAI "
        "feature vector exists for every "
        "training question."
    )

    print_section(
        "LEAKAGE CONTROL"
    )

    print(
        "Groq received only:"
    )

    print(
        "- question text"
    )

    print(
        "- visual description"
    )

    print()
    print(
        "Groq did NOT receive:"
    )

    print(
        "- difficulty"
    )

    print(
        "- SE_beta"
    )

    print(
        "- n_answers"
    )

    print(
        "- correct_rate"
    )

    print(
        "- error_rate"
    )

    print_section(
        "OUTPUT"
    )

    print(
        "Raw extraction history:"
    )

    print(
        RAW_OUTPUT_PATH
    )

    print()
    print(
        "Primary 580-item feature table:"
    )

    print(
        FINAL_OUTPUT_PATH
    )

    print_section(
        "EXTRACTION COMPLETE"
    )

    print(
        "The 145 held-out test items "
        "remain untouched."
    )

    print(
        "Next step: train the "
        "Complexity Model using frozen "
        "5-fold cross-validation."
    )


if __name__ == "__main__":
    main()