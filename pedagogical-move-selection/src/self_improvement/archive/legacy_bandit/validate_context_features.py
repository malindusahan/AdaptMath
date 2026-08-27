import json
from pathlib import Path


FILE = Path(
    "data/processed/self_improvement/context_features.jsonl"
)


REQUIRED_FIELDS = [
    "episode_id",
    "attempt_index",
    "turn_index",
    "problem_text",
    "skill_id",
    "mastery_before",
    "dialogue_history",
    "turn_number",
    "student_response_length",
    "asked_question",
    "selector_probabilities"
]


FORBIDDEN_FIELDS = [
    "evaluator_score",
    "mastery_after",
    "mastery_delta",
    "reward",
    "session_success"
]


def main():

    errors = 0
    records = 0


    with open(FILE, "r", encoding="utf-8") as f:

        for line in f:

            item=json.loads(line)

            records += 1


            for field in REQUIRED_FIELDS:

                if field not in item:
                    print(
                        "Missing:",
                        field
                    )
                    errors += 1


            for field in FORBIDDEN_FIELDS:

                if field in item:
                    print(
                        "LEAKAGE:",
                        field
                    )
                    errors += 1


            if not 0 <= item["mastery_before"] <= 1:

                print(
                    "Invalid mastery:",
                    item["mastery_before"]
                )

                errors += 1


            if item["student_response_length"] < 0:

                print(
                    "Invalid response length"
                )

                errors += 1



    print("="*70)
    print("SI6-A CONTEXT VALIDATION")
    print("="*70)

    print()

    print("Records:", records)

    print()

    print("Errors:", errors)


    if errors == 0:

        print()
        print("VALIDATION PASSED")

    else:

        print()
        print("VALIDATION FAILED")



if __name__ == "__main__":
    main()