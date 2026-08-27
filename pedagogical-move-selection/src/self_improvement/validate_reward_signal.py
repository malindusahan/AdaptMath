import json
from pathlib import Path


FILE = Path(
    "data/processed/self_improvement/reward_signals.jsonl"
)



def main():

    errors = 0
    records = 0


    with open(FILE, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            records += 1


            reward = item["reward"]


            if reward < 0 or reward > 1:

                print(
                    "Invalid reward:",
                    reward
                )

                errors += 1



            required = [
                "episode_id",
                "attempt_index",
                "evaluator_rate",
                "mastery_delta",
                "reward"
            ]


            for field in required:

                if field not in item:

                    print(
                        "Missing field:",
                        field
                    )

                    errors += 1



            # Future information leakage check

            forbidden = [
                "teacher_response",
                "student_response",
                "mastery_after",
                "evaluator_score"
            ]


            for field in forbidden:

                if field in item:

                    print(
                        "Leakage detected:",
                        field
                    )

                    errors += 1



    print("=" * 70)
    print("SI6-B REWARD VALIDATION")
    print("=" * 70)

    print()

    print("Records:", records)
    print("Errors:", errors)


    if errors == 0:

        print()
        print("VALIDATION PASSED")

    else:

        print()
        print("VALIDATION FAILED")



if __name__ == "__main__":
    main()
