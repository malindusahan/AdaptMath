import json
from pathlib import Path


INPUT_FILE = Path(
    "data/synthetic/self_improvement/trajectory_prefix_examples.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/self_improvement/reward_signals.jsonl"
)


# Reward design:
# 80% evaluator success
# 20% mastery improvement

EVAL_WEIGHT = 0.8
MASTERY_WEIGHT = 0.2



def normalize_mastery_delta(delta):
    """
    Convert mastery change into [0,1].

    Example:
    +0.25 -> 0.625
     0.00 -> 0.5
    -0.50 -> 0.25
    """

    value = (delta + 1) / 2

    return max(0, min(1, value))



def calculate_reward(
    evaluator_rate,
    mastery_delta
):

    mastery_signal = normalize_mastery_delta(
        mastery_delta
    )

    reward = (
        EVAL_WEIGHT * evaluator_rate
        +
        MASTERY_WEIGHT * mastery_signal
    )

    return reward



def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0


    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:


        for line in fin:

            record = json.loads(line)


            targets = record["targets"]
            metadata = record["metadata"]


            evaluator_rate = targets["evaluator_rate"]
            mastery_delta = targets["mastery_delta"]


            reward = calculate_reward(
                evaluator_rate,
                mastery_delta
            )


            output = {

                "episode_id":
                    metadata["episode_id"],

                "attempt_index":
                    metadata["attempt_index"],

                "turn_index":
                    metadata["turn_index"],

                "evaluator_rate":
                    evaluator_rate,

                "mastery_delta":
                    mastery_delta,

                "reward":
                    reward
            }


            fout.write(
                json.dumps(output)
                + "\n"
            )


            count += 1



    print("=" * 70)
    print("SI6-B REWARD SIGNAL BUILDER")
    print("=" * 70)

    print()

    print("Evaluator weight:", EVAL_WEIGHT)
    print("Mastery weight:", MASTERY_WEIGHT)

    print()

    print("Records created:", count)

    print()

    print("Saved:")
    print(OUTPUT_FILE)



if __name__ == "__main__":
    main()
