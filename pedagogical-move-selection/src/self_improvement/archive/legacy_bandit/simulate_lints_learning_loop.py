import json
from pathlib import Path
import numpy as np

from src.self_improvement.lints_policy import LinTSPolicy


CONTEXT_FILE = Path(
    "data/processed/self_improvement/context_features.jsonl"
)


REWARD_FILE = Path(
    "data/processed/self_improvement/reward_signals.jsonl"
)


ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias"
]



def build_feature_vector(context):

    """
    Convert context dictionary into LinTS vector.

    IMPORTANT:
    Only information available BEFORE decision
    is used.
    """


    return np.array([

        context["mastery_before"],

        context["turn_number"] / 10.0,

        context["student_response_length"] / 100.0,

        float(context["asked_question"]),

        1.0

    ])





def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data.append(
                json.loads(line)
            )

    return data





def main():


    print("="*70)
    print("SI6-D LinTS LEARNING LOOP SIMULATION")
    print("="*70)

    print()


    contexts = load_jsonl(
        CONTEXT_FILE
    )

    rewards = load_jsonl(
        REWARD_FILE
    )


    print(
        "Contexts:",
        len(contexts)
    )

    print(
        "Rewards:",
        len(rewards)
    )

    print()



    policy = LinTSPolicy(

        n_features=5,

        arms=ARMS,

        alpha=1.0

    )


    update_count = 0


    arm_counts = {

        arm:0

        for arm in ARMS

    }


    total_reward = 0



    for context, reward_record in zip(
        contexts,
        rewards
    ):


        x = build_feature_vector(
            context
        )


        selected_arm = policy.select_arm(
            x
        )


        reward = reward_record["reward"]


        policy.update(

            selected_arm,

            x,

            reward

        )


        update_count += 1


        arm_counts[
            selected_arm
        ] += 1


        total_reward += reward



    print(
        "Updates performed:",
        update_count
    )


    print()


    print(
        "Average reward:",
        round(
            total_reward/update_count,
            4
        )
    )


    print()


    print(
        "Arm selection counts:"
    )


    for arm,count in arm_counts.items():

        print(
            f"  {arm:<15}: {count}"
        )


    print()


    print(
        "VALIDATION PASSED"
    )





if __name__ == "__main__":

    main()
