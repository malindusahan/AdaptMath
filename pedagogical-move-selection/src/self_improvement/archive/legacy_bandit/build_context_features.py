import json
from pathlib import Path


INPUT_FILE = Path(
    "data/synthetic/self_improvement/trajectory_prefix_examples.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/self_improvement/context_features.jsonl"
)


def count_words(text):
    if not text:
        return 0

    return len(text.split())


def detect_question(text):

    if not text:
        return False

    text = text.lower().strip()

    question_words = [
        "why",
        "how",
        "what",
        "can",
        "could",
        "would",
        "explain"
    ]

    if "?" in text:
        return True

    for word in question_words:
        if text.startswith(word):
            return True

    return False



def build_context(example):

    state = example["outcome_model_input"]
    metadata = example["metadata"]


    history = state["history"]


    # Latest student message available before this decision
    student_text = ""

    for turn in reversed(history):

        if turn["user"].lower() == "student":
            student_text = turn["text"]
            break


    context = {

        # identifiers
        "episode_id":
            metadata["episode_id"],

        "attempt_index":
            metadata["attempt_index"],

        "turn_index":
            metadata["turn_index"],


        # problem information
        "problem_text":
            state["problem"],

        "skill_id":
            state["skill_id"],


        # student state
        "mastery_before":
            state["mastery_before"],


        # conversation state
        "dialogue_history":
            history,


        "turn_number":
            metadata["turn_index"],


        "student_response_length":
            count_words(student_text),


        "asked_question":
            detect_question(student_text),


        # frozen selector output
        "selector_probabilities":
            metadata["base_move_probs"]

    }


    return context




def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    count = 0


    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:


        for line in fin:

            example = json.loads(line)


            context = build_context(example)


            fout.write(
                json.dumps(context)
                + "\n"
            )

            count += 1



    print("="*70)
    print("SI6-A CONTEXT FEATURE BUILDER")
    print("="*70)

    print()

    print("Input:")
    print(INPUT_FILE)

    print()

    print("Output:")
    print(OUTPUT_FILE)

    print()

    print("Records created:", count)



if __name__ == "__main__":
    main()