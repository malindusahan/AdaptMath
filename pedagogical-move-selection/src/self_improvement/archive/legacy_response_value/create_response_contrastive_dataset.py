import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


SEED = 42
NEGATIVES_PER_POSITIVE = 3
MIN_RATIO = 0.70
MAX_RATIO = 1.30

TRAIN_IN = Path(
    "data/processed/self_improvement/value_transitions_train.jsonl"
)

VAL_IN = Path(
    "data/processed/self_improvement/value_transitions_validation.jsonl"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize(text):
    return " ".join(
        str(text).lower().split()
    )


def word_count(text):
    return len(
        str(text).split()
    )


def build_context(row):
    history = row["pre_history"]

    if history:
        history_text = "\n".join(
            f"{turn['user']}: {turn['text']}"
            for turn in history
        )
    else:
        history_text = "[No previous conversation]"

    return (
        f"Problem:\n{row['problem']}\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Teacher strategy:\n{row['teacher_move']}\n\n"
        f"Teacher response:\n{row['teacher_text']}\n\n"
        f"Next student response:"
    )


def build_pools(rows):
    pools = defaultdict(list)

    for index, row in enumerate(rows):
        pools[row["teacher_move"]].append({
            "index": index,
            "qid": str(row["qid"]),
            "source": int(row["source_row_index"]),
            "response": row["student_reply"],
            "normalized": normalize(row["student_reply"]),
            "length": word_count(row["student_reply"]),
        })

    return pools


def valid_negatives(row, row_index, pools):
    positive_text = normalize(row["student_reply"])
    positive_length = word_count(row["student_reply"])

    min_words = max(
        1,
        int(np.floor(positive_length * MIN_RATIO)),
    )

    max_words = max(
        min_words,
        int(np.ceil(positive_length * MAX_RATIO)),
    )

    candidates = []

    for candidate in pools[row["teacher_move"]]:

        if candidate["index"] == row_index:
            continue

        if candidate["qid"] == str(row["qid"]):
            continue

        if candidate["source"] == int(row["source_row_index"]):
            continue

        if candidate["normalized"] == positive_text:
            continue

        if not (
            min_words
            <= candidate["length"]
            <= max_words
        ):
            continue

        candidates.append(candidate)

    return candidates


def build_split(name, rows, seed):
    rng = random.Random(seed)

    pools = build_pools(rows)

    examples = []
    excluded = []

    position_counts = Counter()

    for row_index, row in enumerate(rows):

        negatives = valid_negatives(
            row,
            row_index,
            pools,
        )

        if len(negatives) < NEGATIVES_PER_POSITIVE:
            excluded.append({
                "transition_id": row["transition_id"],
                "teacher_move": row["teacher_move"],
                "candidate_count": len(negatives),
            })
            continue

        sampled = rng.sample(
            negatives,
            NEGATIVES_PER_POSITIVE,
        )

        candidate_list = [{
            "response": row["student_reply"],
            "qid": str(row["qid"]),
            "source_row_index": int(
                row["source_row_index"]
            ),
            "_positive": True,
        }]

        for negative in sampled:
            candidate_list.append({
                "response": negative["response"],
                "qid": negative["qid"],
                "source_row_index": negative["source"],
                "_positive": False,
            })

        rng.shuffle(candidate_list)

        correct_index = next(
            index
            for index, candidate
            in enumerate(candidate_list)
            if candidate["_positive"]
        )

        position_counts[correct_index] += 1

        final_candidates = []

        for candidate in candidate_list:
            final_candidates.append({
                "response": candidate["response"],
                "qid": candidate["qid"],
                "source_row_index": (
                    candidate["source_row_index"]
                ),
            })

        examples.append({
            "example_id": row["transition_id"],
            "group_id": str(row["group_id"]),
            "qid": str(row["qid"]),
            "source_row_index": int(
                row["source_row_index"]
            ),
            "teacher_move": row["teacher_move"],
            "context": build_context(row),
            "candidates": final_candidates,
            "correct_index": correct_index,
        })

    print(f"\n{name}")
    print("Source contexts:", len(rows))
    print("Built examples:", len(examples))
    print("Excluded:", len(excluded))

    if excluded:
        print("Excluded details:")
        for item in excluded:
            print(
                item["transition_id"],
                item["teacher_move"],
                item["candidate_count"],
            )

    return examples, excluded, position_counts


def validate_structure(train_data, val_data):

    bad_candidate_count = 0
    bad_index = 0
    duplicate_candidate_text = 0

    for dataset in [train_data, val_data]:

        for example in dataset:

            candidates = example["candidates"]

            if len(candidates) != 4:
                bad_candidate_count += 1

            if not (
                0 <= example["correct_index"] < 4
            ):
                bad_index += 1

            normalized = [
                normalize(candidate["response"])
                for candidate in candidates
            ]

            if len(set(normalized)) != 4:
                duplicate_candidate_text += 1

    print("\n" + "=" * 70)
    print("STRUCTURAL VALIDATION")
    print("=" * 70)

    print(
        "Invalid candidate counts:",
        bad_candidate_count,
    )

    print(
        "Invalid correct indices:",
        bad_index,
    )

    print(
        "Duplicate candidate text:",
        duplicate_candidate_text,
    )

    assert bad_candidate_count == 0
    assert bad_index == 0
    assert duplicate_candidate_text == 0


def validate_leakage(train_data, val_data):

    train_groups = {
        row["group_id"]
        for row in train_data
    }

    val_groups = {
        row["group_id"]
        for row in val_data
    }

    train_qids = {
        row["qid"]
        for row in train_data
    }

    val_qids = {
        row["qid"]
        for row in val_data
    }

    train_sources = {
        row["source_row_index"]
        for row in train_data
    }

    val_sources = {
        row["source_row_index"]
        for row in val_data
    }

    group_overlap = len(
        train_groups & val_groups
    )

    qid_overlap = len(
        train_qids & val_qids
    )

    source_overlap = len(
        train_sources & val_sources
    )

    print("\n" + "=" * 70)
    print("LEAKAGE VALIDATION")
    print("=" * 70)

    print("Group overlap:", group_overlap)
    print("QID overlap:", qid_overlap)
    print("Source overlap:", source_overlap)

    assert group_overlap == 0
    assert qid_overlap == 0
    assert source_overlap == 0


def print_positions(name, counts, total):

    print(f"\n{name}")

    for position in range(4):

        count = counts[position]

        rate = count / total

        print(
            f"Position {position}: "
            f"{count} "
            f"({rate * 100:.2f}%)"
        )


def main():

    train_rows = load_jsonl(TRAIN_IN)
    val_rows = load_jsonl(VAL_IN)

    assert len(train_rows) == 9989
    assert len(val_rows) == 2509

    print("=" * 70)
    print("SI3-B — CONTRASTIVE DATASET BUILD VALIDATION")
    print("=" * 70)

    train_data, train_excluded, train_positions = (
        build_split(
            "SI TRAIN",
            train_rows,
            SEED,
        )
    )

    val_data, val_excluded, val_positions = (
        build_split(
            "SI VALIDATION",
            val_rows,
            SEED + 1,
        )
    )

    validate_structure(
        train_data,
        val_data,
    )

    validate_leakage(
        train_data,
        val_data,
    )

    print("\n" + "=" * 70)
    print("CORRECT-CANDIDATE POSITION DISTRIBUTION")
    print("=" * 70)

    print_positions(
        "TRAIN",
        train_positions,
        len(train_data),
    )

    print_positions(
        "VALIDATION",
        val_positions,
        len(val_data),
    )

    print("\n" + "=" * 70)
    print("BUILD VALIDATION")
    print("=" * 70)

    print(
        "Expected train exclusions:",
        1,
    )

    print(
        "Actual train exclusions:",
        len(train_excluded),
    )

    print(
        "Expected validation exclusions:",
        2,
    )

    print(
        "Actual validation exclusions:",
        len(val_excluded),
    )

    assert len(train_excluded) == 1
    assert len(val_excluded) == 2

    print("\nSI3-B BUILD VALIDATION: PASSED")
    print("No dataset files written yet.")


if __name__ == "__main__":
    main()
