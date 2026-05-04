import json
from pathlib import Path


TRAIN_PATH = Path(
    "data/raw/mathdial/train.jsonl"
)

TEST_PATH = Path(
    "data/raw/mathdial/test.jsonl"
)


def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def normalize_question(text):

    return " ".join(
        text.lower().split()
    )


train = load_jsonl(
    TRAIN_PATH
)

test = load_jsonl(
    TEST_PATH
)


train_qids = {
    str(row["qid"])
    for row in train
}

test_qids = {
    str(row["qid"])
    for row in test
}


train_questions = {
    normalize_question(
        row["question"]
    )
    for row in train
}

test_questions = {
    normalize_question(
        row["question"]
    )
    for row in test
}


qid_overlap = (
    train_qids
    & test_qids
)

question_overlap = (
    train_questions
    & test_questions
)


print("=" * 70)
print("MATHDIAL OFFICIAL SPLIT OVERLAP CHECK")
print("=" * 70)

print(
    f"\nTrain rows: {len(train)}"
)

print(
    f"Test rows: {len(test)}"
)

print(
    f"\nTrain unique qids: "
    f"{len(train_qids)}"
)

print(
    f"Test unique qids: "
    f"{len(test_qids)}"
)

print(
    f"\nQID overlap: "
    f"{len(qid_overlap)}"
)

print(
    f"Exact normalized question overlap: "
    f"{len(question_overlap)}"
)


if qid_overlap:

    print(
        "\nFirst overlapping qids:"
    )

    print(
        sorted(qid_overlap)[:20]
    )


print(
    "\n"
    + "=" * 70
)

if (
    len(qid_overlap) == 0
    and len(question_overlap) == 0
):

    print(
        "OFFICIAL SPLIT CHECK: PASSED"
    )

else:

    print(
        "OFFICIAL SPLIT CHECK: OVERLAP FOUND"
    )

print("=" * 70)
