import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_train.jsonl"
)

VAL_PATH = Path(
    "data/processed/self_improvement/"
    "value_transitions_validation.jsonl"
)

RESULT_PATH = Path(
    "results/self_improvement/"
    "si3_retrieval_hard_negative_audit.json"
)


# ============================================================
# CONFIG
# ============================================================

MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]

NEGATIVES_REQUIRED = 3

LENGTH_MIN = 0.50
LENGTH_MAX = 1.50

INITIAL_NEIGHBORS = 100

SEED = 42

random.seed(SEED)


# ============================================================
# LOAD
# ============================================================

def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


train_rows = load_jsonl(
    TRAIN_PATH
)

val_rows = load_jsonl(
    VAL_PATH
)


assert len(train_rows) == 9989
assert len(val_rows) == 2509


print("=" * 70)

print(
    "SI3-D â€” RETRIEVAL HARD-NEGATIVE FEASIBILITY"
)

print("=" * 70)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def word_count(text):

    return len(
        str(text).split()
    )


def recent_history(
    history,
    max_turns=4,
):

    if not history:

        return "[no history]"

    selected = history[
        -max_turns:
    ]

    return "\n".join(

        f"{turn['user']}: "
        f"{turn['text']}"

        for turn in selected
    )


# ============================================================
# RETRIEVAL CONTEXT
#
# IMPORTANT:
#
# Student reply is NOT included here.
#
# Retrieval only sees information that exists
# BEFORE the next student response.
# ============================================================

def retrieval_text(row):

    return (
        "Problem:\n"
        f"{row['problem']}\n\n"

        "Recent conversation:\n"
        f"{recent_history(row['pre_history'])}\n\n"

        "Teacher move:\n"
        f"{row['teacher_move']}\n\n"

        "Teacher response:\n"
        f"{row['teacher_text']}"
    )


# ============================================================
# PREPARE TEXT
# ============================================================

train_context_texts = [

    retrieval_text(row)

    for row in train_rows
]


val_context_texts = [

    retrieval_text(row)

    for row in val_rows
]


print(
    "\nTrain contexts:",
    len(train_context_texts),
)

print(
    "Validation contexts:",
    len(val_context_texts),
)


# ============================================================
# TF-IDF
#
# Fit ONLY on SI train.
#
# Validation uses the train vocabulary.
# ============================================================

print(
    "\nFitting train-only TF-IDF retrieval model..."
)


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(
        1,
        2,
    ),

    min_df=2,

    max_df=0.98,

    sublinear_tf=True,

    max_features=50000,
)


X_train = vectorizer.fit_transform(
    train_context_texts
)

X_val = vectorizer.transform(
    val_context_texts
)


print(
    "Vocabulary size:",
    len(
        vectorizer.vocabulary_
    ),
)

print(
    "Train matrix:",
    X_train.shape,
)

print(
    "Validation matrix:",
    X_val.shape,
)


# ============================================================
# MOVE-SPECIFIC INDEX
# ============================================================

def build_move_indices(rows):

    result = defaultdict(
        list
    )

    for index, row in enumerate(rows):

        move = row[
            "teacher_move"
        ]

        assert move in MOVES

        result[
            move
        ].append(
            index
        )

    return result


train_move_indices = (
    build_move_indices(
        train_rows
    )
)

val_move_indices = (
    build_move_indices(
        val_rows
    )
)


# ============================================================
# FIT RETRIEVERS INSIDE EACH SPLIT / MOVE
# ============================================================

def build_retrievers(
    rows,
    X,
    move_indices,
):

    retrievers = {}

    for move in MOVES:

        indices = move_indices[
            move
        ]

        matrix = X[
            indices
        ]

        model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
        )

        model.fit(
            matrix
        )

        retrievers[
            move
        ] = {
            "model":
                model,

            "indices":
                indices,
        }

    return retrievers


train_retrievers = (
    build_retrievers(
        train_rows,
        X_train,
        train_move_indices,
    )
)


val_retrievers = (
    build_retrievers(
        val_rows,
        X_val,
        val_move_indices,
    )
)


# ============================================================
# CANDIDATE VALIDITY
# ============================================================

def candidate_is_valid(
    query_row,
    candidate_row,
):

    # --------------------------------------------
    # Different source dialogue
    # --------------------------------------------

    if (
        int(
            query_row[
                "source_row_index"
            ]
        )
        ==
        int(
            candidate_row[
                "source_row_index"
            ]
        )
    ):

        return False


    # --------------------------------------------
    # Different QID
    # --------------------------------------------

    if (
        str(
            query_row[
                "qid"
            ]
        )
        ==
        str(
            candidate_row[
                "qid"
            ]
        )
    ):

        return False


    # --------------------------------------------
    # Different leakage-safe group
    # --------------------------------------------

    if (
        str(
            query_row[
                "group_id"
            ]
        )
        ==
        str(
            candidate_row[
                "group_id"
            ]
        )
    ):

        return False


    # --------------------------------------------
    # Same move
    # --------------------------------------------

    if (
        query_row[
            "teacher_move"
        ]
        !=
        candidate_row[
            "teacher_move"
        ]
    ):

        return False


    # --------------------------------------------
    # Negative response cannot equal positive
    # --------------------------------------------

    if (
        normalize_text(
            query_row[
                "student_reply"
            ]
        )
        ==
        normalize_text(
            candidate_row[
                "student_reply"
            ]
        )
    ):

        return False


    # --------------------------------------------
    # Similar student-response length
    # --------------------------------------------

    query_words = word_count(
        query_row[
            "student_reply"
        ]
    )

    candidate_words = word_count(
        candidate_row[
            "student_reply"
        ]
    )


    minimum = max(
        1,
        int(
            np.floor(
                query_words
                * LENGTH_MIN
            )
        ),
    )

    maximum = max(
        minimum,
        int(
            np.ceil(
                query_words
                * LENGTH_MAX
            )
        ),
    )


    if not (
        minimum
        <= candidate_words
        <= maximum
    ):

        return False


    return True


# ============================================================
# RETRIEVE TOP HARD NEGATIVES
# ============================================================

def retrieve_negatives(
    query_index,
    rows,
    X,
    retrievers,
):

    query_row = rows[
        query_index
    ]

    move = query_row[
        "teacher_move"
    ]

    bundle = retrievers[
        move
    ]

    model = bundle[
        "model"
    ]

    move_indices = bundle[
        "indices"
    ]


    pool_size = len(
        move_indices
    )


    k = min(
        INITIAL_NEIGHBORS,
        pool_size,
    )


    distances, local_indices = (
        model.kneighbors(
            X[
                query_index
            ],
            n_neighbors=k,
        )
    )


    selected = []


    for distance, local_index in zip(
        distances[0],
        local_indices[0],
    ):

        global_index = (
            move_indices[
                int(
                    local_index
                )
            ]
        )

        candidate = rows[
            global_index
        ]


        if not candidate_is_valid(
            query_row,
            candidate,
        ):

            continue


        selected.append({

            "index":
                global_index,

            "similarity":
                float(
                    1.0
                    - distance
                ),

            "row":
                candidate,
        })


        if (
            len(selected)
            >=
            NEGATIVES_REQUIRED
        ):

            break


    # ========================================================
    # Rare fallback:
    #
    # If top-100 does not contain enough valid candidates,
    # search the whole same-move pool.
    # ========================================================

    if (
        len(selected)
        <
        NEGATIVES_REQUIRED
        and
        k
        <
        pool_size
    ):

        distances, local_indices = (
            model.kneighbors(
                X[
                    query_index
                ],
                n_neighbors=pool_size,
            )
        )


        selected = []


        for distance, local_index in zip(
            distances[0],
            local_indices[0],
        ):

            global_index = (
                move_indices[
                    int(
                        local_index
                    )
                ]
            )

            candidate = rows[
                global_index
            ]


            if not candidate_is_valid(
                query_row,
                candidate,
            ):

                continue


            selected.append({

                "index":
                    global_index,

                "similarity":
                    float(
                        1.0
                        - distance
                    ),

                "row":
                    candidate,
            })


            if (
                len(selected)
                >=
                NEGATIVES_REQUIRED
            ):

                break


    return selected


# ============================================================
# AUDIT SPLIT
# ============================================================

def audit_split(
    split_name,
    rows,
    X,
    retrievers,
):

    counts = []

    similarities = []

    insufficient = []

    examples = []

    by_move = defaultdict(
        list
    )


    for index, row in enumerate(
        rows
    ):

        retrieved = retrieve_negatives(
            index,
            rows,
            X,
            retrievers,
        )


        count = len(
            retrieved
        )


        counts.append(
            count
        )


        by_move[
            row[
                "teacher_move"
            ]
        ].append(
            count
        )


        if (
            count
            <
            NEGATIVES_REQUIRED
        ):

            insufficient.append(
                index
            )


        if retrieved:

            similarities.extend([

                item[
                    "similarity"
                ]

                for item
                in retrieved

            ])


        # --------------------------------------------
        # Save several examples with all 3 negatives.
        # --------------------------------------------

        if (
            count
            >= 3
            and
            len(examples)
            < 5
        ):

            examples.append({

                "index":
                    index,

                "row":
                    row,

                "retrieved":
                    retrieved,
            })


    counts_array = np.asarray(
        counts,
        dtype=np.int64,
    )


    similarity_array = np.asarray(
        similarities,
        dtype=np.float64,
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{split_name} RETRIEVAL COVERAGE"
    )

    print(
        "=" * 70
    )


    print(
        "\nContexts:",
        len(rows),
    )


    print(
        "Contexts with >=3 negatives:",
        f"{np.sum(counts_array >= 3)}/"
        f"{len(rows)} "
        f"({(
            np.mean(
                counts_array >= 3
            )
            * 100
        ):.2f}%)",
    )


    print(
        "Insufficient contexts:",
        len(
            insufficient
        ),
    )


    if len(
        similarity_array
    ) > 0:

        print(
            "\nRetrieved-context cosine similarity:"
        )

        print(
            "  Mean:",
            f"{np.mean(similarity_array):.4f}",
        )

        print(
            "  Median:",
            f"{np.median(similarity_array):.4f}",
        )

        print(
            "  P10:",
            f"{np.percentile(similarity_array, 10):.4f}",
        )

        print(
            "  P90:",
            f"{np.percentile(similarity_array, 90):.4f}",
        )


    print(
        "\nCoverage by move:"
    )


    move_results = {}


    for move in MOVES:

        values = np.asarray(
            by_move[
                move
            ],
            dtype=np.int64,
        )


        coverage = float(
            np.mean(
                values >= 3
            )
        )


        print(
            f"\n  {move}"
        )

        print(
            "    Contexts:",
            len(values),
        )

        print(
            "    >=3 negatives:",
            int(
                np.sum(
                    values >= 3
                )
            ),
        )

        print(
            "    Coverage:",
            f"{coverage * 100:.2f}%",
        )


        move_results[
            move
        ] = {
            "contexts":
                len(values),

            "coverage":
                coverage,
        }


    return {

        "contexts":
            len(rows),

        "sufficient":
            int(
                np.sum(
                    counts_array >= 3
                )
            ),

        "insufficient":
            len(
                insufficient
            ),

        "coverage":
            float(
                np.mean(
                    counts_array >= 3
                )
            ),

        "mean_similarity":
            float(
                np.mean(
                    similarity_array
                )
            ),

        "median_similarity":
            float(
                np.median(
                    similarity_array
                )
            ),

        "by_move":
            move_results,

        "examples":
            examples,
    }


# ============================================================
# RUN
# ============================================================

train_result = audit_split(
    "SI TRAIN",
    train_rows,
    X_train,
    train_retrievers,
)


val_result = audit_split(
    "SI VALIDATION",
    val_rows,
    X_val,
    val_retrievers,
)


# ============================================================
# MANUAL EXAMPLES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RETRIEVAL HARD-NEGATIVE EXAMPLES"
)

print(
    "=" * 70
)


def print_examples(
    split_name,
    examples,
):

    print(
        f"\n### {split_name}"
    )


    for example in examples[
        :3
    ]:

        row = example[
            "row"
        ]


        print(
            "\n"
            + "-" * 70
        )


        print(
            "QID:",
            row[
                "qid"
            ],
        )

        print(
            "Move:",
            row[
                "teacher_move"
            ],
        )


        print(
            "\nProblem:"
        )

        print(
            row[
                "problem"
            ]
        )


        print(
            "\nTeacher:"
        )

        print(
            row[
                "teacher_text"
            ]
        )


        print(
            "\nACTUAL NEXT STUDENT RESPONSE:"
        )

        print(
            row[
                "student_reply"
            ]
        )


        for i, item in enumerate(
            example[
                "retrieved"
            ],
            start=1,
        ):

            candidate = item[
                "row"
            ]


            print(
                f"\nRETRIEVED NEGATIVE {i}"
            )

            print(
                "Similarity:",
                f"{item['similarity']:.4f}",
            )

            print(
                "Candidate QID:",
                candidate[
                    "qid"
                ],
            )

            print(
                "Candidate problem:"
            )

            print(
                candidate[
                    "problem"
                ]
            )

            print(
                "Candidate student response:"
            )

            print(
                candidate[
                    "student_reply"
                ]
            )


print_examples(
    "SI TRAIN",
    train_result[
        "examples"
    ],
)


print_examples(
    "SI VALIDATION",
    val_result[
        "examples"
    ],
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RETRIEVAL VALIDATION"
)

print(
    "=" * 70
)


minimum_required_coverage = 0.95


passed = (

    train_result[
        "coverage"
    ]
    >= minimum_required_coverage

    and

    val_result[
        "coverage"
    ]
    >= minimum_required_coverage
)


print(
    "\nRequired coverage:",
    f"{minimum_required_coverage * 100:.0f}%",
)


print(
    "Train coverage:",
    f"{train_result['coverage'] * 100:.2f}%",
)


print(
    "Validation coverage:",
    f"{val_result['coverage'] * 100:.2f}%",
)


if passed:

    print(
        "\nSTRUCTURAL VALIDATION: PASSED"
    )

    print(
        "Manual inspection of retrieved "
        "negative examples is still required."
    )

else:

    print(
        "\nSTRUCTURAL VALIDATION: FAILED"
    )

    print(
        "Do not construct/train retrieval "
        "contrastive data yet."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "experiment":
        "SI3-D_retrieval_hard_negative_feasibility",

    "retrieval_input":
        "problem + recent_history + move + teacher_text",

    "student_reply_used_for_retrieval":
        False,

    "tfidf_fit":
        "SI_train_only",

    "candidate_constraints": {

        "same_move":
            True,

        "different_qid":
            True,

        "different_group":
            True,

        "different_source":
            True,

        "different_response_text":
            True,

        "response_length_min":
            LENGTH_MIN,

        "response_length_max":
            LENGTH_MAX,
    },

    "train": {

        key: value

        for key, value
        in train_result.items()

        if key != "examples"
    },

    "validation": {

        key: value

        for key, value
        in val_result.items()

        if key != "examples"
    },

    "structural_validation_passed":
        bool(
            passed
        ),
}


RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with RESULT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\nSaved:"
)

print(
    RESULT_PATH
)
