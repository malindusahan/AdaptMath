import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND_ROOT),
)


from app.agents.evaluator.evaluator_agent import (  # noqa: E402
    EvaluatorAgent,
)
from app.schemas.evaluator import (  # noqa: E402
    EvaluatorInput,
    StudentAnswer,
)
from app.schemas.assessment import (  # noqa: E402
    AssessmentQuestion,
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

REPEATS = 3

# Small pause between hosted-model calls to reduce pressure
# on development-tier provider rate limits.
INTER_CALL_DELAY_SECONDS = 3.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "evaluator_evaluation_results.json"
)


# ============================================================
# OBJECTIVELY LABELLED TEST CASES
#
# 16 cases:
# - 8 correct
# - 8 incorrect
#
# Includes:
# - exact correctness
# - semantic equivalence
# - algebra
# - fractions
# - percentages
# - geometry
# - representation/instruction compliance
# - common mathematical errors
# ============================================================

TEST_CASES: list[dict[str, Any]] = [
    # --------------------------------------------------------
    # ARITHMETIC
    # --------------------------------------------------------
    {
        "case_id": "arithmetic_correct",
        "category": "arithmetic",
        "original_question": "What is 7 + 5?",
        "topic": "arithmetic",
        "subtopic": "addition",
        "assessment_question": "What is 8 + 6?",
        "expected_answer": "14",
        "student_answer": "14",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "arithmetic_wrong",
        "category": "arithmetic",
        "original_question": "What is 7 + 5?",
        "topic": "arithmetic",
        "subtopic": "addition",
        "assessment_question": "What is 8 + 6?",
        "expected_answer": "14",
        "student_answer": "13",
        "expected_is_correct": False,
        "expected_error_concept": (
            "Incorrect arithmetic result."
        ),
    },

    # --------------------------------------------------------
    # FRACTIONS
    # --------------------------------------------------------
    {
        "case_id": "fraction_equivalent_correct",
        "category": "semantic_equivalence",
        "original_question": (
            "Give a fraction equivalent to 1/2."
        ),
        "topic": "fractions",
        "subtopic": "equivalent fractions",
        "assessment_question": (
            "Give a fraction equivalent to 1/2."
        ),
        "expected_answer": "1/2",
        "student_answer": "2/4",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "fraction_not_equivalent",
        "category": "fractions",
        "original_question": (
            "Give a fraction equivalent to 1/2."
        ),
        "topic": "fractions",
        "subtopic": "equivalent fractions",
        "assessment_question": (
            "Give a fraction equivalent to 1/2."
        ),
        "expected_answer": "1/2",
        "student_answer": "3/5",
        "expected_is_correct": False,
        "expected_error_concept": (
            "The fraction is not equivalent to 1/2."
        ),
    },

    # --------------------------------------------------------
    # NUMERIC SEMANTIC EQUIVALENCE
    # --------------------------------------------------------
    {
        "case_id": "numeric_equivalent_form_correct",
        "category": "semantic_equivalence",
        "original_question": (
            "Give a number equal to one half."
        ),
        "topic": "fractions",
        "subtopic": "equivalent representations",
        "assessment_question": (
            "Give any numerical value equal to one half."
        ),
        "expected_answer": "0.5",
        "student_answer": "1/2",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },

    # --------------------------------------------------------
    # REPRESENTATION / TASK COMPLIANCE
    # --------------------------------------------------------
    {
        "case_id": "decimal_format_not_followed",
        "category": "instruction_compliance",
        "original_question": (
            "Convert one half to a decimal."
        ),
        "topic": "fractions",
        "subtopic": "fraction to decimal",
        "assessment_question": (
            "Write 1/2 as a decimal."
        ),
        "expected_answer": "0.5",
        "student_answer": "1/2",
        "expected_is_correct": False,
        "expected_error_concept": (
            "The value is mathematically equivalent "
            "but was not written in the requested "
            "decimal form."
        ),
    },

    # --------------------------------------------------------
    # LINEAR EQUATIONS
    # --------------------------------------------------------
    {
        "case_id": "linear_equation_correct",
        "category": "algebra",
        "original_question": (
            "Solve 2x + 6 = 14."
        ),
        "topic": "algebra",
        "subtopic": "linear equations",
        "assessment_question": (
            "Solve 3x + 9 = 24."
        ),
        "expected_answer": "x = 5",
        "student_answer": "5",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "linear_equation_wrong",
        "category": "algebra",
        "original_question": (
            "Solve 2x + 6 = 14."
        ),
        "topic": "algebra",
        "subtopic": "linear equations",
        "assessment_question": (
            "Solve 3x + 9 = 24."
        ),
        "expected_answer": "x = 5",
        "student_answer": "x = 11",
        "expected_is_correct": False,
        "expected_error_concept": (
            "Incorrect solution of the linear equation."
        ),
    },

    # --------------------------------------------------------
    # ALGEBRAIC EXPANSION
    # --------------------------------------------------------
    {
        "case_id": "expansion_correct",
        "category": "algebra",
        "original_question": (
            "Expand 2(x + 3)."
        ),
        "topic": "algebra",
        "subtopic": "expansion",
        "assessment_question": (
            "Expand 3(x + 4)."
        ),
        "expected_answer": "3x + 12",
        "student_answer": "3x + 12",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "expansion_not_performed",
        "category": "instruction_compliance",
        "original_question": (
            "Expand 2(x + 3)."
        ),
        "topic": "algebra",
        "subtopic": "expansion",
        "assessment_question": (
            "Expand 3(x + 4)."
        ),
        "expected_answer": "3x + 12",
        "student_answer": "3(x + 4)",
        "expected_is_correct": False,
        "expected_error_concept": (
            "The expression is equivalent but has "
            "not been expanded as requested."
        ),
    },
    {
        "case_id": "expansion_sign_error",
        "category": "algebra_error",
        "original_question": (
            "Expand 2(x - 3)."
        ),
        "topic": "algebra",
        "subtopic": "expansion",
        "assessment_question": (
            "Expand 4(x - 2)."
        ),
        "expected_answer": "4x - 8",
        "student_answer": "4x + 8",
        "expected_is_correct": False,
        "expected_error_concept": (
            "Sign error when distributing across "
            "the subtraction."
        ),
    },

    # --------------------------------------------------------
    # ALGEBRAIC SEMANTIC EQUIVALENCE
    # --------------------------------------------------------
    {
        "case_id": "factorised_equivalent_correct",
        "category": "semantic_equivalence",
        "original_question": (
            "Give an expression equivalent to x^2 - 1."
        ),
        "topic": "algebra",
        "subtopic": "equivalent expressions",
        "assessment_question": (
            "Give any expression mathematically "
            "equivalent to x^2 - 1."
        ),
        "expected_answer": "x^2 - 1",
        "student_answer": "(x - 1)(x + 1)",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },

    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------
    {
        "case_id": "percentage_correct",
        "category": "percentages",
        "original_question": (
            "Find 10% of 100."
        ),
        "topic": "percentages",
        "subtopic": "percentage of an amount",
        "assessment_question": (
            "Find 15% of 200."
        ),
        "expected_answer": "30",
        "student_answer": "30",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "percentage_wrong",
        "category": "percentages",
        "original_question": (
            "Find 10% of 100."
        ),
        "topic": "percentages",
        "subtopic": "percentage of an amount",
        "assessment_question": (
            "Find 15% of 200."
        ),
        "expected_answer": "30",
        "student_answer": "35",
        "expected_is_correct": False,
        "expected_error_concept": (
            "Incorrect calculation of the percentage."
        ),
    },

    # --------------------------------------------------------
    # GEOMETRY / UNITS
    # --------------------------------------------------------
    {
        "case_id": "area_with_units_correct",
        "category": "geometry",
        "original_question": (
            "Find the area of a 3 cm by 4 cm rectangle."
        ),
        "topic": "geometry",
        "subtopic": "area",
        "assessment_question": (
            "A rectangle is 5 cm long and 4 cm wide. "
            "Give its area including units."
        ),
        "expected_answer": "20 cm^2",
        "student_answer": "20 cm²",
        "expected_is_correct": True,
        "expected_error_concept": None,
    },
    {
        "case_id": "area_wrong_units",
        "category": "instruction_compliance",
        "original_question": (
            "Find the area of a 3 cm by 4 cm rectangle."
        ),
        "topic": "geometry",
        "subtopic": "area",
        "assessment_question": (
            "A rectangle is 5 cm long and 4 cm wide. "
            "Give its area including units."
        ),
        "expected_answer": "20 cm^2",
        "student_answer": "20 cm",
        "expected_is_correct": False,
        "expected_error_concept": (
            "Area requires square centimetres rather "
            "than centimetres."
        ),
    },
]


# ============================================================
# BUILD INPUT
# ============================================================

def build_input(
    case: dict[str, Any],
) -> EvaluatorInput:
    question = AssessmentQuestion(
        question_id="q1",
        question=case[
            "assessment_question"
        ],
        expected_answer=case[
            "expected_answer"
        ],
    )

    answer = StudentAnswer(
        question_id="q1",
        answer=case[
            "student_answer"
        ],
    )

    return EvaluatorInput(
        original_question=case[
            "original_question"
        ],
        topic=case["topic"],
        subtopic=case[
            "subtopic"
        ],
        assessment_questions=[
            question
        ],
        student_answers=[
            answer
        ],
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    95% Wilson score interval for a binomial proportion.
    """

    if total == 0:
        return 0.0, 0.0

    proportion = successes / total

    denominator = (
        1
        + (z ** 2 / total)
    )

    centre = (
        proportion
        + (z ** 2 / (2 * total))
    )

    adjustment = (
        z
        * math.sqrt(
            (
                proportion
                * (1 - proportion)
                / total
            )
            + (
                z ** 2
                / (4 * total ** 2)
            )
        )
    )

    lower = (
        centre - adjustment
    ) / denominator

    upper = (
        centre + adjustment
    ) / denominator

    return lower, upper


def majority_boolean(
    values: list[bool],
) -> bool:
    true_count = sum(
        1
        for value in values
        if value
    )

    return (
        true_count
        > len(values) / 2
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    evaluator = EvaluatorAgent()

    case_results: list[
        dict[str, Any]
    ] = []

    all_latencies: list[float] = []

    print(
        "\n=== AdaptMath Evaluator "
        "Formal Evaluation ===\n"
    )

    print(
        "Unique benchmark cases:",
        len(TEST_CASES),
    )

    print(
        "Repeats per case:",
        REPEATS,
    )

    print(
        "Total evaluator calls:",
        len(TEST_CASES) * REPEATS,
    )

    print()

    # ========================================================
    # CASE LOOP
    # ========================================================

    for case_index, case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(
            "===================================="
        )

        print(
            f"Case {case_index}/{len(TEST_CASES)}:"
        )

        print(
            case["case_id"]
        )

        print(
            "Expected correctness:",
            case["expected_is_correct"],
        )

        run_results: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # REPEATED JUDGMENTS
        # ====================================================

        for repeat_index in range(
            1,
            REPEATS + 1,
        ):
            evaluator_input = build_input(
                case
            )

            start_time = (
                time.perf_counter()
            )

            output = evaluator.evaluate(
                evaluator_input
            )

            latency_seconds = (
                time.perf_counter()
                - start_time
            )

            all_latencies.append(
                latency_seconds
            )

            evaluated_answers = (
                list(
                    output.correct_answers
                )
                + list(
                    output.wrong_answers
                )
            )

            if len(
                evaluated_answers
            ) != 1:
                raise RuntimeError(
                    "Expected exactly one evaluated "
                    "answer for this benchmark case."
                )

            evaluated = (
                evaluated_answers[0]
            )

            predicted_is_correct = (
                evaluated.is_correct
            )

            classification_matches_gold = (
                predicted_is_correct
                == case[
                    "expected_is_correct"
                ]
            )

            expected_reteaching_from_prediction = (
                not predicted_is_correct
            )

            policy_consistent = (
                output.needs_reteaching
                == expected_reteaching_from_prediction
            )

            run_result = {
                "repeat":
                    repeat_index,

                "predicted_is_correct":
                    predicted_is_correct,

                "classification_matches_gold":
                    classification_matches_gold,

                "needs_reteaching":
                    output.needs_reteaching,

                "policy_consistent":
                    policy_consistent,

                "feedback":
                    evaluated.feedback,

                "identified_error":
                    evaluated.identified_error,

                "overall_feedback":
                    output.overall_feedback,

                "latency_seconds":
                    latency_seconds,
            }

            run_results.append(
                run_result
            )

            print(
                f"  Run {repeat_index}:",
                (
                    "correct"
                    if predicted_is_correct
                    else "incorrect"
                ),
                "| gold match:",
                classification_matches_gold,
                "| reteach:",
                output.needs_reteaching,
                "| latency:",
                round(
                    latency_seconds,
                    2,
                ),
                "s",
            )

            is_last_call = (
                case_index
                == len(TEST_CASES)
                and repeat_index
                == REPEATS
            )

            if not is_last_call:
                time.sleep(
                    INTER_CALL_DELAY_SECONDS
                )

        # ====================================================
        # CASE-LEVEL AGGREGATION
        # ====================================================

        predictions = [
            run[
                "predicted_is_correct"
            ]
            for run
            in run_results
        ]

        majority_prediction = (
            majority_boolean(
                predictions
            )
        )

        majority_matches_gold = (
            majority_prediction
            == case[
                "expected_is_correct"
            ]
        )

        unanimous = (
            len(
                set(
                    predictions
                )
            )
            == 1
        )

        diagnosis_present_runs = sum(
            1
            for run
            in run_results
            if run[
                "identified_error"
            ]
        )

        case_result = {
            "case_id":
                case["case_id"],

            "category":
                case["category"],

            "assessment_question":
                case[
                    "assessment_question"
                ],

            "expected_answer":
                case[
                    "expected_answer"
                ],

            "student_answer":
                case[
                    "student_answer"
                ],

            "expected_is_correct":
                case[
                    "expected_is_correct"
                ],

            "expected_error_concept":
                case[
                    "expected_error_concept"
                ],

            "majority_predicted_is_correct":
                majority_prediction,

            "majority_matches_gold":
                majority_matches_gold,

            "repeat_predictions":
                predictions,

            "repeat_unanimous":
                unanimous,

            "diagnosis_present_runs":
                diagnosis_present_runs,

            "runs":
                run_results,
        }

        case_results.append(
            case_result
        )

        print(
            "  Majority prediction:",
            majority_prediction,
        )

        print(
            "  Majority matches gold:",
            majority_matches_gold,
        )

        print(
            "  Repeats unanimous:",
            unanimous,
        )

        if not case[
            "expected_is_correct"
        ]:
            print(
                "  Expected error concept:",
                case[
                    "expected_error_concept"
                ],
            )

            print(
                "  Model diagnosis:",
                run_results[-1][
                    "identified_error"
                ],
            )

        print()

    # ========================================================
    # CASE-LEVEL CLASSIFICATION METRICS
    #
    # Positive class = INCORRECT answer.
    #
    # This is the safety-critical direction for adaptation,
    # because failing to detect an incorrect answer can prevent
    # necessary reteaching.
    # ========================================================

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for result in case_results:
        gold_incorrect = (
            not result[
                "expected_is_correct"
            ]
        )

        predicted_incorrect = (
            not result[
                "majority_predicted_is_correct"
            ]
        )

        if (
            gold_incorrect
            and predicted_incorrect
        ):
            true_positive += 1

        elif (
            not gold_incorrect
            and predicted_incorrect
        ):
            false_positive += 1

        elif (
            gold_incorrect
            and not predicted_incorrect
        ):
            false_negative += 1

        else:
            true_negative += 1

    total_cases = len(
        case_results
    )

    correct_case_classifications = sum(
        1
        for result
        in case_results
        if result[
            "majority_matches_gold"
        ]
    )

    accuracy = safe_divide(
        correct_case_classifications,
        total_cases,
    )

    incorrect_precision = safe_divide(
        true_positive,
        true_positive
        + false_positive,
    )

    incorrect_recall = safe_divide(
        true_positive,
        true_positive
        + false_negative,
    )

    incorrect_f1 = safe_divide(
        2
        * incorrect_precision
        * incorrect_recall,
        incorrect_precision
        + incorrect_recall,
    )

    accuracy_ci_lower, accuracy_ci_upper = (
        wilson_interval(
            correct_case_classifications,
            total_cases,
        )
    )

    # ========================================================
    # RELIABILITY
    # ========================================================

    unanimous_cases = sum(
        1
        for result
        in case_results
        if result[
            "repeat_unanimous"
        ]
    )

    repeat_unanimity_rate = safe_divide(
        unanimous_cases,
        total_cases,
    )

    all_runs = [
        run
        for case_result
        in case_results
        for run
        in case_result[
            "runs"
        ]
    ]

    policy_consistency_rate = (
        safe_divide(
            sum(
                1
                for run
                in all_runs
                if run[
                    "policy_consistent"
                ]
            ),
            len(all_runs),
        )
    )

    # ========================================================
    # ERROR-DIAGNOSIS PRESENCE
    #
    # IMPORTANT:
    # This measures whether a diagnosis was produced.
    # It does NOT claim that the diagnosis was correct.
    # Diagnosis correctness should be manually audited
    # against expected_error_concept.
    # ========================================================

    wrong_gold_runs = [
        run
        for case_result
        in case_results
        if not case_result[
            "expected_is_correct"
        ]
        for run
        in case_result[
            "runs"
        ]
    ]

    diagnosis_presence_rate = (
        safe_divide(
            sum(
                1
                for run
                in wrong_gold_runs
                if run[
                    "identified_error"
                ]
            ),
            len(wrong_gold_runs),
        )
    )

    # ========================================================
    # LATENCY
    # ========================================================

    mean_latency = (
        statistics.mean(
            all_latencies
        )
    )

    median_latency = (
        statistics.median(
            all_latencies
        )
    )

    max_latency = max(
        all_latencies
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "unique_cases":
            total_cases,

        "correct_gold_cases":
            sum(
                1
                for case
                in TEST_CASES
                if case[
                    "expected_is_correct"
                ]
            ),

        "incorrect_gold_cases":
            sum(
                1
                for case
                in TEST_CASES
                if not case[
                    "expected_is_correct"
                ]
            ),

        "repeats_per_case":
            REPEATS,

        "total_evaluator_calls":
            len(all_runs),

        "majority_vote_accuracy":
            accuracy,

        "majority_vote_accuracy_95_percent_wilson_ci": {
            "lower":
                accuracy_ci_lower,

            "upper":
                accuracy_ci_upper,
        },

        "incorrect_answer_precision":
            incorrect_precision,

        "incorrect_answer_recall":
            incorrect_recall,

        "incorrect_answer_f1":
            incorrect_f1,

        "confusion_matrix_incorrect_as_positive": {
            "true_positive":
                true_positive,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "true_negative":
                true_negative,
        },

        "repeat_unanimity_rate":
            repeat_unanimity_rate,

        "reteaching_policy_consistency_rate":
            policy_consistency_rate,

        "wrong_gold_runs_with_diagnosis_presence_rate":
            diagnosis_presence_rate,

        "diagnosis_metric_note": (
            "Diagnosis presence is not diagnosis "
            "accuracy. identified_error outputs must "
            "be manually compared with each case's "
            "expected_error_concept."
        ),

        "mean_latency_seconds":
            mean_latency,

        "median_latency_seconds":
            median_latency,

        "max_latency_seconds":
            max_latency,
    }

    evaluation = {
        "experiment":
            "AdaptMath Evaluator targeted benchmark",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "methodology": {
            "description": (
                "Balanced objectively labelled mathematics "
                "benchmark with repeated GenAI evaluator "
                "judgments."
            ),

            "majority_vote_unit":
                "unique benchmark case",

            "positive_class_for_precision_recall":
                "incorrect student answer",

            "repeat_count":
                REPEATS,

            "limitations": [
                (
                    "This is a small targeted benchmark "
                    "and does not establish universal "
                    "evaluator accuracy."
                ),
                (
                    "Error-diagnosis correctness requires "
                    "manual review."
                ),
                (
                    "Hosted-model behaviour may change "
                    "across provider/model versions."
                ),
            ],
        },

        "summary":
            summary,

        "cases":
            case_results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "===================================="
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "===================================="
    )

    print(
        "Unique cases:",
        total_cases,
    )

    print(
        "Correct / incorrect gold:",
        (
            summary[
                "correct_gold_cases"
            ]
        ),
        "/",
        (
            summary[
                "incorrect_gold_cases"
            ]
        ),
    )

    print(
        "Total evaluator calls:",
        len(all_runs),
    )

    print()

    print(
        "Majority-vote accuracy:",
        round(
            accuracy,
            3,
        ),
    )

    print(
        "95% Wilson CI:",
        (
            round(
                accuracy_ci_lower,
                3,
            ),
            round(
                accuracy_ci_upper,
                3,
            ),
        ),
    )

    print(
        "Incorrect-answer precision:",
        round(
            incorrect_precision,
            3,
        ),
    )

    print(
        "Incorrect-answer recall:",
        round(
            incorrect_recall,
            3,
        ),
    )

    print(
        "Incorrect-answer F1:",
        round(
            incorrect_f1,
            3,
        ),
    )

    print()

    print(
        "Repeat unanimity:",
        round(
            repeat_unanimity_rate,
            3,
        ),
    )

    print(
        "Reteaching policy consistency:",
        round(
            policy_consistency_rate,
            3,
        ),
    )

    print(
        "Wrong-answer diagnosis presence:",
        round(
            diagnosis_presence_rate,
            3,
        ),
    )

    print()

    print(
        "Mean latency:",
        round(
            mean_latency,
            2,
        ),
        "seconds",
    )

    print(
        "Median latency:",
        round(
            median_latency,
            2,
        ),
        "seconds",
    )

    print(
        "Maximum latency:",
        round(
            max_latency,
            2,
        ),
        "seconds",
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "IMPORTANT: manually review identified_error "
        "against expected_error_concept before reporting "
        "diagnosis accuracy."
    )


if __name__ == "__main__":
    main()