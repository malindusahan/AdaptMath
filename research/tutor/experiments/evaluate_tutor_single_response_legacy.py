import json
import sys
import time
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


from app.agents.tutor.tutor_agent import TutorAgent  # noqa: E402
from app.schemas.tutor import TutorInput  # noqa: E402


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "tutor_evaluation_results.json"
)


# ============================================================
# DIVERSE MATHEMATICS BENCHMARK
# ============================================================

TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "arithmetic_percentage",
        "category": "arithmetic",
        "question": "What is 15% of 240?",
        "topic": "percentage",
        "subtopic": "percentage of a quantity",
        "age": 12,
        "complexity_score": 0.10,
        "reference_answer": "36",
        "reference_notes": (
            "15% of 240 = 0.15 × 240 = 36."
        ),
    },
    {
        "case_id": "linear_equation",
        "category": "algebra",
        "question": "Solve 3x + 5 = 20.",
        "topic": "algebra",
        "subtopic": "linear equations",
        "age": 14,
        "complexity_score": 0.27,
        "reference_answer": "x = 5",
        "reference_notes": (
            "Subtract 5 to obtain 3x = 15, "
            "then divide by 3."
        ),
    },
    {
        "case_id": "fraction_addition",
        "category": "fractions",
        "question": "Calculate 3/4 + 1/2.",
        "topic": "fractions",
        "subtopic": "fraction addition",
        "age": 10,
        "complexity_score": 0.15,
        "reference_answer": "5/4 or 1 1/4",
        "reference_notes": (
            "1/2 = 2/4, therefore "
            "3/4 + 2/4 = 5/4."
        ),
    },
    {
        "case_id": "geometry_triangle",
        "category": "geometry",
        "question": (
            "An isosceles triangle has two equal sides "
            "of length 5 cm and a base of length 6 cm. "
            "Find its area."
        ),
        "topic": "geometry",
        "subtopic": "triangle area",
        "age": 14,
        "complexity_score": 0.50,
        "reference_answer": "12 cm^2",
        "reference_notes": (
            "The altitude bisects the base into lengths 3. "
            "Height = sqrt(5^2 - 3^2) = 4. "
            "Area = 1/2 × 6 × 4 = 12."
        ),
    },
    {
        "case_id": "compound_interest",
        "category": "word_problem",
        "question": (
            "A bank offers 3.2% compound interest per year. "
            "Sarah invests £4,500 for 6 years. "
            "Calculate the total interest earned."
        ),
        "topic": "percentage",
        "subtopic": "compound interest",
        "age": 16,
        "complexity_score": 0.55,
        "reference_answer": (
    "Approximately £936.14 interest"
),
        "reference_notes": (
            "Use 4500(1.032)^6 - 4500. "
            "Reference value should be independently "
            "rechecked before final reporting."
        ),
    },
    {
        "case_id": "commuting_cubics",
        "category": "symbolic_reasoning",
        "question": (
            "Let p(x) and q(x) be two cubic polynomials "
            "such that p(0) = -24, q(0) = 30, and "
            "p(q(x)) = q(p(x)) for all real x. "
            "Find (p(3), q(6))."
        ),
        "topic": "algebra",
        "subtopic": "polynomials",
        "age": 17,
        "complexity_score": 0.83401269,
        "reference_answer": "(3, -24)",
        "reference_notes": (
            "Independently verified using the complete "
            "coefficient constraints and Groebner "
            "reduction in SymPy."
        ),
    },
]


# ============================================================
# MANUAL REVIEW RUBRIC
# ============================================================

REVIEW_RUBRIC = {
    "mathematical_correctness": (
        "Does the teaching response reach and explain "
        "the mathematically correct result?"
    ),
    "reasoning_validity": (
        "Are the mathematical steps logically valid, "
        "without invented assumptions?"
    ),
    "age_appropriateness": (
        "Is the explanation appropriate for the supplied "
        "student age?"
    ),
    "instructional_clarity": (
        "Is the explanation understandable and structured?"
    ),
    "assessment_quality": (
        "Are the generated assessment questions relevant "
        "to the concept just taught?"
    ),
    "assessment_answer_correctness": (
        "Are the internal expected answers for generated "
        "assessment questions mathematically correct?"
    ),
}


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    tutor = TutorAgent()

    results: list[dict[str, Any]] = []

    print(
        "\n=== AdaptMath Tutor Evaluation ===\n"
    )

    for case in TEST_CASES:
        print(
            f"Case: {case['case_id']}"
        )

        tutor_input = TutorInput(
            question=case["question"],
            topic=case["topic"],
            subtopic=case["subtopic"],
            student_age=case["age"],
            complexity_score=case[
                "complexity_score"
            ],
            planner_output=None,
            teaching_strategy=None,
            previous_errors=[],
            reteaching=False,
        )

        start_time = time.perf_counter()

        try:
            output = tutor.teach(
                tutor_input
            )

            latency_seconds = (
                time.perf_counter()
                - start_time
            )

            result = {
                "case_id":
                    case["case_id"],

                "category":
                    case["category"],

                "question":
                    case["question"],

                "student_age":
                    case["age"],

                "complexity_score":
                    case["complexity_score"],

                "reference_answer":
                    case["reference_answer"],

                "reference_notes":
                    case["reference_notes"],

                "teaching_response":
                    output.teaching_response,

                "assessment_questions": [
                    question.model_dump()
                    for question
                    in output.assessment_questions
                ],

                "number_of_assessment_questions":
                    len(
                        output.assessment_questions
                    ),

                "latency_seconds":
                    latency_seconds,

                "execution_success":
                    True,

                "error":
                    None,

                # These are intentionally left unscored.
                # They must be filled only after independent
                # review against the rubric.
                "manual_review": {
                    "mathematical_correctness":
                        None,

                    "reasoning_validity":
                        None,

                    "age_appropriateness":
                        None,

                    "instructional_clarity":
                        None,

                    "assessment_quality":
                        None,

                    "assessment_answer_correctness":
                        None,

                    "review_notes":
                        None,
                },
            }

            print(
                "  Completed in:",
                round(
                    latency_seconds,
                    2,
                ),
                "seconds",
            )

            print(
                "  Assessment questions:",
                len(
                    output.assessment_questions
                ),
            )

        except Exception as exc:
            latency_seconds = (
                time.perf_counter()
                - start_time
            )

            result = {
                "case_id":
                    case["case_id"],

                "category":
                    case["category"],

                "question":
                    case["question"],

                "student_age":
                    case["age"],

                "complexity_score":
                    case["complexity_score"],

                "reference_answer":
                    case["reference_answer"],

                "reference_notes":
                    case["reference_notes"],

                "teaching_response":
                    None,

                "assessment_questions":
                    [],

                "number_of_assessment_questions":
                    0,

                "latency_seconds":
                    latency_seconds,

                "execution_success":
                    False,

                "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                "manual_review":
                    None,
            }

            print(
                "  FAILED:",
                type(exc).__name__,
            )

        results.append(
            result
        )

        print()

    # ========================================================
    # AUTOMATIC EXECUTION METRICS
    # ========================================================

    successful = sum(
        1
        for result in results
        if result[
            "execution_success"
        ]
    )

    successful_latencies = [
        result[
            "latency_seconds"
        ]
        for result in results
        if result[
            "execution_success"
        ]
    ]

    mean_latency = (
        sum(
            successful_latencies
        )
        / len(
            successful_latencies
        )
        if successful_latencies
        else None
    )

    summary = {
        "number_of_cases":
            len(results),

        "execution_success_rate":
            (
                successful
                / len(results)
            ),

        "mean_latency_seconds":
            mean_latency,

        "mathematical_accuracy":
            None,

        "note": (
            "Mathematical accuracy is intentionally "
            "not calculated automatically. Tutor outputs "
            "must first be independently reviewed against "
            "the supplied reference answers and rubric."
        ),
    }

    evaluation = {
        "rubric":
            REVIEW_RUBRIC,

        "summary":
            summary,

        "cases":
            results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Execution Summary ==="
    )

    print(
        "Execution success rate:",
        round(
            summary[
                "execution_success_rate"
            ],
            3,
        ),
    )

    if mean_latency is not None:
        print(
            "Mean latency:",
            round(
                mean_latency,
                2,
            ),
            "seconds",
        )

    print(
        "Mathematical accuracy: "
        "pending independent review"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()