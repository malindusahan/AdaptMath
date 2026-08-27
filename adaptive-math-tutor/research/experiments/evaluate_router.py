import json
import math
import statistics
import sys
import time
from collections import Counter
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


from app.agents.complexity.service import (  # noqa: E402
    get_complexity_service,
)
from app.agents.router.router_agent import (  # noqa: E402
    RouterAgent,
)
from app.core.config import get_settings  # noqa: E402
from app.schemas.memory import (  # noqa: E402
    LearnerHistoryItem,
    MemoryContext,
)
from app.schemas.profile import StudentProfile  # noqa: E402
from app.schemas.router import RouterInput  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

REPEATS_PER_CASE = 3

INTER_CALL_DELAY_SECONDS = 2.0

OUTPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "router_evaluation_results_v5.json"
)

VALID_ROUTES = {
    "direct_tutor",
    "planned_tutor",
}

# This is pre-tutoring workflow depth, not a gold correctness label.
# Assessment is universal after Tutor and therefore is not a routing level.
WORKFLOW_DEPTH = {
    "direct_tutor": 1,
    "planned_tutor": 2,
}


# ============================================================
# CONTROLLED TEST CASES
# ============================================================

TEST_CASES: list[dict[str, Any]] = [
    # ========================================================
    # PROBLEM SENSITIVITY
    #
    # Learner context is deliberately held constant:
    # - same age
    # - no learner history
    # - no previous errors
    # - no previous strategies
    #
    # Only the mathematical problem/topic changes.
    # Complexity scores come from the trained ML model.
    # ========================================================

    {
        "case_id": "problem_arithmetic",
        "group": "problem_sensitivity",
        "question": "What is 7 + 5?",
        "age": 16,
        "topic": "arithmetic",
        "subtopic": "addition",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "problem_fraction",
        "group": "problem_sensitivity",
        "question": "What is 3/4 + 1/2?",
        "age": 16,
        "topic": "fractions",
        "subtopic": "fraction addition",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "problem_linear_equation",
        "group": "problem_sensitivity",
        "question": "Solve 3x + 5 = 20.",
        "age": 16,
        "topic": "algebra",
        "subtopic": "linear equations",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "problem_geometry",
        "group": "problem_sensitivity",
        "question": (
            "An isosceles triangle has equal sides of "
            "5 cm and a base of 6 cm. Find its area."
        ),
        "age": 16,
        "topic": "geometry",
        "subtopic": "triangle area",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "problem_compound_interest",
        "group": "problem_sensitivity",
        "question": (
            "A bank offers 3.2% compound interest per "
            "year. Sarah invests GBP 4,500 for 6 years. "
            "Calculate the total interest earned."
        ),
        "age": 16,
        "topic": "percentage",
        "subtopic": "compound interest",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "problem_complex_polynomial",
        "group": "problem_sensitivity",
        "question": (
            "Let p(x) and q(x) be two cubic polynomials "
            "such that p(0) = -24, q(0) = 30, and "
            "p(q(x)) = q(p(x)) for all real x. "
            "Find (p(3), q(6))."
        ),
        "age": 16,
        "topic": "algebra",
        "subtopic": "polynomials",
        "history": [],
        "previous_errors": [],
        "previous_strategies": [],
    },

    # ========================================================
    # LEARNER-CONTEXT PAIR 1
    #
    # Same:
    # - question
    # - age
    # - topic
    # - ML complexity score
    #
    # Different:
    # - learner history/errors
    # ========================================================

    {
        "case_id": "linear_strong",
        "group": "learner_context_linear",
        "learner_condition": "strong",
        "question": "Solve 3x + 5 = 20.",
        "age": 15,
        "topic": "algebra",
        "subtopic": "linear equations",
        "history": [
            {
                "topic": "algebra",
                "event_type": "successful_attempt",
                "details": {
                    "description": (
                        "Solved several similar linear "
                        "equations correctly without help."
                    )
                },
            }
        ],
        "previous_errors": [],
        "previous_strategies": [
            "Step-by-step explanation"
        ],
    },
    {
        "case_id": "linear_struggling",
        "group": "learner_context_linear",
        "learner_condition": "struggling",
        "question": "Solve 3x + 5 = 20.",
        "age": 15,
        "topic": "algebra",
        "subtopic": "linear equations",
        "history": [
            {
                "topic": "algebra",
                "event_type": "difficulty",
                "details": {
                    "description": (
                        "Repeatedly struggled with similar "
                        "linear equations and required "
                        "re-explanation."
                    )
                },
            }
        ],
        "previous_errors": [
            "Incorrect use of inverse operations",
            "Stopped before isolating the variable",
        ],
        "previous_strategies": [
            "Step-by-step explanation",
            "Worked example",
        ],
    },

    # ========================================================
    # LEARNER-CONTEXT PAIR 2
    # ========================================================

    {
        "case_id": "fraction_strong",
        "group": "learner_context_fraction",
        "learner_condition": "strong",
        "question": "What is 3/4 + 1/2?",
        "age": 13,
        "topic": "fractions",
        "subtopic": "fraction addition",
        "history": [
            {
                "topic": "fractions",
                "event_type": "successful_attempt",
                "details": {
                    "description": (
                        "Consistently added fractions with "
                        "different denominators correctly."
                    )
                },
            }
        ],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "fraction_struggling",
        "group": "learner_context_fraction",
        "learner_condition": "struggling",
        "question": "What is 3/4 + 1/2?",
        "age": 13,
        "topic": "fractions",
        "subtopic": "fraction addition",
        "history": [
            {
                "topic": "fractions",
                "event_type": "difficulty",
                "details": {
                    "description": (
                        "Has repeatedly struggled to find "
                        "common denominators when adding "
                        "fractions."
                    )
                },
            }
        ],
        "previous_errors": [
            (
                "Added denominators directly instead of "
                "finding a common denominator"
            )
        ],
        "previous_strategies": [
            "Visual fraction explanation"
        ],
    },

    # ========================================================
    # LEARNER-CONTEXT PAIR 3
    # ========================================================

    {
        "case_id": "percentage_strong",
        "group": "learner_context_percentage",
        "learner_condition": "strong",
        "question": (
            "A bank offers 3.2% compound interest per "
            "year. Sarah invests GBP 4,500 for 6 years. "
            "Calculate the total interest earned."
        ),
        "age": 16,
        "topic": "percentage",
        "subtopic": "compound interest",
        "history": [
            {
                "topic": "percentage",
                "event_type": "successful_attempt",
                "details": {
                    "description": (
                        "Previously solved percentage and "
                        "compound-interest problems correctly."
                    )
                },
            }
        ],
        "previous_errors": [],
        "previous_strategies": [],
    },
    {
        "case_id": "percentage_struggling",
        "group": "learner_context_percentage",
        "learner_condition": "struggling",
        "question": (
            "A bank offers 3.2% compound interest per "
            "year. Sarah invests GBP 4,500 for 6 years. "
            "Calculate the total interest earned."
        ),
        "age": 16,
        "topic": "percentage",
        "subtopic": "compound interest",
        "history": [
            {
                "topic": "percentage",
                "event_type": "difficulty",
                "details": {
                    "description": (
                        "Has struggled to distinguish simple "
                        "interest from compound interest."
                    )
                },
            }
        ],
        "previous_errors": [
            (
                "Used a simple-interest calculation for "
                "a compound-interest problem"
            )
        ],
        "previous_strategies": [
            "Formula explanation"
        ],
    },
]


CONTEXT_PAIRS = [
    (
        "linear_strong",
        "linear_struggling",
    ),
    (
        "fraction_strong",
        "fraction_struggling",
    ),
    (
        "percentage_strong",
        "percentage_struggling",
    ),
]


# ============================================================
# INPUT BUILDER
# ============================================================

def build_router_input(
    case: dict[str, Any],
    complexity_score: float,
) -> RouterInput:
    profile = StudentProfile(
        student_id=case["case_id"],
        age=case["age"],
    )

    history = [
        LearnerHistoryItem.model_validate(
            item
        )
        for item in case["history"]
    ]

    memory = MemoryContext(
        student_id=case["case_id"],
        topic=case["topic"],
        subtopic=case["subtopic"],
        relevant_history=history,
        previous_errors=case[
            "previous_errors"
        ],
        previous_strategies=case[
            "previous_strategies"
        ],
    )

    return RouterInput(
        question=case["question"],
        profile=profile,
        memory=memory,
        complexity_score=complexity_score,
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def modal_route_information(
    routes: list[str],
) -> tuple[
    str | None,
    list[str],
    float,
]:
    """
    Return:
    - unique modal route, or None if tied/no output
    - all tied modal routes
    - modal support fraction
    """

    if not routes:
        return None, [], 0.0

    counts = Counter(routes)

    highest_count = max(
        counts.values()
    )

    winners = sorted(
        route
        for route, count
        in counts.items()
        if count == highest_count
    )

    stability = (
        highest_count
        / len(routes)
    )

    if len(winners) == 1:
        return (
            winners[0],
            winners,
            stability,
        )

    return (
        None,
        winners,
        stability,
    )


def average_ranks(
    values: list[float],
) -> list[float]:
    """
    Assign average ranks to tied values.
    Smallest value receives rank 1.
    """

    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [
        0.0
        for _ in values
    ]

    position = 0

    while position < len(indexed):
        end = position

        current_value = indexed[
            position
        ][1]

        while (
            end + 1
            < len(indexed)
            and math.isclose(
                indexed[end + 1][1],
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            end += 1

        average_rank = (
            (position + 1)
            + (end + 1)
        ) / 2

        for rank_position in range(
            position,
            end + 1,
        ):
            original_index = indexed[
                rank_position
            ][0]

            ranks[
                original_index
            ] = average_rank

        position = end + 1

    return ranks


def pearson_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if (
        len(x_values) != len(y_values)
        or len(x_values) < 2
    ):
        return None

    x_mean = statistics.mean(
        x_values
    )

    y_mean = statistics.mean(
        y_values
    )

    numerator = sum(
        (
            x - x_mean
        )
        * (
            y - y_mean
        )
        for x, y
        in zip(
            x_values,
            y_values,
        )
    )

    x_squared = sum(
        (
            x - x_mean
        ) ** 2
        for x
        in x_values
    )

    y_squared = sum(
        (
            y - y_mean
        ) ** 2
        for y
        in y_values
    )

    denominator = math.sqrt(
        x_squared
        * y_squared
    )

    if denominator == 0:
        return None

    return numerator / denominator


def spearman_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if (
        len(x_values) != len(y_values)
        or len(x_values) < 2
    ):
        return None

    x_ranks = average_ranks(
        x_values
    )

    y_ranks = average_ranks(
        y_values
    )

    return pearson_correlation(
        x_ranks,
        y_ranks,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    settings = get_settings()

    complexity_service = (
        get_complexity_service()
    )

    router = RouterAgent()

    results: list[
        dict[str, Any]
    ] = []

    all_call_latencies: list[
        float
    ] = []

    total_calls = (
        len(TEST_CASES)
        * REPEATS_PER_CASE
    )

    completed_calls = 0

    print(
        "\n=== AdaptMath Router "
        "Formal Evaluation ===\n"
    )

    print(
        "Unique cases:",
        len(TEST_CASES),
    )

    print(
        "Repeats per case:",
        REPEATS_PER_CASE,
    )

    print(
        "Total router calls:",
        total_calls,
    )

    print(
        "Model:",
        settings.groq_model,
    )

    print()

    # ========================================================
    # RUN CASES
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

        complexity_score = (
            complexity_service.predict(
                case["question"]
            )
        )

        router_input = (
            build_router_input(
                case,
                complexity_score,
            )
        )

        run_results: list[
            dict[str, Any]
        ] = []

        successful_routes: list[
            str
        ] = []

        reasons: list[
            str
        ] = []

        for repeat_index in range(
            1,
            REPEATS_PER_CASE + 1,
        ):
            start_time = (
                time.perf_counter()
            )

            route = None
            reason = None
            error = None

            try:
                output = router.route(
                    router_input
                )

                route = output.route
                reason = output.reason

            except Exception as exc:
                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            latency_seconds = (
                time.perf_counter()
                - start_time
            )

            all_call_latencies.append(
                latency_seconds
            )

            valid_route = (
                route in VALID_ROUTES
                if route is not None
                else False
            )

            if valid_route:
                successful_routes.append(
                    route
                )

            if reason:
                reasons.append(
                    reason
                )

            run_result = {
                "repeat":
                    repeat_index,

                "route":
                    route,

                "reason":
                    reason,

                "valid_route":
                    valid_route,

                "error":
                    error,

                "latency_seconds":
                    latency_seconds,
            }

            run_results.append(
                run_result
            )

            if error:
                print(
                    f"  Run {repeat_index}: "
                    f"ERROR - {error}"
                )
            else:
                print(
                    f"  Run {repeat_index}: "
                    f"{route}"
                )

            completed_calls += 1

            if completed_calls < total_calls:
                time.sleep(
                    INTER_CALL_DELAY_SECONDS
                )

        (
            modal_route,
            tied_modal_routes,
            stability,
        ) = modal_route_information(
            successful_routes
        )

        unique_modal = (
            modal_route is not None
        )

        case_result = {
            "case_id":
                case["case_id"],

            "group":
                case["group"],

            "learner_condition":
                case.get(
                    "learner_condition"
                ),

            "question":
                case["question"],

            "age":
                case["age"],

            "topic":
                case["topic"],

            "subtopic":
                case["subtopic"],

            "complexity_score":
                complexity_score,

            "runs":
                run_results,

            "routes":
                successful_routes,

            "reasons":
                reasons,

            "modal_route":
                modal_route,

            "tied_modal_routes":
                tied_modal_routes,

            "has_unique_modal_route":
                unique_modal,

            "modal_workflow_depth":
                (
                    WORKFLOW_DEPTH[
                        modal_route
                    ]
                    if modal_route
                    else None
                ),

            "valid_output_rate":
                (
                    sum(
                        1
                        for run
                        in run_results
                        if run[
                            "valid_route"
                        ]
                    )
                    / REPEATS_PER_CASE
                ),

            "repeat_stability":
                stability,

            "unanimous":
                (
                    len(
                        successful_routes
                    )
                    == REPEATS_PER_CASE
                    and len(
                        set(
                            successful_routes
                        )
                    )
                    == 1
                ),
        }

        results.append(
            case_result
        )

        print(
            "  Complexity:",
            round(
                complexity_score,
                4,
            ),
        )

        print(
            "  Modal route:",
            (
                modal_route
                if modal_route
                else (
                    "TIE: "
                    + str(
                        tied_modal_routes
                    )
                )
            ),
        )

        print(
            "  Stability:",
            round(
                stability,
                3,
            ),
        )

        print()

    # ========================================================
    # GLOBAL RELIABILITY / VALIDITY
    # ========================================================

    all_runs = [
        run
        for result in results
        for run
        in result["runs"]
    ]

    successful_call_count = sum(
        1
        for run
        in all_runs
        if run["error"] is None
    )

    valid_call_count = sum(
        1
        for run
        in all_runs
        if run["valid_route"]
    )

    unanimous_case_count = sum(
        1
        for result
        in results
        if result["unanimous"]
    )

    unique_modal_case_count = sum(
        1
        for result
        in results
        if result[
            "has_unique_modal_route"
        ]
    )

    mean_stability = (
        statistics.mean(
            result[
                "repeat_stability"
            ]
            for result
            in results
        )
    )

    # ========================================================
    # PROBLEM SENSITIVITY
    # ========================================================

    problem_results = [
        result
        for result
        in results
        if result["group"]
        == "problem_sensitivity"
    ]

    usable_problem_results = [
        result
        for result
        in problem_results
        if result[
            "modal_route"
        ]
        is not None
    ]

    problem_scores = [
        result[
            "complexity_score"
        ]
        for result
        in usable_problem_results
    ]

    problem_depths = [
        float(
            result[
                "modal_workflow_depth"
            ]
        )
        for result
        in usable_problem_results
    ]

    score_route_spearman = (
        spearman_correlation(
            problem_scores,
            problem_depths,
        )
    )

    ordered_problem_results = sorted(
        usable_problem_results,
        key=lambda item: item[
            "complexity_score"
        ],
    )

    pairwise_comparisons = 0
    pairwise_nondecreasing = 0
    pairwise_strict_increase = 0

    for lower_index in range(
        len(ordered_problem_results)
    ):
        for higher_index in range(
            lower_index + 1,
            len(
                ordered_problem_results
            ),
        ):
            lower = (
                ordered_problem_results[
                    lower_index
                ]
            )

            higher = (
                ordered_problem_results[
                    higher_index
                ]
            )

            if math.isclose(
                lower[
                    "complexity_score"
                ],
                higher[
                    "complexity_score"
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue

            pairwise_comparisons += 1

            lower_depth = (
                lower[
                    "modal_workflow_depth"
                ]
            )

            higher_depth = (
                higher[
                    "modal_workflow_depth"
                ]
            )

            if (
                higher_depth
                >= lower_depth
            ):
                pairwise_nondecreasing += 1

            if (
                higher_depth
                > lower_depth
            ):
                pairwise_strict_increase += 1

    problem_nondecreasing_rate = (
        safe_divide(
            pairwise_nondecreasing,
            pairwise_comparisons,
        )
    )

    problem_strict_increase_rate = (
        safe_divide(
            pairwise_strict_increase,
            pairwise_comparisons,
        )
    )

    distinct_problem_routes = sorted(
        {
            result[
                "modal_route"
            ]
            for result
            in usable_problem_results
        }
    )

    problem_sensitivity = {
        "number_of_problem_cases":
            len(problem_results),

        "usable_unique_modal_cases":
            len(
                usable_problem_results
            ),

        "distinct_modal_routes":
            distinct_problem_routes,

        "number_of_distinct_modal_routes":
            len(
                distinct_problem_routes
            ),

        "complexity_score_to_workflow_depth_spearman":
            score_route_spearman,

        "pairwise_nondecreasing_depth_rate":
            problem_nondecreasing_rate,

        "pairwise_strict_increase_rate":
            problem_strict_increase_rate,

        "ordered_by_complexity_score": [
            {
                "case_id":
                    result[
                        "case_id"
                    ],

                "complexity_score":
                    result[
                        "complexity_score"
                    ],

                "modal_route":
                    result[
                        "modal_route"
                    ],

                "workflow_depth":
                    result[
                        "modal_workflow_depth"
                    ],
            }
            for result
            in ordered_problem_results
        ],

        "interpretation_note": (
            "These are sensitivity metrics, not "
            "routing accuracy. No gold workflow labels "
            "were assigned."
        ),
    }

    # ========================================================
    # LEARNER-CONTEXT SENSITIVITY
    # ========================================================

    result_by_id = {
        result[
            "case_id"
        ]: result
        for result
        in results
    }

    learner_context_pairs: list[
        dict[str, Any]
    ] = []

    comparable_pair_count = 0
    route_changed_count = 0
    struggling_more_support_count = 0
    struggling_same_or_more_count = 0

    for strong_id, struggling_id in (
        CONTEXT_PAIRS
    ):
        strong = result_by_id[
            strong_id
        ]

        struggling = result_by_id[
            struggling_id
        ]

        same_question = (
            strong["question"]
            == struggling["question"]
        )

        same_complexity_score = (
            math.isclose(
                strong[
                    "complexity_score"
                ],
                struggling[
                    "complexity_score"
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        strong_route = (
            strong[
                "modal_route"
            ]
        )

        struggling_route = (
            struggling[
                "modal_route"
            ]
        )

        comparable = (
            strong_route is not None
            and struggling_route
            is not None
        )

        route_changed = None
        struggling_more_support = None
        struggling_same_or_more = None
        depth_difference = None

        if comparable:
            comparable_pair_count += 1

            strong_depth = (
                WORKFLOW_DEPTH[
                    strong_route
                ]
            )

            struggling_depth = (
                WORKFLOW_DEPTH[
                    struggling_route
                ]
            )

            depth_difference = (
                struggling_depth
                - strong_depth
            )

            route_changed = (
                strong_route
                != struggling_route
            )

            struggling_more_support = (
                struggling_depth
                > strong_depth
            )

            struggling_same_or_more = (
                struggling_depth
                >= strong_depth
            )

            if route_changed:
                route_changed_count += 1

            if struggling_more_support:
                struggling_more_support_count += 1

            if struggling_same_or_more:
                struggling_same_or_more_count += 1

        pair_result = {
            "strong_case":
                strong_id,

            "struggling_case":
                struggling_id,

            "same_question":
                same_question,

            "same_complexity_score":
                same_complexity_score,

            "strong_modal_route":
                strong_route,

            "struggling_modal_route":
                struggling_route,

            "strong_workflow_depth":
                (
                    WORKFLOW_DEPTH[
                        strong_route
                    ]
                    if strong_route
                    else None
                ),

            "struggling_workflow_depth":
                (
                    WORKFLOW_DEPTH[
                        struggling_route
                    ]
                    if struggling_route
                    else None
                ),

            "workflow_depth_difference":
                depth_difference,

            "route_changed":
                route_changed,

            "struggling_received_more_support":
                struggling_more_support,

            "struggling_received_same_or_more_support":
                struggling_same_or_more,

            "comparable":
                comparable,
        }

        learner_context_pairs.append(
            pair_result
        )

    learner_context_summary = {
        "number_of_pairs":
            len(CONTEXT_PAIRS),

        "comparable_pairs":
            comparable_pair_count,

        "route_change_rate":
            safe_divide(
                route_changed_count,
                comparable_pair_count,
            ),

        "struggling_more_support_rate":
            safe_divide(
                struggling_more_support_count,
                comparable_pair_count,
            ),

        "struggling_same_or_more_support_rate":
            safe_divide(
                struggling_same_or_more_count,
                comparable_pair_count,
            ),

        "interpretation_note": (
            "These controlled pairs measure whether "
            "learner evidence affects workflow depth. "
            "They are directional behavioral tests, "
            "not gold-label routing accuracy."
        ),
    }

    # ========================================================
    # ROUTE DISTRIBUTION
    # ========================================================

    route_distribution = dict(
        Counter(
            run["route"]
            for run
            in all_runs
            if run["valid_route"]
        )
    )

    # ========================================================
    # LATENCY
    # ========================================================

    mean_latency = (
        statistics.mean(
            all_call_latencies
        )
    )

    median_latency = (
        statistics.median(
            all_call_latencies
        )
    )

    max_latency = max(
        all_call_latencies
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "number_of_cases":
            len(results),

        "repeats_per_case":
            REPEATS_PER_CASE,

        "total_router_calls":
            len(all_runs),

        "call_success_rate":
            (
                successful_call_count
                / len(all_runs)
            ),

        "valid_route_rate":
            (
                valid_call_count
                / len(all_runs)
            ),

        "mean_repeat_stability":
            mean_stability,

        "unanimous_case_rate":
            (
                unanimous_case_count
                / len(results)
            ),

        "unique_modal_route_case_rate":
            (
                unique_modal_case_count
                / len(results)
            ),

        "route_distribution":
            route_distribution,

        "mean_latency_seconds":
            mean_latency,

        "median_latency_seconds":
            median_latency,

        "maximum_latency_seconds":
            max_latency,
    }

    # ========================================================
    # SAVE
    # ========================================================

    evaluation = {
        "experiment":
            "AdaptMath GenAI Router targeted evaluation",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model":
            settings.groq_model,

        "methodology": {
            "description": (
                "Controlled behavioral evaluation of "
                "the GenAI workflow Router."
            ),

            "gold_route_labels_used":
                False,

            "why_no_accuracy_metric": (
                "The three AdaptMath workflows do not "
                "have independently established ground-"
                "truth route labels. Assigning labels "
                "ourselves would create circular evidence."
            ),

            "evaluated_properties": [
                "structured route validity",
                "repeated-run stability",
                (
                    "problem-complexity sensitivity "
                    "under controlled learner context"
                ),
                (
                    "learner-context sensitivity while "
                    "holding the mathematical problem "
                    "and complexity score constant"
                ),
                "latency",
            ],

            "workflow_depth_definition": {
                "direct_tutor":
                    1,

                "planned_tutor":
                    2,
            },

            "workflow_depth_note": (
                "Workflow depth is used only for "
                "directional sensitivity analysis. "
                "It does not imply that a deeper route "
                "is universally more correct."
            ),

            "limitations": [
                (
                    "This is a small targeted behavioral "
                    "benchmark rather than a population-"
                    "level routing benchmark."
                ),
                (
                    "No independent expert-labelled "
                    "optimal workflow dataset currently "
                    "exists for these custom workflows."
                ),
                (
                    "Hosted-model behavior may vary "
                    "across provider/model versions."
                ),
                (
                    "Router reasons are preserved for "
                    "manual qualitative grounding review."
                ),
            ],
        },

        "summary":
            summary,

        "problem_sensitivity":
            problem_sensitivity,

        "learner_context_summary":
            learner_context_summary,

        "learner_context_pairs":
            learner_context_pairs,

        "cases":
            results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print(
        "===================================="
    )

    print(
        "ROUTER EVALUATION SUMMARY"
    )

    print(
        "===================================="
    )

    print(
        "Cases:",
        summary[
            "number_of_cases"
        ],
    )

    print(
        "Total calls:",
        summary[
            "total_router_calls"
        ],
    )

    print(
        "Call success rate:",
        round(
            summary[
                "call_success_rate"
            ],
            3,
        ),
    )

    print(
        "Valid route rate:",
        round(
            summary[
                "valid_route_rate"
            ],
            3,
        ),
    )

    print(
        "Mean repeat stability:",
        round(
            summary[
                "mean_repeat_stability"
            ],
            3,
        ),
    )

    print(
        "Unanimous case rate:",
        round(
            summary[
                "unanimous_case_rate"
            ],
            3,
        ),
    )

    print(
        "Unique modal-route case rate:",
        round(
            summary[
                "unique_modal_route_case_rate"
            ],
            3,
        ),
    )

    print()

    print(
        "--- Problem sensitivity ---"
    )

    print(
        "Distinct modal routes:",
        problem_sensitivity[
            "distinct_modal_routes"
        ],
    )

    print(
        "Score-route Spearman:",
        (
            round(
                score_route_spearman,
                3,
            )
            if score_route_spearman
            is not None
            else None
        ),
    )

    print(
        "Pairwise non-decreasing depth:",
        (
            round(
                problem_nondecreasing_rate,
                3,
            )
            if problem_nondecreasing_rate
            is not None
            else None
        ),
    )

    print(
        "Pairwise strict-increase rate:",
        (
            round(
                problem_strict_increase_rate,
                3,
            )
            if problem_strict_increase_rate
            is not None
            else None
        ),
    )

    print()

    print(
        "Problems ordered by learned "
        "complexity score:"
    )

    for item in problem_sensitivity[
        "ordered_by_complexity_score"
    ]:
        print(
            " ",
            item["case_id"],
            "-> score",
            round(
                item[
                    "complexity_score"
                ],
                4,
            ),
            "->",
            item[
                "modal_route"
            ],
        )

    print()

    print(
        "--- Learner-context sensitivity ---"
    )

    print(
        "Route-change rate:",
        learner_context_summary[
            "route_change_rate"
        ],
    )

    print(
        "Struggling learner received "
        "more support rate:",
        learner_context_summary[
            "struggling_more_support_rate"
        ],
    )

    print(
        "Struggling learner received "
        "same-or-more support rate:",
        learner_context_summary[
            "struggling_same_or_more_support_rate"
        ],
    )

    print()

    for pair in learner_context_pairs:
        print(
            " ",
            pair["strong_case"],
            "->",
            pair[
                "strong_modal_route"
            ],
        )

        print(
            " ",
            pair["struggling_case"],
            "->",
            pair[
                "struggling_modal_route"
            ],
        )

        print(
            "   same question:",
            pair[
                "same_question"
            ],
        )

        print(
            "   same complexity:",
            pair[
                "same_complexity_score"
            ],
        )

        print(
            "   route changed:",
            pair[
                "route_changed"
            ],
        )

        print(
            "   more support:",
            pair[
                "struggling_received_more_support"
            ],
        )

        print()

    print(
        "--- Latency ---"
    )

    print(
        "Mean:",
        round(
            mean_latency,
            2,
        ),
        "seconds",
    )

    print(
        "Median:",
        round(
            median_latency,
            2,
        ),
        "seconds",
    )

    print(
        "Maximum:",
        round(
            max_latency,
            2,
        ),
        "seconds",
    )

    print()

    print(
        "IMPORTANT: these are behavioral "
        "sensitivity/reliability results, not "
        "a gold-label routing accuracy score."
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()