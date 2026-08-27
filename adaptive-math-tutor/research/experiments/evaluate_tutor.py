"""
Current Tutor evaluation entrypoint.

The previous experiment evaluated a single-message Tutor that generated a full
lesson plus assessment questions in one call. That architecture is no longer
active. AdaptMath now uses a multi-turn teacher-student dialogue, a fresh
externally selected MathDial move for every teacher turn, a GenAI teaching-
progress judgment, and Evaluator-generated three-question assessment.

The original experiment is preserved verbatim as:
    evaluate_tutor_single_response_legacy.py

Do not report that historical single-response experiment as evidence for the
current multi-turn Tutor. A controlled multi-turn evaluation must be designed
against the new interaction protocol before final reporting.
"""


def main() -> None:
    raise SystemExit(
        "Legacy single-response Tutor benchmark is archived. "
        "Use a new multi-turn evaluation protocol for the current architecture."
    )


if __name__ == "__main__":
    main()
