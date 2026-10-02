import json
import time
from typing import Any

from app.agents.tutor.tutor_agent import TutorAgent
from app.schemas.tutor import TutorInput


# ============================================================
# TRACE GENAI ACTION SELECTION
# ============================================================

class TracingActionModel:
    def __init__(
        self,
        original: Any,
    ) -> None:
        self.original = original
        self.round_number = 0

    def invoke(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:

        self.round_number += 1

        print()
        print(
            f"===== MODEL ROUND "
            f"{self.round_number} ====="
        )

        start = time.perf_counter()

        response = self.original.invoke(
            messages
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Model time: {elapsed:.2f}s"
        )

        print(
            "ACTION:",
            response["action"],
        )

        print(
            "ARGUMENTS:"
        )

        print(
            json.dumps(
                response["arguments"],
                indent=2,
            )
        )

        return response


# ============================================================
# TRACE MATHEMATICAL TOOL EXECUTION
# ============================================================

class TracingTool:
    def __init__(
        self,
        name: str,
        original: Any,
    ) -> None:
        self.name = name
        self.original = original

    def invoke(
        self,
        arguments: dict[str, Any],
    ) -> Any:

        print()
        print(
            f">>>>> TOOL: {self.name}"
        )

        print(
            "NORMALIZED ARGUMENTS:"
        )

        print(
            json.dumps(
                arguments,
                indent=2,
                default=str,
            )
        )

        start = time.perf_counter()

        try:
            result = self.original.invoke(
                arguments
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"TOOL ERROR after "
                f"{elapsed:.2f}s"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            raise

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Tool time: {elapsed:.2f}s"
        )

        print(
            "RESULT:"
        )

        print(
            result
        )

        print(
            f"<<<<< END TOOL: "
            f"{self.name}"
        )

        return result


# ============================================================
# CREATE TUTOR
# ============================================================

print(
    "Creating Tutor Agent..."
)

agent = TutorAgent()


# Wrap GenAI action selector.
agent.action_model = TracingActionModel(
    agent.action_model
)


# Wrap deterministic mathematical tools.
agent.tool_map = {
    name: TracingTool(
        name,
        current_tool,
    )
    for name, current_tool
    in agent.tool_map.items()
}


# ============================================================
# HARD SYMBOLIC TEST
# ============================================================

tutor_input = TutorInput(
    question=(
        "Let p(x) and q(x) be two cubic polynomials "
        "such that p(0)=-24, q(0)=30, and "
        "p(q(x))=q(p(x)) for all real x. "
        "Find (p(3), q(6))."
    ),
    topic="algebra",
    subtopic="polynomials",
    student_age=17,
    complexity_score=0.83401269,
    planner_output=None,
    teaching_strategy=None,
    previous_errors=[],
    reteaching=False,
)


# ============================================================
# RUN
# ============================================================

print(
    "Starting hard Tutor test..."
)

start = time.perf_counter()

try:
    result = agent.teach(
        tutor_input
    )

except Exception as exc:
    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print(
        "===== TUTOR FAILED ====="
    )

    print(
        f"Total time: {elapsed:.2f}s"
    )

    print(
        f"{type(exc).__name__}: {exc}"
    )

    raise


elapsed = (
    time.perf_counter()
    - start
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print(
    "===================================="
)
print(
    "FINAL TUTOR OUTPUT"
)
print(
    "===================================="
)

print(
    f"Total time: {elapsed:.2f}s"
)

print()
print(
    "TEACHING:"
)

print(
    result.teaching_response
)

print()
print(
    "ASSESSMENT:"
)

for question in (
    result.assessment_questions
):
    print(
        json.dumps(
            question.model_dump(),
            indent=2,
        )
    )