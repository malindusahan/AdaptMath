"""Development-only targeted Router V4 regression check.

This script lives outside the backend package, so it resolves the project root
and adds ``backend`` to ``sys.path`` explicitly. The production complexity
service returns the continuous complexity score directly as a ``float``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.complexity.service import get_complexity_service  # noqa: E402
from app.agents.router.router_agent import ROUTER_VERSION, RouterAgent  # noqa: E402
from app.schemas.memory import MemoryContext  # noqa: E402
from app.schemas.profile import StudentProfile  # noqa: E402
from app.schemas.router import RouterInput  # noqa: E402

QUESTION = """Three positive integers a, b, and c satisfy:
a + b + c = 30
ab + bc + ca = 269
abc = 660
Find a, b, and c, and explain how you know your answer is the only possible solution."""


def main() -> None:
    complexity_score = float(get_complexity_service().predict(QUESTION))

    inp = RouterInput(
        question=QUESTION,
        profile=StudentProfile(student_id="router-v4-targeted", age=18),
        memory=MemoryContext(
            student_id="router-v4-targeted",
            topic="Algebra",
            subtopic="Systems of equations / symmetric polynomials",
            relevant_history=[],
            previous_errors=[],
            previous_strategies=[],
        ),
        complexity_score=complexity_score,
    )

    result = RouterAgent().route(inp)

    print("ROUTER VERSION:", ROUTER_VERSION)
    print("COMPLEXITY SCORE:", complexity_score)
    print("ROUTE:", result.route)
    print("REASON:", result.reason)
    print("\nJSON")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
