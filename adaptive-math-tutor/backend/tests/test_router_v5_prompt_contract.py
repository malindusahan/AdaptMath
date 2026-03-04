from pydantic import ValidationError
import pytest

from app.agents.router.router_agent import ROUTER_VERSION, RouterAgent
from app.schemas.memory import MemoryContext
from app.schemas.profile import StudentProfile
from app.schemas.router import RouterInput, RouterOutput


def _input(score: float) -> RouterInput:
    return RouterInput(
        question="Find the remaining area of two concentric circles.",
        profile=StudentProfile(student_id="student-1", age=15),
        memory=MemoryContext(
            student_id="student-1",
            topic="Geometry",
            subtopic="Area of circles",
        ),
        complexity_score=score,
    )


def test_router_has_only_direct_and_planned_routes() -> None:
    RouterOutput(route="direct_tutor", reason="Direct preparation.")
    RouterOutput(route="planned_tutor", reason="Include a plan.")
    with pytest.raises(ValidationError):
        RouterOutput(route="planned_tutor_evaluate", reason="Legacy route.")


def test_router_is_deterministic_and_uses_no_model_client() -> None:
    router = RouterAgent.__new__(RouterAgent)
    router.planning_threshold = 0.5

    assert ROUTER_VERSION == "6.0-deterministic-local"
    assert not hasattr(router, "model")
    assert router.route(_input(0.49)).route == "direct_tutor"
    assert router.route(_input(0.50)).route == "planned_tutor"
    assert router.route(_input(0.90)).route == "planned_tutor"


def test_router_does_not_select_move_or_assessment() -> None:
    router = RouterAgent.__new__(RouterAgent)
    router.planning_threshold = 0.5
    result = router.route(_input(0.4)).model_dump()

    assert set(result) == {"route", "reason"}
    assert "pedagogical_move" not in result
    assert "assessment" not in result
