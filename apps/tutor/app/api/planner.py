from fastapi import APIRouter

from app.agents.planner.planner_agent import PlannerAgent
from app.core.model_errors import model_service_http_exception
from app.schemas.planner import PlannerInput, PlannerOutput


router = APIRouter(
    prefix="/planner",
    tags=["Planner"],
)

planner_agent = PlannerAgent()


@router.post(
    "/plan",
    response_model=PlannerOutput,
)
def create_plan(
    planner_input: PlannerInput,
) -> PlannerOutput:
    try:
        return planner_agent.create_plan(planner_input)

    except Exception as exc:
        raise model_service_http_exception(
            exc
        ) from exc