from app.core.config import get_settings
from app.schemas.router import RouterInput, RouterOutput


ROUTER_VERSION = "6.0-deterministic-local"


class RouterAgent:
    """Route locally using the trained continuous complexity score.

    Routing remains an explicit workflow process, but it no longer consumes a
    Gemini request. Assessment is universal and is NOT part of this routing
    decision. The pedagogical move is selected by the external MD7/Turn-LinTS
    component before every Tutor turn; this router does not select it.

    Missing learner history remains unknown. It is forwarded to question
    preparation but is not converted into an invented learner-state signal.
    There are no fixed topic/problem mappings and no Easy/Medium/Hard labels.
    """

    def __init__(self) -> None:
        self.planning_threshold = get_settings().planning_complexity_threshold

    def route(self, router_input: RouterInput) -> RouterOutput:
        score = router_input.complexity_score
        if score >= self.planning_threshold:
            return RouterOutput(
                route="planned_tutor",
                reason=(
                    "The trained local complexity score meets the configured "
                    "deterministic planning threshold, so question preparation "
                    "will include an explicit interactive plan."
                ),
            )

        return RouterOutput(
            route="direct_tutor",
            reason=(
                "The trained local complexity score is below the configured "
                "deterministic planning threshold, so question preparation "
                "will verify the mathematics without a separate plan."
            ),
        )
