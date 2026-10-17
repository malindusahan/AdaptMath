"""Thin boundaries joining AdaptMath to the frozen adaptive subsystem."""

from app.integrations.adaptive_component_coordinator import (
    AdaptiveComponentCoordinator,
    AdaptiveConcurrencyError,
    AdaptiveIntegrationError,
    AdaptiveRestartRequiredError,
    build_default_adaptive_component_coordinator,
)
from app.integrations.adaptive_tutor_agent_adapter import (
    AdaptiveTutorAgentAdapter,
)
from app.integrations.tutor_state_memory_adapter import (
    TutorStateMemoryAdapter,
)

__all__ = (
    "AdaptiveComponentCoordinator",
    "AdaptiveConcurrencyError",
    "AdaptiveIntegrationError",
    "AdaptiveRestartRequiredError",
    "AdaptiveTutorAgentAdapter",
    "TutorStateMemoryAdapter",
    "build_default_adaptive_component_coordinator",
)
