from app.api.tutor import _adaptive_http_exception
from app.integrations.adaptive_component_coordinator import (
    AdaptiveConcurrencyError,
    AdaptiveRestartRequiredError,
)


def test_concurrent_adaptive_attempt_is_safe_409() -> None:
    result = _adaptive_http_exception(AdaptiveConcurrencyError("internal"))
    assert result.status_code == 409
    assert result.detail == (
        "Another adaptive tutoring session is currently active. "
        "Please retry shortly."
    )


def test_interrupted_adaptive_attempt_requires_new_session() -> None:
    result = _adaptive_http_exception(AdaptiveRestartRequiredError("internal"))
    assert result.status_code == 409
    assert result.detail == (
        "This tutoring session was interrupted by a server restart and "
        "cannot be safely continued. Start a new session."
    )
