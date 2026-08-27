from __future__ import annotations

from typing import TypedDict

from app.agents.complexity.service import get_complexity_service
from app.core.persistence import check_persistence_ready


class ReadinessCheck(TypedDict):
    status: str
    detail: str


class ReadinessReport(TypedDict):
    status: str
    checks: dict[str, ReadinessCheck]


def _check_checkpoint_database() -> ReadinessCheck:
    try:
        if check_persistence_ready():
            return {
                "status": "ok",
                "detail": "SQLite checkpoint database is available.",
            }
    except Exception as exc:  # readiness must report, not crash
        return {
            "status": "error",
            "detail": f"Checkpoint database check failed: {type(exc).__name__}",
        }

    return {
        "status": "error",
        "detail": "SQLite checkpoint database is unavailable.",
    }


def _check_complexity_model() -> ReadinessCheck:
    try:
        service = get_complexity_service()
        if service.model is not None:
            return {
                "status": "ok",
                "detail": "Complexity model is loaded.",
            }
    except Exception as exc:  # readiness must report, not crash
        return {
            "status": "error",
            "detail": f"Complexity model check failed: {type(exc).__name__}",
        }

    return {
        "status": "error",
        "detail": "Complexity model is unavailable.",
    }


def get_readiness_report() -> ReadinessReport:
    """
    Check dependencies required for the current standalone backend.

    This endpoint deliberately does not call Gemini or teammate-owned APIs.
    Readiness probes should be fast, deterministic, and should not consume
    model-provider quota. External-service checks can be added after the
    real integration contracts are connected.
    """

    checks: dict[str, ReadinessCheck] = {
        "checkpoint_database": _check_checkpoint_database(),
        "complexity_model": _check_complexity_model(),
    }

    ready = all(
        check["status"] == "ok"
        for check in checks.values()
    )

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
