"""Narrow runtime boundaries between independently owned repositories."""

from .student_model_v3_bridge import (
    ActiveCrossRepositoryAttempt,
    StudentModelFrozenV3Bridge,
)

__all__ = (
    "ActiveCrossRepositoryAttempt",
    "StudentModelFrozenV3Bridge",
)
