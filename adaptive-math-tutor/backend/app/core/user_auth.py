"""Learner authentication boundary for public Tutor routes."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.clients.memory.memory_client import (
    MemoryAuthenticatedUser,
    get_memory_client,
)


bearer_scheme = HTTPBearer(auto_error=False)


def require_authenticated_student(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> MemoryAuthenticatedUser:
    """Resolve a Memory-backed student identity or reject the request."""
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials.strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_memory_client().authenticate_user(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session is invalid or unavailable. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
