"""Authentication and user session routes for students."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.schemas.auth import (
    LoginRequest,
    LoginResponse,
    SignupRequest,
    SignupResponse,
    UserMeResponse,
)
from src.services.auth_service import AuthService, get_auth_service

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

bearer_scheme = HTTPBearer(auto_error=False)


def get_token_from_header(
    auth_header: str | None = Header(default=None, alias="Authorization"),
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    """Extract Bearer token from either Authorization header or Bearer credentials."""
    if creds and creds.credentials:
        return creds.credentials
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "").strip()
    return None


def get_current_user_optional(
    token: str | None = Depends(get_token_from_header),
) -> dict[str, Any] | None:
    """Resolve current logged in user dict from token, or None if not authenticated."""
    if not token:
        return None
    auth_service = get_auth_service()
    return auth_service.get_current_user(token)


def get_current_user_required(
    current_user: dict[str, Any] | None = Depends(get_current_user_optional),
) -> dict[str, Any]:
    """Ensure user is logged in, or raise 401."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_student_user(
    current_user: dict[str, Any] = Depends(get_current_user_required),
) -> dict[str, Any]:
    """Ensure user is a registered STUDENT."""
    if current_user.get("role") != "STUDENT":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Student role required.",
        )
    return current_user


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Student Signup & Identity Provisioning",
)
def signup_student(
    request: SignupRequest,
) -> SignupResponse:
    """
    Register a new student with username, exact birth date, and password.
    Automatically provisions internal student identity and calculates age.
    """
    auth_service = get_auth_service()
    return auth_service.signup_student(request)


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
)
def login_user(
    request: LoginRequest,
) -> LoginResponse:
    """Authenticate a student and issue a session token."""
    auth_service = get_auth_service()
    return auth_service.login(request)


@router.get(
    "/me",
    response_model=UserMeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current User Profile & Identity",
)
def get_current_user_profile(
    current_user: dict[str, Any] = Depends(require_student_user),
) -> UserMeResponse:
    """Return profile of currently authenticated user."""
    student_id = current_user.get("student_id")
    if not student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated student account is not linked to a student identity.",
        )
    return UserMeResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        role=current_user["role"],
        student_id=student_id,
        age=current_user.get("age"),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Log Out User",
)
def logout_user(
    token: str | None = Depends(get_token_from_header),
) -> dict[str, str]:
    """Invalidate active session token."""
    auth_service = get_auth_service()
    auth_service.logout(token)
    return {"message": "Successfully logged out."}
