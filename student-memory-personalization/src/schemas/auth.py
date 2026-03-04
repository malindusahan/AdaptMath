"""Pydantic schemas for authentication, signup, login, and user profile."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


PASSWORD_MIN_LENGTH = 8


class SignupRequest(BaseModel):
    """Payload for student self-registration."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, max_length=255, description="Unique username.")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format.")
    password: str = Field(
        ...,
        min_length=PASSWORD_MIN_LENGTH,
        max_length=255,
        description=f"Account password (minimum {PASSWORD_MIN_LENGTH} characters).",
    )
    confirm_password: str = Field(
        ...,
        min_length=PASSWORD_MIN_LENGTH,
        max_length=255,
        description="Confirm password matching password.",
    )


class SignupResponse(BaseModel):
    """Sanitized registration response."""

    model_config = ConfigDict(extra="ignore")

    username: str
    student_id: str
    age: int
    role: str = "STUDENT"


class LoginRequest(BaseModel):
    """Payload for user login."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class LoginResponse(BaseModel):
    """Authentication token and logged-in user profile."""

    model_config = ConfigDict(extra="ignore")

    token: str
    username: str
    role: str
    student_id: str | None = None
    age: int | None = None


class UserMeResponse(BaseModel):
    """Currently authenticated user identity."""

    model_config = ConfigDict(extra="ignore")

    user_id: str
    username: str
    role: str
    student_id: str
    age: int | None = None
