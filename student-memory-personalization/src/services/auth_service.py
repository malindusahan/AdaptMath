"""Authentication and Identity Service for Student Personalization Memory."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import secrets
from typing import Any
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select

from src.database.models.core import Student
from src.database.models.user_account import UserAccount
from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.schemas.auth import (
    LoginRequest,
    LoginResponse,
    SignupRequest,
    SignupResponse,
    UserMeResponse,
)

# In-memory session store mapping session_token -> user metadata
_ACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt}${hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against its PBKDF2 salt/hash record."""
    try:
        salt, hashed = password_hash.split("$", 1)
        computed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000,
        ).hex()
        return secrets.compare_digest(computed, hashed)
    except Exception:
        return False


def calculate_age(dob: date, today: date | None = None) -> int:
    """Calculate exact age from date of birth.
    
    Formula: current_year - birth_year minus one if birthday has not occurred yet this year.
    """
    if today is None:
        today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


class AuthService:
    """Provides user account registration, login verification, and identity resolution."""

    def __init__(self, session_factory: SessionFactory | None = None):
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )

    def signup_student(self, request: SignupRequest) -> SignupResponse:
        """Atomically register a new student and linked user account."""
        username = request.username.strip()
        if not username:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Username cannot be empty.",
            )

        if request.password != request.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Password and confirmation password do not match.",
            )

        if not request.password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Password cannot be empty.",
            )

        try:
            dob = date.fromisoformat(request.date_of_birth.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid date_of_birth format. Expected YYYY-MM-DD.",
            )

        today = date.today()
        if dob > today:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Date of birth cannot be in the future.",
            )

        age = calculate_age(dob, today=today)

        with UnitOfWork(self.session_factory) as uow:
            assert uow.session is not None

            # Check for existing username in user_accounts
            existing_account = uow.session.scalar(
                select(UserAccount).where(UserAccount.username == username)
            )
            if existing_account is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Username '{username}' is already registered.",
                )

            # Check for existing external_student_id
            existing_student = uow.session.scalar(
                select(Student).where(Student.external_student_id == username)
            )
            if existing_student is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Student ID '{username}' already exists.",
                )

            # 1. Create Student Domain Entity
            student_uuid = uuid.uuid4()
            student = Student(
                student_id=student_uuid,
                external_student_id=username,
                display_name=username,
                is_active=True,
            )
            uow.session.add(student)
            uow.session.flush()

            # 2. Create User Account Entity
            pw_hash = hash_password(request.password)
            account = UserAccount(
                user_id=uuid.uuid4(),
                username=username,
                password_hash=pw_hash,
                date_of_birth=dob,
                age=age,
                role="STUDENT",
                student_id=student.student_id,
            )
            uow.session.add(account)
            uow.session.commit()

            return SignupResponse(
                username=username,
                student_id=username,
                age=age,
                role="STUDENT",
            )

    def login(self, request: LoginRequest) -> LoginResponse:
        """Authenticate user by username and password, update login metadata, and return session token."""
        username = request.username.strip()

        with UnitOfWork(self.session_factory) as uow:
            assert uow.session is not None

            account = uow.session.scalar(
                select(UserAccount).where(UserAccount.username == username)
            )
            if account is None and username.lower() in {"demo_new", "demo_developing", "demo_strong", "demo_support", "student01"}:
                try:
                    from src.demo.seed_demo_data import seed_demo_environment
                    seed_demo_environment(self.session_factory)
                    account = uow.session.scalar(
                        select(UserAccount).where(UserAccount.username == username)
                    )
                except Exception:
                    pass

            if (
                account is None
                or account.role != "STUDENT"
                or not verify_password(request.password, account.password_hash)
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password.",
                )

            # Recalculate age if date_of_birth present
            if account.date_of_birth:
                new_age = calculate_age(account.date_of_birth)
                if new_age != account.age:
                    account.age = new_age

            account.last_login_at = datetime.now()

            # Resolve external student identifier
            student_id_str: str | None = None
            if account.student_id:
                stud = uow.session.get(Student, account.student_id)
                student_id_str = stud.external_student_id if stud else str(account.student_id)

            token = f"mem_sess_{secrets.token_urlsafe(32)}"
            user_data = {
                "user_id": str(account.user_id),
                "username": account.username,
                "role": account.role,
                "student_id": student_id_str,
                "age": account.age,
            }
            _ACTIVE_SESSIONS[token] = user_data

            uow.session.commit()

            return LoginResponse(
                token=token,
                username=account.username,
                role=account.role,
                student_id=student_id_str,
                age=account.age,
            )

    login_user = login

    def get_current_user(self, token: str | None) -> dict[str, Any] | None:
        """Resolve active user session from token."""
        if not token:
            return None
        return _ACTIVE_SESSIONS.get(token)

    def logout(self, token: str | None) -> None:
        """Terminate active session."""
        if token and token in _ACTIVE_SESSIONS:
            del _ACTIVE_SESSIONS[token]


_AUTH_SERVICE: AuthService | None = None


def get_auth_service() -> AuthService:
    """Factory helper for dependency injection."""
    global _AUTH_SERVICE
    if _AUTH_SERVICE is None:
        _AUTH_SERVICE = AuthService()
    return _AUTH_SERVICE


def set_auth_service(auth_service: AuthService | None) -> None:
    """Setter for test monkeypatching."""
    global _AUTH_SERVICE
    _AUTH_SERVICE = auth_service
