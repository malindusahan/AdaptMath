"""Verify PostgreSQL auth-session survival across service object recreation."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
)
from src.schemas.auth import LoginRequest, SignupRequest
from src.services.auth_service import AuthService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    args = parser.parse_args()
    if args.database_url:
        os.environ["MEMORY_DATABASE_URL"] = args.database_url
    engine = create_postgres_engine()
    factory = create_session_factory(engine)
    username = f"pg_session_verify_{uuid.uuid4().hex[:12]}"
    password = "VerifyOnly-Password-123"
    first_process = AuthService(
        session_factory=factory,
        session_persistence="postgres",
        session_ttl_seconds=3600,
    )
    first_process.signup_student(
        SignupRequest(
            username=username,
            date_of_birth=(date.today() - timedelta(days=365 * 15)).isoformat(),
            password=password,
            confirm_password=password,
        )
    )
    login = first_process.login(LoginRequest(username=username, password=password))
    assert login.token.startswith("mem_sess_")

    restarted_process = AuthService(
        session_factory=factory,
        session_persistence="postgres",
        session_ttl_seconds=3600,
    )
    current = restarted_process.get_current_user(login.token)
    assert current is not None
    assert current["username"] == username
    assert current["student_id"] == username
    restarted_process.logout(login.token)

    second_restart = AuthService(
        session_factory=factory,
        session_persistence="postgres",
        session_ttl_seconds=3600,
    )
    assert second_restart.get_current_user(login.token) is None
    engine.dispose()
    print("postgres_sessions=true")
    print("survives_service_recreation=true")
    print("logout_revocation=true")
    print("token_format=mem_sess_compatible")


if __name__ == "__main__":
    main()
