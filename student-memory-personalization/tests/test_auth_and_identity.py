"""Focused tests for User Accounts, Password Hashing, Age Calculation, Signup, Login, and Identity."""

from __future__ import annotations

from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
from src.database.base import Base
from src.database.models.core import Student
from src.database.models.user_account import UserAccount
import src.database.postgres_session as postgres_session
import src.services.auth_service as auth_service_module
from src.services.auth_service import AuthService, calculate_age, hash_password, set_auth_service, verify_password


@pytest.fixture
def auth_test_env(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    monkeypatch.setattr(postgres_session, "get_session_factory", lambda: factory)
    auth_svc = AuthService(session_factory=factory)
    set_auth_service(auth_svc)

    client = TestClient(app, raise_server_exceptions=False)
    yield {"client": client, "factory": factory, "auth_service": auth_svc}
    set_auth_service(None)


def test_password_hashing_and_verification():
    """Verify PBKDF2 hashing produces unique salts and verifies correctly."""
    pw = "SuperSecret123!"
    h1 = hash_password(pw)
    h2 = hash_password(pw)
    assert h1 != h2, "Salts must be unique"
    assert verify_password(pw, h1) is True
    assert verify_password(pw, h2) is True
    assert verify_password("WrongPassword", h1) is False
    assert pw not in h1, "Plaintext password must not appear in hash"


def test_exact_age_calculation():
    """Verify age calculation handles before and after birthday correctly."""
    today = date(2026, 8, 19)
    # Birthday not occurred yet this year (Dec 15)
    dob_later = date(2006, 12, 15)
    assert calculate_age(dob_later, today=today) == 19

    # Birthday already occurred this year (Jan 10)
    dob_earlier = date(2006, 1, 10)
    assert calculate_age(dob_earlier, today=today) == 20

    # Birthday is today
    dob_today = date(2006, 8, 19)
    assert calculate_age(dob_today, today=today) == 20


def test_student_signup_success(auth_test_env):
    """Verify valid student registration creates Student and UserAccount rows."""
    client = auth_test_env["client"]
    factory = auth_test_env["factory"]

    dob = "2006-04-12"
    payload = {
        "username": "student_alice",
        "date_of_birth": dob,
        "password": "Password123!",
        "confirm_password": "Password123!",
    }

    res = client.post("/auth/signup", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["username"] == "student_alice"
    assert data["student_id"] == "student_alice"
    assert data["role"] == "STUDENT"
    assert data["age"] >= 19

    # Verify Database state
    with factory() as session:
        stud = session.scalar(select(Student).where(Student.external_student_id == "student_alice"))
        assert stud is not None

        acc = session.scalar(select(UserAccount).where(UserAccount.username == "student_alice"))
        assert acc is not None
        assert acc.student_id == stud.student_id
        assert acc.password_hash != "Password123!"
        assert verify_password("Password123!", acc.password_hash) is True
        assert acc.date_of_birth == date(2006, 4, 12)
        assert acc.role == "STUDENT"


def test_signup_validation_rejections(auth_test_env):
    """Verify signup rejects duplicate username, mismatched password, and future DOB."""
    client = auth_test_env["client"]

    # 1. Successful first signup
    client.post("/auth/signup", json={
        "username": "unique_user",
        "date_of_birth": "2007-01-01",
        "password": "Pass1234",
        "confirm_password": "Pass1234",
    })

    # 2. Duplicate username
    res_dup = client.post("/auth/signup", json={
        "username": "unique_user",
        "date_of_birth": "2007-01-01",
        "password": "Pass1234",
        "confirm_password": "Pass1234",
    })
    assert res_dup.status_code in (409, 422)

    # 3. Password mismatch
    res_mismatch = client.post("/auth/signup", json={
        "username": "other_user",
        "date_of_birth": "2007-01-01",
        "password": "Pass1234",
        "confirm_password": "DifferentPassword",
    })
    assert res_mismatch.status_code == 422

    # 4. Future DOB
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    res_future = client.post("/auth/signup", json={
        "username": "future_user",
        "date_of_birth": tomorrow,
        "password": "Pass1234",
        "confirm_password": "Pass1234",
    })
    assert res_future.status_code == 422


def test_login_me_logout_lifecycle(auth_test_env):
    """Verify login provides session token, /auth/me returns identity, and logout invalidates token."""
    client = auth_test_env["client"]

    # Create account
    client.post("/auth/signup", json={
        "username": "login_user",
        "date_of_birth": "2005-05-05",
        "password": "SecretPassword123",
        "confirm_password": "SecretPassword123",
    })

    # Wrong password
    res_bad_pw = client.post("/auth/login", json={"username": "login_user", "password": "WrongPassword"})
    assert res_bad_pw.status_code == 401

    # Valid login
    res_login = client.post("/auth/login", json={"username": "login_user", "password": "SecretPassword123"})
    assert res_login.status_code == 200
    login_data = res_login.json()
    assert "token" in login_data
    token = login_data["token"]
    assert login_data["username"] == "login_user"
    assert login_data["role"] == "STUDENT"

    # GET /auth/me with Bearer token
    res_me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["username"] == "login_user"
    assert me_data["student_id"] == "login_user"

    # Logout
    res_logout = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert res_logout.status_code == 200

    # Subsequent /auth/me fails
    res_me_after = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me_after.status_code == 401
