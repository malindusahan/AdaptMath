"""Verify an authenticated session survives a real PostgreSQL container restart."""

from __future__ import annotations

import hashlib
import secrets
import subprocess
import time

import psycopg

from verify_live_postgres_e2e import (
    MEMORY_ROOT,
    MEMORY_URL,
    TUTOR_URL,
    api,
    database_url,
    require,
    wait_ready,
)


def main() -> None:
    wait_ready()
    username = f"pgauthrestart_{int(time.time())}_{secrets.token_hex(3)}"
    password = f"C7!{secrets.token_urlsafe(18)}"
    signup = api(
        "POST",
        f"{MEMORY_URL}/auth/signup",
        expected=201,
        payload={
            "username": username,
            "date_of_birth": "2011-02-03",
            "password": password,
            "confirm_password": password,
        },
    ).json()
    token = api(
        "POST",
        f"{MEMORY_URL}/auth/login",
        payload={"username": username, "password": password},
    ).json()["token"]
    me = api("GET", f"{MEMORY_URL}/auth/me", token=token).json()
    require(me["student_id"] == signup["student_id"], "pre-restart identity drift")

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with psycopg.connect(database_url()) as connection:
        row = connection.execute(
            "SELECT token_hash, revoked_at, expires_at > CURRENT_TIMESTAMP "
            "FROM auth.sessions WHERE token_hash = %s",
            (token_hash,),
        ).fetchone()
    require(row == (token_hash, None, True), "hashed session was not durable")

    restarted = subprocess.run(
        ["docker", "compose", "restart", "postgres"],
        cwd=MEMORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
    )
    require(restarted.returncode == 0, "PostgreSQL container restart failed")
    wait_ready()

    persisted = api("GET", f"{MEMORY_URL}/auth/me", token=token).json()
    require(persisted["student_id"] == signup["student_id"], "token lost on restart")
    profile = api("GET", f"{TUTOR_URL}/profile", token=token).json()
    require(profile["username"] == username, "Tutor profile identity drift")
    api("POST", f"{MEMORY_URL}/auth/logout", token=token)
    api("GET", f"{MEMORY_URL}/auth/me", token=token, expected=401)

    with psycopg.connect(database_url()) as connection:
        revoked = connection.execute(
            "SELECT revoked_at IS NOT NULL FROM auth.sessions WHERE token_hash = %s",
            (token_hash,),
        ).fetchone()
    require(revoked == (True,), "logout revocation was not durable")
    print("postgres_container_restart=passed")
    print("token_restart_persistence=passed")
    print("tutor_profile_after_restart=passed")
    print("logout_revocation=passed")
    print("LIVE_AUTH_POSTGRES_RESTART=PASS")


if __name__ == "__main__":
    main()
