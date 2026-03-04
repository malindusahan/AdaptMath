"""Controlled live acceptance test for unified AdaptMath PostgreSQL persistence.

This script intentionally prints no password, bearer token, assessment answer, or
checkpoint payload.  It requires the Final-SKL live stack on ports 8400/8402.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import psycopg
import requests
from langgraph.checkpoint.postgres import PostgresSaver


WORKSPACE = Path(__file__).resolve().parents[1]
MEMORY_ROOT = WORKSPACE / "student-memory-personalization"
MEMORY_URL = "http://127.0.0.1:8400"
TUTOR_URL = "http://127.0.0.1:8402"
HTTP_TIMEOUT = 180


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def dotenv_value(name: str) -> str | None:
    path = MEMORY_ROOT / ".env"
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def database_url() -> str:
    password = (
        os.getenv("MEMORY_POSTGRES_PASSWORD")
        or dotenv_value("MEMORY_POSTGRES_PASSWORD")
        or "memory-local-dev-only"
    )
    return (
        "postgresql://memory_app:"
        f"{quote(password, safe='')}@127.0.0.1:5433/adaptmath_memory_integration"
    )


def api(
    method: str,
    url: str,
    *,
    token: str | None = None,
    expected: int | tuple[int, ...] = 200,
    payload: dict[str, Any] | None = None,
    timeout: int = HTTP_TIMEOUT,
) -> requests.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(
        method,
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    expected_codes = (expected,) if isinstance(expected, int) else expected
    if response.status_code not in expected_codes:
        detail = response.text[:800].replace("\n", " ")
        raise AssertionError(
            f"{method} {url} returned {response.status_code}; expected "
            f"{expected_codes}: {detail}"
        )
    return response


def wait_ready() -> None:
    deadline = time.monotonic() + 360
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            memory = requests.get(f"{MEMORY_URL}/ready", timeout=5)
            tutor = requests.get(f"{TUTOR_URL}/ready", timeout=5)
            if memory.status_code == 200 and tutor.status_code == 200:
                return
            last_error = f"memory={memory.status_code}, tutor={tutor.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        time.sleep(2)
    raise TimeoutError(f"stack did not become ready: {last_error}")


def checkpoint_answers(thread_id: str) -> list[dict[str, str]]:
    conninfo = database_url() + "?options=-c%20search_path%3Dtutor%2Cpublic"
    with psycopg.connect(conninfo, autocommit=True) as connection:
        saved = PostgresSaver(connection).get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
    require(saved is not None, "Tutor checkpoint disappeared")
    values = saved.checkpoint.get("channel_values", {})
    questions = values.get("assessment_questions", [])
    require(len(questions) == 3, "expected exactly three persisted questions")
    answers: list[dict[str, str]] = []
    for question in questions:
        question_id = question.get("question_id")
        answer = question.get("expected_answer")
        require(isinstance(question_id, str) and question_id, "missing question id")
        require(isinstance(answer, str) and answer, "missing expected answer")
        answers.append({"question_id": question_id, "answer": answer})
    return answers


def run_session(
    *,
    token: str,
    student_id: str,
    question: str,
    learner_solution: str,
    ordinal: int,
) -> dict[str, Any]:
    started = api(
        "POST",
        f"{TUTOR_URL}/tutor/start",
        token=token,
        payload={
            "student_id": student_id,
            "age": 15,
            "question": question,
            "topic": "Percent Of",
            "subtopic": "Percent Of",
            "target_skill": "Percent Of",
        },
    ).json()
    thread_id = started["thread_id"]
    state = started

    # The active integrated selector must ignore the legacy manual move seam.
    if state["status"] == "student_response_required":
        rejected = api(
            "POST",
            f"{TUTOR_URL}/tutor/{thread_id}/move",
            token=token,
            expected=409,
            payload={"pedagogical_move": "telling"},
        )
        require(rejected.status_code == 409, "manual selector seam was not rejected")

    def transition(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Retry transient model-service failures only if the checkpoint did not move."""

        before = api("GET", f"{TUTOR_URL}/tutor/{thread_id}", token=token).json()
        for retry in range(5):
            response = requests.post(
                f"{TUTOR_URL}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code not in {502, 503}:
                raise AssertionError(
                    f"POST {path} returned {response.status_code}: {response.text[:800]}"
                )
            current = api(
                "GET", f"{TUTOR_URL}/tutor/{thread_id}", token=token
            ).json()
            if (
                current.get("status") != before.get("status")
                or current.get("turn_count") != before.get("turn_count")
                or current.get("reteach_round") != before.get("reteach_round")
            ):
                return current
            if retry == 4:
                raise AssertionError(f"transient model failure persisted for {path}")
            time.sleep(3 * (retry + 1))
        raise AssertionError("unreachable transition retry state")

    for step in range(18):
        status = state["status"]
        if status == "complete":
            require(state.get("evaluation") is not None, "session lacked evaluation")
            print(
                f"session_{ordinal}=complete thread_id={thread_id} "
                f"turn_count={state.get('turn_count')} "
                f"reteach_round={state.get('reteach_round')}"
            )
            return state
        if status == "student_response_required":
            state = transition(
                f"/tutor/{thread_id}/turn",
                {
                    "response": (
                        f"{learner_solution} I converted the percentage to a "
                        "decimal, multiplied by the whole quantity, and checked "
                        "the result. I can explain each calculation and I am "
                        "ready for the assessment."
                    )
                },
            )
            continue
        if status == "assessment_required":
            state = transition(
                f"/tutor/{thread_id}/answers",
                {"answers": checkpoint_answers(thread_id)},
            )
            continue
        if status == "recovery_required":
            state = transition(f"/tutor/{thread_id}/resume")
            continue
        raise AssertionError(f"unexpected integrated session status: {status}")
    raise AssertionError(f"session {ordinal} did not finish within 18 transitions")


def scalar(connection: psycopg.Connection[Any], query: str, params=()) -> int:
    row = connection.execute(query, params).fetchone()
    require(row is not None, "count query returned no row")
    return int(row[0])


def snapshot_counts(username: str | None = None) -> dict[str, int]:
    where = "" if username is None else " WHERE ua.username = %s"
    params = () if username is None else (username,)
    with psycopg.connect(database_url()) as connection:
        counts = {
            "accounts": scalar(
                connection,
                "SELECT COUNT(*) FROM student_memory.user_accounts ua" + where,
                params,
            ),
            "auth_sessions": scalar(connection, "SELECT COUNT(*) FROM auth.sessions"),
            "tutor_threads": scalar(connection, "SELECT COUNT(*) FROM tutor.threads"),
            "student_attempts": scalar(connection, "SELECT COUNT(*) FROM student_model.attempts"),
            "research_actions": scalar(connection, "SELECT COUNT(*) FROM research.turn_actions"),
            "policy_updates": scalar(connection, "SELECT COUNT(*) FROM research.policy_updates"),
            "interactions": scalar(connection, "SELECT COUNT(*) FROM student_memory.interaction_logs"),
        }
    return counts


def restart_full_stack() -> None:
    stop = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WORKSPACE / "stop-adaptmath-local.ps1"),
        ],
        cwd=WORKSPACE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
    )
    require(stop.returncode == 0, "launcher stop failed")
    postgres_restart = subprocess.run(
        ["docker", "compose", "restart", "postgres"],
        cwd=MEMORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
    )
    require(postgres_restart.returncode == 0, "PostgreSQL container restart failed")
    start = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WORKSPACE / "start-adaptmath-local.ps1"),
            "-FinalSKLLive",
        ],
        cwd=WORKSPACE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=600,
    )
    if start.returncode != 0:
        raise AssertionError("launcher restart failed; inspect _local_runtime_logs")
    wait_ready()
    print("full_stack_restart=passed postgres_container_restarted=true")


def main() -> int:
    wait_ready()
    run_id = f"pgaccept_{int(time.time())}_{secrets.token_hex(3)}"
    username = run_id
    password = f"A9!{secrets.token_urlsafe(18)}"

    baseline = snapshot_counts()
    signup = api(
        "POST",
        f"{MEMORY_URL}/auth/signup",
        expected=201,
        payload={
            "username": username,
            "date_of_birth": "2011-01-15",
            "password": password,
            "confirm_password": password,
        },
    ).json()
    student_id = signup["student_id"]
    login = api(
        "POST",
        f"{MEMORY_URL}/auth/login",
        payload={"username": username, "password": password},
    ).json()
    token = login["token"]
    me = api("GET", f"{MEMORY_URL}/auth/me", token=token).json()
    require(me["student_id"] == student_id, "authenticated identity drifted")

    impersonation = api(
        "POST",
        f"{TUTOR_URL}/tutor/start",
        token=token,
        expected=403,
        payload={
            "student_id": "00000000-0000-0000-0000-000000000001",
            "age": 15,
            "question": "What is 20 percent of 50?",
            "topic": "Percent Of",
            "target_skill": "Percent Of",
        },
    )
    require(impersonation.status_code == 403, "student impersonation was accepted")

    session_one = run_session(
        token=token,
        student_id=student_id,
        question="What is 20 percent of 50?",
        learner_solution="Twenty percent of 50 is 0.20 times 50, which equals 10.",
        ordinal=1,
    )
    first_thread = session_one["thread_id"]
    after_one = snapshot_counts()
    require(after_one["student_attempts"] > baseline["student_attempts"], "BKT attempt missing")
    require(after_one["policy_updates"] > baseline["policy_updates"], "policy update missing")
    require(after_one["interactions"] > baseline["interactions"], "Memory interaction missing")

    restart_full_stack()
    persisted_me = api("GET", f"{MEMORY_URL}/auth/me", token=token).json()
    require(persisted_me["student_id"] == student_id, "token did not survive restart")
    persisted_thread = api(
        "GET", f"{TUTOR_URL}/tutor/{first_thread}", token=token
    ).json()
    require(persisted_thread["status"] == "complete", "Tutor thread did not survive restart")

    session_two = run_session(
        token=token,
        student_id=student_id,
        question="A class has 80 students. What is 25 percent of 80?",
        learner_solution="Twenty-five percent of 80 is 0.25 times 80, which equals 20.",
        ordinal=2,
    )
    require(session_two["thread_id"] != first_thread, "session thread was reused")

    # A second student must not be able to read the first student's checkpoint.
    other_username = run_id + "_other"
    other_password = f"B8!{secrets.token_urlsafe(18)}"
    api(
        "POST",
        f"{MEMORY_URL}/auth/signup",
        expected=201,
        payload={
            "username": other_username,
            "date_of_birth": "2010-06-10",
            "password": other_password,
            "confirm_password": other_password,
        },
    )
    other_token = api(
        "POST",
        f"{MEMORY_URL}/auth/login",
        payload={"username": other_username, "password": other_password},
    ).json()["token"]
    api(
        "GET",
        f"{TUTOR_URL}/tutor/{first_thread}",
        token=other_token,
        expected=403,
    )

    profile = api("GET", f"{TUTOR_URL}/profile", token=token).json()
    require(profile.get("username") == username, "profile identity drifted")

    final = snapshot_counts()
    require(final["tutor_threads"] >= baseline["tutor_threads"] + 2, "Tutor threads missing")
    require(final["policy_updates"] >= after_one["policy_updates"] + 1, "second policy update missing")
    require(final["student_attempts"] >= after_one["student_attempts"] + 1, "second BKT update missing")
    require(snapshot_counts(username)["accounts"] == 1, "primary account was not durable")

    api("POST", f"{MEMORY_URL}/auth/logout", token=token)
    api("GET", f"{MEMORY_URL}/auth/me", token=token, expected=401)
    api("POST", f"{MEMORY_URL}/auth/logout", token=other_token)
    api("GET", f"{MEMORY_URL}/auth/me", token=other_token, expected=401)

    deltas = {key: final[key] - baseline[key] for key in final}
    print("same_learner_sessions=2")
    print("token_restart_persistence=passed")
    print("student_impersonation_rejection=passed")
    print("cross_student_thread_isolation=passed")
    print("logout_revocation=passed")
    print("postgres_deltas=" + json.dumps(deltas, sort_keys=True))
    print("LIVE_POSTGRES_E2E=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"LIVE_POSTGRES_E2E=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
