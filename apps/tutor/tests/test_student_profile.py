from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient

from app.api import profile as profile_api
from app.clients.memory.memory_client import MemoryAuthenticatedUser
import app.core.user_auth as user_auth
from app.main import app
from app.schemas.student_profile import StudentProfileResponse
from app.services import student_profile_service
from app.services.student_profile_service import StudentProfileReader


STUDENT_A = MemoryAuthenticatedUser(
    user_id="user-a", username="Alex", role="STUDENT", student_id="student-a", age=15
)


class AuthClient:
    def __init__(self, user):
        self.user = user

    def authenticate_user(self, token):
        return self.user if token == "valid-a" else None


class CapturingReader:
    def __init__(self):
        self.student_ids = []

    def read(self, *, student_id, username, age):
        self.student_ids.append(student_id)
        return StudentProfileResponse(
            username=username,
            age=age,
            weekly_summary={
                "sessions": 0,
                "skills_practiced": 0,
                "questions_answered": 0,
                "correct_answers": 0,
            },
            skills=[],
            total_practiced_skills=0,
        )


def _db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE mastery (
                student_id TEXT, skill_name TEXT, mastery_probability REAL,
                mastery_label TEXT, previous_mastery_probability REAL,
                last_updated TIMESTAMP
            );
            CREATE TABLE sessions (
                session_id TEXT, student_id TEXT, processed_at TIMESTAMP,
                concept_count INTEGER
            );
            CREATE TABLE attempts (
                attempt_id INTEGER PRIMARY KEY, student_id TEXT, skill_name TEXT,
                correct INTEGER, session_id TEXT, created_at TIMESTAMP
            );
            """
        )


def test_profile_requires_authentication():
    response = TestClient(app, raise_server_exceptions=False).get("/profile")
    assert response.status_code == 401


def test_authenticated_profile_uses_only_token_student(monkeypatch):
    reader = CapturingReader()
    monkeypatch.setattr(user_auth, "get_memory_client", lambda: AuthClient(STUDENT_A))
    monkeypatch.setattr(profile_api, "get_student_profile_reader", lambda: reader)

    response = TestClient(app).get(
        "/profile?student_id=student-b",
        headers={"Authorization": "Bearer valid-a"},
    )

    assert response.status_code == 200
    assert response.json()["username"] == "Alex"
    assert reader.student_ids == ["student-a"]
    assert "student_id" not in response.json()


def test_profile_maps_real_bkt_sessions_trend_and_focus(tmp_path):
    path = tmp_path / "bkt.sqlite3"
    _db(path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO mastery VALUES (?, ?, ?, ?, ?, ?)",
            ("student-a", "Linear Equations", 0.72, "strong", 0.55, now),
        )
        connection.execute(
            "INSERT INTO mastery VALUES (?, ?, ?, ?, ?, ?)",
            ("student-a", "Fractions", 0.25, "weak", 0.30, now),
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            ("thread:1", "student-a", now, 1),
        )
        connection.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "student-a", "Linear Equations", 1, "thread:1", now),
                (2, "student-a", "Linear Equations", 0, "thread:1", now),
                (3, "student-a", "Linear Equations", 1, "thread:1", now),
            ],
        )

    def recommendation(graph):
        assert len(graph) == 2
        return {
            "recommended_order": [
                {"skill": "Fractions", "mastery_status": "weak", "is_regression": False}
            ]
        }

    profile = StudentProfileReader(path, recommendation).read(
        student_id="student-a", username="Alex", age=15
    )

    assert profile.recent_session is not None
    assert profile.recent_session.questions_answered == 3
    assert profile.recent_session.correct_answers == 2
    assert profile.weekly_summary.sessions == 1
    assert profile.weekly_summary.skills_practiced == 1
    assert profile.focus_next is not None
    assert profile.focus_next.skill == "Fractions"
    assert {item.skill: item.trend for item in profile.skills} == {
        "Fractions": "declining",
        "Linear Equations": "improving",
    }


def test_profile_uses_configured_student_model_persistence(monkeypatch, tmp_path):
    path = tmp_path / "authoritative.sqlite3"
    _db(path)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO mastery VALUES (?, ?, ?, ?, ?, ?)",
            ("student-a", "Linear Equations", 0.91, "strong", 0.72, now),
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            ("thread:1", "student-a", now, 1),
        )
        connection.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
            (1, "student-a", "Linear Equations", 1, "thread:1", now),
        )

    calls: list[Path] = []

    class ConfiguredConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, parameters=()):
            # Literal percent signs are interpreted as placeholders by
            # psycopg, so configured-persistence queries must avoid them.
            assert "%" not in sql
            return self.connection.execute(sql, parameters)

    @contextmanager
    def configured_connection(db_path):
        calls.append(db_path)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            yield ConfiguredConnection(connection)
        finally:
            connection.close()

    monkeypatch.setattr(
        student_profile_service,
        "_student_model_connection_factory",
        lambda _root: configured_connection,
    )

    profile = StudentProfileReader(
        recommendation_builder=lambda _graph: {"recommended_order": []}
    ).read(student_id="student-a", username="Alex", age=15)

    assert len(calls) == 1
    assert profile.recent_session is not None
    assert profile.recent_session.correct_answers == 1
    assert profile.skills[0].mastery_probability == 0.91


def test_new_learner_has_truthful_empty_state_and_official_starting_recommendation(tmp_path):
    path = tmp_path / "empty.sqlite3"
    _db(path)
    reader = StudentProfileReader(
        path,
        lambda graph: {
            "recommended_order": [
                {"skill": "Whole Number Addition", "mastery_status": "unseen"}
            ]
        },
    )

    profile = reader.read(student_id="new", username="New learner", age=None)

    assert profile.recent_session is None
    assert profile.weekly_summary.questions_answered == 0
    assert profile.skills == []
    assert profile.age is None
    assert profile.focus_next is not None
    assert profile.focus_next.skill == "Whole Number Addition"
