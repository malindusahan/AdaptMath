"""Bounded read model for the authenticated learner profile."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable

from app.schemas.student_profile import (
    ProfileFocusNext,
    ProfileRecentSession,
    ProfileSkill,
    ProfileWeeklySummary,
    StudentProfileResponse,
)

logger = logging.getLogger(__name__)
RecommendationBuilder = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _student_model_root() -> Path:
    workspace = Path(__file__).resolve().parents[4]
    return Path(os.getenv("STUDENT_MODEL_REPO", workspace / "student-modeling")).resolve()


@lru_cache(maxsize=2)
def _curriculum_for(root_text: str):
    root = Path(root_text)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from core.curriculum import load_curriculum

    return load_curriculum(
        root / "models" / "skill_prerequisites.json",
        bkt_params_path=root / "models" / "bkt_params.json",
    )


def _official_learning_path(graph: list[dict[str, Any]]) -> dict[str, Any]:
    root = _student_model_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from core.learning_path import generate_learning_path

    return generate_learning_path(graph, curriculum=_curriculum_for(root_text))


def _iso_utc(value: str | None) -> str | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _trend(current: float, previous: float | None) -> str:
    if previous is None:
        return "not_available"
    if current > previous:
        return "improving"
    if current < previous:
        return "declining"
    return "steady"


def _focus_reason(entry: dict[str, Any]) -> str:
    if entry.get("is_regression"):
        return "Recent mastery evidence indicates that revisiting this skill would be useful."
    status = entry.get("mastery_status")
    if status == "weak":
        return "This prerequisite-ready skill currently needs more practice."
    if status == "partial":
        return "You are developing this skill; another lesson can help strengthen it."
    return "This is the next prerequisite-ready skill in your learning path."


class StudentProfileReader:
    """Read only BKT-backed learner data without initializing scientific writers."""

    def __init__(
        self,
        db_path: Path | None = None,
        recommendation_builder: RecommendationBuilder = _official_learning_path,
    ) -> None:
        root = _student_model_root()
        self.db_path = (db_path or root / "data" / "meta_agent.db").resolve()
        self.recommendation_builder = recommendation_builder

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.db_path.as_posix()}?mode=ro",
            uri=True,
            timeout=2.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def read(self, *, student_id: str, username: str, age: int | None) -> StudentProfileResponse:
        if not self.db_path.is_file():
            return self._empty(username=username, age=age, graph=[])

        with self._connect() as connection:
            mastery_rows = connection.execute(
                """
                SELECT skill_name, mastery_probability, mastery_label,
                       previous_mastery_probability, last_updated
                FROM mastery WHERE student_id = ?
                ORDER BY last_updated DESC, skill_name ASC LIMIT 100
                """,
                (student_id,),
            ).fetchall()
            session_rows = connection.execute(
                """
                SELECT session_id, processed_at FROM sessions
                WHERE student_id = ? AND session_id NOT LIKE '%:dialogue:%'
                ORDER BY processed_at DESC LIMIT 50
                """,
                (student_id,),
            ).fetchall()
            session_ids = [str(row["session_id"]) for row in session_rows]
            attempt_rows: list[sqlite3.Row] = []
            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                attempt_rows = connection.execute(
                    f"""
                    SELECT session_id, skill_name, correct, created_at
                    FROM attempts
                    WHERE student_id = ? AND session_id IN ({placeholders})
                    ORDER BY created_at DESC LIMIT 500
                    """,
                    (student_id, *session_ids),
                ).fetchall()

        graph = [
            {
                "skill": str(row["skill_name"]),
                "mastery_probability": float(row["mastery_probability"]),
                "mastery_label": str(row["mastery_label"]),
                "previous_mastery_probability": (
                    None if row["previous_mastery_probability"] is None
                    else float(row["previous_mastery_probability"])
                ),
                "last_updated": row["last_updated"],
            }
            for row in mastery_rows
        ]
        skills = [
            ProfileSkill(
                skill=item["skill"],
                mastery_probability=item["mastery_probability"],
                mastery_status=item["mastery_label"],
                previous_mastery_probability=item["previous_mastery_probability"],
                trend=_trend(item["mastery_probability"], item["previous_mastery_probability"]),
                last_practiced_at=_iso_utc(item["last_updated"]),
            )
            for item in graph
        ]

        attempts_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in attempt_rows:
            attempts_by_session[str(row["session_id"])].append(row)

        summaries: list[ProfileRecentSession] = []
        weekly_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        weekly_sessions: list[ProfileRecentSession] = []
        for row in session_rows:
            rows = attempts_by_session.get(str(row["session_id"]), [])
            completed_at = _iso_utc(str(row["processed_at"]))
            if completed_at is None:
                continue
            correct = sum(int(item["correct"]) for item in rows)
            summary = ProfileRecentSession(
                completed_at=completed_at,
                skills=sorted({str(item["skill_name"]) for item in rows}),
                questions_answered=len(rows),
                correct_answers=correct,
                incorrect_answers=len(rows) - correct,
            )
            summaries.append(summary)
            if datetime.fromisoformat(completed_at.replace("Z", "+00:00")) >= weekly_cutoff:
                weekly_sessions.append(summary)

        focus = self._focus(graph)
        return StudentProfileResponse(
            username=username,
            age=age,
            recent_session=summaries[0] if summaries else None,
            weekly_summary=ProfileWeeklySummary(
                sessions=len(weekly_sessions),
                skills_practiced=len({skill for item in weekly_sessions for skill in item.skills}),
                questions_answered=sum(item.questions_answered for item in weekly_sessions),
                correct_answers=sum(item.correct_answers for item in weekly_sessions),
            ),
            focus_next=focus,
            skills=skills,
            total_practiced_skills=len(skills),
        )

    def _focus(self, graph: list[dict[str, Any]]) -> ProfileFocusNext | None:
        try:
            path = self.recommendation_builder(graph)
            recommended = path.get("recommended_order") or []
            if not recommended:
                return None
            entry = recommended[0]
            return ProfileFocusNext(skill=str(entry["skill"]), reason=_focus_reason(entry))
        except Exception:
            logger.exception("student_profile_recommendation_unavailable")
            return None

    def _empty(self, *, username: str, age: int | None, graph: list[dict[str, Any]]) -> StudentProfileResponse:
        return StudentProfileResponse(
            username=username,
            age=age,
            weekly_summary=ProfileWeeklySummary(
                sessions=0, skills_practiced=0, questions_answered=0, correct_answers=0
            ),
            focus_next=self._focus(graph),
            skills=[],
            total_practiced_skills=0,
        )


@lru_cache(maxsize=1)
def get_student_profile_reader() -> StudentProfileReader:
    return StudentProfileReader()
