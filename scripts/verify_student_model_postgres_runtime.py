"""Focused PostgreSQL runtime smoke for unchanged BKT behavior/idempotency."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import uuid


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "student-modeling"))

from core.knowledge_graph import KnowledgeGraph  # noqa: E402
from db.database import get_connection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--skill", default="Percent Of")
    args = parser.parse_args()
    os.environ["STUDENT_MODEL_PERSISTENCE"] = "postgres"
    os.environ["STUDENT_MODEL_DATABASE_URL"] = args.database_url
    student_id = f"pg_bkt_verify_{uuid.uuid4().hex[:12]}"
    event_id = f"{student_id}:event:1"
    graph = KnowledgeGraph()
    inserted = graph.record_attempt(
        student_id,
        args.skill,
        1,
        confidence=1.0,
        signal_type="verification",
        session_id=f"{student_id}:session",
        resolver_event_id=event_id,
    )
    assert inserted is True
    first = graph.update_mastery(student_id, args.skill)
    duplicate = graph.record_attempt(
        student_id,
        args.skill,
        1,
        confidence=1.0,
        signal_type="verification",
        session_id=f"{student_id}:session",
        resolver_event_id=event_id,
    )
    assert duplicate is False
    second = graph.update_mastery(student_id, args.skill)
    assert first["probability"] == second["probability"]
    with get_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM attempts WHERE resolver_event_id = ?",
            (event_id,),
        ).fetchone()["count"]
    assert int(count) == 1
    print("student_model_postgres=true")
    print("duplicate_protection=true")
    print("mastery_continuity=true")


if __name__ == "__main__":
    main()

