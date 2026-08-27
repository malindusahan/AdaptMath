"""Script to safely clear all student data, interactions, sessions, accounts, and memories while keeping the database schema and 111 canonical skills ontology intact."""

from __future__ import annotations

from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA
from src.database.postgres_session import get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_seed_service import OntologySeedService
from sqlalchemy import text


def reset_data():
    schema = DEFAULT_POSTGRES_SCHEMA
    session_factory = get_session_factory()

    with UnitOfWork(session_factory) as uow:
        print(f"Cleaning all student data and memory projections from schema '{schema}'...")

        # Truncate all student data and memory tables
        tables_to_clear = [
            "repair_outcomes",
            "topic_extraction_logs",
            "current_learning_state",
            "learning_state_snapshots",
            "student_misconceptions",
            "short_term_memory",
            "concept_memory",
            "long_term_memory",
            "interaction_logs",
            "learning_sessions",
            "user_accounts",
            "students",
        ]

        for table in tables_to_clear:
            uow.session.execute(text(f"TRUNCATE TABLE {schema}.{table} CASCADE;"))
            print(f"  [OK] Truncated {table}")

        uow.commit()

    # Verify or reseed canonical skills ontology (111 skills)
    seed_service = OntologySeedService(session_factory=session_factory)
    seed_summary = seed_service.seed_ontology()
    print(f"  [OK] Canonical skills ontology verified: {seed_summary.total_canonical_skills} skills, {seed_summary.total_skill_aliases} aliases ready.")

    # Print final counts
    with UnitOfWork(session_factory) as uow:
        print("\n--- Current Database State ---")
        all_tables = uow.session.execute(text(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}' AND table_type='BASE TABLE'"
        )).fetchall()

        for t in all_tables:
            cnt = uow.session.execute(text(f"SELECT count(*) FROM {schema}.{t[0]}")).scalar()
            print(f"  {t[0]}: {cnt} rows")


if __name__ == "__main__":
    reset_data()
