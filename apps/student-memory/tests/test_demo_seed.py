"""Focused test for demo environment seeding and idempotency."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.models.core import Student
from src.database.models.user_account import UserAccount
from src.demo.seed_demo_data import seed_demo_environment


def test_demo_seed_creation_and_idempotency():
    """Verify demo accounts and memory records are created and can be re-run safely without duplication."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in Base.metadata.tables.values():
        table.schema = None
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # First run
    res1 = seed_demo_environment(session_factory=factory)
    assert res1["demo_new"] == "demo_new"
    assert res1["demo_developing"] == "demo_developing"
    assert res1["demo_strong"] == "demo_strong"
    assert res1["demo_support"] == "demo_support"

    with factory() as session:
        students = session.scalars(select(Student)).all()
        assert len(students) >= 4
        accounts = session.scalars(select(UserAccount)).all()
        assert len(accounts) >= 5

    # Second run (Idempotency test)
    res2 = seed_demo_environment(session_factory=factory)
    assert res2 == res1

    with factory() as session:
        students_after = session.scalars(select(Student)).all()
        assert len(students_after) == len(students)
        accounts_after = session.scalars(select(UserAccount)).all()
        assert len(accounts_after) == len(accounts)
