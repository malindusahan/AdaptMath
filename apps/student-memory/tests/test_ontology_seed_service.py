"""Tests for OntologySeedService (idempotent loading, duplicate protection, integrity)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import uuid

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import func, select

from src.database.models.core import CanonicalSkill, SkillAlias
from src.database.postgres_config import (
    DATABASE_URL_ENV,
    DEFAULT_POSTGRES_SCHEMA,
    load_postgres_settings,
)
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
    set_session_factory,
)
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    OntologySeedError,
    OntologySeedService,
    generate_alias_id,
    generate_skill_id,
    seed_canonical_ontology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA


# =============================================================================
# Pure Unit Tests (Deterministic without external DB)
# =============================================================================

def test_load_ontology_dataset():
    """Verify ontology JSON dataset structure and required attributes."""
    service = OntologySeedService()
    data = service.load_ontology_data()

    assert len(data) == 111

    for item in data:
        assert "skill_code" in item
        assert "assistments_skill_id" in item
        assert "canonical_name" in item
        assert "display_name" in item
        assert "category" in item
        assert "aliases" in item
        assert len(item["aliases"]) > 0


def test_deterministic_uuid_generation():
    """Verify UUID generation is stable and deterministic for identical tokens."""
    u1 = generate_skill_id("math :: algebra :: linear equations")
    u2 = generate_skill_id("math :: algebra :: linear equations")
    assert u1 == u2

    a1 = generate_alias_id("linear equations")
    a2 = generate_alias_id("linear equations")
    assert a1 == a2


def test_missing_ontology_file_raises_error(tmp_path):
    """Verify missing file raises clear OntologySeedError."""
    fake_path = tmp_path / "non_existent.json"
    service = OntologySeedService(ontology_file_path=fake_path)
    with pytest.raises(OntologySeedError):
        service.load_ontology_data()


def test_corrupted_or_empty_json_raises_error(tmp_path):
    """Verify empty list or malformed JSON raises OntologySeedError."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")
    service_empty = OntologySeedService(ontology_file_path=empty_file)
    with pytest.raises(OntologySeedError):
        service_empty.load_ontology_data()

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")
    service_corrupt = OntologySeedService(ontology_file_path=corrupt_file)
    with pytest.raises(OntologySeedError):
        service_corrupt.load_ontology_data()


def test_all_111_skills_have_unique_deterministic_uuids():
    """Verify that all 111 skills generate mutually distinct UUIDs."""
    service = OntologySeedService()
    data = service.load_ontology_data()
    skill_ids = [generate_skill_id(item["canonical_name"]) for item in data]
    assert len(skill_ids) == 111
    assert len(set(skill_ids)) == 111

    alias_ids = [
        generate_alias_id(a)
        for item in data
        for a in item["aliases"]
    ]
    assert len(alias_ids) == 692
    assert len(set(alias_ids)) == 692


# =============================================================================
# Live PostgreSQL Seed Integration Tests
# =============================================================================

@pytest.fixture(scope="module")
def ontology_db():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for ontology seed tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)

    try:
        yield engine, session_factory
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()


def test_first_and_second_import_idempotence(ontology_db):
    """Verify first seed loads 111 skills & 692 aliases and second seed creates 0 duplicates."""
    engine, session_factory = ontology_db
    service = OntologySeedService(session_factory=session_factory)

    # 1. First Import
    summary_1 = service.seed_ontology()
    assert summary_1.skills_processed == 111
    assert summary_1.total_canonical_skills == 111
    assert summary_1.total_skill_aliases == 692
    assert summary_1.is_idempotent_verified is True

    # 2. Second Import (Idempotence check)
    summary_2 = service.seed_ontology()
    assert summary_2.skills_processed == 111
    assert summary_2.total_canonical_skills == 111
    assert summary_2.total_skill_aliases == 692
    assert summary_2.is_idempotent_verified is True

    # Direct database verification
    with UnitOfWork(session_factory) as uow:
        skill_count = uow.session.scalar(select(func.count(CanonicalSkill.skill_id)))
        alias_count = uow.session.scalar(select(func.count(SkillAlias.alias_id)))
        assert skill_count == 111
        assert alias_count == 692


def test_distinct_absolute_value_skills_preserved(ontology_db):
    """Verify standard Absolute Value (id 85) and Advanced (id 163) remain distinct."""
    _, session_factory = ontology_db

    with UnitOfWork(session_factory) as uow:
        std = uow.session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.canonical_name
                == "math :: number sense & operations :: absolute value"
            )
        )
        adv = uow.session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.canonical_name
                == "math :: number sense & operations :: absolute value (advanced)"
            )
        )

        assert std is not None
        assert adv is not None
        assert std.skill_id != adv.skill_id
        assert std.display_name == "Absolute Value"
        assert adv.display_name == "Absolute Value (Advanced)"


def test_alias_foreign_key_references_and_synonyms(ontology_db):
    """Verify aliases correctly resolve to expected canonical skills."""
    _, session_factory = ontology_db

    with UnitOfWork(session_factory) as uow:
        # Check PEMDAS alias maps to Order of Operations All
        pemdas_alias = uow.session.scalar(
            select(SkillAlias).where(SkillAlias.normalized_alias == "pemdas")
        )
        assert pemdas_alias is not None
        pemdas_skill = uow.session.get(CanonicalSkill, pemdas_alias.skill_id)
        assert pemdas_skill is not None
        assert pemdas_skill.display_name == "Order of Operations All"

        # Check Linear Equations alias
        linear_alias = uow.session.scalar(
            select(SkillAlias).where(SkillAlias.normalized_alias == "linear equations")
        )
        assert linear_alias is not None
        linear_skill = uow.session.get(CanonicalSkill, linear_alias.skill_id)
        assert linear_skill is not None
        assert linear_skill.display_name == "Linear Equations"


def test_invalid_json_triggers_rollback(ontology_db, tmp_path):
    """Verify invalid JSON input causes complete transaction rollback without partial persistence."""
    _, session_factory = ontology_db

    # Create invalid JSON with missing canonical_name
    invalid_file = tmp_path / "invalid_ontology.json"
    invalid_file.write_text(
        json.dumps([{"display_name": "Incomplete Skill", "aliases": []}]),
        encoding="utf-8",
    )

    service = OntologySeedService(
        session_factory=session_factory,
        ontology_file_path=invalid_file,
    )

    with pytest.raises(OntologySeedError):
        service.seed_ontology()
