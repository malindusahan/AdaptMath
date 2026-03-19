"""Final end-to-end validation for Phase 13 Canonical Skill Ontology."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
from src.ontology.ontology_lookup_service import (
    OntologyLookupService,
    get_ontology_lookup_service,
)
from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    OntologySeedService,
    generate_alias_id,
    generate_skill_id,
    normalize_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA


# =============================================================================
# Pure Unit Tests (Offline Validation against JSON Dataset)
# =============================================================================

def test_json_ontology_completeness_and_uniqueness():
    """Verify JSON dataset contains exactly 111 unique skills and 692 unique aliases."""
    with open(DEFAULT_ONTOLOGY_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 111, f"Expected 111 skills, got {len(data)}"

    canonical_names = [d["canonical_name"] for d in data]
    skill_codes = [d["skill_code"] for d in data]
    all_aliases = [normalize_token(a) for d in data for a in d["aliases"]]

    # Assert 0 duplicate canonical names
    assert len(canonical_names) == len(set(canonical_names))

    # Assert 0 duplicate skill codes
    assert len(skill_codes) == len(set(skill_codes))

    # Assert exactly 692 aliases with 0 collisions
    assert len(all_aliases) == 692
    assert len(all_aliases) == len(set(all_aliases))


def test_offline_lookup_service_validation():
    """Verify lookup service resolves canonical skills, aliases, and searches deterministically."""
    service = OntologyLookupService()

    # Exact alias resolution
    res = service.get_by_alias("pemdas")
    assert res is not None
    assert res.display_name == "Order of Operations All"

    # Skill code resolution
    res_code = service.get_by_skill_code("SKILL_193")
    assert res_code is not None
    assert res_code.display_name == "Linear Equations"

    # Absolute value distinction
    std = service.get_by_alias("Absolute Value")
    adv = service.get_by_alias("Absolute Value (Advanced)")
    assert std is not None
    assert adv is not None
    assert std.skill_id != adv.skill_id
    assert std.skill_code == "SKILL_85"
    assert adv.skill_code == "SKILL_163"

    # Unknown lookup returns None
    assert service.get_by_alias("quantum mechanics unknown") is None
    assert service.get_by_skill_id(uuid.uuid4()) is None

    # Deterministic candidate search
    candidates_1 = service.search_candidates("equations", limit=5)
    candidates_2 = service.search_candidates("equations", limit=5)
    assert len(candidates_1) == len(candidates_2)
    assert [c.skill_id for c in candidates_1] == [c.skill_id for c in candidates_2]


# =============================================================================
# Live PostgreSQL End-to-End Validation
# =============================================================================

@pytest.fixture(scope="module")
def validated_postgres_db():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for final ontology validation.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)

    # Initial seed
    seed_service = OntologySeedService(session_factory=session_factory)
    seed_service.seed_ontology()

    try:
        yield engine, session_factory, seed_service
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()


def test_postgres_ontology_final_metrics(validated_postgres_db):
    """
    Verify full database constraints and metrics:
    - 111 canonical skills
    - 692 skill aliases
    - 0 duplicate canonical names
    - 0 duplicate skill codes
    - 0 duplicate normalized aliases
    - 0 orphan aliases
    """
    _, session_factory, _ = validated_postgres_db

    with UnitOfWork(session_factory) as uow:
        # Total canonical skills
        skills_count = uow.session.scalar(select(func.count(CanonicalSkill.skill_id)))
        assert skills_count == 111

        # Total skill aliases
        aliases_count = uow.session.scalar(select(func.count(SkillAlias.alias_id)))
        assert aliases_count == 692

        # Distinct canonical names count
        distinct_names = uow.session.scalar(
            select(func.count(func.distinct(CanonicalSkill.canonical_name)))
        )
        assert distinct_names == 111

        # Distinct normalized aliases count
        distinct_aliases = uow.session.scalar(
            select(func.count(func.distinct(SkillAlias.normalized_alias)))
        )
        assert distinct_aliases == 692

        # Check for orphan aliases (aliases without a matching canonical skill)
        orphan_aliases_stmt = (
            select(func.count(SkillAlias.alias_id))
            .outerjoin(CanonicalSkill, SkillAlias.skill_id == CanonicalSkill.skill_id)
            .where(CanonicalSkill.skill_id.is_(None))
        )
        orphan_count = uow.session.scalar(orphan_aliases_stmt)
        assert orphan_count == 0, f"Found {orphan_count} orphan aliases!"


def test_reseed_idempotence_and_stability(validated_postgres_db):
    """Verify running the seed service a second time does not alter counts."""
    _, session_factory, seed_service = validated_postgres_db

    summary = seed_service.seed_ontology()
    assert summary.total_canonical_skills == 111
    assert summary.total_skill_aliases == 692
    assert summary.is_idempotent_verified is True

    with UnitOfWork(session_factory) as uow:
        skills_count = uow.session.scalar(select(func.count(CanonicalSkill.skill_id)))
        aliases_count = uow.session.scalar(select(func.count(SkillAlias.alias_id)))
        assert skills_count == 111
        assert aliases_count == 692


def test_live_postgres_lookup_service(validated_postgres_db):
    """Verify lookup service queries PostgreSQL database directly."""
    _, session_factory, _ = validated_postgres_db
    lookup = OntologyLookupService(session_factory=session_factory)

    # By alias
    pyth = lookup.get_by_alias("Pythagorean Theorem")
    assert pyth is not None
    assert pyth.display_name == "Pythagorean Theorem"
    assert pyth.category == "Geometry & Measurement"

    # By skill code
    s85 = lookup.get_by_skill_code("SKILL_85")
    s163 = lookup.get_by_skill_code("SKILL_163")
    assert s85 is not None
    assert s163 is not None
    assert s85.skill_id != s163.skill_id
    assert s85.display_name == "Absolute Value"
    assert s163.display_name == "Absolute Value (Advanced)"

    # Unknown lookup returns None
    assert lookup.get_by_alias("unknown alias string xyz") is None
    assert lookup.get_by_skill_id(uuid.uuid4()) is None
