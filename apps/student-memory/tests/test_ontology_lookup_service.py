"""Tests for OntologyLookupService (UUID lookup, skill code, alias, candidate search)."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

from alembic import command
from alembic.config import Config
import pytest

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
from src.ontology.ontology_lookup_service import (
    CanonicalSkillDetail,
    OntologyLookupService,
    get_ontology_lookup_service,
)
from src.ontology.ontology_seed_service import (
    OntologySeedService,
    generate_skill_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DEFAULT_POSTGRES_SCHEMA


# =============================================================================
# Pure Unit Tests (Fast, In-Memory Dataset Lookups)
# =============================================================================

@pytest.fixture
def lookup_service():
    return OntologyLookupService()


def test_lookup_by_skill_id(lookup_service):
    """Verify lookup by exact UUID returns expected skill details."""
    expected_uuid = generate_skill_id("math :: algebra & functions :: linear equations")
    skill = lookup_service.get_by_skill_id(expected_uuid)

    assert skill is not None
    assert skill.skill_id == expected_uuid
    assert skill.display_name == "Linear Equations"
    assert skill.category == "Algebra & Functions"
    assert skill.skill_code == "SKILL_193"


def test_lookup_by_skill_code(lookup_service):
    """Verify lookup by various skill code surface forms ('SKILL_193', 'skill_193', '193')."""
    s1 = lookup_service.get_by_skill_code("SKILL_193")
    assert s1 is not None
    assert s1.display_name == "Linear Equations"

    s2 = lookup_service.get_by_skill_code("skill_193")
    assert s2 is not None
    assert s2.skill_id == s1.skill_id

    s3 = lookup_service.get_by_skill_code("assistments:193")
    assert s3 is not None
    assert s3.skill_id == s1.skill_id

    s4 = lookup_service.get_by_skill_code("193")
    assert s4 is not None
    assert s4.skill_id == s1.skill_id


def test_lookup_by_canonical_name(lookup_service):
    """Verify lookup by canonical name."""
    skill = lookup_service.get_by_canonical_name(
        "math :: geometry & measurement :: pythagorean theorem"
    )
    assert skill is not None
    assert skill.display_name == "Pythagorean Theorem"
    assert skill.category == "Geometry & Measurement"


def test_lookup_by_alias_and_synonyms(lookup_service):
    """Verify lookup by common synonyms and surface aliases."""
    # Linear Equations synonyms
    s_linear = lookup_service.get_by_alias("solving linear equations")
    assert s_linear is not None
    assert s_linear.display_name == "Linear Equations"

    # PEMDAS -> Order of Operations All
    s_pemdas = lookup_service.get_by_alias("pemdas")
    assert s_pemdas is not None
    assert s_pemdas.display_name == "Order of Operations All"

    # GCF -> Greatest Common Factor
    s_gcf = lookup_service.get_by_alias("GCF")
    assert s_gcf is not None
    assert s_gcf.display_name == "Greatest Common Factor"

    # Pythagoras Theorem -> Pythagorean Theorem
    s_pyth = lookup_service.get_by_alias("Pythagoras Theorem")
    assert s_pyth is not None
    assert s_pyth.display_name == "Pythagorean Theorem"


def test_case_insensitivity_and_whitespace_normalization(lookup_service):
    """Verify case differences and extra whitespace resolve to the same skill."""
    s1 = lookup_service.get_by_alias("  LiNeAr   EqUaTiOnS  ")
    s2 = lookup_service.get_by_alias("linear equations")
    assert s1 is not None
    assert s2 is not None
    assert s1.skill_id == s2.skill_id


def test_unknown_lookups_return_none(lookup_service):
    """Verify unknown IDs, codes, or alias strings cleanly return None."""
    assert lookup_service.get_by_skill_id(uuid.uuid4()) is None
    assert lookup_service.get_by_skill_id("not-a-uuid") is None
    assert lookup_service.get_by_skill_code("UNKNOWN_99999") is None
    assert lookup_service.get_by_canonical_name("math :: quantum :: entanglement") is None
    assert lookup_service.get_by_alias("random unregistered text") is None
    assert lookup_service.get_by_alias("") is None
    assert lookup_service.get_by_alias("   ") is None


def test_absolute_value_skills_distinction(lookup_service):
    """Verify introductory and advanced Absolute Value resolve to separate skills."""
    std = lookup_service.get_by_alias("Absolute Value")
    adv = lookup_service.get_by_alias("Absolute Value (Advanced)")

    assert std is not None
    assert adv is not None
    assert std.skill_id != adv.skill_id
    assert std.display_name == "Absolute Value"
    assert adv.display_name == "Absolute Value (Advanced)"
    assert std.skill_code == "SKILL_85"
    assert adv.skill_code == "SKILL_163"


def test_candidate_search(lookup_service):
    """Verify search_candidates retrieves matching candidate skills with ranking."""
    # Search for slope
    slope_candidates = lookup_service.search_candidates("slope", limit=5)
    assert len(slope_candidates) > 0
    # First match should be the exact 'Slope' skill
    assert slope_candidates[0].display_name == "Slope"
    assert all("slope" in s.display_name.lower() or any("slope" in a.lower() for a in s.aliases) for s in slope_candidates)

    # Search for linear
    linear_candidates = lookup_service.search_candidates("linear", limit=5)
    assert len(linear_candidates) > 0
    assert any(s.display_name == "Linear Equations" for s in linear_candidates)

    # Empty search query returns empty list
    assert lookup_service.search_candidates("") == []
    assert lookup_service.search_candidates("   ") == []


# =============================================================================
# Live PostgreSQL Lookup Tests
# =============================================================================

@pytest.fixture(scope="module")
def seeded_postgres_db():
    if not os.environ.get(DATABASE_URL_ENV):
        pytest.skip(f"{DATABASE_URL_ENV} is not configured for ontology lookup DB tests.")

    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_postgres_engine(settings)

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    session_factory = create_session_factory(engine)
    set_session_factory(session_factory)

    # Seed the ontology
    OntologySeedService(session_factory=session_factory).seed_ontology()

    try:
        yield engine, session_factory
    finally:
        set_session_factory(None)
        command.upgrade(config, "head")
        engine.dispose()


def test_live_postgres_lookup_by_alias_and_code(seeded_postgres_db):
    """Verify PostgreSQL-backed lookup service executes queries directly against DB."""
    _, session_factory = seeded_postgres_db
    db_lookup = OntologyLookupService(session_factory=session_factory)

    # Lookup by alias from PostgreSQL
    skill_1 = db_lookup.get_by_alias("PEMDAS")
    assert skill_1 is not None
    assert skill_1.display_name == "Order of Operations All"

    # Lookup by skill code from PostgreSQL
    skill_2 = db_lookup.get_by_skill_code("SKILL_193")
    assert skill_2 is not None
    assert skill_2.display_name == "Linear Equations"

    # Lookup by UUID from PostgreSQL
    skill_3 = db_lookup.get_by_skill_id(skill_2.skill_id)
    assert skill_3 is not None
    assert skill_3.canonical_name == skill_2.canonical_name
