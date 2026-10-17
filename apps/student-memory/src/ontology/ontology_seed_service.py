"""Idempotent seed and import service for Canonical Skill Ontology and Aliases."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models.core import CanonicalSkill, SkillAlias
from src.database.postgres_config import DATABASE_URL_ENV
from src.database.postgres_session import (
    SessionFactory,
    get_session_factory,
    session_scope,
)
from src.database.unit_of_work import UnitOfWork


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_JSON_PATH = (
    PROJECT_ROOT / "data" / "processed" / "canonical_skill_ontology.json"
)

ONTOLOGY_NAMESPACE = uuid.UUID("a7b8c9d0-1234-5678-90ab-cdef12345678")


def generate_skill_id(canonical_name: str) -> uuid.UUID:
    """Generate a stable deterministic UUID for a canonical skill name."""
    return uuid.uuid5(ONTOLOGY_NAMESPACE, f"skill:{canonical_name.strip().lower()}")


def generate_alias_id(normalized_alias: str) -> uuid.UUID:
    """Generate a stable deterministic UUID for a normalized alias."""
    return uuid.uuid5(ONTOLOGY_NAMESPACE, f"alias:{normalized_alias.strip().lower()}")


def normalize_token(text: str) -> str:
    """Normalize token string to collapsed lowercase."""
    return " ".join(text.strip().split()).lower()


@dataclass(frozen=True)
class OntologySeedSummary:
    """Summary of ontology seeding operation."""

    skills_processed: int
    skills_inserted_or_updated: int
    aliases_processed: int
    aliases_inserted_or_updated: int
    total_canonical_skills: int
    total_skill_aliases: int
    is_idempotent_verified: bool


class OntologySeedError(Exception):
    """Raised when ontology import or validation fails."""


class OntologySeedService:
    """Service to load and sync canonical skill ontology data into PostgreSQL."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        ontology_file_path: Path | str | None = None,
    ):
        self.session_factory = session_factory
        self.ontology_file_path = Path(
            ontology_file_path
            if ontology_file_path is not None
            else DEFAULT_ONTOLOGY_JSON_PATH
        )

    def load_ontology_data(self) -> list[dict[str, Any]]:
        """Load and validate JSON ontology dataset."""
        if not self.ontology_file_path.exists():
            raise OntologySeedError(
                f"Ontology JSON dataset not found at: {self.ontology_file_path}"
            )

        try:
            with open(self.ontology_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise OntologySeedError(
                f"Failed to read/parse ontology JSON: {exc}"
            ) from exc

        if not isinstance(data, list) or len(data) == 0:
            raise OntologySeedError(
                "Ontology JSON must contain a non-empty list of skills."
            )

        return data

    def seed_ontology(
        self,
        session: Session | None = None,
    ) -> OntologySeedSummary:
        """
        Idempotently insert or update all canonical skills and aliases.

        If a session is provided, operations execute within that session's transaction.
        Otherwise, a new UnitOfWork transaction is opened with session_factory.
        """
        data = self.load_ontology_data()

        if session is not None:
            return self._seed_with_session(session, data)

        factory = (
            self.session_factory
            if self.session_factory is not None
            else get_session_factory()
        )

        with UnitOfWork(factory) as uow:
            summary = self._seed_with_session(uow.session, data)
            uow.commit()
            return summary

    def _seed_with_session(
        self,
        session: Session,
        data: list[dict[str, Any]],
    ) -> OntologySeedSummary:
        """Internal execution logic operating on a specific database session."""
        bind = session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            from src.database.postgres_config import DEFAULT_POSTGRES_SCHEMA

            CanonicalSkill.__table__.schema = DEFAULT_POSTGRES_SCHEMA
            SkillAlias.__table__.schema = DEFAULT_POSTGRES_SCHEMA

        skills_count = 0
        aliases_count = 0

        # Pass 1: Upsert all canonical skills
        for item in data:
            canonical_name = item.get("canonical_name", "").strip()
            display_name = item.get("display_name", "").strip()
            if not canonical_name or not display_name:
                raise OntologySeedError(
                    f"Skill record missing required canonical_name or display_name: {item}"
                )

            skill_id = generate_skill_id(canonical_name)
            description = item.get("description")
            ontology_version = str(item.get("ontology_version", "1.0"))
            is_active = bool(item.get("is_active", True))

            # Upsert canonical skill
            stmt = (
                insert(CanonicalSkill)
                .values(
                    skill_id=skill_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    description=description,
                    ontology_version=ontology_version,
                    is_active=is_active,
                )
                .on_conflict_do_update(
                    index_elements=["canonical_name"],
                    set_={
                        "display_name": display_name,
                        "description": description,
                        "ontology_version": ontology_version,
                        "is_active": is_active,
                        "updated_at": func.now(),
                    },
                )
            )
            session.execute(stmt)
            skills_count += 1

        session.flush()

        # Build lookup of canonical_name -> skill_id from DB
        db_skills = session.scalars(select(CanonicalSkill)).all()
        name_to_id = {s.canonical_name: s.skill_id for s in db_skills}

        # Pass 2: Upsert aliases
        for item in data:
            canonical_name = item["canonical_name"].strip()
            skill_id = name_to_id[canonical_name]
            aliases = item.get("aliases", [])

            for alias_raw in aliases:
                alias_str = alias_raw.strip()
                norm_alias = normalize_token(alias_str)
                if not norm_alias:
                    continue

                alias_id = generate_alias_id(norm_alias)
                stmt = (
                    insert(SkillAlias)
                    .values(
                        alias_id=alias_id,
                        skill_id=skill_id,
                        alias=alias_str,
                        normalized_alias=norm_alias,
                        source="assistments_ontology",
                    )
                    .on_conflict_do_update(
                        index_elements=["normalized_alias"],
                        set_={
                            "skill_id": skill_id,
                            "alias": alias_str,
                            "source": "assistments_ontology",
                        },
                    )
                )
                session.execute(stmt)
                aliases_count += 1

        session.flush()

        # Query final counts in schema
        total_skills = int(
            session.scalar(select(func.count(CanonicalSkill.skill_id))) or 0
        )
        total_aliases = int(
            session.scalar(select(func.count(SkillAlias.alias_id))) or 0
        )

        return OntologySeedSummary(
            skills_processed=len(data),
            skills_inserted_or_updated=skills_count,
            aliases_processed=aliases_count,
            aliases_inserted_or_updated=aliases_count,
            total_canonical_skills=total_skills,
            total_skill_aliases=total_aliases,
            is_idempotent_verified=(total_skills == len(data)),
        )


def seed_canonical_ontology(
    session_factory: SessionFactory | None = None,
    ontology_file_path: Path | str | None = None,
) -> OntologySeedSummary:
    """Convenience entry point for seeding the canonical ontology."""
    service = OntologySeedService(
        session_factory=session_factory,
        ontology_file_path=ontology_file_path,
    )
    return service.seed_ontology()


def main():
    """CLI entry point for running the ontology seed directly."""
    print("Seeding canonical skill ontology into PostgreSQL...")
    try:
        summary = seed_canonical_ontology()
        print("=" * 60)
        print("Ontology Seed Completed Successfully!")
        print(f"Skills Processed:             {summary.skills_processed}")
        print(f"Total Canonical Skills in DB: {summary.total_canonical_skills}")
        print(f"Aliases Processed:            {summary.aliases_processed}")
        print(f"Total Skill Aliases in DB:    {summary.total_skill_aliases}")
        print(f"Idempotent Integrity Check:   {summary.is_idempotent_verified}")
        print("=" * 60)
    except Exception as exc:
        print(f"ERROR: Ontology seeding failed: {exc}")
        raise


if __name__ == "__main__":
    main()
