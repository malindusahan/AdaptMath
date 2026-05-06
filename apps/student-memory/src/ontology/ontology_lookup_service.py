"""Ontology lookup service for querying canonical skills and aliases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.core import CanonicalSkill, SkillAlias
from src.database.postgres_config import DATABASE_URL_ENV
from src.database.postgres_session import (
    SessionFactory,
    get_session_factory,
)
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    generate_skill_id,
    normalize_token,
)


@dataclass(frozen=True)
class CanonicalSkillDetail:
    """Rich domain representation of a canonical skill in the ontology."""

    skill_id: uuid.UUID
    skill_code: str
    canonical_name: str
    display_name: str
    category: str | None
    description: str | None
    ontology_version: str
    is_active: bool
    aliases: list[str]


class OntologyLookupService:
    """Lookup and search service for canonical skills and aliases."""

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
        self._in_memory_index: dict[str, Any] | None = None

    def _ensure_in_memory_index(self) -> dict[str, Any]:
        """Lazy-load and index the JSON ontology dataset for fast local/offline resolution."""
        if self._in_memory_index is not None:
            return self._in_memory_index

        if not self.ontology_file_path.exists():
            self._in_memory_index = {
                "by_id": {},
                "by_code": {},
                "by_name": {},
                "by_display": {},
                "by_alias": {},
                "all_skills": [],
            }
            return self._in_memory_index

        with open(self.ontology_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        by_id: dict[uuid.UUID, CanonicalSkillDetail] = {}
        by_code: dict[str, CanonicalSkillDetail] = {}
        by_name: dict[str, CanonicalSkillDetail] = {}
        by_display: dict[str, CanonicalSkillDetail] = {}
        by_alias: dict[str, CanonicalSkillDetail] = {}
        all_skills: list[CanonicalSkillDetail] = []

        for item in raw_data:
            c_name = item["canonical_name"].strip()
            sk_id = generate_skill_id(c_name)
            sk_code = item.get("skill_code", f"SKILL_{item.get('assistments_skill_id', '')}")
            detail = CanonicalSkillDetail(
                skill_id=sk_id,
                skill_code=sk_code,
                canonical_name=c_name,
                display_name=item["display_name"],
                category=item.get("category"),
                description=item.get("description"),
                ontology_version=str(item.get("ontology_version", "1.0")),
                is_active=bool(item.get("is_active", True)),
                aliases=list(item.get("aliases", [])),
            )
            by_id[sk_id] = detail
            by_code[normalize_token(sk_code)] = detail
            if "assistments_skill_id" in item:
                by_code[str(item["assistments_skill_id"])] = detail
            by_name[normalize_token(c_name)] = detail
            by_display[normalize_token(detail.display_name)] = detail
            all_skills.append(detail)

            for a in detail.aliases:
                norm_a = normalize_token(a)
                if norm_a:
                    by_alias[norm_a] = detail

        self._in_memory_index = {
            "by_id": by_id,
            "by_code": by_code,
            "by_name": by_name,
            "by_display": by_display,
            "by_alias": by_alias,
            "all_skills": all_skills,
        }
        return self._in_memory_index

    def get_by_skill_id(
        self,
        skill_id: uuid.UUID | str,
        session: Session | None = None,
    ) -> CanonicalSkillDetail | None:
        """Retrieve a canonical skill by its unique UUID."""
        if isinstance(skill_id, str):
            try:
                skill_id = uuid.UUID(skill_id)
            except ValueError:
                return None

        # Check DB if session or session_factory is active
        if session is not None:
            return self._db_get_by_skill_id(session, skill_id)
        if self.session_factory is not None:
            with UnitOfWork(self.session_factory) as uow:
                return self._db_get_by_skill_id(uow.session, skill_id)

        index = self._ensure_in_memory_index()
        return index["by_id"].get(skill_id)

    def get_by_skill_code(
        self,
        skill_code: str,
        session: Session | None = None,
    ) -> CanonicalSkillDetail | None:
        """Retrieve a canonical skill by its code (e.g. 'SKILL_193', 'skill_193', '193')."""
        if not skill_code or not skill_code.strip():
            return None

        norm_code = normalize_token(skill_code)

        # Alias lookup handles assistments codes as well
        return self.get_by_alias(norm_code, session=session)

    def get_by_canonical_name(
        self,
        canonical_name: str,
        session: Session | None = None,
    ) -> CanonicalSkillDetail | None:
        """Retrieve a canonical skill by exact or normalized canonical name."""
        if not canonical_name or not canonical_name.strip():
            return None

        norm_name = normalize_token(canonical_name)

        if session is not None:
            return self._db_get_by_canonical_name(session, norm_name)
        if self.session_factory is not None:
            with UnitOfWork(self.session_factory) as uow:
                return self._db_get_by_canonical_name(uow.session, norm_name)

        index = self._ensure_in_memory_index()
        return index["by_name"].get(norm_name)

    def get_by_alias(
        self,
        alias: str,
        session: Session | None = None,
    ) -> CanonicalSkillDetail | None:
        """Retrieve a canonical skill by alias, display name, or normalized surface form."""
        if not alias or not alias.strip():
            return None

        norm_alias = normalize_token(alias)

        if session is not None:
            return self._db_get_by_alias(session, norm_alias)
        if self.session_factory is not None:
            with UnitOfWork(self.session_factory) as uow:
                return self._db_get_by_alias(uow.session, norm_alias)

        index = self._ensure_in_memory_index()
        # Direct alias index
        if norm_alias in index["by_alias"]:
            return index["by_alias"][norm_alias]
        # Check by code
        if norm_alias in index["by_code"]:
            return index["by_code"][norm_alias]
        # Check by name
        if norm_alias in index["by_name"]:
            return index["by_name"][norm_alias]
        if norm_alias in index["by_display"]:
            return index["by_display"][norm_alias]

        return None

    def search_candidates(
        self,
        query: str,
        limit: int = 10,
        session: Session | None = None,
    ) -> list[CanonicalSkillDetail]:
        """Search candidate skills matching keywords or substrings."""
        if not query or not query.strip():
            return []

        norm_query = normalize_token(query)
        index = self._ensure_in_memory_index()
        results: list[CanonicalSkillDetail] = []
        seen_ids: set[uuid.UUID] = set()

        # 1. Exact alias / code / name matches first
        exact_match = self.get_by_alias(norm_query, session=session)
        if exact_match is not None and exact_match.skill_id not in seen_ids:
            results.append(exact_match)
            seen_ids.add(exact_match.skill_id)

        # 2. Substring search across all skills in index
        for skill in index["all_skills"]:
            if len(results) >= limit:
                break
            if skill.skill_id in seen_ids:
                continue

            # Check if query matches display name, canonical name, or any alias
            if (
                norm_query in normalize_token(skill.display_name)
                or norm_query in normalize_token(skill.canonical_name)
                or any(norm_query in normalize_token(a) for a in skill.aliases)
            ):
                results.append(skill)
                seen_ids.add(skill.skill_id)

        return results[:limit]

    def list_all_skills(self) -> list[CanonicalSkillDetail]:
        """Return all canonical skills in the ontology."""
        index = self._ensure_in_memory_index()
        return list(index.get("all_skills", []))

    # -------------------------------------------------------------------------
    # Internal Database Helpers
    # -------------------------------------------------------------------------

    def _build_detail_from_db_model(
        self,
        session: Session,
        skill: CanonicalSkill,
    ) -> CanonicalSkillDetail:
        aliases_db = session.scalars(
            select(SkillAlias.alias).where(SkillAlias.skill_id == skill.skill_id)
        ).all()
        # Extract category from canonical name if available ("math :: category :: name")
        parts = [p.strip() for p in skill.canonical_name.split("::")]
        category = parts[1].title() if len(parts) >= 3 else None

        index = self._ensure_in_memory_index()
        mem_item = index["by_id"].get(skill.skill_id)
        skill_code = mem_item.skill_code if mem_item else f"SKILL_{skill.skill_id.hex[:8]}"

        return CanonicalSkillDetail(
            skill_id=skill.skill_id,
            skill_code=skill_code,
            canonical_name=skill.canonical_name,
            display_name=skill.display_name,
            category=mem_item.category if mem_item else category,
            description=skill.description,
            ontology_version=skill.ontology_version,
            is_active=skill.is_active,
            aliases=list(aliases_db) if aliases_db else (mem_item.aliases if mem_item else []),
        )

    def _db_get_by_skill_id(
        self,
        session: Session,
        skill_id: uuid.UUID,
    ) -> CanonicalSkillDetail | None:
        skill = session.get(CanonicalSkill, skill_id)
        if skill is None:
            return None
        return self._build_detail_from_db_model(session, skill)

    def _db_get_by_canonical_name(
        self,
        session: Session,
        norm_name: str,
    ) -> CanonicalSkillDetail | None:
        skill = session.scalar(
            select(CanonicalSkill).where(CanonicalSkill.canonical_name == norm_name)
        )
        if skill is None:
            return None
        return self._build_detail_from_db_model(session, skill)

    def _db_get_by_alias(
        self,
        session: Session,
        norm_alias: str,
    ) -> CanonicalSkillDetail | None:
        # Check direct alias match
        alias_record = session.scalar(
            select(SkillAlias).where(SkillAlias.normalized_alias == norm_alias)
        )
        if alias_record is not None:
            skill = session.get(CanonicalSkill, alias_record.skill_id)
            if skill is not None:
                return self._build_detail_from_db_model(session, skill)

        display_match = session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.display_name.ilike(norm_alias)
            )
        )
        if display_match is not None:
            return self._build_detail_from_db_model(session, display_match)

        # Fallback to direct canonical name check
        return self._db_get_by_canonical_name(session, norm_alias)


_default_lookup_service: OntologyLookupService | None = None


def get_ontology_lookup_service() -> OntologyLookupService:
    """Retrieve global default OntologyLookupService."""
    global _default_lookup_service
    if _default_lookup_service is None:
        _default_lookup_service = OntologyLookupService()
    return _default_lookup_service
