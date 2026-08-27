"""Deterministic compatibility resolver for mapping topic and subtopic to CanonicalSkill."""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.core import CanonicalSkill, SkillAlias


def normalize_skill_name(topic: str, subtopic: str | None) -> str:
    """Deterministically format canonical skill name from topic and subtopic."""
    parts = [" ".join(topic.strip().split()).lower()]
    if subtopic and subtopic.strip():
        parts.append(" ".join(subtopic.strip().split()).lower())
    return " :: ".join(parts)


def format_skill_display_name(topic: str, subtopic: str | None) -> str:
    """Format human-readable display name for topic and subtopic."""
    clean_topic = topic.strip()
    if subtopic and subtopic.strip():
        return f"{clean_topic} / {subtopic.strip()}"
    return clean_topic


class SkillContextResolver:
    """Compatibility resolver bridging legacy topic/subtopic strings to PostgreSQL CanonicalSkills."""

    @staticmethod
    def resolve_skill(
        session: Session,
        topic: str,
        subtopic: str | None,
    ) -> CanonicalSkill | None:
        """Find an existing canonical skill by topic and optional subtopic, including aliases."""
        clean_topic = topic.strip().lower()
        canonical_name = normalize_skill_name(topic, subtopic)

        # 1. Check aliases first (maps legacy names like "Linear equations" to canonical ontology)
        alias_candidates = [canonical_name]
        if subtopic and subtopic.strip():
            alias_candidates.append(subtopic.strip().lower())
        alias_candidates.append(clean_topic)

        for cand in alias_candidates:
            alias = session.scalar(
                select(SkillAlias).where(
                    (SkillAlias.normalized_alias == cand) | (SkillAlias.alias == cand)
                )
            )
            if alias is not None:
                resolved = session.get(CanonicalSkill, alias.skill_id)
                if resolved is not None:
                    return resolved

        # 2. Check exact canonical name match
        skill = session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.canonical_name == canonical_name
            )
        )
        if skill is not None:
            return skill

        skill = session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.canonical_name == clean_topic
            )
        )
        if skill is not None:
            return skill

        # 3. Check display name case-insensitive match
        skill = session.scalar(
            select(CanonicalSkill).where(
                CanonicalSkill.display_name.ilike(topic.strip())
            )
        )
        if skill is not None:
            return skill

        return None

    @staticmethod
    def resolve_or_create_skill(
        session: Session,
        topic: str,
        subtopic: str | None,
    ) -> CanonicalSkill:
        """Find or create a canonical skill by topic and optional subtopic with concurrency safety."""
        existing = SkillContextResolver.resolve_skill(session, topic, subtopic)
        if existing is not None:
            return existing

        canonical_name = normalize_skill_name(topic, subtopic)
        display_name = format_skill_display_name(topic, subtopic)

        try:
            with session.begin_nested():
                skill = CanonicalSkill(
                    skill_id=uuid.uuid4(),
                    canonical_name=canonical_name,
                    display_name=display_name,
                    ontology_version="1.0",
                    is_active=True,
                )
                session.add(skill)
                session.flush()
                return skill
        except Exception:
            # Handle race condition where another concurrent worker inserted the skill
            existing = SkillContextResolver.resolve_skill(session, topic, subtopic)
            if existing is not None:
                return existing
            raise
