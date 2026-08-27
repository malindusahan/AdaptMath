"""Readiness and dependency health probe service."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
from fastapi import HTTPException, status
from sqlalchemy import inspect, text

from src.database.postgres_session import SessionFactory, get_session_factory
from src.schemas.api_common import ReadinessResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOPIC_EXTRACTOR_DIR = ARTIFACTS_DIR / "topic_extractor"
EXPECTED_ALEMBIC_REVISION = "0011_adaptmath_ingestion"
REQUIRED_TABLES = {
    "students",
    "learning_sessions",
    "canonical_skills",
    "interaction_logs",
    "short_term_memory",
    "long_term_memory",
    "concept_memory",
    "student_misconceptions",
    "completed_attempt_receipts",
}


def _topic_runtime_available() -> bool:
    """Return whether the configured topic extractor can be imported at runtime."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


class ReadinessService:
    """Evaluates readiness of database connectivity, schema migrations, and ML model artifacts."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        artifacts_dir: Path | None = None,
        topic_extractor_dir: Path | None = None,
    ):
        self._session_factory = session_factory
        self.artifacts_dir = artifacts_dir if artifacts_dir is not None else ARTIFACTS_DIR
        self.topic_extractor_dir = (
            topic_extractor_dir
            if topic_extractor_dir is not None
            else (self.artifacts_dir / "topic_extractor")
        )

    def _get_factory(self) -> SessionFactory | None:
        if self._session_factory is not None:
            return self._session_factory
        try:
            return get_session_factory()
        except Exception:
            return None

    def check_readiness(self) -> ReadinessResponse:
        """
        Deep check of service dependencies:
        1. PostgreSQL connection
        2. Schema / migrations health
        3. Topic extractor fine-tuned model artifacts
        4. Learning state classification model artifacts

        Returns ReadinessResponse if all ready, or raises HTTPException(503) with details.
        """
        statuses: dict[str, str] = {
            "database": "unavailable",
            "migrations": "unavailable",
            "topic_extractor": "unavailable",
            "learning_state_model": "unavailable",
        }
        failures: list[dict[str, str]] = []

        # 1. Database Connection & 2. Migrations
        factory = self._get_factory()
        if factory is not None:
            try:
                with factory() as session:
                    # Connection check
                    session.execute(text("SELECT 1"))
                    statuses["database"] = "ready"
            except Exception:
                statuses["database"] = "unavailable"
                failures.append({"component": "database", "status": "unavailable"})

            if statuses["database"] == "ready":
                try:
                    with factory() as session:
                        bind = session.get_bind()
                        inspector = inspect(bind)
                        schema = (
                            "student_memory"
                            if bind.dialect.name == "postgresql"
                            else None
                        )
                        tables = set(inspector.get_table_names(schema=schema))
                        missing_tables = REQUIRED_TABLES - tables
                        if missing_tables:
                            raise RuntimeError("Required migration tables are missing.")
                        if bind.dialect.name == "postgresql":
                            revision = session.execute(
                                text("SELECT version_num FROM alembic_version")
                            ).scalar_one_or_none()
                            if revision != EXPECTED_ALEMBIC_REVISION:
                                raise RuntimeError("Database migration revision is stale.")
                        statuses["migrations"] = "ready"
                except Exception:
                    statuses["migrations"] = "unavailable"
                    failures.append({"component": "migrations", "status": "unavailable"})
            else:
                statuses["migrations"] = "unavailable"
                failures.append({"component": "migrations", "status": "unavailable"})
        else:
            failures.append({"component": "database", "status": "unavailable"})

        # 3. Topic Extractor Artifacts
        ft_model = (
            self.topic_extractor_dir / "minilm_finetuned_v2"
            if (self.topic_extractor_dir / "minilm_finetuned_v2").exists()
            else self.topic_extractor_dir / "minilm_finetuned"
        )
        centroids = (
            self.topic_extractor_dir / "minilm_finetuned_v2_skill_centroids.npz"
            if (self.topic_extractor_dir / "minilm_finetuned_v2_skill_centroids.npz").exists()
            else self.topic_extractor_dir / "minilm_finetuned_skill_centroids.npz"
        )
        thresholds = self.topic_extractor_dir / "confidence_thresholds.json"

        if (
            ft_model.exists()
            and centroids.exists()
            and thresholds.exists()
            and _topic_runtime_available()
        ):
            statuses["topic_extractor"] = "ready"
        else:
            statuses["topic_extractor"] = "unavailable"
            failures.append({"component": "topic_extractor", "status": "unavailable"})

        # 4. Learning State Model Artifacts
        ls_model = self.artifacts_dir / "learning_state_model.joblib"
        ls_imputer = self.artifacts_dir / "learning_state_imputer.joblib"
        ls_meta = self.artifacts_dir / "learning_state_model_metadata.json"

        if ls_model.exists() and ls_imputer.exists() and ls_meta.exists():
            statuses["learning_state_model"] = "ready"
        else:
            statuses["learning_state_model"] = "unavailable"
            failures.append({"component": "learning_state_model", "status": "unavailable"})

        # If any component failed, raise HTTP 503 with standardized envelope details
        if failures:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "SERVICE_NOT_READY",
                    "message": "Memory service is not ready.",
                    "details": failures,
                },
            )

        return ReadinessResponse(
            status="ready",
            service="student-personalization-memory",
            database=statuses["database"],
            migrations=statuses["migrations"],
            topic_extractor=statuses["topic_extractor"],
            learning_state_model=statuses["learning_state_model"],
        )


# Singleton factory cache
_READINESS_SERVICE: ReadinessService | None = None


def get_readiness_service(
    session_factory: SessionFactory | None = None,
) -> ReadinessService:
    global _READINESS_SERVICE
    if _READINESS_SERVICE is None or session_factory is not None:
        _READINESS_SERVICE = ReadinessService(session_factory=session_factory)
    return _READINESS_SERVICE
