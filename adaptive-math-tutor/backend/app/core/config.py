from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ENV_FILE = BACKEND_ROOT / ".env"
SHARED_STUDENT_MODEL_ENV_FILE = WORKSPACE_ROOT / "student-modeling" / ".env"


class Settings(BaseSettings):
    app_name: str = "AdaptMath API"
    app_env: str = "development"
    log_level: str = "INFO"

    profile_api_url: str
    memory_api_url: str
    memory_enabled: bool = True
    memory_service_api_key: SecretStr | None = None
    memory_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    memory_topic_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    memory_history_limit: int = Field(default=5, ge=1, le=10)

    # Legacy field retained for backwards-compatible local .env files.
    # The active tutoring workflow no longer uses a local/external
    # generic Strategy service.
    strategy_api_url: str | None = None

    # Omash owns pedagogical-move selection. This remains optional
    # until the real teammate API contract is available and integrated.
    move_selector_api_url: str | None = None

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-3.6-flash"
    gemini_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    gemini_max_retries: int = Field(default=0, ge=0, le=2)
    planning_complexity_threshold: float = Field(default=0.5, gt=0, lt=1)

    @field_validator("gemini_api_key")
    @classmethod
    def require_nonempty_gemini_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("GEMINI_API_KEY must not be empty.")
        return value

    # Persistent LangGraph checkpoint database. Relative paths are
    # resolved from the backend project root.
    checkpoint_db_path: str = "runtime/adaptmath_checkpoints.sqlite3"
    tutor_persistence: Literal["sqlite", "postgres"] = "sqlite"
    adaptmath_database_url: SecretStr | None = None
    tutor_persistence_schema: str = "tutor"

    @field_validator("tutor_persistence_schema")
    @classmethod
    def require_tutor_schema(cls, value: str) -> str:
        if value != "tutor":
            raise ValueError("TUTOR_PERSISTENCE_SCHEMA must be exactly 'tutor'.")
        return value

    # Browser clients are allowed only from explicitly configured origins.
    # Keep this comma-separated in .env so local and deployed frontends can
    # be configured without changing application code.
    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )
    cors_allow_credentials: bool = True

    @property
    def allowed_cors_origins(self) -> list[str]:
        """Return normalized, unique browser origins from configuration."""
        origins: list[str] = []

        for raw_origin in self.cors_origins.split(","):
            origin = raw_origin.strip().rstrip("/")

            if not origin:
                continue

            if origin == "*":
                raise ValueError(
                    "CORS wildcard '*' is not allowed. Configure explicit origins."
                )

            if origin not in origins:
                origins.append(origin)

        if not origins:
            raise ValueError(
                "At least one explicit CORS origin must be configured."
            )

        return origins

    model_config = SettingsConfigDict(
        # The shared student-modeling file is loaded last so both projects use
        # the same Gemini credential. Real process environment variables still
        # take precedence over dotenv files under pydantic-settings semantics.
        env_file=(BACKEND_ENV_FILE, SHARED_STUDENT_MODEL_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
