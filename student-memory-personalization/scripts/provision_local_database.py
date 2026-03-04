"""Explicitly migrate and seed the configured local/research Memory database."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from src.database.postgres_config import load_postgres_settings
from src.database.postgres_session import (
    create_postgres_engine,
    create_session_factory,
)
from src.ontology.ontology_seed_service import OntologySeedService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISION = "0013_policy_state_ordered_json"


def main() -> None:
    settings = load_postgres_settings()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")

    engine = create_postgres_engine(settings)
    factory = create_session_factory(engine)
    OntologySeedService(session_factory=factory).seed_ontology()

    with engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    tables = sorted(inspect(engine).get_table_names(schema=settings.schema))
    engine.dispose()

    if revision != EXPECTED_REVISION:
        raise RuntimeError("Provisioning completed at an unexpected revision.")
    if "completed_attempt_receipts" not in tables:
        raise RuntimeError("Completed-attempt receipt table was not provisioned.")

    print(f"database={settings.database_url.database}")
    print(f"schema={settings.schema}")
    print(f"alembic_revision={revision}")
    print("tables=" + ",".join(tables))


if __name__ == "__main__":
    main()
