"""Import durable pending Turn-LinTS action ledgers into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pedagogical-move-selection"))

from src.self_improvement.postgres_repository import stage_action  # noqa: E402
from postgres_unification_common import connect  # noqa: E402


def _files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.database_url:
        os.environ["RESEARCH_DATABASE_URL"] = args.database_url
    os.environ["RESEARCH_PERSISTENCE"] = "postgres"

    action_ids: set[str] = set()
    digest = hashlib.sha256()
    files = _files(args.paths)
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Pending action must be an object: {path}")
        action_id = str(record.get("action_event_id", "")).strip()
        if not action_id:
            raise ValueError(f"Pending action has no action_event_id: {path}")
        action_ids.add(action_id)
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if not args.dry_run:
            stage_action(record)

    destination = None
    if not args.dry_run:
        connection = connect(args.database_url)
        try:
            destination = int(
                connection.execute(
                    "SELECT COUNT(*) FROM research.turn_actions "
                    "WHERE action_event_id = ANY(%s)",
                    (list(action_ids),),
                ).fetchone()[0]
            ) if action_ids else 0
        finally:
            connection.close()
        if destination != len(action_ids):
            raise RuntimeError("Not every pending action identity exists in PostgreSQL.")

    print(
        f"files={len(files)} unique_actions={len(action_ids)} "
        f"destination_matching={destination} sha256={digest.hexdigest()} "
        f"dry_run={str(args.dry_run).lower()}"
    )


if __name__ == "__main__":
    main()
