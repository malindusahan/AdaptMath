"""Import versioned LinTS policy JSON snapshots without mutating policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
MOVE_ROOT = WORKSPACE / "pedagogical-move-selection"
sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.postgres_repository import (  # noqa: E402
    policy_key_for_path,
    save_policy_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--policy-key")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.policy_key and len(args.paths) != 1:
        parser.error("--policy-key is valid only with one path")
    if args.database_url:
        os.environ["RESEARCH_DATABASE_URL"] = args.database_url
    os.environ["RESEARCH_PERSISTENCE"] = "postgres"
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Policy state must be an object: {path}")
        inner = payload.get("policy_state")
        state = inner if isinstance(inner, dict) else payload
        serialized = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        key = args.policy_key or policy_key_for_path(path)
        print(
            f"policy={key} schema={payload.get('schema_version', state.get('schema_version'))} "
            f"dimension={state.get('context_dim', state.get('context_dimension'))} "
            f"updates={state.get('total_updates', state.get('update_count', 0))} "
            f"sha256={digest} dry_run={str(args.dry_run).lower()}"
        )
        if not args.dry_run:
            save_policy_snapshot(
                policy_key=key,
                payload=payload,
                policy_class=str(payload.get("policy_class", "DirectTurnLinTS")),
            )


if __name__ == "__main__":
    main()
