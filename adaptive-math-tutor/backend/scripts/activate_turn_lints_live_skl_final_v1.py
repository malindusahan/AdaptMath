"""Authorize the audited S+K+L LIVE lineage immediately before launch."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
MOVE_ROOT = ROOT / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    REWARD_MODE,
    SELECTOR_SHA256,
)
from src.self_improvement.turn_lints_live_lineage import (  # noqa: E402
    ARCHITECTURE_MANIFEST_SHA256,
    LIVE_LINEAGE_SCHEMA,
)


RESULTS = MOVE_ROOT / "results" / "turn_lints_live_skl_synthetic_warmstart_v1"
LIVE_ROOT = BACKEND / "runtime" / "turn_lints_live_skl_final_v1"
STATE = LIVE_ROOT / "policy_state.json"
MANIFEST = LIVE_ROOT / "lineage_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    checks = {
        "decision": summary.get("activation_decision") == "A",
        "all_gates": all(summary.get("predeclared_sanity_checks", {}).values()),
        "manifest_schema": manifest.get("schema_version") == LIVE_LINEAGE_SCHEMA,
        "architecture_hash": manifest.get("architecture_manifest_sha256") == ARCHITECTURE_MANIFEST_SHA256,
        "context": manifest.get("context_preset") == CONTEXT_PRESET,
        "dimension": manifest.get("context_dimension") == CONTEXT_DIMENSION,
        "reward": manifest.get("reward_mode") == REWARD_MODE,
        "selector": manifest.get("selector_sha256") == SELECTOR_SHA256,
        "state_hash": manifest.get("initial_state_sha256") == sha256(STATE),
        "exact_initial_updates": state.get("update_count") == 500,
        "exact_initial_ids": state.get("applied_update_ids") == [
            f"synthetic-init:{index:04d}" for index in range(1, 501)
        ],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"LIVE activation refused; failed checks: {failed}")
    if manifest.get("live_activated_at") is not None:
        print(f"LIVE lineage already authorized at {manifest['live_activated_at']}")
        return
    manifest["live_activated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Authorized audited LIVE lineage at {manifest['live_activated_at']}")


if __name__ == "__main__":
    main()
