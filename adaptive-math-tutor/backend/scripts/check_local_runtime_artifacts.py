"""Non-destructive preflight for persistent local-demo Tutor artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.integrations.adaptive_component_coordinator import (  # noqa: E402
    _ProductionAdaptiveResources,
)


DEMO_ROOT = (BACKEND_ROOT / "runtime" / "adaptive_demo").resolve()


def _configured_demo_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured.")
    path = Path(value)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    resolved = path.resolve()
    if resolved != DEMO_ROOT and DEMO_ROOT not in resolved.parents:
        raise RuntimeError(f"{name} must remain under {DEMO_ROOT}.")
    return resolved


def _readonly_count(database: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def main() -> int:
    if os.getenv("ADAPTIVE_DATA_MODE") != "synthetic":
        raise RuntimeError("ADAPTIVE_DATA_MODE must be synthetic for local demo.")

    checkpoint = _configured_demo_path("CHECKPOINT_DB_PATH")
    policy_state = _configured_demo_path("ADAPTIVE_POLICY_STATE_PATH")
    experience_log = _configured_demo_path("ADAPTIVE_EXPERIENCE_LOG_PATH")
    student_root = _configured_demo_path("STUDENT_MODEL_REPO")
    student_database = student_root / "data" / "meta_agent.db"

    required = (checkpoint, policy_state, experience_log, student_database)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local-demo artifacts: " + ", ".join(missing))

    with policy_state.open("r", encoding="utf-8") as handle:
        policy_payload = json.load(handle)
    serialized_policy = policy_payload.get("policy_state", {})
    if not isinstance(serialized_policy, dict):
        raise RuntimeError("The local policy artifact has an invalid structure.")
    if serialized_policy.get("data_mode") != "synthetic":
        raise RuntimeError("The local policy artifact is not synthetic.")

    resources = _ProductionAdaptiveResources()
    predictor = resources._ensure_predictor()
    resources._ensure_shared()
    if resources.policy.data_mode != "synthetic":
        raise RuntimeError("Loaded LinTS policy is not synthetic.")

    checkpoint_connection = sqlite3.connect(
        f"file:{checkpoint.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        checkpoint_tables = {
            row[0]
            for row in checkpoint_connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        checkpoint_connection.close()
    if "checkpoints" not in checkpoint_tables:
        raise RuntimeError("Tutor checkpoint database is not initialized.")

    print("LOCAL RUNTIME ARTIFACTS: PASS")
    print(f"canonical_skill_count: {len(predictor.params)}")
    print(f"policy_updates_retained: {resources.policy.total_updates}")
    print(f"student_records_retained: {_readonly_count(student_database, 'students')}")
    print(f"attempt_records_retained: {_readonly_count(student_database, 'attempts')}")
    print("md6: loaded")
    print("mrb1: loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
