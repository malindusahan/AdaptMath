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


DATA_MODE = os.getenv("ADAPTIVE_DATA_MODE", "").strip()
if DATA_MODE not in {"real", "synthetic"}:
    raise ValueError("ADAPTIVE_DATA_MODE must be exactly 'real' or 'synthetic'.")
DEMO_ROOT = Path(
    os.getenv(
        "ADAPTIVE_LOCAL_RUNTIME_ROOT",
        BACKEND_ROOT
        / "runtime"
        / ("adaptive_live_real_v1" if DATA_MODE == "real" else "adaptive_demo"),
    )
).resolve()
MD6_POLICY_ROOT = (
    BACKEND_ROOT
    / "runtime"
    / (
        "adaptive_demo_md6_real_v1"
        if DATA_MODE == "real"
        else "adaptive_demo"
    )
).resolve()
MD7_POLICY_ROOT = Path(
    os.getenv(
        "ADAPTIVE_MD7_RUNTIME_ROOT",
        BACKEND_ROOT
        / "runtime"
        / (
            "adaptive_demo_md7r1_real_v1"
            if DATA_MODE == "real"
            else "adaptive_demo_md7r1_v1"
        ),
    )
).resolve()
TURN_LINTS_ROOT = Path(
    os.getenv(
        "ADAPTIVE_TURN_LINTS_EVENT_ROOT",
        BACKEND_ROOT / "runtime" / "adaptive_turn_lints_v1_real",
    )
).resolve()


def _configured_demo_path(
    name: str,
    *,
    allowed_roots: tuple[Path, ...] = (DEMO_ROOT,),
) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured.")
    path = Path(value)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    resolved = path.resolve()
    if not any(
        resolved == root or root in resolved.parents for root in allowed_roots
    ):
        raise RuntimeError(
            f"{name} must remain under one of: "
            + ", ".join(str(root) for root in allowed_roots)
        )
    return resolved


def _readonly_count(database: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def main() -> int:
    checkpoint = _configured_demo_path("CHECKPOINT_DB_PATH")
    student_root = _configured_demo_path("STUDENT_MODEL_REPO")
    student_database = student_root / "data" / "meta_agent.db"

    resources = _ProductionAdaptiveResources()
    predictor = resources._ensure_predictor()
    resources._ensure_shared()
    policy_state = resources.policy_state_path
    experience_log = resources.experience_log_path

    required = [checkpoint, policy_state, student_database]
    if not resources.selector_mode.uses_turn_lints:
        required.append(experience_log)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local-demo artifacts: " + ", ".join(missing))

    with policy_state.open("r", encoding="utf-8") as handle:
        policy_payload = json.load(handle)
    serialized_policy = (
        policy_payload
        if resources.selector_mode.uses_turn_lints
        else policy_payload.get("policy_state", {})
    )
    if not isinstance(serialized_policy, dict):
        raise RuntimeError("The local policy artifact has an invalid structure.")
    if serialized_policy.get("data_mode") != DATA_MODE:
        raise RuntimeError(
            "The configured policy artifact does not match ADAPTIVE_DATA_MODE."
        )

    if resources.policy.data_mode != DATA_MODE:
        raise RuntimeError("Loaded LinTS policy has the wrong data provenance.")
    if resources.policy_state_path != policy_state:
        raise RuntimeError("Resolved selector mode uses a different policy lineage.")
    if not resources.selector_mode.uses_turn_lints and (
        resources.experience_log_path != experience_log
    ):
        raise RuntimeError("Resolved selector mode uses a different experience lineage.")

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

    student_connection = sqlite3.connect(
        f"file:{student_database.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        student_tables = {
            row[0]
            for row in student_connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "resolved_events" not in student_tables:
            raise RuntimeError("Student database is missing the idempotency ledger.")
        attempt_columns = {
            row[1]
            for row in student_connection.execute("PRAGMA table_info(attempts)")
        }
        if "resolver_event_id" not in attempt_columns:
            raise RuntimeError("Student database is missing resolver event identity.")
    finally:
        student_connection.close()

    print("LOCAL RUNTIME ARTIFACTS: PASS")
    print(f"data_mode: {DATA_MODE}")
    print(f"canonical_skill_count: {len(predictor.params)}")
    print(f"policy_updates_retained: {resources.policy.total_updates}")
    print(f"student_records_retained: {_readonly_count(student_database, 'students')}")
    print(f"attempt_records_retained: {_readonly_count(student_database, 'attempts')}")
    print(f"selector_mode: {resources.selector_mode.value}")
    print(f"policy_lineage: {resources.policy_lineage}")
    print(f"active_selector_model: {resources.active_model_path}")
    print("mrb1: loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
