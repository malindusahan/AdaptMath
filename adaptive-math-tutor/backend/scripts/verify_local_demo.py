from __future__ import annotations

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
OLD_CHECKPOINT = (
    BACKEND_ROOT / "runtime" / "adaptmath_checkpoints.sqlite3"
).resolve()


def _configured_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured.")
    path = Path(value)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path.resolve()


def _require_demo_path(name: str) -> Path:
    path = _configured_path(name)
    if path != DEMO_ROOT and DEMO_ROOT not in path.parents:
        raise RuntimeError(f"{name} must remain under {DEMO_ROOT}; got {path}")
    return path


def main() -> int:
    if os.getenv("ADAPTIVE_DATA_MODE") != "synthetic":
        raise RuntimeError("ADAPTIVE_DATA_MODE must be synthetic for local demo.")

    checkpoint = _require_demo_path("CHECKPOINT_DB_PATH")
    policy_state = _require_demo_path("ADAPTIVE_POLICY_STATE_PATH")
    experience_log = _require_demo_path("ADAPTIVE_EXPERIENCE_LOG_PATH")
    student_root = _require_demo_path("STUDENT_MODEL_REPO")
    if checkpoint == OLD_CHECKPOINT:
        raise RuntimeError("Local demo must not use the legacy checkpoint DB.")

    from app.core.persistence import close_persistence, get_checkpoint_db_path

    resolved_checkpoint = get_checkpoint_db_path().resolve()
    if resolved_checkpoint != checkpoint:
        raise RuntimeError(
            "Tutor persistence resolved a different checkpoint path: "
            f"{resolved_checkpoint}"
        )
    close_persistence()

    required = (
        checkpoint,
        policy_state,
        experience_log,
        student_root / "data" / "meta_agent.db",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared demo paths: " + ", ".join(missing))

    resources = _ProductionAdaptiveResources()
    predictor = resources._ensure_predictor()
    resources._ensure_shared()
    if resources.policy.data_mode != "synthetic":
        raise RuntimeError("Loaded policy is not synthetic.")
    if resources.policy.total_updates != 0:
        raise RuntimeError("Fresh demo policy unexpectedly contains learned updates.")

    student_db = student_root / "data" / "meta_agent.db"
    connection = sqlite3.connect(f"file:{student_db.as_posix()}?mode=ro", uri=True)
    try:
        student_count = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        attempt_count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    finally:
        connection.close()
    if student_count or attempt_count:
        raise RuntimeError("Prepared demo student model must contain no learner records.")

    samples = list(predictor.params)[:4]
    print("LOCAL DEMO CONFIGURATION VERIFIED")
    print(f"checkpoint_db: {checkpoint}")
    print("checkpoint_config_resolution: pass")
    print(f"policy_state: {policy_state}")
    print(f"experience_log: {experience_log}")
    print(f"student_model_state: {student_db}")
    print("policy_mode: synthetic")
    print("policy_updates: 0")
    print("student_records: 0")
    print("attempt_records: 0")
    print("valid_skill_sample: " + " | ".join(samples))
    print("md6: loaded")
    print("mrb1: loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
