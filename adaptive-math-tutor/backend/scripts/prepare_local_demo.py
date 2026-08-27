from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parents[1]
DEMO_ROOT = BACKEND_ROOT / "runtime" / "adaptive_demo"
STUDENT_SOURCE = WORKSPACE_ROOT / "student-modeling"
MOVE_SOURCE = WORKSPACE_ROOT / "pedagogical-move-selection"


def _require_source(path: Path, required: tuple[str, ...], label: str) -> Path:
    resolved = path.resolve()
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{label} is incomplete at {resolved}; missing: {', '.join(missing)}"
        )
    return resolved


def _copy_student_runtime(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Demo student-model runtime already exists: {destination}"
        )
    destination.mkdir(parents=True)
    for directory in ("bkt", "core", "db", "models"):
        shutil.copytree(
            source / directory,
            destination / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(source / "config.py", destination / "config.py")
    (destination / "data").mkdir()


def _initialize_student_database(student_root: Path) -> Path:
    root_text = str(student_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from db.database import DEFAULT_DB_PATH, initialise_database

    initialise_database()
    return DEFAULT_DB_PATH.resolve()


def _initialize_checkpoint_database(path: Path) -> None:
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(path)
    try:
        SqliteSaver(connection).setup()
    finally:
        connection.close()


def _initialize_synthetic_policy(move_root: Path, path: Path) -> None:
    root_text = str(move_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.self_improvement.lints_policy import TrueDisjointLinTS
    from src.self_improvement.state_io import save_policy_state
    from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES

    policy = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=42,
        data_mode="synthetic",
    )
    save_policy_state(policy, path)


def main() -> int:
    student_source = _require_source(
        STUDENT_SOURCE,
        ("bkt", "core", "db", "models", "config.py"),
        "student-modeling checkout",
    )
    move_source = _require_source(
        MOVE_SOURCE,
        ("src/self_improvement/lints_policy.py", "src/self_improvement/state_io.py"),
        "pedagogical-move-selection checkout",
    )

    if DEMO_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing demo state: {DEMO_ROOT}"
        )
    DEMO_ROOT.mkdir(parents=True)

    student_root = DEMO_ROOT / "student_model"
    checkpoint_path = DEMO_ROOT / "checkpoints.sqlite3"
    policy_path = DEMO_ROOT / "policy_state.json"
    experience_path = DEMO_ROOT / "attempts.jsonl"

    _copy_student_runtime(student_source, student_root)
    student_db_path = _initialize_student_database(student_root)
    _initialize_checkpoint_database(checkpoint_path)
    _initialize_synthetic_policy(move_source, policy_path)
    experience_path.touch(exist_ok=False)

    manifest = {
        "mode": "synthetic-local-demo",
        "student_model_source": str(student_source),
        "student_model_state": str(student_db_path),
        "checkpoint_db": str(checkpoint_path.resolve()),
        "policy_state": str(policy_path.resolve()),
        "experience_log": str(experience_path.resolve()),
    }
    (DEMO_ROOT / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prepared isolated synthetic local-demo storage:")
    for name in (
        "checkpoint_db",
        "policy_state",
        "experience_log",
        "student_model_state",
    ):
        print(f"  {name}: {manifest[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
