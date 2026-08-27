from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


_TEST_RUNTIME: tempfile.TemporaryDirectory[str] | None = None
if "CHECKPOINT_DB_PATH" not in os.environ:
    _TEST_RUNTIME = tempfile.TemporaryDirectory(
        prefix="adaptmath_pytest_runtime_"
    )
    _runtime_root = Path(_TEST_RUNTIME.name)
    os.environ["CHECKPOINT_DB_PATH"] = str(
        _runtime_root / "checkpoints.sqlite3"
    )
    os.environ.setdefault("ADAPTIVE_DATA_MODE", "synthetic")
    os.environ.setdefault(
        "ADAPTIVE_POLICY_STATE_PATH",
        str(_runtime_root / "policy_state.json"),
    )
    os.environ.setdefault(
        "ADAPTIVE_EXPERIENCE_LOG_PATH",
        str(_runtime_root / "attempts.jsonl"),
    )


def pytest_sessionfinish(session, exitstatus) -> None:
    del session, exitstatus
    if _TEST_RUNTIME is None:
        return
    if "app.core.persistence" in sys.modules:
        from app.core.persistence import close_persistence

        close_persistence()
    _TEST_RUNTIME.cleanup()
