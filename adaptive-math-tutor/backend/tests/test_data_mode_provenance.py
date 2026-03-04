import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.integrations.adaptive_component_coordinator import (
    _ProductionAdaptiveResources,
)
from app.integrations.adaptive_selector_mode import (
    AdaptiveSelectorMode,
    resolve_selector_mode,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def test_normal_interactive_launcher_is_explicitly_real():
    source = (WORKSPACE_ROOT / "start-adaptmath-local.ps1").read_text(
        encoding="utf-8"
    )
    assert "$env:ADAPTIVE_DATA_MODE = 'real'" in source
    assert "$env:ADAPTIVE_DATA_MODE = 'synthetic'" not in source
    assert "adaptive_demo_md7r1_real_v1" in source
    assert "adaptive_demo_md6_real_v1" in source
    assert "adaptive_live_real_v1" in source
    assert "$env:ADAPTIVE_LOCAL_RUNTIME_ROOT = $demoRoot" in source
    assert "dataMode = 'real'" in source


def test_runtime_configuration_accepts_explicit_real_mode(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_DATA_MODE", "real")
    resources = _ProductionAdaptiveResources()
    assert resources.data_mode == "real"


def test_no_flag_selector_mode_is_promoted_md7_r2_turn_lints_base():
    assert resolve_selector_mode({}) is AdaptiveSelectorMode.NORMAL_MD7
    assert AdaptiveSelectorMode.NORMAL_MD7.value == "ordinary-md7-r2-tell-c1-v1"


def test_runtime_configuration_fails_closed_on_invalid_mode(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_DATA_MODE", "human-ish")
    with pytest.raises(ValueError, match="exactly 'synthetic' or 'real'"):
        _ProductionAdaptiveResources()


def test_lineage_preparer_uses_mode_specific_nonhistorical_directory():
    source = (
        WORKSPACE_ROOT
        / "adaptive-math-tutor"
        / "backend"
        / "scripts"
        / "prepare_md7r1_policy_lineage.py"
    ).read_text(encoding="utf-8")
    assert 'DATA_MODE = os.getenv("ADAPTIVE_DATA_MODE", "real").strip()' in source
    assert 'data_mode=DATA_MODE' in source
    assert 'expected_data_mode=DATA_MODE' in source
    assert '"adaptive_demo_md7r1_real_v1"' in source


def test_student_runtime_preparer_separates_real_and_synthetic_storage():
    source = (
        WORKSPACE_ROOT
        / "adaptive-math-tutor"
        / "backend"
        / "scripts"
        / "prepare_local_demo.py"
    ).read_text(encoding="utf-8")
    assert 'DATA_MODE = os.getenv("ADAPTIVE_DATA_MODE", "synthetic").strip()' in source
    assert '"adaptive_live_real_v1"' in source
    assert '"adaptive_demo"' in source
    assert 'data_mode=DATA_MODE' in source
    assert 'expected_data_mode=DATA_MODE' in source
    assert '"mode": DATA_MODE' in source


def test_real_runtime_prepare_is_idempotent_migrated_and_preserves_data(tmp_path):
    runtime_root = tmp_path / "real-runtime"
    script = (
        WORKSPACE_ROOT
        / "adaptive-math-tutor"
        / "backend"
        / "scripts"
        / "prepare_local_demo.py"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "ADAPTIVE_DATA_MODE": "real",
            "ADAPTIVE_LOCAL_RUNTIME_ROOT": str(runtime_root),
        }
    )

    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=script.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
    )
    database = runtime_root / "student_model" / "data" / "meta_agent.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO students (student_id) VALUES ('preserved-student')"
        )

    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=script.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(
        (runtime_root / "demo_manifest.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (runtime_root / "policy_state.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        attempt_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(attempts)")
        }
        retained = connection.execute(
            "SELECT COUNT(*) FROM students WHERE student_id = 'preserved-student'"
        ).fetchone()[0]

    assert manifest["mode"] == "real"
    assert manifest["runtime_kind"] == "interactive-human-tutoring"
    assert policy["policy_state"]["data_mode"] == "real"
    assert "resolved_events" in tables
    assert "resolver_event_id" in attempt_columns
    assert retained == 1
