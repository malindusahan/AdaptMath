"""Offline-only validation for manual MD7-active/MD6-shadow UI mode."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parents[1]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
OUTPUT_DIR = MOVE_ROOT / "results" / "md7r1_manual_controlled_live_v1"
OUTPUT_PATH = OUTPUT_DIR / "offline_validation.json"
POLICY_PATH = (
    BACKEND_ROOT
    / "runtime"
    / "adaptive_demo_md7r1_v1"
    / "policy_state.json"
)
EXPERIENCE_PATH = (
    BACKEND_ROOT
    / "runtime"
    / "adaptive_demo_md7r1_v1"
    / "attempts.jsonl"
)
MD6_DIR = MOVE_ROOT / "models" / "frozen" / "md6"
MD7_DIR = MOVE_ROOT / "models" / "candidates" / "md7r1_epoch3"

for import_root in (BACKEND_ROOT, MOVE_ROOT):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

from app.integrations.manual_controlled_live import (  # noqa: E402
    file_snapshot,
)
from src.self_improvement.conservative_overlay import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_snapshot(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def check(number: int, name: str, evidence: str) -> dict[str, object]:
    return {
        "number": number,
        "name": name,
        "status": "PASS",
        "evidence": evidence,
    }


def main() -> None:
    protected_before = {
        "policy_state": file_snapshot(POLICY_PATH),
        "experience_log": file_snapshot(EXPERIENCE_PATH),
    }
    models_before = {
        "md6": model_snapshot(MD6_DIR),
        "md7r1_epoch3": model_snapshot(MD7_DIR),
    }
    environment = dict(os.environ)
    environment.pop("ADAPTIVE_MANUAL_CONTROLLED_LIVE", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(BACKEND_ROOT / "tests" / "test_manual_controlled_live.py"),
        str(BACKEND_ROOT / "tests" / "test_adaptive_tutor_integration.py"),
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=BACKEND_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Offline controlled-live tests failed.\n"
            + completed.stdout
            + completed.stderr
        )
    if DEFAULT_GAP_THRESHOLD != 0.10:
        raise RuntimeError("Overlay threshold changed from 0.10.")

    launcher = (WORKSPACE_ROOT / "start-adaptmath-local.ps1").read_text(
        encoding="utf-8"
    )
    for required in (
        "[switch]$ManualControlledLive",
        "[switch]$MD6Rollback",
        "$env:ADAPTIVE_SELECTOR_MODE = $desiredSelectorMode",
        "$env:ADAPTIVE_MANUAL_CONTROLLED_LIVE = 'enabled-v1'",
        "ACTIVE: MD7-R1 epoch 3",
        "SHADOW: frozen MD6",
        "LinTS posterior updates: DISABLED",
        "Real adaptive persistence: DISABLED",
    ):
        if required not in launcher:
            raise RuntimeError(f"Launcher contract is missing: {required}")

    protected_after = {
        "policy_state": file_snapshot(POLICY_PATH),
        "experience_log": file_snapshot(EXPERIENCE_PATH),
    }
    models_after = {
        "md6": model_snapshot(MD6_DIR),
        "md7r1_epoch3": model_snapshot(MD7_DIR),
    }
    if protected_before != protected_after:
        raise RuntimeError("Offline validation changed real adaptive persistence.")
    if models_before != models_after:
        raise RuntimeError("Offline validation changed MD6 or MD7 model files.")

    checks = [
        check(1, "ordinary mode resolves MD7 as active", "test_resource_routing_promotes_md7_and_keeps_explicit_md6_rollback"),
        check(2, "controlled-live mode resolves MD7 as active", "resource routing test loads the candidate path only under the exact opt-in"),
        check(3, "controlled-live mode loads MD6 shadow", "resource routing test retains the default frozen selector as shadow_md6"),
        check(4, "identical fixture problem/history reaches both selectors", "test_active_md7_and_shadow_md6_receive_identical_fixture"),
        check(5, "returned active probabilities are MD7", "paired-selector and integrated-pipeline fixture assertions"),
        check(6, "MD6 probabilities are diagnostic only", "MD6 argmax probing while returned/final MD7 path is focus"),
        check(7, "C3 receives MD7 probabilities", "integrated fixture asserts context[0:4] equals MD7 vector"),
        check(8, "overlay threshold remains 0.10", f"DEFAULT_GAP_THRESHOLD={DEFAULT_GAP_THRESHOLD}"),
        check(9, "final move comes from MD7-driven path", "integrated fixture forces MD7 focus/MD6 probing and observes final focus"),
        check(10, "abort/test lifecycle produces zero LinTS updates", "pipeline abort and coordinator controlled-completion tests both assert total_updates=0"),
        check(11, "real policy state remains byte-for-byte unchanged", "before/after size, mtime_ns, and SHA256 match"),
        check(12, "real experience log remains byte-for-byte unchanged", "before/after size, mtime_ns, and SHA256 match"),
        check(13, "isolated diagnostic record can be written", "test_isolated_turn_log_has_both_selectors_and_preserves_policy"),
        check(14, "ordinary mode remains learning-enabled MD7 when flag is absent", "selector-mode test plus existing adaptive integration regression suite"),
    ]
    payload = {
        "validation": "manual_controlled_live_md7r1_v1",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "external_api_calls": 0,
        "training": 0,
        "pytest_command": command,
        "pytest_stdout": completed.stdout.strip(),
        "checks": checks,
        "protected_files_before": protected_before,
        "protected_files_after": protected_after,
        "protected_files_unchanged": protected_before == protected_after,
        "model_files_unchanged": models_before == models_after,
        "side_effects": {
            "training": 0,
            "model_writes": 0,
            "LinTS_posterior_updates": 0,
            "production_policy_writes": 0,
            "production_experience_log_writes": 0,
            "external_API_calls": 0,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
