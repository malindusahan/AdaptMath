"""Create or validate the separate MD7-R1 local-runtime LinTS lineage."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parents[1]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
DATA_MODE = os.getenv("ADAPTIVE_DATA_MODE", "real").strip()
if DATA_MODE not in {"real", "synthetic"}:
    raise ValueError("ADAPTIVE_DATA_MODE must be exactly 'real' or 'synthetic'.")
LINEAGE_ROOT = Path(
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
POLICY_PATH = LINEAGE_ROOT / "policy_state.json"
EXPERIENCE_PATH = LINEAGE_ROOT / "attempts.jsonl"
LINEAGE_MANIFEST_PATH = LINEAGE_ROOT / "lineage_manifest.json"


def _imports() -> tuple[object, object, object, tuple[object, object]]:
    root_text = str(MOVE_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.self_improvement.lints_policy import (
        ARMS,
        TrueDisjointLinTS,
    )
    from src.self_improvement.state_io import load_policy_state, save_policy_state
    from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES

    return ARMS, TrueDisjointLinTS, load_policy_state, (
        save_policy_state,
        TURN_FEATURE_NAMES,
    )


def prepare_lineage() -> dict[str, object]:
    arms, policy_type, load_policy_state, helpers = _imports()
    save_policy_state, feature_names = helpers
    LINEAGE_ROOT.mkdir(parents=True, exist_ok=True)
    policy = policy_type(
        context_dim=len(feature_names),
        seed=42,
        data_mode=DATA_MODE,
    )

    created_policy = False
    if POLICY_PATH.exists():
        load_policy_state(policy, POLICY_PATH, expected_data_mode=DATA_MODE)
    else:
        if EXPERIENCE_PATH.exists() and EXPERIENCE_PATH.stat().st_size:
            raise RuntimeError(
                "MD7 experience log exists without its policy state; refusing repair."
            )
        save_policy_state(policy, POLICY_PATH)
        created_policy = True

    created_log = False
    if not EXPERIENCE_PATH.exists():
        EXPERIENCE_PATH.touch(exist_ok=False)
        created_log = True

    payload = {
        "lineage_version": "md7r1_lints_v1",
        "selector": "MD7-R1 epoch 3",
        "policy_state": str(POLICY_PATH.resolve()),
        "experience_log": str(EXPERIENCE_PATH.resolve()),
        "context_dim": policy.context_dim,
        "ridge_lambda": policy.ridge_lambda,
        "exploration_scale": policy.exploration_scale,
        "arms": list(arms),
        "data_mode": policy.data_mode,
        "total_updates": policy.total_updates,
        "created_policy": created_policy,
        "created_log": created_log,
    }
    LINEAGE_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    print(json.dumps(prepare_lineage(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
