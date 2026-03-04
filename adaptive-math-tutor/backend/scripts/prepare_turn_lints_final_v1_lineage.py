"""Create the frozen v1 S-context warm-start lineage without activating LIVE."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    SELECTOR_NAME,
    SELECTOR_SHA256,
)
from src.self_improvement.turn_lints_context import context_schema_id  # noqa: E402
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DirectTurnLinTS,
    save_turn_lints_state,
)
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
    TurnActionLedger,
    TurnLinTSMode,
)


LINEAGE_ROOT = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "turn_lints_randomized_warmstart_final_v1"
)
CONTEXT_PRESET = "S"
ORDERED_CONTEXT_FEATURES = (
    "selector_p_generic",
    "selector_p_probing",
    "selector_p_focus",
    "selector_p_telling",
)


def run() -> dict[str, object]:
    state_path = LINEAGE_ROOT / "policy_state.json"
    policy = DirectTurnLinTS(
        context_schema_id=context_schema_id(CONTEXT_PRESET),
        enabled_context_blocks=(CONTEXT_PRESET,),
        feature_names=ORDERED_CONTEXT_FEATURES,
        selector_version=SELECTOR_NAME,
        selector_sha256=SELECTOR_SHA256,
        reward_mode="headroom_normalized",
        anchor_mode="none",
        anchor_gamma=0.0,
    )
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        policy.load_state_dict(existing)
        if policy.total_updates != 0:
            raise RuntimeError("Final warm-start preparation refuses a learned posterior.")
    else:
        save_turn_lints_state(policy, state_path)

    DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=(CONTEXT_PRESET,),
        state_path=state_path,
        ledger=TurnActionLedger(LINEAGE_ROOT),
    )
    if (LINEAGE_ROOT / "turn_events.jsonl").exists():
        raise RuntimeError("Preparation refuses a lineage containing treatments.")
    result = {
        "lineage_root": str(LINEAGE_ROOT),
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "posterior_updates": policy.total_updates,
        "event_count": 0,
        "mode_prepared_not_activated": TurnLinTSMode.RANDOMIZED_WARMSTART.value,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
