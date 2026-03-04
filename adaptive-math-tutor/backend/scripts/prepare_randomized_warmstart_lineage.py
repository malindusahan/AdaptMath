"""Create the inactive, zero-update MD7-proportional warm-start lineage."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.turn_lints_context import (  # noqa: E402
    BLOCK_ORDER,
    context_schema_id,
    feature_names_for_blocks,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DirectTurnLinTS,
    save_turn_lints_state,
)
from src.self_improvement.turn_lints_runtime import (  # noqa: E402
    DirectTurnController,
    TurnActionLedger,
    TurnLinTSMode,
)


SELECTOR_SHA256 = (
    "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
)
LINEAGE_ROOT = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "turn_lints_randomized_warmstart_v1"
)


def run() -> dict[str, object]:
    state_path = LINEAGE_ROOT / "policy_state.json"
    policy = DirectTurnLinTS(
        context_schema_id=context_schema_id(BLOCK_ORDER),
        enabled_context_blocks=BLOCK_ORDER,
        feature_names=feature_names_for_blocks(BLOCK_ORDER),
        selector_version="MD7-R2-TELL-C1 epoch 2",
        selector_sha256=SELECTOR_SHA256,
        reward_mode="headroom_normalized",
        anchor_mode="none",
        anchor_gamma=0.0,
    )
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        policy.load_state_dict(existing)
        if policy.total_updates != 0:
            raise RuntimeError("Warm-start preparation refuses a non-fresh posterior.")
    else:
        save_turn_lints_state(policy, state_path)

    # Construction writes and validates the immutable lineage manifest. It does
    # not start an attempt, sample a treatment, or update the posterior.
    DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=BLOCK_ORDER,
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
