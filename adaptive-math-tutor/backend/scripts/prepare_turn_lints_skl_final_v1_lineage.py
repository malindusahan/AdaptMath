"""Create the frozen S+K+L warm-start lineage without treatments or LIVE."""

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
    ANCHOR_GAMMA,
    ANCHOR_MODE,
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    ORDERED_CONTEXT_FEATURES,
    REWARD_MODE,
    SELECTOR_NAME,
    SELECTOR_SHA256,
)
from src.self_improvement.turn_lints_context import (  # noqa: E402
    context_schema_id,
    parse_context_blocks,
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


LINEAGE_ROOT = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "turn_lints_randomized_warmstart_skl_final_v1"
)


def run() -> dict[str, object]:
    blocks = parse_context_blocks(CONTEXT_PRESET)
    if len(ORDERED_CONTEXT_FEATURES) != CONTEXT_DIMENSION:
        raise RuntimeError("Frozen S+K+L feature contract is inconsistent.")
    state_path = LINEAGE_ROOT / "policy_state.json"
    policy = DirectTurnLinTS(
        context_schema_id=context_schema_id(blocks),
        enabled_context_blocks=blocks,
        feature_names=ORDERED_CONTEXT_FEATURES,
        selector_version=SELECTOR_NAME,
        selector_sha256=SELECTOR_SHA256,
        reward_mode=REWARD_MODE,
        anchor_mode=ANCHOR_MODE,
        anchor_gamma=ANCHOR_GAMMA,
    )
    if state_path.exists():
        policy.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        if policy.total_updates != 0:
            raise RuntimeError("S+K+L preparation refuses a learned posterior.")
    else:
        save_turn_lints_state(policy, state_path)

    DirectTurnController(
        policy=policy,
        mode=TurnLinTSMode.RANDOMIZED_WARMSTART,
        enabled_context_blocks=blocks,
        state_path=state_path,
        ledger=TurnActionLedger(LINEAGE_ROOT),
    )
    if (LINEAGE_ROOT / "turn_events.jsonl").exists():
        raise RuntimeError("S+K+L preparation refuses a lineage containing treatments.")
    result = {
        "lineage_root": str(LINEAGE_ROOT),
        "policy_context": CONTEXT_PRESET,
        "policy_dimension": CONTEXT_DIMENSION,
        "diagnostic_context": "S+K+L+H+Q",
        "reward_mode": REWARD_MODE,
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "posterior_updates": policy.total_updates,
        "event_count": 0,
        "live_enabled": False,
        "mode_prepared_not_activated": TurnLinTSMode.RANDOMIZED_WARMSTART.value,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
