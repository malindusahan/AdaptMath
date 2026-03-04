"""Deterministic, no-LLM smoke check for the final S+K+L LIVE runtime."""

from __future__ import annotations

import json
import copy
import sys
import urllib.request
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parents[1]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
LAUNCHER_STATE = WORKSPACE_ROOT / "_local_runtime_state" / "adaptmath-local.json"
EXPECTED_ROOT = (BACKEND_ROOT / "runtime" / "turn_lints_live_skl_final_v1").resolve()
EXPECTED_STATE = EXPECTED_ROOT / "policy_state.json"
EXPECTED_MANIFEST = EXPECTED_ROOT / "lineage_manifest.json"

if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    ORDERED_CONTEXT_FEATURES,
    REWARD_MODE,
    SELECTOR_SHA256,
)
from src.self_improvement.turn_lints_context import context_schema_id  # noqa: E402
from src.self_improvement.turn_lints_live_lineage import (  # noqa: E402
    SYNTHETIC_EFFECTIVE_SAMPLE_SIZE,
    SYNTHETIC_OBSERVATION_COUNT,
    validate_live_lineage,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DirectTurnLinTS,
    load_turn_lints_state,
)


def _http_200(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def _policy() -> DirectTurnLinTS:
    blocks = ("S", "K", "L")
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(blocks),
        enabled_context_blocks=blocks,
        feature_names=ORDERED_CONTEXT_FEATURES,
        selector_version="MD7-R2-TELL-C1 Epoch 2",
        selector_sha256=SELECTOR_SHA256,
        reward_mode=REWARD_MODE,
        anchor_mode="none",
        anchor_gamma=0.0,
        ridge_lambda=1.0,
        exploration_scale=0.20,
        seed=42,
        data_mode="real",
    )


def run() -> dict[str, object]:
    launcher = json.loads(LAUNCHER_STATE.read_text(encoding="utf-8-sig"))
    expected_launcher = {
        "selectorMode": "ordinary-md7-r2-tell-c1-v1",
        "policyLineage": "direct_disjoint_turn_lints_v1",
        "dataMode": "real",
        "turnLinTSMode": "LIVE",
        "turnLinTSContextBlocks": CONTEXT_PRESET,
        "turnLinTSDiagnosticContextBlocks": "S+K+L+H+Q",
        "turnLinTSRewardMode": REWARD_MODE,
        "turnLinTSAnchorMode": "none",
        "turnLinTSAnchorGamma": 0,
    }
    for key, expected in expected_launcher.items():
        if launcher.get(key) != expected:
            raise RuntimeError(
                f"Final LIVE runtime drift for {key}: "
                f"{launcher.get(key)!r} != {expected!r}."
            )

    state_path = Path(launcher["turnLinTSStatePath"]).resolve()
    event_root = Path(launcher["turnLinTSEventRoot"]).resolve()
    manifest_path = Path(launcher["turnLinTSLiveManifestPath"]).resolve()
    if (
        state_path != EXPECTED_STATE
        or event_root != EXPECTED_ROOT
        or manifest_path != EXPECTED_MANIFEST
    ):
        raise RuntimeError("Final LIVE runtime uses the wrong isolated lineage paths.")

    policy = _policy()
    load_turn_lints_state(policy, state_path)
    counters = validate_live_lineage(policy, manifest_path, state_path=state_path)
    if policy.context_dim != CONTEXT_DIMENSION:
        raise RuntimeError("Final LIVE posterior dimension mismatch.")
    if policy.feature_names != ORDERED_CONTEXT_FEATURES:
        raise RuntimeError("Final LIVE ordered feature schema mismatch.")
    if policy.total_updates < SYNTHETIC_OBSERVATION_COUNT:
        raise RuntimeError("Final LIVE posterior lost its synthetic initialization.")

    # Exercise sampling and the immediate unit-weight update path on an
    # in-memory clone only. No event or policy state is written by this smoke.
    smoke_context = [0.25, 0.25, 0.25, 0.25, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    sampled = policy.select_arm(smoke_context)
    sampled_arm = str(sampled["selected_arm"])
    clone = copy.deepcopy(policy)
    before_updates = clone.total_updates
    applied = clone.update_once(
        update_id="deterministic-runtime-smoke:not-a-real-observation",
        arm=sampled_arm,
        context=smoke_context,
        reward=0.25,
        weight=1.0,
    )
    if not applied or clone.total_updates != before_updates + 1:
        raise RuntimeError("Final LIVE cloned unit-weight update smoke failed.")

    ports = launcher["ports"]
    tutor_health = _http_200(f"http://127.0.0.1:{int(ports['tutor'])}/health")
    tutor_ready = _http_200(f"http://127.0.0.1:{int(ports['tutor'])}/ready")
    if not tutor_health or not tutor_ready:
        raise RuntimeError("Final LIVE Tutor health/readiness smoke failed.")

    result = {
        "status": "PASS",
        "service_health": "UP",
        "service_ready": "READY",
        "mode": "LIVE",
        "algorithm": "direct disjoint Turn-LinTS",
        "policy_context": CONTEXT_PRESET,
        "dimension": policy.context_dim,
        "reward_mode": policy.reward_mode,
        "selector_sha256": policy.selector_sha256,
        "anchor_mode": policy.anchor_mode,
        "tau_gate": None,
        "synthetic_initialization_records": SYNTHETIC_OBSERVATION_COUNT,
        "synthetic_effective_weight": SYNTHETIC_EFFECTIVE_SAMPLE_SIZE,
        "real_live_updates": counters["real_live_updates"],
        "all_arms_available": True,
        "deterministic_sampled_arm": sampled_arm,
        "cloned_unit_weight_update": "PASS (not persisted; not a real event)",
        "state_path": str(state_path),
        "manifest_path": str(manifest_path),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
