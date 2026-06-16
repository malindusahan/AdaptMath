"""Fail-closed validation for the audited S+K+L LIVE posterior lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .turn_lints_architecture_v1 import (
    ANCHOR_MODE,
    CONTEXT_DIMENSION,
    CONTEXT_PRESET,
    ORDERED_CONTEXT_FEATURES,
    REWARD_MODE,
    SELECTOR_NAME,
    SELECTOR_SHA256,
)
from .turn_lints_policy import DIRECT_ARMS, DirectTurnLinTS


LIVE_LINEAGE_SCHEMA = "adaptmath_turn_lints_live_skl_lineage_v1"
ARCHITECTURE_MANIFEST_SHA256 = (
    "a860d385797eaaced4788463fb789197c614bef7d7934f8006705276fdb57dfa"
)
SYNTHETIC_GENERATOR_VERSION = "adaptmath_skl_synthetic_warmstart_v1"
SYNTHETIC_OBSERVATION_COUNT = 500
SYNTHETIC_OBSERVATION_WEIGHT = 0.02
SYNTHETIC_EFFECTIVE_SAMPLE_SIZE = 10.0
REAL_LIVE_OBSERVATION_WEIGHT = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_live_lineage(
    policy: DirectTurnLinTS,
    manifest_path: str | Path,
    *,
    state_path: str | Path,
) -> dict[str, object]:
    """Validate the LIVE lineage and derive source-separated update counts."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"LIVE Turn-LinTS lineage manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("LIVE Turn-LinTS lineage manifest must be an object.")
    expected = {
        "schema_version": LIVE_LINEAGE_SCHEMA,
        "mode": "LIVE",
        "architecture_manifest_sha256": ARCHITECTURE_MANIFEST_SHA256,
        "context_preset": CONTEXT_PRESET,
        "context_dimension": CONTEXT_DIMENSION,
        "ordered_context_features": list(ORDERED_CONTEXT_FEATURES),
        "reward_mode": REWARD_MODE,
        "selector_name": SELECTOR_NAME,
        "selector_sha256": SELECTOR_SHA256,
        "anchor_mode": ANCHOR_MODE,
        "synthetic_generator_version": SYNTHETIC_GENERATOR_VERSION,
        "synthetic_observation_count": SYNTHETIC_OBSERVATION_COUNT,
        "synthetic_observation_weight": SYNTHETIC_OBSERVATION_WEIGHT,
        "synthetic_effective_sample_size": SYNTHETIC_EFFECTIVE_SAMPLE_SIZE,
        "real_live_observation_weight": REAL_LIVE_OBSERVATION_WEIGHT,
        "activation_decision": "A",
        "live_enabled": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"LIVE Turn-LinTS lineage mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}."
            )
    if not str(manifest.get("live_activated_at") or "").strip():
        raise RuntimeError("LIVE Turn-LinTS lineage lacks activation timestamp.")
    synthetic_ids = policy.applied_update_ids[:SYNTHETIC_OBSERVATION_COUNT]
    expected_ids = [
        f"synthetic-init:{index:04d}"
        for index in range(1, SYNTHETIC_OBSERVATION_COUNT + 1)
    ]
    if synthetic_ids != expected_ids:
        raise RuntimeError("LIVE posterior does not contain the exact audited synthetic prefix.")
    if policy.total_updates < SYNTHETIC_OBSERVATION_COUNT:
        raise RuntimeError("LIVE posterior has fewer updates than its initialization.")
    real_updates = policy.total_updates - SYNTHETIC_OBSERVATION_COUNT
    synthetic_arm_counts = manifest.get("synthetic_arm_update_counts")
    if not isinstance(synthetic_arm_counts, Mapping):
        raise RuntimeError("LIVE manifest lacks synthetic per-arm counts.")
    real_arm_counts: dict[str, int] = {}
    for arm in DIRECT_ARMS:
        synthetic_count = int(synthetic_arm_counts.get(arm, -1))
        real_count = int(policy.arm_update_counts[arm]) - synthetic_count
        if synthetic_count < 1 or real_count < 0:
            raise RuntimeError(f"LIVE posterior per-arm lineage is invalid for {arm}.")
        real_arm_counts[arm] = real_count
    if sum(real_arm_counts.values()) != real_updates:
        raise RuntimeError("LIVE posterior aggregate/per-arm real-update counts disagree.")
    state = Path(state_path).resolve()
    if real_updates == 0 and _sha256(state) != manifest.get("initial_state_sha256"):
        raise RuntimeError("Zero-real-update LIVE state differs from the audited initial state.")
    return {
        "synthetic_initialization_updates": SYNTHETIC_OBSERVATION_COUNT,
        "synthetic_effective_sample_size": SYNTHETIC_EFFECTIVE_SAMPLE_SIZE,
        "real_live_updates": real_updates,
        "real_live_arm_updates": real_arm_counts,
    }


__all__ = ("LIVE_LINEAGE_SCHEMA", "validate_live_lineage")
