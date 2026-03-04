"""Safe versioned persistence for canonical true-LinTS policy state.

This module provides reproducible policy-state persistence only. Synthetic
learned posterior state must never seed real tutoring. The authoritative state
payload and all posterior/configuration validation remain in
``TrueDisjointLinTS.state_dict`` and ``TrueDisjointLinTS.load_state_dict``.

State persistence does not establish educational efficacy or an optimal
context, reward, threshold, hyperparameter configuration, or credit-assignment
scheme. Policy snapshots remain separate from attempt trajectory logs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

from .lints_policy import (
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    LINTS_STATE_SCHEMA_VERSION,
    TrueDisjointLinTS,
)
from .turn_context_builder import TURN_FEATURE_NAMES


POLICY_STATE_SCHEMA_VERSION: Final[str] = "turn_lints_state_v3"
POLICY_CLASS_NAME: Final[str] = "TrueDisjointLinTS"
DATA_MODES: Final[tuple[str, ...]] = ("synthetic", "real")

_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "policy_class", "policy_state"}
)


def _validate_policy(policy: object) -> TrueDisjointLinTS:
    if not isinstance(policy, TrueDisjointLinTS):
        raise TypeError("policy must be a TrueDisjointLinTS instance.")
    if policy.context_dim != len(TURN_FEATURE_NAMES):
        raise ValueError(
            "Canonical persisted policy must use the 9-D C3 turn context."
        )
    if policy.ridge_lambda != DEFAULT_RIDGE_LAMBDA:
        raise ValueError(
            "Canonical persisted policy must use ridge_lambda=1.0."
        )
    if policy.exploration_scale != DEFAULT_EXPLORATION_SCALE:
        raise ValueError(
            "Canonical persisted policy must use exploration_scale=0.20."
        )
    return policy


def _validate_data_mode(value: object, name: str) -> str:
    if not isinstance(value, str) or value not in DATA_MODES:
        raise ValueError(f"{name} must be exactly 'synthetic' or 'real'.")
    return value


def _reject_nonstandard_constant(value: str) -> object:
    raise ValueError(f"State JSON contains forbidden non-finite value {value!r}.")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"State JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _serialize_envelope(policy: TrueDisjointLinTS) -> str:
    """Build and serialize the complete state before any filesystem write."""

    policy_state = policy.state_dict()
    if not isinstance(policy_state, Mapping):
        raise TypeError("policy.state_dict() must return a mapping.")
    if policy_state.get("schema_version") != LINTS_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Policy state does not use the canonical signed-reward v3 schema."
        )

    state_mode = _validate_data_mode(
        policy_state.get("data_mode"),
        "policy_state.data_mode",
    )
    if state_mode != policy.data_mode:
        raise ValueError(
            "policy.state_dict() data_mode does not match policy.data_mode."
        )

    envelope = {
        "schema_version": POLICY_STATE_SCHEMA_VERSION,
        "policy_class": POLICY_CLASS_NAME,
        "policy_state": policy_state,
    }
    try:
        serialized = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    except TypeError as exc:
        raise TypeError(
            "Canonical policy state contains a non-JSON-serializable object."
        ) from exc
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "Canonical policy state contains an invalid JSON numeric value."
        ) from exc
    return serialized + "\n"


def save_policy_state(
    policy: TrueDisjointLinTS,
    path: str | Path,
) -> Path:
    """Atomically replace one JSON state file with a canonical snapshot.

    Serialization completes before a temporary file is created. The temporary
    file is written, flushed, and ``fsync``-ed in the destination directory,
    then ``os.replace`` atomically installs it. On any failure before replace,
    a pre-existing valid destination remains unchanged and the temporary file
    is removed where practical.
    """

    validated_policy = _validate_policy(policy)
    serialized = _serialize_envelope(validated_policy)
    destination = Path(path)
    from .postgres_repository import (
        policy_key_for_path,
        postgres_enabled,
        save_policy_snapshot,
    )
    if postgres_enabled():
        envelope = json.loads(serialized)
        save_policy_snapshot(
            policy_key=policy_key_for_path(destination),
            payload=envelope,
            policy_class=POLICY_CLASS_NAME,
        )
        return destination
    if destination.exists() and not destination.is_file():
        raise ValueError(f"State destination is not a regular file: {destination}.")
    if destination.exists():
        existing_envelope = _read_state_envelope(destination)
        existing_policy_state = existing_envelope["policy_state"]
        if not isinstance(existing_policy_state, Mapping) or (
            existing_policy_state.get("schema_version")
            != LINTS_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Refusing to overwrite policy state from an incompatible reward "
                "lineage. Use a new v3 state path."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def _read_state_envelope(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Policy state file does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"Policy state path is not a regular file: {path}.")

    try:
        serialized = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ValueError("Policy state file is not valid UTF-8.") from exc

    try:
        envelope = json.loads(
            serialized,
            parse_constant=_reject_nonstandard_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Policy state file contains malformed JSON.") from exc

    if not isinstance(envelope, dict):
        raise ValueError("Policy state file must contain one top-level object.")
    actual_fields = set(envelope)
    if actual_fields != _ENVELOPE_FIELDS:
        missing = sorted(_ENVELOPE_FIELDS - actual_fields)
        extra = sorted(actual_fields - _ENVELOPE_FIELDS)
        raise ValueError(
            "Policy state envelope fields are incompatible; "
            f"missing={missing}, extra={extra}."
        )
    if envelope["schema_version"] != POLICY_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported policy state schema version: "
            f"{envelope['schema_version']!r}."
        )
    if envelope["policy_class"] != POLICY_CLASS_NAME:
        raise ValueError(
            f"policy_class must be exactly {POLICY_CLASS_NAME!r}."
        )
    if not isinstance(envelope["policy_state"], Mapping):
        raise TypeError("policy_state must be a mapping.")
    if (
        envelope["policy_state"].get("schema_version")
        != LINTS_STATE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported internal LinTS policy state schema.")
    return envelope


def load_policy_state(
    policy: TrueDisjointLinTS,
    path: str | Path,
    expected_data_mode: str,
) -> None:
    """Validate a versioned file and atomically load its canonical payload.

    File-level schema, policy class, and hard data-mode checks complete before
    delegating to ``policy.load_state_dict``. That canonical method remains
    authoritative for configuration, posterior, counts, shapes, and RNG state
    and performs its own validate-before-mutate commit.
    """

    validated_policy = _validate_policy(policy)
    expected_mode = _validate_data_mode(
        expected_data_mode,
        "expected_data_mode",
    )
    if validated_policy.data_mode != expected_mode:
        raise ValueError(
            "expected_data_mode must match the receiving policy; "
            f"expected={expected_mode!r}, policy={validated_policy.data_mode!r}."
        )

    source = Path(path)
    from .postgres_repository import (
        load_policy_snapshot,
        policy_key_for_path,
        postgres_enabled,
    )
    envelope = (
        load_policy_snapshot(policy_key_for_path(source))
        if postgres_enabled()
        else _read_state_envelope(source)
    )
    policy_state = envelope["policy_state"]
    if not isinstance(policy_state, Mapping):
        raise TypeError("policy_state must be a mapping.")
    state_mode = _validate_data_mode(
        policy_state.get("data_mode"),
        "policy_state.data_mode",
    )
    if state_mode != expected_mode:
        raise ValueError(
            "Synthetic/real policy-state barrier triggered: "
            f"state={state_mode!r}, expected={expected_mode!r}."
        )

    validated_policy.load_state_dict(
        policy_state,
        expected_data_mode=expected_mode,
    )


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA256 hex digest of one state file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash non-file policy state: {source}.")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "DATA_MODES",
    "POLICY_CLASS_NAME",
    "POLICY_STATE_SCHEMA_VERSION",
    "load_policy_state",
    "save_policy_state",
    "sha256_file",
)
