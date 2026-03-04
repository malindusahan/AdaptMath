"""Passive, append-only turn-outcome collection for future selector research.

The collector only projects values already computed by the live pipeline. It
does not call a selector, evaluator, resolver, BKT model, tutor, or scorer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import RLock
from typing import Any, Iterator


SCHEMA_VERSION = "adaptmath_turn_outcome_v1"
DATASET_DIRECTORY_NAME = "md_self_improvement_turn_data_v1"
COMPLETION_STATUSES = frozenset({"completed", "aborted"})
OBSERVATION_STATUSES = frozenset(
    {
        "observed_update",
        "resolved_no_update",
        "censored_no_response",
        "processing_failed",
    }
)


class TurnCollectionIntegrityError(RuntimeError):
    """Raised when one durable identity is presented with different content."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"))
        except TypeError:
            return _json_safe(dump())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(record: Mapping[str, object]) -> str:
    return json.dumps(
        _json_safe(record),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_action_snapshot(record: Mapping[str, object]) -> str:
    """Canonicalize the immutable action portion of a staged/final row."""

    snapshot = copy.deepcopy(dict(_json_safe(record)))
    snapshot.pop("next_learner_observation", None)
    snapshot.pop("learning_state_update", None)
    provenance = snapshot.get("provenance")
    if isinstance(provenance, dict):
        # A retry naturally observes a different wall-clock time. The first
        # durable timestamp remains authoritative; it is not action content.
        provenance.pop("timestamp", None)
    return _canonical_json(snapshot)


def _without_local_policy_paths(value: object) -> object:
    """Remove runtime-only policy paths from a copied scientific payload."""

    safe = _json_safe(value)
    if isinstance(safe, Mapping):
        return {
            str(key): _without_local_policy_paths(item)
            for key, item in safe.items()
            if str(key) != "policy_state_path"
        }
    if isinstance(safe, Sequence) and not isinstance(safe, (str, bytes)):
        return [_without_local_policy_paths(item) for item in safe]
    return safe


def _validate_utc_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an explicit UTC ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an explicit UTC ISO-8601 timestamp."
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC.")
    return value


def _nested_value(record: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = record
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Record is missing unique identity path {path!r}.")
        current = current[key]
    return current


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Use a one-byte cross-process lock around scan-plus-append."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class PassiveTurnCollector:
    """Durably stage actions and append their eventual observed/censored result."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        data_mode: str,
        version_provenance: Mapping[str, object] | None = None,
        preserve_pending_for_restart: bool = False,
    ) -> None:
        mode = str(data_mode).strip()
        if mode not in {"real", "synthetic"}:
            raise ValueError("data_mode must be exactly 'real' or 'synthetic'.")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.output_dir / "pending_actions"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir = self.output_dir / "pending_processing_failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.turn_path = self.output_dir / "turn_outcomes.jsonl"
        self.assessment_path = self.output_dir / "assessment_outcomes.jsonl"
        self.attempt_path = self.output_dir / "attempt_summaries.jsonl"
        self.manifest_path = self.output_dir / "dataset_manifest.json"
        self.lock_path = self.output_dir / ".append.lock"
        self.data_mode = mode
        self.version_provenance = dict(version_provenance or {})
        if not isinstance(preserve_pending_for_restart, bool):
            raise TypeError("preserve_pending_for_restart must be boolean.")
        self.preserve_pending_for_restart = preserve_pending_for_restart
        self._lock = RLock()
        self._write_manifest_once()
        if not self.preserve_pending_for_restart:
            self._recover_abandoned_pending()

    @staticmethod
    def pseudonymize_student_id(student_id: object) -> str:
        if student_id is None or isinstance(student_id, bool):
            raise ValueError("student_id must be available for pseudonymization.")
        value = str(student_id).strip()
        if not value:
            raise ValueError("student_id must be available for pseudonymization.")
        digest = hashlib.sha256(
            f"{SCHEMA_VERSION}:student:{value}".encode("utf-8")
        ).hexdigest()
        return f"psn_{digest[:24]}"

    def _write_manifest_once(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_directory": DATASET_DIRECTORY_NAME,
            "format": "JSONL authoritative; one final row per action_event_id",
            "observation_statuses": sorted(OBSERVATION_STATUSES),
            "artifacts": {
                "turn_outcomes": "turn_outcomes.jsonl",
                "assessment_outcomes": "assessment_outcomes.jsonl",
                "attempt_summaries": "attempt_summaries.jsonl",
                "bkt_activity": "bkt_activity.jsonl",
            },
            "privacy": {
                "student_identifier": (
                    "deterministic dataset-scoped SHA-256 pseudonym; raw id omitted"
                ),
                "retained_potentially_identifying_fields": [
                    "problem text",
                    "mathematical dialogue needed by the selector",
                    "tutor response",
                    "thread and attempt runtime identifiers",
                ],
                "excluded_fields": [
                    "names/account identifiers",
                    "authentication secrets",
                    "API keys",
                    "age/profile fields not consumed by move selection",
                ],
            },
            "semantics": {
                "delta_mastery": (
                    "confidence-weighted BKT posterior belief update after "
                    "resolved evidence; not a reward or causal effect"
                ),
                "formal_assessment": (
                    "stored separately and never joined to the last Tutor action"
                ),
                "collector_execution": "copy-only; no model or evaluator execution",
                "portable_policy_provenance": (
                    "scientific rows omit runtime policy_state_path values and retain "
                    "only lineage/repository-relative identifiers"
                ),
            },
        }
        canonical = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ) + "\n"
        with self._lock, _exclusive_lock(self.lock_path):
            if self.manifest_path.exists():
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if existing.get("schema_version") != SCHEMA_VERSION:
                    raise TurnCollectionIntegrityError(
                        "Dataset directory contains a different schema version."
                    )
                artifacts = existing.get("artifacts")
                if not isinstance(artifacts, dict):
                    raise TurnCollectionIntegrityError(
                        "Dataset manifest artifacts must be an object."
                    )
                if artifacts.get("bkt_activity") == "bkt_activity.jsonl":
                    return
                artifacts["bkt_activity"] = "bkt_activity.jsonl"
                canonical = json.dumps(
                    existing,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ) + "\n"
            temporary = self.manifest_path.with_name(
                f".{self.manifest_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(canonical, encoding="utf-8")
            os.replace(temporary, self.manifest_path)

    @staticmethod
    def _postgres_enabled() -> bool:
        try:
            from src.self_improvement.postgres_repository import postgres_enabled
        except ImportError:
            return False
        return postgres_enabled()

    @staticmethod
    def _postgres_action_id(action_event_id: str) -> str:
        return f"passive:{action_event_id}"

    def _append_unique(
        self,
        path: Path,
        record: Mapping[str, object],
        identity_path: tuple[str, ...],
    ) -> bool:
        canonical = _canonical_json(record)
        identity = _nested_value(record, identity_path)
        if identity is None or not str(identity).strip():
            raise ValueError("Append identity must be non-empty.")
        try:
            from src.self_improvement.postgres_repository import (
                append_experience,
                postgres_enabled,
            )
        except ImportError:
            postgres_enabled = lambda: False
        if postgres_enabled():
            append_experience(
                dict(record),
                event_id_override=f"passive:{path.stem}:{identity}",
            )
            return True
        with self._lock, _exclusive_lock(self.lock_path):
            if path.exists():
                for line_number, raw_line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if not raw_line.strip():
                        continue
                    try:
                        existing = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise TurnCollectionIntegrityError(
                            f"{path.name} line {line_number} is not valid JSON."
                        ) from exc
                    if _nested_value(existing, identity_path) != identity:
                        continue
                    if _canonical_json(existing) != canonical:
                        raise TurnCollectionIntegrityError(
                            f"Identity {identity!r} already has different content."
                        )
                    return False
            encoded = (canonical + "\n").encode("utf-8")
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                offset = 0
                while offset < len(encoded):
                    offset += os.write(descriptor, encoded[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return True

    def _pending_path(self, action_event_id: str) -> Path:
        digest = hashlib.sha256(action_event_id.encode("utf-8")).hexdigest()
        return self.pending_dir / f"{digest}.json"

    def _failure_path(self, action_event_id: str) -> Path:
        digest = hashlib.sha256(action_event_id.encode("utf-8")).hexdigest()
        return self.failure_dir / f"{digest}.json"

    def _read_final_action(
        self,
        action_event_id: str,
    ) -> dict[str, object] | None:
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import get_action

            stored = get_action(self._postgres_action_id(action_event_id))
            if stored is None or stored[1] != "COMPLETED":
                return None
            return stored[0]
        if not self.turn_path.exists():
            return None
        for line_number, raw_line in enumerate(
            self.turn_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TurnCollectionIntegrityError(
                    f"{self.turn_path.name} line {line_number} is not valid JSON."
                ) from exc
            if _nested_value(
                record,
                ("identity", "action_event_id"),
            ) == action_event_id:
                return record
        return None

    def finalized_actions_for_attempt(
        self,
        attempt_id: str,
    ) -> list[dict[str, object]]:
        """Return durable finalized or restart-preserved actions in order."""

        canonical_attempt_id = str(attempt_id).strip()
        if not canonical_attempt_id:
            raise ValueError("attempt_id must be non-empty.")
        records_by_id: dict[str, dict[str, object]] = {}
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import actions_for_attempt

            records = actions_for_attempt(
                canonical_attempt_id,
                include_pending=self.preserve_pending_for_restart,
                schema_version=SCHEMA_VERSION,
            )
            records_by_id = {
                str(_nested_value(record, ("identity", "action_event_id"))): record
                for record in records
            }
        else:
            records_by_id = {}
        with self._lock, _exclusive_lock(self.lock_path):
            if not self._postgres_enabled() and self.turn_path.exists():
                for line_number, raw_line in enumerate(
                    self.turn_path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise TurnCollectionIntegrityError(
                            f"{self.turn_path.name} line {line_number} is not valid JSON."
                        ) from exc
                    identity = record.get("identity")
                    if not isinstance(identity, Mapping):
                        continue
                    if identity.get("attempt_id") == canonical_attempt_id:
                        records_by_id[str(identity.get("action_event_id"))] = record
            if not self._postgres_enabled() and self.preserve_pending_for_restart:
                for pending_path in sorted(self.pending_dir.glob("*.json")):
                    try:
                        record = json.loads(pending_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise TurnCollectionIntegrityError(
                            f"{pending_path.name} is not valid JSON."
                        ) from exc
                    identity = record.get("identity")
                    if not isinstance(identity, Mapping):
                        continue
                    if identity.get("attempt_id") != canonical_attempt_id:
                        continue
                    action_event_id = str(identity.get("action_event_id"))
                    existing = records_by_id.get(action_event_id)
                    if existing is not None and _canonical_action_snapshot(existing) != (
                        _canonical_action_snapshot(record)
                    ):
                        raise TurnCollectionIntegrityError(
                            "Final and pending action snapshots disagree."
                        )
                    records_by_id.setdefault(action_event_id, record)

        records = list(records_by_id.values())
        records.sort(key=lambda item: item["identity"]["action_turn_index"])
        for expected_index, record in enumerate(records, start=1):
            identity = record["identity"]
            expected_event_id = f"{canonical_attempt_id}:action:{expected_index}"
            if (
                identity.get("action_turn_index") != expected_index
                or identity.get("action_event_id") != expected_event_id
            ):
                raise TurnCollectionIntegrityError(
                    "Finalized attempt actions are not consecutive."
                )
        return records

    @staticmethod
    def _empty_learning_state_update() -> dict[str, object]:
        return {
            "skill": None,
            "resolver_event_id": None,
            "should_update": None,
            "observation_source": None,
            "outcome": None,
            "update_confidence": None,
            "mastery_before": None,
            "mastery_after": None,
            "delta_mastery": None,
        }

    def _recover_abandoned_pending(self) -> None:
        """Close action snapshots left by a prior, unreconstructable runtime."""

        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import list_pending_actions

            for record in list_pending_actions(SCHEMA_VERSION):
                action_event_id = str(
                    _nested_value(record, ("identity", "action_event_id"))
                )
                failure = record.get("_processing_failure")
                if isinstance(failure, Mapping):
                    self.complete_action(
                        action_event_id,
                        next_learner_observation=failure[
                            "next_learner_observation"
                        ],
                        learning_state_update=self._empty_learning_state_update(),
                    )
                else:
                    self.finalize_without_response(
                        action_event_id,
                        failure_reason="runtime_restart_before_response",
                    )
            return

        for pending_path in sorted(self.pending_dir.glob("*.json")):
            try:
                record = json.loads(pending_path.read_text(encoding="utf-8"))
                action_event_id = str(
                    _nested_value(record, ("identity", "action_event_id"))
                )
                failure_path = self._failure_path(action_event_id)
                finalized = self._read_final_action(action_event_id)
                if finalized is not None:
                    if _canonical_action_snapshot(finalized) != (
                        _canonical_action_snapshot(record)
                    ):
                        raise TurnCollectionIntegrityError(
                            "Final row and recoverable pending action disagree."
                        )
                    # The append was durable and the process stopped before
                    # cleanup. The final row wins; never reclassify it.
                    pending_path.unlink(missing_ok=True)
                    failure_path.unlink(missing_ok=True)
                    continue
                if failure_path.exists():
                    failure = json.loads(
                        failure_path.read_text(encoding="utf-8")
                    )
                    if failure.get("action_event_id") != action_event_id:
                        raise TurnCollectionIntegrityError(
                            "Pending processing-failure identity mismatch."
                        )
                    record["next_learner_observation"] = failure[
                        "next_learner_observation"
                    ]
                else:
                    record["next_learner_observation"] = {
                        "status": "censored_no_response",
                        "student_text": None,
                        "evaluator": None,
                        "resolver": None,
                        "failure_reason": "runtime_restart_before_response",
                    }
                record["learning_state_update"] = (
                    self._empty_learning_state_update()
                )
                self._append_unique(
                    self.turn_path,
                    record,
                    ("identity", "action_event_id"),
                )
                pending_path.unlink(missing_ok=True)
                failure_path.unlink(missing_ok=True)
            except Exception as exc:
                raise TurnCollectionIntegrityError(
                    f"Could not recover pending action file {pending_path.name}."
                ) from exc

    def stage_action(self, base_record: Mapping[str, object]) -> bool:
        record = copy.deepcopy(dict(_json_safe(base_record)))
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}.")
        raw_action_id = _nested_value(record, ("identity", "action_event_id"))
        if not isinstance(raw_action_id, str) or not raw_action_id.strip():
            raise ValueError("identity.action_event_id must be a non-empty string.")
        action_id = raw_action_id.strip()
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import (
                get_action,
                stage_action as postgres_stage_action,
            )

            storage_action_id = self._postgres_action_id(action_id)
            existing = get_action(storage_action_id)
            postgres_stage_action(record, action_id_override=storage_action_id)
            return existing is None
        pending_path = self._pending_path(action_id)
        canonical = _canonical_json(record)
        with self._lock, _exclusive_lock(self.lock_path):
            if self.turn_path.exists():
                for raw_line in self.turn_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if not raw_line.strip():
                        continue
                    existing = json.loads(raw_line)
                    if _nested_value(
                        existing,
                        ("identity", "action_event_id"),
                    ) == action_id:
                        if _canonical_action_snapshot(existing) != (
                            _canonical_action_snapshot(record)
                        ):
                            raise TurnCollectionIntegrityError(
                                "Finalized action identity was replayed with "
                                "different action content."
                            )
                        return False
            if pending_path.exists():
                existing = json.loads(
                    pending_path.read_text(encoding="utf-8")
                )
                if _canonical_action_snapshot(existing) != (
                    _canonical_action_snapshot(record)
                ):
                    raise TurnCollectionIntegrityError(
                        "Staged action identity was replayed with different content."
                    )
                return False
            temporary = pending_path.with_name(
                f".{pending_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(canonical + "\n", encoding="utf-8")
            os.replace(temporary, pending_path)
        return True

    def stage_runtime_action(
        self,
        *,
        state: Mapping[str, object],
        history_before_action: Sequence[Mapping[str, object]],
        turn_result: Mapping[str, object],
    ) -> dict[str, object]:
        selector = copy.deepcopy(dict(turn_result.get("selector", {})))
        selector.update(copy.deepcopy(self.version_provenance.get("selector", {})))
        history = copy.deepcopy(list(history_before_action))
        latest_student_text = None
        for turn in reversed(history):
            if str(turn.get("user", "")).casefold() == "student":
                text = turn.get("text")
                if isinstance(text, str) and text.strip():
                    latest_student_text = text.strip()
                    break
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "provenance": {
                "data_mode": self.data_mode,
                "timestamp": _utc_now(),
                **copy.deepcopy(self.version_provenance.get("runtime", {})),
            },
            "identity": {
                "student_pseudonymous_id": self.pseudonymize_student_id(
                    state.get("student_id")
                ),
                "thread_id": state.get("thread_id"),
                "attempt_id": turn_result.get("attempt_id"),
                "adaptive_attempt_index": turn_result.get(
                    "adaptive_attempt_index"
                ),
                "action_event_id": turn_result.get("action_event_id"),
                "action_turn_index": turn_result.get("action_turn_index"),
                "thread_turn_count": turn_result.get("thread_turn_count"),
            },
            "state_before_action": {
                "problem": state.get("question"),
                "history_before_action": history,
                "latest_student_text": latest_student_text,
                "target_skill": state.get("target_skill"),
                "teaching_phase": state.get("teaching_phase"),
                "attempt_start_mastery": turn_result.get(
                    "attempt_start_mastery"
                ),
                "attempt_started_at": turn_result.get("attempt_started_at"),
                "mastery_at_action": turn_result.get("mastery_at_action"),
            },
            "selector": selector,
            "adaptive_decision": copy.deepcopy(
                turn_result.get("adaptive_decision", {})
            ),
            "tutor_response": turn_result.get("tutor_response"),
            "tutor_quality": {
                "source": "frozen_mrb1",
                "scores": copy.deepcopy(turn_result.get("mrb1_scores", {})),
                **copy.deepcopy(
                    self.version_provenance.get("tutor_quality", {})
                ),
            },
        }
        self.stage_action(record)
        return record

    def note_processing_failure(
        self,
        action_event_id: str,
        *,
        dialogue_evidence: Mapping[str, object],
        failure_reason: str,
    ) -> bool:
        """Durably remember that a response existed but processing failed.

        This is provisional: a successful retry replaces it with the actual
        resolver/BKT result. A restart or abort closes it as processing_failed.
        """

        action_id = str(action_event_id).strip()
        if not action_id:
            raise ValueError("action_event_id must be non-empty.")
        pending_path, _ = self._load_pending(action_id)
        del pending_path
        reason = str(failure_reason).strip()
        if not reason:
            raise ValueError("failure_reason must be non-empty.")
        payload = {
            "action_event_id": action_id,
            "next_learner_observation": {
                "status": "processing_failed",
                "student_text": dialogue_evidence.get("student_text"),
                "evaluator": {
                    "correctness": dialogue_evidence.get("correctness"),
                    "reported_confidence": dialogue_evidence.get(
                        "reported_evaluator_confidence"
                    ),
                    "applied_confidence": dialogue_evidence.get(
                        "applied_evaluator_confidence"
                    ),
                    "reason": dialogue_evidence.get("evaluator_reason"),
                    "source": dialogue_evidence.get("evaluator_source"),
                },
                "resolver": None,
                "failure_reason": reason,
            },
        }
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import patch_pending_action

            return patch_pending_action(
                self._postgres_action_id(action_id),
                {"_processing_failure": payload},
            )
        failure_path = self._failure_path(action_id)
        canonical = _canonical_json(payload)
        with self._lock, _exclusive_lock(self.lock_path):
            if failure_path.exists():
                existing = failure_path.read_text(encoding="utf-8").strip()
                if existing != canonical:
                    raise TurnCollectionIntegrityError(
                        "Processing failure was replayed with different evidence."
                    )
                return False
            temporary = failure_path.with_name(
                f".{failure_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(canonical + "\n", encoding="utf-8")
            os.replace(temporary, failure_path)
        return True

    def _load_pending(self, action_event_id: str) -> tuple[Path, dict[str, object]]:
        pending_path = self._pending_path(action_event_id)
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import get_action

            stored = get_action(self._postgres_action_id(action_event_id))
            if stored is None or stored[1] != "PENDING":
                raise TurnCollectionIntegrityError(
                    f"No staged action exists for {action_event_id!r}."
                )
            return pending_path, stored[0]
        if not pending_path.exists():
            raise TurnCollectionIntegrityError(
                f"No staged action exists for {action_event_id!r}."
            )
        value = json.loads(pending_path.read_text(encoding="utf-8"))
        if _nested_value(value, ("identity", "action_event_id")) != action_event_id:
            raise TurnCollectionIntegrityError("Pending action file identity mismatch.")
        return pending_path, value

    def complete_action(
        self,
        action_event_id: str,
        *,
        next_learner_observation: Mapping[str, object],
        learning_state_update: Mapping[str, object],
    ) -> bool:
        action_id = str(action_event_id).strip()
        if not action_id:
            raise ValueError("action_event_id must be non-empty.")
        status = next_learner_observation.get("status")
        if status not in OBSERVATION_STATUSES:
            raise ValueError(f"Unknown next-learner status: {status!r}.")
        safe_next_observation = copy.deepcopy(
            dict(_json_safe(next_learner_observation))
        )
        safe_learning_update = copy.deepcopy(
            dict(_json_safe(learning_state_update))
        )
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import (
                complete_action as postgres_complete_action,
                get_action,
            )

            storage_action_id = self._postgres_action_id(action_id)
            stored = get_action(storage_action_id)
            if stored is None:
                raise TurnCollectionIntegrityError(
                    f"No staged action exists for {action_id!r}."
                )
            if stored[1] == "COMPLETED":
                if (
                    stored[0].get("next_learner_observation")
                    != safe_next_observation
                    or stored[0].get("learning_state_update")
                    != safe_learning_update
                ):
                    raise TurnCollectionIntegrityError(
                        "Finalized action outcome was replayed with different content."
                    )
                return False
            postgres_complete_action(
                action_id=storage_action_id,
                outcome={
                    "next_learner_observation": safe_next_observation,
                    "learning_state_update": safe_learning_update,
                },
                experience_event_id=f"passive:turn_outcomes:{action_id}",
            )
            return True
        if not self._pending_path(action_id).exists():
            with self._lock, _exclusive_lock(self.lock_path):
                finalized = self._read_final_action(action_id)
                if finalized is None:
                    raise TurnCollectionIntegrityError(
                        f"No staged action exists for {action_id!r}."
                    )
                if (
                    finalized.get("next_learner_observation")
                    != safe_next_observation
                    or finalized.get("learning_state_update")
                    != safe_learning_update
                ):
                    raise TurnCollectionIntegrityError(
                        "Finalized action outcome was replayed with different content."
                    )
                return False
        pending_path, record = self._load_pending(action_id)
        record["next_learner_observation"] = safe_next_observation
        record["learning_state_update"] = safe_learning_update
        appended = self._append_unique(
            self.turn_path,
            record,
            ("identity", "action_event_id"),
        )
        with self._lock, _exclusive_lock(self.lock_path):
            pending_path.unlink(missing_ok=True)
            self._failure_path(action_id).unlink(missing_ok=True)
        return appended

    def complete_dialogue_action(
        self,
        *,
        action_event_id: str,
        dialogue_evidence: Mapping[str, object],
        student_model_result: Mapping[str, object],
    ) -> bool:
        if dialogue_evidence.get("action_event_id") != action_event_id:
            raise TurnCollectionIntegrityError(
                "Dialogue evidence action identity does not match completion."
            )
        raw_resolved = student_model_result.get("resolved_events", [])
        resolved = [
            dict(_json_safe(item))
            for item in raw_resolved
            if isinstance(_json_safe(item), Mapping)
        ]
        matching = [
            item
            for item in resolved
            if item.get("source_action_event_id") == action_event_id
        ]
        if len(matching) != 1:
            raise TurnCollectionIntegrityError(
                "Dialogue result must retain exactly one source action identity."
            )
        resolver = matching[0]
        graph_result = student_model_result.get("knowledge_graph_result", {})
        raw_observations = (
            graph_result.get("observation_results", [])
            if isinstance(graph_result, Mapping)
            else []
        )
        observations = [
            dict(item)
            for item in raw_observations
            if isinstance(item, Mapping)
            and item.get("event_id") == resolver.get("event_id")
        ]
        if len(observations) != 1:
            raise TurnCollectionIntegrityError(
                "Dialogue resolver event has no unique persisted result."
            )
        observation = observations[0]
        should_update = observation.get("should_update")
        if not isinstance(should_update, bool):
            raise TurnCollectionIntegrityError(
                "Persisted observation must expose an explicit should_update flag."
            )
        status = "observed_update" if should_update else "resolved_no_update"
        evaluator = {
            "correctness": dialogue_evidence.get("correctness"),
            "reported_confidence": dialogue_evidence.get(
                "reported_evaluator_confidence"
            ),
            "applied_confidence": dialogue_evidence.get(
                "applied_evaluator_confidence"
            ),
            "reason": dialogue_evidence.get("evaluator_reason"),
            "source": dialogue_evidence.get("evaluator_source"),
        }
        bkt_update = resolver.get("bkt_update", {})
        resolver_projection = {
            "version": resolver.get("resolver_version"),
            "primary_signal": resolver.get("primary_signal"),
            "observation_source": (
                bkt_update.get("observation_source")
                if isinstance(bkt_update, Mapping)
                else None
            ),
            "evidence_weight": (
                bkt_update.get("evidence_weight")
                if isinstance(bkt_update, Mapping)
                else None
            ),
            "behaviour_factor": (
                bkt_update.get("behaviour_factor")
                if isinstance(bkt_update, Mapping)
                else None
            ),
            "contributors": (
                bkt_update.get("contributors")
                if isinstance(bkt_update, Mapping)
                else []
            ),
        }
        next_observation = {
            "status": status,
            "student_text": dialogue_evidence.get("student_text"),
            "evaluator": evaluator,
            "resolver": resolver_projection,
        }
        learning_update = {
            "skill": observation.get("skill"),
            "resolver_event_id": observation.get("event_id"),
            "should_update": should_update,
            "observation_source": observation.get("observation_source"),
            "outcome": observation.get("outcome"),
            "update_confidence": observation.get("update_confidence"),
            "mastery_before": observation.get("mastery_before"),
            "mastery_after": observation.get("mastery_after"),
            "delta_mastery": observation.get("delta_mastery"),
        }
        return self.complete_action(
            action_event_id,
            next_learner_observation=next_observation,
            learning_state_update=learning_update,
        )

    def finalize_pending_action(self, action_event_id: str) -> bool:
        """Close a pending action using its durable failure marker if present."""

        action_id = str(action_event_id).strip()
        if self._postgres_enabled():
            from src.self_improvement.postgres_repository import get_action

            stored = get_action(self._postgres_action_id(action_id))
            failure = None if stored is None else stored[0].get("_processing_failure")
            if not isinstance(failure, Mapping):
                return self.finalize_without_response(action_id)
            return self.complete_action(
                action_id,
                next_learner_observation=failure["next_learner_observation"],
                learning_state_update=self._empty_learning_state_update(),
            )
        failure_path = self._failure_path(action_id)
        if not failure_path.exists():
            return self.finalize_without_response(action_id)
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("action_event_id") != action_id:
            raise TurnCollectionIntegrityError(
                "Pending processing-failure identity mismatch."
            )
        return self.complete_action(
            action_id,
            next_learner_observation=failure["next_learner_observation"],
            learning_state_update=self._empty_learning_state_update(),
        )

    def finalize_without_response(
        self,
        action_event_id: str,
        *,
        status: str = "censored_no_response",
        failure_reason: str | None = None,
    ) -> bool:
        if status not in {"censored_no_response", "processing_failed"}:
            raise ValueError("Terminal no-evidence status is invalid.")
        return self.complete_action(
            action_event_id,
            next_learner_observation={
                "status": status,
                "student_text": None,
                "evaluator": None,
                "resolver": None,
                "failure_reason": failure_reason,
            },
            learning_state_update={
                **self._empty_learning_state_update(),
            },
        )

    def _artifact_provenance(
        self,
        supplied: object,
    ) -> dict[str, object]:
        """Build standalone, portable provenance without executing components."""

        safe_supplied = _without_local_policy_paths(supplied)
        provided = (
            copy.deepcopy(dict(safe_supplied))
            if isinstance(safe_supplied, Mapping)
            else {}
        )
        timestamp = provided.pop("timestamp", None)
        if timestamp is None:
            timestamp = _utc_now()
        else:
            timestamp = _validate_utc_timestamp(timestamp, "provenance.timestamp")

        runtime = copy.deepcopy(
            dict(_json_safe(self.version_provenance.get("runtime", {})))
        )
        selector = copy.deepcopy(
            dict(_json_safe(self.version_provenance.get("selector", {})))
        )
        tutor_quality = copy.deepcopy(
            dict(_json_safe(self.version_provenance.get("tutor_quality", {})))
        )
        safe_policy = _without_local_policy_paths(
            self.version_provenance.get("policy", {})
        )
        policy = (
            copy.deepcopy(dict(safe_policy))
            if isinstance(safe_policy, Mapping)
            else {}
        )
        relative_policy_path = policy.get("policy_state_relative_path")
        if isinstance(relative_policy_path, str) and (
            PureWindowsPath(relative_policy_path).is_absolute()
            or PurePosixPath(relative_policy_path).is_absolute()
        ):
            policy["policy_state_relative_path"] = None
        return {
            **provided,
            **runtime,
            "timestamp": timestamp,
            "data_mode": self.data_mode,
            "active_selector": selector,
            "tutor_quality": tutor_quality,
            "policy": policy,
        }

    def record_assessment_outcome(self, record: Mapping[str, object]) -> bool:
        safe_record = copy.deepcopy(dict(_json_safe(record)))
        safe_record["provenance"] = self._artifact_provenance(
            safe_record.get("provenance")
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **safe_record,
        }
        return self._append_unique(
            self.assessment_path,
            payload,
            ("identity", "attempt_id"),
        )

    def record_attempt_summary(self, record: Mapping[str, object]) -> bool:
        safe_record = _without_local_policy_paths(record)
        if not isinstance(safe_record, Mapping):
            raise TypeError("Attempt summary must be a mapping.")
        safe_record = copy.deepcopy(dict(safe_record))
        safe_record["attempt_started_at"] = _validate_utc_timestamp(
            safe_record.get("attempt_started_at"),
            "attempt_started_at",
        )
        completion_status = safe_record.get("completion_status")
        if completion_status not in COMPLETION_STATUSES:
            raise ValueError(
                "completion_status must be exactly 'completed' or 'aborted'."
            )
        safe_record["provenance"] = self._artifact_provenance(
            safe_record.get("provenance")
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **safe_record,
        }
        return self._append_unique(
            self.attempt_path,
            payload,
            ("identity", "attempt_id"),
        )


__all__ = (
    "COMPLETION_STATUSES",
    "DATASET_DIRECTORY_NAME",
    "OBSERVATION_STATUSES",
    "PassiveTurnCollector",
    "SCHEMA_VERSION",
    "TurnCollectionIntegrityError",
)
