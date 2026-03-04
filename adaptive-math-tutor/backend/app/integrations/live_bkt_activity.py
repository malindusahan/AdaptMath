"""Append-only BKT activity logging for the ordinary adaptive runtime.

The manual controlled-live experiment introduced the authoritative
``adaptmath_bkt_activity_v2`` serializers.  This adapter deliberately reuses
those serializers without the manual experiment's policy immutability guards
or summary artifacts, so ordinary LIVE runs can expose the same audit stream.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from app.integrations.manual_controlled_live import (
    BKT_ACTIVITY_SCHEMA_VERSION,
    ManualControlledLiveDiagnostics,
)


class LiveBKTActivityLog(ManualControlledLiveDiagnostics):
    """Persist ordinary LIVE dialogue and assessment BKT transactions.

    Existing durable identities are loaded on startup. Re-delivery of an
    already-recorded event is therefore an idempotent no-op, including after a
    backend restart.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.bkt_activity_log_path = self.output_dir / "bkt_activity.jsonl"
        self.runtime_id = uuid4().hex
        self._lock = RLock()
        self._bkt_activity_count = 0
        self._recorded_bkt_attempts: set[str] = set()
        self._recorded_dialogue_turns: set[tuple[str, int]] = set()
        self._assessment_records: dict[str, dict[str, object]] = {}
        self._dialogue_records: dict[tuple[str, int], dict[str, object]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bkt_activity_log_path.touch(exist_ok=True)
        self._load_existing_records()

    def _load_existing_records(self) -> None:
        for line_number, raw_line in enumerate(
            self.bkt_activity_log_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"bkt_activity.jsonl line {line_number} is not valid JSON."
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"bkt_activity.jsonl line {line_number} must be an object."
                )
            if record.get("schema_version") != BKT_ACTIVITY_SCHEMA_VERSION:
                raise ValueError(
                    "bkt_activity.jsonl contains an incompatible schema version."
                )
            attempt_id = str(record.get("attempt_id", "")).strip()
            if not attempt_id:
                raise ValueError(
                    f"bkt_activity.jsonl line {line_number} has no attempt_id."
                )
            event = record.get("event")
            safe_record = deepcopy(dict(record))
            if event == "assessment_cycle_bkt_update":
                if attempt_id in self._assessment_records:
                    raise ValueError(
                        f"Duplicate assessment BKT activity for {attempt_id!r}."
                    )
                self._recorded_bkt_attempts.add(attempt_id)
                self._assessment_records[attempt_id] = safe_record
            elif event == "dialogue_turn_bkt_update":
                raw_turn_index = record.get("turn_index")
                if (
                    isinstance(raw_turn_index, bool)
                    or not isinstance(raw_turn_index, int)
                    or raw_turn_index < 1
                ):
                    raise ValueError(
                        f"bkt_activity.jsonl line {line_number} has an invalid turn_index."
                    )
                key = (attempt_id, raw_turn_index)
                if key in self._dialogue_records:
                    raise ValueError(
                        "Duplicate dialogue BKT activity for "
                        f"{attempt_id!r} turn {raw_turn_index}."
                    )
                self._recorded_dialogue_turns.add(key)
                self._dialogue_records[key] = safe_record
            else:
                raise ValueError(
                    f"bkt_activity.jsonl line {line_number} has an unknown event."
                )
        self._bkt_activity_count = (
            len(self._assessment_records) + len(self._dialogue_records)
        )

    def assert_safety(self) -> None:
        """Ordinary LIVE mode intentionally permits policy persistence."""

    def _write_summary(self) -> None:
        """The ordinary runtime already has authoritative attempt summaries."""

    def record_bkt_activity(self, **activity: Any) -> dict[str, object]:
        attempt_id = str(activity.get("attempt_id", "")).strip()
        with self._lock:
            existing = self._assessment_records.get(attempt_id)
            if existing is not None:
                return deepcopy(existing)
            record = super().record_bkt_activity(**activity)
            self._assessment_records[attempt_id] = deepcopy(record)
            return record

    def record_dialogue_bkt_activity(
        self,
        **activity: Any,
    ) -> dict[str, object]:
        attempt_id = str(activity.get("attempt_id", "")).strip()
        dialogue_evidence = activity.get("dialogue_evidence")
        turn_index = (
            dialogue_evidence.get("turn_index")
            if isinstance(dialogue_evidence, Mapping)
            else None
        )
        key = (
            (attempt_id, turn_index)
            if isinstance(turn_index, int) and not isinstance(turn_index, bool)
            else None
        )
        with self._lock:
            existing = self._dialogue_records.get(key) if key is not None else None
            if existing is not None:
                return deepcopy(existing)
            record = super().record_dialogue_bkt_activity(**activity)
            canonical_key = (attempt_id, int(record["turn_index"]))
            self._dialogue_records[canonical_key] = deepcopy(record)
            return record


__all__ = ("LiveBKTActivityLog",)
