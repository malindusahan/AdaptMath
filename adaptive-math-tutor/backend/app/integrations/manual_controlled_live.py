"""Manual controlled-live configuration, logging, and safety guards."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from math import isclose
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, Final
from uuid import uuid4


MANUAL_CONTROLLED_LIVE_ENV: Final[str] = (
    "ADAPTIVE_MANUAL_CONTROLLED_LIVE"
)
MANUAL_CONTROLLED_LIVE_VALUE: Final[str] = "enabled-v1"
TURN_LOG_SCHEMA_VERSION: Final[str] = "md7r1_manual_controlled_live_turn_v2"
BKT_ACTIVITY_SCHEMA_VERSION: Final[str] = "adaptmath_bkt_activity_v2"
SUMMARY_SCHEMA_VERSION: Final[str] = "md7r1_manual_controlled_live_summary_v1"
MOVE_ORDER: Final[tuple[str, ...]] = (
    "generic",
    "probing",
    "focus",
    "telling",
)


def manual_controlled_live_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Enable only for the exact versioned value; reject ambiguous values."""

    source = os.environ if environment is None else environment
    raw = source.get(MANUAL_CONTROLLED_LIVE_ENV)
    if raw is None or raw == "":
        return False
    if raw != MANUAL_CONTROLLED_LIVE_VALUE:
        raise ValueError(
            f"{MANUAL_CONTROLLED_LIVE_ENV} must be absent or exactly "
            f"{MANUAL_CONTROLLED_LIVE_VALUE!r}; got {raw!r}."
        )
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(path: str | Path) -> dict[str, object]:
    """Capture existence, bytes, timestamp, and content hash for one path."""

    source = Path(path).resolve()
    if not source.exists():
        return {"path": str(source), "exists": False}
    if not source.is_file():
        raise ValueError(f"Protected path is not a regular file: {source}")
    stat = source.stat()
    return {
        "path": str(source),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(source),
    }


def posterior_snapshot(policy: Any) -> dict[str, object]:
    """Snapshot learned posterior fields while intentionally excluding RNG."""

    state = policy.state_dict()
    return {
        "A": state["A"],
        "b": state["b"],
        "arm_update_counts": state["arm_update_counts"],
        "total_updates": state["total_updates"],
    }


def _probability_diagnostics(
    probabilities: Mapping[str, object],
) -> dict[str, object]:
    if tuple(probabilities) != MOVE_ORDER:
        raise ValueError("Selector probabilities use an incompatible label order.")
    ranked = sorted(
        ((move, float(probabilities[move])) for move in MOVE_ORDER),
        key=lambda item: (-item[1], MOVE_ORDER.index(item[0])),
    )
    return {
        **{f"p_{move}": float(probabilities[move]) for move in MOVE_ORDER},
        "argmax": ranked[0][0],
        "top1_probability": ranked[0][1],
        "top2_probability": ranked[1][1],
        "top1_top2_gap": float(ranked[0][1] - ranked[1][1]),
    }


def _latest_student_text(
    history: Sequence[Mapping[str, object]],
) -> str | None:
    for turn in reversed(history):
        if turn.get("user") == "student":
            text = turn.get("text")
            return str(text) if text is not None else None
    return None


def _json_ready(value: object) -> object:
    """Project Pydantic/enums/containers into strict JSON-compatible values."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_ready(model_dump(mode="json"))
        except TypeError:
            return _json_ready(model_dump())
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _resolved_event_mapping(value: object) -> dict[str, object]:
    converted = _json_ready(value)
    if not isinstance(converted, dict):
        raise TypeError("Resolved BKT event must serialize to a mapping.")
    return converted


def _bkt_observation(event: Mapping[str, object]) -> tuple[int, float] | None:
    update = event.get("bkt_update")
    if not isinstance(update, Mapping) or update.get("should_update") is not True:
        return None
    outcome = update.get("outcome")
    confidence = update.get("update_confidence")
    if isinstance(outcome, bool) or outcome not in (0, 1):
        raise ValueError("Applied BKT event requires binary outcome.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("Applied BKT event requires numeric update confidence.")
    numeric_confidence = float(confidence)
    if not 0.0 <= numeric_confidence <= 1.0:
        raise ValueError("BKT update confidence must be in [0, 1].")
    return int(outcome), numeric_confidence


class ManualControlledLiveDiagnostics:
    """Append paired turn diagnostics and continuously prove no policy writes."""

    def __init__(
        self,
        *,
        selector: Any,
        policy: Any,
        output_dir: str | Path,
        production_policy_path: str | Path,
        production_experience_path: str | Path,
    ) -> None:
        if not isinstance(getattr(selector, "records", None), list):
            raise TypeError("selector must expose paired diagnostic records.")
        if not callable(getattr(policy, "state_dict", None)):
            raise TypeError("policy must expose state_dict().")
        self.selector = selector
        self.policy = policy
        self.output_dir = Path(output_dir).resolve()
        self.turn_log_path = self.output_dir / "manual_live_turns.jsonl"
        self.bkt_activity_log_path = self.output_dir / "bkt_activity.jsonl"
        self.summary_path = (
            self.output_dir / "manual_live_session_summary.json"
        )
        self.production_policy_path = Path(production_policy_path).resolve()
        self.production_experience_path = Path(
            production_experience_path
        ).resolve()
        self.runtime_id = uuid4().hex
        self.started_at_utc = datetime.now(UTC).isoformat()
        self._lock = RLock()
        self._initial_posterior = posterior_snapshot(policy)
        self._production_before = {
            "policy_state": file_snapshot(self.production_policy_path),
            "experience_log": file_snapshot(self.production_experience_path),
        }
        self._selector_record_count = len(selector.records)
        self._turn_count = 0
        self._bkt_activity_count = 0
        self._recorded_bkt_attempts: set[str] = set()
        self._recorded_dialogue_turns: set[tuple[str, int]] = set()
        self._completed_attempts: list[dict[str, object]] = []
        self._write_summary()

    def _current_protected_files(self) -> dict[str, dict[str, object]]:
        return {
            "policy_state": file_snapshot(self.production_policy_path),
            "experience_log": file_snapshot(self.production_experience_path),
        }

    def assert_safety(self) -> None:
        if posterior_snapshot(self.policy) != self._initial_posterior:
            raise RuntimeError(
                "Manual controlled-live LinTS posterior changed unexpectedly."
            )
        current = self._current_protected_files()
        if current != self._production_before:
            raise RuntimeError(
                "Manual controlled-live changed a protected adaptive persistence file."
            )

    def _summary_payload(self) -> dict[str, object]:
        current = self._current_protected_files()
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "runtime_id": self.runtime_id,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "mode": "manual_controlled_live",
            "active_selector": "MD7-R1 epoch 3",
            "shadow_selector": "frozen MD6",
            "learner_agency_escalation": (
                "explicit answer request or two consecutive non-engagement "
                "turns force telling"
            ),
            "lints_posterior_updates": 0,
            "real_adaptive_persistence_writes": 0,
            "turn_count": self._turn_count,
            "bkt_activity_count": self._bkt_activity_count,
            "bkt_activity_log_path": str(self.bkt_activity_log_path),
            "completed_attempts": list(self._completed_attempts),
            "turn_log_path": str(self.turn_log_path),
            "protected_files_before": self._production_before,
            "protected_files_current": current,
            "protected_files_unchanged": current == self._production_before,
        }

    def _write_summary(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                self._summary_payload(),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.summary_path.name}.",
                suffix=".tmp",
                dir=self.output_dir,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.summary_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def record_turn(
        self,
        *,
        attempt_id: str,
        turn_result: Mapping[str, object],
        decision: Any,
    ) -> dict[str, object]:
        """Append exactly one completed live turn without any model/API call."""

        with self._lock:
            self.assert_safety()
            observed_count = len(self.selector.records)
            if observed_count <= self._selector_record_count:
                raise RuntimeError(
                    "Expected a new paired selector record for this tutor turn."
                )
            # A downstream tutor failure can leave an unlogged selector record.
            # The newest record belongs to the completed turn being committed;
            # older uncommitted records remain intentionally absent from the log.
            paired = self.selector.last_record
            history = paired.get("history")
            if isinstance(history, (str, bytes)) or not isinstance(
                history,
                Sequence,
            ):
                raise TypeError("Paired selector history is invalid.")
            md6_probabilities = paired.get("md6_probabilities")
            md7_probabilities = paired.get("md7_probabilities")
            if not isinstance(md6_probabilities, Mapping) or not isinstance(
                md7_probabilities,
                Mapping,
            ):
                raise TypeError("Paired selector probabilities are invalid.")
            effective_probabilities = paired.get(
                "effective_probabilities",
                md7_probabilities,
            )
            if not isinstance(effective_probabilities, Mapping):
                raise TypeError("Effective selector probabilities are invalid.")
            raw_agency = paired.get("learner_agency_escalation")
            learner_agency = (
                dict(raw_agency)
                if isinstance(raw_agency, Mapping)
                else {
                    "triggered": False,
                    "reason": None,
                    "latest_student_text": _latest_student_text(history),
                    "consecutive_non_engagement_turns": 0,
                }
            )
            active_probabilities = turn_result.get("md6_probabilities")
            if not isinstance(active_probabilities, Mapping):
                raise TypeError("Turn result has no active probability mapping.")
            if any(
                abs(
                    float(active_probabilities[move])
                    - float(effective_probabilities[move])
                )
                > 1e-9
                for move in MOVE_ORDER
            ):
                raise RuntimeError(
                    "C3 did not receive the effective selector probabilities."
                )

            md6 = _probability_diagnostics(md6_probabilities)
            md7 = _probability_diagnostics(md7_probabilities)
            effective = _probability_diagnostics(effective_probabilities)
            if getattr(decision, "base_move") != effective["argmax"]:
                raise RuntimeError(
                    "Overlay base move is not the effective selector argmax."
                )
            record: dict[str, object] = {
                "schema_version": TURN_LOG_SCHEMA_VERSION,
                "runtime_id": self.runtime_id,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "attempt_id": attempt_id,
                "turn_index": int(turn_result["turn_index"]),
                "problem": paired["problem"],
                "latest_student_text": _latest_student_text(history),
                "history_turn_count": len(history),
                "md6": md6,
                "md7": md7,
                "transition": f"{md6['argmax']} -> {md7['argmax']}",
                "selector_effective": effective,
                "learner_agency": learner_agency,
                "overlay": {
                    "eligible_arms": list(turn_result["eligible_arms"]),
                    "selected_arm": turn_result["selected_arm"],
                    "base_move": getattr(decision, "base_move"),
                    "final_move": turn_result["pedagogical_move"],
                    "overridden": bool(turn_result["overridden"]),
                    "gap": float(getattr(decision, "gap")),
                    "gap_threshold": float(
                        getattr(decision, "gap_threshold")
                    ),
                },
                "tutor": {"tutor_response": turn_result["tutor_response"]},
            }
            scores = turn_result.get("mrb1_scores")
            if isinstance(scores, Mapping):
                record["mrb1_scores"] = {
                    str(name): float(value) for name, value in scores.items()
                }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            from src.self_improvement.postgres_repository import (
                append_experience,
                postgres_enabled,
            )
            if postgres_enabled():
                append_experience(
                    record,
                    event_id_override=(
                        f"manual-turn:{record.get('action_event_id', observed_count)}"
                    ),
                )
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with self.turn_log_path.open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._selector_record_count = observed_count
            self._turn_count += 1
            self.assert_safety()
            self._write_summary()
            return record

    def record_bkt_activity(
        self,
        *,
        student_id: object,
        attempt_id: object,
        target_skill: object,
        evaluated_answers: object,
        student_model_result: object,
        knowledge_graph: Any,
    ) -> dict[str, object]:
        """Append one exact assessment-cycle BKT transaction as JSONL."""

        with self._lock:
            self.assert_safety()
            for name, value in (
                ("student_id", student_id),
                ("attempt_id", attempt_id),
                ("target_skill", target_skill),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string.")
            canonical_attempt_id = str(attempt_id).strip()
            if canonical_attempt_id in self._recorded_bkt_attempts:
                raise RuntimeError(
                    f"BKT activity already recorded for {canonical_attempt_id}."
                )
            if not isinstance(student_model_result, Mapping):
                raise TypeError("student_model_result must be a mapping.")
            if isinstance(evaluated_answers, (str, bytes)) or not isinstance(
                evaluated_answers,
                Sequence,
            ):
                raise TypeError("evaluated_answers must be a sequence.")

            learning_outcome = student_model_result.get("learning_outcome")
            knowledge_result = student_model_result.get("knowledge_graph_result")
            raw_events = student_model_result.get("resolved_events")
            if not isinstance(learning_outcome, Mapping):
                raise ValueError("BKT activity requires learning_outcome.")
            if not isinstance(knowledge_result, Mapping):
                raise ValueError("BKT activity requires knowledge_graph_result.")
            if isinstance(raw_events, (str, bytes)) or not isinstance(
                raw_events,
                Sequence,
            ):
                raise TypeError("BKT activity requires resolved_events.")

            skill = str(target_skill).strip()
            if learning_outcome.get("attempt_id") != canonical_attempt_id:
                raise ValueError("BKT learning outcome attempt_id mismatch.")
            if learning_outcome.get("skill") != skill:
                raise ValueError("BKT learning outcome skill mismatch.")
            mastery_before = float(learning_outcome["mastery_before"])
            mastery_after = float(learning_outcome["mastery_after"])
            mastery_delta = float(learning_outcome["delta_mastery"])
            if not isclose(
                mastery_after - mastery_before,
                mastery_delta,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("BKT learning outcome delta is inconsistent.")

            events = [_resolved_event_mapping(event) for event in raw_events]
            target_events = [
                event for event in events if event.get("skill_id") == skill
            ]
            new_observations = [
                observation
                for observation in (
                    _bkt_observation(event) for event in target_events
                )
                if observation is not None
            ]

            predictor = getattr(knowledge_graph, "predictor", None)
            predict = getattr(predictor, "predict", None)
            params_by_skill = getattr(predictor, "params", None)
            get_attempts = getattr(knowledge_graph, "get_attempts", None)
            get_prior = getattr(
                knowledge_graph,
                "get_effective_initial_prior",
                None,
            )
            if not callable(predict) or not isinstance(params_by_skill, Mapping):
                raise TypeError("Knowledge graph has no compatible BKT predictor.")
            if not callable(get_attempts) or not callable(get_prior):
                raise TypeError("Knowledge graph has no BKT history/prior accessors.")
            raw_params = params_by_skill.get(skill)
            if not isinstance(raw_params, Mapping):
                raise KeyError(f"No BKT parameters found for {skill!r}.")
            parameters = {
                name: float(raw_params[name])
                for name in ("prior", "learns", "guesses", "slips", "forgets")
            }
            prior_record = get_prior(str(student_id).strip(), skill)
            if not isinstance(prior_record, Mapping):
                raise ValueError("Effective BKT initial prior is unavailable.")
            effective_prior = float(prior_record["effective_initial_prior"])

            all_attempts = [
                (int(label), float(confidence))
                for label, confidence in get_attempts(
                    str(student_id).strip(),
                    skill,
                )
            ]
            new_count = len(new_observations)
            history_prefix = all_attempts[:-new_count] if new_count else all_attempts
            persisted_suffix = all_attempts[-new_count:] if new_count else []
            suffix_matches = persisted_suffix == new_observations
            raw_dialogue_summary = student_model_result.get(
                "dialogue_evidence_summary",
                {},
            )
            dialogue_summary = (
                dict(raw_dialogue_summary)
                if isinstance(raw_dialogue_summary, Mapping)
                else {}
            )
            dialogue_observation_count = int(
                dialogue_summary.get("observations_applied", 0)
            )
            if dialogue_observation_count < 0 or dialogue_observation_count > len(
                history_prefix
            ):
                raise ValueError("Dialogue observation count is inconsistent.")
            attempt_history_prefix = (
                history_prefix[:-dialogue_observation_count]
                if dialogue_observation_count
                else history_prefix
            )

            def mastery_for(observations: list[tuple[int, float]]) -> float:
                if not observations:
                    return effective_prior
                return float(
                    predict(
                        skill,
                        observations,
                        initial_prior=effective_prior,
                    )
                )

            recomputed_attempt_start = mastery_for(attempt_history_prefix)
            recomputed_before = mastery_for(history_prefix)
            observation_records: list[dict[str, object]] = []
            applied_so_far: list[tuple[int, float]] = []
            answer_records = [
                dict(answer) if isinstance(answer, Mapping) else {"value": answer}
                for answer in evaluated_answers
            ]
            for index, event in enumerate(target_events):
                observation = _bkt_observation(event)
                transition: dict[str, float] | None = None
                if observation is not None:
                    before = mastery_for(history_prefix + applied_so_far)
                    applied_so_far.append(observation)
                    after = mastery_for(history_prefix + applied_so_far)
                    transition = {
                        "mastery_before_observation": before,
                        "mastery_after_observation": after,
                        "delta_mastery": float(after - before),
                    }
                observation_records.append(
                    {
                        "sequence_in_assessment": index + 1,
                        "assessment_answer": (
                            _json_ready(answer_records[index])
                            if index < len(answer_records)
                            else None
                        ),
                        "resolved_signal": event,
                        "mastery_transition": transition,
                    }
                )

            recomputed_after = mastery_for(all_attempts)
            raw_skill_updates = knowledge_result.get("skills_updated", [])
            skill_updates = (
                list(raw_skill_updates)
                if isinstance(raw_skill_updates, Sequence)
                and not isinstance(raw_skill_updates, (str, bytes))
                else []
            )
            persisted_skill_update = next(
                (
                    _json_ready(update)
                    for update in skill_updates
                    if isinstance(update, Mapping) and update.get("skill") == skill
                ),
                None,
            )
            correct_count = sum(
                answer.get("is_correct") is True
                for answer in answer_records
            )
            record: dict[str, object] = {
                "schema_version": BKT_ACTIVITY_SCHEMA_VERSION,
                "runtime_id": self.runtime_id,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "event": "assessment_cycle_bkt_update",
                "student_id": str(student_id).strip(),
                "attempt_id": canonical_attempt_id,
                "session_id": knowledge_result.get(
                    "session_id",
                    canonical_attempt_id,
                ),
                "target_skill": skill,
                "update_timing": "formal_assessment_completion",
                "assessment": {
                    "answer_count": len(answer_records),
                    "correct_count": correct_count,
                    "incorrect_count": len(answer_records) - correct_count,
                },
                "bkt_parameters": parameters,
                "effective_initial_prior": {
                    "probability": effective_prior,
                    "source": prior_record.get("prior_source"),
                },
                "history": {
                    "observation_count_before_attempt": len(attempt_history_prefix),
                    "dialogue_turns_processed_before_assessment": int(
                        dialogue_summary.get("turns_processed", 0)
                    ),
                    "dialogue_observations_applied_before_assessment": (
                        dialogue_observation_count
                    ),
                    "observation_count_before_assessment": len(history_prefix),
                    "observations_applied_in_assessment": new_count,
                    "observation_count_after_assessment": len(all_attempts),
                    "persisted_suffix_matches_resolved_updates": suffix_matches,
                },
                "mastery": {
                    "before": mastery_before,
                    "after": mastery_after,
                    "delta": mastery_delta,
                    "recomputed_before": recomputed_attempt_start,
                    "before_assessment": recomputed_before,
                    "dialogue_delta_before_assessment": (
                        recomputed_before - recomputed_attempt_start
                    ),
                    "assessment_delta": recomputed_after - recomputed_before,
                    "recomputed_after": recomputed_after,
                    "before_consistent": isclose(
                        mastery_before,
                        recomputed_attempt_start,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                    "after_consistent": isclose(
                        mastery_after,
                        recomputed_after,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                },
                "observations": observation_records,
                "persisted_skill_update": persisted_skill_update,
            }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            from src.self_improvement.postgres_repository import (
                append_experience,
                postgres_enabled,
            )
            if postgres_enabled():
                append_experience(
                    record,
                    event_id_override=f"bkt-assessment:{canonical_attempt_id}",
                )
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with self.bkt_activity_log_path.open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._recorded_bkt_attempts.add(canonical_attempt_id)
            self._bkt_activity_count += 1
            self.assert_safety()
            self._write_summary()
            return record

    def record_dialogue_bkt_activity(
        self,
        *,
        student_id: object,
        attempt_id: object,
        target_skill: object,
        dialogue_evidence: object,
        student_model_result: object,
        knowledge_graph: Any,
    ) -> dict[str, object]:
        """Append one live student-dialogue BKT resolution as JSONL."""

        with self._lock:
            self.assert_safety()
            for name, value in (
                ("student_id", student_id),
                ("attempt_id", attempt_id),
                ("target_skill", target_skill),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string.")
            if not isinstance(dialogue_evidence, Mapping):
                raise TypeError("dialogue_evidence must be a mapping.")
            if not isinstance(student_model_result, Mapping):
                raise TypeError("student_model_result must be a mapping.")

            raw_turn_index = dialogue_evidence.get("turn_index")
            if (
                isinstance(raw_turn_index, bool)
                or not isinstance(raw_turn_index, int)
                or raw_turn_index < 1
            ):
                raise ValueError("Dialogue BKT turn_index must be positive.")
            canonical_attempt_id = str(attempt_id).strip()
            activity_key = (canonical_attempt_id, raw_turn_index)
            if activity_key in self._recorded_dialogue_turns:
                raise RuntimeError(
                    "Dialogue BKT activity already recorded for "
                    f"{canonical_attempt_id} turn {raw_turn_index}."
                )

            knowledge_result = student_model_result.get("knowledge_graph_result")
            incremental = student_model_result.get("incremental_learning_outcome")
            raw_events = student_model_result.get("resolved_events")
            if not isinstance(knowledge_result, Mapping):
                raise ValueError("Dialogue BKT activity requires knowledge_graph_result.")
            if not isinstance(incremental, Mapping):
                raise ValueError(
                    "Dialogue BKT activity requires incremental_learning_outcome."
                )
            if isinstance(raw_events, (str, bytes)) or not isinstance(
                raw_events,
                Sequence,
            ):
                raise TypeError("Dialogue BKT activity requires resolved_events.")

            skill = str(target_skill).strip()
            events = [_resolved_event_mapping(event) for event in raw_events]
            target_events = [
                event for event in events if event.get("skill_id") == skill
            ]
            new_observations = [
                observation
                for observation in (
                    _bkt_observation(event) for event in target_events
                )
                if observation is not None
            ]
            if len(new_observations) > 1:
                raise ValueError(
                    "One dialogue turn cannot create multiple BKT observations."
                )

            mastery_before = float(incremental["mastery_before"])
            mastery_after = float(incremental["mastery_after"])
            mastery_delta = float(incremental["delta_mastery"])
            if not isclose(
                mastery_after - mastery_before,
                mastery_delta,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("Dialogue BKT mastery delta is inconsistent.")

            predictor = getattr(knowledge_graph, "predictor", None)
            predict = getattr(predictor, "predict", None)
            params_by_skill = getattr(predictor, "params", None)
            get_attempts = getattr(knowledge_graph, "get_attempts", None)
            get_prior = getattr(
                knowledge_graph,
                "get_effective_initial_prior",
                None,
            )
            if not callable(predict) or not isinstance(params_by_skill, Mapping):
                raise TypeError("Knowledge graph has no compatible BKT predictor.")
            if not callable(get_attempts) or not callable(get_prior):
                raise TypeError("Knowledge graph has no BKT history/prior accessors.")
            raw_params = params_by_skill.get(skill)
            if not isinstance(raw_params, Mapping):
                raise KeyError(f"No BKT parameters found for {skill!r}.")
            parameters = {
                name: float(raw_params[name])
                for name in ("prior", "learns", "guesses", "slips", "forgets")
            }
            canonical_student_id = str(student_id).strip()
            prior_record = get_prior(canonical_student_id, skill)
            if not isinstance(prior_record, Mapping):
                raise ValueError("Effective BKT initial prior is unavailable.")
            effective_prior = float(prior_record["effective_initial_prior"])
            all_attempts = [
                (int(label), float(confidence))
                for label, confidence in get_attempts(canonical_student_id, skill)
            ]
            new_count = len(new_observations)
            history_prefix = all_attempts[:-new_count] if new_count else all_attempts
            persisted_suffix = all_attempts[-new_count:] if new_count else []
            suffix_matches = persisted_suffix == new_observations

            def mastery_for(observations: list[tuple[int, float]]) -> float:
                if not observations:
                    return effective_prior
                return float(
                    predict(
                        skill,
                        observations,
                        initial_prior=effective_prior,
                    )
                )

            recomputed_before = mastery_for(history_prefix)
            recomputed_after = mastery_for(all_attempts)
            raw_skill_updates = knowledge_result.get("skills_updated", [])
            skill_updates = (
                list(raw_skill_updates)
                if isinstance(raw_skill_updates, Sequence)
                and not isinstance(raw_skill_updates, (str, bytes))
                else []
            )
            persisted_skill_update = next(
                (
                    _json_ready(update)
                    for update in skill_updates
                    if isinstance(update, Mapping) and update.get("skill") == skill
                ),
                None,
            )

            observation_records: list[dict[str, object]] = []
            for event in target_events:
                transition: dict[str, float] | None = None
                if _bkt_observation(event) is not None:
                    transition = {
                        "mastery_before_observation": recomputed_before,
                        "mastery_after_observation": recomputed_after,
                        "delta_mastery": recomputed_after - recomputed_before,
                    }
                observation_records.append(
                    {
                        "dialogue_evidence": _json_ready(dialogue_evidence),
                        "resolved_signal": event,
                        "mastery_transition": transition,
                    }
                )

            record: dict[str, object] = {
                "schema_version": BKT_ACTIVITY_SCHEMA_VERSION,
                "runtime_id": self.runtime_id,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "event": "dialogue_turn_bkt_update",
                "student_id": canonical_student_id,
                "attempt_id": canonical_attempt_id,
                "session_id": knowledge_result.get(
                    "session_id",
                    f"{canonical_attempt_id}:dialogue:{raw_turn_index}",
                ),
                "target_skill": skill,
                "turn_index": raw_turn_index,
                "update_timing": "after_student_dialogue_turn",
                "dialogue": _json_ready(dialogue_evidence),
                "bkt_parameters": parameters,
                "effective_initial_prior": {
                    "probability": effective_prior,
                    "source": prior_record.get("prior_source"),
                },
                "history": {
                    "observation_count_before_dialogue_turn": len(history_prefix),
                    "observations_applied_in_dialogue_turn": new_count,
                    "observation_count_after_dialogue_turn": len(all_attempts),
                    "persisted_suffix_matches_resolved_updates": suffix_matches,
                },
                "mastery": {
                    "before": mastery_before,
                    "after": mastery_after,
                    "delta": mastery_delta,
                    "recomputed_before": recomputed_before,
                    "recomputed_after": recomputed_after,
                    "before_consistent": isclose(
                        mastery_before,
                        recomputed_before,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                    "after_consistent": isclose(
                        mastery_after,
                        recomputed_after,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                },
                "observations": observation_records,
                "persisted_skill_update": persisted_skill_update,
            }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            from src.self_improvement.postgres_repository import (
                append_experience,
                postgres_enabled,
            )
            if postgres_enabled():
                append_experience(
                    record,
                    event_id_override=(
                        f"bkt-dialogue:{canonical_attempt_id}:{record['turn_index']}"
                    ),
                )
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with self.bkt_activity_log_path.open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._recorded_dialogue_turns.add(activity_key)
            self._bkt_activity_count += 1
            self.assert_safety()
            self._write_summary()
            return record

    def record_attempt_completion(
        self,
        *,
        attempt_id: str,
        turn_count: int,
        mastery_before: float,
        mastery_after: float,
    ) -> None:
        """Record an abort-only policy completion after BKT has finished."""

        with self._lock:
            self.assert_safety()
            self._completed_attempts.append(
                {
                    "attempt_id": attempt_id,
                    "turn_count": int(turn_count),
                    "mastery_before": float(mastery_before),
                    "mastery_after": float(mastery_after),
                    "policy_completion": "aborted_without_update",
                }
            )
            self._write_summary()


__all__ = (
    "BKT_ACTIVITY_SCHEMA_VERSION",
    "MANUAL_CONTROLLED_LIVE_ENV",
    "MANUAL_CONTROLLED_LIVE_VALUE",
    "ManualControlledLiveDiagnostics",
    "file_snapshot",
    "manual_controlled_live_enabled",
    "posterior_snapshot",
)
