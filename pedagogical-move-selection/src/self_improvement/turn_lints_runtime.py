"""Runtime lifecycle for the fresh direct-arm, immediate-credit Turn-LinTS."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

from .context_builder import MOVE_ORDER
from .explicit_learner_agency import (
    AGENCY_ASSIGNMENT_SOURCE,
    AGENCY_BEHAVIOR_POLICY,
    decide_explicit_learner_agency,
)
from .randomized_warmstart import (
    BEHAVIOR_POLICY_ID,
    MD7ProportionalAssigner,
    WARMSTART_LINEAGE_SCHEMA,
)
from .turn_context_builder import MRB1_TASKS, validate_mrb1_scores
from .turn_lints_context import TurnContext, build_turn_lints_context
from .turn_lints_policy import DirectTurnLinTS, save_turn_lints_state
from .turn_lints_reward import turn_rewards
from .postgres_repository import (
    add_response_quality as postgres_add_response_quality,
    complete_action as postgres_complete_action,
    policy_key_for_path,
    postgres_enabled,
    stage_action as postgres_stage_action,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TurnLinTSMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    RANDOMIZED_WARMSTART = "RANDOMIZED_WARMSTART"
    LIVE = "LIVE"

    @classmethod
    def parse(cls, value: object) -> "TurnLinTSMode":
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(
                "Turn-LinTS mode must be OFF, SHADOW, RANDOMIZED_WARMSTART, or LIVE."
            ) from exc


@runtime_checkable
class Selector(Protocol):
    def predict_probabilities(self, problem: str,
                              conversation_history: Sequence[Mapping[str, object]]) -> Mapping[str, float]: ...


@runtime_checkable
class Critic(Protocol):
    def score_response(self, conversation_history: Sequence[Mapping[str, object]],
                       tutor_response: str) -> Mapping[str, float]: ...


@runtime_checkable
class TutorAgent(Protocol):
    def generate(self, *, problem: str,
                 conversation_history: Sequence[Mapping[str, object]],
                 pedagogical_move: str) -> str: ...


@dataclass(frozen=True, slots=True)
class DirectTurnDecision:
    action_event_id: str
    action_turn_index: int
    turn_index: int
    context: tuple[float, ...]
    context_snapshot: Mapping[str, object]
    diagnostic_context_snapshot: Mapping[str, object]
    selector_probabilities: Mapping[str, float]
    selected_arm: str | None
    hypothetical_arm: str | None
    sampled_scores: Mapping[str, float]
    contextual_ts_reward_scores: Mapping[str, float]
    anchor_scores: Mapping[str, float]
    policy_scores: Mapping[str, float]
    shadow_p0_move: str | None
    shadow_p1_move: str | None
    base_move: str
    final_move: str
    policy_mode: str
    behavior_policy: str
    behavior_propensity: float | None
    full_behavior_probability_vector: Mapping[str, float]
    randomized_assignment: bool
    treatment_assignment_source: str
    assignment_rng_source: str | None
    assignment_seed: int | None
    assignment_random_draw: float | None
    explicit_agency_request_category: str | None


@dataclass(frozen=True, slots=True)
class DirectCompletedTurn:
    decision: DirectTurnDecision
    mrb1_scores: Mapping[str, float]


class TurnActionLedger:
    """Derived action/outcome log; authoritative student-model JSONLs stay untouched."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending_actions"
        self.events = self.root / "turn_events.jsonl"
        if not postgres_enabled():
            self.pending.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, action_id: str) -> Path:
        name = hashlib.sha256(action_id.encode("utf-8")).hexdigest() + ".json"
        return self.pending / name

    def stage(self, record: Mapping[str, object]) -> None:
        action_id = str(record.get("action_event_id", "")).strip()
        if not action_id:
            raise ValueError("Action ledger record needs action_event_id.")
        if postgres_enabled():
            postgres_stage_action(record)
            return
        path = self._path(action_id)
        serialized = json.dumps(dict(record), ensure_ascii=False, allow_nan=False,
                                sort_keys=True, indent=2) + "\n"
        with self._lock:
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != dict(record):
                    raise RuntimeError("Action identity was replayed with different context.")
                return
            path.write_text(serialized, encoding="utf-8", newline="\n")

    def add_response_quality(
        self,
        action_id: str,
        scores: Mapping[str, float],
        tutor_response: str | None = None,
    ) -> None:
        if postgres_enabled():
            postgres_add_response_quality(action_id, scores, tutor_response)
            return
        with self._lock:
            path = self._path(action_id)
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = record.get("current_response_mrb1")
            payload = dict(scores)
            if expected is not None and expected != payload:
                raise RuntimeError("MRB1 retry differs for one action identity.")
            record["current_response_mrb1"] = payload
            if tutor_response is not None:
                response = str(tutor_response).strip()
                if not response:
                    raise ValueError("Tutor response provenance cannot be empty.")
                existing_response = record.get("tutor_response")
                if existing_response is not None and existing_response != response:
                    raise RuntimeError("Tutor response retry differs for one action identity.")
                record["tutor_response"] = response
            path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                       sort_keys=True, indent=2) + "\n",
                            encoding="utf-8", newline="\n")

    def complete(
        self,
        action_id: str,
        outcome: Mapping[str, object],
        *,
        policy: DirectTurnLinTS | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        if postgres_enabled():
            postgres_complete_action(
                action_id=action_id,
                outcome=outcome,
                policy_key=(
                    None if state_path is None else policy_key_for_path(state_path)
                ),
                policy_payload=(None if policy is None else policy.state_dict()),
                policy_class=(
                    "DirectTurnLinTS" if policy is None else type(policy).__name__
                ),
            )
            return
        with self._lock:
            path = self._path(action_id)
            action = json.loads(path.read_text(encoding="utf-8"))
            payload = {**action, **dict(outcome)}
            canonical = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                   sort_keys=True, separators=(",", ":"))
            if self.events.exists():
                for line in self.events.read_text(encoding="utf-8").splitlines():
                    existing = json.loads(line)
                    if existing.get("action_event_id") == action_id:
                        if json.dumps(existing, ensure_ascii=False, allow_nan=False,
                                      sort_keys=True, separators=(",", ":")) != canonical:
                            raise RuntimeError("Completed action retry differs from ledger.")
                        path.unlink(missing_ok=True)
                        return
            self.events.parent.mkdir(parents=True, exist_ok=True)
            with self.events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            path.unlink(missing_ok=True)


class DirectTurnController:
    def __init__(self, *, policy: DirectTurnLinTS, mode: TurnLinTSMode,
                 enabled_context_blocks: Sequence[str], state_path: str | Path,
                 ledger: TurnActionLedger,
                 warmstart_assigner: MD7ProportionalAssigner | None = None) -> None:
        self.policy = policy
        self.mode = mode
        self.enabled_context_blocks = tuple(enabled_context_blocks)
        self.state_path = Path(state_path)
        self.ledger = ledger
        self.warmstart_assigner = warmstart_assigner or MD7ProportionalAssigner()
        if self.mode is TurnLinTSMode.RANDOMIZED_WARMSTART:
            manifest = {
                "schema_version": WARMSTART_LINEAGE_SCHEMA,
                "mode": TurnLinTSMode.RANDOMIZED_WARMSTART.value,
                "behavior_policy": BEHAVIOR_POLICY_ID,
                "selector_version": self.policy.selector_version,
                "selector_sha256": self.policy.selector_sha256,
                "context_schema_id": self.policy.context_schema_id,
                "context_blocks": list(self.enabled_context_blocks),
                "context_dimension": self.policy.context_dim,
                "reward_mode": self.policy.reward_mode,
                "assignment_rng_source": self.warmstart_assigner.rng_source,
                "assignment_seed": self.warmstart_assigner.seed,
                "posterior_updates_enabled": False,
                "event_schema_version": "adaptmath_turn_lints_event_v4",
                "state_path": str(self.state_path),
            }
            if self.enabled_context_blocks == ("S", "K", "L"):
                manifest["diagnostic_context_blocks"] = ["S", "K", "L", "H", "Q"]
                manifest["diagnostic_context_dimension"] = 22
            if not postgres_enabled():
                manifest_path = self.ledger.root / "lineage_manifest.json"
                if manifest_path.exists():
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if existing != manifest:
                        raise RuntimeError("Randomized warm-start lineage mismatch.")
                else:
                    manifest_path.write_text(
                        json.dumps(
                            manifest,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
        self._attempt_id: str | None = None
        self.completed_turns: list[DirectCompletedTurn] = []
        self._pending: DirectTurnDecision | None = None
        self._previous_move: str | None = None
        self._previous_mrb1: dict[str, float] | None = None
        self._completed_outcomes: dict[str, dict[str, object]] = {}
        self._processing_failures: set[str] = set()

    @property
    def is_active(self) -> bool:
        return self._attempt_id is not None

    def start_attempt(self, _mastery_before: object, *, attempt_id: object) -> None:
        if self.is_active:
            raise RuntimeError("A Turn-LinTS attempt is already active.")
        identity = str(attempt_id).strip()
        if not identity:
            raise ValueError("attempt_id must be non-empty.")
        self._attempt_id = identity
        self.completed_turns = []
        self._pending = None
        self._previous_move = None
        self._previous_mrb1 = None
        self._completed_outcomes = {}
        self._processing_failures = set()

    def select_turn(
        self,
        probabilities: Mapping[str, float],
        pre_action: Mapping[str, object],
        *,
        externally_forced_move: str | None = None,
        external_assignment_source: str | None = None,
        external_assignment_category: str | None = None,
    ) -> DirectTurnDecision:
        if self._attempt_id is None or self._pending is not None:
            raise RuntimeError("Turn-LinTS selection lifecycle is invalid.")
        turn_index = len(self.completed_turns) + 1
        context: TurnContext = build_turn_lints_context(
            enabled_blocks=self.enabled_context_blocks,
            selector_probabilities=probabilities,
            mastery_before=pre_action["mastery_before"],
            previous_mastery_delta=pre_action.get("previous_mastery_delta"),
            previous_learner_signals=pre_action.get("previous_learner_signals"),
            previous_move=self._previous_move,
            prior_tutor_turn_count=turn_index - 1,
            previous_mrb1_scores=self._previous_mrb1,
        )
        diagnostic_context: TurnContext = build_turn_lints_context(
            enabled_blocks=("S", "K", "L", "H", "Q"),
            selector_probabilities=probabilities,
            mastery_before=pre_action["mastery_before"],
            previous_mastery_delta=pre_action.get("previous_mastery_delta"),
            previous_learner_signals=pre_action.get("previous_learner_signals"),
            previous_move=self._previous_move,
            prior_tutor_turn_count=turn_index - 1,
            previous_mrb1_scores=self._previous_mrb1,
        )
        base_move = max(MOVE_ORDER, key=lambda move: probabilities[move])
        sampled_scores: Mapping[str, float] = {}
        contextual_ts_reward_scores: Mapping[str, float] = {}
        anchor_scores: Mapping[str, float] = {}
        policy_scores: Mapping[str, float] = {}
        selected: str | None = None
        hypothetical: str | None = None
        shadow_p0_move: str | None = None
        shadow_p1_move: str | None = None
        behavior_policy = "frozen_md7_top1_v1"
        behavior_propensity: float | None = None
        randomized_assignment = False
        treatment_assignment_source = "frozen_md7_top1"
        assignment_rng_source: str | None = None
        assignment_seed: int | None = None
        assignment_random_draw: float | None = None
        explicit_agency_request_category: str | None = None
        externally_forced_final_move: str | None = None
        full_behavior_probability_vector = dict(probabilities)
        if externally_forced_move is not None:
            if externally_forced_move not in MOVE_ORDER:
                raise ValueError("Externally forced move is not a direct Tutor move.")
            source = str(external_assignment_source or "").strip()
            if not source:
                raise ValueError("Externally forced action requires its assignment source.")
            # Learner agency sits above policy assignment.  It is deliberately
            # not recorded as a LinTS-selected arm and can never update the
            # LinTS posterior.
            externally_forced_final_move = externally_forced_move
            behavior_policy = (
                AGENCY_BEHAVIOR_POLICY
                if source == AGENCY_ASSIGNMENT_SOURCE
                else "external_override"
            )
            treatment_assignment_source = source
            if behavior_policy == AGENCY_BEHAVIOR_POLICY:
                category = str(external_assignment_category or "").strip()
                if not category:
                    raise ValueError("Learner-agency override requires a request category.")
                explicit_agency_request_category = category
        elif self.mode is TurnLinTSMode.RANDOMIZED_WARMSTART:
            assignment = self.warmstart_assigner.assign(probabilities)
            selected = assignment.selected_move
            behavior_policy = assignment.behavior_policy
            behavior_propensity = assignment.behavior_propensity
            full_behavior_probability_vector = dict(
                assignment.full_probability_vector
            )
            randomized_assignment = assignment.randomized_assignment
            treatment_assignment_source = assignment.treatment_assignment_source
            assignment_rng_source = assignment.assignment_rng_source
            assignment_seed = assignment.assignment_seed
            assignment_random_draw = assignment.assignment_random_draw
        elif self.mode is not TurnLinTSMode.OFF:
            sample = self.policy.select_arm(
                context.vector(), selector_probabilities=probabilities
            )
            sampled_move = str(sample["selected_arm"])
            sampled_scores = dict(sample["sampled_scores"])  # type: ignore[arg-type]
            contextual_ts_reward_scores = dict(  # type: ignore[arg-type]
                sample["contextual_ts_reward_scores"]
            )
            anchor_scores = dict(sample["anchor_scores"])  # type: ignore[arg-type]
            policy_scores = dict(sample["policy_scores"])  # type: ignore[arg-type]
            if self.mode is TurnLinTSMode.SHADOW:
                hypothetical = sampled_move
                shadow_p0_move = str(sample["p0_selected_arm"])
                if sample["p1_selected_arm"] is not None:
                    shadow_p1_move = str(sample["p1_selected_arm"])
            else:
                selected = sampled_move
                behavior_policy = "turn_lints_live_v1"
                treatment_assignment_source = "turn_lints_posterior_sample"
        final_move = (
            externally_forced_final_move
            if externally_forced_final_move is not None
            else selected if selected is not None else base_move
        )
        action_id = f"{self._attempt_id}:action:{turn_index}"
        decision = DirectTurnDecision(
            action_event_id=action_id, action_turn_index=turn_index,
            turn_index=turn_index, context=context.values,
            context_snapshot=context.as_mapping(),
            diagnostic_context_snapshot=diagnostic_context.as_mapping(),
            selector_probabilities=dict(probabilities), selected_arm=selected,
            hypothetical_arm=hypothetical, sampled_scores=sampled_scores,
            contextual_ts_reward_scores=contextual_ts_reward_scores,
            anchor_scores=anchor_scores, policy_scores=policy_scores,
            shadow_p0_move=shadow_p0_move, shadow_p1_move=shadow_p1_move,
            base_move=base_move, final_move=final_move, policy_mode=self.mode.value,
            behavior_policy=behavior_policy,
            behavior_propensity=behavior_propensity,
            full_behavior_probability_vector=full_behavior_probability_vector,
            randomized_assignment=randomized_assignment,
            treatment_assignment_source=treatment_assignment_source,
            assignment_rng_source=assignment_rng_source,
            assignment_seed=assignment_seed,
            assignment_random_draw=assignment_random_draw,
            explicit_agency_request_category=explicit_agency_request_category,
        )
        self.ledger.stage({
            "schema_version": "adaptmath_turn_lints_event_v4",
            "attempt_id": self._attempt_id,
            "action_event_id": action_id,
            "action_turn_index": turn_index,
            "policy_lineage": "direct_disjoint_turn_lints_v1",
            "collection_lineage": self.ledger.root.name,
            "policy_mode": self.mode.value,
            "selector_version": self.policy.selector_version,
            "selector_sha256": self.policy.selector_sha256,
            "context": context.as_mapping(),
            "policy_context": context.as_mapping(),
            "diagnostic_logged_context": diagnostic_context.as_mapping(),
            "md7_raw_probabilities": dict(probabilities),
            "selected_arm": selected,
            "hypothetical_arm": hypothetical,
            "actual_move": final_move,
            "actual_selector_move": base_move,
            "actual_treatment_move": final_move,
            "decision_source": treatment_assignment_source,
            "treatment_assignment_source": treatment_assignment_source,
            "behavior_policy": behavior_policy,
            "behavior_propensity": behavior_propensity,
            "full_behavior_probability_vector": dict(
                full_behavior_probability_vector
            ),
            "randomized_assignment": randomized_assignment,
            "assignment_rng_source": assignment_rng_source,
            "assignment_seed": assignment_seed,
            "assignment_random_draw": assignment_random_draw,
            "explicit_learner_agency": {
                "triggered": explicit_agency_request_category is not None,
                "request_category": explicit_agency_request_category,
                "inferred_learner_state_used": False,
            },
            "shadow_hypothetical_move": hypothetical,
            "shadow_p0_move": shadow_p0_move,
            "shadow_p1_move": shadow_p1_move,
            "shadow_moves_are_observed_treatments": False,
            "hypothetical_is_observed_treatment": False,
            "contextual_ts_reward_scores": dict(contextual_ts_reward_scores),
            "anchor_scores": dict(anchor_scores),
            "policy_scores": dict(policy_scores),
            "anchor_mode": self.policy.anchor_mode,
            "anchor_gamma": self.policy.anchor_gamma,
            "anchor_selector_version": self.policy.anchor_selector_version,
            "anchor_selector_sha256": self.policy.anchor_selector_sha256,
            "reward_attribution": "actual_treatment_move",
            "reward_mode": self.policy.reward_mode,
            "configured_reward_mode": self.policy.reward_mode,
            "observation_origin": (
                "live_tutor" if self.mode is TurnLinTSMode.LIVE else "runtime_tutor"
            ),
            "synthetic_initialization": False,
            "real_live_observation": self.mode is TurnLinTSMode.LIVE,
            "posterior_initialization_observation": False,
        })
        self._pending = decision
        return decision

    def record_mrb1_scores(
        self,
        scores: Mapping[str, float],
        tutor_response: str | None = None,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("No selected action is awaiting MRB1.")
        validated = validate_mrb1_scores(scores)
        self.ledger.add_response_quality(
            self._pending.action_event_id,
            validated,
            tutor_response,
        )
        self.completed_turns.append(DirectCompletedTurn(self._pending, validated))
        self._previous_move = self._pending.final_move
        self._previous_mrb1 = validated
        self._pending = None

    def complete_outcome(self, *, action_event_id: str, should_update: bool,
                         mastery_before: object | None,
                         mastery_after: object | None,
                         resolver_event_id: object | None,
                         previous_learner_signals: Mapping[str, object] | None) -> dict[str, object]:
        replay_input = {
            "should_update": should_update,
            "mastery_before": mastery_before,
            "mastery_after": mastery_after,
            "resolver_event_id": resolver_event_id,
            "previous_learner_signals": (
                None if previous_learner_signals is None
                else dict(previous_learner_signals)
            ),
        }
        prior = self._completed_outcomes.get(action_event_id)
        if prior is not None:
            if prior["replay_input"] != replay_input:
                raise RuntimeError("Turn outcome retry differs for one action identity.")
            return dict(prior["outcome"])  # type: ignore[arg-type]
        matches = [turn for turn in self.completed_turns
                   if turn.decision.action_event_id == action_event_id]
        if len(matches) != 1:
            raise RuntimeError("BKT outcome has no unique originating action.")
        decision = matches[0].decision
        rewards: dict[str, float] | None = None
        updated = False
        update_id: str | None = None
        if should_update:
            if mastery_before is None or mastery_after is None:
                raise ValueError("Observed BKT update lacks mastery values.")
            rewards = turn_rewards(mastery_before, mastery_after)
            if (
                self.mode is TurnLinTSMode.LIVE
                and decision.explicit_agency_request_category is None
            ):
                if decision.selected_arm is None:
                    raise RuntimeError("LIVE policy action has no selected direct arm.")
                update_id = action_event_id
                updated = self.policy.update_once(
                    update_id=update_id, arm=decision.selected_arm,
                    context=decision.context,
                    reward=rewards[self.policy.reward_mode],
                    weight=1.0,
                )
                # The idempotency identity is part of the atomically replaced state.
                if not postgres_enabled():
                    save_turn_lints_state(self.policy, self.state_path)
        outcome = {
            "resolver_event_id": resolver_event_id,
            "mastery_before": mastery_before,
            "mastery_after": mastery_after,
            "raw_delta": None if rewards is None else rewards["raw_delta"],
            "headroom_normalized_delta": (
                None if rewards is None else rewards["headroom_normalized"]
            ),
            "configured_reward_mode": self.policy.reward_mode,
            "configured_reward_value": (
                None if rewards is None else rewards[self.policy.reward_mode]
            ),
            "reward_attributed_to_move": decision.final_move,
            "reward_attributed_to_shadow_hypothetical": False,
            "learner_signals_for_next_action": (
                None if previous_learner_signals is None
                else dict(previous_learner_signals)
            ),
            "posterior_update_occurred": updated,
            "posterior_updated": updated,
            "posterior_update_origin": "real_live" if updated else None,
            "posterior_update_weight": 1.0 if updated else None,
            "agency_forced_no_policy_update": bool(
                decision.explicit_agency_request_category is not None
            ),
            "update_id": update_id,
            "update_timestamp": _utc_now() if updated else None,
            "outcome_observed": bool(should_update),
            "resolution_status": (
                "observed_bkt_update" if should_update else "no_bkt_update"
            ),
        }
        self.ledger.complete(
            action_event_id,
            outcome,
            policy=self.policy if updated else None,
            state_path=self.state_path if updated else None,
        )
        self._completed_outcomes[action_event_id] = {
            "replay_input": replay_input,
            "outcome": dict(outcome),
        }
        return outcome

    def finish_attempt(self) -> int:
        if self._pending is not None:
            raise RuntimeError("Cannot finish while a Tutor response is pending.")
        count = len(self.completed_turns)
        self._attempt_id = None
        return count

    def note_processing_failure(self, action_event_id: str) -> None:
        if not any(turn.decision.action_event_id == action_event_id
                   for turn in self.completed_turns):
            raise RuntimeError("Processing failure has no selected Tutor action.")
        self._processing_failures.add(action_event_id)

    def abort_attempt(self) -> None:
        pending_ids = {
            turn.decision.action_event_id for turn in self.completed_turns
        }
        if self._pending is not None:
            pending_ids.add(self._pending.action_event_id)
        for action_id in sorted(pending_ids - set(self._completed_outcomes)):
            self.ledger.complete(
                action_id,
                {
                    "resolver_event_id": None,
                    "mastery_before": None,
                    "mastery_after": None,
                    "raw_delta": None,
                    "headroom_normalized_delta": None,
                    "configured_reward_mode": self.policy.reward_mode,
                    "configured_reward_value": None,
                    "reward_attributed_to_move": next(
                        turn.decision.final_move
                        for turn in self.completed_turns
                        if turn.decision.action_event_id == action_id
                    ) if any(
                        turn.decision.action_event_id == action_id
                        for turn in self.completed_turns
                    ) else None,
                    "reward_attributed_to_shadow_hypothetical": False,
                    "learner_signals_for_next_action": None,
                    "posterior_update_occurred": False,
                    "posterior_updated": False,
                    "update_id": None,
                    "update_timestamp": None,
                    "outcome_observed": False,
                    "resolution_status": (
                        "processing_failed"
                        if action_id in self._processing_failures
                        else "censored_no_response"
                    ),
                    "outcome_status": (
                        "processing_failed"
                        if action_id in self._processing_failures
                        else "censored_no_response"
                    ),
                },
            )
        self._attempt_id = None
        self._pending = None


class DirectTurnLinTSPipeline:
    """Selector/Tutor/MRB1 pipeline with immediate updates supplied separately."""

    def __init__(self, *, selector: Selector, mrb1: Critic,
                 controller: DirectTurnController, memory_adapter: object) -> None:
        self.selector, self.mrb1 = selector, mrb1
        self.controller, self.memory_adapter = controller, memory_adapter
        self._active_attempt_id: str | None = None
        self._pending_decision: DirectTurnDecision | None = None
        self._pending_response: str | None = None
        self._pending_scores: dict[str, float] | None = None
        self._pending_scores_recorded = False

    @property
    def active_attempt_id(self) -> str | None:
        return self._active_attempt_id

    def start_attempt(self, memory: object) -> dict[str, object]:
        attempt_id = str(self.memory_adapter.get_attempt_id(memory))
        mastery = self.memory_adapter.get_mastery_before(memory)
        self.controller.start_attempt(mastery, attempt_id=attempt_id)
        self._active_attempt_id = attempt_id
        self._pending_decision = self._pending_response = None
        self._pending_scores = None
        self._pending_scores_recorded = False
        return {"attempt_id": attempt_id, "mastery_before": float(mastery)}

    def restore_attempt(self, memory: object,
                        completed_turns: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if completed_turns:
            raise RuntimeError("Direct Turn-LinTS restart restoration is not yet required outside controlled legacy mode.")
        return self.start_attempt(memory)

    def run_tutor_turn(self, memory: object, tutor_agent: TutorAgent) -> dict[str, object]:
        attempt_id = str(self.memory_adapter.get_attempt_id(memory))
        if attempt_id != self._active_attempt_id:
            raise ValueError("Tutor memory differs from active Turn-LinTS attempt.")
        problem = self.memory_adapter.get_problem(memory)
        history = copy.deepcopy(list(self.memory_adapter.get_conversation_history(memory)))
        decision = self._pending_decision
        if decision is None:
            probabilities = dict(self.selector.predict_probabilities(problem, history))
            getter = getattr(self.memory_adapter, "get_turn_lints_pre_action_state", None)
            if not callable(getter):
                raise TypeError("Memory adapter lacks Turn-LinTS pre-action state.")
            agency = decide_explicit_learner_agency(history)
            decision = self.controller.select_turn(
                probabilities,
                getter(memory),
                externally_forced_move=agency.requested_move,
                external_assignment_source=agency.assignment_source,
                external_assignment_category=agency.request_category,
            )
            self._pending_decision = decision
        response = self._pending_response
        if response is None:
            response = tutor_agent.generate(
                problem=problem, conversation_history=copy.deepcopy(history),
                pedagogical_move=decision.final_move,
            )
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Tutor response must be non-empty.")
            response = response.strip()
            self._pending_response = response
        scores = self._pending_scores
        if scores is None:
            scores = dict(self.mrb1.score_response(history, response))
            self._pending_scores = dict(scores)
        if not self._pending_scores_recorded:
            self.controller.record_mrb1_scores(scores, response)
            self._pending_scores_recorded = True
        result = {
            "attempt_id": attempt_id,
            "action_event_id": decision.action_event_id,
            "action_turn_index": decision.action_turn_index,
            "turn_index": decision.turn_index,
            "tutor_response": response,
            "pedagogical_move": decision.final_move,
            "selected_arm": decision.selected_arm,
            "base_move": decision.base_move,
            "overridden": decision.final_move != decision.base_move,
            "context": list(decision.context),
            # Compatibility alias for the immutable legacy turn collector.
            "md6_probabilities": dict(decision.selector_probabilities),
            "md7_probabilities": dict(decision.selector_probabilities),
            "mrb1_scores": scores,
            "selector": {"raw": {"probabilities": dict(decision.selector_probabilities),
                                    "argmax": decision.base_move},
                         "effective": {
                             "probabilities": (
                                 {
                                     move: float(move == decision.final_move)
                                     for move in MOVE_ORDER
                                 }
                                 if decision.explicit_agency_request_category is not None
                                 else dict(decision.selector_probabilities)
                             ),
                             "argmax": (
                                 decision.final_move
                                 if decision.explicit_agency_request_category is not None
                                 else decision.base_move
                             ),
                         }},
            "adaptive_decision": {
                **dict(decision.context_snapshot),
                "policy_context": dict(decision.context_snapshot),
                "diagnostic_logged_context": dict(
                    decision.diagnostic_context_snapshot
                ),
                "policy_mode": decision.policy_mode,
                "available_arms": list(MOVE_ORDER),
                "sampled_scores": dict(decision.sampled_scores),
                "contextual_ts_reward_scores": dict(
                    decision.contextual_ts_reward_scores
                ),
                "anchor_scores": dict(decision.anchor_scores),
                "policy_scores": dict(decision.policy_scores),
                "selected_arm": decision.selected_arm,
                "hypothetical_arm": decision.hypothetical_arm,
                "shadow_p0_move": decision.shadow_p0_move,
                "shadow_p1_move": decision.shadow_p1_move,
                "base_move": decision.base_move,
                "final_move": decision.final_move,
                "behavior_policy": decision.behavior_policy,
                "behavior_propensity": decision.behavior_propensity,
                "full_behavior_probability_vector": dict(
                    decision.full_behavior_probability_vector
                ),
                "randomized_assignment": decision.randomized_assignment,
                "treatment_assignment_source": (
                    decision.treatment_assignment_source
                ),
                "explicit_learner_agency": {
                    "triggered": decision.explicit_agency_request_category is not None,
                    "request_category": decision.explicit_agency_request_category,
                    "inferred_learner_state_used": False,
                },
                "reward_mode": self.controller.policy.reward_mode,
                "anchor_mode": self.controller.policy.anchor_mode,
                "anchor_gamma": self.controller.policy.anchor_gamma,
            },
        }
        self.memory_adapter.append_tutor_response(memory, response)
        self._pending_decision = self._pending_response = None
        self._pending_scores = None
        self._pending_scores_recorded = False
        return result

    def complete_turn_outcome(self, **kwargs: object) -> dict[str, object]:
        return self.controller.complete_outcome(**kwargs)  # type: ignore[arg-type]

    def note_turn_processing_failure(self, action_event_id: str) -> None:
        self.controller.note_processing_failure(action_event_id)

    def finish_attempt(self, memory: object, learning_outcome: Mapping[str, object],
                       **_kwargs: object) -> dict[str, object]:
        attempt_id = str(self.memory_adapter.get_attempt_id(memory))
        if attempt_id != self._active_attempt_id:
            raise ValueError("Attempt completion identity mismatch.")
        count = self.controller.finish_attempt()
        self._active_attempt_id = None
        return {
            "attempt_id": attempt_id,
            "skill": learning_outcome["skill"],
            "mastery_before": float(learning_outcome["mastery_before"]),
            "mastery_after": float(learning_outcome["mastery_after"]),
            "mastery_delta": float(learning_outcome["delta_mastery"]),
            "turn_count": count,
            "policy_completion": "formal_assessment_not_credited_to_actions",
            "turn_lints_update_count": self.controller.policy.total_updates,
        }

    def abort_attempt(self, memory: object | None = None) -> None:
        self.controller.abort_attempt()
        self._active_attempt_id = None
        self._pending_decision = self._pending_response = None
        self._pending_scores = None
        self._pending_scores_recorded = False


__all__ = (
    "DirectCompletedTurn", "DirectTurnController", "DirectTurnDecision",
    "DirectTurnLinTSPipeline", "TurnActionLedger", "TurnLinTSMode",
)
