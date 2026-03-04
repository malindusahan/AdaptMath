"""Canonical candidate turn-level controller with delayed terminal credit.

Each tutor turn builds a fresh nine-dimensional C3 context, computes the overlay
arms currently capable of changing frozen MD6, and performs a fresh true-LinTS
posterior draw. The controller never updates LinTS during an active attempt.

After the attempt completes, its validated signed mastery-delta reward is
distributed over the saved turn decisions with normalized equal weight
``1 / T``. Thus an
attempt with four turns contributes four updates of weight ``0.25``, not four
independent full rewards. Equal credit is a simple candidate assumption, not a
scientifically established optimum; later work may compare recency or other
defensible delayed-credit schemes.

The policy's ``total_updates`` and arm counts continue to count low-level
weighted update calls for backwards compatibility. This controller separately
counts completed attempts and weighted turn updates. It performs no logging,
persistence, model inference, simulation, or scientific ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Real
from types import MappingProxyType
from uuid import uuid4

from .conservative_overlay import (
    DEFAULT_GAP_THRESHOLD,
    apply_conservative_overlay,
    eligible_arms,
)
from .context_builder import MOVE_ORDER
from .lints_policy import ARMS, TrueDisjointLinTS
from .reward import validate_signed_reward
from .turn_context_builder import (
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
    RunningTutorQuality,
    TutorQualitySnapshot,
    build_turn_context,
    validate_mastery_probability,
    validate_mrb1_scores,
)


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Immutable record returned when one tutor turn is selected."""

    action_event_id: str
    action_turn_index: int
    turn_index: int
    context: tuple[float, ...]
    md6_probabilities: Mapping[str, float]
    eligible_arms: tuple[str, ...]
    sampled_scores: Mapping[str, float]
    selected_arm: str
    base_move: str
    final_move: str
    target_move: str | None
    overridden: bool
    gap: float
    gap_threshold: float
    base_probability: float
    target_probability: float | None


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    """One immutable turn decision paired with its completed MRB1 scores."""

    decision: TurnDecision
    mrb1_scores: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class AttemptCompletion:
    """Immutable terminal result after all delayed updates have been applied."""

    reward: float
    mastery_before: float
    mastery_after: float
    mastery_delta: float
    final_quality: TutorQualitySnapshot
    turns: tuple[CompletedTurn, ...]
    turn_count: int
    sample_weight_per_turn: float
    total_attempt_weight: float
    completed_attempts: int
    number_of_weighted_turn_updates: int


def _validate_gap_threshold(value: object) -> float:
    """Return a strict finite threshold in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "gap_threshold must be a non-boolean real number, "
            f"got {value!r}."
        )
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "gap_threshold could not be represented as a float."
        ) from exc
    if not isfinite(result):
        raise ValueError(f"gap_threshold must be finite, got {result!r}.")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"gap_threshold must be in [0, 1], got {result}.")
    return result


def _immutable_float_mapping(
    values: Mapping[str, float],
    order: tuple[str, ...],
) -> Mapping[str, float]:
    """Copy ordered float values into an immutable mapping."""

    return MappingProxyType({key: float(values[key]) for key in order})


class TurnLevelAttemptController:
    """Coordinate turn selection and normalized terminal delayed updates."""

    def __init__(
        self,
        policy: TrueDisjointLinTS,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> None:
        if not isinstance(policy, TrueDisjointLinTS):
            raise TypeError("policy must be a TrueDisjointLinTS instance.")
        if policy.context_dim != len(TURN_FEATURE_NAMES):
            raise ValueError(
                "Turn-level policy context_dim must be "
                f"{len(TURN_FEATURE_NAMES)}, got {policy.context_dim}."
            )

        self.policy = policy
        self.gap_threshold = _validate_gap_threshold(gap_threshold)
        self.completed_attempts = 0
        self.number_of_weighted_turn_updates = 0

        self._active = False
        self._attempt_id: str | None = None
        self._mastery_before: float | None = None
        self._running_quality: RunningTutorQuality | None = None
        self._completed_turns: list[CompletedTurn] = []
        self._pending_turn: TurnDecision | None = None
        self._policy_updates_at_start: int | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def awaiting_mrb1(self) -> bool:
        return self._pending_turn is not None

    @property
    def completed_turns(self) -> tuple[CompletedTurn, ...]:
        """Return the immutable completed-turn history of the active attempt."""

        return tuple(self._completed_turns)

    @property
    def running_quality(self) -> TutorQualitySnapshot | None:
        """Return current active-attempt quality, or ``None`` when inactive."""

        if self._running_quality is None:
            return None
        return self._running_quality.snapshot()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("No turn-level tutoring attempt is active.")

    def _assert_no_within_attempt_updates(self) -> None:
        if self._policy_updates_at_start is None:
            raise RuntimeError("Active attempt is missing its policy update guard.")
        if self.policy.total_updates != self._policy_updates_at_start:
            raise RuntimeError(
                "LinTS policy was updated before the active attempt completed."
            )

    def _clear_active_attempt(self) -> None:
        self._active = False
        self._attempt_id = None
        self._mastery_before = None
        self._running_quality = None
        self._completed_turns = []
        self._pending_turn = None
        self._policy_updates_at_start = None

    def start_attempt(
        self,
        mastery_before: float,
        attempt_id: str | int | None = None,
    ) -> None:
        """Start an empty attempt without selecting or updating the policy."""

        if self._active:
            raise RuntimeError("Cannot start a second attempt while one is active.")

        mastery = validate_mastery_probability(mastery_before)
        if attempt_id is None:
            canonical_attempt_id = f"anonymous-{uuid4().hex}"
        elif isinstance(attempt_id, bool) or not isinstance(attempt_id, (str, int)):
            raise TypeError("attempt_id must be a string or integer when supplied.")
        else:
            canonical_attempt_id = str(attempt_id).strip()
            if not canonical_attempt_id:
                raise ValueError("attempt_id cannot be empty or whitespace.")
        self._active = True
        self._attempt_id = canonical_attempt_id
        self._mastery_before = mastery
        self._running_quality = RunningTutorQuality()
        self._completed_turns = []
        self._pending_turn = None
        self._policy_updates_at_start = self.policy.total_updates

    def restore_completed_turns(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> None:
        """Restore exact completed-turn records after a controlled restart.

        The caller must supply the original immutable decisions and MRB1 scores
        in consecutive action order. This method performs no Thompson draws and
        no policy updates; it only reconstructs the active attempt's causal
        running-quality state and action index.
        """

        self._require_active()
        if self._pending_turn is not None or self._completed_turns:
            raise RuntimeError("Completed turns can only restore into a fresh attempt.")
        if self._attempt_id is None or self._running_quality is None:
            raise RuntimeError("Active attempt state is incomplete.")

        restored: list[CompletedTurn] = []
        for expected_index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise TypeError("Each restored turn must be a mapping.")
            raw_decision = record.get("decision")
            raw_scores = record.get("mrb1_scores")
            if not isinstance(raw_decision, Mapping):
                raise TypeError("Restored turn decision must be a mapping.")
            if not isinstance(raw_scores, Mapping):
                raise TypeError("Restored turn MRB1 scores must be a mapping.")

            action_index = raw_decision.get("action_turn_index")
            action_event_id = raw_decision.get("action_event_id")
            expected_event_id = f"{self._attempt_id}:action:{expected_index}"
            if action_index != expected_index or action_event_id != expected_event_id:
                raise ValueError("Restored turn identities are not consecutive.")

            context = tuple(float(value) for value in raw_decision["context"])
            if len(context) != len(TURN_FEATURE_NAMES):
                raise ValueError("Restored turn context has the wrong dimension.")
            md6_probabilities = _immutable_float_mapping(
                raw_decision["md6_probabilities"],
                MOVE_ORDER,
            )
            sampled_scores = _immutable_float_mapping(
                raw_decision["sampled_scores"],
                ARMS,
            )
            validated_scores = validate_mrb1_scores(raw_scores)
            decision = TurnDecision(
                action_event_id=str(action_event_id),
                action_turn_index=expected_index,
                turn_index=expected_index,
                context=context,
                md6_probabilities=md6_probabilities,
                eligible_arms=tuple(str(value) for value in raw_decision["eligible_arms"]),
                sampled_scores=sampled_scores,
                selected_arm=str(raw_decision["selected_arm"]),
                base_move=str(raw_decision["base_move"]),
                final_move=str(raw_decision["final_move"]),
                target_move=(
                    None
                    if raw_decision.get("target_move") is None
                    else str(raw_decision["target_move"])
                ),
                overridden=bool(raw_decision["overridden"]),
                gap=float(raw_decision["gap"]),
                gap_threshold=float(raw_decision["gap_threshold"]),
                base_probability=float(raw_decision["base_probability"]),
                target_probability=(
                    None
                    if raw_decision.get("target_probability") is None
                    else float(raw_decision["target_probability"])
                ),
            )
            self._running_quality.add_scores(validated_scores)
            restored.append(
                CompletedTurn(
                    decision=decision,
                    mrb1_scores=_immutable_float_mapping(
                        validated_scores,
                        MRB1_TASKS,
                    ),
                )
            )

        self._completed_turns = restored
        self._assert_no_within_attempt_updates()

    def select_turn(
        self,
        md6_probabilities: Mapping[str, float],
    ) -> TurnDecision:
        """Perform one fresh eligible-arm Thompson decision for this turn."""

        self._require_active()
        if self._pending_turn is not None:
            raise RuntimeError(
                "The previous selected turn must receive MRB1 scores first."
            )
        self._assert_no_within_attempt_updates()

        if self._mastery_before is None or self._running_quality is None:
            raise RuntimeError("Active attempt state is incomplete.")
        if self._attempt_id is None:
            raise RuntimeError("Active attempt is missing its explicit identity.")

        context = build_turn_context(
            md6_probabilities=md6_probabilities,
            running_quality=self._running_quality,
        )
        canonical_probabilities = {
            move: float(context[index])
            for index, move in enumerate(MOVE_ORDER)
        }
        candidates = eligible_arms(
            canonical_probabilities,
            gap_threshold=self.gap_threshold,
        )

        selection = self.policy.select_arm(
            context,
            eligible_arms=candidates,
        )
        self._assert_no_within_attempt_updates()
        selected_arm = selection["selected_arm"]
        if selected_arm not in candidates:
            raise RuntimeError("LinTS selected an ineligible overlay arm.")

        overlay = apply_conservative_overlay(
            selected_arm,
            canonical_probabilities,
            gap_threshold=self.gap_threshold,
        )
        sampled_scores = _immutable_float_mapping(
            selection["sampled_scores"],
            ARMS,
        )

        action_turn_index = len(self._completed_turns) + 1
        decision = TurnDecision(
            action_event_id=(
                f"{self._attempt_id}:action:{action_turn_index}"
            ),
            action_turn_index=action_turn_index,
            # Backwards-compatible alias. Scientific joins use
            # action_event_id/action_turn_index, never a host thread counter.
            turn_index=action_turn_index,
            context=tuple(float(value) for value in context),
            md6_probabilities=_immutable_float_mapping(
                canonical_probabilities,
                MOVE_ORDER,
            ),
            eligible_arms=candidates,
            sampled_scores=sampled_scores,
            selected_arm=selected_arm,
            base_move=overlay.base_move,
            final_move=overlay.final_move,
            target_move=overlay.target_move,
            overridden=overlay.overridden,
            gap=overlay.gap,
            gap_threshold=overlay.gap_threshold,
            base_probability=overlay.base_probability,
            target_probability=overlay.target_probability,
        )
        self._pending_turn = decision
        return decision

    def record_mrb1_scores(
        self,
        scores: Mapping[str, float],
    ) -> CompletedTurn:
        """Close the pending turn and expose its quality to later turns."""

        self._require_active()
        if self._pending_turn is None:
            raise RuntimeError("No selected tutor turn is awaiting MRB1 scores.")
        self._assert_no_within_attempt_updates()
        if self._running_quality is None:
            raise RuntimeError("Active attempt is missing running MRB1 state.")

        validated = validate_mrb1_scores(scores)
        self._running_quality.add_scores(validated)
        completed = CompletedTurn(
            decision=self._pending_turn,
            mrb1_scores=_immutable_float_mapping(validated, MRB1_TASKS),
        )
        self._completed_turns.append(completed)
        self._pending_turn = None
        self._assert_no_within_attempt_updates()
        return completed

    def finish_attempt(
        self,
        *,
        reward: float,
        mastery_after: float,
    ) -> AttemptCompletion:
        """Apply normalized signed delayed credit and close the active attempt."""

        self._require_active()
        if self._pending_turn is not None:
            raise RuntimeError("Cannot finish while a turn awaits MRB1 scores.")
        if not self._completed_turns:
            raise RuntimeError("Cannot finish an attempt with zero completed turns.")
        self._assert_no_within_attempt_updates()

        validated_reward = validate_signed_reward(reward)
        validated_mastery_after = validate_mastery_probability(mastery_after)
        if self._mastery_before is None or self._running_quality is None:
            raise RuntimeError("Active attempt state is incomplete.")

        mastery_delta = float(validated_mastery_after - self._mastery_before)
        if not isclose(
            validated_reward,
            mastery_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "reward must equal mastery_after - mastery_before within "
                "abs_tol=1e-9."
            )

        turns = tuple(self._completed_turns)
        turn_count = len(turns)
        sample_weight = float(1.0 / turn_count)
        total_attempt_weight = float(fsum([sample_weight] * turn_count))
        if not isclose(
            total_attempt_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Normalized delayed-credit weights do not sum to 1.")

        for completed_turn in turns:
            self.policy.update(
                completed_turn.decision.selected_arm,
                completed_turn.decision.context,
                validated_reward,
                sample_weight=sample_weight,
            )

        expected_policy_updates = self._policy_updates_at_start + turn_count
        if self.policy.total_updates != expected_policy_updates:
            raise RuntimeError("Delayed weighted update count is inconsistent.")

        self.completed_attempts += 1
        self.number_of_weighted_turn_updates += turn_count
        mastery_before = self._mastery_before
        result = AttemptCompletion(
            reward=validated_reward,
            mastery_before=mastery_before,
            mastery_after=validated_mastery_after,
            mastery_delta=mastery_delta,
            final_quality=self._running_quality.snapshot(),
            turns=turns,
            turn_count=turn_count,
            sample_weight_per_turn=sample_weight,
            total_attempt_weight=total_attempt_weight,
            completed_attempts=self.completed_attempts,
            number_of_weighted_turn_updates=(
                self.number_of_weighted_turn_updates
            ),
        )
        self._clear_active_attempt()
        return result

    def abort_attempt(self) -> None:
        """Discard the active attempt without any LinTS posterior update.

        Turn selections may already have advanced the policy RNG, as required
        by fresh Thompson sampling. Aborting never changes learned ``A``/``b``
        posterior matrices, reward vectors, or update counters.
        """

        self._require_active()
        self._clear_active_attempt()


__all__ = (
    "AttemptCompletion",
    "CompletedTurn",
    "TurnDecision",
    "TurnLevelAttemptController",
)
