from __future__ import annotations

import copy
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_REPO = WORKSPACE_ROOT / "pedagogical-move-selection"
if str(MOVE_REPO) not in sys.path:
    sys.path.insert(0, str(MOVE_REPO))

from src.self_improvement.learner_agency_escalation import (  # noqa: E402
    LearnerAgencyEscalationSelector,
    TELLING_PROBABILITIES,
    decide_learner_agency_escalation,
    is_direct_answer_request,
    is_non_engagement,
)
from src.self_improvement.lints_policy import TrueDisjointLinTS  # noqa: E402
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


PROBING_PROBABILITIES = {
    "generic": 0.01,
    "probing": 0.97,
    "focus": 0.01,
    "telling": 0.01,
}


class FixtureSelector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict_probabilities(self, problem, conversation_history):
        self.calls.append((problem, copy.deepcopy(list(conversation_history))))
        return dict(PROBING_PROBABILITIES)


def _turn(role: str, text: str) -> dict[str, object]:
    return {"user": role, "text": text}


def test_direct_answer_request_is_detected_without_false_positive():
    assert is_direct_answer_request("just give me the answer")
    assert is_direct_answer_request("What's the answer for this problem?")
    assert is_direct_answer_request("Could you please solve it for me?")
    assert not is_direct_answer_request("I think the answer is 12.")


def test_non_engagement_is_narrow_and_content_free():
    assert is_non_engagement("no idea")
    assert is_non_engagement("I don't know")
    assert is_non_engagement("i cannot understand anything")
    assert not is_non_engagement("I don't know whether to add or multiply")
    assert not is_non_engagement("I do not understand why 6 is divided by 2")


def test_one_non_engaged_turn_does_not_force_telling():
    decision = decide_learner_agency_escalation(
        [_turn("student", "no idea")]
    )
    assert decision.triggered is False
    assert decision.consecutive_non_engagement_turns == 1


def test_two_latest_non_engaged_student_turns_force_telling():
    decision = decide_learner_agency_escalation(
        [
            _turn("student", "no idea"),
            _turn("teacher", "What part is unclear?"),
            _turn("student", "I don't understand anything"),
        ]
    )
    assert decision.triggered is True
    assert decision.reason == "repeated_non_engagement"
    assert decision.consecutive_non_engagement_turns == 2


def test_explicit_request_forces_telling_immediately():
    decision = decide_learner_agency_escalation(
        [_turn("student", "just give me the answer")]
    )
    assert decision.triggered is True
    assert decision.reason == "explicit_answer_request"


def test_wrapper_preserves_raw_probabilities_but_returns_effective_telling():
    base = FixtureSelector()
    selector = LearnerAgencyEscalationSelector(base, retain_records=True)
    history = [_turn("student", "just give me the answer")]

    returned = selector.predict_probabilities("is 1 + 1 = 11?", history)

    assert returned == TELLING_PROBABILITIES
    assert selector.last_raw_probabilities == PROBING_PROBABILITIES
    assert selector.last_record["raw_active_probabilities"] == PROBING_PROBABILITIES
    assert selector.last_record["effective_probabilities"] == TELLING_PROBABILITIES
    assert selector.last_record["learner_agency_escalation"] == {
        "triggered": True,
        "reason": "explicit_answer_request",
        "latest_student_text": "just give me the answer",
        "consecutive_non_engagement_turns": 0,
    }
    assert base.calls == [("is 1 + 1 = 11?", history)]


def test_wrapper_leaves_engaged_dialogue_under_model_control():
    selector = LearnerAgencyEscalationSelector(FixtureSelector())
    history = [_turn("student", "I think I should add the two values.")]

    returned = selector.predict_probabilities("What is 1 + 1?", history)

    assert returned == PROBING_PROBABILITIES
    assert selector.last_decision is not None
    assert selector.last_decision.triggered is False


def test_effective_distribution_forces_controller_and_tutor_move_to_telling():
    selector = LearnerAgencyEscalationSelector(FixtureSelector())
    effective = selector.predict_probabilities(
        "is 1 + 1 = 11?",
        [_turn("student", "just give me the answer")],
    )
    controller = TurnLevelAttemptController(
        TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    )
    controller.start_attempt(0.2)

    decision = controller.select_turn(effective)

    assert decision.eligible_arms == ("baseline",)
    assert decision.base_move == "telling"
    assert decision.final_move == "telling"
