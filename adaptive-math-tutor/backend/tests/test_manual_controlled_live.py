from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.integrations.adaptive_component_coordinator import (
    AdaptiveAttemptComponents,
    AdaptiveComponentCoordinator,
    _ProductionAdaptiveResources,
)
from app.integrations.adaptive_selector_mode import (
    AdaptiveSelectorMode,
    SELECTOR_MODE_ENV,
    resolve_selector_mode,
)
from app.integrations.manual_controlled_live import (
    MANUAL_CONTROLLED_LIVE_ENV,
    MANUAL_CONTROLLED_LIVE_VALUE,
    ManualControlledLiveDiagnostics,
    file_snapshot,
    manual_controlled_live_enabled,
)
from app.integrations.tutor_state_memory_adapter import TutorStateMemoryAdapter
from app.schemas.tutor import TutorOutput


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_REPO = WORKSPACE_ROOT / "pedagogical-move-selection"
STUDENT_MODEL_REPO = WORKSPACE_ROOT / "student-modeling"
if str(MOVE_REPO) not in sys.path:
    sys.path.insert(0, str(MOVE_REPO))
if str(STUDENT_MODEL_REPO) not in sys.path:
    sys.path.insert(0, str(STUDENT_MODEL_REPO))

from src.self_improvement.controlled_live_selector import (  # noqa: E402
    ActiveMD7WithMD6Shadow,
)
from src.self_improvement.learner_agency_escalation import (  # noqa: E402
    LearnerAgencyEscalationSelector,
)
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.experience_logger import ExperienceLogger  # noqa: E402
from src.self_improvement.lints_policy import TrueDisjointLinTS  # noqa: E402
from src.self_improvement.turn_context_builder import MRB1_TASKS  # noqa: E402
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)
from bkt.predict import BKTPredictor  # noqa: E402


class FixtureSelector:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict_probabilities(self, problem, conversation_history):
        self.calls.append((problem, copy.deepcopy(list(conversation_history))))
        return dict(self.probabilities)


MD6 = {
    "generic": 0.02,
    "probing": 0.94,
    "focus": 0.02,
    "telling": 0.02,
}
MD7 = {
    "generic": 0.02,
    "probing": 0.02,
    "focus": 0.94,
    "telling": 0.02,
}


def test_manual_flag_is_exact_and_fail_closed():
    assert not manual_controlled_live_enabled({})
    assert resolve_selector_mode({}) is AdaptiveSelectorMode.NORMAL_MD7
    assert manual_controlled_live_enabled(
        {MANUAL_CONTROLLED_LIVE_ENV: MANUAL_CONTROLLED_LIVE_VALUE}
    )
    assert resolve_selector_mode(
        {MANUAL_CONTROLLED_LIVE_ENV: MANUAL_CONTROLLED_LIVE_VALUE}
    ) is AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE
    with pytest.raises(ValueError):
        manual_controlled_live_enabled({MANUAL_CONTROLLED_LIVE_ENV: "true"})
    with pytest.raises(ValueError):
        resolve_selector_mode({SELECTOR_MODE_ENV: "md7"})
    with pytest.raises(ValueError):
        resolve_selector_mode(
            {
                SELECTOR_MODE_ENV: AdaptiveSelectorMode.MD6_ROLLBACK.value,
                MANUAL_CONTROLLED_LIVE_ENV: MANUAL_CONTROLLED_LIVE_VALUE,
            }
        )


def test_active_md7_and_shadow_md6_receive_identical_fixture():
    active = FixtureSelector(MD7)
    shadow = FixtureSelector(MD6)
    paired = ActiveMD7WithMD6Shadow(active_md7=active, shadow_md6=shadow)
    history = [{"user": "student", "text": "I think it is 12."}]

    returned = paired.predict_probabilities("What is 25% of 80?", history)

    assert returned == MD7
    assert active.calls == shadow.calls
    assert active.calls[0] == ("What is 25% of 80?", history)
    assert paired.last_record["md6_probabilities"] == MD6
    assert paired.last_record["md7_probabilities"] == MD7


def test_resource_routing_promotes_md7_and_keeps_explicit_md6_rollback(monkeypatch):
    class Loader:
        def __init__(self) -> None:
            self.paths: list[Path | None] = []

        def __call__(self, model_dir=None):
            self.paths.append(None if model_dir is None else Path(model_dir))
            return FixtureSelector(MD6 if model_dir is None else MD7)

    monkeypatch.delenv(MANUAL_CONTROLLED_LIVE_ENV, raising=False)
    monkeypatch.delenv(SELECTOR_MODE_ENV, raising=False)
    ordinary_resources = _ProductionAdaptiveResources()
    ordinary_loader = Loader()
    ordinary = ordinary_resources._build_base_selector(ordinary_loader)
    assert ordinary_resources.selector_mode is AdaptiveSelectorMode.NORMAL_MD7
    assert isinstance(ordinary, LearnerAgencyEscalationSelector)
    assert isinstance(ordinary.base_selector, FixtureSelector)
    assert ordinary_loader.paths == [ordinary_resources.md7_model_dir]

    monkeypatch.setenv(
        SELECTOR_MODE_ENV,
        AdaptiveSelectorMode.MD6_ROLLBACK.value,
    )
    rollback_resources = _ProductionAdaptiveResources()
    rollback_loader = Loader()
    rollback = rollback_resources._build_base_selector(rollback_loader)
    assert rollback_resources.selector_mode is AdaptiveSelectorMode.MD6_ROLLBACK
    assert isinstance(rollback, LearnerAgencyEscalationSelector)
    assert isinstance(rollback.base_selector, FixtureSelector)
    assert rollback_loader.paths == [None]

    monkeypatch.delenv(SELECTOR_MODE_ENV, raising=False)
    monkeypatch.setenv(
        MANUAL_CONTROLLED_LIVE_ENV,
        MANUAL_CONTROLLED_LIVE_VALUE,
    )
    manual_resources = _ProductionAdaptiveResources()
    manual_loader = Loader()
    manual = manual_resources._build_base_selector(manual_loader)
    assert manual_resources.manual_controlled_live is True
    assert isinstance(manual, LearnerAgencyEscalationSelector)
    assert isinstance(manual.base_selector, ActiveMD7WithMD6Shadow)
    assert manual_loader.paths == [manual_resources.md7_model_dir, None]
    assert manual.base_selector.shadow_md6 is manual_resources.shadow_md6
    assert manual.base_selector.active_md7 is manual_resources.active_md7


def test_isolated_turn_log_has_both_selectors_and_preserves_policy(tmp_path):
    active = FixtureSelector(MD7)
    shadow = FixtureSelector(MD6)
    paired = ActiveMD7WithMD6Shadow(active_md7=active, shadow_md6=shadow)
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    production_policy = tmp_path / "real" / "policy_state.json"
    production_experience = tmp_path / "real" / "attempts.jsonl"
    production_policy.parent.mkdir(parents=True)
    production_policy.write_bytes(b"policy-before\n")
    production_experience.write_bytes(b"experience-before\n")
    policy_before = file_snapshot(production_policy)
    experience_before = file_snapshot(production_experience)
    output_dir = tmp_path / "isolated-results"
    diagnostics = ManualControlledLiveDiagnostics(
        selector=paired,
        policy=policy,
        output_dir=output_dir,
        production_policy_path=production_policy,
        production_experience_path=production_experience,
    )

    history = [{"user": "student", "text": "I think it is 12."}]
    returned = paired.predict_probabilities("What is 25% of 80?", history)
    record = diagnostics.record_turn(
        attempt_id="fixture-thread:1",
        turn_result={
            "turn_index": 1,
            "md6_probabilities": returned,
            "eligible_arms": ["baseline"],
            "selected_arm": "baseline",
            "pedagogical_move": "focus",
            "overridden": False,
            "tutor_response": "Which multiplication represents 25 percent?",
            "mrb1_scores": {"coherence": 1.0},
        },
        decision=SimpleNamespace(
            base_move="focus",
            gap=0.0,
            gap_threshold=0.10,
        ),
    )

    assert record["md6"]["argmax"] == "probing"
    assert record["md7"]["argmax"] == "focus"
    assert record["transition"] == "probing -> focus"
    assert record["overlay"]["final_move"] == "focus"
    assert policy.total_updates == 0
    assert file_snapshot(production_policy) == policy_before
    assert file_snapshot(production_experience) == experience_before
    lines = diagnostics.turn_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record
    summary = json.loads(diagnostics.summary_path.read_text(encoding="utf-8"))
    assert summary["protected_files_unchanged"] is True
    assert summary["lints_posterior_updates"] == 0


def test_manual_log_preserves_raw_md7_and_records_effective_agency_telling(
    tmp_path,
):
    paired = ActiveMD7WithMD6Shadow(
        active_md7=FixtureSelector(MD7),
        shadow_md6=FixtureSelector(MD6),
    )
    selector = LearnerAgencyEscalationSelector(
        paired,
        retain_records=True,
    )
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    production_policy = tmp_path / "real" / "policy_state.json"
    production_experience = tmp_path / "real" / "attempts.jsonl"
    production_policy.parent.mkdir(parents=True)
    production_policy.write_bytes(b"policy-before\n")
    production_experience.write_bytes(b"experience-before\n")
    diagnostics = ManualControlledLiveDiagnostics(
        selector=selector,
        policy=policy,
        output_dir=tmp_path / "isolated-results",
        production_policy_path=production_policy,
        production_experience_path=production_experience,
    )
    history = [{"user": "student", "text": "just give me the answer"}]

    effective = selector.predict_probabilities("is 1 + 1 = 11?", history)
    record = diagnostics.record_turn(
        attempt_id="fixture-thread:1",
        turn_result={
            "turn_index": 1,
            "md6_probabilities": effective,
            "eligible_arms": ["baseline"],
            "selected_arm": "baseline",
            "pedagogical_move": "telling",
            "overridden": False,
            "tutor_response": "No. In ordinary arithmetic, 1 + 1 = 2.",
            "mrb1_scores": {"coherence": 1.0},
        },
        decision=SimpleNamespace(
            base_move="telling",
            gap=0.0,
            gap_threshold=0.10,
        ),
    )

    assert record["md7"]["argmax"] == "focus"
    assert record["selector_effective"]["argmax"] == "telling"
    assert record["learner_agency"]["triggered"] is True
    assert record["learner_agency"]["reason"] == "explicit_answer_request"
    assert record["overlay"]["final_move"] == "telling"


def test_bkt_activity_log_records_observation_level_mastery_transitions(tmp_path):
    paired = ActiveMD7WithMD6Shadow(
        active_md7=FixtureSelector(MD7),
        shadow_md6=FixtureSelector(MD6),
    )
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    production_policy = tmp_path / "real" / "policy_state.json"
    production_experience = tmp_path / "real" / "attempts.jsonl"
    production_policy.parent.mkdir(parents=True)
    production_policy.write_bytes(b"policy-before\n")
    production_experience.write_bytes(b"experience-before\n")
    diagnostics = ManualControlledLiveDiagnostics(
        selector=paired,
        policy=policy,
        output_dir=tmp_path / "isolated-results",
        production_policy_path=production_policy,
        production_experience_path=production_experience,
    )
    params = {
        "Target": {
            "prior": 0.2,
            "learns": 0.1,
            "guesses": 0.2,
            "slips": 0.1,
            "forgets": 0.0,
        }
    }
    predictor = BKTPredictor(params)
    observations = [(1, 1.0), (1, 1.0), (0, 1.0)]

    class Graph:
        def __init__(self):
            self.predictor = predictor

        @staticmethod
        def get_attempts(student_id, skill):
            assert student_id == "student-1"
            assert skill == "Target"
            return list(observations)

        @staticmethod
        def get_effective_initial_prior(student_id, skill):
            assert student_id == "student-1"
            assert skill == "Target"
            return {
                "effective_initial_prior": 0.2,
                "prior_source": "population",
            }

    answers = [
        {
            "question_id": f"q{index}",
            "question": f"Question {index}",
            "student_answer": answer,
            "is_correct": correct,
        }
        for index, (answer, correct) in enumerate(
            (("yes", True), ("yes", True), ("no", False)),
            start=1,
        )
    ]
    resolved_events = [
        {
            "event_id": f"attempt-1:{index}",
            "student_id": "student-1",
            "skill_id": "Target",
            "primary_signal": "correct_answer" if outcome else "incorrect_answer",
            "resolver_version": "2.0",
            "behaviour": {},
            "history": {"repeated_misunderstanding": False},
            "bkt_update": {
                "should_update": True,
                "outcome": outcome,
                "evidence_weight": 1.0,
                "evaluator_confidence": 1.0,
                "behavioural_confidence": 0.0,
                "behaviour_factor": 1.0,
                "update_confidence": confidence,
                "observation_source": "evaluator",
                "contributors": [],
            },
        }
        for index, (outcome, confidence) in enumerate(observations, start=1)
    ]
    mastery_after = predictor.predict(
        "Target",
        observations,
        initial_prior=0.2,
    )
    student_model_result = {
        "learning_outcome": {
            "attempt_id": "attempt-1",
            "skill": "Target",
            "mastery_before": 0.2,
            "mastery_after": mastery_after,
            "delta_mastery": mastery_after - 0.2,
        },
        "resolved_events": resolved_events,
        "knowledge_graph_result": {
            "session_id": "attempt-1",
            "skills_updated": [
                {
                    "skill": "Target",
                    "probability": mastery_after,
                    "label": "partial",
                }
            ],
        },
    }

    record = diagnostics.record_bkt_activity(
        student_id="student-1",
        attempt_id="attempt-1",
        target_skill="Target",
        evaluated_answers=answers,
        student_model_result=student_model_result,
        knowledge_graph=Graph(),
    )

    assert record["event"] == "assessment_cycle_bkt_update"
    assert record["assessment"]["correct_count"] == 2
    assert record["history"]["observations_applied_in_assessment"] == 3
    assert record["mastery"]["before_consistent"] is True
    assert record["mastery"]["after_consistent"] is True
    transitions = [
        item["mastery_transition"] for item in record["observations"]
    ]
    assert transitions[0]["mastery_before_observation"] == pytest.approx(0.2)
    assert transitions[-1]["mastery_after_observation"] == pytest.approx(
        mastery_after
    )
    lines = diagnostics.bkt_activity_log_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record
    summary = json.loads(diagnostics.summary_path.read_text(encoding="utf-8"))
    assert summary["bkt_activity_count"] == 1
    assert summary["bkt_activity_log_path"] == str(
        diagnostics.bkt_activity_log_path
    )


def test_bkt_activity_log_records_live_dialogue_updates(tmp_path):
    paired = ActiveMD7WithMD6Shadow(
        active_md7=FixtureSelector(MD7),
        shadow_md6=FixtureSelector(MD6),
    )
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    production_policy = tmp_path / "real" / "policy_state.json"
    production_experience = tmp_path / "real" / "attempts.jsonl"
    production_policy.parent.mkdir(parents=True)
    production_policy.write_bytes(b"policy-before\n")
    production_experience.write_bytes(b"experience-before\n")
    diagnostics = ManualControlledLiveDiagnostics(
        selector=paired,
        policy=policy,
        output_dir=tmp_path / "isolated-results",
        production_policy_path=production_policy,
        production_experience_path=production_experience,
    )
    predictor = BKTPredictor(
        {
            "Target": {
                "prior": 0.2,
                "learns": 0.1,
                "guesses": 0.2,
                "slips": 0.1,
                "forgets": 0.0,
            }
        }
    )
    observations = [(0, 0.2)]
    mastery_after = predictor.predict(
        "Target",
        observations,
        initial_prior=0.2,
    )

    class Graph:
        def __init__(self):
            self.predictor = predictor

        @staticmethod
        def get_attempts(student_id, skill):
            assert (student_id, skill) == ("student-1", "Target")
            return list(observations)

        @staticmethod
        def get_effective_initial_prior(student_id, skill):
            assert (student_id, skill) == ("student-1", "Target")
            return {
                "effective_initial_prior": 0.2,
                "prior_source": "population",
            }

    resolved_event = {
        "event_id": "attempt-1:dialogue:1:turn_1:target",
        "student_id": "student-1",
        "skill_id": "Target",
        "primary_signal": "behavioural_difficulty",
        "resolver_version": "2.0",
        "behaviour": {"uncertainty_present": True},
        "history": {"repeated_misunderstanding": False},
        "bkt_update": {
            "should_update": True,
            "outcome": 0,
            "evidence_weight": 0.25,
            "evaluator_confidence": 0.0,
            "behavioural_confidence": 0.8,
            "behaviour_factor": 1.0,
            "update_confidence": 0.2,
            "observation_source": "behavioural_proxy",
            "contributors": ["uncertainty"],
        },
    }
    student_model_result = {
        "incremental_learning_outcome": {
            "mastery_before": 0.2,
            "mastery_after": mastery_after,
            "delta_mastery": mastery_after - 0.2,
        },
        "resolved_events": [resolved_event],
        "knowledge_graph_result": {
            "session_id": "attempt-1:dialogue:1",
            "skills_updated": [
                {
                    "skill": "Target",
                    "probability": mastery_after,
                    "label": "weak",
                }
            ],
        },
    }
    evidence = {
        "turn_index": 1,
        "teacher_text": "What would you try first?",
        "student_text": "I don't know",
        "correctness": "unknown",
        "reported_evaluator_confidence": 0.0,
        "applied_evaluator_confidence": 0.0,
        "evaluator_confidence_cap": 0.5,
        "evaluator_reason": "No assessable mathematical answer.",
        "evaluator_source": "adaptmath_teaching_progress_judge",
    }

    record = diagnostics.record_dialogue_bkt_activity(
        student_id="student-1",
        attempt_id="attempt-1",
        target_skill="Target",
        dialogue_evidence=evidence,
        student_model_result=student_model_result,
        knowledge_graph=Graph(),
    )

    assert record["schema_version"] == "adaptmath_bkt_activity_v2"
    assert record["event"] == "dialogue_turn_bkt_update"
    assert record["update_timing"] == "after_student_dialogue_turn"
    assert record["history"]["observations_applied_in_dialogue_turn"] == 1
    assert record["mastery"]["before_consistent"] is True
    assert record["mastery"]["after_consistent"] is True
    assert record["observations"][0]["mastery_transition"] is not None
    assert len(
        diagnostics.bkt_activity_log_path.read_text(encoding="utf-8").splitlines()
    ) == 1


def test_diagnostics_recovers_after_uncommitted_selector_record(tmp_path):
    active = FixtureSelector(MD7)
    shadow = FixtureSelector(MD6)
    paired = ActiveMD7WithMD6Shadow(active_md7=active, shadow_md6=shadow)
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    production_policy = tmp_path / "real" / "policy_state.json"
    production_experience = tmp_path / "real" / "attempts.jsonl"
    production_policy.parent.mkdir(parents=True)
    production_policy.write_bytes(b"policy-before\n")
    production_experience.write_bytes(b"experience-before\n")
    diagnostics = ManualControlledLiveDiagnostics(
        selector=paired,
        policy=policy,
        output_dir=tmp_path / "isolated-results",
        production_policy_path=production_policy,
        production_experience_path=production_experience,
    )

    # Simulate a selector call whose downstream tutor generation failed before
    # the diagnostics record could be committed.
    paired.predict_probabilities("Fixture problem", [])
    paired.predict_probabilities(
        "Fixture problem",
        [{"user": "student", "text": "Try again."}],
    )
    record = diagnostics.record_turn(
        attempt_id="fixture-thread:1",
        turn_result={
            "turn_index": 1,
            "md6_probabilities": MD7,
            "eligible_arms": ["baseline"],
            "selected_arm": "baseline",
            "pedagogical_move": "focus",
            "overridden": False,
            "tutor_response": "Recovered fixture response.",
            "mrb1_scores": {"coherence": 1.0},
        },
        decision=SimpleNamespace(
            base_move="focus",
            gap=0.0,
            gap_threshold=0.10,
        ),
    )

    assert record["latest_student_text"] == "Try again."
    assert len(diagnostics.turn_log_path.read_text(encoding="utf-8").splitlines()) == 1
    assert policy.total_updates == 0


def test_md7_drives_c3_overlay_final_move_and_abort_has_zero_updates(tmp_path):
    active = FixtureSelector(MD7)
    shadow = FixtureSelector(MD6)
    paired = ActiveMD7WithMD6Shadow(active_md7=active, shadow_md6=shadow)
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    experience_path = tmp_path / "attempts.jsonl"
    policy_path = tmp_path / "policy.json"

    class StudentPipeline:
        def start_attempt(self, *, student_id, attempt_id, attempt_skill=None):
            return SimpleNamespace(
                student_id=student_id,
                attempt_id=attempt_id,
                skill=attempt_skill,
                mastery_before=0.2,
            )

    class MRB1:
        def score_response(self, conversation_history, tutor_response):
            del conversation_history, tutor_response
            return {task: 1.0 for task in MRB1_TASKS}

    class Tutor:
        def teach_turn(self, tutor_input, evidence):
            del evidence
            return TutorOutput(
                teaching_response=f"Fixture tutor used {tutor_input.pedagogical_move}."
            )

    def factory(**identity):
        del identity
        memory_adapter = TutorStateMemoryAdapter()
        adaptive = AdaptiveTutorPipeline(
            md6=paired,
            mrb1=MRB1(),
            controller=TurnLevelAttemptController(policy),
            experience_logger=ExperienceLogger(
                experience_path,
                data_mode="synthetic",
                source_policy=policy,
            ),
            policy_state_path=policy_path,
            memory_adapter=memory_adapter,
        )
        student = StudentPipeline()
        return AdaptiveAttemptComponents(
            bridge=StudentModelFrozenV3Bridge(
                student_model_pipeline=student,
                adaptive_pipeline=adaptive,
            ),
            student_model_pipeline=student,
            memory_adapter=memory_adapter,
        )

    coordinator = AdaptiveComponentCoordinator(
        component_factory=factory,
        skill_validator=lambda skill: skill == "Percent Of",
    )
    state = {
        "thread_id": "offline-fixture",
        "student_id": "offline-student",
        "attempt_id": "offline-fixture:1",
        "adaptive_attempt_index": 1,
        "target_skill": "Percent Of",
        "adaptive_lifecycle_status": "not_started",
        "question": "What is 25 percent of 80?",
        "topic": "Percentages",
        "subtopic": "Percent of a quantity",
        "age": 14,
        "complexity_score": 0.3,
        "planner_output": {},
        "previous_errors": [],
        "teaching_phase": "initial",
        "verified_math_evidence": [
            {"action": "offline_fixture", "result": "verified"}
        ],
        "conversation_history": [
            {"role": "student", "content": "I think it is 12."}
        ],
        "turn_count": 0,
    }
    state.update(coordinator.start_attempt(state))
    result = coordinator.run_tutor_turn(state, tutor_agent=Tutor())

    assert result["md6_probabilities"] == MD7
    assert result["context"][:4] == [MD7[move] for move in MD7]
    assert result["eligible_arms"] == ["baseline"]
    assert result["base_move"] == "focus"
    assert result["pedagogical_move"] == "focus"
    assert result["tutor_response"] == "Fixture tutor used focus."
    coordinator.abort_attempt(state)
    assert policy.total_updates == 0
    assert not experience_path.exists()
    assert not policy_path.exists()
