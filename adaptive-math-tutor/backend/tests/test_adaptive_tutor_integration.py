from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.integrations.adaptive_component_coordinator import (
    _AssessmentEvidenceSource,
    AdaptiveAttemptComponents,
    AdaptiveComponentCoordinator,
    AdaptiveConcurrencyError,
    AdaptiveRestartRequiredError,
    TutorAssessmentStudentModelPipeline,
)
from app.integrations.adaptive_tutor_agent_adapter import (
    AdaptiveTutorAgentAdapter,
)
from app.integrations.tutor_state_memory_adapter import (
    TutorStateMemoryAdapter,
)
from app.schemas.tutor import TutorOutput


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_REPO = WORKSPACE_ROOT / "pedagogical-move-selection"
if str(MOVE_REPO) not in sys.path:
    sys.path.insert(0, str(MOVE_REPO))

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


class RecordingMD6:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[str, list[dict[str, object]]]] = []
        self.events = events

    def predict_probabilities(self, problem, conversation_history):
        if self.events is not None:
            self.events.append("md6")
        self.calls.append((problem, copy.deepcopy(list(conversation_history))))
        return {
            "generic": 0.97,
            "probing": 0.01,
            "focus": 0.01,
            "telling": 0.01,
        }


class RecordingMRB1:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[list[dict[str, object]], str]] = []
        self.events = events

    def score_response(self, conversation_history, tutor_response):
        if self.events is not None:
            self.events.append("mrb1")
        self.calls.append(
            (copy.deepcopy(list(conversation_history)), tutor_response)
        )
        return {task: 1.0 for task in MRB1_TASKS}


class RecordingTutor:
    def __init__(self, response: str = "What would you try first?", events=None):
        self.response = response
        self.events = events
        self.calls = []

    def teach_turn(self, tutor_input, evidence):
        if self.events is not None:
            self.events.append("tutor")
        self.calls.append((tutor_input, copy.deepcopy(evidence)))
        return TutorOutput(teaching_response=self.response)


class RejectingTutor:
    def teach_turn(self, tutor_input, evidence):
        del tutor_input, evidence
        raise RuntimeError("Existing verifier rejected a conflicting move.")


class FakeStudentPipeline:
    def __init__(self, mastery_store: dict[tuple[str, str], float]) -> None:
        self.mastery_store = mastery_store
        self.context = None
        self.process_calls = 0

    def start_attempt(self, *, student_id, attempt_id, attempt_skill=None):
        if attempt_skill is None:
            raise ValueError("attempt_skill is required")
        mastery = self.mastery_store.setdefault((student_id, attempt_skill), 0.2)
        self.context = SimpleNamespace(
            student_id=student_id,
            attempt_id=attempt_id,
            skill=attempt_skill,
            mastery_before=mastery,
        )
        return self.context

    def process_assessment_cycle(
        self,
        *,
        student_id,
        attempt_id,
        evaluated_answers,
    ):
        self.process_calls += 1
        if self.context is None:
            raise RuntimeError("attempt not started")
        if student_id != self.context.student_id or attempt_id != self.context.attempt_id:
            raise ValueError("identity mismatch")
        correct = sum(bool(item["is_correct"]) for item in evaluated_answers)
        delta = 0.1 if correct >= 2 else -0.05
        after = min(1.0, max(0.0, self.context.mastery_before + delta))
        self.mastery_store[(student_id, self.context.skill)] = after
        return {
            "learning_outcome": {
                "attempt_id": attempt_id,
                "skill": self.context.skill,
                "mastery_before": self.context.mastery_before,
                "mastery_after": after,
                "delta_mastery": after - self.context.mastery_before,
            }
        }


def tutor_state(
    thread_id: str = "thread-a",
    student_id: str = "student-a",
    *,
    attempt_index: int = 1,
    target_skill: str = "Skill A",
):
    return {
        "thread_id": thread_id,
        "student_id": student_id,
        "attempt_id": f"{thread_id}:{attempt_index}",
        "adaptive_attempt_index": attempt_index,
        "target_skill": target_skill,
        "adaptive_lifecycle_status": "not_started",
        "question": "Solve x + 2 = 5.",
        "topic": "Algebra",
        "subtopic": "Equations",
        "age": 14,
        "complexity_score": 0.4,
        "planner_output": {},
        "previous_errors": [],
        "teaching_phase": "initial",
        "verified_math_evidence": [
            {"action": "final_claim_verification", "result": "true"}
        ],
        "conversation_history": [
            {"role": "student", "content": "I am not sure where to begin."}
        ],
        "turn_count": 0,
    }


def assessment_state(state, *, needs_reteaching: bool):
    updated = copy.deepcopy(state)
    questions = [
        {"question_id": f"q{i}", "question": f"Question {i}?", "expected_answer": str(i)}
        for i in range(1, 4)
    ]
    evaluated = [
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "student_answer": str(i),
            "expected_answer": str(i),
            "is_correct": i != 3 if needs_reteaching else True,
            "feedback": "checked",
            "identified_error": "error" if i == 3 and needs_reteaching else None,
        }
        for i in range(1, 4)
    ]
    wrong = [item for item in evaluated if not item["is_correct"]]
    correct = [item for item in evaluated if item["is_correct"]]
    updated.update(
        {
            "assessment_questions": questions,
            "student_answers": [
                {"question_id": f"q{i}", "answer": str(i)}
                for i in range(1, 4)
            ],
            "evaluation_result": {
                "correct_answers": correct,
                "wrong_answers": wrong,
                "identified_errors": [],
                "needs_reteaching": needs_reteaching,
                "overall_feedback": "checked",
            },
            "needs_reteaching": needs_reteaching,
        }
    )
    return updated


class AdaptiveTutorAdapterTests(unittest.TestCase):
    def test_all_canonical_moves_reach_tutor_input_unchanged(self):
        for move in ("generic", "probing", "focus", "telling"):
            with self.subTest(move=move):
                state = tutor_state()
                memory_adapter = TutorStateMemoryAdapter()
                tutor = RecordingTutor()
                adapter = AdaptiveTutorAgentAdapter(
                    tutor_agent=tutor,
                    tutor_state=state,
                    memory_adapter=memory_adapter,
                )
                history = memory_adapter.get_conversation_history(state)
                response = adapter.generate(
                    problem=state["question"],
                    conversation_history=history,
                    pedagogical_move=move,
                )
                self.assertEqual(tutor.calls[0][0].pedagogical_move, move)
                self.assertEqual(tutor.calls[0][1], state["verified_math_evidence"])
                memory_adapter.append_tutor_response(state, response)
                self.assertEqual(
                    state["conversation_history"][-1]["pedagogical_move"],
                    move,
                )

    def test_invalid_move_and_tutor_verifier_failure_do_not_append(self):
        state = tutor_state()
        memory_adapter = TutorStateMemoryAdapter()
        adapter = AdaptiveTutorAgentAdapter(
            tutor_agent=RecordingTutor(),
            tutor_state=state,
            memory_adapter=memory_adapter,
        )
        with self.assertRaises(ValueError):
            adapter.generate(
                problem=state["question"],
                conversation_history=memory_adapter.get_conversation_history(state),
                pedagogical_move="strategy_override",
            )

        rejecting = AdaptiveTutorAgentAdapter(
            tutor_agent=RejectingTutor(),
            tutor_state=state,
            memory_adapter=memory_adapter,
        )
        with self.assertRaises(RuntimeError):
            rejecting.generate(
                problem=state["question"],
                conversation_history=memory_adapter.get_conversation_history(state),
                pedagogical_move="focus",
            )
        self.assertEqual(len(state["conversation_history"]), 1)


class AdaptivePipelineHistoryTests(unittest.TestCase):
    def test_history_snapshot_causality_and_single_append(self):
        events: list[str] = []
        md6 = RecordingMD6(events)
        mrb1 = RecordingMRB1(events)
        tutor = RecordingTutor(events=events)
        state = tutor_state()
        pre_history = copy.deepcopy(state["conversation_history"])

        with tempfile.TemporaryDirectory() as temp_dir:
            policy = TrueDisjointLinTS(context_dim=9, seed=7, data_mode="synthetic")
            memory_adapter = TutorStateMemoryAdapter()
            adaptive = AdaptiveTutorPipeline(
                md6=md6,
                mrb1=mrb1,
                controller=TurnLevelAttemptController(policy),
                experience_logger=ExperienceLogger(
                    Path(temp_dir) / "attempts.jsonl",
                    data_mode="synthetic",
                    source_policy=policy,
                ),
                policy_state_path=Path(temp_dir) / "policy.json",
                memory_adapter=memory_adapter,
            )
            student = FakeStudentPipeline({})
            bridge = StudentModelFrozenV3Bridge(
                student_model_pipeline=student,
                adaptive_pipeline=adaptive,
            )
            bridge.start_attempt(
                student_id=state["student_id"],
                attempt_id=state["attempt_id"],
                target_skill=state["target_skill"],
                adaptive_memory=state,
            )
            adapter = AdaptiveTutorAgentAdapter(
                tutor_agent=tutor,
                tutor_state=state,
                memory_adapter=memory_adapter,
            )
            result = adaptive.run_tutor_turn(state, adapter)

        projected_pre = [
            {"user": turn["role"], "text": turn["content"]}
            for turn in pre_history
        ]
        self.assertEqual(events, ["md6", "tutor", "mrb1"])
        self.assertEqual(md6.calls[0][1], projected_pre)
        self.assertEqual(
            [turn.model_dump(exclude_none=True) for turn in tutor.calls[0][0].conversation_history],
            pre_history,
        )
        self.assertEqual(mrb1.calls[0], (projected_pre, result["tutor_response"]))
        self.assertEqual(len(state["conversation_history"]), len(pre_history) + 1)
        self.assertEqual(state["conversation_history"][-1]["role"], "teacher")


class AssessmentEvidenceProjectionTests(unittest.TestCase):
    def test_structured_tutor_verdicts_enter_cross_session_pipeline(self):
        class CapturingCrossSession:
            def __init__(self):
                self.context = None
                self.process_kwargs = None

            def start_attempt(self, *, student_id, attempt_id, attempt_skill):
                self.context = SimpleNamespace(
                    student_id=student_id,
                    attempt_id=attempt_id,
                    skill=attempt_skill,
                    mastery_before=0.4,
                )
                return self.context

            def process_transcript(self, **kwargs):
                self.process_kwargs = kwargs
                return {"learning_outcome": {"attempt_id": "thread-a:1"}}

        source = _AssessmentEvidenceSource("Skill A")
        cross_session = CapturingCrossSession()
        pipeline = TutorAssessmentStudentModelPipeline(
            target_skill="Skill A",
            cross_session_pipeline=cross_session,
            evidence_source=source,
        )
        context = pipeline.start_attempt(
            student_id="student-a",
            attempt_id="thread-a:1",
            attempt_skill="Skill A",
        )
        evaluated = [
            {"question": "What is 2 + 2?", "student_answer": "4", "is_correct": True},
            {"question": "What is 3 + 3?", "student_answer": "5", "is_correct": False},
        ]
        pipeline.process_assessment_cycle(
            student_id="student-a",
            attempt_id="thread-a:1",
            evaluated_answers=evaluated,
        )

        kwargs = cross_session.process_kwargs
        self.assertIsNotNone(kwargs)
        self.assertEqual(
            [turn["role"] for turn in kwargs["transcript"]],
            ["tutor", "student", "tutor", "student"],
        )
        self.assertIs(kwargs["attempt_context"], context)
        self.assertEqual(kwargs["behavioural_skill"], "Skill A")
        extraction = source.extract(kwargs["transcript"])
        self.assertEqual([event["turn_index"] for event in extraction["events"]], [1, 3])
        self.assertEqual(
            source(extraction["events"][0], kwargs["transcript"])["correctness"],
            "correct",
        )
        self.assertEqual(
            source(extraction["events"][1], kwargs["transcript"])["correctness"],
            "incorrect",
        )


class AdaptiveCoordinatorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp.name)
        self.policy = TrueDisjointLinTS(
            context_dim=9,
            seed=11,
            data_mode="synthetic",
        )
        self.logger = ExperienceLogger(
            self.temp_path / "attempts.jsonl",
            data_mode="synthetic",
            source_policy=self.policy,
        )
        self.mastery_store: dict[tuple[str, str], float] = {}
        self.student_pipelines: list[FakeStudentPipeline] = []
        self.cached_evidence_by_thread: dict[str, list[dict[str, str]]] = {}

        def factory(**identity):
            memory_adapter = TutorStateMemoryAdapter()
            adaptive = AdaptiveTutorPipeline(
                md6=RecordingMD6(),
                mrb1=RecordingMRB1(),
                controller=TurnLevelAttemptController(self.policy),
                experience_logger=self.logger,
                policy_state_path=self.temp_path / "policy.json",
                memory_adapter=memory_adapter,
            )
            student = FakeStudentPipeline(self.mastery_store)
            self.student_pipelines.append(student)
            bridge = StudentModelFrozenV3Bridge(
                student_model_pipeline=student,
                adaptive_pipeline=adaptive,
            )
            self.cached_evidence_by_thread[identity["thread_id"]] = []
            return AdaptiveAttemptComponents(
                bridge=bridge,
                student_model_pipeline=student,
                memory_adapter=memory_adapter,
            )

        self.factory = factory
        self.coordinator = AdaptiveComponentCoordinator(
            component_factory=factory,
            skill_validator=lambda skill: skill in {"Skill A", "Skill B"},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_failed_assessment_finishes_once_and_starts_next_attempt(self):
        state = tutor_state()
        started = self.coordinator.start_attempt(state)
        state.update(started)
        turn = self.coordinator.run_tutor_turn(
            state,
            tutor_agent=RecordingTutor(),
        )
        state["turn_count"] += 1
        self.assertEqual(turn["attempt_id"], "thread-a:1")
        self.assertEqual(len(state["conversation_history"]), 2)

        old_state = assessment_state(state, needs_reteaching=True)
        completed = self.coordinator.finish_attempt(old_state)
        self.assertEqual(completed["completed_attempt_id"], "thread-a:1")
        self.assertEqual(completed["attempt_id"], "thread-a:2")
        self.assertEqual(completed["adaptive_attempt_index"], 2)
        self.assertEqual(completed["adaptive_lifecycle_status"], "active")
        self.assertAlmostEqual(completed["mastery_before"], 0.3)
        self.assertEqual(self.policy.total_updates, 1)
        self.assertEqual(len(self.student_pipelines), 2)
        self.assertEqual(self.student_pipelines[0].process_calls, 1)

        duplicate = self.coordinator.finish_attempt(old_state)
        self.assertEqual(duplicate, completed)
        self.assertEqual(self.policy.total_updates, 1)
        self.assertEqual(self.student_pipelines[0].process_calls, 1)
        log_lines = (self.temp_path / "attempts.jsonl").read_text().splitlines()
        self.assertEqual(len(log_lines), 1)

        next_state = copy.deepcopy(old_state)
        next_state.update(completed)
        self.coordinator.abort_attempt(next_state)

    def test_two_threads_are_isolated_and_overlap_is_rejected(self):
        first = tutor_state("thread-a", "student-a", target_skill="Skill A")
        second = tutor_state("thread-b", "student-b", target_skill="Skill B")
        first["verified_math_evidence"] = [{"action": "a", "result": "1"}]
        second["verified_math_evidence"] = [{"action": "b", "result": "2"}]

        first.update(self.coordinator.start_attempt(first))
        with self.assertRaises(AdaptiveConcurrencyError):
            self.coordinator.start_attempt(second)

        self.coordinator.run_tutor_turn(first, tutor_agent=RecordingTutor("A"))
        first_completed = assessment_state(first, needs_reteaching=False)
        self.coordinator.finish_attempt(first_completed)
        second.update(self.coordinator.start_attempt(second))
        self.coordinator.run_tutor_turn(second, tutor_agent=RecordingTutor("B"))

        self.assertEqual(first["student_id"], "student-a")
        self.assertEqual(second["student_id"], "student-b")
        self.assertEqual(first["conversation_history"][0]["content"], "I am not sure where to begin.")
        self.assertEqual(second["conversation_history"][-1]["content"], "B")
        self.assertNotEqual(
            first["verified_math_evidence"],
            second["verified_math_evidence"],
        )
        self.coordinator.abort_attempt(second)

    def test_invalid_skill_and_restart_reconciliation_fail_closed(self):
        invalid = tutor_state(target_skill="topic guessed as a skill")
        with self.assertRaises(KeyError):
            self.coordinator.start_attempt(invalid)

        persisted = tutor_state()
        persisted["adaptive_lifecycle_status"] = "active"
        restarted = AdaptiveComponentCoordinator(
            component_factory=self.factory,
            skill_validator=lambda skill: skill == "Skill A",
        )
        with self.assertRaises(AdaptiveRestartRequiredError) as raised:
            restarted.start_attempt(persisted)
        self.assertIn("process restart", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
