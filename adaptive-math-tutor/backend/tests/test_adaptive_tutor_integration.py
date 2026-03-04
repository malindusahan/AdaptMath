from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from app.integrations.adaptive_component_coordinator import (
    DIALOGUE_EVALUATOR_CONFIDENCE_CAP,
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
from app.integrations.self_improvement_turn_collector import PassiveTurnCollector
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


class FailOnceTutor:
    def __init__(self) -> None:
        self.calls = 0

    def teach_turn(self, tutor_input, evidence):
        del tutor_input, evidence
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Temporary tutor failure.")
        return TutorOutput(teaching_response="What would you try first?")


class FailOnceMRB1(RecordingMRB1):
    def score_response(self, conversation_history, tutor_response):
        if not self.calls:
            self.calls.append(
                (copy.deepcopy(list(conversation_history)), tutor_response)
            )
            raise RuntimeError("Temporary MRB1 failure.")
        return super().score_response(conversation_history, tutor_response)


class FakeStudentPipeline:
    def __init__(self, mastery_store: dict[tuple[str, str], float]) -> None:
        self.mastery_store = mastery_store
        self.context = None
        self.process_calls = 0
        self.dialogue_calls = []

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

    def current_mastery(self, *, student_id, attempt_id):
        if self.context is None:
            raise RuntimeError("attempt not started")
        if student_id != self.context.student_id or attempt_id != self.context.attempt_id:
            raise ValueError("identity mismatch")
        return self.mastery_store[(student_id, self.context.skill)]

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

    def process_dialogue_turn(self, **kwargs):
        self.dialogue_calls.append(copy.deepcopy(kwargs))
        if self.context is None:
            raise RuntimeError("attempt not started")
        return {
            "incremental_learning_outcome": {
                "mastery_before": self.context.mastery_before,
                "mastery_after": self.context.mastery_before,
                "delta_mastery": 0.0,
            },
            "learning_outcome": {
                "attempt_id": self.context.attempt_id,
                "skill": self.context.skill,
                "mastery_before": self.context.mastery_before,
                "mastery_after": self.context.mastery_before,
                "delta_mastery": 0.0,
            },
            "resolved_events": [],
            "knowledge_graph_result": {
                "session_id": (
                    f"{self.context.attempt_id}:dialogue:{kwargs['turn_index']}"
                ),
                "skills_updated": [],
            },
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

    def test_dialogue_batches_advance_mastery_without_replaying_prior_turns(self):
        source = _AssessmentEvidenceSource("Skill A")

        class Graph:
            def __init__(self):
                self.mastery = 0.2

            def get_current_mastery_probability(self, *args, **kwargs):
                del args, kwargs
                return self.mastery

        class IncrementalCrossSession:
            def __init__(self):
                self.knowledge_graph = Graph()
                self.calls = []

            def start_attempt(self, *, student_id, attempt_id, attempt_skill):
                return SimpleNamespace(
                    student_id=student_id,
                    attempt_id=attempt_id,
                    skill=attempt_skill,
                    mastery_before=self.knowledge_graph.mastery,
                )

            def process_transcript(self, **kwargs):
                self.calls.append(copy.deepcopy(kwargs))
                extraction = source.extract(kwargs["transcript"])
                events = []
                for event in extraction["events"]:
                    verdict = source(event, kwargs["transcript"])
                    if verdict["correctness"] == "correct":
                        self.knowledge_graph.mastery += 0.1
                    elif verdict["correctness"] == "incorrect":
                        self.knowledge_graph.mastery -= 0.05
                    events.append(
                        SimpleNamespace(
                            bkt_update=SimpleNamespace(should_update=True)
                        )
                    )
                return {
                    "resolved_events": events,
                    "knowledge_graph_result": {
                        "session_id": kwargs["session_id"],
                        "skills_updated": [],
                    },
                }

        cross_session = IncrementalCrossSession()
        pipeline = TutorAssessmentStudentModelPipeline(
            target_skill="Skill A",
            cross_session_pipeline=cross_session,
            evidence_source=source,
        )
        pipeline.start_attempt(
            student_id="student-a",
            attempt_id="thread-a:1",
            attempt_skill="Skill A",
        )
        dialogue = pipeline.process_dialogue_turn(
            student_id="student-a",
            attempt_id="thread-a:1",
            turn_index=1,
            teacher_text="What is x if x + 2 = 5?",
            student_text="x is 3",
            correctness="correct",
            evaluator_confidence=0.5,
        )
        completed = pipeline.process_assessment_cycle(
            student_id="student-a",
            attempt_id="thread-a:1",
            evaluated_answers=[
                {"question": f"Question {index}", "student_answer": "yes", "is_correct": True}
                for index in range(1, 4)
            ],
        )

        self.assertAlmostEqual(
            dialogue["incremental_learning_outcome"]["mastery_after"],
            0.3,
        )
        self.assertEqual(len(cross_session.calls), 2)
        self.assertTrue(
            all(call["attempt_context"] is None for call in cross_session.calls)
        )
        self.assertEqual(
            [len(call["transcript"]) for call in cross_session.calls],
            [2, 6],
        )
        self.assertAlmostEqual(completed["learning_outcome"]["mastery_before"], 0.2)
        self.assertAlmostEqual(completed["learning_outcome"]["mastery_after"], 0.6)
        self.assertEqual(
            completed["dialogue_evidence_summary"],
            {"turns_processed": 1, "observations_applied": 1},
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
        self.adaptive_pipelines: list[AdaptiveTutorPipeline] = []
        self.md6_selectors: list[RecordingMD6] = []
        self.cached_evidence_by_thread: dict[str, list[dict[str, str]]] = {}

        def factory(**identity):
            memory_adapter = TutorStateMemoryAdapter()
            md6 = RecordingMD6()
            adaptive = AdaptiveTutorPipeline(
                md6=md6,
                mrb1=RecordingMRB1(),
                controller=TurnLevelAttemptController(self.policy),
                experience_logger=self.logger,
                policy_state_path=self.temp_path / "policy.json",
                memory_adapter=memory_adapter,
            )
            self.adaptive_pipelines.append(adaptive)
            self.md6_selectors.append(md6)
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
        first_started_at = started["attempt_started_at"]
        self.assertEqual(
            datetime.fromisoformat(first_started_at.replace("Z", "+00:00")).utcoffset(),
            datetime.fromisoformat("2026-01-01T00:00:00+00:00").utcoffset(),
        )
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
        self.assertNotEqual(completed["attempt_started_at"], first_started_at)
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

    def test_research_lifecycle_rows_bind_start_status_and_full_provenance(self):
        move_order = ["generic", "probing", "focus", "telling"]
        collector = PassiveTurnCollector(
            self.temp_path / "research",
            data_mode="synthetic",
            version_provenance={
                "runtime": {
                    "selector_mode": "ordinary-md7r1-v1",
                    "selector_deployment_status": "active_runtime",
                    "learner_agency_version": (
                        "learner_agency_telling_escalation_v1"
                    ),
                    "lints_policy_lineage": "MD7-R1 fresh LinTS v1",
                    "lints_policy_version": "true_disjoint_lints_v3",
                    "c3_version": "turn_lints_v3_c3_9d",
                    "tau": 0.10,
                    "move_order": move_order,
                    "resolver_version": "2.0",
                    "bkt_config_version": "confidence_weighted_bkt_v1",
                    "bkt_config_sha256": "bkt-sha",
                },
                "selector": {
                    "checkpoint": "md7r1_epoch3",
                    "model_sha256": "md7-sha",
                    "move_order": move_order,
                },
                "tutor_quality": {
                    "model_version": "frozen_mrb1",
                    "model_sha256": "mrb1-sha",
                },
                "policy": {
                    "policy_lineage": "MD7-R1 fresh LinTS v1",
                    "policy_state_relative_path": (
                        "adaptive-math-tutor/backend/runtime/policy_state.json"
                    ),
                },
            },
        )
        coordinator = AdaptiveComponentCoordinator(
            component_factory=self.factory,
            skill_validator=lambda skill: skill in {"Skill A", "Skill B"},
            research_collector=collector,
        )
        state = tutor_state()
        first_start = coordinator.start_attempt(state)
        state.update(first_start)
        turn = coordinator.run_tutor_turn(state, tutor_agent=RecordingTutor())
        state.update(turn)
        completed = coordinator.finish_attempt(
            assessment_state(state, needs_reteaching=True)
        )
        second_start = completed["attempt_started_at"]
        self.assertNotEqual(first_start["attempt_started_at"], second_start)

        next_state = assessment_state(state, needs_reteaching=True)
        next_state.update(completed)
        coordinator.abort_attempt(next_state)

        assessment = json.loads(
            collector.assessment_path.read_text(encoding="utf-8").splitlines()[0]
        )
        summaries = [
            json.loads(line)
            for line in collector.attempt_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(
            assessment["attempt_started_at"], first_start["attempt_started_at"]
        )
        self.assertEqual(
            assessment["provenance"]["active_selector"]["move_order"],
            move_order,
        )
        self.assertEqual(
            [row["completion_status"] for row in summaries],
            ["completed", "aborted"],
        )
        self.assertEqual(
            [row["attempt_started_at"] for row in summaries],
            [first_start["attempt_started_at"], second_start],
        )
        self.assertNotIn("policy_state_path", summaries[0]["adaptive_completion"])
        self.assertEqual(
            summaries[0]["provenance"]["selector_mode"],
            "ordinary-md7r1-v1",
        )
        self.assertEqual(summaries[0]["provenance"]["data_mode"], "synthetic")

    def test_each_action_reads_live_mastery_but_retains_attempt_start_mastery(self):
        state = tutor_state()
        state.update(self.coordinator.start_attempt(state))

        first = self.coordinator.run_tutor_turn(
            state,
            tutor_agent=RecordingTutor(),
        )
        self.mastery_store[("student-a", "Skill A")] = 0.47
        state["turn_count"] = 1
        second = self.coordinator.run_tutor_turn(
            state,
            tutor_agent=RecordingTutor(),
        )

        self.assertAlmostEqual(first["mastery_at_action"], 0.2)
        self.assertAlmostEqual(second["mastery_at_action"], 0.47)
        self.assertAlmostEqual(second["attempt_start_mastery"], 0.2)
        self.assertEqual(second["action_turn_index"], 2)
        self.coordinator.abort_attempt(state)

    def test_controlled_live_completion_processes_bkt_but_aborts_policy(self):
        observed_bkt_activity = []
        manual = AdaptiveComponentCoordinator(
            component_factory=self.factory,
            skill_validator=lambda skill: skill in {"Skill A", "Skill B"},
            suppress_policy_completion=True,
            bkt_activity_observer=lambda **activity: observed_bkt_activity.append(
                activity
            ),
        )
        state = tutor_state()
        state.update(manual.start_attempt(state))
        turn = manual.run_tutor_turn(state, tutor_agent=RecordingTutor())
        state.update(turn)
        completed = manual.finish_attempt(
            assessment_state(state, needs_reteaching=False)
        )

        result = completed["adaptive_completion_result"]
        self.assertTrue(result["controlled_live"])
        self.assertEqual(result["policy_completion"], "aborted_without_update")
        self.assertTrue(result["policy_update_suppressed"])
        self.assertTrue(result["experience_log_write_suppressed"])
        self.assertTrue(result["policy_state_write_suppressed"])
        self.assertEqual(self.policy.total_updates, 0)
        self.assertFalse((self.temp_path / "attempts.jsonl").exists())
        self.assertFalse((self.temp_path / "policy.json").exists())
        self.assertEqual(self.student_pipelines[0].process_calls, 1)
        self.assertEqual(len(observed_bkt_activity), 1)
        self.assertEqual(observed_bkt_activity[0]["student_id"], "student-a")
        self.assertEqual(observed_bkt_activity[0]["attempt_id"], "thread-a:1")
        self.assertEqual(observed_bkt_activity[0]["target_skill"], "Skill A")
        self.assertEqual(len(observed_bkt_activity[0]["evaluated_answers"]), 3)
        duplicate = manual.finish_attempt(
            assessment_state(state, needs_reteaching=False)
        )
        self.assertEqual(duplicate, completed)
        self.assertEqual(len(observed_bkt_activity), 1)
        self.assertFalse(manual.has_active_attempt("thread-a"))

    def test_dialogue_bkt_is_confidence_capped_and_retry_idempotent(self):
        observed = []
        coordinator = AdaptiveComponentCoordinator(
            component_factory=self.factory,
            skill_validator=lambda skill: skill in {"Skill A", "Skill B"},
            dialogue_bkt_activity_observer=lambda **activity: observed.append(
                activity
            ),
        )
        state = tutor_state()
        state.update(coordinator.start_attempt(state))
        coordinator.run_tutor_turn(state, tutor_agent=RecordingTutor())
        state["turn_count"] = 1
        state["conversation_history"].append(
            {"role": "student", "content": "x is 3"}
        )

        first = coordinator.process_dialogue_turn(
            state,
            correctness="correct",
            evaluator_confidence=0.95,
            evaluator_reason="The response solves the requested step.",
        )
        replay = coordinator.process_dialogue_turn(
            state,
            correctness="correct",
            evaluator_confidence=0.95,
            evaluator_reason="The response solves the requested step.",
        )

        self.assertEqual(first, replay)
        pipeline = self.student_pipelines[0]
        self.assertEqual(len(pipeline.dialogue_calls), 1)
        self.assertEqual(
            pipeline.dialogue_calls[0]["evaluator_confidence"],
            DIALOGUE_EVALUATOR_CONFIDENCE_CAP,
        )
        self.assertEqual(len(observed), 1)
        self.assertEqual(
            observed[0]["dialogue_evidence"]["student_text"],
            "x is 3",
        )
        coordinator.abort_attempt(state)

    def test_temporary_tutor_failure_reuses_one_sampled_decision_on_resume(self):
        state = tutor_state()
        state.update(self.coordinator.start_attempt(state))
        tutor = FailOnceTutor()

        with self.assertRaisesRegex(RuntimeError, "Temporary tutor failure"):
            self.coordinator.run_tutor_turn(state, tutor_agent=tutor)

        self.assertTrue(self.coordinator.has_active_attempt("thread-a"))
        self.assertEqual(len(self.md6_selectors[0].calls), 1)
        result = self.coordinator.run_tutor_turn(state, tutor_agent=tutor)

        self.assertEqual(tutor.calls, 2)
        self.assertEqual(len(self.md6_selectors[0].calls), 1)
        self.assertEqual(result["turn_index"], 1)
        self.assertEqual(
            len(self.adaptive_pipelines[0].controller.completed_turns),
            1,
        )
        self.assertEqual(self.policy.total_updates, 0)
        self.coordinator.abort_attempt(state)

    def test_temporary_mrb1_failure_reuses_tutor_response_on_resume(self):
        state = tutor_state()
        state.update(self.coordinator.start_attempt(state))
        tutor = RecordingTutor()
        scorer = FailOnceMRB1()
        self.adaptive_pipelines[0].mrb1 = scorer

        with self.assertRaisesRegex(RuntimeError, "Temporary MRB1 failure"):
            self.coordinator.run_tutor_turn(state, tutor_agent=tutor)

        self.assertTrue(self.coordinator.has_active_attempt("thread-a"))
        result = self.coordinator.run_tutor_turn(state, tutor_agent=tutor)

        self.assertEqual(len(tutor.calls), 1)
        self.assertEqual(len(self.md6_selectors[0].calls), 1)
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual(result["turn_index"], 1)
        self.assertEqual(
            len(self.adaptive_pipelines[0].controller.completed_turns),
            1,
        )
        self.assertEqual(self.policy.total_updates, 0)
        self.coordinator.abort_attempt(state)

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
