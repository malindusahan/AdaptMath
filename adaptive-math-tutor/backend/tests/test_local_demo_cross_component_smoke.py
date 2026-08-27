from __future__ import annotations

import copy
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.response_verifier import TeachingResponseVerifier
from app.agents.tutor.tutor_agent import TutorAgent
from app.core.config import get_settings
from app.graph import workflow
from app.integrations.adaptive_component_coordinator import (
    _AssessmentEvidenceSource,
    AdaptiveAttemptComponents,
    AdaptiveComponentCoordinator,
    TutorAssessmentStudentModelPipeline,
)
from app.integrations.tutor_state_memory_adapter import TutorStateMemoryAdapter


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STUDENT_ROOT = WORKSPACE_ROOT / "student-modeling"
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
for import_root in (STUDENT_ROOT, MOVE_ROOT):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from bkt.predict import BKTPredictor  # noqa: E402
from core.cross_session_pipeline import (  # noqa: E402
    CrossSessionStudentModelPipeline,
)
from core.detector_service import DetectorService  # noqa: E402
import core.knowledge_graph as knowledge_graph_module  # noqa: E402
from core.knowledge_graph import KnowledgeGraph  # noqa: E402
from db.database import get_connection, initialise_database  # noqa: E402
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.experience_logger import ExperienceLogger  # noqa: E402
from src.self_improvement.lints_policy import TrueDisjointLinTS  # noqa: E402
from src.self_improvement.turn_context_builder import (  # noqa: E402
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
)
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


TARGET_SKILL = "Equation Solving Two or Fewer Steps"


class DeterministicMD6:
    def __init__(self) -> None:
        self.histories: list[list[dict[str, object]]] = []

    def predict_probabilities(self, problem, conversation_history):
        assert problem == "Solve x + 2 = 5."
        self.histories.append(copy.deepcopy(list(conversation_history)))
        return {
            "generic": 0.97,
            "probing": 0.01,
            "focus": 0.01,
            "telling": 0.01,
        }


class DeterministicMRB1:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], str]] = []

    def score_response(self, conversation_history, tutor_response):
        self.calls.append(
            (copy.deepcopy(list(conversation_history)), tutor_response)
        )
        score = (0.25, 0.75, 0.5)[len(self.calls) - 1]
        return {task: score for task in MRB1_TASKS}


class MockGeminiCompletions:
    def __init__(self) -> None:
        self.teacher_turns = 0
        self.audits = 0

    @staticmethod
    def _response(payload: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )

    def create(self, **kwargs):
        schema = kwargs["response_format"]["json_schema"]
        if schema["name"] == "adaptmath_atomic_teacher_turn":
            self.teacher_turns += 1
            move = schema["schema"]["properties"]["selected_move"]["enum"][0]
            return self._response(
                {
                    "selected_move": move,
                    "immediate_target": "the learner's next idea",
                    "learner_action": "describe one next step",
                    "teacher_message": (
                        f"What would you try next? (demo turn {self.teacher_turns})"
                    ),
                    "reveals_future_steps": False,
                    "multiple_independent_requests": False,
                    "blends_multiple_pedagogical_actions": False,
                }
            )
        if schema["name"] == "adaptmath_teaching_response_audit":
            self.audits += 1
            return self._response(
                {
                    "checkable_claims": [],
                    "logical_issues": [],
                    "turn_scope_issues": [],
                    "turn_scope_verdict": "pass",
                    "move_alignment_verdict": "pass",
                    "scope_reason": "One atomic learner action.",
                    "move_alignment_reason": "The selected move is preserved.",
                }
            )
        raise AssertionError(f"Unexpected mocked Gemini schema: {schema['name']}")


def _mocked_real_tutor() -> tuple[TutorAgent, MockGeminiCompletions]:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "synthetic-test-key"}):
        get_settings.cache_clear()
        tutor = TutorAgent()
    get_settings.cache_clear()

    completions = MockGeminiCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    tutor.client = client
    tutor.response_verifier = TeachingResponseVerifier(
        client=client,
        model_name=tutor.model_name,
        math_verifier=SymPyMathVerifier(),
    )
    return tutor, completions


def _state(thread_id: str) -> dict[str, object]:
    return {
        "student_id": "synthetic-demo-student",
        "thread_id": thread_id,
        "attempt_id": f"{thread_id}:1",
        "adaptive_attempt_index": 1,
        "target_skill": TARGET_SKILL,
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
        "conversation_history": [],
        "turn_count": 0,
    }


def _append_student_response(state: dict[str, object], response: str) -> None:
    with patch.object(workflow, "interrupt", return_value={"response": response}):
        state.update(workflow.await_student_response_node(state))


def _assessment(
    state: dict[str, object],
    *,
    is_correct: bool,
    needs_reteaching: bool,
) -> dict[str, object]:
    result = copy.deepcopy(state)
    questions = [
        {"question_id": f"q{i}", "question": f"Demo question {i}?"}
        for i in range(1, 4)
    ]
    answers = [
        {
            "question_id": f"q{i}",
            "question": f"Demo question {i}?",
            "student_answer": "synthetic answer",
            "is_correct": is_correct,
        }
        for i in range(1, 4)
    ]
    result.update(
        {
            "assessment_questions": questions,
            "student_answers": [
                {"question_id": f"q{i}", "answer": "synthetic answer"}
                for i in range(1, 4)
            ],
            "evaluation_result": {
                "correct_answers": answers if is_correct else [],
                "wrong_answers": [] if is_correct else answers,
                "identified_errors": [] if is_correct else ["synthetic error"],
                "needs_reteaching": needs_reteaching,
                "overall_feedback": "synthetic evaluation",
            },
            "needs_reteaching": needs_reteaching,
        }
    )
    return result


def test_real_cross_component_two_attempt_smoke(tmp_path: Path) -> None:
    student_db = tmp_path / "student_model" / "meta_agent.db"
    policy_path = tmp_path / "policy_state.json"
    experience_path = tmp_path / "attempts.jsonl"

    predictor = BKTPredictor.load(STUDENT_ROOT / "models" / "bkt_params.json")
    detectors = DetectorService.from_project_defaults(STUDENT_ROOT)
    policy = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=42,
        data_mode="synthetic",
    )
    md6 = DeterministicMD6()
    mrb1 = DeterministicMRB1()
    logger = ExperienceLogger(
        experience_path,
        data_mode="synthetic",
        source_policy=policy,
    )
    pipelines: list[TutorAssessmentStudentModelPipeline] = []

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                knowledge_graph_module,
                "initialise_database",
                lambda: initialise_database(student_db),
            )
        )
        stack.enter_context(
            patch.object(
                knowledge_graph_module,
                "get_connection",
                lambda: get_connection(student_db),
            )
        )
        knowledge_graph = KnowledgeGraph(predictor=predictor)

        def factory(**identity) -> AdaptiveAttemptComponents:
            source = _AssessmentEvidenceSource(identity["target_skill"])
            cross_session = CrossSessionStudentModelPipeline(
                concept_extractor=source,
                evaluator=source,
                detectors=detectors,
                knowledge_graph=knowledge_graph,
                curriculum=None,
            )
            student_pipeline = TutorAssessmentStudentModelPipeline(
                target_skill=identity["target_skill"],
                cross_session_pipeline=cross_session,
                evidence_source=source,
            )
            pipelines.append(student_pipeline)
            memory = TutorStateMemoryAdapter()
            adaptive = AdaptiveTutorPipeline(
                md6=md6,
                mrb1=mrb1,
                controller=TurnLevelAttemptController(policy),
                experience_logger=logger,
                policy_state_path=policy_path,
                memory_adapter=memory,
            )
            return AdaptiveAttemptComponents(
                bridge=StudentModelFrozenV3Bridge(
                    student_model_pipeline=student_pipeline,
                    adaptive_pipeline=adaptive,
                ),
                student_model_pipeline=student_pipeline,
                memory_adapter=memory,
            )

        coordinator = AdaptiveComponentCoordinator(
            component_factory=factory,
            skill_validator=predictor.has_skill,
        )
        tutor, completions = _mocked_real_tutor()
        state = _state("demo-thread")
        with (
            patch.object(workflow, "adaptive_coordinator", coordinator),
            patch.object(workflow, "tutor_agent", tutor),
        ):
            state.update(workflow.start_adaptive_attempt_node(state))
            attempt_1_id = state["attempt_id"]
            attempt_1_before = float(state["mastery_before"])

            first_turn = workflow.adaptive_tutor_turn_node(state)
            state.update(first_turn)
            context_after_turn_1 = coordinator._contexts[state["thread_id"]]
            assert context_after_turn_1.status == "active"
            assert context_after_turn_1.attempt_id == attempt_1_id
            assert (
                context_after_turn_1.bridge.adaptive_pipeline.active_attempt_id
                == attempt_1_id
            )
            assert context_after_turn_1.bridge.adaptive_pipeline.controller.is_active

            _append_student_response(state, "I would subtract two.")
            with patch.object(
                workflow.progress_agent,
                "judge",
                return_value=SimpleNamespace(
                    status="continue_teaching",
                    reason="The learner should take another conversational turn.",
                ),
            ):
                state.update(workflow.teaching_progress_node(state))
            assert workflow.after_teaching_progress(state) == "adaptive_turn"

            second_turn = workflow.adaptive_tutor_turn_node(state)
            state.update(second_turn)

        context_after_turn_2 = coordinator._contexts[state["thread_id"]]
        assert context_after_turn_2 is context_after_turn_1
        assert context_after_turn_2.status == "active"
        assert state["attempt_id"] == attempt_1_id
        assert context_after_turn_2.attempt_id == attempt_1_id
        assert (
            context_after_turn_2.bridge.adaptive_pipeline.active_attempt_id
            == attempt_1_id
        )
        assert context_after_turn_2.bridge.adaptive_pipeline.controller.is_active
        assert [turn["role"] for turn in state["conversation_history"]] == [
            "teacher",
            "student",
            "teacher",
        ]
        assert state["conversation_history"][0]["content"] != state[
            "conversation_history"
        ][2]["content"]
        _append_student_response(state, "That gives x equals three.")

        first_assessment = _assessment(
            state,
            is_correct=False,
            needs_reteaching=True,
        )
        first_completion = coordinator.finish_attempt(first_assessment)
        duplicate_first = coordinator.finish_attempt(first_assessment)
        assert duplicate_first == first_completion
        attempt_1_after = float(
            first_completion["adaptive_completion_result"]["mastery_after"]
        )

        state.update(first_completion)
        state["teaching_phase"] = "reteaching"
        attempt_2_id = state["attempt_id"]
        attempt_2_before = float(state["mastery_before"])
        assert state["thread_id"] == "demo-thread"
        assert state["target_skill"] == TARGET_SKILL
        assert attempt_2_id != attempt_1_id
        assert attempt_2_before == attempt_1_after

        reteaching_turn = coordinator.run_tutor_turn(state, tutor_agent=tutor)
        state.update(reteaching_turn)
        _append_student_response(state, "I corrected the equation.")
        second_assessment = _assessment(
            state,
            is_correct=True,
            needs_reteaching=False,
        )
        second_completion = coordinator.finish_attempt(second_assessment)
        duplicate_second = coordinator.finish_attempt(second_assessment)
        assert duplicate_second == second_completion

        assert first_turn["pedagogical_move"] == "generic"
        assert second_turn["pedagogical_move"] == "generic"
        assert reteaching_turn["pedagogical_move"] == "generic"
        assert first_turn["adaptive_turn_diagnostics"]["context"][4:] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        assert second_turn["adaptive_turn_diagnostics"]["context"][4:] == [
            0.25,
            0.25,
            0.25,
            0.25,
            1.0,
        ]

        teacher_turns = [
            turn for turn in state["conversation_history"]
            if turn["role"] == "teacher"
        ]
        student_turns = [
            turn for turn in state["conversation_history"]
            if turn["role"] == "student"
        ]
        assert len(teacher_turns) == 3
        assert len(student_turns) == 3
        assert completions.teacher_turns == 3
        assert completions.audits == 3
        assert len(mrb1.calls) == 3
        for history, current_response in mrb1.calls:
            assert all(item["text"] != current_response for item in history)

        with get_connection(student_db) as connection:
            observation_count = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE student_id = ?",
                ("synthetic-demo-student",),
            ).fetchone()[0]
        assert observation_count == 6
        assert policy.total_updates == 3
        records = [
            json.loads(line)
            for line in experience_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["attempt_id"] for record in records] == [
            attempt_1_id,
            attempt_2_id,
        ]
        assert policy_path.exists()
        assert len(pipelines) == 2
        assert attempt_1_before == first_completion["adaptive_completion_result"][
            "mastery_before"
        ]
        print(
            "SMOKE_MASTERY "
            f"attempt_1_before={attempt_1_before:.12f} "
            f"attempt_1_after={attempt_1_after:.12f} "
            f"attempt_2_before={attempt_2_before:.12f}"
        )
