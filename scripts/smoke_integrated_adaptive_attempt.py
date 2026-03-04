"""Final isolated smoke test for the integrated adaptive-attempt lifecycle.

This script intentionally exercises production implementations without writing
to the production student database, policy state, or experience log.  Only the
TutorAgent is replaced with a deterministic offline stub; the smoke test is
about scientific integration boundaries rather than generated prose quality.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "adaptive-math-tutor" / "backend"
STUDENT_MODEL_ROOT = WORKSPACE_ROOT / "student-modeling"
MOVE_SELECTOR_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"

for import_root in (BACKEND_ROOT, STUDENT_MODEL_ROOT, MOVE_SELECTOR_ROOT):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


from app.integrations.adaptive_component_coordinator import (  # noqa: E402
    AdaptiveAttemptComponents,
    AdaptiveComponentCoordinator,
    TutorAssessmentStudentModelPipeline,
)
from app.integrations.tutor_state_memory_adapter import (  # noqa: E402
    TutorStateMemoryAdapter,
)
from app.schemas.dialogue import DialogueTurn  # noqa: E402
from bkt.predict import BKTPredictor  # noqa: E402
from core.cross_session_pipeline import (  # noqa: E402
    CrossSessionStudentModelPipeline,
)
from core.detector_service import DetectorService  # noqa: E402
from core.evaluation_contract import EvaluationVerdict  # noqa: E402
import core.knowledge_graph as knowledge_graph_module  # noqa: E402
from core.knowledge_graph import KnowledgeGraph  # noqa: E402
from core.signal_resolver import Correctness  # noqa: E402
from core.student_answer_evaluator import (  # noqa: E402
    EvaluatorContext,
    StudentAnswerEvaluator,
)
from db.database import get_connection, initialise_database  # noqa: E402
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.experience_logger import ExperienceLogger  # noqa: E402
from src.self_improvement.lints_policy import TrueDisjointLinTS  # noqa: E402
from src.self_improvement.md6_inference import FrozenMD6Inference  # noqa: E402
from src.self_improvement.mrb1_inference import FrozenMRB1Inference  # noqa: E402
from src.self_improvement.turn_context_builder import (  # noqa: E402
    TURN_FEATURE_NAMES,
)
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


TARGET_SKILL = "Percent Of"
TUTOR_TURNS = 2
FLOAT_TOLERANCE = 1e-9

ASSESSMENT_ITEMS: tuple[dict[str, str], ...] = (
    {
        "question_id": "q1",
        "question": "What is 20 percent of 50?",
        "reference_answer": "10",
        "student_answer": "The answer is 10.",
    },
    {
        "question_id": "q2",
        "question": "What is 25 percent of 80?",
        "reference_answer": "20",
        "student_answer": "I think the answer is 20.",
    },
    {
        "question_id": "q3",
        "question": "What is 10 percent of 90?",
        "reference_answer": "9",
        "student_answer": "I think the answer is 8.",
    },
)


class SmokeFailure(AssertionError):
    """Raised with actual/expected evidence for a failed smoke invariant."""


def require(
    condition: bool,
    label: str,
    *,
    actual: object | None = None,
    expected: object | None = None,
) -> None:
    if condition:
        return
    print(f"CHECK FAILED: {label}")
    print(f"actual:   {actual!r}")
    print(f"expected: {expected!r}")
    raise SmokeFailure(label)


def require_close(
    actual: float,
    expected: float,
    label: str,
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        label,
        actual=actual,
        expected=expected,
    )


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


class _FailIfGeminiIsUsed:
    """Guard proving numeric objective scoring did not use a network fallback."""

    class _Models:
        @staticmethod
        def generate_content(**_: object) -> object:
            raise SmokeFailure(
                "StudentAnswerEvaluator unexpectedly requested Gemini fallback"
            )

    models = _Models()


def objective_evaluator(question: str, reference_answer: str) -> StudentAnswerEvaluator:
    return StudentAnswerEvaluator(
        EvaluatorContext(
            problem=question,
            reference_answer=reference_answer,
            assessed_skills=(TARGET_SKILL,),
        ),
        client=_FailIfGeminiIsUsed(),
    )


class ObjectiveAssessmentEvidenceSource:
    """Test-only adapter from three objective items to the real evaluator contract."""

    def __init__(self, target_skill: str) -> None:
        self.target_skill = target_skill
        self.context = SimpleNamespace(assessed_skills=(target_skill,))
        self._reference_by_question = {
            item["question"]: item["reference_answer"]
            for item in ASSESSMENT_ITEMS
        }
        self._evaluators: dict[int, StudentAnswerEvaluator] = {}
        self._expected_correctness: dict[int, bool] = {}
        self.verdicts: dict[int, EvaluationVerdict] = {}

    def build_transcript(
        self,
        evaluated_answers: Sequence[Mapping[str, object]],
    ) -> list[dict[str, str]]:
        transcript: list[dict[str, str]] = []
        self._evaluators.clear()
        self._expected_correctness.clear()
        self.verdicts.clear()

        require(
            len(evaluated_answers) == 3,
            "assessment must contain exactly three evaluated answers",
            actual=len(evaluated_answers),
            expected=3,
        )

        for index, answer in enumerate(evaluated_answers):
            question = answer.get("question")
            student_answer = answer.get("student_answer")
            is_correct = answer.get("is_correct")
            require(
                isinstance(question, str) and question in self._reference_by_question,
                f"assessment question {index} is recognized",
                actual=question,
                expected=tuple(self._reference_by_question),
            )
            require(
                isinstance(student_answer, str) and bool(student_answer.strip()),
                f"assessment answer {index} is non-empty",
                actual=student_answer,
                expected="non-empty string",
            )
            require(
                isinstance(is_correct, bool),
                f"assessment answer {index} has a boolean host verdict",
                actual=is_correct,
                expected="bool",
            )

            transcript.append({"role": "tutor", "text": question})
            turn_index = len(transcript)
            transcript.append(
                {"role": "student", "text": student_answer.strip()}
            )
            self._evaluators[turn_index] = objective_evaluator(
                question,
                self._reference_by_question[question],
            )
            self._expected_correctness[turn_index] = is_correct

        return transcript

    def extract(self, transcript: list[dict[str, object]]) -> dict[str, object]:
        return {
            "events": [
                {
                    "turn_index": turn_index,
                    "skill": self.target_skill,
                    "evidence_span": str(transcript[turn_index]["text"]),
                }
                for turn_index in sorted(self._evaluators)
            ],
            "misconceptions": [],
        }

    def __call__(
        self,
        event: dict[str, Any],
        transcript: list[dict[str, Any]],
    ) -> EvaluationVerdict:
        turn_index = event.get("turn_index")
        require(
            isinstance(turn_index, int) and turn_index in self._evaluators,
            "extracted event maps to one assessment evaluator",
            actual=turn_index,
            expected=tuple(sorted(self._evaluators)),
        )
        verdict = self._evaluators[turn_index](event, transcript)
        expected_correct = self._expected_correctness[turn_index]
        observed_correct = verdict.correctness == Correctness.CORRECT
        require(
            verdict.correctness in {Correctness.CORRECT, Correctness.INCORRECT},
            "objective evaluator returns assessable correctness",
            actual=verdict.correctness.value,
            expected="correct or incorrect",
        )
        require(
            observed_correct == expected_correct,
            "host assessment verdict matches StudentAnswerEvaluator",
            actual=observed_correct,
            expected=expected_correct,
        )
        self.verdicts[turn_index] = verdict
        return verdict


class RecordingStudentModelPipeline(TutorAssessmentStudentModelPipeline):
    """Record public boundary values without changing production behavior."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.started_contexts: list[object] = []
        self.results: list[Mapping[str, object]] = []

    def start_attempt(self, **kwargs: object) -> object:
        context = super().start_attempt(**kwargs)
        self.started_contexts.append(context)
        return context

    def process_assessment_cycle(self, **kwargs: object) -> Mapping[str, object]:
        result = super().process_assessment_cycle(**kwargs)
        self.results.append(result)
        return result


class DeterministicTutorAgentStub:
    """Offline response generator; MD6 still controls the pedagogical move."""

    def __init__(self) -> None:
        self.calls = 0

    def teach_turn(self, tutor_input: object, verified_math_evidence: object) -> object:
        del verified_math_evidence
        self.calls += 1
        move = getattr(tutor_input, "pedagogical_move")
        messages = (
            "How would you convert the percentage into a decimal before multiplying?",
            "Using that decimal, what multiplication would you perform next?",
        )
        require(
            self.calls <= len(messages),
            "TutorAgent stub call count",
            actual=self.calls,
            expected=f"at most {len(messages)}",
        )
        return SimpleNamespace(
            teaching_response=f"[{move}] {messages[self.calls - 1]}"
        )


def assessment_verdicts() -> list[dict[str, object]]:
    verdicts: list[dict[str, object]] = []
    for item in ASSESSMENT_ITEMS:
        evaluator = objective_evaluator(
            item["question"],
            item["reference_answer"],
        )
        transcript = [
            {"role": "tutor", "text": item["question"]},
            {"role": "student", "text": item["student_answer"]},
        ]
        verdict = evaluator(
            {
                "turn_index": 1,
                "skill": TARGET_SKILL,
                "student_text": item["student_answer"],
                "evidence_span": item["student_answer"],
            },
            transcript,
        )
        require(
            verdict.correctness in {Correctness.CORRECT, Correctness.INCORRECT},
            "preflight objective assessment is binary",
            actual=verdict.correctness.value,
            expected="correct or incorrect",
        )
        verdicts.append(
            {
                **item,
                "is_correct": verdict.correctness == Correctness.CORRECT,
                "confidence": verdict.confidence,
                "source": verdict.source,
            }
        )
    require(
        sum(bool(item["is_correct"]) for item in verdicts) == 2,
        "assessment preflight is two correct and one incorrect",
        actual=[item["is_correct"] for item in verdicts],
        expected=[True, True, False],
    )
    return verdicts


def build_assessment_state(
    state: Mapping[str, object],
    verdicts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result = copy.deepcopy(dict(state))
    questions = [
        {
            "question_id": item["question_id"],
            "question": item["question"],
        }
        for item in verdicts
    ]
    evaluated = [
        {
            "question_id": item["question_id"],
            "question": item["question"],
            "student_answer": item["student_answer"],
            "is_correct": item["is_correct"],
        }
        for item in verdicts
    ]
    result.update(
        {
            "assessment_questions": questions,
            "student_answers": [
                {
                    "question_id": item["question_id"],
                    "answer": item["student_answer"],
                }
                for item in verdicts
            ],
            "evaluation_result": {
                "correct_answers": [
                    item for item in evaluated if item["is_correct"]
                ],
                "wrong_answers": [
                    item for item in evaluated if not item["is_correct"]
                ],
                "identified_errors": ["Incorrect percentage calculation"],
                "needs_reteaching": True,
                "overall_feedback": "Two of three objective answers were correct.",
            },
            "needs_reteaching": True,
        }
    )
    return result


def table_count(database_path: Path, table: str, student_id: str) -> int:
    allowed = {"attempts", "mastery", "sessions"}
    require(
        table in allowed,
        "table_count allowlist",
        actual=table,
        expected=sorted(allowed),
    )
    with get_connection(database_path) as connection:
        row = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE student_id = ?",
            (student_id,),
        ).fetchone()
    return int(row[0])


def prior_rows(
    database_path: Path,
    student_id: str,
    skill: str,
) -> list[tuple[float | None, str]]:
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT effective_initial_prior, prior_source
            FROM bkt_initial_priors
            WHERE student_id = ? AND skill_name = ?
            """,
            (student_id, skill),
        ).fetchall()
    return [
        (
            None if row["effective_initial_prior"] is None else float(
                row["effective_initial_prior"]
            ),
            str(row["prior_source"]),
        )
        for row in rows
    ]


def experience_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def initial_state(
    *,
    student_id: str,
    thread_id: str,
    attempt_id: str,
) -> dict[str, object]:
    return {
        "student_id": student_id,
        "thread_id": thread_id,
        "attempt_id": attempt_id,
        "adaptive_attempt_index": 1,
        "target_skill": TARGET_SKILL,
        "adaptive_lifecycle_status": "not_started",
        "question": "A shirt costs 50 dollars. What is 20 percent of its price?",
        "topic": "Percentages",
        "subtopic": "Percent of a quantity",
        "age": 14,
        "complexity_score": 0.35,
        "planner_output": {},
        "previous_errors": [],
        "teaching_phase": "initial",
        "verified_math_evidence": [
            {"action": "final_claim_verification", "result": "true"}
        ],
        "conversation_history": [],
        "turn_count": 0,
    }


def append_student_turn(state: dict[str, object], response: str) -> None:
    history = state.get("conversation_history")
    require(
        isinstance(history, list),
        "TutorState conversation history is mutable",
        actual=type(history).__name__,
        expected="list",
    )
    history.append(DialogueTurn(role="student", content=response).model_dump())


def run_smoke(temp_root: Path) -> dict[str, object]:
    student_id = f"smoke_integrated_student_{uuid4().hex[:12]}"
    thread_id = f"smoke_integrated_thread_{uuid4().hex[:12]}"

    student_database = temp_root / "student-model" / "meta_agent.db"
    policy_path = temp_root / "policy" / "frozen_v3_policy_state.json"
    experience_path = temp_root / "policy" / "frozen_v3_experience.jsonl"

    print("Loading real production artifacts...")
    predictor = BKTPredictor.load(
        STUDENT_MODEL_ROOT / "models" / "bkt_params.json"
    )
    detectors = DetectorService.from_project_defaults(STUDENT_MODEL_ROOT)
    md6 = FrozenMD6Inference()
    mrb1 = FrozenMRB1Inference()
    policy = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=42,
        data_mode="synthetic",
    )
    logger = ExperienceLogger(
        experience_path,
        data_mode="synthetic",
        source_policy=policy,
    )

    print("TutorAgent: TEST STUB")
    print("SQLite: TEMPORARY ISOLATED DATABASE")
    print(f"temporary root: {temp_root}")

    sources: list[ObjectiveAssessmentEvidenceSource] = []
    pipelines: list[RecordingStudentModelPipeline] = []

    with (
        patch.object(
            knowledge_graph_module,
            "initialise_database",
            lambda: initialise_database(student_database),
        ),
        patch.object(
            knowledge_graph_module,
            "get_connection",
            lambda: get_connection(student_database),
        ),
    ):
        knowledge_graph = KnowledgeGraph(predictor=predictor)

        def component_factory(**identity: object) -> AdaptiveAttemptComponents:
            source = ObjectiveAssessmentEvidenceSource(
                str(identity["target_skill"])
            )
            cross_session = CrossSessionStudentModelPipeline(
                concept_extractor=source,
                evaluator=source,
                detectors=detectors,
                knowledge_graph=knowledge_graph,
                curriculum=None,
            )
            pipeline = RecordingStudentModelPipeline(
                target_skill=str(identity["target_skill"]),
                cross_session_pipeline=cross_session,
                evidence_source=source,
            )
            memory_adapter = TutorStateMemoryAdapter()
            adaptive_pipeline = AdaptiveTutorPipeline(
                md6=md6,
                mrb1=mrb1,
                controller=TurnLevelAttemptController(policy),
                experience_logger=logger,
                policy_state_path=policy_path,
                memory_adapter=memory_adapter,
            )
            bridge = StudentModelFrozenV3Bridge(
                student_model_pipeline=pipeline,
                adaptive_pipeline=adaptive_pipeline,
            )
            sources.append(source)
            pipelines.append(pipeline)
            return AdaptiveAttemptComponents(
                bridge=bridge,
                student_model_pipeline=pipeline,
                memory_adapter=memory_adapter,
            )

        coordinator = AdaptiveComponentCoordinator(
            component_factory=component_factory,
            skill_validator=predictor.has_skill,
        )
        tutor = DeterministicTutorAgentStub()

        require(
            predictor.has_skill(TARGET_SKILL),
            "target skill is in the real BKT vocabulary",
            actual=TARGET_SKILL,
            expected="trained skill",
        )
        attempt_1_id = coordinator.derive_attempt_id(thread_id, 1)
        attempt_2_id = coordinator.derive_attempt_id(thread_id, 2)
        state = initial_state(
            student_id=student_id,
            thread_id=thread_id,
            attempt_id=attempt_1_id,
        )

        start_policy_updates = policy.total_updates
        state.update(coordinator.start_attempt(state))
        mastery_before = float(state["mastery_before"])
        require(
            0.0 <= mastery_before <= 1.0,
            "mastery_before is a probability",
            actual=mastery_before,
            expected="[0, 1]",
        )

        require(len(pipelines) == 1, "one attempt-1 student pipeline exists")
        attempt_1_pipeline = pipelines[0]
        require(
            len(attempt_1_pipeline.started_contexts) == 1,
            "student model captured exactly one attempt-1 start context",
            actual=len(attempt_1_pipeline.started_contexts),
            expected=1,
        )
        student_context = attempt_1_pipeline.started_contexts[0]
        active_context = coordinator._contexts[thread_id]
        bridge_attempt = active_context.bridge.active_attempt
        require(bridge_attempt is not None, "bridge attempt is active")
        selector_attempt_id = active_context.bridge.adaptive_pipeline.active_attempt_id

        prior_at_start = knowledge_graph.get_effective_initial_prior(
            student_id,
            TARGET_SKILL,
        )
        require(prior_at_start is not None, "effective initial prior exists")
        effective_initial_prior = float(
            prior_at_start["effective_initial_prior"]
        )
        initial_prior_provenance = str(prior_at_start["prior_source"])

        start_counts = {
            "attempts": table_count(student_database, "attempts", student_id),
            "mastery": table_count(student_database, "mastery", student_id),
            "sessions": table_count(student_database, "sessions", student_id),
        }
        require(
            start_counts == {"attempts": 0, "mastery": 0, "sessions": 0},
            "attempt start creates no BKT attempt, mastery, or session row",
            actual=start_counts,
            expected={"attempts": 0, "mastery": 0, "sessions": 0},
        )
        require(
            not attempt_1_pipeline.results,
            "attempt start creates no completed learning outcome",
            actual=len(attempt_1_pipeline.results),
            expected=0,
        )
        require(
            policy.total_updates == start_policy_updates,
            "attempt start creates no LinTS update",
            actual=policy.total_updates,
            expected=start_policy_updates,
        )

        print("\nATTEMPT 1 START")
        print(f"student_id: {student_id}")
        print(f"attempt_id: {attempt_1_id}")
        print(f"target_skill: {TARGET_SKILL}")
        print(f"mastery_before: {mastery_before:.12f}")
        print(f"effective_initial_prior: {effective_initial_prior:.12f}")
        print(f"initial_prior_provenance: {initial_prior_provenance}")
        print("Start is read-only: PASS")

        student_responses = (
            "I would rewrite twenty percent as 0.2.",
            "Then I would multiply 0.2 by 50.",
        )
        turn_results: list[dict[str, object]] = []
        for turn_number, student_response in enumerate(student_responses, start=1):
            result = coordinator.run_tutor_turn(state, tutor_agent=tutor)
            state.update(result)
            turn_results.append(dict(result))
            append_student_turn(state, student_response)

            print(f"\nTUTOR TURN {turn_number}")
            print("MD6 probabilities:")
            print_json(result["md6_probabilities"])
            print(f"eligible moves after conservative overlay: {result['eligible_arms']}")
            print(f"selected pedagogical move: {result['pedagogical_move']}")
            print(f"TutorAgent response: {result['tutor_response']}")
            print("MRB1 scores:")
            print_json(result["mrb1_scores"])
            print(f"LinTS posterior update count: {policy.total_updates}")

            require(
                policy.total_updates == start_policy_updates,
                f"turn {turn_number} performs no within-attempt LinTS update",
                actual=policy.total_updates,
                expected=start_policy_updates,
            )
            require(
                table_count(student_database, "attempts", student_id) == 0,
                f"turn {turn_number} creates no BKT observation",
                actual=table_count(student_database, "attempts", student_id),
                expected=0,
            )
            require(
                table_count(student_database, "mastery", student_id) == 0,
                f"turn {turn_number} creates no mastery row",
                actual=table_count(student_database, "mastery", student_id),
                expected=0,
            )

        print("\nWITHIN-ATTEMPT UPDATE CHECK: PASS")
        policy_updates_before_completion = policy.total_updates
        mastery_before_completion = knowledge_graph.get_mastery(
            student_id,
            TARGET_SKILL,
        )
        require(
            mastery_before_completion is None,
            "MD6, MRB1, Tutor, and selected moves did not recalculate mastery",
            actual=mastery_before_completion,
            expected=None,
        )

        verdicts = assessment_verdicts()
        assessment_state = build_assessment_state(state, verdicts)
        first_completion = coordinator.finish_attempt(assessment_state)

        require(
            len(attempt_1_pipeline.results) == 1,
            "exactly one student-model result exists for attempt 1",
            actual=len(attempt_1_pipeline.results),
            expected=1,
        )
        student_model_result = attempt_1_pipeline.results[0]
        learning_outcome = student_model_result.get("learning_outcome")
        require(
            isinstance(learning_outcome, Mapping),
            "pipeline produced one learning_outcome mapping",
            actual=type(learning_outcome).__name__,
            expected="Mapping",
        )
        learning_outcome = dict(learning_outcome)

        evaluated_events = list(
            student_model_result["evaluated_extraction"]["events"]
        )
        resolved_events = list(student_model_result["resolved_events"])
        require(
            len(evaluated_events) == 3 and len(resolved_events) == 3,
            "exactly three assessment events reached the resolver",
            actual=(len(evaluated_events), len(resolved_events)),
            expected=(3, 3),
        )

        print("\n3-QUESTION ASSESSMENT")
        for index, (item, evaluated, resolved) in enumerate(
            zip(ASSESSMENT_ITEMS, evaluated_events, resolved_events, strict=True),
            start=1,
        ):
            print(f"\nQuestion {index}")
            print(f"question: {item['question']}")
            print(f"student answer: {item['student_answer']}")
            print(f"evaluator correctness: {evaluated['correctness']}")
            print(f"evaluator confidence: {evaluated['evaluator_confidence']:.12f}")
            print(f"evaluator source: {evaluated['evaluator_source']}")
            print(
                "reasoning probability: "
                f"{resolved.behaviour.reasoning_probability:.12f}"
            )
            print(
                "uncertainty probability: "
                f"{resolved.behaviour.uncertainty_probability:.12f}"
            )
            print(
                "clarification probability: "
                f"{resolved.behaviour.clarification_probability:.12f}"
            )
            print(f"resolved primary signal: {resolved.primary_signal.value}")
            print(f"BKT should_update: {resolved.bkt_update.should_update}")
            print(f"BKT outcome: {resolved.bkt_update.outcome}")
            print(
                "BKT update_confidence: "
                f"{resolved.bkt_update.update_confidence:.12f}"
            )

        require(
            learning_outcome["attempt_id"] == attempt_1_id,
            "learning outcome attempt ID",
            actual=learning_outcome["attempt_id"],
            expected=attempt_1_id,
        )
        require(
            learning_outcome["skill"] == TARGET_SKILL,
            "learning outcome target skill",
            actual=learning_outcome["skill"],
            expected=TARGET_SKILL,
        )
        calculated_delta = float(learning_outcome["mastery_after"]) - float(
            learning_outcome["mastery_before"]
        )
        require_close(
            float(learning_outcome["delta_mastery"]),
            calculated_delta,
            "learning outcome delta arithmetic",
        )

        print("\nATTEMPT 1 learning_outcome")
        print_json(learning_outcome)
        print("ONE ATTEMPT-LEVEL OUTCOME: PASS")
        print("DELTA ARITHMETIC: PASS")

        bkt_observations = sum(
            1 for event in resolved_events if event.bkt_update.should_update
        )
        adaptive_learning_outcomes = len(attempt_1_pipeline.results)
        records = experience_records(experience_path)
        require(
            adaptive_learning_outcomes == 1 and len(records) == 1,
            "three answers create one adaptive reward record",
            actual={
                "pipeline_outcomes": adaptive_learning_outcomes,
                "experience_records": len(records),
            },
            expected={"pipeline_outcomes": 1, "experience_records": 1},
        )
        print("\nAssessment answers: 3")
        print(f"BKT observations: {bkt_observations}")
        print(f"Adaptive learning outcomes: {adaptive_learning_outcomes}")

        adaptive_completion = first_completion["adaptive_completion_result"]
        selector_reward = float(adaptive_completion["reward"])
        delta_mastery = float(learning_outcome["delta_mastery"])
        require_close(
            selector_reward,
            delta_mastery,
            "selector reward equals student-model delta",
        )
        print(f"\nstudent-model delta: {delta_mastery:.12f}")
        print(f"selector reward: {selector_reward:.12f}")
        print("REWARD PARITY: PASS")

        policy_updates_after_completion = policy.total_updates
        require(
            policy_updates_before_completion == start_policy_updates,
            "posterior updates before completion remain zero",
            actual=policy_updates_before_completion,
            expected=start_policy_updates,
        )
        require(
            policy_updates_after_completion - policy_updates_before_completion
            == TUTOR_TURNS,
            "completion creates one weighted update per tutor turn",
            actual=(
                policy_updates_after_completion
                - policy_updates_before_completion
            ),
            expected=TUTOR_TURNS,
        )
        require_close(
            float(adaptive_completion["total_attempt_weight"]),
            1.0,
            "delayed-credit total normalized weight",
            tolerance=1e-12,
        )
        require_close(
            float(adaptive_completion["sample_weight_per_turn"]),
            1.0 / TUTOR_TURNS,
            "delayed-credit per-turn sample weight",
            tolerance=1e-12,
        )
        print(
            "posterior updates before completion: "
            f"{policy_updates_before_completion}"
        )
        print(
            "posterior updates after completion: "
            f"{policy_updates_after_completion}"
        )
        print(
            "sample weight per turn: "
            f"{float(adaptive_completion['sample_weight_per_turn']):.12f}"
        )
        print("DELAYED POLICY UPDATE: PASS")

        require(
            first_completion["completed_attempt_id"] == attempt_1_id,
            "coordinator completed attempt 1",
            actual=first_completion["completed_attempt_id"],
            expected=attempt_1_id,
        )
        require(
            first_completion["attempt_id"] == attempt_2_id,
            "failed assessment starts attempt 2",
            actual=first_completion["attempt_id"],
            expected=attempt_2_id,
        )
        attempt_2_mastery_before = float(first_completion["mastery_before"])
        attempt_1_mastery_after = float(learning_outcome["mastery_after"])
        require_close(
            attempt_2_mastery_before,
            attempt_1_mastery_after,
            "cross-attempt mastery continuity",
        )
        require(
            len(pipelines) == 2 and len(pipelines[1].started_contexts) == 1,
            "attempt 2 has exactly one start context",
            actual=(len(pipelines), len(pipelines[1].started_contexts)),
            expected=(2, 1),
        )
        attempt_2_context = pipelines[1].started_contexts[0]
        require(
            getattr(attempt_2_context, "attempt_id") == attempt_2_id,
            "attempt 2 student-model context identity",
            actual=getattr(attempt_2_context, "attempt_id"),
            expected=attempt_2_id,
        )
        print(f"\nATTEMPT 1 mastery_after: {attempt_1_mastery_after:.12f}")
        print(f"ATTEMPT 2 mastery_before: {attempt_2_mastery_before:.12f}")
        print("CROSS-ATTEMPT CONTINUITY: PASS")

        prior_after_attempt_2_start = knowledge_graph.get_effective_initial_prior(
            student_id,
            TARGET_SKILL,
        )
        rows = prior_rows(student_database, student_id, TARGET_SKILL)
        require(
            len(rows) == 1,
            "exactly one effective-initial-prior row exists",
            actual=rows,
            expected="one row",
        )
        require(prior_after_attempt_2_start is not None, "attempt-2 prior exists")
        require_close(
            float(prior_after_attempt_2_start["effective_initial_prior"]),
            effective_initial_prior,
            "attempt 2 reuses the persisted effective initial prior",
            tolerance=1e-12,
        )
        require(
            prior_after_attempt_2_start["prior_source"]
            == initial_prior_provenance,
            "attempt 2 preserves initial-prior provenance",
            actual=prior_after_attempt_2_start["prior_source"],
            expected=initial_prior_provenance,
        )
        print(f"\neffective_initial_prior: {effective_initial_prior:.12f}")
        print(f"provenance: {initial_prior_provenance}")
        print("INITIAL PRIOR PERSISTENCE: PASS")

        complete_history = knowledge_graph.get_attempts(student_id, TARGET_SKILL)
        stored_mastery_record = knowledge_graph.get_mastery(
            student_id,
            TARGET_SKILL,
        )
        require(stored_mastery_record is not None, "persisted mastery exists")
        stored_mastery = float(stored_mastery_record["mastery_probability"])
        independent_prediction = predictor.predict(
            TARGET_SKILL,
            complete_history,
            initial_prior=effective_initial_prior,
        )
        require_close(
            stored_mastery,
            independent_prediction,
            "persisted mastery equals independent real-BKT prediction",
        )
        raw_fraction = 2.0 / 3.0
        require(
            not math.isclose(
                stored_mastery,
                raw_fraction,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "two correct answers out of three did not become raw 2/3 mastery",
            actual=stored_mastery,
            expected="real BKT result, not 0.666666...",
        )
        print(f"\nstored mastery: {stored_mastery:.12f}")
        print(f"independent real-BKT prediction: {independent_prediction:.12f}")
        print("BKT FINAL-STATE PARITY: PASS")

        finish_attempt_id = str(adaptive_completion["attempt_id"])
        identity_values = {
            "selector": selector_attempt_id,
            "student_model_context": getattr(student_context, "attempt_id"),
            "bridge": bridge_attempt.attempt_id,
            "learning_outcome": learning_outcome["attempt_id"],
            "finish_attempt": finish_attempt_id,
        }
        require(
            set(identity_values.values()) == {attempt_1_id},
            "shared attempt identity across repositories",
            actual=identity_values,
            expected=attempt_1_id,
        )
        require(
            getattr(student_context, "student_id")
            == bridge_attempt.student_id
            == student_id,
            "shared student identity",
            actual=(
                getattr(student_context, "student_id"),
                bridge_attempt.student_id,
                student_id,
            ),
            expected=student_id,
        )
        require(
            getattr(student_context, "skill")
            == bridge_attempt.target_skill
            == learning_outcome["skill"]
            == TARGET_SKILL,
            "shared target-skill identity",
            actual=(
                getattr(student_context, "skill"),
                bridge_attempt.target_skill,
                learning_outcome["skill"],
            ),
            expected=TARGET_SKILL,
        )
        print("\nSHARED ATTEMPT IDENTITY: PASS")

        counts_before_unauthorized = {
            "attempts": table_count(student_database, "attempts", student_id),
            "mastery": table_count(student_database, "mastery", student_id),
        }
        unauthorized_rejected = False
        try:
            coordinator.validate_target_skill("smoke_unauthorized_skill")
        except KeyError:
            unauthorized_rejected = True
        require(
            unauthorized_rejected,
            "unauthorized skill is rejected",
            actual=unauthorized_rejected,
            expected=True,
        )
        counts_after_unauthorized = {
            "attempts": table_count(student_database, "attempts", student_id),
            "mastery": table_count(student_database, "mastery", student_id),
        }
        require(
            counts_after_unauthorized == counts_before_unauthorized,
            "unauthorized skill creates no student-model writes",
            actual=counts_after_unauthorized,
            expected=counts_before_unauthorized,
        )
        require(
            len(complete_history) == bkt_observations,
            "move selector did not create or recompute BKT observations",
            actual=len(complete_history),
            expected=bkt_observations,
        )

        return {
            "learning_outcome": learning_outcome,
            "attempt_2_mastery_before": attempt_2_mastery_before,
            "selector_reward": selector_reward,
            "policy_updates_before": policy_updates_before_completion,
            "policy_updates_after": policy_updates_after_completion,
            "effective_initial_prior": effective_initial_prior,
            "initial_prior_provenance": initial_prior_provenance,
            "stored_mastery": stored_mastery,
            "independent_prediction": independent_prediction,
        }


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="adaptmath-integrated-smoke-"
    ) as temporary_directory:
        report = run_smoke(Path(temporary_directory))
        # Windows will not delete SQLite files while even an unreachable
        # connection object still owns a native handle.  The smoke fixtures
        # are already out of scope here; collect them before temp cleanup.
        gc.collect()

    summary = (
        ("Shared attempt identity", True),
        ("Attempt start read-only", True),
        ("No within-attempt BKT completion", True),
        ("No within-attempt LinTS update", True),
        ("3-question assessment", True),
        ("One learning outcome", True),
        ("No per-answer adaptive rewards", True),
        ("Delta arithmetic", True),
        ("Move-selector reward parity", True),
        ("Delayed policy update", True),
        ("Initial-prior persistence", True),
        ("Cross-attempt mastery continuity", True),
        ("BKT final-state parity", True),
        ("Unauthorized-skill isolation", True),
    )

    print("\nREPORT VALUES")
    print("attempt_1 learning_outcome:")
    print_json(report["learning_outcome"])
    print(
        "attempt_2 mastery_before: "
        f"{float(report['attempt_2_mastery_before']):.12f}"
    )
    print(f"selector reward: {float(report['selector_reward']):.12f}")
    print(
        "posterior update counts before/after completion: "
        f"{report['policy_updates_before']}/{report['policy_updates_after']}"
    )
    print(
        "persisted initial prior + provenance: "
        f"{float(report['effective_initial_prior']):.12f} / "
        f"{report['initial_prior_provenance']}"
    )
    print(
        "BKT final-state parity values: "
        f"{float(report['stored_mastery']):.12f} / "
        f"{float(report['independent_prediction']):.12f}"
    )

    print("\n============================================================")
    print("INTEGRATED ADAPTIVE TUTOR SMOKE TEST")
    print("============================================================")
    print()
    for label, passed in summary:
        print(f"{label:<36}{'PASS' if passed else 'FAIL'}")
    print()
    print("OVERALL: PASS")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n============================================================")
        print("INTEGRATED ADAPTIVE TUTOR SMOKE TEST")
        print("============================================================")
        print(f"OVERALL: FAIL ({type(exc).__name__}: {exc})")
        print("============================================================")
        raise
