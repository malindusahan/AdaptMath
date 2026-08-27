"""Real-model smoke validation of the working adaptive tutor pipeline."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.self_improvement.adaptive_tutor_pipeline import AdaptiveTutorPipeline
from src.self_improvement.experience_logger import ExperienceLogger
from src.self_improvement.lints_policy import TrueDisjointLinTS
from src.self_improvement.md6_inference import FrozenMD6Inference
from src.self_improvement.mrb1_inference import FrozenMRB1Inference
from src.self_improvement.state_io import load_policy_state
from src.self_improvement.turn_context_builder import MRB1_TASKS, TURN_FEATURE_NAMES
from src.self_improvement.turn_level_controller import TurnLevelAttemptController


class CapturingMD6(FrozenMD6Inference):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> dict[str, float]:
        self.calls.append(
            {
                "problem": problem,
                "conversation_history": copy.deepcopy(list(conversation_history)),
            }
        )
        return super().predict_probabilities(problem, conversation_history)


class CapturingMRB1(FrozenMRB1Inference):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def score_response(
        self,
        conversation_history: Sequence[Mapping[str, object]],
        tutor_response: str,
    ) -> dict[str, float]:
        self.calls.append(
            {
                "conversation_history": copy.deepcopy(list(conversation_history)),
                "tutor_response": tutor_response,
            }
        )
        return super().score_response(conversation_history, tutor_response)


class CapturingTutorAgent:
    """Test-only capture stub; production tutor behavior remains external."""

    RESPONSES = {
        "generic": "Let's work through the quantities carefully.",
        "probing": "What operation connects equal groups to the total?",
        "focus": "Focus on the three equal groups of four marbles.",
        "telling": "Multiply 3 by 4 to find the total number of marbles.",
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
        pedagogical_move: str,
    ) -> str:
        self.calls.append(
            {
                "problem": problem,
                "conversation_history": copy.deepcopy(list(conversation_history)),
                "pedagogical_move": pedagogical_move,
            }
        )
        return self.RESPONSES[pedagogical_move]


def _posterior_snapshot(
    policy: TrueDisjointLinTS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], int]:
    return (
        {arm: matrix.copy() for arm, matrix in policy.A.items()},
        {arm: vector.copy() for arm, vector in policy.b.items()},
        dict(policy.arm_update_counts),
        policy.total_updates,
    )


def _assert_posterior_equal(
    policy: TrueDisjointLinTS,
    snapshot: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, int],
        int,
    ],
) -> None:
    matrices, vectors, counts, total = snapshot
    for arm in policy.arms:
        np.testing.assert_array_equal(policy.A[arm], matrices[arm])
        np.testing.assert_array_equal(policy.b[arm], vectors[arm])
    assert policy.arm_update_counts == counts
    assert policy.total_updates == total


def _valid_probability_mapping(probabilities: object) -> bool:
    if not isinstance(probabilities, Mapping):
        return False
    expected = {"generic", "probing", "focus", "telling"}
    values = [float(probabilities[key]) for key in expected]
    return (
        set(probabilities) == expected
        and all(np.isfinite(value) and value >= 0.0 for value in values)
        and abs(sum(values) - 1.0) <= 1e-5
    )


def main() -> None:
    problem = (
        "A student has 3 bags with 4 marbles in each bag. "
        "How many marbles are there in total?"
    )
    memory: dict[str, object] = {
        "attempt_id": "real-model-smoke-001",
        "problem": problem,
        "conversation_history": [
            {"user": "student", "text": "I added 3 and 4 and got 7."}
        ],
        "mastery_before": 0.42,
    }

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        policy = TrueDisjointLinTS(
            context_dim=len(TURN_FEATURE_NAMES),
            seed=20260824,
            data_mode="real",
        )
        controller = TurnLevelAttemptController(policy)
        logger = ExperienceLogger(
            root / "attempts.jsonl",
            data_mode="real",
            source_policy=policy,
        )
        md6 = CapturingMD6()
        mrb1 = CapturingMRB1()
        tutor = CapturingTutorAgent()
        state_path = root / "policy_state.json"
        pipeline = AdaptiveTutorPipeline(
            md6=md6,
            mrb1=mrb1,
            controller=controller,
            experience_logger=logger,
            policy_state_path=state_path,
        )

        initial_history = copy.deepcopy(memory["conversation_history"])
        start_result = pipeline.start_attempt(memory)
        assert start_result == {
            "attempt_id": "real-model-smoke-001",
            "mastery_before": 0.42,
        }
        assert md6.calls == []
        before_attempt = _posterior_snapshot(policy)

        turn_results: list[dict[str, object]] = []
        history_before_turns: list[list[dict[str, object]]] = []
        student_followups = (
            "Maybe multiplication represents the equal groups.",
            "Three groups of four gives twelve.",
        )

        for turn_index in range(1, 4):
            current_history = copy.deepcopy(memory["conversation_history"])
            assert isinstance(current_history, list)
            history_before_turns.append(current_history)
            result = pipeline.run_tutor_turn(memory, tutor)
            turn_results.append(result)

            assert result["turn_index"] == turn_index
            assert _valid_probability_mapping(result["md6_probabilities"])
            assert len(result["context"]) == 9
            assert result["selected_arm"] in result["eligible_arms"]
            assert result["pedagogical_move"] in {
                "generic",
                "probing",
                "focus",
                "telling",
            }
            assert tutor.calls[-1]["pedagogical_move"] == result[
                "pedagogical_move"
            ]
            assert result["tutor_response"] == mrb1.calls[-1]["tutor_response"]
            assert set(result["mrb1_scores"]) == set(MRB1_TASKS)
            assert all(
                0.0 <= float(value) <= 1.0
                for value in result["mrb1_scores"].values()
            )
            assert memory["conversation_history"][-1] == {
                "user": "teacher",
                "text": result["tutor_response"],
            }
            assert policy.total_updates == 0

            if turn_index <= 2:
                history = memory["conversation_history"]
                assert isinstance(history, list)
                history.append(
                    {"user": "student", "text": student_followups[turn_index - 1]}
                )

        assert md6.calls[0]["problem"] == problem
        assert md6.calls[0]["conversation_history"] == initial_history
        assert [call["conversation_history"] for call in md6.calls] == (
            history_before_turns
        )
        assert [call["conversation_history"] for call in tutor.calls] == (
            history_before_turns
        )
        assert [call["conversation_history"] for call in mrb1.calls] == (
            history_before_turns
        )

        assert turn_results[0]["context"][4:] == [0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(
            turn_results[1]["context"][4:8],
            [turn_results[0]["mrb1_scores"][task] for task in MRB1_TASKS],
            rtol=0.0,
            atol=1e-12,
        )
        assert turn_results[1]["context"][8] == 1.0
        expected_turn_three = [
            (
                turn_results[0]["mrb1_scores"][task]
                + turn_results[1]["mrb1_scores"][task]
            )
            / 2.0
            for task in MRB1_TASKS
        ]
        np.testing.assert_allclose(
            turn_results[2]["context"][4:8],
            expected_turn_three,
            rtol=0.0,
            atol=1e-12,
        )
        assert turn_results[2]["context"][8] == 1.0
        _assert_posterior_equal(policy, before_attempt)

        completion = pipeline.finish_attempt(
            memory,
            evaluator_score=3,
            mastery_after=0.55,
            metadata={"validation": "real-model-offline-smoke"},
        )
        assert completion["evaluator_score"] == 3
        assert completion["reward"] == 1.0
        assert completion["mastery_before"] == 0.42
        assert completion["mastery_after"] == 0.55
        assert abs(completion["mastery_delta"] - 0.13) <= 1e-15
        assert completion["turn_count"] == 3
        assert completion["sample_weight_per_turn"] == 1.0 / 3.0
        assert completion["total_attempt_weight"] == 1.0
        assert policy.total_updates == 3

        log_path = root / "attempts.jsonl"
        assert log_path.is_file()
        records = log_path.read_text(encoding="utf-8").splitlines()
        assert len(records) == 1
        record = json.loads(records[0])
        assert record["schema_version"] == "turn_lints_v2"
        assert record["num_turns"] == 3
        assert record["policy_config"]["context"]["dimension"] == 9
        assert state_path.is_file()
        state_envelope = json.loads(state_path.read_text(encoding="utf-8"))
        assert state_envelope["schema_version"] == "turn_lints_state_v2"
        assert state_envelope["policy_state"]["schema_version"] == (
            "true_disjoint_lints_v2"
        )

        clone = TrueDisjointLinTS(
            context_dim=9,
            seed=1,
            data_mode="real",
        )
        load_policy_state(clone, state_path, expected_data_mode="real")
        assert clone.state_dict() == policy.state_dict()

        abort_memory: dict[str, object] = {
            "attempt_id": "real-model-smoke-abort",
            "problem": problem,
            "conversation_history": [],
            "mastery_before": 0.55,
        }
        before_abort = _posterior_snapshot(policy)
        pipeline.start_attempt(abort_memory)
        pipeline.abort_attempt(abort_memory)
        _assert_posterior_equal(policy, before_abort)

        example_move = str(turn_results[0]["pedagogical_move"])
        print("ADAPTIVE TUTOR PIPELINE REAL-MODEL VALIDATION")
        checks = (
            "attempt inputs read from authoritative memory",
            "real MD6 receives memory problem/history",
            "valid MD6 probabilities reach controller",
            "controller creates exact 9-D context",
            "final pedagogical move reaches tutor agent",
            "final move is a canonical pedagogical move",
            "tutor response is returned",
            "real MRB1 scores the returned response",
            "Turn 1 MRB1 cannot affect Turn 1 choice",
            "Turn 1 MRB1 enters Turn 2 context",
            "running MRB1 mean enters Turn 3 context",
            "later turns use current authoritative history",
            "no within-attempt posterior update",
            "evaluator and mastery completion accepted",
            "delayed updates occur only at completion",
            "v2 experience record is logged",
            "v2 policy state is persisted and reloadable",
            "abort produces no posterior update",
        )
        for index, check in enumerate(checks, start=1):
            print(f"{index}. {check}: PASS")
        print(f"EXAMPLE FINAL MOVE: {example_move}")
        print("OVERALL: PASS")


if __name__ == "__main__":
    main()
