from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from app.integrations.adaptive_component_coordinator import (
    _ProductionAdaptiveResources,
)
from app.integrations.adaptive_selector_mode import (
    AdaptiveSelectorMode,
    SELECTOR_MODE_ENV,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.conservative_overlay import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD,
)
from src.self_improvement.experience_logger import (  # noqa: E402
    CREDIT_SCHEME,
    ExperienceLogger,
    REWARD_NAME,
    SCHEMA_VERSION,
    UPDATE_TIMING,
)
from src.self_improvement.lints_policy import (  # noqa: E402
    ARMS,
    DEFAULT_EXPLORATION_SCALE,
    DEFAULT_RIDGE_LAMBDA,
    LINTS_STATE_SCHEMA_VERSION,
    TrueDisjointLinTS,
)
from src.self_improvement.md6_inference import MAX_LENGTH  # noqa: E402
from src.self_improvement.turn_context_builder import (  # noqa: E402
    MRB1_TASKS,
    TURN_FEATURE_NAMES,
)
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


MD6_PROBABILITIES = {
    "generic": 0.02,
    "probing": 0.94,
    "focus": 0.02,
    "telling": 0.02,
}
MD7_PROBABILITIES = {
    "generic": 0.02,
    "probing": 0.02,
    "focus": 0.94,
    "telling": 0.02,
}


class FixtureSelector:
    def __init__(self, probabilities: dict[str, float], model_dir=None) -> None:
        self.probabilities = probabilities
        self.model_dir = model_dir
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict_probabilities(self, problem, conversation_history):
        self.calls.append((problem, copy.deepcopy(list(conversation_history))))
        return dict(self.probabilities)


class FixtureLoader:
    def __init__(self) -> None:
        self.paths: list[Path | None] = []

    def __call__(self, model_dir=None):
        path = None if model_dir is None else Path(model_dir)
        self.paths.append(path)
        probabilities = MD6_PROBABILITIES if path is None else MD7_PROBABILITIES
        return FixtureSelector(probabilities, model_dir=path)


class FixtureTutor:
    def generate(self, *, problem, conversation_history, pedagogical_move):
        del problem, conversation_history
        return f"Offline tutor used {pedagogical_move}."


class FixtureMRB1:
    def score_response(self, conversation_history, tutor_response):
        del conversation_history, tutor_response
        return {task: 0.75 for task in MRB1_TASKS}


def _run_completion(
    *,
    selector,
    policy_path: Path,
    experience_path: Path,
) -> tuple[TrueDisjointLinTS, dict[str, object], dict[str, object]]:
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    pipeline = AdaptiveTutorPipeline(
        md6=selector,
        mrb1=FixtureMRB1(),
        controller=TurnLevelAttemptController(policy),
        experience_logger=ExperienceLogger(
            experience_path,
            data_mode="synthetic",
            source_policy=policy,
        ),
        policy_state_path=policy_path,
    )
    memory = {
        "attempt_id": "promotion-fixture:1",
        "problem": "What is 25 percent of 80?",
        "conversation_history": [
            {"user": "student", "text": "I think it is 12."}
        ],
        "mastery_before": 0.30,
    }
    pipeline.start_attempt(memory)
    first_turn = pipeline.run_tutor_turn(memory, FixtureTutor())
    memory["conversation_history"].append(
        {"user": "student", "text": "Maybe I should multiply."}
    )
    pipeline.run_tutor_turn(memory, FixtureTutor())
    completion = pipeline.finish_attempt(
        memory,
        {
            "skill": "Percent Of",
            "mastery_before": 0.30,
            "mastery_after": 0.50,
            "delta_mastery": 0.20,
        },
    )
    return policy, first_turn, completion


def _configure_lineages(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "md6_policy": tmp_path / "md6" / "policy_state.json",
        "md6_log": tmp_path / "md6" / "attempts.jsonl",
        "md7_policy": tmp_path / "md7" / "policy_state.json",
        "md7_log": tmp_path / "md7" / "attempts.jsonl",
        "turn_lints_policy": tmp_path / "turn-lints" / "policy_state.json",
        "turn_lints_events": tmp_path / "turn-lints",
    }
    monkeypatch.setenv("ADAPTIVE_MD6_POLICY_STATE_PATH", str(paths["md6_policy"]))
    monkeypatch.setenv("ADAPTIVE_MD6_EXPERIENCE_LOG_PATH", str(paths["md6_log"]))
    monkeypatch.setenv("ADAPTIVE_MD7_POLICY_STATE_PATH", str(paths["md7_policy"]))
    monkeypatch.setenv("ADAPTIVE_MD7_EXPERIENCE_LOG_PATH", str(paths["md7_log"]))
    monkeypatch.setenv(
        "ADAPTIVE_TURN_LINTS_STATE_PATH", str(paths["turn_lints_policy"])
    )
    monkeypatch.setenv(
        "ADAPTIVE_TURN_LINTS_EVENT_ROOT", str(paths["turn_lints_events"])
    )
    monkeypatch.setenv("ADAPTIVE_TURN_LINTS_MODE", "SHADOW")
    monkeypatch.delenv("ADAPTIVE_MANUAL_CONTROLLED_LIVE", raising=False)
    return paths


def test_scientific_settings_are_unchanged():
    assert len(TURN_FEATURE_NAMES) == 9
    assert TURN_FEATURE_NAMES[:4] == (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
    )
    assert DEFAULT_GAP_THRESHOLD == 0.10
    assert DEFAULT_RIDGE_LAMBDA == 1.0
    assert DEFAULT_EXPLORATION_SCALE == 0.20
    assert ARMS == (
        "baseline",
        "generic_bias",
        "probing_bias",
        "focus_bias",
        "telling_bias",
    )
    assert REWARD_NAME == "mastery_delta"
    assert CREDIT_SCHEME == "equal_normalized"
    assert UPDATE_TIMING == "after attempt completion only"
    assert SCHEMA_VERSION == "turn_lints_v3"
    assert LINTS_STATE_SCHEMA_VERSION == "true_disjoint_lints_v3"
    assert MAX_LENGTH == 512


def test_normal_md7_turn_lints_does_not_infer_legacy_attempt_policy_paths(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(SELECTOR_MODE_ENV, raising=False)
    monkeypatch.delenv("ADAPTIVE_MANUAL_CONTROLLED_LIVE", raising=False)
    turn_state = tmp_path / "turn-lints" / "policy_state.json"
    turn_events = tmp_path / "turn-lints"
    monkeypatch.setenv("ADAPTIVE_TURN_LINTS_MODE", "SHADOW")
    monkeypatch.setenv("ADAPTIVE_TURN_LINTS_STATE_PATH", str(turn_state))
    monkeypatch.setenv("ADAPTIVE_TURN_LINTS_EVENT_ROOT", str(turn_events))
    monkeypatch.setenv(
        "ADAPTIVE_POLICY_STATE_PATH",
        str(tmp_path / "legacy-policy.json"),
    )
    monkeypatch.setenv(
        "ADAPTIVE_EXPERIENCE_LOG_PATH",
        str(tmp_path / "legacy-attempts.jsonl"),
    )

    resources = _ProductionAdaptiveResources()

    assert resources.selector_mode is AdaptiveSelectorMode.NORMAL_MD7
    assert resources.policy_lineage == "direct_disjoint_turn_lints_v1"
    assert resources.policy_state_path == turn_state
    assert resources.experience_log_path == turn_events / "turn_events.jsonl"
    assert resources.policy_state_path != tmp_path / "legacy-policy.json"


def test_normal_md7_uses_promoted_selector_and_isolated_turn_lints_lineage(
    monkeypatch,
    tmp_path,
):
    paths = _configure_lineages(monkeypatch, tmp_path)
    paths["md6_policy"].parent.mkdir(parents=True)
    paths["md6_policy"].write_bytes(b"preserved-md6-policy\n")
    paths["md6_log"].write_bytes(b"preserved-md6-log\n")
    md6_before = {
        name: paths[name].read_bytes() for name in ("md6_policy", "md6_log")
    }
    monkeypatch.delenv(SELECTOR_MODE_ENV, raising=False)

    resources = _ProductionAdaptiveResources()
    loader = FixtureLoader()
    selector = resources._build_base_selector(loader)
    probabilities = selector.predict_probabilities("Fixture problem", [])

    assert resources.selector_mode is AdaptiveSelectorMode.NORMAL_MD7
    assert loader.paths == [resources.md7_model_dir]
    assert resources.policy_lineage == "direct_disjoint_turn_lints_v1"
    assert resources.turn_lints_mode == "SHADOW"
    assert resources.policy_state_path == paths["turn_lints_policy"]
    assert resources.experience_log_path == paths["turn_lints_events"] / "turn_events.jsonl"
    assert probabilities == MD7_PROBABILITIES
    assert max(MD7_PROBABILITIES, key=MD7_PROBABILITIES.__getitem__) == "focus"
    assert not paths["turn_lints_policy"].exists()
    assert not (paths["turn_lints_events"] / "turn_events.jsonl").exists()
    assert {
        name: paths[name].read_bytes() for name in ("md6_policy", "md6_log")
    } == md6_before
    assert not paths["md7_policy"].exists()
    assert not paths["md7_log"].exists()


def test_md6_rollback_pairs_selector_with_preserved_md6_lineage(
    monkeypatch,
    tmp_path,
):
    paths = _configure_lineages(monkeypatch, tmp_path)
    paths["md7_policy"].parent.mkdir(parents=True)
    paths["md7_policy"].write_bytes(b"fresh-md7-policy\n")
    paths["md7_log"].write_bytes(b"")
    md7_before = {
        name: paths[name].read_bytes() for name in ("md7_policy", "md7_log")
    }
    monkeypatch.setenv(
        SELECTOR_MODE_ENV,
        AdaptiveSelectorMode.MD6_ROLLBACK.value,
    )

    resources = _ProductionAdaptiveResources()
    loader = FixtureLoader()
    selector = resources._build_base_selector(loader)
    policy, first_turn, completion = _run_completion(
        selector=selector,
        policy_path=resources.policy_state_path,
        experience_path=resources.experience_log_path,
    )

    assert resources.selector_mode is AdaptiveSelectorMode.MD6_ROLLBACK
    assert loader.paths == [None]
    assert resources.policy_state_path == paths["md6_policy"]
    assert resources.experience_log_path == paths["md6_log"]
    assert first_turn["md6_probabilities"] == MD6_PROBABILITIES
    assert policy.total_updates == 2
    assert completion["total_attempt_weight"] == pytest.approx(1.0)
    assert paths["md6_policy"].is_file()
    assert paths["md6_log"].is_file()
    assert {
        name: paths[name].read_bytes() for name in ("md7_policy", "md7_log")
    } == md7_before
