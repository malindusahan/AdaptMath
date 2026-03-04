"""CPU-only promotion validation for MD7-R1 epoch 3.

This script uses explicit model/runtime paths only. It does not discover or
read held-out datasets, invoke a tutor API, train a model, or run a real user
tutoring session.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parents[1]
MOVE_ROOT = WORKSPACE_ROOT / "pedagogical-move-selection"
MD7_MODEL_ROOT = MOVE_ROOT / "models" / "candidates" / "md7r1_epoch3"
MD6_MODEL_ROOT = MOVE_ROOT / "models" / "frozen" / "md6"
MD7_LINEAGE_ROOT = BACKEND_ROOT / "runtime" / "adaptive_demo_md7r1_v1"
MD7_POLICY_PATH = MD7_LINEAGE_ROOT / "policy_state.json"
MD7_EXPERIENCE_PATH = MD7_LINEAGE_ROOT / "attempts.jsonl"
MD6_LINEAGE_ROOT = BACKEND_ROOT / "runtime" / "adaptive_demo"
MD6_POLICY_PATH = MD6_LINEAGE_ROOT / "policy_state.json"
MD6_EXPERIENCE_PATH = MD6_LINEAGE_ROOT / "attempts.jsonl"
PROMOTION_ROOT = MOVE_ROOT / "results" / "md7r1_promotion_v1"
PROMOTION_MANIFEST_PATH = PROMOTION_ROOT / "promotion_manifest.json"

MD7_EXPECTED_HASHES = {
    "config.json": "9877dd11007c8af1d2841a2c1bb470023ef2ceeb748b47c24686ea82479e9127",
    "model.safetensors": "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13",
    "tokenizer.json": "80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b",
    "tokenizer_config.json": "bb5706f71e3d5cb4bf9878ad9b14a2079e6fbd618fc0a0ba2ac1b257d7824cc9",
}
MD6_EXPECTED_HASHES = {
    "config.json": "9877dd11007c8af1d2841a2c1bb470023ef2ceeb748b47c24686ea82479e9127",
    "model.safetensors": "674f1abbaf8c6c23d4b736a80f40b3532e4681c125695ebbe3863e404acb56e7",
    "tokenizer.json": "80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b",
    "tokenizer_config.json": "4903bcd294e8ff8b840eb9c21909d2f910b466d250ac6e298a7438a7da63ef0d",
}

for import_root in (BACKEND_ROOT, MOVE_ROOT):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from app.integrations.adaptive_component_coordinator import (  # noqa: E402
    _ProductionAdaptiveResources,
)
from app.integrations.adaptive_selector_mode import (  # noqa: E402
    AdaptiveSelectorMode,
    SELECTOR_MODE_ENV,
)
from app.integrations.manual_controlled_live import (  # noqa: E402
    MANUAL_CONTROLLED_LIVE_ENV,
    file_snapshot,
)
from scripts.prepare_md7r1_policy_lineage import prepare_lineage  # noqa: E402
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.conservative_overlay import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD,
)
from src.self_improvement.context_builder import MOVE_ORDER  # noqa: E402
from src.self_improvement.controlled_live_selector import (  # noqa: E402
    ActiveMD7WithMD6Shadow,
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
from src.self_improvement.md6_inference import (  # noqa: E402
    MAX_LENGTH,
    FrozenMD6Inference,
)
from src.self_improvement.state_io import (  # noqa: E402
    POLICY_STATE_SCHEMA_VERSION,
    load_policy_state,
    save_policy_state,
)
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
LINEAGE_ENV_NAMES = (
    "ADAPTIVE_POLICY_STATE_PATH",
    "ADAPTIVE_EXPERIENCE_LOG_PATH",
    "ADAPTIVE_MD6_POLICY_STATE_PATH",
    "ADAPTIVE_MD6_EXPERIENCE_LOG_PATH",
    "ADAPTIVE_MD7_POLICY_STATE_PATH",
    "ADAPTIVE_MD7_EXPERIENCE_LOG_PATH",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_snapshot(
    root: Path,
    expected_hashes: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, expected in expected_hashes.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Required model file is missing: {path}")
        stat = path.stat()
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Accepted model hash mismatch for {path}: "
                f"expected={expected}, actual={actual}"
            )
        result[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual,
        }
    return result


@contextmanager
def _environment(
    values: Mapping[str, str],
    *,
    remove: tuple[str, ...] = (),
) -> Iterator[None]:
    names = set(values) | set(remove)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in remove:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class FixtureSelector:
    def __init__(self, probabilities: Mapping[str, float], model_dir=None) -> None:
        self.probabilities = dict(probabilities)
        self.model_dir = model_dir
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict_probabilities(self, problem, conversation_history):
        self.calls.append((problem, copy.deepcopy(list(conversation_history))))
        return dict(self.probabilities)


class FixtureLoader:
    def __init__(self) -> None:
        self.paths: list[Path | None] = []

    def __call__(self, model_dir=None):
        path = None if model_dir is None else Path(model_dir).resolve()
        self.paths.append(path)
        probabilities = MD6_PROBABILITIES if path is None else MD7_PROBABILITIES
        return FixtureSelector(probabilities, model_dir=path)


class FixtureTutor:
    def generate(self, *, problem, conversation_history, pedagogical_move):
        del problem, conversation_history
        return f"Offline fixture tutor used {pedagogical_move}."


class FixtureMRB1:
    def score_response(self, conversation_history, tutor_response):
        del conversation_history, tutor_response
        return {task: 0.75 for task in MRB1_TASKS}


def _run_completion(
    selector: object,
    *,
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
        "attempt_id": "promotion-offline-fixture:1",
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


def _lineage_values(root: Path) -> dict[str, str]:
    return {
        "ADAPTIVE_DATA_MODE": "synthetic",
        "ADAPTIVE_MD6_POLICY_STATE_PATH": str(root / "md6" / "policy_state.json"),
        "ADAPTIVE_MD6_EXPERIENCE_LOG_PATH": str(root / "md6" / "attempts.jsonl"),
        "ADAPTIVE_MD7_POLICY_STATE_PATH": str(root / "md7" / "policy_state.json"),
        "ADAPTIVE_MD7_EXPERIENCE_LOG_PATH": str(root / "md7" / "attempts.jsonl"),
    }


def _save_fresh_policy(path: Path) -> None:
    save_policy_state(
        TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic"),
        path,
    )


def _normal_completion_fixture(root: Path) -> dict[str, object]:
    values = _lineage_values(root)
    md6_policy = Path(values["ADAPTIVE_MD6_POLICY_STATE_PATH"])
    md6_log = Path(values["ADAPTIVE_MD6_EXPERIENCE_LOG_PATH"])
    _save_fresh_policy(md6_policy)
    md6_log.touch()
    md6_before = {
        "policy": file_snapshot(md6_policy),
        "experience_log": file_snapshot(md6_log),
    }
    values[SELECTOR_MODE_ENV] = AdaptiveSelectorMode.NORMAL_MD7.value
    with _environment(
        values,
        remove=(MANUAL_CONTROLLED_LIVE_ENV,) + LINEAGE_ENV_NAMES[:2],
    ):
        resources = _ProductionAdaptiveResources()
        loader = FixtureLoader()
        selector = resources._build_base_selector(loader)
        policy, first_turn, completion = _run_completion(
            selector,
            policy_path=resources.policy_state_path,
            experience_path=resources.experience_log_path,
        )
    md6_after = {
        "policy": file_snapshot(md6_policy),
        "experience_log": file_snapshot(md6_log),
    }
    if md6_before != md6_after:
        raise RuntimeError("Normal MD7 fixture mutated the MD6 fixture lineage.")
    if loader.paths != [MD7_MODEL_ROOT.resolve()]:
        raise RuntimeError(f"Normal mode loaded unexpected selectors: {loader.paths}")
    if first_turn["md6_probabilities"] != MD7_PROBABILITIES:
        raise RuntimeError("Normal C3 did not receive fixture MD7 probabilities.")
    if first_turn["context"][:4] != list(MD7_PROBABILITIES.values()):
        raise RuntimeError("Normal C3 first four fields are not MD7 probabilities.")
    if policy.total_updates != 2:
        raise RuntimeError("Normal completion did not produce two delayed updates.")
    if completion["reward"] != 0.20:
        raise RuntimeError("Normal completion reward is not raw mastery delta 0.20.")
    if completion["sample_weight_per_turn"] != 0.5:
        raise RuntimeError("Normal completion did not use 1/T=0.5 credit.")
    if completion["total_attempt_weight"] != 1.0:
        raise RuntimeError("Normal completion total delayed weight is not 1.")
    return {
        "selector_loads": [str(path) for path in loader.paths],
        "policy_path": str(resources.policy_state_path),
        "experience_log_path": str(resources.experience_log_path),
        "md7_probabilities": first_turn["md6_probabilities"],
        "c3_first_four": first_turn["context"][:4],
        "base_move": first_turn["base_move"],
        "reward": completion["reward"],
        "delayed_updates": policy.total_updates,
        "per_update_weight": completion["sample_weight_per_turn"],
        "total_attempt_weight": completion["total_attempt_weight"],
        "md6_lineage_unchanged": True,
    }


def _rollback_completion_fixture(root: Path) -> dict[str, object]:
    values = _lineage_values(root)
    md7_policy = Path(values["ADAPTIVE_MD7_POLICY_STATE_PATH"])
    md7_log = Path(values["ADAPTIVE_MD7_EXPERIENCE_LOG_PATH"])
    _save_fresh_policy(md7_policy)
    md7_log.touch()
    md7_before = {
        "policy": file_snapshot(md7_policy),
        "experience_log": file_snapshot(md7_log),
    }
    values[SELECTOR_MODE_ENV] = AdaptiveSelectorMode.MD6_ROLLBACK.value
    with _environment(
        values,
        remove=(MANUAL_CONTROLLED_LIVE_ENV,) + LINEAGE_ENV_NAMES[:2],
    ):
        resources = _ProductionAdaptiveResources()
        loader = FixtureLoader()
        selector = resources._build_base_selector(loader)
        policy, first_turn, completion = _run_completion(
            selector,
            policy_path=resources.policy_state_path,
            experience_path=resources.experience_log_path,
        )
    md7_after = {
        "policy": file_snapshot(md7_policy),
        "experience_log": file_snapshot(md7_log),
    }
    if md7_before != md7_after:
        raise RuntimeError("MD6 rollback fixture mutated the MD7 fixture lineage.")
    if loader.paths != [None]:
        raise RuntimeError(f"Rollback mode loaded unexpected selectors: {loader.paths}")
    if first_turn["md6_probabilities"] != MD6_PROBABILITIES:
        raise RuntimeError("Rollback C3 did not receive fixture MD6 probabilities.")
    if policy.total_updates != 2 or completion["total_attempt_weight"] != 1.0:
        raise RuntimeError("Rollback delayed-credit behavior changed.")
    return {
        "selector_loads": loader.paths,
        "policy_path": str(resources.policy_state_path),
        "experience_log_path": str(resources.experience_log_path),
        "active_probabilities": first_turn["md6_probabilities"],
        "delayed_updates": policy.total_updates,
        "total_attempt_weight": completion["total_attempt_weight"],
        "md7_lineage_unchanged": True,
    }


def _controlled_live_fixture(root: Path) -> dict[str, object]:
    active = FixtureSelector(MD7_PROBABILITIES)
    shadow = FixtureSelector(MD6_PROBABILITIES)
    selector = ActiveMD7WithMD6Shadow(active_md7=active, shadow_md6=shadow)
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    policy_path = root / "manual" / "policy_state.json"
    experience_path = root / "manual" / "attempts.jsonl"
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
        "attempt_id": "controlled-live-offline-fixture:1",
        "problem": "What is 25 percent of 80?",
        "conversation_history": [],
        "mastery_before": 0.30,
    }
    pipeline.start_attempt(memory)
    result = pipeline.run_tutor_turn(memory, FixtureTutor())
    pipeline.abort_attempt(memory)
    if policy.total_updates != 0 or policy_path.exists() or experience_path.exists():
        raise RuntimeError("Controlled-live abort fixture persisted policy learning.")
    return {
        "active_probabilities": result["md6_probabilities"],
        "shadow_probabilities": selector.last_record["md6_probabilities"],
        "posterior_updates": policy.total_updates,
        "policy_written": policy_path.exists(),
        "experience_log_written": experience_path.exists(),
    }


class RecordingTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self.truncation_side = tokenizer.truncation_side
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return self._tokenizer(*args, **kwargs)


def _actual_cpu_smoke(mode: AdaptiveSelectorMode) -> dict[str, object]:
    values = {
        "ADAPTIVE_DATA_MODE": "synthetic",
        "ADAPTIVE_MD6_POLICY_STATE_PATH": str(MD6_POLICY_PATH),
        "ADAPTIVE_MD6_EXPERIENCE_LOG_PATH": str(MD6_EXPERIENCE_PATH),
        "ADAPTIVE_MD7_POLICY_STATE_PATH": str(MD7_POLICY_PATH),
        "ADAPTIVE_MD7_EXPERIENCE_LOG_PATH": str(MD7_EXPERIENCE_PATH),
        SELECTOR_MODE_ENV: mode.value,
        "CUDA_VISIBLE_DEVICES": "",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    with _environment(
        values,
        remove=(MANUAL_CONTROLLED_LIVE_ENV,) + LINEAGE_ENV_NAMES[:2],
    ):
        resources = _ProductionAdaptiveResources()
        selector = resources._build_base_selector(FrozenMD6Inference)
        recorder = RecordingTokenizer(selector.tokenizer)
        selector.tokenizer = recorder
        probabilities = selector.predict_probabilities(
            "What is 25 percent of 80?",
            [{"user": "student", "text": "I think it is 12."}],
        )
    if recorder.truncation_side != "left":
        raise RuntimeError("Selector tokenizer no longer uses left truncation.")
    if len(recorder.calls) != 1:
        raise RuntimeError("CPU smoke produced an unexpected tokenizer call count.")
    tokenization = recorder.calls[0]
    if tokenization.get("truncation") != "only_second":
        raise RuntimeError("Selector preprocessing is not truncation=only_second.")
    if tokenization.get("max_length") != 512:
        raise RuntimeError("Selector preprocessing max_length changed from 512.")
    if tuple(probabilities) != MOVE_ORDER:
        raise RuntimeError("Selector probability label order changed.")
    result = {
        "mode": mode.value,
        "model_path": str(resources.active_model_path),
        "probabilities": probabilities,
        "preprocessing": {
            "truncation_side": recorder.truncation_side,
            "truncation": tokenization["truncation"],
            "max_length": tokenization["max_length"],
        },
    }
    del selector
    gc.collect()
    return result


def _assert_fresh_production_policy() -> dict[str, object]:
    policy = TrueDisjointLinTS(context_dim=9, seed=42, data_mode="synthetic")
    load_policy_state(policy, MD7_POLICY_PATH, expected_data_mode="synthetic")
    state = policy.state_dict()
    if policy.total_updates != 0 or any(policy.arm_update_counts.values()):
        raise RuntimeError("Production MD7 policy is not a zero-update prior.")
    for arm in policy.arms:
        for row_index, row in enumerate(state["A"][arm]):
            expected = [0.0] * policy.context_dim
            expected[row_index] = policy.ridge_lambda
            if row != expected:
                raise RuntimeError("Production MD7 precision matrix is not fresh.")
        if state["b"][arm] != [0.0] * policy.context_dim:
            raise RuntimeError("Production MD7 reward vector is not fresh.")
    if not MD7_EXPERIENCE_PATH.is_file() or MD7_EXPERIENCE_PATH.stat().st_size != 0:
        raise RuntimeError("Production MD7 experience log is not empty.")
    return {
        "policy_state": file_snapshot(MD7_POLICY_PATH),
        "experience_log": file_snapshot(MD7_EXPERIENCE_PATH),
        "total_updates": policy.total_updates,
        "arm_update_counts": dict(policy.arm_update_counts),
        "fresh_identity_precision": True,
        "fresh_zero_reward_vectors": True,
        "no_migrated_md6_posterior": True,
    }


def _test_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        SELECTOR_MODE_ENV,
        MANUAL_CONTROLLED_LIVE_ENV,
        "CHECKPOINT_DB_PATH",
        "ADAPTIVE_DATA_MODE",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ) + LINEAGE_ENV_NAMES:
        environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_pytest(paths: list[Path], cwd: Path) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", *(str(path) for path in paths), "-q"]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_test_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Offline pytest failed: {' '.join(command)}\n{output}")
    return {"command": command, "returncode": completed.returncode, "output": output}


def _pass(number: int, name: str, evidence: object) -> dict[str, object]:
    return {"number": number, "name": name, "status": "PASS", "evidence": evidence}


def _test_summary(result: Mapping[str, object]) -> str:
    lines = str(result["output"]).splitlines()
    return lines[-1] if lines else "PASS"


def _manifest_test_result(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "command": result["command"],
        "returncode": result["returncode"],
        "summary": _test_summary(result),
    }


def main() -> int:
    md6_lineage_before = {
        "policy_state": file_snapshot(MD6_POLICY_PATH),
        "experience_log": file_snapshot(MD6_EXPERIENCE_PATH),
    }
    if not all(item.get("exists") for item in md6_lineage_before.values()):
        raise RuntimeError("The preserved MD6 production lineage is incomplete.")
    md7_models_before = _model_snapshot(MD7_MODEL_ROOT, MD7_EXPECTED_HASHES)
    md6_models_before = _model_snapshot(MD6_MODEL_ROOT, MD6_EXPECTED_HASHES)

    prepared = prepare_lineage()
    if prepared["total_updates"] != 0:
        raise RuntimeError("Fresh MD7 production lineage already contains updates.")
    fresh_before = _assert_fresh_production_policy()

    targeted_tests = _run_pytest(
        [
            BACKEND_ROOT / "tests" / "test_md7r1_promotion.py",
            BACKEND_ROOT / "tests" / "test_manual_controlled_live.py",
        ],
        BACKEND_ROOT,
    )
    backend_tests = _run_pytest([BACKEND_ROOT / "tests"], BACKEND_ROOT)
    bridge_tests = _run_pytest(
        [MOVE_ROOT / "tests" / "test_student_model_v3_bridge.py"],
        MOVE_ROOT,
    )

    with tempfile.TemporaryDirectory(prefix="md7r1_promotion_fixture_") as temp:
        fixture_root = Path(temp)
        normal_fixture = _normal_completion_fixture(fixture_root / "normal")
        rollback_fixture = _rollback_completion_fixture(fixture_root / "rollback")
        controlled_fixture = _controlled_live_fixture(fixture_root / "manual")

    normal_cpu = _actual_cpu_smoke(AdaptiveSelectorMode.NORMAL_MD7)
    rollback_cpu = _actual_cpu_smoke(AdaptiveSelectorMode.MD6_ROLLBACK)

    if MOVE_ORDER != ("generic", "probing", "focus", "telling"):
        raise RuntimeError("Move label order changed.")
    if len(TURN_FEATURE_NAMES) != 9:
        raise RuntimeError("C3 context dimension changed.")
    if TURN_FEATURE_NAMES[:4] != (
        "md6_p_generic",
        "md6_p_probing",
        "md6_p_focus",
        "md6_p_telling",
    ):
        raise RuntimeError("C3 compatibility field names changed.")
    if DEFAULT_GAP_THRESHOLD != 0.10:
        raise RuntimeError("Overlay threshold changed.")
    if DEFAULT_RIDGE_LAMBDA != 1.0 or DEFAULT_EXPLORATION_SCALE != 0.20:
        raise RuntimeError("LinTS hyperparameters changed.")
    if ARMS != (
        "baseline",
        "generic_bias",
        "probing_bias",
        "focus_bias",
        "telling_bias",
    ):
        raise RuntimeError("LinTS arms changed.")
    if REWARD_NAME != "mastery_delta":
        raise RuntimeError("Reward is no longer raw mastery delta.")
    if CREDIT_SCHEME != "equal_normalized" or UPDATE_TIMING != "after attempt completion only":
        raise RuntimeError("Delayed-credit semantics changed.")
    if MAX_LENGTH != 512:
        raise RuntimeError("Selector maximum length changed.")

    fresh_after = _assert_fresh_production_policy()
    md6_lineage_after = {
        "policy_state": file_snapshot(MD6_POLICY_PATH),
        "experience_log": file_snapshot(MD6_EXPERIENCE_PATH),
    }
    if md6_lineage_before != md6_lineage_after:
        raise RuntimeError("Validation mutated the preserved MD6 production lineage.")
    md7_models_after = _model_snapshot(MD7_MODEL_ROOT, MD7_EXPECTED_HASHES)
    md6_models_after = _model_snapshot(MD6_MODEL_ROOT, MD6_EXPECTED_HASHES)
    if md7_models_before != md7_models_after or md6_models_before != md6_models_after:
        raise RuntimeError("Validation mutated a model artifact.")

    checks = [
        _pass(1, "ordinary mode resolves MD7-R1 epoch 3 active", normal_cpu["model_path"]),
        _pass(2, "ordinary mode does not load/use MD6 as active", normal_fixture["selector_loads"]),
        _pass(3, "rollback mode resolves frozen MD6 active", rollback_cpu["model_path"]),
        _pass(4, "controlled-live resolves MD7 active plus MD6 shadow", _test_summary(targeted_tests)),
        _pass(5, "accepted MD7 hashes match", {name: item["sha256"] for name, item in md7_models_after.items()}),
        _pass(6, "frozen MD6 hashes match", {name: item["sha256"] for name, item in md6_models_after.items()}),
        _pass(7, "label order unchanged", list(MOVE_ORDER)),
        _pass(8, "preprocessing unchanged", normal_cpu["preprocessing"]),
        _pass(9, "normal C3 receives MD7 probabilities", normal_fixture["c3_first_four"]),
        _pass(10, "C3 dimension remains 9", len(TURN_FEATURE_NAMES)),
        _pass(11, "overlay threshold remains 0.10", DEFAULT_GAP_THRESHOLD),
        _pass(12, "LinTS lambda remains 1", DEFAULT_RIDGE_LAMBDA),
        _pass(13, "exploration scale remains 0.20", DEFAULT_EXPLORATION_SCALE),
        _pass(14, "arms unchanged", list(ARMS)),
        _pass(15, "reward remains raw signed mastery delta", {"name": REWARD_NAME, "fixture_reward": normal_fixture["reward"]}),
        _pass(16, "delayed credit remains 1/T", {"updates": normal_fixture["delayed_updates"], "per_update_weight": normal_fixture["per_update_weight"], "total_weight": normal_fixture["total_attempt_weight"]}),
        _pass(17, "fresh MD7 state begins with zero updates", fresh_after["total_updates"]),
        _pass(18, "old MD6 policy state unchanged byte-for-byte", md6_lineage_after["policy_state"]),
        _pass(19, "old MD6 attempt log unchanged byte-for-byte", md6_lineage_after["experience_log"]),
        _pass(20, "normal fixture updates only fresh MD7 lineage", normal_fixture),
        _pass(21, "normal fixture total delayed weight is exactly 1", normal_fixture["total_attempt_weight"]),
        _pass(22, "rollback fixture touches only MD6 lineage", rollback_fixture),
        _pass(23, "controlled-live fixture produces zero updates", controlled_fixture),
        _pass(24, "full backend tests pass", _test_summary(backend_tests)),
        _pass(25, "cross-repository bridge tests pass", _test_summary(bridge_tests)),
    ]

    manifest = {
        "promotion_version": "md7r1_promotion_v1",
        "promoted_selector": "MD7-R1 epoch 3",
        "md7_path": str(MD7_MODEL_ROOT.resolve()),
        "md7_hashes": {name: item["sha256"] for name, item in md7_models_after.items()},
        "md6_rollback_path": str(MD6_MODEL_ROOT.resolve()),
        "md6_hashes": {name: item["sha256"] for name, item in md6_models_after.items()},
        "context_dim": len(TURN_FEATURE_NAMES),
        "context_features": list(TURN_FEATURE_NAMES),
        "overlay_threshold": DEFAULT_GAP_THRESHOLD,
        "lints": {
            "policy": "true disjoint LinTS",
            "ridge_lambda": DEFAULT_RIDGE_LAMBDA,
            "exploration_scale": DEFAULT_EXPLORATION_SCALE,
            "arms": list(ARMS),
            "state_envelope_schema": POLICY_STATE_SCHEMA_VERSION,
            "policy_state_schema": LINTS_STATE_SCHEMA_VERSION,
        },
        "reward_type": "raw signed mastery delta",
        "credit_rule": "T equal delayed updates; weight=1/T each; total attempt weight=1",
        "md7_policy_lineage": {
            "policy_state": str(MD7_POLICY_PATH.resolve()),
            "experience_log": str(MD7_EXPERIENCE_PATH.resolve()),
            "final_total_updates": fresh_after["total_updates"],
            "no_migrated_md6_posterior": True,
        },
        "md6_rollback_policy_lineage": {
            "policy_state": str(MD6_POLICY_PATH.resolve()),
            "experience_log": str(MD6_EXPERIENCE_PATH.resolve()),
            "unchanged_during_validation": True,
        },
        "mode_pairing": {
            AdaptiveSelectorMode.NORMAL_MD7.value: "MD7 selector -> fresh MD7 lineage",
            AdaptiveSelectorMode.MD6_ROLLBACK.value: "frozen MD6 selector -> preserved MD6 lineage",
            AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE.value: "MD7 active + MD6 shadow -> MD7 lineage read-only; completion aborted",
        },
        "routing_fixture": normal_fixture,
        "rollback_fixture": rollback_fixture,
        "controlled_live_fixture": controlled_fixture,
        "cpu_model_smoke": {"normal": normal_cpu, "rollback": rollback_cpu},
        "promotion_validation_result": "PASS",
        "validation_checks": checks,
        "test_results": {
            "targeted": _manifest_test_result(targeted_tests),
            "backend": _manifest_test_result(backend_tests),
            "bridge": _manifest_test_result(bridge_tests),
        },
        "side_effects": {
            "training_writes": 0,
            "model_writes": 0,
            "external_api_calls": 0,
            "held_out_test_use": 0,
            "accidental_protected_file_touches": 0,
            "md6_policy_mutations": 0,
            "md6_experience_log_mutations": 0,
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "notes": [
            "MD6 remains the immutable frozen rollback baseline.",
            "No evaluation metrics were generated or fabricated.",
            "Offline fixture learning used temporary paths; production MD7 state remains at zero updates.",
        ],
    }
    PROMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    PROMOTION_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
